"""Enhanced entity linking between OpenIE and spaCy NER entities."""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..models import Entity as GraphEntity
from ..processors.triplet_processor import ValidatedTriplet

logger = structlog.get_logger(__name__)

# Optional imports for enhanced functionality
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class EntityMatch(NamedTuple):
    """Represents a match between OpenIE and spaCy entities."""
    openie_entity: str
    spacy_entity: GraphEntity
    match_type: str  # exact, normalized, fuzzy, semantic, hybrid
    confidence: float
    similarity_score: float
    semantic_score: float = 0.0
    fuzzy_score: float = 0.0
    type_compatibility: float = 0.0
    metadata: Dict[str, Any] = {}


class LinkedTriplet(NamedTuple):
    """Represents a triplet with linked entities."""
    subject: str
    predicate: str
    object: str
    subject_entity: Optional[GraphEntity]
    object_entity: Optional[GraphEntity]
    subject_match: Optional[EntityMatch]
    object_match: Optional[EntityMatch]
    confidence: float
    validation_score: float
    sentence: str
    sentence_id: str
    source_doc_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class EntityLinker:
    """Enhanced entity linker with semantic similarity and improved fuzzy matching."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced entity linker.

        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}

        # Configuration
        self.min_match_confidence = self.config.get('min_match_confidence', 0.6)
        self.enable_fuzzy_matching = self.config.get('enable_fuzzy_matching', True)
        self.enable_semantic_matching = self.config.get('enable_semantic_matching', SENTENCE_TRANSFORMERS_AVAILABLE)
        self.enable_hybrid_matching = self.config.get('enable_hybrid_matching', True)
        self.fuzzy_threshold = self.config.get('fuzzy_threshold', 0.75)
        self.semantic_threshold = self.config.get('semantic_threshold', 0.7)
        self.max_edit_distance = self.config.get('max_edit_distance', 2)
        self.enable_type_filtering = self.config.get('enable_type_filtering', True)
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 1000)

        # Weights for hybrid scoring
        self.fuzzy_weight = self.config.get('fuzzy_weight', 0.4)
        self.semantic_weight = self.config.get('semantic_weight', 0.4)
        self.type_weight = self.config.get('type_weight', 0.2)

        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="entity_linker")

        # Compiled patterns for normalization
        self._normalization_patterns = self._compile_normalization_patterns()

        # Initialize semantic model if available
        self._semantic_model = None
        self._semantic_cache = {} if self.enable_caching else None
        self._match_cache = {} if self.enable_caching else None

        # Initialize semantic model
        if self.enable_semantic_matching and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_semantic_model()

        logger.info(
            "Enhanced entity linker initialized",
            min_match_confidence=self.min_match_confidence,
            enable_fuzzy_matching=self.enable_fuzzy_matching,
            enable_semantic_matching=self.enable_semantic_matching,
            enable_hybrid_matching=self.enable_hybrid_matching,
            fuzzy_threshold=self.fuzzy_threshold,
            semantic_threshold=self.semantic_threshold,
            enable_type_filtering=self.enable_type_filtering,
            sentence_transformers_available=SENTENCE_TRANSFORMERS_AVAILABLE,
            fuzzywuzzy_available=FUZZYWUZZY_AVAILABLE
        )
    
    def _init_semantic_model(self) -> None:
        """Initialize semantic similarity model."""
        try:
            # Use a lightweight but effective model
            model_name = self.config.get('semantic_model', 'all-MiniLM-L6-v2')
            self._semantic_model = SentenceTransformer(model_name)
            logger.info(f"Semantic model initialized: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic model: {e}")
            self.enable_semantic_matching = False

    def _compile_normalization_patterns(self) -> Dict[str, re.Pattern]:
        """Compile enhanced regex patterns for entity normalization."""
        return {
            # Remove extra whitespace
            'whitespace': re.compile(r'\s+'),

            # Remove common prefixes/suffixes
            'prefixes': re.compile(r'^(?:the|a|an|this|that|these|those)\s+', re.IGNORECASE),
            'suffixes': re.compile(r'\s+(?:inc|corp|ltd|llc|co|company|organization|university|college|institute|foundation|association)\.?$', re.IGNORECASE),

            # Remove punctuation for matching
            'punctuation': re.compile(r'[^\w\s]'),

            # Common titles and abbreviations
            'titles': re.compile(r'\b(?:dr|mr|mrs|ms|prof|sr|jr|phd|md|ceo|cto|cfo|president|director)\b\.?', re.IGNORECASE),

            # Remove possessive forms
            'possessive': re.compile(r"'s\b", re.IGNORECASE),

            # Normalize quotes
            'quotes': re.compile(r'[""''`]'),

            # Remove common noise words
            'noise_words': re.compile(r'\b(?:said|says|according|reported|stated|mentioned|noted|announced)\b', re.IGNORECASE),
        }
    
    async def link_triplets(
        self, 
        triplets: List[ValidatedTriplet], 
        spacy_entities: List[GraphEntity],
        source_doc_id: Optional[str] = None
    ) -> List[LinkedTriplet]:
        """Link OpenIE triplets to spaCy entities.
        
        Args:
            triplets: List of validated OpenIE triplets
            spacy_entities: List of spaCy extracted entities
            source_doc_id: Optional source document ID
            
        Returns:
            List of triplets with linked entities
            
        Raises:
            ProcessingError: If linking fails
        """
        if not triplets:
            return []
        
        try:
            logger.debug(
                "Starting entity linking",
                triplet_count=len(triplets),
                spacy_entity_count=len(spacy_entities),
                source_doc_id=source_doc_id
            )
            
            # Build entity lookup structures
            entity_lookup = await self._build_entity_lookup(spacy_entities)
            
            # Link triplets
            linked_triplets = []
            for triplet in triplets:
                try:
                    linked = await self._link_single_triplet(triplet, entity_lookup, source_doc_id)
                    if linked:
                        linked_triplets.append(linked)
                except Exception as e:
                    logger.warning(
                        "Failed to link triplet",
                        triplet=f"{triplet.subject} | {triplet.predicate} | {triplet.object}",
                        error=str(e)
                    )
            
            logger.info(
                "Entity linking completed",
                input_triplets=len(triplets),
                linked_triplets=len(linked_triplets),
                source_doc_id=source_doc_id
            )
            
            return linked_triplets
            
        except Exception as e:
            logger.error(
                "Entity linking failed",
                error=str(e),
                error_type=type(e).__name__,
                triplet_count=len(triplets),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Entity linking failed: {e}")
    
    async def _build_entity_lookup(self, entities: List[GraphEntity]) -> Dict[str, Any]:
        """Build lookup structures for efficient entity matching."""
        def build_lookup_sync():
            lookup = {
                'exact_matches': {},
                'normalized_matches': {},
                'type_groups': defaultdict(list),
                'all_entities': entities
            }
            
            for entity in entities:
                entity_text = entity.name.strip()
                entity_type = entity.type
                
                if not entity_text:
                    continue
                
                # Exact match lookup
                lookup['exact_matches'][entity_text.lower()] = entity
                
                # Normalized match lookup
                normalized = self._normalize_entity_text(entity_text)
                lookup['normalized_matches'][normalized] = entity
                
                # Type-based grouping
                lookup['type_groups'][entity_type].append(entity)
                
                # Add common variations
                variations = self._generate_entity_variations(entity_text)
                for variation in variations:
                    lookup['exact_matches'][variation.lower()] = entity
                    normalized_var = self._normalize_entity_text(variation)
                    lookup['normalized_matches'][normalized_var] = entity
            
            return lookup
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, build_lookup_sync
        )
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for matching (legacy method)."""
        if not text:
            return ""

        normalized = text.strip().lower()

        # Remove prefixes and suffixes
        normalized = self._normalization_patterns['prefixes'].sub('', normalized)

        # Remove punctuation
        normalized = self._normalization_patterns['punctuation'].sub(' ', normalized)

        # Normalize whitespace
        normalized = self._normalization_patterns['whitespace'].sub(' ', normalized)

        return normalized.strip()
    
    def _generate_entity_variations(self, text: str) -> List[str]:
        """Generate common variations of entity text."""
        variations = []
        
        # Add the original text
        variations.append(text)
        
        # Add without common prefixes/suffixes
        no_prefix = self._normalization_patterns['prefixes'].sub('', text).strip()
        if no_prefix != text:
            variations.append(no_prefix)
        
        no_suffix = self._normalization_patterns['suffixes'].sub('', text).strip()
        if no_suffix != text:
            variations.append(no_suffix)
        
        # Add title case and upper case
        variations.append(text.title())
        variations.append(text.upper())
        
        # Add without punctuation
        no_punct = self._normalization_patterns['punctuation'].sub('', text)
        if no_punct != text:
            variations.append(no_punct)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen and var.strip():
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations
    
    async def _link_single_triplet(
        self, 
        triplet: ValidatedTriplet, 
        entity_lookup: Dict[str, Any],
        source_doc_id: Optional[str] = None
    ) -> Optional[LinkedTriplet]:
        """Link a single triplet to spaCy entities."""
        def link_sync():
            # Find matches for subject and object
            subject_match = self._find_entity_match(triplet.subject, entity_lookup)
            object_match = self._find_entity_match(triplet.object, entity_lookup)
            
            # Extract entities from matches
            subject_entity = subject_match.spacy_entity if subject_match else None
            object_entity = object_match.spacy_entity if object_match else None
            
            # Create metadata
            metadata = triplet.metadata.copy()
            metadata.update({
                'subject_linked': subject_entity is not None,
                'object_linked': object_entity is not None,
                'subject_match_type': subject_match.match_type if subject_match else None,
                'object_match_type': object_match.match_type if object_match else None,
                'subject_match_confidence': subject_match.confidence if subject_match else 0.0,
                'object_match_confidence': object_match.confidence if object_match else 0.0,
            })
            
            return LinkedTriplet(
                subject=triplet.subject,
                predicate=triplet.predicate,
                object=triplet.object,
                subject_entity=subject_entity,
                object_entity=object_entity,
                subject_match=subject_match,
                object_match=object_match,
                confidence=triplet.confidence,
                validation_score=triplet.validation_score,
                sentence=triplet.sentence,
                sentence_id=triplet.sentence_id,
                source_doc_id=source_doc_id or triplet.source_doc_id,
                metadata=metadata
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, link_sync
        )
    
    def _find_entity_match(self, openie_entity: str, entity_lookup: Dict[str, Any]) -> Optional[EntityMatch]:
        """Find the best matching spaCy entity for an OpenIE entity with enhanced matching."""
        if not openie_entity or not openie_entity.strip():
            return None

        # Check cache first
        cache_key = f"{openie_entity.lower()}:{len(entity_lookup['all_entities'])}"
        if self._match_cache and cache_key in self._match_cache:
            return self._match_cache[cache_key]

        # Try exact match first
        exact_match = entity_lookup['exact_matches'].get(openie_entity.lower())
        if exact_match:
            match = EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=exact_match,
                match_type='exact',
                confidence=1.0,
                similarity_score=1.0,
                semantic_score=1.0,
                fuzzy_score=1.0,
                type_compatibility=1.0,
                metadata={'match_method': 'exact_lookup'}
            )
            self._cache_match(cache_key, match)
            return match

        # Try normalized match
        normalized_openie = self._normalize_entity_text_enhanced(openie_entity)
        normalized_match = entity_lookup['normalized_matches'].get(normalized_openie)
        if normalized_match:
            match = EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=normalized_match,
                match_type='normalized',
                confidence=0.95,
                similarity_score=0.95,
                semantic_score=0.9,
                fuzzy_score=0.9,
                type_compatibility=0.8,
                metadata={'match_method': 'normalized_lookup', 'normalized_text': normalized_openie}
            )
            self._cache_match(cache_key, match)
            return match

        # Try advanced matching strategies
        best_match = None

        if self.enable_hybrid_matching:
            best_match = self._find_hybrid_match(openie_entity, entity_lookup['all_entities'])
        elif self.enable_semantic_matching:
            best_match = self._find_semantic_match(openie_entity, entity_lookup['all_entities'])
        elif self.enable_fuzzy_matching:
            best_match = self._find_fuzzy_match_enhanced(openie_entity, entity_lookup['all_entities'])

        if best_match:
            self._cache_match(cache_key, best_match)

        return best_match
    
    def _cache_match(self, cache_key: str, match: EntityMatch) -> None:
        """Cache a match result."""
        if not self._match_cache:
            return

        if len(self._match_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._match_cache))
            del self._match_cache[oldest_key]

        self._match_cache[cache_key] = match

    def _normalize_entity_text_enhanced(self, text: str) -> str:
        """Enhanced entity text normalization."""
        if not text:
            return ""

        # Apply all normalization patterns
        normalized = text.lower()

        # Remove noise words
        normalized = self._normalization_patterns['noise_words'].sub('', normalized)

        # Remove titles
        normalized = self._normalization_patterns['titles'].sub('', normalized)

        # Remove possessive forms
        normalized = self._normalization_patterns['possessive'].sub('', normalized)

        # Normalize quotes
        normalized = self._normalization_patterns['quotes'].sub('"', normalized)

        # Remove prefixes and suffixes
        normalized = self._normalization_patterns['prefixes'].sub('', normalized)
        normalized = self._normalization_patterns['suffixes'].sub('', normalized)

        # Remove punctuation
        normalized = self._normalization_patterns['punctuation'].sub(' ', normalized)

        # Normalize whitespace
        normalized = self._normalization_patterns['whitespace'].sub(' ', normalized)

        return normalized.strip()

    def _find_hybrid_match(self, openie_entity: str, all_entities: List[GraphEntity]) -> Optional[EntityMatch]:
        """Find best match using hybrid approach combining fuzzy and semantic similarity."""
        if not all_entities:
            return None

        best_match = None
        best_score = 0.0
        best_fuzzy_score = 0.0
        best_semantic_score = 0.0
        best_type_score = 0.0

        normalized_openie = self._normalize_entity_text_enhanced(openie_entity)

        for entity in all_entities:
            # Calculate fuzzy similarity
            fuzzy_score = self._calculate_fuzzy_similarity_enhanced(normalized_openie, self._normalize_entity_text_enhanced(entity.name))

            # Calculate semantic similarity if available
            semantic_score = 0.0
            if self.enable_semantic_matching and self._semantic_model:
                semantic_score = self._calculate_semantic_similarity(openie_entity, entity.name)

            # Calculate type compatibility
            type_score = self._calculate_type_compatibility(openie_entity, entity)

            # Calculate hybrid score
            hybrid_score = (
                self.fuzzy_weight * fuzzy_score +
                self.semantic_weight * semantic_score +
                self.type_weight * type_score
            )

            if hybrid_score > best_score and hybrid_score >= self.min_match_confidence:
                best_score = hybrid_score
                best_match = entity
                best_fuzzy_score = fuzzy_score
                best_semantic_score = semantic_score
                best_type_score = type_score

        if best_match:
            return EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=best_match,
                match_type='hybrid',
                confidence=best_score,
                similarity_score=best_score,
                semantic_score=best_semantic_score,
                fuzzy_score=best_fuzzy_score,
                type_compatibility=best_type_score,
                metadata={
                    'match_method': 'hybrid_matching',
                    'fuzzy_weight': self.fuzzy_weight,
                    'semantic_weight': self.semantic_weight,
                    'type_weight': self.type_weight
                }
            )

        return None
    
    def _find_semantic_match(self, openie_entity: str, all_entities: List[GraphEntity]) -> Optional[EntityMatch]:
        """Find best match using semantic similarity."""
        if not self._semantic_model or not all_entities:
            return None

        best_match = None
        best_score = 0.0

        for entity in all_entities:
            semantic_score = self._calculate_semantic_similarity(openie_entity, entity.name)

            if semantic_score >= self.semantic_threshold and semantic_score > best_score:
                best_score = semantic_score
                best_match = entity

        if best_match and best_score >= self.min_match_confidence:
            return EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=best_match,
                match_type='semantic',
                confidence=best_score,
                similarity_score=best_score,
                semantic_score=best_score,
                fuzzy_score=0.0,
                type_compatibility=0.0,
                metadata={'match_method': 'semantic_similarity'}
            )

        return None

    def _find_fuzzy_match_enhanced(self, openie_entity: str, all_entities: List[GraphEntity]) -> Optional[EntityMatch]:
        """Enhanced fuzzy matching with multiple algorithms."""
        if not all_entities:
            return None

        best_match = None
        best_score = 0.0

        normalized_openie = self._normalize_entity_text_enhanced(openie_entity)

        for entity in all_entities:
            # Calculate enhanced fuzzy similarity
            fuzzy_score = self._calculate_fuzzy_similarity_enhanced(normalized_openie, self._normalize_entity_text_enhanced(entity.name))

            if fuzzy_score >= self.fuzzy_threshold and fuzzy_score > best_score:
                best_score = fuzzy_score
                best_match = entity

        if best_match and best_score >= self.min_match_confidence:
            return EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=best_match,
                match_type='fuzzy',
                confidence=best_score,
                similarity_score=best_score,
                semantic_score=0.0,
                fuzzy_score=best_score,
                type_compatibility=0.0,
                metadata={'match_method': 'enhanced_fuzzy_matching'}
            )

        return None

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not self._semantic_model or not text1 or not text2:
            return 0.0

        # Check cache first
        cache_key = f"{text1.lower()}:{text2.lower()}"
        if self._semantic_cache and cache_key in self._semantic_cache:
            return self._semantic_cache[cache_key]

        try:
            # Get embeddings
            embeddings = self._semantic_model.encode([text1, text2])

            # Calculate cosine similarity
            if NUMPY_AVAILABLE:
                similarity = float(np.dot(embeddings[0], embeddings[1]) /
                                 (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            else:
                # Fallback without numpy
                dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
                norm1 = sum(a * a for a in embeddings[0]) ** 0.5
                norm2 = sum(b * b for b in embeddings[1]) ** 0.5
                similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

            # Cache result
            if self._semantic_cache:
                if len(self._semantic_cache) >= self.cache_size:
                    oldest_key = next(iter(self._semantic_cache))
                    del self._semantic_cache[oldest_key]
                self._semantic_cache[cache_key] = similarity

            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def _calculate_fuzzy_similarity_enhanced(self, text1: str, text2: str) -> float:
        """Calculate enhanced fuzzy similarity using multiple algorithms."""
        if not text1 or not text2:
            return 0.0

        if text1 == text2:
            return 1.0

        # Use fuzzywuzzy if available for better performance
        if FUZZYWUZZY_AVAILABLE:
            # Combine multiple fuzzy metrics
            ratio = fuzz.ratio(text1, text2) / 100.0
            partial_ratio = fuzz.partial_ratio(text1, text2) / 100.0
            token_sort_ratio = fuzz.token_sort_ratio(text1, text2) / 100.0
            token_set_ratio = fuzz.token_set_ratio(text1, text2) / 100.0

            # Weighted average of different metrics
            return (0.3 * ratio + 0.2 * partial_ratio + 0.25 * token_sort_ratio + 0.25 * token_set_ratio)
        else:
            # Fallback to Levenshtein ratio
            return self._levenshtein_ratio(text1, text2)

    def _calculate_type_compatibility(self, openie_entity: str, spacy_entity: GraphEntity) -> float:
        """Calculate type compatibility between entities."""
        if not self.enable_type_filtering:
            return 1.0

        # Simple heuristics for type compatibility
        # This could be enhanced with more sophisticated type inference

        # Check if both are likely person names
        if self._is_likely_person(openie_entity) and spacy_entity.type in ['PERSON', 'PER']:
            return 1.0

        # Check if both are likely organizations
        if self._is_likely_organization(openie_entity) and spacy_entity.type in ['ORG', 'ORGANIZATION']:
            return 1.0

        # Check if both are likely locations
        if self._is_likely_location(openie_entity) and spacy_entity.type in ['GPE', 'LOC', 'LOCATION']:
            return 1.0

        # Default compatibility
        return 0.5

    def _is_likely_person(self, text: str) -> bool:
        """Check if text is likely a person name."""
        # Simple heuristics
        words = text.split()
        if len(words) >= 2:
            # Check for common titles
            if any(word.lower() in ['dr', 'mr', 'mrs', 'ms', 'prof', 'president', 'ceo'] for word in words):
                return True
            # Check for capitalized words (names)
            if all(word[0].isupper() for word in words if word):
                return True
        return False

    def _is_likely_organization(self, text: str) -> bool:
        """Check if text is likely an organization."""
        org_indicators = ['inc', 'corp', 'ltd', 'llc', 'company', 'university', 'college', 'institute', 'foundation']
        return any(indicator in text.lower() for indicator in org_indicators)

    def _is_likely_location(self, text: str) -> bool:
        """Check if text is likely a location."""
        location_indicators = ['city', 'state', 'country', 'county', 'province', 'district', 'region']
        return any(indicator in text.lower() for indicator in location_indicators)
    
    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein ratio between two strings."""
        if len(s1) == 0:
            return 0.0 if len(s2) > 0 else 1.0
        if len(s2) == 0:
            return 0.0
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        # Convert to ratio (1.0 = identical, 0.0 = completely different)
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'semantic_cache_size': len(self._semantic_cache) if self._semantic_cache else 0,
            'match_cache_size': len(self._match_cache) if self._match_cache else 0,
            'semantic_model_loaded': self._semantic_model is not None,
            'fuzzy_matching_enabled': self.enable_fuzzy_matching,
            'semantic_matching_enabled': self.enable_semantic_matching,
            'hybrid_matching_enabled': self.enable_hybrid_matching,
            'fuzzywuzzy_available': FUZZYWUZZY_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)

            # Clear caches
            if self._semantic_cache:
                self._semantic_cache.clear()
            if self._match_cache:
                self._match_cache.clear()

            logger.info("Enhanced entity linker closed")
        except Exception as e:
            logger.warning("Error during entity linker cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
