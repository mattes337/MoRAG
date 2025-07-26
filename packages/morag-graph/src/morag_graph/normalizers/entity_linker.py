"""Entity linking between OpenIE and spaCy NER entities."""

import asyncio
import re
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..models import Entity as GraphEntity
from ..processors.triplet_processor import ValidatedTriplet

logger = structlog.get_logger(__name__)


class EntityMatch(NamedTuple):
    """Represents a match between OpenIE and spaCy entities."""
    openie_entity: str
    spacy_entity: GraphEntity
    match_type: str  # exact, normalized, fuzzy, semantic
    confidence: float
    similarity_score: float
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
    """Links OpenIE entities to spaCy NER entities with fuzzy matching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize entity linker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration
        self.min_match_confidence = self.config.get('min_match_confidence', 0.6)
        self.enable_fuzzy_matching = self.config.get('enable_fuzzy_matching', True)
        self.enable_semantic_matching = self.config.get('enable_semantic_matching', False)
        self.fuzzy_threshold = self.config.get('fuzzy_threshold', 0.8)
        self.max_edit_distance = self.config.get('max_edit_distance', 2)
        self.enable_type_filtering = self.config.get('enable_type_filtering', True)
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="entity_linker")
        
        # Compiled patterns for normalization
        self._normalization_patterns = self._compile_normalization_patterns()
        
        logger.info(
            "Entity linker initialized",
            min_match_confidence=self.min_match_confidence,
            enable_fuzzy_matching=self.enable_fuzzy_matching,
            fuzzy_threshold=self.fuzzy_threshold,
            enable_type_filtering=self.enable_type_filtering
        )
    
    def _compile_normalization_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for entity normalization."""
        return {
            # Remove extra whitespace
            'whitespace': re.compile(r'\s+'),
            
            # Remove common prefixes/suffixes
            'prefixes': re.compile(r'^(?:the|a|an)\s+', re.IGNORECASE),
            'suffixes': re.compile(r'\s+(?:inc|corp|ltd|llc|co|company|organization)\.?$', re.IGNORECASE),
            
            # Remove punctuation for matching
            'punctuation': re.compile(r'[^\w\s]'),
            
            # Common abbreviations
            'abbreviations': re.compile(r'\b(?:dr|mr|mrs|ms|prof|sr|jr)\b\.?', re.IGNORECASE),
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
        """Normalize entity text for matching."""
        normalized = text.strip().lower()
        
        # Remove prefixes and suffixes
        normalized = self._normalization_patterns['prefixes'].sub('', normalized)
        normalized = self._normalization_patterns['suffixes'].sub('', normalized)
        
        # Remove abbreviations
        normalized = self._normalization_patterns['abbreviations'].sub('', normalized)
        
        # Remove punctuation
        normalized = self._normalization_patterns['punctuation'].sub('', normalized)
        
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
        """Find the best matching spaCy entity for an OpenIE entity."""
        if not openie_entity or not openie_entity.strip():
            return None
        
        # Try exact match first
        exact_match = entity_lookup['exact_matches'].get(openie_entity.lower())
        if exact_match:
            return EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=exact_match,
                match_type='exact',
                confidence=1.0,
                similarity_score=1.0,
                metadata={'match_method': 'exact_lookup'}
            )
        
        # Try normalized match
        normalized_openie = self._normalize_entity_text(openie_entity)
        normalized_match = entity_lookup['normalized_matches'].get(normalized_openie)
        if normalized_match:
            return EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=normalized_match,
                match_type='normalized',
                confidence=0.9,
                similarity_score=0.9,
                metadata={'match_method': 'normalized_lookup', 'normalized_text': normalized_openie}
            )
        
        # Try fuzzy matching if enabled
        if self.enable_fuzzy_matching:
            fuzzy_match = self._find_fuzzy_match(openie_entity, entity_lookup['all_entities'])
            if fuzzy_match:
                return fuzzy_match
        
        return None
    
    def _find_fuzzy_match(self, openie_entity: str, all_entities: List[GraphEntity]) -> Optional[EntityMatch]:
        """Find fuzzy matches using string similarity."""
        best_match = None
        best_score = 0.0
        
        normalized_openie = self._normalize_entity_text(openie_entity)
        
        for entity in all_entities:
            # Calculate similarity
            similarity = self._calculate_similarity(normalized_openie, self._normalize_entity_text(entity.name))
            
            if similarity >= self.fuzzy_threshold and similarity > best_score:
                best_score = similarity
                best_match = entity
        
        if best_match and best_score >= self.min_match_confidence:
            return EntityMatch(
                openie_entity=openie_entity,
                spacy_entity=best_match,
                match_type='fuzzy',
                confidence=best_score,
                similarity_score=best_score,
                metadata={
                    'match_method': 'fuzzy_matching',
                    'similarity_algorithm': 'levenshtein_ratio'
                }
            )
        
        return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        if text1 == text2:
            return 1.0
        
        # Use Levenshtein distance ratio
        return self._levenshtein_ratio(text1, text2)
    
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
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Entity linker closed")
        except Exception as e:
            logger.warning("Error during entity linker cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
