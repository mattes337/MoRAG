"""Entity normalization with LLM-based multilingual support."""

import asyncio
import re
import json
from typing import List, Dict, Any, Optional, NamedTuple, Set, Callable, Tuple, Type
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from morag_core.ai import MoRAGBaseAgent
from pydantic import BaseModel, Field
from ..config.normalization_config import get_config_for_component
from ..monitoring.normalization_metrics import NormalizationMonitor, PerformanceTimer

logger = structlog.get_logger(__name__)


@dataclass
class EntityVariation:
    """Represents a variation of an entity during normalization."""
    original: str
    normalized: str
    confidence: float
    rule_applied: str


@dataclass
class EntityMergeCandidate:
    """Represents entities that should be merged."""
    entities: List[str]
    canonical_form: str
    confidence: float
    merge_reason: str


class NormalizedEntity(BaseModel):
    """Represents a normalized entity."""
    original_text: str
    normalized_text: str
    canonical_form: str
    language: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: float = 1.0
    variations: List[str] = Field(default_factory=list)
    normalization_method: str = "rule_based"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityNormalizationResult(BaseModel):
    """Result from LLM entity normalization."""
    entities: List[NormalizedEntity]


class EntityMergeAnalysisResult(BaseModel):
    """Result from LLM entity merge analysis."""
    merge_groups: List[Dict[str, Any]] = Field(description="Groups of entities that should be merged")


class EntityNormalizationAgent(MoRAGBaseAgent[EntityNormalizationResult]):
    """PydanticAI agent for entity normalization."""

    def get_result_type(self) -> Type[EntityNormalizationResult]:
        return EntityNormalizationResult

    def get_system_prompt(self) -> str:
        return """You are an expert entity normalizer. Your task is to normalize entity names to their canonical, singular, non-conjugated forms while preserving their semantic meaning.

Guidelines:
1. Convert to singular form (e.g., "pilots" → "pilot", "companies" → "company")
2. Use non-conjugated forms (e.g., "running" → "run", "worked" → "work")
3. Preserve proper nouns in their standard form (e.g., "John Smith" stays "John Smith")
4. Handle multiple languages (English, Spanish, German)
5. Remove unnecessary articles and prepositions
6. Maintain semantic meaning and entity type

For each entity, provide:
- original_text: The input entity text
- normalized_text: The normalized form
- canonical_form: The canonical representation
- language: Detected language (en, es, de, or null)
- confidence: Confidence in normalization (0.0-1.0)
- variations: Alternative forms of the entity"""


class EntityMergeAnalysisAgent(MoRAGBaseAgent[EntityMergeAnalysisResult]):
    """PydanticAI agent for analyzing entity merge candidates."""

    def get_result_type(self) -> Type[EntityMergeAnalysisResult]:
        return EntityMergeAnalysisResult

    def get_system_prompt(self) -> str:
        return """You are an expert entity deduplication analyst. Your task is to identify entities that refer to the same real-world entity and should be merged.

Consider variations in:
- Spelling and formatting
- Abbreviations vs full forms
- Different name formats (e.g., "John Smith" vs "Smith, John")
- Language variations
- Punctuation differences

Only suggest merges with high confidence (>0.7). Be conservative to avoid incorrect merges."""


class EnhancedEntityNormalizer:
    """Enhanced entity normalizer with LLM-based deduplication and merge detection."""

    def __init__(self, llm_service=None, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced entity normalizer.

        Args:
            llm_service: LLM service for normalization (optional)
            config: Optional configuration dictionary
        """
        self.llm_service = llm_service

        # Load configuration from file or use provided config
        if config is None:
            self.config = get_config_for_component('normalizer')
        else:
            # Merge provided config with defaults
            default_config = get_config_for_component('normalizer')
            default_config.update(config)
            self.config = default_config

        # Cache for LLM-based normalization results to avoid repeated calls
        self.normalization_cache = {}
        self.entity_type_cache = {}
        self.similarity_cache = {}

        # Configuration
        self.batch_size = self.config.get('batch_size', 20)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.merge_confidence_threshold = self.config.get('merge_confidence_threshold', 0.8)
        self.enable_llm_normalization = self.config.get('enable_llm_normalization', True)
        self.enable_rule_based_fallback = self.config.get('enable_rule_based_fallback', True)

        # Initialize LLM agents if service is available and enabled
        if self.llm_service and self.enable_llm_normalization:
            self.normalization_agent = EntityNormalizationAgent()
            self.merge_analysis_agent = EntityMergeAnalysisAgent()
        else:
            self.normalization_agent = None
            self.merge_analysis_agent = None

        # Initialize monitoring
        monitoring_enabled = self.config.get('monitoring', {}).get('enabled', True)
        self.monitor = NormalizationMonitor(enabled=monitoring_enabled)

        logger.info(
            "Enhanced entity normalizer initialized",
            has_llm_service=self.llm_service is not None,
            batch_size=self.batch_size,
            min_confidence=self.min_confidence,
            monitoring_enabled=monitoring_enabled
        )

    async def normalize_entity(self, entity: str, language: str = None) -> EntityVariation:
        """Normalize a single entity using LLM-based analysis."""
        original = entity

        # Record entity processing
        self.monitor.record_entity_processed(entity, language=language)

        with PerformanceTimer(self.monitor, "normalize_entity"):
            # Check cache first
            cache_key = f"{entity}:{language or 'auto'}"
            if cache_key in self.normalization_cache:
                cached_result = self.normalization_cache[cache_key]
                variation = EntityVariation(
                    original=original,
                    normalized=cached_result['normalized'],
                    confidence=cached_result['confidence'],
                    rule_applied=cached_result['rule_applied']
                )

                # Record normalization from cache
                self.monitor.record_entity_normalized(
                    original, variation.normalized, variation.confidence,
                    "cached", variation.rule_applied
                )
                return variation

            # Use LLM for normalization
            try:
                normalized, confidence, rule_applied = await self._llm_normalize_entity(entity, language)

                # Cache result
                self.normalization_cache[cache_key] = {
                    'normalized': normalized,
                    'confidence': confidence,
                    'rule_applied': rule_applied
                }

                variation = EntityVariation(
                    original=original,
                    normalized=normalized,
                    confidence=confidence,
                    rule_applied=rule_applied
                )

                # Record normalization
                self.monitor.record_entity_normalized(
                    original, normalized, confidence, "llm", rule_applied
                )

                return variation

            except Exception as e:
                self.monitor.record_error(e, f"normalize_entity: {entity}")
                raise

    async def find_merge_candidates(self, entities: List[str], language: str = None) -> List[EntityMergeCandidate]:
        """Find entities that should be merged using LLM-based similarity analysis."""
        candidates = []

        with PerformanceTimer(self.monitor, "find_merge_candidates"):
            try:
                # Use LLM to analyze entity similarities in batches
                batch_size = 20  # Process entities in batches to avoid token limits

                for i in range(0, len(entities), batch_size):
                    batch = entities[i:i + batch_size]
                    batch_candidates = await self._llm_find_merge_candidates(batch, language)
                    candidates.extend(batch_candidates)

                # Record merge candidates found
                self.monitor.record_merge_candidates(len(candidates))

                return candidates

            except Exception as e:
                self.monitor.record_error(e, f"find_merge_candidates: {len(entities)} entities")
                raise

    async def _llm_normalize_entity(self, entity: str, language: str = None) -> Tuple[str, float, str]:
        """Use LLM to normalize entity."""
        if not self.normalization_agent:
            # Fallback to basic normalization
            return entity.strip(), 0.5, "basic_cleanup"

        prompt = f"""
        Normalize the following entity name to its canonical form. Consider:
        - Remove unnecessary punctuation and formatting
        - Standardize capitalization appropriately
        - Expand common abbreviations if beneficial
        - Maintain the core meaning and identity

        Entity: "{entity}"
        Language context: {language or "auto-detect"}

        Respond with JSON:
        {{
            "normalized": "canonical form",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation of changes made"
        }}
        """

        try:
            result = await self.normalization_agent.run(prompt)

            if result and result.entities and len(result.entities) > 0:
                normalized_entity = result.entities[0]
                return (
                    normalized_entity.normalized_text,
                    normalized_entity.confidence,
                    normalized_entity.normalization_method
                )
            else:
                # Fallback if no result
                return entity.strip(), 0.3, "llm_no_result"

        except Exception as e:
            logger.warning("LLM normalization failed", error=str(e))
            # Fallback on LLM failure
            return entity.strip(), 0.3, f"llm_error: {str(e)}"

    async def _llm_find_merge_candidates(self, entities: List[str], language: str = None) -> List[EntityMergeCandidate]:
        """Use LLM to find entities that should be merged."""
        if not self.merge_analysis_agent or len(entities) < 2:
            return []

        entities_text = "\n".join([f"{i+1}. {entity}" for i, entity in enumerate(entities)])

        prompt = f"""
        Analyze the following entities and identify which ones refer to the same real-world entity and should be merged.
        Consider variations in:
        - Spelling and formatting
        - Abbreviations vs full forms
        - Different name formats (e.g., "John Smith" vs "Smith, John")
        - Language variations
        - Punctuation differences

        Entities:
        {entities_text}

        Language context: {language or "auto-detect"}

        Respond with JSON array of merge groups:
        [
            {{
                "entities": ["entity1", "entity2"],
                "canonical_form": "preferred form",
                "confidence": 0.0-1.0,
                "reason": "explanation"
            }}
        ]

        Only include groups with confidence > 0.7.
        """

        try:
            result = await self.merge_analysis_agent.run(prompt)

            candidates = []
            if result and result.merge_groups:
                for group in result.merge_groups:
                    if (isinstance(group, dict) and
                        'entities' in group and
                        len(group['entities']) >= 2 and
                        group.get('confidence', 0) > 0.7):

                        candidates.append(EntityMergeCandidate(
                            entities=group['entities'],
                            canonical_form=group.get('canonical_form', group['entities'][0]),
                            confidence=float(group.get('confidence', 0.8)),
                            merge_reason=group.get('reason', 'llm_similarity_analysis')
                        ))

            return candidates

        except Exception as e:
            logger.warning("LLM merge analysis failed", error=str(e))
            # Fallback to simple string similarity
            return self._fallback_similarity_matching(entities)

    def get_metrics(self) -> Dict[str, Any]:
        """Get normalization metrics.

        Returns:
            Dictionary of current metrics
        """
        return self.monitor.get_metrics()

    def log_metrics_summary(self):
        """Log a summary of normalization metrics."""
        self.monitor.log_summary()

    def reset_metrics(self):
        """Reset normalization metrics."""
        self.monitor.reset_metrics()

    def _fallback_similarity_matching(self, entities: List[str]) -> List[EntityMergeCandidate]:
        """Fallback similarity matching when LLM is unavailable."""
        candidates = []

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Simple string similarity
                similarity = SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()

                if similarity > 0.85:  # High similarity threshold
                    canonical = entity1 if len(entity1) >= len(entity2) else entity2
                    candidates.append(EntityMergeCandidate(
                        entities=[entity1, entity2],
                        canonical_form=canonical,
                        confidence=similarity,
                        merge_reason="string_similarity_fallback"
                    ))

        return candidates


class EntityNormalizer:
    """Advanced entity normalizer with LLM-based multilingual support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize entity normalizer.

        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}

        # Configuration
        self.enable_llm_normalization = self.config.get('enable_llm_normalization', True)
        self.enable_rule_based_fallback = self.config.get('enable_rule_based_fallback', True)
        self.supported_languages = self.config.get('supported_languages', ['en', 'es', 'de'])
        self.batch_size = self.config.get('batch_size', 20)
        self.min_confidence = self.config.get('min_confidence', 0.7)

        # Initialize LLM agent
        if self.enable_llm_normalization:
            self.llm_agent = EntityNormalizationAgent()
        else:
            self.llm_agent = None

        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="entity_norm")

        # Compiled patterns for rule-based normalization
        self._normalization_patterns = self._compile_normalization_patterns()

        # Language-specific normalizers
        self._language_normalizers = self._setup_language_normalizers()

        logger.info(
            "Entity normalizer initialized",
            enable_llm_normalization=self.enable_llm_normalization,
            supported_languages=self.supported_languages,
            batch_size=self.batch_size
        )
    
    def _compile_normalization_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for rule-based normalization."""
        return {
            # English plurals
            'en_plural_s': re.compile(r'\b(\w+)s\b$'),
            'en_plural_es': re.compile(r'\b(\w+)es\b$'),
            'en_plural_ies': re.compile(r'\b(\w+)ies\b$'),
            'en_irregular_plurals': re.compile(r'\b(?:children|people|men|women|feet|teeth|mice|geese)\b', re.IGNORECASE),
            
            # Spanish plurals
            'es_plural_s': re.compile(r'\b(\w+)s\b$'),
            'es_plural_es': re.compile(r'\b(\w+)es\b$'),
            
            # German plurals (simplified)
            'de_plural_e': re.compile(r'\b(\w+)e\b$'),
            'de_plural_en': re.compile(r'\b(\w+)en\b$'),
            'de_plural_er': re.compile(r'\b(\w+)er\b$'),
            
            # Common verb forms
            'en_ing': re.compile(r'\b(\w+)ing\b$'),
            'en_ed': re.compile(r'\b(\w+)ed\b$'),
            'es_ando': re.compile(r'\b(\w+)ando\b$'),
            'es_iendo': re.compile(r'\b(\w+)iendo\b$'),
            'de_end': re.compile(r'\b(\w+)end\b$'),
            
            # Articles and prepositions
            'articles': re.compile(r'\b(?:the|a|an|el|la|los|las|un|una|der|die|das|ein|eine)\b\s+', re.IGNORECASE),
            'prepositions': re.compile(r'\b(?:of|in|on|at|by|for|with|from|to|de|en|con|por|para|von|mit|zu|auf|in)\b\s+', re.IGNORECASE),
            
            # Whitespace normalization
            'whitespace': re.compile(r'\s+'),
        }
    
    def _setup_language_normalizers(self) -> Dict[str, Callable[[str], str]]:
        """Setup language-specific normalization functions."""
        return {
            'en': self._normalize_english,
            'es': self._normalize_spanish,
            'de': self._normalize_german,
        }
    
    async def normalize_entities(
        self, 
        entities: List[str], 
        entity_types: Optional[List[str]] = None,
        source_doc_id: Optional[str] = None
    ) -> List[NormalizedEntity]:
        """Normalize a list of entities.
        
        Args:
            entities: List of entity texts to normalize
            entity_types: Optional list of entity types (same length as entities)
            source_doc_id: Optional source document ID
            
        Returns:
            List of normalized entities
            
        Raises:
            ProcessingError: If normalization fails
        """
        if not entities:
            return []
        
        try:
            logger.debug(
                "Starting entity normalization",
                entity_count=len(entities),
                source_doc_id=source_doc_id
            )
            
            # Prepare entity data
            entity_data = []
            for i, entity_text in enumerate(entities):
                entity_type = entity_types[i] if entity_types and i < len(entity_types) else None
                entity_data.append({
                    'text': entity_text,
                    'type': entity_type,
                    'index': i
                })
            
            # Process in batches
            normalized_entities = []
            for i in range(0, len(entity_data), self.batch_size):
                batch = entity_data[i:i + self.batch_size]
                batch_normalized = await self._normalize_batch(batch, source_doc_id)
                normalized_entities.extend(batch_normalized)
            
            logger.info(
                "Entity normalization completed",
                input_entities=len(entities),
                normalized_entities=len(normalized_entities),
                source_doc_id=source_doc_id
            )
            
            return normalized_entities
            
        except Exception as e:
            logger.error(
                "Entity normalization failed",
                error=str(e),
                error_type=type(e).__name__,
                entity_count=len(entities),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Entity normalization failed: {e}")
    
    async def _normalize_batch(
        self, 
        entity_batch: List[Dict[str, Any]], 
        source_doc_id: Optional[str] = None
    ) -> List[NormalizedEntity]:
        """Normalize a batch of entities."""
        try:
            # Try LLM normalization first if enabled
            if self.enable_llm_normalization and self.llm_agent:
                try:
                    llm_results = await self._normalize_with_llm(entity_batch)
                    if llm_results:
                        return llm_results
                except Exception as e:
                    logger.warning("LLM normalization failed, falling back to rule-based", error=str(e))
            
            # Fall back to rule-based normalization
            if self.enable_rule_based_fallback:
                return await self._normalize_with_rules(entity_batch)
            else:
                # If no fallback, return original entities as normalized
                return [
                    NormalizedEntity(
                        original_text=entity['text'],
                        normalized_text=entity['text'],
                        canonical_form=entity['text'],
                        entity_type=entity.get('type'),
                        confidence=0.5,
                        normalization_method="none"
                    )
                    for entity in entity_batch
                ]
                
        except Exception as e:
            logger.error("Batch normalization failed", error=str(e))
            # Return original entities as fallback
            return [
                NormalizedEntity(
                    original_text=entity['text'],
                    normalized_text=entity['text'],
                    canonical_form=entity['text'],
                    entity_type=entity.get('type'),
                    confidence=0.3,
                    normalization_method="error_fallback"
                )
                for entity in entity_batch
            ]
    
    async def _normalize_with_llm(self, entity_batch: List[Dict[str, Any]]) -> Optional[List[NormalizedEntity]]:
        """Normalize entities using LLM."""
        try:
            # Prepare prompt
            entity_texts = [entity['text'] for entity in entity_batch]
            entity_types = [entity.get('type', 'UNKNOWN') for entity in entity_batch]
            
            prompt = f"""Normalize the following entities to their canonical, singular, non-conjugated forms:

Entities to normalize:
{chr(10).join(f"{i+1}. {text} (type: {entity_type})" for i, (text, entity_type) in enumerate(zip(entity_texts, entity_types)))}

Provide normalized forms that:
1. Are singular (not plural)
2. Are in base form (not conjugated)
3. Preserve proper nouns correctly
4. Handle multiple languages appropriately
5. Remove unnecessary articles/prepositions"""
            
            # Run LLM normalization
            result = await self.llm_agent.run(prompt)
            
            if result and result.entities:
                # Validate and return results
                validated_entities = []
                for i, normalized in enumerate(result.entities):
                    if i < len(entity_batch) and normalized.confidence >= self.min_confidence:
                        # Add metadata from original entity
                        normalized.entity_type = entity_batch[i].get('type')
                        normalized.normalization_method = "llm"
                        validated_entities.append(normalized)
                
                return validated_entities
            
        except Exception as e:
            logger.warning("LLM normalization failed", error=str(e))
        
        return None
    
    async def _normalize_with_rules(self, entity_batch: List[Dict[str, Any]]) -> List[NormalizedEntity]:
        """Normalize entities using rule-based approach."""
        def normalize_batch_sync():
            normalized_entities = []
            
            for entity in entity_batch:
                entity_text = entity['text']
                entity_type = entity.get('type')
                
                # Detect language (simple heuristic)
                language = self._detect_language(entity_text)
                
                # Apply language-specific normalization
                if language in self._language_normalizers:
                    normalized_text = self._language_normalizers[language](entity_text)
                else:
                    normalized_text = self._normalize_generic(entity_text)
                
                # Generate variations
                variations = self._generate_variations(normalized_text, language)
                
                # Calculate confidence based on changes made
                confidence = self._calculate_rule_confidence(entity_text, normalized_text)
                
                normalized_entity = NormalizedEntity(
                    original_text=entity_text,
                    normalized_text=normalized_text,
                    canonical_form=normalized_text,
                    language=language,
                    entity_type=entity_type,
                    confidence=confidence,
                    variations=variations,
                    normalization_method="rule_based",
                    metadata={
                        'detected_language': language,
                        'changes_made': entity_text != normalized_text
                    }
                )
                
                normalized_entities.append(normalized_entity)
            
            return normalized_entities
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, normalize_batch_sync
        )
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Simple language detection based on patterns."""
        text_lower = text.lower()
        
        # German indicators
        if any(pattern in text_lower for pattern in ['ß', 'ä', 'ö', 'ü', 'der ', 'die ', 'das ']):
            return 'de'
        
        # Spanish indicators
        if any(pattern in text_lower for pattern in ['ñ', 'á', 'é', 'í', 'ó', 'ú', 'el ', 'la ', 'los ', 'las ']):
            return 'es'
        
        # Default to English
        return 'en'
    
    def _normalize_english(self, text: str) -> str:
        """Normalize English entities."""
        normalized = text.strip()
        
        # Remove articles
        normalized = self._normalization_patterns['articles'].sub('', normalized)
        
        # Handle irregular plurals
        irregular_map = {
            'children': 'child',
            'people': 'person',
            'men': 'man',
            'women': 'woman',
            'feet': 'foot',
            'teeth': 'tooth',
            'mice': 'mouse',
            'geese': 'goose'
        }
        
        words = normalized.split()
        normalized_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in irregular_map:
                normalized_words.append(irregular_map[word_lower])
            elif word_lower.endswith('ies'):
                normalized_words.append(word[:-3] + 'y')
            elif word_lower.endswith('es') and len(word) > 3:
                normalized_words.append(word[:-2])
            elif word_lower.endswith('s') and len(word) > 2 and not word_lower.endswith('ss'):
                normalized_words.append(word[:-1])
            elif word_lower.endswith('ing') and len(word) > 4:
                normalized_words.append(word[:-3])
            elif word_lower.endswith('ed') and len(word) > 3:
                normalized_words.append(word[:-2])
            else:
                normalized_words.append(word)
        
        normalized = ' '.join(normalized_words)
        
        # Normalize whitespace
        normalized = self._normalization_patterns['whitespace'].sub(' ', normalized)
        
        return normalized.strip()
    
    def _normalize_spanish(self, text: str) -> str:
        """Normalize Spanish entities."""
        normalized = text.strip()
        
        # Remove articles
        normalized = self._normalization_patterns['articles'].sub('', normalized)
        
        # Simple Spanish plural handling
        words = normalized.split()
        normalized_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower.endswith('es') and len(word) > 3:
                normalized_words.append(word[:-2])
            elif word_lower.endswith('s') and len(word) > 2:
                normalized_words.append(word[:-1])
            elif word_lower.endswith('ando') and len(word) > 5:
                normalized_words.append(word[:-4] + 'ar')
            elif word_lower.endswith('iendo') and len(word) > 6:
                normalized_words.append(word[:-5] + 'ir')
            else:
                normalized_words.append(word)
        
        normalized = ' '.join(normalized_words)
        normalized = self._normalization_patterns['whitespace'].sub(' ', normalized)
        
        return normalized.strip()
    
    def _normalize_german(self, text: str) -> str:
        """Normalize German entities."""
        normalized = text.strip()
        
        # Remove articles
        normalized = self._normalization_patterns['articles'].sub('', normalized)
        
        # Simple German plural handling
        words = normalized.split()
        normalized_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower.endswith('en') and len(word) > 3:
                normalized_words.append(word[:-2])
            elif word_lower.endswith('er') and len(word) > 3:
                normalized_words.append(word[:-2])
            elif word_lower.endswith('e') and len(word) > 2:
                normalized_words.append(word[:-1])
            elif word_lower.endswith('end') and len(word) > 4:
                normalized_words.append(word[:-3] + 'en')
            else:
                normalized_words.append(word)
        
        normalized = ' '.join(normalized_words)
        normalized = self._normalization_patterns['whitespace'].sub(' ', normalized)
        
        return normalized.strip()
    
    def _normalize_generic(self, text: str) -> str:
        """Generic normalization for unknown languages."""
        normalized = text.strip()
        
        # Remove common articles and prepositions
        normalized = self._normalization_patterns['articles'].sub('', normalized)
        normalized = self._normalization_patterns['prepositions'].sub('', normalized)
        
        # Normalize whitespace
        normalized = self._normalization_patterns['whitespace'].sub(' ', normalized)
        
        return normalized.strip()
    
    def _generate_variations(self, normalized_text: str, language: Optional[str] = None) -> List[str]:
        """Generate variations of the normalized entity."""
        variations = [normalized_text]
        
        # Add case variations
        variations.append(normalized_text.lower())
        variations.append(normalized_text.upper())
        variations.append(normalized_text.title())
        
        # Add plural forms based on language
        if language == 'en':
            variations.extend([
                normalized_text + 's',
                normalized_text + 'es',
                normalized_text[:-1] + 'ies' if normalized_text.endswith('y') else normalized_text + 'ies'
            ])
        elif language == 'es':
            variations.extend([
                normalized_text + 's',
                normalized_text + 'es'
            ])
        elif language == 'de':
            variations.extend([
                normalized_text + 'e',
                normalized_text + 'en',
                normalized_text + 'er'
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen and var.strip():
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations
    
    def _calculate_rule_confidence(self, original: str, normalized: str) -> float:
        """Calculate confidence for rule-based normalization."""
        if original == normalized:
            return 1.0
        
        # Base confidence for rule-based normalization
        confidence = 0.8
        
        # Adjust based on the type of changes
        if len(normalized) < len(original):
            # Likely removed plural/conjugation - good
            confidence += 0.1
        
        if normalized.lower() == original.lower():
            # Only case changes - high confidence
            confidence += 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Entity normalizer closed")
        except Exception as e:
            logger.warning("Error during entity normalizer cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
