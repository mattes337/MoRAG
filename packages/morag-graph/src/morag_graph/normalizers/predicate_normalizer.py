"""Predicate normalization and standardization for OpenIE relationships."""

import asyncio
import re
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from morag_core.ai import MoRAGBaseAgent
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class RelationshipType(Enum):
    """Standard relationship types for predicate categorization."""
    IDENTITY = "identity"           # is, are, was, were
    POSSESSION = "possession"       # has, have, owns, possesses
    LOCATION = "location"          # located in, based in, at
    EMPLOYMENT = "employment"      # works at, employed by
    MEMBERSHIP = "membership"      # member of, part of, belongs to
    CREATION = "creation"          # creates, makes, builds, develops
    MANAGEMENT = "management"      # manages, leads, supervises
    EDUCATION = "education"        # teaches, studies, learns
    COMMUNICATION = "communication" # says, tells, speaks, writes
    TEMPORAL = "temporal"          # before, after, during, since
    CAUSAL = "causal"             # causes, leads to, results in
    COMPARISON = "comparison"      # similar to, different from
    ACTION = "action"             # performs, does, executes
    RELATIONSHIP = "relationship"  # related to, connected to
    OTHER = "other"               # catch-all for unclassified


@dataclass
class NormalizedPredicate:
    """Represents a normalized predicate with metadata."""
    original: str
    normalized: str
    canonical_form: str
    relationship_type: RelationshipType
    confidence: float
    language: Optional[str] = None
    variations: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.variations is None:
            self.variations = []
        if self.metadata is None:
            self.metadata = {}


class PredicateNormalizationResult(BaseModel):
    """Result from LLM predicate normalization."""
    predicates: List[Dict[str, Any]]


class PredicateNormalizationAgent(MoRAGBaseAgent[PredicateNormalizationResult]):
    """PydanticAI agent for predicate normalization."""
    
    def __init__(self, **kwargs):
        super().__init__(
            result_type=PredicateNormalizationResult,
            system_prompt="""You are an expert predicate normalizer for knowledge graphs. Your task is to normalize relationship predicates to their canonical, standardized forms while preserving semantic meaning.

Guidelines:
1. Convert verbose predicates to concise, standard forms
2. Use consistent verb forms (present tense, active voice when possible)
3. Handle multiple languages (English, Spanish, German)
4. Group semantically similar predicates under canonical forms
5. Maintain relationship semantics and directionality

Standard canonical forms to prefer:
- Identity: "is", "are" 
- Employment: "works_at", "employed_by"
- Location: "located_in", "based_in"
- Possession: "has", "owns"
- Creation: "creates", "develops"
- Management: "manages", "leads"

For each predicate, provide:
- original: The input predicate
- normalized: The normalized form
- canonical_form: The canonical representation
- relationship_type: The type of relationship
- confidence: Confidence in normalization (0.0-1.0)
- language: Detected language (en, es, de, or null)""",
            **kwargs
        )


class PredicateNormalizer:
    """Advanced predicate normalizer with multilingual support and relationship categorization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize predicate normalizer.
        
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
            self.llm_agent = PredicateNormalizationAgent()
        else:
            self.llm_agent = None
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="predicate_norm")
        
        # Predicate normalization mappings
        self._normalization_mappings = self._build_normalization_mappings()
        
        # Relationship type mappings
        self._relationship_mappings = self._build_relationship_mappings()
        
        # Compiled patterns for rule-based normalization
        self._normalization_patterns = self._compile_normalization_patterns()
        
        logger.info(
            "Predicate normalizer initialized",
            enable_llm_normalization=self.enable_llm_normalization,
            supported_languages=self.supported_languages,
            batch_size=self.batch_size
        )
    
    def _build_normalization_mappings(self) -> Dict[str, Dict[str, str]]:
        """Build predicate normalization mappings for different languages."""
        return {
            'en': {
                # Identity
                'is a': 'is',
                'are a': 'are',
                'was a': 'was',
                'were a': 'were',
                'being': 'is',
                'becomes': 'becomes',
                
                # Employment
                'works for': 'works_at',
                'employed by': 'works_at',
                'working at': 'works_at',
                'working for': 'works_at',
                'employee of': 'works_at',
                'staff at': 'works_at',
                
                # Location
                'located in': 'located_in',
                'situated in': 'located_in',
                'based in': 'based_in',
                'positioned in': 'located_in',
                'found in': 'located_in',
                'resides in': 'located_in',
                
                # Possession
                'owns': 'owns',
                'possesses': 'owns',
                'has got': 'has',
                'is owner of': 'owns',
                'belongs to': 'belongs_to',
                
                # Membership
                'member of': 'member_of',
                'part of': 'part_of',
                'component of': 'part_of',
                'element of': 'part_of',
                'belongs to': 'belongs_to',
                
                # Creation
                'creates': 'creates',
                'makes': 'creates',
                'builds': 'creates',
                'develops': 'develops',
                'produces': 'creates',
                'manufactures': 'creates',
                'constructs': 'creates',
                
                # Management
                'manages': 'manages',
                'leads': 'leads',
                'supervises': 'manages',
                'oversees': 'manages',
                'directs': 'manages',
                'heads': 'leads',
                'is head of': 'leads',
                'is manager of': 'manages',
                
                # Communication
                'says': 'says',
                'tells': 'tells',
                'speaks': 'speaks',
                'talks': 'speaks',
                'communicates': 'communicates',
                'writes': 'writes',
                'publishes': 'publishes',
                
                # Temporal
                'happens before': 'before',
                'occurs before': 'before',
                'happens after': 'after',
                'occurs after': 'after',
                'takes place during': 'during',
                'happens during': 'during',
                
                # Education
                'teaches': 'teaches',
                'instructs': 'teaches',
                'educates': 'teaches',
                'studies': 'studies',
                'learns': 'learns',
                'researches': 'researches',
            },
            'es': {
                # Spanish mappings
                'es': 'is',
                'son': 'are',
                'era': 'was',
                'eran': 'were',
                'trabaja en': 'works_at',
                'trabaja para': 'works_at',
                'empleado de': 'works_at',
                'ubicado en': 'located_in',
                'situado en': 'located_in',
                'tiene': 'has',
                'posee': 'owns',
                'pertenece a': 'belongs_to',
                'miembro de': 'member_of',
                'parte de': 'part_of',
                'crea': 'creates',
                'hace': 'creates',
                'construye': 'creates',
                'desarrolla': 'develops',
                'maneja': 'manages',
                'dirige': 'leads',
                'supervisa': 'manages',
                'enseña': 'teaches',
                'estudia': 'studies',
                'aprende': 'learns',
                'dice': 'says',
                'habla': 'speaks',
                'escribe': 'writes',
            },
            'de': {
                # German mappings
                'ist': 'is',
                'sind': 'are',
                'war': 'was',
                'waren': 'were',
                'arbeitet bei': 'works_at',
                'arbeitet für': 'works_at',
                'angestellt bei': 'works_at',
                'befindet sich in': 'located_in',
                'liegt in': 'located_in',
                'hat': 'has',
                'besitzt': 'owns',
                'gehört zu': 'belongs_to',
                'mitglied von': 'member_of',
                'teil von': 'part_of',
                'erstellt': 'creates',
                'macht': 'creates',
                'baut': 'creates',
                'entwickelt': 'develops',
                'verwaltet': 'manages',
                'führt': 'leads',
                'überwacht': 'manages',
                'lehrt': 'teaches',
                'studiert': 'studies',
                'lernt': 'learns',
                'sagt': 'says',
                'spricht': 'speaks',
                'schreibt': 'writes',
            }
        }
    
    def _build_relationship_mappings(self) -> Dict[str, RelationshipType]:
        """Build mappings from canonical predicates to relationship types."""
        return {
            # Identity
            'is': RelationshipType.IDENTITY,
            'are': RelationshipType.IDENTITY,
            'was': RelationshipType.IDENTITY,
            'were': RelationshipType.IDENTITY,
            'becomes': RelationshipType.IDENTITY,
            
            # Possession
            'has': RelationshipType.POSSESSION,
            'owns': RelationshipType.POSSESSION,
            'possesses': RelationshipType.POSSESSION,
            
            # Location
            'located_in': RelationshipType.LOCATION,
            'based_in': RelationshipType.LOCATION,
            'resides_in': RelationshipType.LOCATION,
            
            # Employment
            'works_at': RelationshipType.EMPLOYMENT,
            'employed_by': RelationshipType.EMPLOYMENT,
            
            # Membership
            'member_of': RelationshipType.MEMBERSHIP,
            'part_of': RelationshipType.MEMBERSHIP,
            'belongs_to': RelationshipType.MEMBERSHIP,
            
            # Creation
            'creates': RelationshipType.CREATION,
            'develops': RelationshipType.CREATION,
            'builds': RelationshipType.CREATION,
            'produces': RelationshipType.CREATION,
            
            # Management
            'manages': RelationshipType.MANAGEMENT,
            'leads': RelationshipType.MANAGEMENT,
            'supervises': RelationshipType.MANAGEMENT,
            'directs': RelationshipType.MANAGEMENT,
            
            # Education
            'teaches': RelationshipType.EDUCATION,
            'studies': RelationshipType.EDUCATION,
            'learns': RelationshipType.EDUCATION,
            'researches': RelationshipType.EDUCATION,
            
            # Communication
            'says': RelationshipType.COMMUNICATION,
            'tells': RelationshipType.COMMUNICATION,
            'speaks': RelationshipType.COMMUNICATION,
            'writes': RelationshipType.COMMUNICATION,
            'publishes': RelationshipType.COMMUNICATION,
            
            # Temporal
            'before': RelationshipType.TEMPORAL,
            'after': RelationshipType.TEMPORAL,
            'during': RelationshipType.TEMPORAL,
            'since': RelationshipType.TEMPORAL,
            
            # Causal
            'causes': RelationshipType.CAUSAL,
            'leads_to': RelationshipType.CAUSAL,
            'results_in': RelationshipType.CAUSAL,
            
            # Action
            'performs': RelationshipType.ACTION,
            'executes': RelationshipType.ACTION,
            'does': RelationshipType.ACTION,
            
            # Relationship
            'related_to': RelationshipType.RELATIONSHIP,
            'connected_to': RelationshipType.RELATIONSHIP,
            'associated_with': RelationshipType.RELATIONSHIP,
        }
    
    def _compile_normalization_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for predicate normalization."""
        return {
            # Remove extra whitespace
            'whitespace': re.compile(r'\s+'),
            
            # Remove common filler words
            'fillers': re.compile(r'\b(?:the|a|an|that|which|who|whom|whose)\b\s*', re.IGNORECASE),
            
            # Normalize verb forms
            'ing_verbs': re.compile(r'\b(\w+)ing\b'),
            'ed_verbs': re.compile(r'\b(\w+)ed\b'),
            'es_verbs': re.compile(r'\b(\w+)es\b'),
            's_verbs': re.compile(r'\b(\w+)s\b'),
            
            # Common predicate patterns
            'is_pattern': re.compile(r'\bis\s+(?:a|an|the)\s+', re.IGNORECASE),
            'has_pattern': re.compile(r'\bhas\s+(?:a|an|the)\s+', re.IGNORECASE),
            'works_pattern': re.compile(r'\bworks?\s+(?:at|for|in)\s+', re.IGNORECASE),
            
            # Punctuation cleanup
            'punctuation': re.compile(r'[^\w\s]'),
        }
    
    async def normalize_predicates(
        self, 
        predicates: List[str],
        source_doc_id: Optional[str] = None
    ) -> List[NormalizedPredicate]:
        """Normalize a list of predicates.
        
        Args:
            predicates: List of predicate strings to normalize
            source_doc_id: Optional source document ID
            
        Returns:
            List of normalized predicates
            
        Raises:
            ProcessingError: If normalization fails
        """
        if not predicates:
            return []
        
        try:
            logger.debug(
                "Starting predicate normalization",
                predicate_count=len(predicates),
                source_doc_id=source_doc_id
            )
            
            # Process in batches
            normalized_predicates = []
            for i in range(0, len(predicates), self.batch_size):
                batch = predicates[i:i + self.batch_size]
                batch_normalized = await self._normalize_batch(batch, source_doc_id)
                normalized_predicates.extend(batch_normalized)
            
            logger.info(
                "Predicate normalization completed",
                input_predicates=len(predicates),
                normalized_predicates=len(normalized_predicates),
                source_doc_id=source_doc_id
            )
            
            return normalized_predicates
            
        except Exception as e:
            logger.error(
                "Predicate normalization failed",
                error=str(e),
                error_type=type(e).__name__,
                predicate_count=len(predicates),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Predicate normalization failed: {e}")
    
    async def _normalize_batch(
        self, 
        predicate_batch: List[str], 
        source_doc_id: Optional[str] = None
    ) -> List[NormalizedPredicate]:
        """Normalize a batch of predicates."""
        try:
            # Try LLM normalization first if enabled
            if self.enable_llm_normalization and self.llm_agent:
                try:
                    llm_results = await self._normalize_with_llm(predicate_batch)
                    if llm_results:
                        return llm_results
                except Exception as e:
                    logger.warning("LLM predicate normalization failed, falling back to rule-based", error=str(e))
            
            # Fall back to rule-based normalization
            if self.enable_rule_based_fallback:
                return await self._normalize_with_rules(predicate_batch)
            else:
                # If no fallback, return original predicates as normalized
                return [
                    NormalizedPredicate(
                        original=predicate,
                        normalized=predicate,
                        canonical_form=predicate,
                        relationship_type=RelationshipType.OTHER,
                        confidence=0.5
                    )
                    for predicate in predicate_batch
                ]
                
        except Exception as e:
            logger.error("Batch predicate normalization failed", error=str(e))
            # Return original predicates as fallback
            return [
                NormalizedPredicate(
                    original=predicate,
                    normalized=predicate,
                    canonical_form=predicate,
                    relationship_type=RelationshipType.OTHER,
                    confidence=0.3,
                    metadata={'error': str(e)}
                )
                for predicate in predicate_batch
            ]

    async def _normalize_with_llm(self, predicate_batch: List[str]) -> Optional[List[NormalizedPredicate]]:
        """Normalize predicates using LLM."""
        try:
            # Prepare prompt
            prompt = f"""Normalize the following predicates to their canonical forms:

Predicates to normalize:
{chr(10).join(f"{i+1}. {predicate}" for i, predicate in enumerate(predicate_batch))}

Provide normalized forms that:
1. Are concise and standardized
2. Use consistent verb forms (present tense, active voice)
3. Group semantically similar predicates
4. Preserve relationship semantics
5. Handle multiple languages appropriately"""

            # Run LLM normalization
            result = await self.llm_agent.run(prompt)

            if result and result.predicates:
                # Convert to NormalizedPredicate objects
                normalized_predicates = []
                for i, pred_data in enumerate(result.predicates):
                    if i < len(predicate_batch):
                        # Determine relationship type
                        canonical = pred_data.get('canonical_form', pred_data.get('normalized', ''))
                        rel_type = self._relationship_mappings.get(canonical, RelationshipType.OTHER)

                        normalized = NormalizedPredicate(
                            original=predicate_batch[i],
                            normalized=pred_data.get('normalized', predicate_batch[i]),
                            canonical_form=canonical,
                            relationship_type=rel_type,
                            confidence=pred_data.get('confidence', 0.8),
                            language=pred_data.get('language'),
                            metadata={'method': 'llm'}
                        )
                        normalized_predicates.append(normalized)

                return normalized_predicates

        except Exception as e:
            logger.warning("LLM predicate normalization failed", error=str(e))

        return None

    async def _normalize_with_rules(self, predicate_batch: List[str]) -> List[NormalizedPredicate]:
        """Normalize predicates using rule-based approach."""
        def normalize_batch_sync():
            normalized_predicates = []

            for predicate in predicate_batch:
                # Detect language
                language = self._detect_language(predicate)

                # Apply normalization
                normalized = self._normalize_single_predicate(predicate, language)

                # Determine relationship type
                rel_type = self._relationship_mappings.get(normalized.canonical_form, RelationshipType.OTHER)

                # Calculate confidence
                confidence = self._calculate_rule_confidence(predicate, normalized.normalized)

                # Generate variations
                variations = self._generate_predicate_variations(normalized.canonical_form, language)

                normalized_predicate = NormalizedPredicate(
                    original=predicate,
                    normalized=normalized.normalized,
                    canonical_form=normalized.canonical_form,
                    relationship_type=rel_type,
                    confidence=confidence,
                    language=language,
                    variations=variations,
                    metadata={
                        'method': 'rule_based',
                        'detected_language': language,
                        'changes_made': predicate != normalized.normalized
                    }
                )

                normalized_predicates.append(normalized_predicate)

            return normalized_predicates

        return await asyncio.get_event_loop().run_in_executor(
            self._executor, normalize_batch_sync
        )

    def _detect_language(self, predicate: str) -> Optional[str]:
        """Simple language detection for predicates."""
        predicate_lower = predicate.lower()

        # German indicators
        if any(word in predicate_lower for word in ['ist', 'sind', 'war', 'waren', 'arbeitet', 'befindet']):
            return 'de'

        # Spanish indicators
        if any(word in predicate_lower for word in ['es', 'son', 'era', 'eran', 'trabaja', 'tiene']):
            return 'es'

        # Default to English
        return 'en'

    def _normalize_single_predicate(self, predicate: str, language: Optional[str] = None) -> NormalizedPredicate:
        """Normalize a single predicate using rules."""
        if not language:
            language = 'en'

        # Clean the predicate
        cleaned = self._clean_predicate(predicate)

        # Apply language-specific mappings
        mappings = self._normalization_mappings.get(language, {})

        # Try exact match first
        if cleaned in mappings:
            canonical = mappings[cleaned]
            return NormalizedPredicate(
                original=predicate,
                normalized=cleaned,
                canonical_form=canonical,
                relationship_type=RelationshipType.OTHER,  # Will be set later
                confidence=0.9
            )

        # Try partial matches
        for pattern, canonical in mappings.items():
            if pattern in cleaned or cleaned in pattern:
                return NormalizedPredicate(
                    original=predicate,
                    normalized=cleaned,
                    canonical_form=canonical,
                    relationship_type=RelationshipType.OTHER,
                    confidence=0.7
                )

        # Apply generic normalization rules
        normalized = self._apply_generic_rules(cleaned)

        return NormalizedPredicate(
            original=predicate,
            normalized=normalized,
            canonical_form=normalized,
            relationship_type=RelationshipType.OTHER,
            confidence=0.6
        )

    def _clean_predicate(self, predicate: str) -> str:
        """Clean predicate text."""
        cleaned = predicate.strip().lower()

        # Remove filler words
        cleaned = self._normalization_patterns['fillers'].sub('', cleaned)

        # Normalize whitespace
        cleaned = self._normalization_patterns['whitespace'].sub(' ', cleaned)

        # Remove punctuation
        cleaned = self._normalization_patterns['punctuation'].sub('', cleaned)

        return cleaned.strip()

    def _apply_generic_rules(self, predicate: str) -> str:
        """Apply generic normalization rules."""
        normalized = predicate

        # Convert verb forms to base form (simplified)
        if normalized.endswith('ing'):
            normalized = normalized[:-3]
        elif normalized.endswith('ed'):
            normalized = normalized[:-2]
        elif normalized.endswith('es'):
            normalized = normalized[:-2]
        elif normalized.endswith('s') and len(normalized) > 2:
            normalized = normalized[:-1]

        # Handle common patterns
        if 'work' in normalized and ('at' in normalized or 'for' in normalized):
            normalized = 'works_at'
        elif 'locate' in normalized and 'in' in normalized:
            normalized = 'located_in'
        elif 'member' in normalized and 'of' in normalized:
            normalized = 'member_of'
        elif 'part' in normalized and 'of' in normalized:
            normalized = 'part_of'

        return normalized

    def _calculate_rule_confidence(self, original: str, normalized: str) -> float:
        """Calculate confidence for rule-based normalization."""
        if original.lower().strip() == normalized:
            return 1.0

        # Base confidence for rule-based normalization
        confidence = 0.7

        # Adjust based on changes
        if len(normalized) < len(original):
            confidence += 0.1  # Likely removed unnecessary words

        if '_' in normalized:
            confidence += 0.1  # Likely standardized compound predicate

        return max(0.0, min(1.0, confidence))

    def _generate_predicate_variations(self, canonical: str, language: Optional[str] = None) -> List[str]:
        """Generate variations of the canonical predicate."""
        variations = [canonical]

        # Add the original form
        variations.append(canonical.replace('_', ' '))

        # Add language-specific variations
        if language == 'en':
            # Add different verb forms
            if canonical.endswith('s'):
                variations.append(canonical[:-1])  # Remove 's'
            else:
                variations.append(canonical + 's')  # Add 's'

            # Add -ing form
            variations.append(canonical + 'ing')

            # Add -ed form
            variations.append(canonical + 'ed')

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen and var.strip():
                seen.add(var)
                unique_variations.append(var)

        return unique_variations

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Predicate normalizer closed")
        except Exception as e:
            logger.warning("Error during predicate normalizer cleanup", error=str(e))

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
