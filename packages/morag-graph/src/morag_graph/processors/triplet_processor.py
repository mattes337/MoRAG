"""Triplet processing and validation for OpenIE pipeline."""

import asyncio
import re
from typing import List, Dict, Any, Optional, NamedTuple, Set
from concurrent.futures import ThreadPoolExecutor
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..services.openie_service import OpenIETriplet
from .sentence_processor import ProcessedSentence

logger = structlog.get_logger(__name__)


class ValidatedTriplet(NamedTuple):
    """Represents a validated OpenIE triplet with quality metrics."""
    subject: str
    predicate: str
    object: str
    confidence: float
    sentence: str
    sentence_id: str
    validation_score: float
    validation_flags: Set[str]
    source_doc_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class TripletProcessor:
    """Advanced triplet processing and validation for OpenIE results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize triplet processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.min_validation_score = self.config.get('min_validation_score', 0.6)
        self.enable_validation = self.config.get('enable_validation', True)
        self.enable_normalization = self.config.get('enable_normalization', True)
        self.max_subject_length = self.config.get('max_subject_length', 100)
        self.max_predicate_length = self.config.get('max_predicate_length', 50)
        self.max_object_length = self.config.get('max_object_length', 100)
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="triplet_proc")
        
        # Validation patterns
        self._validation_patterns = self._compile_validation_patterns()
        
        # Quality indicators
        self._quality_indicators = self._compile_quality_indicators()
        
        logger.info(
            "Triplet processor initialized",
            min_confidence=self.min_confidence,
            min_validation_score=self.min_validation_score,
            enable_validation=self.enable_validation,
            enable_normalization=self.enable_normalization
        )
    
    def _compile_validation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for triplet validation."""
        return {
            # Valid entity patterns (letters, numbers, spaces, basic punctuation)
            'valid_entity': re.compile(r'^[a-zA-Z0-9\s\-\.\,\'\"]+$'),
            
            # Invalid patterns that indicate poor extraction
            'invalid_chars': re.compile(r'[<>{}[\]\\|`~@#$%^&*()+=]'),
            'only_punctuation': re.compile(r'^[^\w\s]+$'),
            'only_numbers': re.compile(r'^\d+$'),
            'url_pattern': re.compile(r'https?://|www\.'),
            'email_pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Predicate quality patterns
            'good_predicate': re.compile(r'\b(?:is|are|was|were|has|have|had|does|do|did|will|would|can|could|should|may|might|must|works|lives|owns|creates|makes|builds|develops|manages|leads|teaches|studies|researches|writes|reads|plays|performs|travels|visits|meets|knows|loves|likes|hates|believes|thinks|feels|says|tells|speaks|talks|gives|takes|gets|puts|goes|comes|stays|leaves|starts|stops|continues|finishes|begins|ends)\b', re.IGNORECASE),
            'weak_predicate': re.compile(r'^\s*(?:of|in|on|at|by|for|with|from|to|and|or|but|the|a|an)\s*$', re.IGNORECASE),
            
            # Subject/Object quality patterns
            'proper_noun': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
            'common_noun': re.compile(r'\b(?:person|people|man|woman|child|company|organization|place|city|country|product|service|system|method|process|result|study|research|project|work|job|task|problem|solution|idea|concept|theory|fact|data|information|knowledge|skill|ability|experience|opportunity|challenge|goal|objective|plan|strategy|approach|technique|tool|resource|material|equipment|technology|software|hardware|application|program|system|network|database|website|platform|service|product|device|machine|instrument|vehicle|building|structure|facility|location|area|region|zone|sector|industry|market|business|economy|finance|money|cost|price|value|benefit|advantage|disadvantage|risk|threat|opportunity|challenge)\b', re.IGNORECASE),
            
            # Temporal and spatial indicators
            'temporal': re.compile(r'\b(?:today|yesterday|tomorrow|now|then|before|after|during|while|when|since|until|always|never|sometimes|often|rarely|frequently|recently|currently|previously|formerly|originally|initially|finally|eventually|soon|later|earlier|ago|years?|months?|weeks?|days?|hours?|minutes?|seconds?|morning|afternoon|evening|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE),
            'spatial': re.compile(r'\b(?:here|there|where|above|below|under|over|inside|outside|within|beyond|near|far|close|distant|left|right|north|south|east|west|up|down|forward|backward|ahead|behind|beside|next|between|among|around|through|across|along|toward|away|home|office|school|university|hospital|store|restaurant|park|street|road|avenue|building|room|floor|city|town|village|country|state|province|region|area|place|location|site|position|spot|point)\b', re.IGNORECASE),
        }
    
    def _compile_quality_indicators(self) -> Dict[str, float]:
        """Define quality score weights for different indicators."""
        return {
            'has_proper_noun_subject': 0.2,
            'has_proper_noun_object': 0.2,
            'has_good_predicate': 0.3,
            'no_weak_predicate': 0.1,
            'appropriate_length': 0.1,
            'no_invalid_chars': 0.1,
            'has_temporal_context': 0.05,
            'has_spatial_context': 0.05,
            'subject_object_different': 0.1,
            'meaningful_entities': 0.2,
        }
    
    async def process_triplets(
        self, 
        triplets: List[OpenIETriplet], 
        sentences: Optional[List[ProcessedSentence]] = None,
        source_doc_id: Optional[str] = None
    ) -> List[ValidatedTriplet]:
        """Process and validate OpenIE triplets.
        
        Args:
            triplets: List of raw OpenIE triplets
            sentences: Optional list of processed sentences for context
            source_doc_id: Optional source document ID
            
        Returns:
            List of validated triplets
            
        Raises:
            ProcessingError: If processing fails
        """
        if not triplets:
            return []
        
        try:
            logger.debug(
                "Starting triplet processing",
                triplet_count=len(triplets),
                source_doc_id=source_doc_id
            )
            
            # Create sentence lookup for context
            sentence_lookup = {}
            if sentences:
                sentence_lookup = {s.text: s for s in sentences}
            
            # Process triplets
            validated_triplets = []
            for triplet in triplets:
                try:
                    validated = await self._process_single_triplet(
                        triplet, sentence_lookup, source_doc_id
                    )
                    if validated:
                        validated_triplets.append(validated)
                except Exception as e:
                    logger.warning(
                        "Failed to process triplet",
                        triplet=f"{triplet.subject} | {triplet.predicate} | {triplet.object}",
                        error=str(e)
                    )
            
            # Filter by validation score
            if self.enable_validation:
                filtered_triplets = [
                    t for t in validated_triplets 
                    if t.validation_score >= self.min_validation_score
                ]
            else:
                filtered_triplets = validated_triplets
            
            logger.info(
                "Triplet processing completed",
                input_triplets=len(triplets),
                validated_triplets=len(validated_triplets),
                filtered_triplets=len(filtered_triplets),
                source_doc_id=source_doc_id
            )
            
            return filtered_triplets
            
        except Exception as e:
            logger.error(
                "Triplet processing failed",
                error=str(e),
                error_type=type(e).__name__,
                triplet_count=len(triplets),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Triplet processing failed: {e}")
    
    async def _process_single_triplet(
        self, 
        triplet: OpenIETriplet, 
        sentence_lookup: Dict[str, ProcessedSentence],
        source_doc_id: Optional[str] = None
    ) -> Optional[ValidatedTriplet]:
        """Process a single triplet."""
        def process_sync():
            # Normalize entities if enabled
            if self.enable_normalization:
                subject = self._normalize_entity(triplet.subject)
                predicate = self._normalize_predicate(triplet.predicate)
                obj = self._normalize_entity(triplet.object)
            else:
                subject = triplet.subject.strip()
                predicate = triplet.predicate.strip()
                obj = triplet.object.strip()
            
            # Basic validation
            if not (subject and predicate and obj):
                return None
            
            # Length validation
            if (len(subject) > self.max_subject_length or 
                len(predicate) > self.max_predicate_length or 
                len(obj) > self.max_object_length):
                return None
            
            # Calculate validation score
            validation_score, validation_flags = self._calculate_validation_score(
                subject, predicate, obj, triplet.sentence
            )
            
            # Get sentence context
            sentence_context = sentence_lookup.get(triplet.sentence)
            sentence_id = sentence_context.sentence_id if sentence_context else f"unknown_{hash(triplet.sentence) % 10000}"
            
            # Create metadata
            metadata = {
                'original_subject': triplet.subject,
                'original_predicate': triplet.predicate,
                'original_object': triplet.object,
                'extraction_confidence': triplet.confidence,
                'sentence_length': len(triplet.sentence),
                'subject_length': len(subject),
                'predicate_length': len(predicate),
                'object_length': len(obj),
            }
            
            if sentence_context:
                metadata.update({
                    'sentence_quality_score': sentence_context.quality_score,
                    'sentence_metadata': sentence_context.metadata
                })
            
            return ValidatedTriplet(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=triplet.confidence,
                sentence=triplet.sentence,
                sentence_id=sentence_id,
                validation_score=validation_score,
                validation_flags=validation_flags,
                source_doc_id=source_doc_id,
                metadata=metadata
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, process_sync
        )
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity text."""
        # Basic normalization
        normalized = entity.strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing punctuation
        normalized = normalized.strip('.,;:!?()[]{}"\'-')
        
        # Capitalize proper nouns
        if self._validation_patterns['proper_noun'].match(normalized):
            normalized = normalized.title()
        
        return normalized
    
    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate text."""
        # Basic normalization
        normalized = predicate.strip().lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing punctuation
        normalized = normalized.strip('.,;:!?()[]{}"\'-')
        
        # Common predicate normalizations
        normalizations = {
            'is a': 'is',
            'are a': 'are',
            'was a': 'was',
            'were a': 'were',
            'works for': 'works at',
            'employed by': 'works at',
            'member of': 'belongs to',
            'part of': 'belongs to',
            'located in': 'located at',
            'situated in': 'located at',
        }
        
        for old, new in normalizations.items():
            if old in normalized:
                normalized = normalized.replace(old, new)
        
        return normalized
    
    def _calculate_validation_score(self, subject: str, predicate: str, obj: str, sentence: str) -> tuple[float, Set[str]]:
        """Calculate validation score and flags for a triplet."""
        score = 0.0
        flags = set()
        
        # Check for proper nouns in subject
        if self._validation_patterns['proper_noun'].search(subject):
            score += self._quality_indicators['has_proper_noun_subject']
            flags.add('proper_noun_subject')
        
        # Check for proper nouns in object
        if self._validation_patterns['proper_noun'].search(obj):
            score += self._quality_indicators['has_proper_noun_object']
            flags.add('proper_noun_object')
        
        # Check for good predicate
        if self._validation_patterns['good_predicate'].search(predicate):
            score += self._quality_indicators['has_good_predicate']
            flags.add('good_predicate')
        
        # Check for weak predicate (penalty)
        if not self._validation_patterns['weak_predicate'].match(predicate):
            score += self._quality_indicators['no_weak_predicate']
        else:
            flags.add('weak_predicate')
        
        # Check appropriate length
        if (5 <= len(subject) <= 50 and 
            2 <= len(predicate) <= 30 and 
            5 <= len(obj) <= 50):
            score += self._quality_indicators['appropriate_length']
            flags.add('appropriate_length')
        
        # Check for invalid characters
        if (not self._validation_patterns['invalid_chars'].search(subject) and
            not self._validation_patterns['invalid_chars'].search(predicate) and
            not self._validation_patterns['invalid_chars'].search(obj)):
            score += self._quality_indicators['no_invalid_chars']
            flags.add('no_invalid_chars')
        
        # Check for temporal context
        if self._validation_patterns['temporal'].search(sentence):
            score += self._quality_indicators['has_temporal_context']
            flags.add('temporal_context')
        
        # Check for spatial context
        if self._validation_patterns['spatial'].search(sentence):
            score += self._quality_indicators['has_spatial_context']
            flags.add('spatial_context')
        
        # Check that subject and object are different
        if subject.lower() != obj.lower():
            score += self._quality_indicators['subject_object_different']
            flags.add('different_entities')
        
        # Check for meaningful entities (not just punctuation or numbers)
        if (not self._validation_patterns['only_punctuation'].match(subject) and
            not self._validation_patterns['only_punctuation'].match(obj) and
            not self._validation_patterns['only_numbers'].match(subject) and
            not self._validation_patterns['only_numbers'].match(obj)):
            score += self._quality_indicators['meaningful_entities']
            flags.add('meaningful_entities')
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score, flags
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Triplet processor closed")
        except Exception as e:
            logger.warning("Error during triplet processor cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
