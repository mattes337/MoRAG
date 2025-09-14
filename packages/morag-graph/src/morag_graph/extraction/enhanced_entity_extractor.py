"""Enhanced entity extractor with multi-round gleaning and confidence scoring."""

import structlog
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    lx = None

from ..models import Entity
from .entity_extractor import EntityExtractor
from .entity_normalizer import LLMEntityNormalizer
from ..utils.retry_utils import retry_with_exponential_backoff
from ..utils.quota_retry import never_fail_extraction

logger = structlog.get_logger(__name__)


@dataclass
class ConfidenceEntity:
    """Entity with confidence scoring and extraction metadata."""
    entity: Entity
    confidence: float
    extraction_round: int
    gleaning_strategy: str
    context_score: float = 0.0
    
    def __post_init__(self):
        """Update entity confidence with calculated confidence."""
        self.entity.confidence = self.confidence


@dataclass
class GleaningResult:
    """Result of a gleaning round."""
    entities: List[ConfidenceEntity]
    missed_entities_detected: bool
    confidence_threshold_met: bool
    processing_time: float


class GleaningStrategy:
    """Base class for entity gleaning strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(strategy=name)
    
    async def extract(
        self,
        text: str,
        existing_entities: List[ConfidenceEntity],
        extractor: EntityExtractor,
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities using this strategy."""
        raise NotImplementedError


class BasicGleaningStrategy(GleaningStrategy):
    """Basic gleaning strategy using standard extraction."""
    
    def __init__(self):
        super().__init__("basic")
    
    async def extract(
        self,
        text: str,
        existing_entities: List[ConfidenceEntity],
        extractor: EntityExtractor,
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities using basic strategy."""
        return await extractor.extract(text, source_doc_id)


class ContextualGleaningStrategy(GleaningStrategy):
    """Contextual gleaning strategy that focuses on missed entity types."""
    
    def __init__(self):
        super().__init__("contextual")
    
    async def extract(
        self,
        text: str,
        existing_entities: List[ConfidenceEntity],
        extractor: EntityExtractor,
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities focusing on missing types."""
        # Analyze existing entity types
        existing_types = {e.entity.type for e in existing_entities}
        
        # Create focused prompt for missing types
        missing_types = self._identify_missing_types(text, existing_types)
        
        if not missing_types:
            return []
        
        # Create temporary extractor with focused prompt
        focused_extractor = self._create_focused_extractor(extractor, missing_types)
        return await focused_extractor.extract(text, source_doc_id)
    
    def _identify_missing_types(self, text: str, existing_types: Set[str]) -> List[str]:
        """Identify potentially missing entity types from text."""
        # Common entity types that might be missed
        all_types = {
            "person", "organization", "location", "concept", 
            "object", "event", "technology", "method", "process"
        }
        return list(all_types - existing_types)
    
    def _create_focused_extractor(
        self, 
        base_extractor: EntityExtractor, 
        focus_types: List[str]
    ) -> EntityExtractor:
        """Create extractor focused on specific entity types."""
        # Create new extractor with focused entity types
        focused_types = {t: f"Focus on {t} entities" for t in focus_types}
        
        return EntityExtractor(
            min_confidence=base_extractor.min_confidence * 0.8,  # Lower threshold for focused search
            chunk_size=base_extractor.chunk_size,
            dynamic_types=True,
            entity_types=focused_types,
            language=base_extractor.language,
            model_id=base_extractor.model_id,
            api_key=base_extractor.api_key,
            max_workers=base_extractor.max_workers,
            extraction_passes=2,  # Fewer passes for focused extraction
            domain=base_extractor.domain
        )


class SemanticGleaningStrategy(GleaningStrategy):
    """Semantic gleaning strategy that uses entity relationships."""
    
    def __init__(self):
        super().__init__("semantic")
    
    async def extract(
        self,
        text: str,
        existing_entities: List[ConfidenceEntity],
        extractor: EntityExtractor,
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities using semantic relationships."""
        if not existing_entities:
            return []
        
        # Create context from existing entities
        entity_context = self._create_entity_context(existing_entities)
        
        # Use relationship-aware extraction
        return await self._extract_with_context(
            text, entity_context, extractor, source_doc_id
        )
    
    def _create_entity_context(self, entities: List[ConfidenceEntity]) -> str:
        """Create context string from existing entities."""
        entity_names = [e.entity.name for e in entities]
        return f"Known entities: {', '.join(entity_names[:10])}"  # Limit context size
    
    async def _extract_with_context(
        self,
        text: str,
        context: str,
        extractor: EntityExtractor,
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities with additional context."""
        # Prepend context to text for extraction
        contextual_text = f"{context}\n\n{text}"
        entities = await extractor.extract(contextual_text, source_doc_id)
        
        # Filter out entities that were in the context
        context_names = {name.strip() for name in context.split(":")[1].split(",")} if ":" in context else set()
        return [e for e in entities if e.name not in context_names]


class EntityConfidenceModel:
    """Model for scoring entity confidence."""
    
    def __init__(self):
        self.logger = logger.bind(component="confidence_model")
    
    async def score_entity(
        self,
        entity: Entity,
        text: str,
        existing_entities: List[ConfidenceEntity]
    ) -> float:
        """Score entity confidence based on multiple factors."""
        base_confidence = entity.confidence
        
        # Context score: how well entity fits in text
        context_score = self._calculate_context_score(entity, text)
        
        # Uniqueness score: penalty for duplicates
        uniqueness_score = self._calculate_uniqueness_score(entity, existing_entities)
        
        # Type consistency score
        type_score = self._calculate_type_consistency_score(entity)
        
        # Combine scores with weights
        final_confidence = (
            base_confidence * 0.4 +
            context_score * 0.3 +
            uniqueness_score * 0.2 +
            type_score * 0.1
        )
        
        return min(1.0, max(0.0, final_confidence))
    
    def _calculate_context_score(self, entity: Entity, text: str) -> float:
        """Calculate how well entity fits in context."""
        # Simple implementation: check if entity name appears in text
        entity_name_lower = entity.name.lower()
        text_lower = text.lower()
        
        if entity_name_lower in text_lower:
            # Bonus for exact match
            return 0.9
        
        # Check for partial matches
        words = entity_name_lower.split()
        matches = sum(1 for word in words if word in text_lower)
        
        if matches > 0:
            return 0.5 + (matches / len(words)) * 0.3
        
        return 0.1  # Low score if no matches
    
    def _calculate_uniqueness_score(
        self, 
        entity: Entity, 
        existing_entities: List[ConfidenceEntity]
    ) -> float:
        """Calculate uniqueness score (penalty for duplicates)."""
        entity_name_lower = entity.name.lower()
        
        for existing in existing_entities:
            existing_name_lower = existing.entity.name.lower()
            
            # Exact match penalty
            if entity_name_lower == existing_name_lower:
                return 0.1
            
            # Partial match penalty
            if (entity_name_lower in existing_name_lower or 
                existing_name_lower in entity_name_lower):
                return 0.5
        
        return 1.0  # Full score for unique entities
    
    def _calculate_type_consistency_score(self, entity: Entity) -> float:
        """Calculate type consistency score."""
        # Simple implementation: check if type is reasonable
        if not entity.type or entity.type.lower() in ["unknown", "other"]:
            return 0.3
        
        return 0.8  # Good score for specific types


class EnhancedEntityExtractor:
    """Enhanced entity extractor with multi-round gleaning and confidence scoring."""
    
    def __init__(
        self,
        base_extractor: Optional[EntityExtractor] = None,
        max_rounds: int = 3,
        target_confidence: float = 0.85,
        confidence_threshold: float = 0.6,
        enable_gleaning: bool = True
    ):
        """Initialize enhanced entity extractor.
        
        Args:
            base_extractor: Base EntityExtractor instance
            max_rounds: Maximum gleaning rounds
            target_confidence: Target overall confidence to stop early
            confidence_threshold: Minimum confidence for individual entities
            enable_gleaning: Whether to enable multi-round gleaning
        """
        self.base_extractor = base_extractor or EntityExtractor()
        self.max_rounds = max_rounds
        self.target_confidence = target_confidence
        self.confidence_threshold = confidence_threshold
        self.enable_gleaning = enable_gleaning
        
        self.confidence_model = EntityConfidenceModel()
        self.gleaning_strategies = [
            BasicGleaningStrategy(),
            ContextualGleaningStrategy(),
            SemanticGleaningStrategy()
        ]
        
        self.logger = logger.bind(component="enhanced_entity_extractor")
    
    async def extract_with_gleaning(
        self,
        text: str,
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities with multi-round gleaning."""
        if not self.enable_gleaning:
            return await self.base_extractor.extract(text, source_doc_id)
        
        start_time = time.time()
        all_entities: List[ConfidenceEntity] = []
        
        self.logger.info(
            "Starting multi-round entity extraction",
            max_rounds=self.max_rounds,
            target_confidence=self.target_confidence
        )
        
        for round_num in range(self.max_rounds):
            round_start = time.time()
            
            # Select gleaning strategy for this round
            strategy = self._select_gleaning_strategy(round_num, all_entities)
            
            self.logger.info(
                f"Round {round_num + 1}/{self.max_rounds}",
                strategy=strategy.name,
                existing_entities=len(all_entities)
            )
            
            # Extract entities for this round
            round_entities = await strategy.extract(
                text, all_entities, self.base_extractor, source_doc_id
            )
            
            # Score confidence for new entities
            confident_entities = []
            for entity in round_entities:
                confidence = await self.confidence_model.score_entity(
                    entity, text, all_entities
                )
                
                if confidence >= self.confidence_threshold:
                    confident_entities.append(ConfidenceEntity(
                        entity=entity,
                        confidence=confidence,
                        extraction_round=round_num + 1,
                        gleaning_strategy=strategy.name
                    ))
            
            # Add new entities
            new_entities = self._deduplicate_entities(all_entities + confident_entities)
            new_count = len(new_entities) - len(all_entities)
            all_entities = new_entities
            
            round_time = time.time() - round_start
            
            self.logger.info(
                f"Round {round_num + 1} completed",
                new_entities=new_count,
                total_entities=len(all_entities),
                processing_time=f"{round_time:.2f}s"
            )
            
            # Check if we should stop early
            overall_confidence = self._calculate_overall_confidence(all_entities)
            if overall_confidence >= self.target_confidence:
                self.logger.info(
                    "Target confidence reached, stopping early",
                    confidence=overall_confidence
                )
                break
            
            # Stop if no new entities found
            if new_count == 0:
                self.logger.info("No new entities found, stopping extraction")
                break
        
        total_time = time.time() - start_time
        final_entities = [ce.entity for ce in all_entities]
        
        self.logger.info(
            "Multi-round extraction completed",
            total_entities=len(final_entities),
            total_time=f"{total_time:.2f}s",
            final_confidence=self._calculate_overall_confidence(all_entities)
        )
        
        return final_entities

    def _select_gleaning_strategy(
        self,
        round_num: int,
        existing_entities: List[ConfidenceEntity]
    ) -> GleaningStrategy:
        """Select appropriate gleaning strategy for the round."""
        if round_num == 0:
            return self.gleaning_strategies[0]  # Basic strategy first
        elif round_num == 1 and len(existing_entities) > 0:
            return self.gleaning_strategies[1]  # Contextual strategy
        else:
            return self.gleaning_strategies[2]  # Semantic strategy

    def _deduplicate_entities(
        self,
        entities: List[ConfidenceEntity]
    ) -> List[ConfidenceEntity]:
        """Deduplicate entities by name and type."""
        seen = {}
        deduplicated = []

        for entity in entities:
            key = (entity.entity.name.lower(), entity.entity.type.lower())

            if key not in seen:
                seen[key] = entity
                deduplicated.append(entity)
            else:
                # Keep entity with higher confidence
                if entity.confidence > seen[key].confidence:
                    # Replace in deduplicated list
                    for i, existing in enumerate(deduplicated):
                        if existing is seen[key]:
                            deduplicated[i] = entity
                            seen[key] = entity
                            break

        return deduplicated

    def _calculate_overall_confidence(
        self,
        entities: List[ConfidenceEntity]
    ) -> float:
        """Calculate overall confidence score for all entities."""
        if not entities:
            return 0.0

        # Weighted average based on entity confidence
        total_weight = sum(e.confidence for e in entities)
        weighted_sum = sum(e.confidence ** 2 for e in entities)  # Square for emphasis

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def assess_missed_entities(
        self,
        text: str,
        existing_entities: List[ConfidenceEntity]
    ) -> bool:
        """Assess if there might be missed entities using LLM."""
        if not existing_entities:
            return True  # Always assume missed entities if none found

        # Create assessment prompt
        entity_list = ", ".join([e.entity.name for e in existing_entities[:10]])

        assessment_prompt = f"""
        Analyze this text and determine if there are likely missed entities:

        Text: {text[:500]}...

        Currently identified entities: {entity_list}

        Are there likely important entities (people, places, organizations, concepts)
        that were missed? Consider:
        - Pronouns that might refer to unnamed entities
        - Implicit references to organizations or locations
        - Technical terms or concepts not captured
        - Relationships that suggest missing entities

        Respond with: YES or NO
        """

        try:
            # Use base extractor's LLM capabilities for assessment
            # This is a simplified implementation - in practice, you'd use a dedicated LLM call
            return len(existing_entities) < 5  # Simple heuristic for now
        except Exception as e:
            self.logger.warning(f"Entity assessment failed: {e}")
            return False  # Conservative approach

    # Compatibility methods to maintain interface
    async def extract(
        self,
        text: str,
        source_doc_id: Optional[str] = None,
        auto_infer_domain: bool = False
    ) -> List[Entity]:
        """Extract entities (compatibility method)."""
        return await self.extract_with_gleaning(text, source_doc_id)

    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self.confidence_threshold

    @property
    def domain(self) -> str:
        """Get extraction domain."""
        return self.base_extractor.domain

    @property
    def language(self) -> Optional[str]:
        """Get extraction language."""
        return self.base_extractor.language
