"""Hybrid entity extractor combining AI and pattern matching."""

import structlog
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass

from ..models import Entity
from ..ai import EntityExtractionAgent
from .pattern_matcher import EntityPatternMatcher

logger = structlog.get_logger(__name__)


@dataclass
class ExtractionResult:
    """Result from a single extraction method."""
    entities: List[Entity]
    method: str
    confidence_boost: float = 0.0


class HybridEntityExtractor:
    """Hybrid entity extractor combining AI-based and pattern-based extraction."""
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        chunk_size: int = 4000,
        enable_pattern_matching: bool = True,
        pattern_confidence_boost: float = 0.1,
        ai_confidence_boost: float = 0.0,
        **kwargs
    ):
        """Initialize the hybrid extractor.
        
        Args:
            min_confidence: Minimum confidence threshold for final entities
            chunk_size: Maximum characters per chunk for large texts
            enable_pattern_matching: Whether to use pattern matching
            pattern_confidence_boost: Confidence boost for pattern-matched entities
            ai_confidence_boost: Confidence boost for AI-extracted entities
            **kwargs: Additional arguments passed to AI agent
        """
        self.min_confidence = min_confidence
        self.chunk_size = chunk_size
        self.enable_pattern_matching = enable_pattern_matching
        self.pattern_confidence_boost = pattern_confidence_boost
        self.ai_confidence_boost = ai_confidence_boost
        
        # Initialize extractors
        self.ai_agent = EntityExtractionAgent(min_confidence=min_confidence * 0.8, **kwargs)  # Lower threshold for AI
        if self.enable_pattern_matching:
            self.pattern_matcher = EntityPatternMatcher()
        else:
            self.pattern_matcher = None
        
        self.logger = logger.bind(component="hybrid_extractor")
    
    async def extract(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Entity]:
        """Extract entities using hybrid approach.
        
        Args:
            text: Text to extract entities from
            doc_id: Optional document ID (deprecated, use source_doc_id)
            source_doc_id: Optional source document ID
            **kwargs: Additional arguments
            
        Returns:
            List of Entity objects with enhanced accuracy
        """
        # Handle backward compatibility
        if source_doc_id is None and doc_id is not None:
            source_doc_id = doc_id
        
        if not text or not text.strip():
            return []
        
        self.logger.info(
            "Starting hybrid entity extraction",
            text_length=len(text),
            enable_pattern_matching=self.enable_pattern_matching,
            source_doc_id=source_doc_id
        )
        
        try:
            # Collect results from different extraction methods
            extraction_results = []
            
            # 1. AI-based extraction
            ai_entities = await self.ai_agent.extract_entities(
                text=text,
                chunk_size=self.chunk_size,
                source_doc_id=source_doc_id
            )
            extraction_results.append(ExtractionResult(
                entities=ai_entities,
                method="ai",
                confidence_boost=self.ai_confidence_boost
            ))
            
            # 2. Pattern-based extraction
            if self.enable_pattern_matching and self.pattern_matcher:
                pattern_entities = self.pattern_matcher.extract_entities(
                    text=text,
                    min_confidence=self.min_confidence * 0.7  # Lower threshold for patterns
                )
                extraction_results.append(ExtractionResult(
                    entities=pattern_entities,
                    method="pattern",
                    confidence_boost=self.pattern_confidence_boost
                ))
            
            # 3. Merge and deduplicate entities
            merged_entities = self._merge_extraction_results(extraction_results, source_doc_id)
            
            # 4. Filter by final confidence threshold
            final_entities = [
                entity for entity in merged_entities
                if entity.confidence >= self.min_confidence
            ]
            
            self.logger.info(
                "Hybrid extraction completed",
                ai_entities=len(ai_entities),
                pattern_entities=len(extraction_results[1].entities) if len(extraction_results) > 1 else 0,
                merged_entities=len(merged_entities),
                final_entities=len(final_entities),
                source_doc_id=source_doc_id
            )
            
            return final_entities
            
        except Exception as e:
            self.logger.error(
                "Hybrid extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text)
            )
            raise
    
    def _merge_extraction_results(
        self,
        results: List[ExtractionResult],
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Merge entities from different extraction methods."""
        if not results:
            return []
        
        # Collect all entities with method information
        all_entities = []
        for result in results:
            for entity in result.entities:
                # Apply confidence boost
                boosted_confidence = min(1.0, entity.confidence + result.confidence_boost)
                
                # Add method information to attributes
                enhanced_attributes = entity.attributes.copy()
                enhanced_attributes.update({
                    "extraction_method": result.method,
                    "original_confidence": entity.confidence,
                    "confidence_boost": result.confidence_boost,
                    "boosted_confidence": boosted_confidence
                })
                
                # Create enhanced entity
                enhanced_entity = Entity(
                    name=entity.name,
                    type=entity.type,
                    confidence=boosted_confidence,
                    source_doc_id=source_doc_id or entity.source_doc_id,
                    attributes=enhanced_attributes
                )
                all_entities.append(enhanced_entity)
        
        # Deduplicate and merge similar entities
        merged_entities = self._deduplicate_and_merge(all_entities)
        
        return merged_entities
    
    def _deduplicate_and_merge(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities and merge similar ones."""
        if not entities:
            return entities
        
        # Group entities by normalized name and type
        entity_groups = {}
        for entity in entities:
            key = (self._normalize_entity_name(entity.name), entity.type)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Merge entities in each group
        merged_entities = []
        for group in entity_groups.values():
            merged_entity = self._merge_entity_group(group)
            merged_entities.append(merged_entity)
        
        return merged_entities
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        return name.lower().strip()
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """Merge a group of similar entities."""
        if len(entities) == 1:
            return entities[0]
        
        # Sort by confidence (highest first)
        entities.sort(key=lambda e: e.confidence, reverse=True)
        
        # Use the highest confidence entity as base
        base_entity = entities[0]
        
        # Calculate merged confidence
        # If multiple methods agree, boost confidence
        methods = set(entity.attributes.get("extraction_method", "unknown") for entity in entities)
        if len(methods) > 1:
            # Multiple methods found this entity - high confidence boost
            confidence_boost = 0.2
        else:
            # Same method found it multiple times - small boost
            confidence_boost = 0.05
        
        merged_confidence = min(1.0, base_entity.confidence + confidence_boost)
        
        # Merge attributes
        merged_attributes = base_entity.attributes.copy()
        merged_attributes.update({
            "merged_from_methods": list(methods),
            "merge_count": len(entities),
            "merge_confidence_boost": confidence_boost,
            "original_entities": [
                {
                    "name": e.name,
                    "confidence": e.confidence,
                    "method": e.attributes.get("extraction_method", "unknown")
                }
                for e in entities
            ]
        })
        
        # Create merged entity
        merged_entity = Entity(
            name=base_entity.name,  # Use the name from highest confidence entity
            type=base_entity.type,
            confidence=merged_confidence,
            source_doc_id=base_entity.source_doc_id,
            attributes=merged_attributes
        )
        
        return merged_entity
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the extraction methods."""
        stats = {
            "ai_agent_available": self.ai_agent is not None,
            "pattern_matcher_available": self.pattern_matcher is not None,
            "min_confidence": self.min_confidence,
            "chunk_size": self.chunk_size,
            "pattern_confidence_boost": self.pattern_confidence_boost,
            "ai_confidence_boost": self.ai_confidence_boost
        }
        
        if self.pattern_matcher:
            stats["pattern_count"] = len(self.pattern_matcher.patterns)
        
        return stats
