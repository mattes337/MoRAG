"""Entity extraction agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..config.entity_types import get_agent_entity_types
from .models import EntityExtractionResult, ExtractedEntity, ConfidenceLevel

logger = structlog.get_logger(__name__)


class EntityExtractionAgent(BaseAgent[EntityExtractionResult]):
    """Agent specialized for extracting named entities from text."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for entity extraction."""
        return AgentConfig(
            name="entity_extraction",
            description="Extracts named entities from text content",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                include_context=True,
                output_format="json",
                strict_json=True,
                include_confidence=True,
                min_confidence=0.4,
            ),
            agent_config={
                "entity_types": get_agent_entity_types("entity_extraction"),
                "include_offsets": True,
                "normalize_entities": True,
                "min_entity_length": 2,
            }
        )
    

    
    def get_result_type(self) -> Type[EntityExtractionResult]:
        """Get the result type for entity extraction."""
        return EntityExtractionResult
    
    async def extract_entities(
        self,
        text: str,
        domain: str = "general",
        entity_types: Optional[List[str]] = None
    ) -> EntityExtractionResult:
        """Extract entities from text.
        
        Args:
            text: Text to extract entities from
            domain: Domain context for extraction
            entity_types: Optional list of entity types to focus on
            
        Returns:
            Entity extraction result
        """
        if not text or not text.strip():
            return EntityExtractionResult(
                entities=[],
                total_entities=0,
                confidence=ConfidenceLevel.HIGH,
                metadata={"error": "Empty text"}
            )
        
        # Update config for this extraction
        if entity_types:
            self.config.set_agent_config("entity_types", entity_types)
        
        self.logger.info(
            "Starting entity extraction",
            text_length=len(text),
            domain=domain,
            entity_types=entity_types
        )
        
        try:
            result = await self.execute(
                text,
                domain=domain
            )
            
            self.logger.info(
                "Entity extraction completed",
                entities_extracted=result.total_entities,
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e))
            raise
    
    async def extract_entities_batch(
        self,
        texts: List[str],
        domain: str = "general",
        entity_types: Optional[List[str]] = None
    ) -> List[EntityExtractionResult]:
        """Extract entities from multiple texts.
        
        Args:
            texts: List of texts to process
            domain: Domain context
            entity_types: Optional entity types to focus on
            
        Returns:
            List of entity extraction results
        """
        if not texts:
            return []
        
        self.logger.info(f"Starting batch entity extraction for {len(texts)} texts")
        
        # Update config if needed
        if entity_types:
            self.config.set_agent_config("entity_types", entity_types)
        
        results = await self.batch_execute(
            texts,
            domain=domain
        )
        
        self.logger.info(f"Batch entity extraction completed for {len(results)} texts")
        return results
