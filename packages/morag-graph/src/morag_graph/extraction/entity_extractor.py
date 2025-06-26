"""PydanticAI-based entity extraction."""

import structlog
from typing import List, Optional, Dict, Any

from ..models import Entity
from ..ai import EntityExtractionAgent

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """PydanticAI-based entity extractor - completely new implementation."""
    
    def __init__(self, min_confidence: float = 0.6, chunk_size: int = 4000, **kwargs):
        """Initialize the entity extractor."""
        self.min_confidence = min_confidence
        self.chunk_size = chunk_size
        self.agent = EntityExtractionAgent(min_confidence=min_confidence, **kwargs)
        self.logger = logger.bind(component="entity_extractor")
    
    async def extract(self, text: str, doc_id: Optional[str] = None, source_doc_id: Optional[str] = None, **kwargs) -> List[Entity]:
        """Extract entities from text using PydanticAI agent."""
        if source_doc_id is None and doc_id is not None:
            source_doc_id = doc_id
        
        try:
            entities = await self.agent.extract_entities(
                text=text,
                chunk_size=self.chunk_size,
                source_doc_id=source_doc_id
            )
            
            self.logger.info(
                "Entity extraction completed",
                text_length=len(text),
                entities_found=len(entities),
                source_doc_id=source_doc_id
            )
            
            return entities
            
        except Exception as e:
            self.logger.error(
                "Entity extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text)
            )
            raise
    
    async def extract_entities(self, text: str, source_doc_id: Optional[str] = None, **kwargs) -> List[Entity]:
        """Alias for extract method for backward compatibility."""
        return await self.extract(text, source_doc_id=source_doc_id, **kwargs)

    async def extract_with_context(
        self,
        text: str,
        source_doc_id: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities with additional context information.

        Args:
            text: Text to extract entities from
            source_doc_id: ID of the source document
            additional_context: Additional context to help with extraction

        Returns:
            List of Entity objects with context information
        """
        # For now, just call the regular extract method
        # The PydanticAI agent handles context internally
        return await self.extract(text, source_doc_id=source_doc_id, **kwargs)
