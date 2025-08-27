"""Relation extraction agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..config.defaults import get_default_config
from .models import RelationExtractionResult

logger = structlog.get_logger(__name__)


class RelationExtractionAgent(BaseAgent[RelationExtractionResult]):
    """Agent specialized for extracting relations between entities."""

    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for relation extraction."""
        return get_default_config("relation_extraction")

    def get_result_type(self) -> Type[RelationExtractionResult]:
        """Get the result type for relation extraction."""
        return RelationExtractionResult

    async def extract_relations(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> RelationExtractionResult:
        """Extract relations between entities from text.

        Args:
            text: Input text to analyze
            entities: Optional list of known entities to focus on
            domain: Optional domain context for relation extraction
            **kwargs: Additional arguments passed to the agent

        Returns:
            RelationExtractionResult containing extracted relations
        """
        logger.info(
            "Starting relation extraction",
            agent=self.__class__.__name__,
            text_length=len(text),
            entities_provided=len(entities) if entities else 0,
            domain=domain or "general",
            version=self.config.version
        )

        try:
            # Prepare context for relation extraction
            context = {
                "domain": domain or "general",
                "entities": entities or [],
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=text,
                **context
            )

            logger.info(
                "Relation extraction completed",
                agent=self.__class__.__name__,
                relations_extracted=len(result.relations),
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Relation extraction failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
