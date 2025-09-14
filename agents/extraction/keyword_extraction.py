"""Keyword extraction agent."""

from typing import Type, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..config.defaults import get_default_config
from .models import KeywordExtractionResult

logger = structlog.get_logger(__name__)


class KeywordExtractionAgent(BaseAgent[KeywordExtractionResult]):
    """Agent specialized for extracting keywords from text."""

    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for keyword extraction."""
        return get_default_config("keyword_extraction")

    def get_result_type(self) -> Type[KeywordExtractionResult]:
        """Get the result type for keyword extraction."""
        return KeywordExtractionResult

    async def extract_keywords(
        self,
        text: str,
        max_keywords: Optional[int] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> KeywordExtractionResult:
        """Extract keywords from text.

        Args:
            text: Input text to analyze
            max_keywords: Maximum number of keywords to extract
            domain: Optional domain context for keyword extraction
            **kwargs: Additional arguments passed to the agent

        Returns:
            KeywordExtractionResult containing extracted keywords
        """
        logger.info(
            "Starting keyword extraction",
            agent=self.__class__.__name__,
            text_length=len(text),
            max_keywords=max_keywords,
            domain=domain or "general",
            version=self.config.version
        )

        try:
            # Prepare context for keyword extraction
            context = {
                "domain": domain or "general",
                "max_keywords": max_keywords,
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=text,
                **context
            )

            logger.info(
                "Keyword extraction completed",
                agent=self.__class__.__name__,
                keywords_extracted=len(result.keywords),
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Keyword extraction failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
