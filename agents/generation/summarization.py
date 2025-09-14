"""Summarization agent."""

from typing import Type, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import SummarizationResult

logger = structlog.get_logger(__name__)


class SummarizationAgent(BaseAgent[SummarizationResult]):
    """Agent specialized for text summarization."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="summarization",
            description="Generates high-quality summaries of text content",
            prompt=PromptConfig(output_format="json", strict_json=True),
            agent_config={"max_summary_length": 1000, "summary_type": "abstractive"}
        )

    def get_result_type(self) -> Type[SummarizationResult]:
        return SummarizationResult

    async def summarize(
        self,
        text: str,
        summary_type: Optional[str] = None,
        max_length: Optional[int] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> SummarizationResult:
        """Generate a summary of the input text.

        Args:
            text: Input text to summarize
            summary_type: Type of summary (abstractive, extractive, bullet_points)
            max_length: Maximum length of the summary
            domain: Optional domain context for summarization
            **kwargs: Additional arguments passed to the agent

        Returns:
            SummarizationResult containing the summary
        """
        logger.info(
            "Starting summarization",
            agent=self.__class__.__name__,
            text_length=len(text),
            summary_type=summary_type,
            max_length=max_length,
            domain=domain or "general",
            version=self.config.version
        )

        try:
            # Prepare context for summarization
            context = {
                "domain": domain or "general",
                "summary_type": summary_type or "abstractive",
                "max_length": max_length,
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=text,
                **context
            )

            logger.info(
                "Summarization completed",
                agent=self.__class__.__name__,
                summary_length=len(result.summary) if hasattr(result, 'summary') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Summarization failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
