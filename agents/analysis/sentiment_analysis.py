"""Sentiment analysis agent."""

from typing import Type, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import SentimentAnalysisResult

logger = structlog.get_logger(__name__)


class SentimentAnalysisAgent(BaseAgent[SentimentAnalysisResult]):
    """Agent specialized for sentiment and emotion analysis."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="sentiment_analysis",
            description="Analyzes sentiment and emotions in text",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[SentimentAnalysisResult]:
        return SentimentAnalysisResult

    async def analyze_sentiment(
        self,
        text: str,
        include_emotions: bool = True,
        domain: Optional[str] = None,
        **kwargs
    ) -> SentimentAnalysisResult:
        """Analyze sentiment and emotions in text.

        Args:
            text: Input text to analyze
            include_emotions: Whether to include detailed emotion analysis
            domain: Optional domain context for sentiment analysis
            **kwargs: Additional arguments passed to the agent

        Returns:
            SentimentAnalysisResult containing sentiment analysis
        """
        logger.info(
            "Starting sentiment analysis",
            agent=self.__class__.__name__,
            text_length=len(text),
            include_emotions=include_emotions,
            domain=domain or "general",
            version=self.config.version
        )

        try:
            # Prepare context for sentiment analysis
            context = {
                "domain": domain or "general",
                "include_emotions": include_emotions,
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=text,
                result_type=SentimentAnalysisResult,
                **context
            )

            logger.info(
                "Sentiment analysis completed",
                agent=self.__class__.__name__,
                sentiment=result.sentiment if hasattr(result, 'sentiment') else "unknown",
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Sentiment analysis failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
