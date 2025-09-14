"""Topic analysis agent."""

from typing import Type, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import TopicAnalysisResult

logger = structlog.get_logger(__name__)


class TopicAnalysisAgent(BaseAgent[TopicAnalysisResult]):
    """Agent specialized for topic modeling and analysis."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="topic_analysis",
            description="Analyzes and models topics in text",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[TopicAnalysisResult]:
        return TopicAnalysisResult

    async def analyze_topics(
        self,
        text: str,
        num_topics: Optional[int] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> TopicAnalysisResult:
        """Analyze and model topics in text.

        Args:
            text: Input text to analyze
            num_topics: Number of topics to identify
            domain: Optional domain context for topic analysis
            **kwargs: Additional arguments passed to the agent

        Returns:
            TopicAnalysisResult containing topic analysis
        """
        logger.info(
            "Starting topic analysis",
            agent=self.__class__.__name__,
            text_length=len(text),
            num_topics=num_topics,
            domain=domain or "general",
            version=self.config.version
        )

        try:
            # Prepare context for topic analysis
            context = {
                "domain": domain or "general",
                "num_topics": num_topics,
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=text,
                **context
            )

            logger.info(
                "Topic analysis completed",
                agent=self.__class__.__name__,
                topics_found=len(result.topics) if hasattr(result, 'topics') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Topic analysis failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
