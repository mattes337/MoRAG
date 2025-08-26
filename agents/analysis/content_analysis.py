"""Content analysis agent."""

from typing import Type, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ContentAnalysisResult

logger = structlog.get_logger(__name__)


class ContentAnalysisAgent(BaseAgent[ContentAnalysisResult]):
    """Agent specialized for analyzing content structure and topics."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="content_analysis",
            description="Analyzes content structure, topics, and complexity",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[ContentAnalysisResult]:
        return ContentAnalysisResult

    async def analyze_content(
        self,
        content: str,
        analysis_type: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> ContentAnalysisResult:
        """Analyze content structure, topics, and complexity.

        Args:
            content: Input content to analyze
            analysis_type: Type of analysis to perform (structure, topics, complexity)
            domain: Optional domain context for analysis
            **kwargs: Additional arguments passed to the agent

        Returns:
            ContentAnalysisResult containing analysis results
        """
        logger.info(
            "Starting content analysis",
            agent=self.__class__.__name__,
            content_length=len(content),
            analysis_type=analysis_type,
            domain=domain or "general",
            version=self.config.version
        )

        try:
            # Prepare context for content analysis
            context = {
                "domain": domain or "general",
                "analysis_type": analysis_type,
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=content,
                result_type=ContentAnalysisResult,
                **context
            )

            logger.info(
                "Content analysis completed",
                agent=self.__class__.__name__,
                topics_found=len(result.topics) if hasattr(result, 'topics') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Content analysis failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
