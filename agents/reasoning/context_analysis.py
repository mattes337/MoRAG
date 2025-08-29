"""Context analysis agent."""

from typing import Type, Optional, List, Dict, Any
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ContextAnalysisResult

logger = structlog.get_logger(__name__)


class ContextAnalysisAgent(BaseAgent[ContextAnalysisResult]):
    """Agent specialized for analyzing context and relevance."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="context_analysis",
            description="Analyzes context and determines relevance",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[ContextAnalysisResult]:
        return ContextAnalysisResult

    async def analyze_context(
        self,
        query: str,
        context_info: Dict[str, Any],
        additional_context: Optional[List[str]] = None,
        **kwargs
    ) -> ContextAnalysisResult:
        """Analyze context and determine relevance for reasoning.

        Args:
            query: The query or problem to analyze
            context_info: Context information and metadata
            additional_context: Additional context sources
            **kwargs: Additional arguments passed to the agent

        Returns:
            ContextAnalysisResult containing context analysis
        """
        logger.info(
            "Starting context analysis",
            agent=self.__class__.__name__,
            query_length=len(query),
            context_keys=list(context_info.keys()) if context_info else [],
            additional_context_count=len(additional_context) if additional_context else 0,
            version=self.config.version
        )

        try:
            # Prepare context for analysis
            context_data = {
                "context_info": context_info,
                "additional_context": additional_context or [],
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=query,
                **context_data
            )

            logger.info(
                "Context analysis completed",
                agent=self.__class__.__name__,
                relevance_score=result.relevance_score,
                context_elements_count=len(result.relevant_context),
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Context analysis failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
