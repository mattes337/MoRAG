"""Explanation agent."""

from typing import Type, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ExplanationResult

logger = structlog.get_logger(__name__)


class ExplanationAgent(BaseAgent[ExplanationResult]):
    """Agent specialized for generating explanations."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="explanation",
            description="Generates clear explanations for complex topics",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[ExplanationResult]:
        return ExplanationResult

    async def explain(
        self,
        topic: str,
        explanation_type: Optional[str] = None,
        audience: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> ExplanationResult:
        """Generate an explanation for a topic.

        Args:
            topic: Topic to explain
            explanation_type: Type of explanation (causal, mechanistic, functional, etc.)
            audience: Target audience for the explanation
            context: Additional context for the explanation
            **kwargs: Additional arguments passed to the agent

        Returns:
            ExplanationResult containing the generated explanation
        """
        logger.info(
            "Starting explanation generation",
            agent=self.__class__.__name__,
            topic_length=len(topic),
            explanation_type=explanation_type,
            audience=audience or "general",
            version=self.config.version
        )

        try:
            # Prepare context for explanation generation
            context_data = {
                "explanation_type": explanation_type or "general",
                "audience": audience or "general",
                "context": context,
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=topic,
                **context_data
            )

            logger.info(
                "Explanation generation completed",
                agent=self.__class__.__name__,
                explanation_length=len(result.explanation) if hasattr(result, 'explanation') else 0,
                reasoning_steps=len(result.reasoning_steps) if hasattr(result, 'reasoning_steps') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Explanation generation failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
