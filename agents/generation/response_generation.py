"""Response generation agent."""

from typing import Type, Optional, List
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ResponseGenerationResult

logger = structlog.get_logger(__name__)


class ResponseGenerationAgent(BaseAgent[ResponseGenerationResult]):
    """Agent specialized for generating responses to queries."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="response_generation",
            description="Generates comprehensive responses to user queries",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[ResponseGenerationResult]:
        return ResponseGenerationResult

    async def generate_response(
        self,
        query: str,
        context: Optional[List[str]] = None,
        response_type: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> ResponseGenerationResult:
        """Generate a comprehensive response to a query.

        Args:
            query: Input query to respond to
            context: Optional context information
            response_type: Type of response (informative, explanatory, instructional)
            domain: Optional domain context for response generation
            **kwargs: Additional arguments passed to the agent

        Returns:
            ResponseGenerationResult containing the generated response
        """
        logger.info(
            "Starting response generation",
            agent=self.__class__.__name__,
            query_length=len(query),
            context_items=len(context) if context else 0,
            response_type=response_type,
            domain=domain or "general",
            version=self.config.version
        )

        try:
            # Prepare context for response generation
            context_data = {
                "domain": domain or "general",
                "context": context or [],
                "response_type": response_type or "informative",
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=query,
                result_type=ResponseGenerationResult,
                **context_data
            )

            logger.info(
                "Response generation completed",
                agent=self.__class__.__name__,
                response_length=len(result.response) if hasattr(result, 'response') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Response generation failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
