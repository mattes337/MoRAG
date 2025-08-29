"""Decision making agent."""

from typing import Type, Optional, List, Dict, Any
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import DecisionResult

logger = structlog.get_logger(__name__)


class DecisionMakingAgent(BaseAgent[DecisionResult]):
    """Agent specialized for decision making and option evaluation."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="decision_making",
            description="Makes decisions and evaluates options",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[DecisionResult]:
        return DecisionResult

    async def make_decision(
        self,
        decision_context: str,
        options: Optional[List[Dict[str, Any]]] = None,
        criteria: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        **kwargs
    ) -> DecisionResult:
        """Make a decision based on context and options.

        Args:
            decision_context: Context and background for the decision
            options: Available options to choose from
            criteria: Decision criteria to consider
            constraints: Constraints that limit the decision
            **kwargs: Additional arguments passed to the agent

        Returns:
            DecisionResult containing the decision and rationale
        """
        logger.info(
            "Starting decision making process",
            agent=self.__class__.__name__,
            context_length=len(decision_context),
            options_count=len(options) if options else 0,
            criteria_count=len(criteria) if criteria else 0,
            constraints_count=len(constraints) if constraints else 0,
            version=self.config.version
        )

        try:
            # Prepare context for decision making
            context_data = {
                "options": options or [],
                "criteria": criteria or [],
                "constraints": constraints or [],
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=decision_context,
                **context_data
            )

            logger.info(
                "Decision making completed",
                agent=self.__class__.__name__,
                decision_length=len(result.decision) if hasattr(result, 'decision') else 0,
                alternatives_considered=len(result.alternatives) if hasattr(result, 'alternatives') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Decision making failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
