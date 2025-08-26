"""Reasoning agent."""

from typing import Type, Optional, List
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ReasoningResult

logger = structlog.get_logger(__name__)


class ReasoningAgent(BaseAgent[ReasoningResult]):
    """Agent specialized for multi-step reasoning."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="reasoning",
            description="Performs multi-step reasoning and inference",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[ReasoningResult]:
        return ReasoningResult

    async def reason(
        self,
        problem: str,
        reasoning_type: Optional[str] = None,
        evidence: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        **kwargs
    ) -> ReasoningResult:
        """Perform multi-step reasoning on a problem.

        Args:
            problem: Problem or question to reason about
            reasoning_type: Type of reasoning (deductive, inductive, abductive, etc.)
            evidence: Available evidence to consider
            constraints: Constraints or limitations to consider
            **kwargs: Additional arguments passed to the agent

        Returns:
            ReasoningResult containing the reasoning process and conclusion
        """
        logger.info(
            "Starting reasoning process",
            agent=self.__class__.__name__,
            problem_length=len(problem),
            reasoning_type=reasoning_type,
            evidence_count=len(evidence) if evidence else 0,
            constraints_count=len(constraints) if constraints else 0,
            version=self.config.version
        )

        try:
            # Prepare context for reasoning
            context_data = {
                "reasoning_type": reasoning_type or "logical",
                "evidence": evidence or [],
                "constraints": constraints or [],
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=problem,
                result_type=ReasoningResult,
                **context_data
            )

            logger.info(
                "Reasoning process completed",
                agent=self.__class__.__name__,
                conclusion_length=len(result.conclusion) if hasattr(result, 'conclusion') else 0,
                reasoning_steps=len(result.reasoning_steps) if hasattr(result, 'reasoning_steps') else 0,
                evidence_used=len(result.evidence) if hasattr(result, 'evidence') else 0,
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Reasoning process failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
