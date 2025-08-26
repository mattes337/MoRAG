"""Validation agent."""

from typing import Type, Optional, List, Dict, Any
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ValidationResult

logger = structlog.get_logger(__name__)


class ValidationAgent(BaseAgent[ValidationResult]):
    """Agent specialized for content validation."""

    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="validation",
            description="Validates content quality and accuracy",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )

    def get_result_type(self) -> Type[ValidationResult]:
        return ValidationResult

    async def validate_fact(
        self,
        fact: str,
        context: Optional[str] = None,
        sources: Optional[List[str]] = None,
        **kwargs
    ) -> ValidationResult:
        """Validate a fact for accuracy and reliability.

        Args:
            fact: The fact to validate
            context: Additional context for validation
            sources: Source materials for verification
            **kwargs: Additional arguments passed to the agent

        Returns:
            ValidationResult containing validation assessment
        """
        logger.info(
            "Starting fact validation",
            agent=self.__class__.__name__,
            fact_length=len(fact),
            has_context=context is not None,
            sources_count=len(sources) if sources else 0,
            version=self.config.version
        )

        try:
            # Prepare context for validation
            context_data = {
                "context": context or "",
                "sources": sources or [],
                **kwargs
            }

            # Execute the agent
            result = await self.execute(
                user_input=fact,
                result_type=ValidationResult,
                **context_data
            )

            logger.info(
                "Fact validation completed",
                agent=self.__class__.__name__,
                is_valid=result.is_valid,
                quality_score=result.quality_score,
                errors_count=len(result.validation_errors),
                confidence=result.confidence,
                version=self.config.version
            )

            return result

        except Exception as e:
            logger.error(
                "Fact validation failed",
                agent=self.__class__.__name__,
                error=str(e),
                version=self.config.version
            )
            raise
