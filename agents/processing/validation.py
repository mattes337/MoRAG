"""Validation agent."""

from typing import Type
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
