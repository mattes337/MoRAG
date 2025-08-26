"""Reasoning agent."""

from typing import Type
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
