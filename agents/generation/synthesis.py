"""Synthesis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import SynthesisResult

logger = structlog.get_logger(__name__)


class SynthesisAgent(BaseAgent[SynthesisResult]):
    """Agent specialized for synthesizing information from multiple sources."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="synthesis",
            description="Synthesizes information from multiple sources",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    

    
    def get_result_type(self) -> Type[SynthesisResult]:
        return SynthesisResult
