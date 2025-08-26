"""Synthesis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
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
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a synthesis expert. Combine information from multiple sources coherently."""
        user_prompt = """Synthesize information about: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[SynthesisResult]:
        return SynthesisResult
