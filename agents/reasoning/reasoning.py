"""Reasoning agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
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
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a reasoning expert. Perform logical reasoning and inference."""
        user_prompt = """Reason about: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[ReasoningResult]:
        return ReasoningResult
