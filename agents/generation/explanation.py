"""Explanation agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
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
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are an explanation expert. Generate clear, step-by-step explanations."""
        user_prompt = """Explain: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[ExplanationResult]:
        return ExplanationResult
