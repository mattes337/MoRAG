"""Response generation agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
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
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a response generation expert. Generate comprehensive, accurate responses."""
        user_prompt = """Generate a response for: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[ResponseGenerationResult]:
        return ResponseGenerationResult
