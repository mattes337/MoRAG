"""Context analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import ContextAnalysisResult

logger = structlog.get_logger(__name__)


class ContextAnalysisAgent(BaseAgent[ContextAnalysisResult]):
    """Agent specialized for analyzing context and relevance."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="context_analysis",
            description="Analyzes context and determines relevance",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a context analysis expert. Analyze context and determine relevance."""
        user_prompt = """Analyze context for: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[ContextAnalysisResult]:
        return ContextAnalysisResult
