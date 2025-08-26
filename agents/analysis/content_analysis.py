"""Content analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import ContentAnalysisResult

logger = structlog.get_logger(__name__)


class ContentAnalysisAgent(BaseAgent[ContentAnalysisResult]):
    """Agent specialized for analyzing content structure and topics."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="content_analysis",
            description="Analyzes content structure, topics, and complexity",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a content analysis expert. Analyze text for topics, structure, and complexity."""
        user_prompt = """Analyze this content: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[ContentAnalysisResult]:
        return ContentAnalysisResult
