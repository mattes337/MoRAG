"""Topic analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import TopicAnalysisResult

logger = structlog.get_logger(__name__)


class TopicAnalysisAgent(BaseAgent[TopicAnalysisResult]):
    """Agent specialized for topic modeling and analysis."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="topic_analysis",
            description="Analyzes and models topics in text",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a topic analysis expert. Identify and analyze topics in text."""
        user_prompt = """Analyze topics in: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[TopicAnalysisResult]:
        return TopicAnalysisResult
