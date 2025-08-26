"""Summarization agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import SummarizationResult

logger = structlog.get_logger(__name__)


class SummarizationAgent(BaseAgent[SummarizationResult]):
    """Agent specialized for text summarization."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="summarization",
            description="Generates high-quality summaries of text content",
            prompt=PromptConfig(output_format="json", strict_json=True),
            agent_config={"max_summary_length": 1000, "summary_type": "abstractive"}
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a summarization expert. Create concise, informative summaries."""
        user_prompt = """Summarize this text: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[SummarizationResult]:
        return SummarizationResult
