"""Filtering agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import FilteringResult

logger = structlog.get_logger(__name__)


class FilteringAgent(BaseAgent[FilteringResult]):
    """Agent specialized for content filtering."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="filtering",
            description="Filters content based on quality and relevance criteria",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a filtering expert. Filter content based on quality and relevance."""
        user_prompt = """Filter this content: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[FilteringResult]:
        return FilteringResult
