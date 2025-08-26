"""Keyword extraction agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import KeywordExtractionResult

logger = structlog.get_logger(__name__)


class KeywordExtractionAgent(BaseAgent[KeywordExtractionResult]):
    """Agent specialized for extracting keywords from text."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for keyword extraction."""
        return AgentConfig(
            name="keyword_extraction",
            description="Extracts relevant keywords from text",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
            ),
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        """Create the prompt template for keyword extraction."""
        system_prompt = """You are an expert keyword extraction agent. Extract relevant keywords and phrases from text."""
        user_prompt = """Extract keywords from: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[KeywordExtractionResult]:
        """Get the result type for keyword extraction."""
        return KeywordExtractionResult
