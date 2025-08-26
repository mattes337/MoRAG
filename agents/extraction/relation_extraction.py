"""Relation extraction agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import RelationExtractionResult

logger = structlog.get_logger(__name__)


class RelationExtractionAgent(BaseAgent[RelationExtractionResult]):
    """Agent specialized for extracting relations between entities."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for relation extraction."""
        return AgentConfig(
            name="relation_extraction",
            description="Extracts semantic relations between entities",
            prompt=PromptConfig(
                domain="general",
                include_examples=True,
                output_format="json",
                strict_json=True,
                include_confidence=True,
            ),
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        """Create the prompt template for relation extraction."""
        system_prompt = """You are an expert relation extraction agent. Extract meaningful relationships between entities in text."""
        user_prompt = """Extract relations from: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[RelationExtractionResult]:
        """Get the result type for relation extraction."""
        return RelationExtractionResult
