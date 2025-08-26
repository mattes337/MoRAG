"""Chunking agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import ChunkingResult

logger = structlog.get_logger(__name__)


class ChunkingAgent(BaseAgent[ChunkingResult]):
    """Agent specialized for intelligent text chunking."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="chunking",
            description="Performs intelligent semantic chunking of text",
            prompt=PromptConfig(output_format="json", strict_json=True),
            agent_config={"max_chunk_size": 4000, "min_chunk_size": 500}
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a chunking expert. Divide text into semantically coherent chunks."""
        user_prompt = """Chunk this text: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[ChunkingResult]:
        return ChunkingResult
