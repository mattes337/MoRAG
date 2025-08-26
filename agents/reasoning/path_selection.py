"""Path selection agent for reasoning."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
from .models import PathSelectionResult

logger = structlog.get_logger(__name__)


class PathSelectionAgent(BaseAgent[PathSelectionResult]):
    """Agent specialized for selecting optimal reasoning paths."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="path_selection",
            description="Selects optimal paths for multi-hop reasoning",
            prompt=PromptConfig(output_format="json", strict_json=True),
            agent_config={"max_paths": 10, "strategy": "bidirectional"}
        )
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a path selection expert. Select the most promising reasoning paths for a given query."""
        user_prompt = """Select reasoning paths for: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[PathSelectionResult]:
        return PathSelectionResult
