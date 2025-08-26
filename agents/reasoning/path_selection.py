"""Path selection agent for reasoning."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
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
    

    
    def get_result_type(self) -> Type[PathSelectionResult]:
        return PathSelectionResult
