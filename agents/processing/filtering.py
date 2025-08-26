"""Filtering agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
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
    

    
    def get_result_type(self) -> Type[FilteringResult]:
        return FilteringResult
