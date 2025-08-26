"""Relation extraction agent."""

from typing import Type, List, Optional
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..config.defaults import get_default_config
from .models import RelationExtractionResult

logger = structlog.get_logger(__name__)


class RelationExtractionAgent(BaseAgent[RelationExtractionResult]):
    """Agent specialized for extracting relations between entities."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for relation extraction."""
        return get_default_config("relation_extraction")
    

    
    def get_result_type(self) -> Type[RelationExtractionResult]:
        """Get the result type for relation extraction."""
        return RelationExtractionResult
