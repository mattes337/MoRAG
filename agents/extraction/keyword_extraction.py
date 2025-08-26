"""Keyword extraction agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..config.defaults import get_default_config
from .models import KeywordExtractionResult

logger = structlog.get_logger(__name__)


class KeywordExtractionAgent(BaseAgent[KeywordExtractionResult]):
    """Agent specialized for extracting keywords from text."""
    
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for keyword extraction."""
        return get_default_config("keyword_extraction")
    

    
    def get_result_type(self) -> Type[KeywordExtractionResult]:
        """Get the result type for keyword extraction."""
        return KeywordExtractionResult
