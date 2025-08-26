"""Content analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ContentAnalysisResult

logger = structlog.get_logger(__name__)


class ContentAnalysisAgent(BaseAgent[ContentAnalysisResult]):
    """Agent specialized for analyzing content structure and topics."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="content_analysis",
            description="Analyzes content structure, topics, and complexity",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    

    
    def get_result_type(self) -> Type[ContentAnalysisResult]:
        return ContentAnalysisResult
