"""Context analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ContextAnalysisResult

logger = structlog.get_logger(__name__)


class ContextAnalysisAgent(BaseAgent[ContextAnalysisResult]):
    """Agent specialized for analyzing context and relevance."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="context_analysis",
            description="Analyzes context and determines relevance",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    

    
    def get_result_type(self) -> Type[ContextAnalysisResult]:
        return ContextAnalysisResult
