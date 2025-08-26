"""Topic analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import TopicAnalysisResult

logger = structlog.get_logger(__name__)


class TopicAnalysisAgent(BaseAgent[TopicAnalysisResult]):
    """Agent specialized for topic modeling and analysis."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="topic_analysis",
            description="Analyzes and models topics in text",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    

    
    def get_result_type(self) -> Type[TopicAnalysisResult]:
        return TopicAnalysisResult
