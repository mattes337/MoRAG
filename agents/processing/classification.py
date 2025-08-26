"""Classification agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import ClassificationResult

logger = structlog.get_logger(__name__)


class ClassificationAgent(BaseAgent[ClassificationResult]):
    """Agent specialized for text classification."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="classification",
            description="Classifies text into predefined categories",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    

    
    def get_result_type(self) -> Type[ClassificationResult]:
        return ClassificationResult
