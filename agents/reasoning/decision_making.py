"""Decision making agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import DecisionResult

logger = structlog.get_logger(__name__)


class DecisionMakingAgent(BaseAgent[DecisionResult]):
    """Agent specialized for decision making and option evaluation."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="decision_making",
            description="Makes decisions and evaluates options",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    

    
    def get_result_type(self) -> Type[DecisionResult]:
        return DecisionResult
