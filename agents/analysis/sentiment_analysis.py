"""Sentiment analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from .models import SentimentAnalysisResult

logger = structlog.get_logger(__name__)


class SentimentAnalysisAgent(BaseAgent[SentimentAnalysisResult]):
    """Agent specialized for sentiment and emotion analysis."""
    
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="sentiment_analysis",
            description="Analyzes sentiment and emotions in text",
            prompt=PromptConfig(output_format="json", strict_json=True),
        )
    

    
    def get_result_type(self) -> Type[SentimentAnalysisResult]:
        return SentimentAnalysisResult
