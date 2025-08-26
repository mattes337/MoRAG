"""Sentiment analysis agent."""

from typing import Type
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig, PromptConfig
from ..base.template import ConfigurablePromptTemplate
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
    
    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = """You are a sentiment analysis expert. Analyze text for sentiment, emotions, and polarity."""
        user_prompt = """Analyze sentiment in: {{ input }}"""
        return ConfigurablePromptTemplate(self.config.prompt, system_prompt, user_prompt)
    
    def get_result_type(self) -> Type[SentimentAnalysisResult]:
        return SentimentAnalysisResult
