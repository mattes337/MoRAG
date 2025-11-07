"""
MoRAG Agents - Centralized AI Agent Framework

This package provides a comprehensive agentic pattern for all LLM interactions in MoRAG.
Each agent is specialized for specific tasks and exposes configurable properties that
reflect in the actual prompts, abstracting prompt management from the rest of the codebase.

Key Features:
- Specialized agents for different LLM use cases
- Configurable prompt templates
- Standardized interfaces and error handling
- Agent factory and registry for easy management
- Dynamic prompt generation based on configuration
"""

from .analysis import (
    ContentAnalysisAgent,
    QueryAnalysisAgent,
    SentimentAnalysisAgent,
    TopicAnalysisAgent,
)
from .base import (
    AgentConfig,
    AgentError,
    BaseAgent,
    ConfigurationError,
    LLMResponseParseError,
    LLMResponseParser,
    PromptGenerationError,
    PromptTemplate,
)
from .config import AgentConfigManager, ModelConfig, PromptConfig, RetryConfig

# Import specialized agents
from .extraction import (
    EntityExtractionAgent,
    FactExtractionAgent,
    KeywordExtractionAgent,
    RelationExtractionAgent,
)
from .factory import (
    AgentFactory,
    AgentRegistry,
    create_agent,
    get_agent,
    register_agent,
)
from .generation import (
    ExplanationAgent,
    ResponseGenerationAgent,
    SummarizationAgent,
    SynthesisAgent,
)
from .processing import (
    ChunkingAgent,
    ClassificationAgent,
    FilteringAgent,
    SemanticChunkingAgent,
    ValidationAgent,
)
from .reasoning import (
    ContextAnalysisAgent,
    DecisionMakingAgent,
    PathSelectionAgent,
    ReasoningAgent,
)

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig",
    "PromptTemplate",
    "LLMResponseParser",
    "LLMResponseParseError",
    "AgentError",
    "ConfigurationError",
    "PromptGenerationError",
    # Factory and registry
    "AgentFactory",
    "AgentRegistry",
    "create_agent",
    "get_agent",
    "register_agent",
    # Configuration
    "AgentConfigManager",
    "PromptConfig",
    "ModelConfig",
    "RetryConfig",
    # Extraction agents
    "FactExtractionAgent",
    "EntityExtractionAgent",
    "RelationExtractionAgent",
    "KeywordExtractionAgent",
    # Analysis agents
    "QueryAnalysisAgent",
    "ContentAnalysisAgent",
    "SentimentAnalysisAgent",
    "TopicAnalysisAgent",
    # Reasoning agents
    "PathSelectionAgent",
    "ReasoningAgent",
    "DecisionMakingAgent",
    "ContextAnalysisAgent",
    # Generation agents
    "SummarizationAgent",
    "ResponseGenerationAgent",
    "ExplanationAgent",
    "SynthesisAgent",
    # Processing agents
    "ChunkingAgent",
    "ClassificationAgent",
    "ValidationAgent",
    "FilteringAgent",
    "SemanticChunkingAgent",
]
