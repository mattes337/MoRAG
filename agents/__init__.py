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

# Core base imports (always available)
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

# Optional imports with fallbacks
try:
    from .config import AgentConfigManager, ModelConfig, PromptConfig, RetryConfig
except ImportError:
    AgentConfigManager = None
    ModelConfig = None
    PromptConfig = None
    RetryConfig = None

try:
    from .analysis import (
        ContentAnalysisAgent,
        QueryAnalysisAgent,
        SentimentAnalysisAgent,
        TopicAnalysisAgent,
    )
except ImportError:
    ContentAnalysisAgent = None
    QueryAnalysisAgent = None
    SentimentAnalysisAgent = None
    TopicAnalysisAgent = None

try:
    from .extraction import (
        EntityExtractionAgent,
        FactExtractionAgent,
        KeywordExtractionAgent,
        RelationExtractionAgent,
    )
except ImportError:
    EntityExtractionAgent = None
    FactExtractionAgent = None
    KeywordExtractionAgent = None
    RelationExtractionAgent = None

try:
    from .factory import (
        AgentFactory,
        AgentRegistry,
        create_agent,
        get_agent,
        register_agent,
    )
except ImportError:
    AgentFactory = None
    AgentRegistry = None
    create_agent = None
    get_agent = None
    register_agent = None

try:
    from .generation import (
        ExplanationAgent,
        ResponseGenerationAgent,
        SummarizationAgent,
        SynthesisAgent,
    )
except ImportError:
    ExplanationAgent = None
    ResponseGenerationAgent = None
    SummarizationAgent = None
    SynthesisAgent = None

try:
    from .processing import (
        ChunkingAgent,
        ClassificationAgent,
        FilteringAgent,
        SemanticChunkingAgent,
        ValidationAgent,
    )
except ImportError:
    ChunkingAgent = None
    ClassificationAgent = None
    FilteringAgent = None
    SemanticChunkingAgent = None
    ValidationAgent = None

try:
    from .reasoning import (
        ContextAnalysisAgent,
        DecisionMakingAgent,
        PathSelectionAgent,
        ReasoningAgent,
    )
except ImportError:
    ContextAnalysisAgent = None
    DecisionMakingAgent = None
    PathSelectionAgent = None
    ReasoningAgent = None

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
