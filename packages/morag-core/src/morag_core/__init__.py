"""MoRAG Core - Essential components for the MoRAG system."""

from .ai import (
    MoRAGBaseAgent,
    AgentConfig,
    GeminiProvider,
    ProviderConfig,
    AgentFactory,
    create_agent,
    create_agent_with_config,
    EntityExtractionResult,
    RelationExtractionResult,
    SummaryResult,
    SemanticChunkingResult,
    ContentAnalysisResult,
    TranscriptAnalysisResult,
)

from .chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    SemanticChunker,
    ChunkerFactory,
    create_chunker,
)

__version__ = "0.1.0"

__all__ = [
    "MoRAGBaseAgent",
    "AgentConfig",
    "GeminiProvider",
    "ProviderConfig",
    "AgentFactory",
    "create_agent",
    "create_agent_with_config",
    "ChunkingConfig",
    "ChunkingStrategy",
    "SemanticChunker",
    "ChunkerFactory",
    "create_chunker",
    "EntityExtractionResult",
    "RelationExtractionResult",
    "SummaryResult",
    "SemanticChunkingResult",
    "ContentAnalysisResult",
    "TranscriptAnalysisResult",
]