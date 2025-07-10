"""MoRAG Reasoning Package.

Multi-hop reasoning capabilities for MoRAG.
"""

__version__ = "0.1.0"

# Core components
from .llm import LLMClient, LLMConfig
from .path_selection import PathSelectionAgent, ReasoningPathFinder, PathRelevanceScore, ReasoningStrategy
from .iterative_retrieval import IterativeRetriever, ContextGap, ContextAnalysis, RetrievalContext

# Intelligent retrieval components
from .intelligent_retrieval import IntelligentRetrievalService
from .intelligent_retrieval_models import (
    IntelligentRetrievalRequest,
    IntelligentRetrievalResponse,
    KeyFact,
    SourceInfo,
    EntityPath,
    RetrievalIteration,
    PathDecision,
)
from .entity_identification import EntityIdentificationService
from .recursive_path_follower import RecursivePathFollower
from .fact_extraction import FactExtractionService

__all__ = [
    "LLMClient",
    "LLMConfig",
    "PathSelectionAgent",
    "ReasoningPathFinder",
    "PathRelevanceScore",
    "ReasoningStrategy",
    "IterativeRetriever",
    "ContextGap",
    "ContextAnalysis",
    "RetrievalContext",
    # Intelligent retrieval
    "IntelligentRetrievalService",
    "IntelligentRetrievalRequest",
    "IntelligentRetrievalResponse",
    "KeyFact",
    "SourceInfo",
    "EntityPath",
    "RetrievalIteration",
    "PathDecision",
    "EntityIdentificationService",
    "RecursivePathFollower",
    "FactExtractionService",
]
