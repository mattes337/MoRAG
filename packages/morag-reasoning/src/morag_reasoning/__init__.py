"""MoRAG Reasoning Package.

Multi-hop reasoning capabilities for MoRAG.
"""

__version__ = "0.1.0"

# Core components
from .llm import LLMClient, LLMConfig
from .batch_processor import (
    BatchProcessor,
    BatchItem,
    BatchResult,
    TextAnalysisBatchProcessor,
    DocumentChunkBatchProcessor,
    batch_llm_calls,
    batch_text_analysis,
    batch_document_chunks
)
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

# Recursive fact retrieval components
from .recursive_fact_models import (
    SourceMetadata, RawFact, ScoredFact, FinalFact,
    RecursiveFactRetrievalRequest, RecursiveFactRetrievalResponse,
    TraversalStep, GTAResponse, FCAResponse
)
from .graph_traversal_agent import GraphTraversalAgent
from .fact_critic_agent import FactCriticAgent
from .recursive_fact_retrieval_service import RecursiveFactRetrievalService

# Enhanced fact gathering and scoring components
from .graph_fact_extractor import GraphFactExtractor, ExtractedFact, FactType
from .fact_scorer import FactRelevanceScorer, ScoredFact as EnhancedScoredFact, ScoringDimensions, ScoringStrategy
from .citation_manager import CitationManager, CitedFact, SourceReference, CitationFormat

# Response generation components
from .response_generator import ResponseGenerator, GeneratedResponse, ResponseFormat, ResponseOptions
from .citation_integrator import CitationIntegrator, CitedResponse, CitationStyle, CitationOptions
from .response_assessor import ResponseQualityAssessor, QualityAssessment, QualityMetrics, AssessmentOptions

__all__ = [
    "LLMClient",
    "LLMConfig",
    "BatchProcessor",
    "BatchItem",
    "BatchResult",
    "TextAnalysisBatchProcessor",
    "DocumentChunkBatchProcessor",
    "batch_llm_calls",
    "batch_text_analysis",
    "batch_document_chunks",
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
    # Recursive fact retrieval
    "SourceMetadata",
    "RawFact",
    "ScoredFact",
    "FinalFact",
    "RecursiveFactRetrievalRequest",
    "RecursiveFactRetrievalResponse",
    "TraversalStep",
    "GTAResponse",
    "FCAResponse",
    "GraphTraversalAgent",
    "FactCriticAgent",
    "RecursiveFactRetrievalService",
    # Enhanced fact gathering and scoring
    "GraphFactExtractor",
    "ExtractedFact",
    "FactType",
    "FactRelevanceScorer",
    "EnhancedScoredFact",
    "ScoringDimensions",
    "ScoringStrategy",
    "CitationManager",
    "CitedFact",
    "SourceReference",
    "CitationFormat",
    # Response generation components
    "ResponseGenerator",
    "GeneratedResponse",
    "ResponseFormat",
    "ResponseOptions",
    "CitationIntegrator",
    "CitedResponse",
    "CitationStyle",
    "CitationOptions",
    "ResponseQualityAssessor",
    "QualityAssessment",
    "QualityMetrics",
    "AssessmentOptions",
]
