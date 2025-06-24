"""Models for hybrid retrieval system."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..models import Entity
from ..operations.traversal import GraphPath


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval system."""
    content: str
    source: str  # "vector", "graph", "hybrid", etc.
    score: float
    metadata: Dict[str, Any]
    entities: Optional[List[str]] = None
    reasoning: str = ""


@dataclass
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval system."""
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    max_vector_results: int = 20
    max_graph_results: int = 15
    fusion_strategy: str = "weighted_combination"  # "weighted_combination", "rank_fusion", "adaptive"
    min_confidence_threshold: float = 0.3


@dataclass
class ContextExpansionConfig:
    """Configuration for context expansion."""
    max_expansion_depth: int = 2
    max_entities_per_hop: int = 10
    relation_type_weights: Optional[Dict[str, float]] = None
    entity_type_priorities: Optional[Dict[str, float]] = None
    expansion_strategies: Optional[List[str]] = None


@dataclass
class ExpandedContext:
    """Result of context expansion."""
    original_entities: List[str]
    expanded_entities: List[Entity]
    expansion_paths: List[GraphPath]
    context_score: float
    expansion_reasoning: str


class RetrievalError(Exception):
    """Exception raised during retrieval operations."""
    pass


class VectorRetriever:
    """Mock interface for vector retriever - to be replaced with actual implementation."""
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of document results with content, score, and metadata
        """
        # This is a placeholder - actual implementation would use Qdrant or similar
        return []


@dataclass
class DocumentResult:
    """Document result from retrieval."""
    content: str
    score: float
    metadata: Dict[str, Any]
