"""Query processing models for graph-guided retrieval."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..models import Entity


@dataclass
class QueryEntity:
    """Entity extracted from a user query."""
    text: str
    entity_type: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0
    linked_entity_id: Optional[str] = None
    linked_entity: Optional[Entity] = None


@dataclass
class QueryAnalysis:
    """Complete analysis of a user query."""
    original_query: str
    entities: List[QueryEntity]
    intent: str
    query_type: str  # "factual", "exploratory", "comparative", etc.
    complexity_score: float
    intent_scores: Optional[Dict[str, float]] = None


class QueryProcessingError(Exception):
    """Exception raised during query processing."""
    pass
