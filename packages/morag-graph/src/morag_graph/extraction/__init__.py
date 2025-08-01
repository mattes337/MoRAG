"""LangExtract-based entity and relation extraction."""

from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor

# All extraction is now handled by LangExtract
# Old systems (OpenIE, SpaCy, Pattern Matching, Hybrid) have been removed

__all__ = [
    "EntityExtractor",
    "RelationExtractor",
]