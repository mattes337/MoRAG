"""LangExtract-based entity and relation extraction."""

try:
    from .entity_extractor import EntityExtractor
    from .relation_extractor import RelationExtractor
    from .base import BaseExtractor, Entity, Relation, DummyExtractor

    __all__ = ["EntityExtractor", "RelationExtractor", "BaseExtractor", "Entity", "Relation", "DummyExtractor"]
except ImportError:
    from .base import BaseExtractor, Entity, Relation, DummyExtractor

    __all__ = ["BaseExtractor", "Entity", "Relation", "DummyExtractor"]

# All extraction is now handled by LangExtract
# Old systems (OpenIE, SpaCy, Pattern Matching, Hybrid) have been removed

__all__ = [
    "EntityExtractor",
    "RelationExtractor",
]