"""LangExtract-based entity and relation extraction, and fact-based knowledge extraction."""

try:
    from .base import BaseExtractor, Entity, Relation, DummyExtractor
    # Removed obsolete fact_prompts - now using agents framework

    __all__ = [
        "EntityExtractor", "RelationExtractor", "BaseExtractor", "Entity", "Relation", "DummyExtractor",
        "FactExtractor", "FactGraphBuilder", "FactValidator",
        "EntityNormalizer"
    ]
except ImportError:
    from .base import BaseExtractor, Entity, Relation, DummyExtractor

    __all__ = ["BaseExtractor", "Entity", "Relation", "DummyExtractor"]
