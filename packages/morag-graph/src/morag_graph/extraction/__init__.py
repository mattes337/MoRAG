"""LangExtract-based entity and relation extraction, and fact-based knowledge extraction."""

try:
    from .base import BaseExtractor, DummyExtractor, Entity, Relation
    from .entity_extractor import EntityExtractor
    from .entity_normalizer import EntityNormalizer
    from .fact_extractor import FactExtractor
    from .fact_graph_builder import FactGraphBuilder
    from .fact_validator import FactValidator
    from .relation_extractor import RelationExtractor

    __all__ = [
        "EntityExtractor",
        "RelationExtractor",
        "BaseExtractor",
        "Entity",
        "Relation",
        "DummyExtractor",
        "FactExtractor",
        "FactGraphBuilder",
        "FactValidator",
        "EntityNormalizer",
    ]
except ImportError as e:
    # Fallback to base classes only if imports fail
    from .base import BaseExtractor, DummyExtractor, Entity, Relation

    __all__ = ["BaseExtractor", "Entity", "Relation", "DummyExtractor"]
