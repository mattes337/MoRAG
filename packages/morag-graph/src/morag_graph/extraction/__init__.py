"""LangExtract-based entity and relation extraction, and fact-based knowledge extraction."""

try:
    from .entity_extractor import EntityExtractor
    from .relation_extractor import RelationExtractor
    from .base import BaseExtractor, Entity, Relation, DummyExtractor
    from .fact_extractor import FactExtractor
    from .fact_graph_builder import FactGraphBuilder
    from .fact_validator import FactValidator
    # Removed obsolete fact_prompts - now using agents framework
    from .entity_normalizer import LLMEntityNormalizer as EntityNormalizer

    __all__ = [
        "EntityExtractor", "RelationExtractor", "BaseExtractor", "Entity", "Relation", "DummyExtractor",
        "FactExtractor", "FactGraphBuilder", "FactValidator",
        "EntityNormalizer"
    ]
except ImportError:
    from .base import BaseExtractor, Entity, Relation, DummyExtractor

    __all__ = ["BaseExtractor", "Entity", "Relation", "DummyExtractor"]