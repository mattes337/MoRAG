"""LangExtract-based entity and relation extraction, and fact-based knowledge extraction."""

try:
    from .entity_extractor import EntityExtractor
    from .relation_extractor import RelationExtractor
    from .base import BaseExtractor, Entity, Relation, DummyExtractor
    from .fact_extractor import FactExtractor
    from .fact_graph_builder import FactGraphBuilder
    from .fact_validator import FactValidator
    from .fact_prompts import FactExtractionPrompts, FactPromptTemplates

    __all__ = [
        "EntityExtractor", "RelationExtractor", "BaseExtractor", "Entity", "Relation", "DummyExtractor",
        "FactExtractor", "FactGraphBuilder", "FactValidator", "FactExtractionPrompts", "FactPromptTemplates"
    ]
except ImportError:
    from .base import BaseExtractor, Entity, Relation, DummyExtractor

    __all__ = ["BaseExtractor", "Entity", "Relation", "DummyExtractor"]