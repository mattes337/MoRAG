"""LLM-based entity and relation extraction with pattern matching."""

from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
from .base import BaseExtractor
from .pattern_matcher import EntityPatternMatcher, EntityPattern, PatternType
from .hybrid_extractor import HybridEntityExtractor

# SpaCy extractor
try:
    from .spacy_extractor import SpacyEntityExtractor
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    SpacyEntityExtractor = None

# OpenIE extractors
try:
    from ..extractors import OpenIEExtractor
    _OPENIE_AVAILABLE = True
except ImportError:
    _OPENIE_AVAILABLE = False
    OpenIEExtractor = None

__all__ = [
    "EntityExtractor",
    "RelationExtractor",
    "BaseExtractor",
    "EntityPatternMatcher",
    "EntityPattern",
    "PatternType",
    "HybridEntityExtractor",
]

if _SPACY_AVAILABLE:
    __all__.append("SpacyEntityExtractor")

if _OPENIE_AVAILABLE:
    __all__.append("OpenIEExtractor")