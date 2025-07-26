"""Normalizers package for MoRAG graph processing."""

from .entity_linker import EntityLinker
from .entity_normalizer import EntityNormalizer
from .confidence_manager import ConfidenceManager
from .predicate_normalizer import PredicateNormalizer
from .relationship_categorizer import RelationshipCategorizer

# SpaCy normalizer
try:
    from .spacy_normalizer import SpacyNormalizer
    _SPACY_NORMALIZER_AVAILABLE = True
except ImportError:
    _SPACY_NORMALIZER_AVAILABLE = False
    SpacyNormalizer = None

__all__ = ["EntityLinker", "EntityNormalizer", "ConfidenceManager", "PredicateNormalizer", "RelationshipCategorizer"]

if _SPACY_NORMALIZER_AVAILABLE:
    __all__.append("SpacyNormalizer")
