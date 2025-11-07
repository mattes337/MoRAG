"""Base classes for graph extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Entity:
    """Represents an entity in the graph."""

    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class Relation:
    """Represents a relation between entities."""

    source: str
    target: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 1.0


class BaseExtractor(ABC):
    """Base class for entity and relation extractors."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""

    @abstractmethod
    async def extract_relations(
        self, text: str, entities: List[Entity]
    ) -> List[Relation]:
        """Extract relations from text given entities."""

    async def extract(self, text: str) -> Dict[str, Any]:
        """Extract both entities and relations from text."""
        entities = await self.extract_entities(text)
        relations = await self.extract_relations(text, entities)

        return {"entities": entities, "relations": relations, "extractor": self.name}


class DummyExtractor(BaseExtractor):
    """Dummy extractor for testing and fallback."""

    def __init__(self):
        super().__init__("dummy")

    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract dummy entities."""
        return []

    async def extract_relations(
        self, text: str, entities: List[Entity]
    ) -> List[Relation]:
        """Extract dummy relations."""
        return []
