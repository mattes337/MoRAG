"""Batch builder service for collecting data during processing."""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from ..models.atomic_storage_batch import (
    AtomicStorageBatch,
    DocumentBatch,
    EntityBatch,
    FactBatch,
    RelationBatch,
)
from ..models.document import Document
from ..models.document_chunk import DocumentChunk
from ..models.entity import Entity
from ..models.fact import Fact, FactRelation
from ..models.relation import Relation


class AtomicBatchBuilder:
    """Builder for collecting data into atomic storage batches."""

    def __init__(self, source_document: Optional[str] = None, language: str = "en"):
        """Initialize the batch builder.

        Args:
            source_document: Source document path
            language: Content language
        """
        self.source_document = source_document
        self.language = language
        self.logger = structlog.get_logger(__name__)

        # Generate batch ID
        timestamp = datetime.utcnow().isoformat()
        source_hash = hashlib.md5((source_document or "unknown").encode()).hexdigest()[
            :8
        ]
        self.batch_id = f"batch_{source_hash}_{timestamp}"

        # Initialize batch
        self.batch = AtomicStorageBatch(
            batch_id=self.batch_id, source_document=source_document, language=language
        )

        # Tracking sets for deduplication
        self._document_ids = set()
        self._chunk_ids = set()
        self._entity_ids = set()
        self._fact_ids = set()
        self._relation_ids = set()
        self._fact_relation_ids = set()

    def add_document(self, document: Document) -> None:
        """Add a document to the batch.

        Args:
            document: Document to add
        """
        if document.id not in self._document_ids:
            self.batch.documents.add_document(document)
            self._document_ids.add(document.id)
            self.logger.debug("Added document to batch", document_id=document.id)

    def add_chunk(
        self, chunk: DocumentChunk, embedding: Optional[List[float]] = None
    ) -> None:
        """Add a document chunk to the batch.

        Args:
            chunk: Document chunk to add
            embedding: Optional embedding vector
        """
        if chunk.id not in self._chunk_ids:
            self.batch.documents.add_chunk(chunk, embedding)
            self._chunk_ids.add(chunk.id)
            self.logger.debug("Added chunk to batch", chunk_id=chunk.id)

    def add_entity(
        self, entity: Entity, embedding: Optional[List[float]] = None
    ) -> None:
        """Add an entity to the batch.

        Args:
            entity: Entity to add
            embedding: Optional embedding vector
        """
        if entity.id not in self._entity_ids:
            self.batch.entities.add_entity(entity, embedding)
            self._entity_ids.add(entity.id)
            self.logger.debug(
                "Added entity to batch", entity_id=entity.id, entity_name=entity.name
            )

    def add_fact(
        self,
        fact: Fact,
        embedding: Optional[List[float]] = None,
        chunk_id: Optional[str] = None,
    ) -> None:
        """Add a fact to the batch.

        Args:
            fact: Fact to add
            embedding: Optional embedding vector
            chunk_id: Optional source chunk ID for mapping
        """
        if fact.id not in self._fact_ids:
            # Use fact's source_chunk_id if chunk_id not provided
            source_chunk_id = chunk_id or fact.source_chunk_id
            self.batch.facts.add_fact(fact, embedding, source_chunk_id)
            self._fact_ids.add(fact.id)
            self.logger.debug(
                "Added fact to batch", fact_id=fact.id, chunk_id=source_chunk_id
            )

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the batch.

        Args:
            relation: Relation to add
        """
        if relation.id not in self._relation_ids:
            self.batch.relations.add_relation(relation)
            self._relation_ids.add(relation.id)
            self.logger.debug("Added relation to batch", relation_id=relation.id)

    def add_fact_relation(self, fact_relation: FactRelation) -> None:
        """Add a fact relation to the batch.

        Args:
            fact_relation: Fact relation to add
        """
        if fact_relation.id not in self._fact_relation_ids:
            self.batch.facts.add_fact_relation(fact_relation)
            self._fact_relation_ids.add(fact_relation.id)
            self.logger.debug(
                "Added fact relation to batch", relation_id=fact_relation.id
            )

    def add_entities_batch(
        self,
        entities: List[Entity],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Add multiple entities to the batch.

        Args:
            entities: List of entities to add
            embeddings: Optional embeddings dictionary (entity_id -> embedding)
        """
        for entity in entities:
            embedding = embeddings.get(entity.id) if embeddings else None
            self.add_entity(entity, embedding)

    def add_facts_batch(
        self, facts: List[Fact], embeddings: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """Add multiple facts to the batch.

        Args:
            facts: List of facts to add
            embeddings: Optional embeddings dictionary (fact_id -> embedding)
        """
        for fact in facts:
            embedding = embeddings.get(fact.id) if embeddings else None
            self.add_fact(fact, embedding)

    def add_relations_batch(self, relations: List[Relation]) -> None:
        """Add multiple relations to the batch.

        Args:
            relations: List of relations to add
        """
        for relation in relations:
            self.add_relation(relation)

    def add_processing_metadata(self, key: str, value: Any) -> None:
        """Add processing metadata to the batch.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.batch.processing_metadata[key] = value

    def get_batch(self) -> AtomicStorageBatch:
        """Get the current batch.

        Returns:
            Current atomic storage batch
        """
        return self.batch

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current batch.

        Returns:
            Summary dictionary with counts and metadata
        """
        return {
            "batch_id": self.batch_id,
            "source_document": self.source_document,
            "language": self.language,
            "created_at": self.batch.created_at,
            "total_items": self.batch.get_total_items(),
            "is_empty": self.batch.is_empty(),
            "processing_metadata": self.batch.processing_metadata,
        }

    def validate_batch(self) -> List[str]:
        """Validate the current batch for consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        return self.batch.validate_consistency()

    def clear(self) -> None:
        """Clear the current batch and start fresh."""
        self.batch = AtomicStorageBatch(
            batch_id=self.batch_id,
            source_document=self.source_document,
            language=self.language,
        )

        # Clear tracking sets
        self._document_ids.clear()
        self._chunk_ids.clear()
        self._entity_ids.clear()
        self._fact_ids.clear()
        self._relation_ids.clear()
        self._fact_relation_ids.clear()

        self.logger.info("Batch cleared", batch_id=self.batch_id)

    def merge_from_enhanced_processing(self, enhanced_data: Dict[str, Any]) -> None:
        """Merge data from enhanced fact processing results.

        Args:
            enhanced_data: Enhanced processing results dictionary
        """
        # Add entities
        entities = enhanced_data.get("entities", [])
        if entities:
            for entity_data in entities:
                if isinstance(entity_data, dict):
                    entity = Entity(**entity_data)
                else:
                    entity = entity_data
                self.add_entity(entity)

        # Add relations
        relations = enhanced_data.get("relations", [])
        if relations:
            for relation_data in relations:
                if isinstance(relation_data, dict):
                    relation = Relation(**relation_data)
                else:
                    relation = relation_data
                self.add_relation(relation)

        # Add processing metadata
        if "statistics" in enhanced_data:
            self.add_processing_metadata(
                "enhanced_processing_stats", enhanced_data["statistics"]
            )

        self.logger.info(
            "Merged enhanced processing data",
            entities_added=len(entities),
            relations_added=len(relations),
        )

    def merge_embeddings(
        self, entity_embeddings: Dict[str, Any], fact_embeddings: Dict[str, Any]
    ) -> None:
        """Merge embedding data into the batch.

        Args:
            entity_embeddings: Entity embeddings dictionary
            fact_embeddings: Fact embeddings dictionary
        """
        # Process entity embeddings
        for entity_key, embedding_data in entity_embeddings.items():
            if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                entity_name = embedding_data["name"]
                embedding_vector = embedding_data["embedding"]

                # Find matching entity by name
                for entity in self.batch.entities.entities:
                    if entity.name.lower() == entity_name.lower():
                        self.batch.entities.embeddings[entity.id] = embedding_vector
                        break

        # Process fact embeddings
        for fact_key, embedding_data in fact_embeddings.items():
            if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                fact_id = embedding_data.get("fact_id", fact_key)
                embedding_vector = embedding_data["embedding"]

                # Add to fact embeddings if fact exists
                if fact_id in self._fact_ids:
                    self.batch.facts.embeddings[fact_id] = embedding_vector

        self.logger.info(
            "Merged embeddings",
            entity_embeddings=len(entity_embeddings),
            fact_embeddings=len(fact_embeddings),
        )
