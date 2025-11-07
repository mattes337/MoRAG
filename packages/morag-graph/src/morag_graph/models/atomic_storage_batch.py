"""Atomic storage batch models for collecting all data before database storage."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from .document import Document
from .document_chunk import DocumentChunk
from .entity import Entity
from .relation import Relation
from .fact import Fact, FactRelation


class EntityBatch(BaseModel):
    """Batch of entities to be stored atomically."""
    entities: List[Entity] = Field(default_factory=list, description="List of entities")
    embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Entity embeddings by entity ID")

    def add_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> None:
        """Add an entity to the batch."""
        self.entities.append(entity)
        if embedding:
            self.embeddings[entity.id] = embedding

    def get_entity_count(self) -> int:
        """Get total number of entities in batch."""
        return len(self.entities)


class FactBatch(BaseModel):
    """Batch of facts to be stored atomically."""
    facts: List[Fact] = Field(default_factory=list, description="List of facts")
    fact_relations: List[FactRelation] = Field(default_factory=list, description="List of fact relationships")
    embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Fact embeddings by fact ID")
    chunk_mappings: Dict[str, List[str]] = Field(default_factory=dict, description="Chunk ID to fact IDs mapping")

    def add_fact(self, fact: Fact, embedding: Optional[List[float]] = None, chunk_id: Optional[str] = None) -> None:
        """Add a fact to the batch."""
        self.facts.append(fact)
        if embedding:
            self.embeddings[fact.id] = embedding
        if chunk_id:
            if chunk_id not in self.chunk_mappings:
                self.chunk_mappings[chunk_id] = []
            self.chunk_mappings[chunk_id].append(fact.id)

    def add_fact_relation(self, relation: FactRelation) -> None:
        """Add a fact relationship to the batch."""
        self.fact_relations.append(relation)

    def get_fact_count(self) -> int:
        """Get total number of facts in batch."""
        return len(self.facts)


class RelationBatch(BaseModel):
    """Batch of relations to be stored atomically."""
    relations: List[Relation] = Field(default_factory=list, description="List of relations")

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the batch."""
        self.relations.append(relation)

    def get_relation_count(self) -> int:
        """Get total number of relations in batch."""
        return len(self.relations)


class DocumentBatch(BaseModel):
    """Batch of documents and chunks to be stored atomically."""
    documents: List[Document] = Field(default_factory=list, description="List of documents")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of document chunks")
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Chunk embeddings by chunk ID")

    def add_document(self, document: Document) -> None:
        """Add a document to the batch."""
        self.documents.append(document)

    def add_chunk(self, chunk: DocumentChunk, embedding: Optional[List[float]] = None) -> None:
        """Add a document chunk to the batch."""
        self.chunks.append(chunk)
        if embedding:
            self.chunk_embeddings[chunk.id] = embedding

    def get_document_count(self) -> int:
        """Get total number of documents in batch."""
        return len(self.documents)

    def get_chunk_count(self) -> int:
        """Get total number of chunks in batch."""
        return len(self.chunks)


class AtomicStorageBatch(BaseModel):
    """Complete batch of all data to be stored atomically."""

    # Metadata
    batch_id: str = Field(..., description="Unique batch identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation timestamp")
    source_document: Optional[str] = Field(None, description="Source document path")
    language: str = Field(default="en", description="Content language")

    # Data batches
    documents: DocumentBatch = Field(default_factory=DocumentBatch, description="Document and chunk batch")
    entities: EntityBatch = Field(default_factory=EntityBatch, description="Entity batch")
    facts: FactBatch = Field(default_factory=FactBatch, description="Fact batch")
    relations: RelationBatch = Field(default_factory=RelationBatch, description="Relation batch")

    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")

    def get_total_items(self) -> Dict[str, int]:
        """Get count of all items in the batch."""
        return {
            "documents": self.documents.get_document_count(),
            "chunks": self.documents.get_chunk_count(),
            "entities": self.entities.get_entity_count(),
            "facts": self.facts.get_fact_count(),
            "fact_relations": len(self.facts.fact_relations),
            "relations": self.relations.get_relation_count()
        }

    def is_empty(self) -> bool:
        """Check if the batch is empty."""
        totals = self.get_total_items()
        return all(count == 0 for count in totals.values())

    def validate_consistency(self) -> List[str]:
        """Validate internal consistency of the batch."""
        errors = []

        # Check that all fact source_chunk_ids reference existing chunks
        chunk_ids = {chunk.id for chunk in self.documents.chunks}
        for fact in self.facts.facts:
            if fact.source_chunk_id and fact.source_chunk_id not in chunk_ids:
                errors.append(f"Fact {fact.id} references non-existent chunk {fact.source_chunk_id}")

        # Check that all relations reference existing entities or facts
        entity_ids = {entity.id for entity in self.entities.entities}
        fact_ids = {fact.id for fact in self.facts.facts}

        for relation in self.relations.relations:
            if relation.source_entity_id not in entity_ids and relation.source_entity_id not in fact_ids:
                errors.append(f"Relation {relation.id} references non-existent source {relation.source_entity_id}")
            if relation.target_entity_id not in entity_ids and relation.target_entity_id not in fact_ids:
                errors.append(f"Relation {relation.id} references non-existent target {relation.target_entity_id}")

        # Check fact relations
        for fact_relation in self.facts.fact_relations:
            if fact_relation.source_fact_id not in fact_ids:
                errors.append(f"FactRelation {fact_relation.id} references non-existent source fact {fact_relation.source_fact_id}")
            if fact_relation.target_fact_id not in fact_ids:
                errors.append(f"FactRelation {fact_relation.id} references non-existent target fact {fact_relation.target_fact_id}")

        return errors


class AtomicStorageResult(BaseModel):
    """Result of atomic storage operation."""

    success: bool = Field(..., description="Whether storage was successful")
    batch_id: str = Field(..., description="Batch identifier")
    stored_counts: Dict[str, int] = Field(default_factory=dict, description="Count of items stored by type")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    storage_duration_seconds: float = Field(default=0.0, description="Time taken for storage operation")
    database_results: Dict[str, Any] = Field(default_factory=dict, description="Database-specific results")

    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False

    def set_stored_count(self, item_type: str, count: int) -> None:
        """Set the stored count for an item type."""
        self.stored_counts[item_type] = count
