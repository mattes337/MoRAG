# Task 2.2: Metadata Synchronization

## Overview

Implement comprehensive metadata synchronization between Neo4j and Qdrant systems to ensure data consistency, enable rich cross-system queries, and maintain synchronized entity information across both databases.

## Objectives

- Synchronize entity metadata between Neo4j and Qdrant
- Implement real-time metadata updates across systems
- Create metadata validation and consistency checking
- Enable metadata-driven query filtering and routing
- Establish metadata versioning and conflict resolution

## Current State Analysis

### Existing Metadata Patterns

**Neo4j**:
- Rich entity properties (name, type, description, etc.)
- Relationship metadata with context information
- Document and chunk metadata with processing information
- No synchronization with vector database metadata

**Qdrant**:
- Basic payload metadata (document_id, chunk_index)
- Limited entity information in vector payloads
- No real-time updates from graph database changes
- Inconsistent metadata schemas across collections

## Implementation Plan

### Step 1: Define Metadata Schema

Create standardized metadata schemas for cross-system synchronization:

```python
# src/morag_graph/schemas/metadata_schemas.py
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class MetadataType(Enum):
    ENTITY = "entity"
    DOCUMENT = "document"
    CHUNK = "chunk"
    RELATION = "relation"

@dataclass
class EntityMetadata:
    """Standardized entity metadata for cross-system sync."""
    entity_id: str
    name: str
    type: str
    description: Optional[str] = None
    aliases: List[str] = None
    confidence_score: float = 1.0
    source_documents: List[str] = None
    properties: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    version: int = 1
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.source_documents is None:
            self.source_documents = []
        if self.properties is None:
            self.properties = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties."""
        return {
            "id": self.entity_id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": self.aliases,
            "confidence_score": self.confidence_score,
            "source_documents": self.source_documents,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata_version": self.version
        }
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant point payload."""
        return {
            "entity_id": self.entity_id,
            "entity_name": self.name,
            "entity_type": self.type,
            "entity_description": self.description,
            "entity_aliases": self.aliases,
            "confidence_score": self.confidence_score,
            "source_documents": self.source_documents,
            "entity_properties": self.properties,
            "metadata_updated": self.updated_at.isoformat(),
            "metadata_version": self.version,
            "metadata_type": MetadataType.ENTITY.value
        }

@dataclass
class ChunkMetadata:
    """Standardized chunk metadata for cross-system sync."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    text_length: int
    mentioned_entities: List[str] = None
    entity_types: List[str] = None
    topics: List[str] = None
    language: str = "en"
    processing_metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    version: int = 1
    
    def __post_init__(self):
        if self.mentioned_entities is None:
            self.mentioned_entities = []
        if self.entity_types is None:
            self.entity_types = []
        if self.topics is None:
            self.topics = []
        if self.processing_metadata is None:
            self.processing_metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.text_length == 0:
            self.text_length = len(self.text)
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties."""
        return {
            "id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "text_length": self.text_length,
            "mentioned_entities": self.mentioned_entities,
            "entity_types": self.entity_types,
            "topics": self.topics,
            "language": self.language,
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata_version": self.version
        }
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant point payload."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "text_length": self.text_length,
            "mentioned_entities": self.mentioned_entities,
            "entity_types": self.entity_types,
            "topics": self.topics,
            "language": self.language,
            "processing_metadata": self.processing_metadata,
            "metadata_updated": self.updated_at.isoformat(),
            "metadata_version": self.version,
            "metadata_type": MetadataType.CHUNK.value
        }

@dataclass
class DocumentMetadata:
    """Standardized document metadata for cross-system sync."""
    document_id: str
    source_file: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    chunk_count: int = 0
    entity_count: int = 0
    processing_status: str = "pending"
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    version: int = 1
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties."""
        return {
            "id": self.document_id,
            "source_file": self.source_file,
            "title": self.title,
            "author": self.author,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count,
            "processing_status": self.processing_status,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata_version": self.version
        }
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant point payload."""
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "document_title": self.title,
            "document_author": self.author,
            "document_created_date": self.created_date.isoformat() if self.created_date else None,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count,
            "processing_status": self.processing_status,
            "document_tags": self.tags,
            "document_metadata": self.metadata,
            "metadata_updated": self.updated_at.isoformat(),
            "metadata_version": self.version,
            "metadata_type": MetadataType.DOCUMENT.value
        }
```

### Step 2: Create Metadata Synchronization Service

Implement `src/morag_graph/services/metadata_sync.py`:

```python
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import logging
from datetime import datetime
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from neo4j import AsyncSession

from ..schemas.metadata_schemas import (
    EntityMetadata, ChunkMetadata, DocumentMetadata, MetadataType
)

class MetadataSynchronizer:
    """Synchronizes metadata between Neo4j and Qdrant systems."""
    
    def __init__(self, neo4j_session: AsyncSession, qdrant_client: QdrantClient):
        self.neo4j_session = neo4j_session
        self.qdrant_client = qdrant_client
        self.logger = logging.getLogger(__name__)
    
    async def sync_entity_metadata(self, 
                                 entity_id: str,
                                 collection_name: str = "morag_entities") -> bool:
        """Sync entity metadata from Neo4j to Qdrant."""
        try:
            # Get entity metadata from Neo4j
            neo4j_query = """
            MATCH (entity:Entity {id: $entity_id})
            RETURN entity.id as id, entity.name as name, entity.type as type,
                   entity.description as description, entity.aliases as aliases,
                   entity.confidence_score as confidence_score,
                   entity.source_documents as source_documents,
                   entity.properties as properties,
                   entity.created_at as created_at,
                   entity.updated_at as updated_at,
                   entity.metadata_version as version
            """
            
            result = await self.neo4j_session.run(neo4j_query, entity_id=entity_id)
            record = await result.single()
            
            if not record:
                self.logger.warning(f"Entity {entity_id} not found in Neo4j")
                return False
            
            # Create EntityMetadata object
            entity_metadata = EntityMetadata(
                entity_id=record["id"],
                name=record["name"],
                type=record["type"],
                description=record["description"],
                aliases=record["aliases"] or [],
                confidence_score=record["confidence_score"] or 1.0,
                source_documents=record["source_documents"] or [],
                properties=record["properties"] or {},
                created_at=datetime.fromisoformat(record["created_at"]) if record["created_at"] else datetime.now(),
                updated_at=datetime.fromisoformat(record["updated_at"]) if record["updated_at"] else datetime.now(),
                version=record["version"] or 1
            )
            
            # Update Qdrant point payload
            await self.qdrant_client.set_payload(
                collection_name=collection_name,
                points=[entity_id],
                payload=entity_metadata.to_qdrant_payload()
            )
            
            self.logger.info(f"Synced entity metadata for {entity_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to sync entity metadata for {entity_id}: {e}")
            return False
    
    async def sync_chunk_metadata(self, 
                                chunk_id: str,
                                collection_name: str = "morag_vectors") -> bool:
        """Sync chunk metadata from Neo4j to Qdrant."""
        try:
            # Get chunk metadata from Neo4j
            neo4j_query = """
            MATCH (chunk:DocumentChunk {id: $chunk_id})
            OPTIONAL MATCH (chunk)-[:MENTIONED_IN]-(entity:Entity)
            WITH chunk, collect(entity.id) as mentioned_entities, 
                 collect(entity.type) as entity_types
            RETURN chunk.id as id, chunk.document_id as document_id,
                   chunk.chunk_index as chunk_index, chunk.text as text,
                   chunk.text_length as text_length,
                   mentioned_entities, entity_types,
                   chunk.topics as topics, chunk.language as language,
                   chunk.processing_metadata as processing_metadata,
                   chunk.created_at as created_at,
                   chunk.updated_at as updated_at,
                   chunk.metadata_version as version
            """
            
            result = await self.neo4j_session.run(neo4j_query, chunk_id=chunk_id)
            record = await result.single()
            
            if not record:
                self.logger.warning(f"Chunk {chunk_id} not found in Neo4j")
                return False
            
            # Create ChunkMetadata object
            chunk_metadata = ChunkMetadata(
                chunk_id=record["id"],
                document_id=record["document_id"],
                chunk_index=record["chunk_index"],
                text=record["text"],
                text_length=record["text_length"] or len(record["text"]),
                mentioned_entities=record["mentioned_entities"] or [],
                entity_types=record["entity_types"] or [],
                topics=record["topics"] or [],
                language=record["language"] or "en",
                processing_metadata=record["processing_metadata"] or {},
                created_at=datetime.fromisoformat(record["created_at"]) if record["created_at"] else datetime.now(),
                updated_at=datetime.fromisoformat(record["updated_at"]) if record["updated_at"] else datetime.now(),
                version=record["version"] or 1
            )
            
            # Update Qdrant point payload
            await self.qdrant_client.set_payload(
                collection_name=collection_name,
                points=[chunk_id],
                payload=chunk_metadata.to_qdrant_payload()
            )
            
            self.logger.info(f"Synced chunk metadata for {chunk_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to sync chunk metadata for {chunk_id}: {e}")
            return False
    
    async def sync_document_metadata(self, 
                                   document_id: str,
                                   collection_name: str = "morag_vectors") -> bool:
        """Sync document metadata to all related chunks in Qdrant."""
        try:
            # Get document metadata from Neo4j
            neo4j_query = """
            MATCH (doc:Document {id: $document_id})
            OPTIONAL MATCH (doc)-[:HAS_CHUNK]->(chunk:DocumentChunk)
            OPTIONAL MATCH (chunk)-[:MENTIONED_IN]-(entity:Entity)
            WITH doc, count(DISTINCT chunk) as chunk_count, 
                 count(DISTINCT entity) as entity_count
            RETURN doc.id as id, doc.source_file as source_file,
                   doc.title as title, doc.author as author,
                   doc.created_date as created_date, doc.file_type as file_type,
                   doc.file_size as file_size, doc.checksum as checksum,
                   chunk_count, entity_count,
                   doc.processing_status as processing_status,
                   doc.tags as tags, doc.metadata as metadata,
                   doc.created_at as created_at,
                   doc.updated_at as updated_at,
                   doc.metadata_version as version
            """
            
            result = await self.neo4j_session.run(neo4j_query, document_id=document_id)
            record = await result.single()
            
            if not record:
                self.logger.warning(f"Document {document_id} not found in Neo4j")
                return False
            
            # Create DocumentMetadata object
            document_metadata = DocumentMetadata(
                document_id=record["id"],
                source_file=record["source_file"],
                title=record["title"],
                author=record["author"],
                created_date=datetime.fromisoformat(record["created_date"]) if record["created_date"] else None,
                file_type=record["file_type"],
                file_size=record["file_size"],
                checksum=record["checksum"],
                chunk_count=record["chunk_count"] or 0,
                entity_count=record["entity_count"] or 0,
                processing_status=record["processing_status"] or "pending",
                tags=record["tags"] or [],
                metadata=record["metadata"] or {},
                created_at=datetime.fromisoformat(record["created_at"]) if record["created_at"] else datetime.now(),
                updated_at=datetime.fromisoformat(record["updated_at"]) if record["updated_at"] else datetime.now(),
                version=record["version"] or 1
            )
            
            # Get all chunks for this document
            chunks_query = """
            MATCH (doc:Document {id: $document_id})-[:HAS_CHUNK]->(chunk:DocumentChunk)
            RETURN chunk.id as chunk_id
            """
            
            chunks_result = await self.neo4j_session.run(chunks_query, document_id=document_id)
            chunk_ids = []
            
            async for chunk_record in chunks_result:
                chunk_ids.append(chunk_record["chunk_id"])
            
            # Update document metadata in all related chunk payloads
            if chunk_ids:
                document_payload = document_metadata.to_qdrant_payload()
                
                await self.qdrant_client.set_payload(
                    collection_name=collection_name,
                    points=chunk_ids,
                    payload=document_payload
                )
                
                self.logger.info(f"Synced document metadata for {document_id} to {len(chunk_ids)} chunks")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to sync document metadata for {document_id}: {e}")
            return False
    
    async def batch_sync_metadata(self, 
                                metadata_type: MetadataType,
                                ids: List[str],
                                collection_name: str,
                                batch_size: int = 50) -> Dict[str, int]:
        """Batch synchronize metadata for multiple items."""
        sync_stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0
        }
        
        # Select appropriate sync method
        if metadata_type == MetadataType.ENTITY:
            sync_method = self.sync_entity_metadata
        elif metadata_type == MetadataType.CHUNK:
            sync_method = self.sync_chunk_metadata
        elif metadata_type == MetadataType.DOCUMENT:
            sync_method = self.sync_document_metadata
        else:
            self.logger.error(f"Unsupported metadata type: {metadata_type}")
            return sync_stats
        
        # Process in batches
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for item_id in batch:
                task = sync_method(item_id, collection_name)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                sync_stats["processed"] += 1
                if isinstance(result, Exception):
                    sync_stats["failed"] += 1
                    self.logger.error(f"Batch sync error: {result}")
                elif result:
                    sync_stats["successful"] += 1
                else:
                    sync_stats["failed"] += 1
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        return sync_stats
    
    async def validate_metadata_consistency(self, 
                                          collection_name: str = "morag_vectors",
                                          sample_size: int = 100) -> Dict[str, Any]:
        """Validate metadata consistency between Neo4j and Qdrant."""
        validation_results = {
            "total_checked": 0,
            "consistent": 0,
            "inconsistent": [],
            "missing_in_qdrant": [],
            "missing_in_neo4j": []
        }
        
        try:
            # Get sample of chunks from Neo4j
            neo4j_query = """
            MATCH (chunk:DocumentChunk)
            WHERE chunk.qdrant_point_id IS NOT NULL
            RETURN chunk.id as chunk_id, chunk.updated_at as neo4j_updated,
                   chunk.metadata_version as neo4j_version
            LIMIT $sample_size
            """
            
            result = await self.neo4j_session.run(neo4j_query, sample_size=sample_size)
            neo4j_chunks = []
            
            async for record in result:
                neo4j_chunks.append({
                    "chunk_id": record["chunk_id"],
                    "neo4j_updated": record["neo4j_updated"],
                    "neo4j_version": record["neo4j_version"]
                })
            
            # Check corresponding Qdrant points
            chunk_ids = [chunk["chunk_id"] for chunk in neo4j_chunks]
            
            if chunk_ids:
                points = await self.qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=chunk_ids,
                    with_payload=True
                )
                
                qdrant_data = {}
                for point in points:
                    if point.payload:
                        qdrant_data[str(point.id)] = {
                            "qdrant_updated": point.payload.get("metadata_updated"),
                            "qdrant_version": point.payload.get("metadata_version")
                        }
                
                # Compare metadata
                for chunk in neo4j_chunks:
                    chunk_id = chunk["chunk_id"]
                    validation_results["total_checked"] += 1
                    
                    if chunk_id not in qdrant_data:
                        validation_results["missing_in_qdrant"].append(chunk_id)
                        continue
                    
                    qdrant_info = qdrant_data[chunk_id]
                    
                    # Compare versions and timestamps
                    neo4j_version = chunk["neo4j_version"] or 1
                    qdrant_version = qdrant_info["qdrant_version"] or 1
                    
                    if neo4j_version == qdrant_version:
                        validation_results["consistent"] += 1
                    else:
                        validation_results["inconsistent"].append({
                            "chunk_id": chunk_id,
                            "neo4j_version": neo4j_version,
                            "qdrant_version": qdrant_version,
                            "neo4j_updated": chunk["neo4j_updated"],
                            "qdrant_updated": qdrant_info["qdrant_updated"]
                        })
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Metadata validation failed: {e}")
            return validation_results
    
    async def resolve_metadata_conflicts(self, 
                                       inconsistent_items: List[Dict[str, Any]],
                                       resolution_strategy: str = "neo4j_wins") -> int:
        """Resolve metadata conflicts between systems."""
        resolved_count = 0
        
        for item in inconsistent_items:
            chunk_id = item["chunk_id"]
            
            try:
                if resolution_strategy == "neo4j_wins":
                    # Sync from Neo4j to Qdrant
                    success = await self.sync_chunk_metadata(chunk_id)
                    if success:
                        resolved_count += 1
                        self.logger.info(f"Resolved conflict for {chunk_id} (Neo4j wins)")
                
                elif resolution_strategy == "latest_wins":
                    # Compare timestamps and sync from the latest
                    neo4j_time = datetime.fromisoformat(item["neo4j_updated"]) if item["neo4j_updated"] else datetime.min
                    qdrant_time = datetime.fromisoformat(item["qdrant_updated"]) if item["qdrant_updated"] else datetime.min
                    
                    if neo4j_time >= qdrant_time:
                        success = await self.sync_chunk_metadata(chunk_id)
                        if success:
                            resolved_count += 1
                            self.logger.info(f"Resolved conflict for {chunk_id} (Neo4j latest)")
                    # Note: Qdrant to Neo4j sync would require additional implementation
                
            except Exception as e:
                self.logger.error(f"Failed to resolve conflict for {chunk_id}: {e}")
        
        return resolved_count
```

### Step 3: Create Real-time Metadata Update Service

Implement `src/morag_graph/services/metadata_events.py`:

```python
from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class MetadataEventType(Enum):
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_DELETED = "entity_deleted"
    CHUNK_CREATED = "chunk_created"
    CHUNK_UPDATED = "chunk_updated"
    CHUNK_DELETED = "chunk_deleted"
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_DELETED = "document_deleted"

@dataclass
class MetadataEvent:
    """Represents a metadata change event."""
    event_type: MetadataEventType
    item_id: str
    item_type: str  # 'Entity', 'DocumentChunk', 'Document'
    changes: Dict[str, Any]
    timestamp: datetime
    source_system: str  # 'neo4j' or 'qdrant'
    
class MetadataEventHandler:
    """Handles real-time metadata synchronization events."""
    
    def __init__(self, metadata_synchronizer):
        self.metadata_synchronizer = metadata_synchronizer
        self.logger = logging.getLogger(__name__)
        self.event_queue = asyncio.Queue()
        self.handlers: Dict[MetadataEventType, List[Callable]] = {}
        self.running = False
    
    def register_handler(self, event_type: MetadataEventType, handler: Callable):
        """Register an event handler for a specific event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    async def emit_event(self, event: MetadataEvent):
        """Emit a metadata change event."""
        await self.event_queue.put(event)
        self.logger.debug(f"Emitted event: {event.event_type} for {event.item_id}")
    
    async def start_processing(self):
        """Start processing metadata events."""
        self.running = True
        self.logger.info("Started metadata event processing")
        
        while self.running:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing metadata event: {e}")
    
    async def stop_processing(self):
        """Stop processing metadata events."""
        self.running = False
        self.logger.info("Stopped metadata event processing")
    
    async def _process_event(self, event: MetadataEvent):
        """Process a single metadata event."""
        try:
            # Call registered handlers
            if event.event_type in self.handlers:
                for handler in self.handlers[event.event_type]:
                    await handler(event)
            
            # Default synchronization behavior
            await self._default_sync_handler(event)
            
        except Exception as e:
            self.logger.error(f"Failed to process event {event.event_type} for {event.item_id}: {e}")
    
    async def _default_sync_handler(self, event: MetadataEvent):
        """Default handler for metadata synchronization."""
        if event.source_system == "neo4j":
            # Sync from Neo4j to Qdrant
            if event.event_type in [MetadataEventType.ENTITY_CREATED, MetadataEventType.ENTITY_UPDATED]:
                await self.metadata_synchronizer.sync_entity_metadata(event.item_id)
            
            elif event.event_type in [MetadataEventType.CHUNK_CREATED, MetadataEventType.CHUNK_UPDATED]:
                await self.metadata_synchronizer.sync_chunk_metadata(event.item_id)
            
            elif event.event_type in [MetadataEventType.DOCUMENT_CREATED, MetadataEventType.DOCUMENT_UPDATED]:
                await self.metadata_synchronizer.sync_document_metadata(event.item_id)
        
        # Note: Qdrant to Neo4j sync would require additional implementation
```

## Testing Strategy

### Unit Tests

Create `tests/test_metadata_sync.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_graph.services.metadata_sync import MetadataSynchronizer
from morag_graph.schemas.metadata_schemas import EntityMetadata, ChunkMetadata

@pytest.fixture
def mock_neo4j_session():
    return AsyncMock()

@pytest.fixture
def mock_qdrant_client():
    return AsyncMock()

@pytest.fixture
def metadata_sync(mock_neo4j_session, mock_qdrant_client):
    return MetadataSynchronizer(mock_neo4j_session, mock_qdrant_client)

@pytest.mark.asyncio
async def test_sync_entity_metadata_success(metadata_sync, mock_neo4j_session, mock_qdrant_client):
    # Mock Neo4j response
    mock_result = AsyncMock()
    mock_result.single.return_value = {
        "id": "entity_123",
        "name": "John Doe",
        "type": "PERSON",
        "description": "A person",
        "aliases": ["Johnny"],
        "confidence_score": 0.95,
        "source_documents": ["doc_1"],
        "properties": {"age": 30},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "version": 1
    }
    mock_neo4j_session.run.return_value = mock_result
    
    # Mock Qdrant response
    mock_qdrant_client.set_payload.return_value = None
    
    # Test
    success = await metadata_sync.sync_entity_metadata("entity_123")
    
    assert success is True
    mock_neo4j_session.run.assert_called_once()
    mock_qdrant_client.set_payload.assert_called_once()

@pytest.mark.asyncio
async def test_batch_sync_metadata(metadata_sync):
    # Mock sync methods
    metadata_sync.sync_entity_metadata = AsyncMock(return_value=True)
    
    # Test
    from morag_graph.schemas.metadata_schemas import MetadataType
    stats = await metadata_sync.batch_sync_metadata(
        MetadataType.ENTITY,
        ["entity_1", "entity_2", "entity_3"],
        "test_collection",
        batch_size=2
    )
    
    assert stats["processed"] == 3
    assert stats["successful"] == 3
    assert stats["failed"] == 0
    assert metadata_sync.sync_entity_metadata.call_count == 3

@pytest.mark.asyncio
async def test_validate_metadata_consistency(metadata_sync, mock_neo4j_session, mock_qdrant_client):
    # Mock Neo4j response
    mock_result = AsyncMock()
    mock_result.__aiter__.return_value = iter([
        {
            "chunk_id": "chunk_1",
            "neo4j_updated": "2024-01-01T00:00:00",
            "neo4j_version": 1
        },
        {
            "chunk_id": "chunk_2",
            "neo4j_updated": "2024-01-01T00:00:00",
            "neo4j_version": 2
        }
    ])
    mock_neo4j_session.run.return_value = mock_result
    
    # Mock Qdrant response
    mock_point_1 = MagicMock()
    mock_point_1.id = "chunk_1"
    mock_point_1.payload = {
        "metadata_updated": "2024-01-01T00:00:00",
        "metadata_version": 1
    }
    
    mock_point_2 = MagicMock()
    mock_point_2.id = "chunk_2"
    mock_point_2.payload = {
        "metadata_updated": "2024-01-01T00:00:00",
        "metadata_version": 1  # Different version
    }
    
    mock_qdrant_client.retrieve.return_value = [mock_point_1, mock_point_2]
    
    # Test
    results = await metadata_sync.validate_metadata_consistency()
    
    assert results["total_checked"] == 2
    assert results["consistent"] == 1
    assert len(results["inconsistent"]) == 1
    assert results["inconsistent"][0]["chunk_id"] == "chunk_2"
```

### Integration Tests

Create `tests/integration/test_metadata_integration.py`:

```python
import pytest
import asyncio
from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase

from morag_graph.services.metadata_sync import MetadataSynchronizer
from morag_graph.schemas.metadata_schemas import EntityMetadata, MetadataType

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_metadata_sync():
    # Setup test databases
    qdrant_client = QdrantClient(":memory:")
    neo4j_driver = AsyncGraphDatabase.driver("bolt://localhost:7687")
    
    async with neo4j_driver.session() as session:
        metadata_sync = MetadataSynchronizer(session, qdrant_client)
        
        # Create test entity in Neo4j
        entity_id = "test_entity_123"
        await session.run(
            "CREATE (e:Entity {id: $id, name: $name, type: $type, metadata_version: 1})",
            id=entity_id, name="Test Entity", type="PERSON"
        )
        
        # Sync metadata
        success = await metadata_sync.sync_entity_metadata(entity_id)
        assert success
        
        # Validate consistency
        validation_results = await metadata_sync.validate_metadata_consistency()
        assert validation_results["total_checked"] >= 1
        
        # Cleanup
        await session.run("MATCH (e:Entity {id: $id}) DELETE e", id=entity_id)
    
    await neo4j_driver.close()
```

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**: Process metadata updates in batches
2. **Async Operations**: Use asyncio for concurrent synchronization
3. **Event-Driven Updates**: Real-time sync for critical changes
4. **Selective Sync**: Only sync changed metadata fields
5. **Caching**: Cache frequently accessed metadata

### Performance Targets

- Single metadata sync: < 20ms
- Batch sync (100 items): < 2s
- Metadata validation: < 10s per 1000 items
- Event processing latency: < 100ms
- Consistency check: < 30s per 10,000 items

## Success Criteria

- [ ] Metadata schemas implemented and validated
- [ ] Real-time synchronization working for all metadata types
- [ ] Batch synchronization handles large datasets efficiently
- [ ] Metadata consistency validation passes >95% accuracy
- [ ] Event-driven updates process within target latency
- [ ] Conflict resolution strategies implemented and tested
- [ ] Comprehensive test coverage (>90%)

## Risk Assessment

**Risk Level**: Medium

**Key Risks**:
- Metadata drift during high-frequency updates
- Performance degradation with large metadata payloads
- Event processing bottlenecks
- Schema evolution compatibility issues

**Mitigation Strategies**:
- Implement metadata versioning
- Use efficient serialization formats
- Add event queue monitoring and scaling
- Design backward-compatible schema changes

## Rollback Plan

1. **Immediate Rollback**: Disable real-time metadata sync
2. **Data Restoration**: Restore metadata from backup snapshots
3. **Service Isolation**: Revert to independent metadata management
4. **Validation**: Verify system functionality without sync

## Next Steps

- **Task 2.3**: [ID Mapping Utilities](./task-2.3-id-mapping-utilities.md)
- **Task 3.1**: [Neo4j Vector Storage](./task-3.1-neo4j-vector-storage.md)

## Dependencies

- **Task 2.1**: Bidirectional Reference Storage (must be completed)
- **Task 1.1**: Unified ID Architecture (must be completed)
- **Task 1.2**: Document and Chunk ID Standardization (must be completed)

## Estimated Time

**Total**: 4-5 days

- Schema design: 1 day
- Implementation: 2.5 days
- Testing: 1 day
- Documentation: 0.5 days

## Status

- [ ] Planning
- [ ] Implementation
- [ ] Testing
- [ ] Documentation
- [ ] Review
- [ ] Deployment