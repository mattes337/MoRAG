# Task 1.2: Core Graph Package Creation

**Phase**: 1 - Foundation Infrastructure  
**Priority**: Critical  
**Total Estimated Time**: 6-8 days  
**Dependencies**: Task 1.1 (Graph Database Setup)

## Overview

This task creates the `morag-graph` package, which provides the core graph data models, storage interfaces, and basic operations for the graph-augmented RAG system.

## Subtasks

### Task 1.2.1: Create morag-graph Package
**Priority**: Critical  
**Estimated Time**: 2-3 days  
**Dependencies**: 1.1.1

#### Implementation Steps

1. **Package Structure Setup**
   - Create package directory structure
   - Set up pyproject.toml with dependencies
   - Initialize core modules and interfaces

2. **Core Data Models**
   - Define Entity, Relation, and Graph models
   - Implement serialization/deserialization
   - Add validation and type checking

3. **Base Storage Interface**
   - Define abstract storage interface
   - Create common exceptions and types
   - Implement base functionality

#### Package Structure
```
packages/morag-graph/
├── src/morag_graph/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── entity.py
│   │   ├── relation.py
│   │   ├── graph.py
│   │   └── types.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── neo4j_storage.py
│   │   └── exceptions.py
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── crud.py
│   │   ├── traversal.py
│   │   └── analytics.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   └── serialization.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_storage.py
│   └── test_operations.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_operations.py
├── pyproject.toml
└── README.md
```

#### Code Examples

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "morag-graph"
version = "0.1.0"
description = "Graph database integration for MoRAG"
authors = [{name = "MoRAG Team"}]
requires-python = ">=3.9"
dependencies = [
    "neo4j>=5.15.0",
    "pydantic>=2.0.0",
    "asyncio-compat>=0.1.0",
    "numpy>=1.24.0",
    "typing-extensions>=4.5.0",
    "uuid>=1.30",
    "python-dateutil>=2.8.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0"
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 88
```

**Entity Model**:
```python
# src/morag_graph/models/entity.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4
import json

@dataclass
class Entity:
    """Represents a knowledge graph entity"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    type: str = ""
    summary: str = ""
    aliases: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_documents: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if not self.name:
            raise ValueError("Entity name cannot be empty")
        if not self.type:
            raise ValueError("Entity type cannot be empty")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for storage"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "summary": self.summary,
            "aliases": self.aliases,
            "embedding": self.embedding,
            "sparse_vector": json.dumps(self.sparse_vector) if self.sparse_vector else None,
            "metadata": json.dumps(self.metadata),
            "source_documents": self.source_documents,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            summary=data.get("summary", ""),
            aliases=data.get("aliases", []),
            embedding=data.get("embedding"),
            sparse_vector=json.loads(data["sparse_vector"]) if data.get("sparse_vector") else None,
            metadata=json.loads(data.get("metadata", "{}")),
            source_documents=data.get("source_documents", []),
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
    
    def add_source_document(self, document_id: str):
        """Add a source document reference"""
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)
            self.updated_at = datetime.utcnow()
    
    def update_confidence(self, new_confidence: float):
        """Update entity confidence score"""
        if new_confidence < 0 or new_confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        self.confidence = new_confidence
        self.updated_at = datetime.utcnow()
```

**Relation Model**:
```python
# src/morag_graph/models/relation.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4
import json

@dataclass
class Relation:
    """Represents a relationship between two entities"""
    id: str = field(default_factory=lambda: str(uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: str = ""
    description: str = ""
    confidence: float = 1.0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_documents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate relation after initialization"""
        if not self.source_entity_id:
            raise ValueError("Source entity ID cannot be empty")
        if not self.target_entity_id:
            raise ValueError("Target entity ID cannot be empty")
        if not self.relation_type:
            raise ValueError("Relation type cannot be empty")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("Source and target entities cannot be the same")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary for storage"""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type,
            "description": self.description,
            "confidence": self.confidence,
            "weight": self.weight,
            "metadata": json.dumps(self.metadata),
            "source_documents": self.source_documents,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create relation from dictionary"""
        return cls(
            id=data["id"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            relation_type=data["relation_type"],
            description=data.get("description", ""),
            confidence=data.get("confidence", 1.0),
            weight=data.get("weight", 1.0),
            metadata=json.loads(data.get("metadata", "{}")),
            source_documents=data.get("source_documents", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
```

**Base Storage Interface**:
```python
# src/morag_graph/storage/base.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.types import GraphPath, TraversalOptions

class BaseGraphStorage(ABC):
    """Abstract base class for graph storage implementations"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the graph database"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the graph database"""
        pass
    
    # Entity operations
    @abstractmethod
    async def create_entity(self, entity: Entity) -> str:
        """Create a new entity and return its ID"""
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID"""
        pass
    
    @abstractmethod
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity"""
        pass
    
    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships"""
        pass
    
    @abstractmethod
    async def find_entities(
        self, 
        query: str, 
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Find entities matching the query"""
        pass
    
    # Relation operations
    @abstractmethod
    async def create_relation(self, relation: Relation) -> str:
        """Create a new relation and return its ID"""
        pass
    
    @abstractmethod
    async def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Retrieve a relation by ID"""
        pass
    
    @abstractmethod
    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation"""
        pass
    
    @abstractmethod
    async def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation"""
        pass
    
    @abstractmethod
    async def find_relations(
        self,
        source_entity_id: Optional[str] = None,
        target_entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Relation]:
        """Find relations matching the criteria"""
        pass
    
    # Graph traversal operations
    @abstractmethod
    async def traverse_graph(
        self,
        start_entity_id: str,
        options: TraversalOptions
    ) -> List[GraphPath]:
        """Traverse the graph from a starting entity"""
        pass
    
    @abstractmethod
    async def find_shortest_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 5
    ) -> Optional[GraphPath]:
        """Find the shortest path between two entities"""
        pass
    
    # Bulk operations
    @abstractmethod
    async def bulk_create_entities(self, entities: List[Entity]) -> List[str]:
        """Create multiple entities in a single transaction"""
        pass
    
    @abstractmethod
    async def bulk_create_relations(self, relations: List[Relation]) -> List[str]:
        """Create multiple relations in a single transaction"""
        pass
```

#### Deliverables
- [ ] Complete package structure
- [ ] Core data models (Entity, Relation)
- [ ] Storage interface definitions
- [ ] Basic validation and serialization
- [ ] Package configuration and dependencies

---

### Task 1.2.2: Graph Storage Implementation
**Priority**: Critical  
**Estimated Time**: 4-5 days  
**Dependencies**: 1.2.1, 1.1.2

#### Implementation Steps

1. **Neo4j Storage Implementation**
   - Implement BaseGraphStorage for Neo4j
   - Handle connection management and transactions
   - Optimize queries for performance

2. **CRUD Operations**
   - Entity creation, update, deletion
   - Relation management
   - Bulk operations for performance
   - Transaction handling

3. **Query Optimization**
   - Index usage optimization
   - Query plan analysis
   - Caching strategies

#### Code Examples

**Neo4j Storage Implementation**:
```python
# src/morag_graph/storage/neo4j_storage.py
from typing import List, Optional, Dict, Any
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.types import GraphPath, TraversalOptions
from .base import BaseGraphStorage
from .exceptions import GraphStorageError, EntityNotFoundError, RelationNotFoundError

class Neo4jGraphStorage(BaseGraphStorage):
    """Neo4j implementation of graph storage"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver: Optional[AsyncDriver] = None
    
    async def connect(self) -> None:
        """Establish connection to Neo4j"""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            async with self._driver.session(database=self.database) as session:
                await session.run("RETURN 1")
        except Exception as e:
            raise GraphStorageError(f"Failed to connect to Neo4j: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to Neo4j"""
        if self._driver:
            await self._driver.close()
            self._driver = None
    
    async def create_entity(self, entity: Entity) -> str:
        """Create a new entity in Neo4j"""
        if not self._driver:
            raise GraphStorageError("Not connected to database")
        
        query = """
        CREATE (e:Entity {
            id: $id,
            name: $name,
            type: $type,
            summary: $summary,
            aliases: $aliases,
            embedding: $embedding,
            sparse_vector: $sparse_vector,
            metadata: $metadata,
            source_documents: $source_documents,
            confidence: $confidence,
            created_at: $created_at,
            updated_at: $updated_at
        })
        RETURN e.id
        """
        
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, entity.to_dict())
                record = await result.single()
                return record["e.id"]
        except Exception as e:
            raise GraphStorageError(f"Failed to create entity: {e}")
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID"""
        if not self._driver:
            raise GraphStorageError("Not connected to database")
        
        query = "MATCH (e:Entity {id: $entity_id}) RETURN e"
        
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, {"entity_id": entity_id})
                record = await result.single()
                if record:
                    return Entity.from_dict(dict(record["e"]))
                return None
        except Exception as e:
            raise GraphStorageError(f"Failed to get entity: {e}")
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity"""
        if not self._driver:
            raise GraphStorageError("Not connected to database")
        
        query = """
        MATCH (e:Entity {id: $id})
        SET e += $properties
        RETURN e.id
        """
        
        try:
            entity_dict = entity.to_dict()
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, {
                    "id": entity.id,
                    "properties": entity_dict
                })
                record = await result.single()
                return record is not None
        except Exception as e:
            raise GraphStorageError(f"Failed to update entity: {e}")
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships"""
        if not self._driver:
            raise GraphStorageError("Not connected to database")
        
        query = """
        MATCH (e:Entity {id: $entity_id})
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """
        
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, {"entity_id": entity_id})
                record = await result.single()
                return record["deleted_count"] > 0
        except Exception as e:
            raise GraphStorageError(f"Failed to delete entity: {e}")
    
    async def find_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Find entities matching the query"""
        if not self._driver:
            raise GraphStorageError("Not connected to database")
        
        # Use full-text search if available, otherwise use CONTAINS
        cypher_query = """
        CALL db.index.fulltext.queryNodes('entity_search', $query)
        YIELD node, score
        WHERE ($entity_type IS NULL OR node.type = $entity_type)
        RETURN node
        ORDER BY score DESC
        LIMIT $limit
        """
        
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run(cypher_query, {
                    "query": query,
                    "entity_type": entity_type,
                    "limit": limit
                })
                entities = []
                async for record in result:
                    entities.append(Entity.from_dict(dict(record["node"])))
                return entities
        except Exception as e:
            # Fallback to simple text search
            fallback_query = """
            MATCH (e:Entity)
            WHERE (e.name CONTAINS $query OR e.summary CONTAINS $query)
            AND ($entity_type IS NULL OR e.type = $entity_type)
            RETURN e
            LIMIT $limit
            """
            try:
                async with self._driver.session(database=self.database) as session:
                    result = await session.run(fallback_query, {
                        "query": query,
                        "entity_type": entity_type,
                        "limit": limit
                    })
                    entities = []
                    async for record in result:
                        entities.append(Entity.from_dict(dict(record["e"])))
                    return entities
            except Exception as fallback_error:
                raise GraphStorageError(f"Failed to find entities: {fallback_error}")
    
    async def create_relation(self, relation: Relation) -> str:
        """Create a new relation in Neo4j"""
        if not self._driver:
            raise GraphStorageError("Not connected to database")
        
        query = """
        MATCH (source:Entity {id: $source_entity_id})
        MATCH (target:Entity {id: $target_entity_id})
        CREATE (source)-[r:RELATES_TO {
            id: $id,
            relation_type: $relation_type,
            description: $description,
            confidence: $confidence,
            weight: $weight,
            metadata: $metadata,
            source_documents: $source_documents,
            created_at: $created_at,
            updated_at: $updated_at
        }]->(target)
        RETURN r.id
        """
        
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, relation.to_dict())
                record = await result.single()
                if not record:
                    raise GraphStorageError("Failed to create relation - entities not found")
                return record["r.id"]
        except Exception as e:
            raise GraphStorageError(f"Failed to create relation: {e}")
    
    async def bulk_create_entities(self, entities: List[Entity]) -> List[str]:
        """Create multiple entities in a single transaction"""
        if not self._driver:
            raise GraphStorageError("Not connected to database")
        
        query = """
        UNWIND $entities as entity_data
        CREATE (e:Entity {
            id: entity_data.id,
            name: entity_data.name,
            type: entity_data.type,
            summary: entity_data.summary,
            aliases: entity_data.aliases,
            embedding: entity_data.embedding,
            sparse_vector: entity_data.sparse_vector,
            metadata: entity_data.metadata,
            source_documents: entity_data.source_documents,
            confidence: entity_data.confidence,
            created_at: entity_data.created_at,
            updated_at: entity_data.updated_at
        })
        RETURN e.id
        """
        
        try:
            entity_dicts = [entity.to_dict() for entity in entities]
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, {"entities": entity_dicts})
                return [record["e.id"] async for record in result]
        except Exception as e:
            raise GraphStorageError(f"Failed to bulk create entities: {e}")
```

#### Deliverables
- [ ] Complete Neo4j storage implementation
- [ ] CRUD operations for entities and relations
- [ ] Bulk operations for performance
- [ ] Error handling and transaction management
- [ ] Query optimization and caching

## Testing Requirements

### Unit Tests
```python
# tests/test_models.py
import pytest
from datetime import datetime
from morag_graph.models.entity import Entity
from morag_graph.models.relation import Relation

def test_entity_creation():
    """Test entity creation and validation"""
    entity = Entity(
        name="Test Entity",
        type="CONCEPT",
        summary="A test entity for validation"
    )
    assert entity.name == "Test Entity"
    assert entity.type == "CONCEPT"
    assert entity.confidence == 1.0
    assert isinstance(entity.created_at, datetime)

def test_entity_validation():
    """Test entity validation rules"""
    with pytest.raises(ValueError):
        Entity(name="", type="CONCEPT")  # Empty name
    
    with pytest.raises(ValueError):
        Entity(name="Test", type="")  # Empty type
    
    with pytest.raises(ValueError):
        Entity(name="Test", type="CONCEPT", confidence=1.5)  # Invalid confidence

def test_relation_creation():
    """Test relation creation and validation"""
    relation = Relation(
        source_entity_id="entity1",
        target_entity_id="entity2",
        relation_type="RELATED_TO"
    )
    assert relation.source_entity_id == "entity1"
    assert relation.target_entity_id == "entity2"
    assert relation.relation_type == "RELATED_TO"

def test_relation_validation():
    """Test relation validation rules"""
    with pytest.raises(ValueError):
        Relation(
            source_entity_id="entity1",
            target_entity_id="entity1",  # Same as source
            relation_type="RELATED_TO"
        )
```

### Integration Tests
```python
# tests/test_storage.py
import pytest
import asyncio
from morag_graph.storage.neo4j_storage import Neo4jGraphStorage
from morag_graph.models.entity import Entity
from morag_graph.models.relation import Relation

@pytest.mark.asyncio
async def test_entity_crud_operations():
    """Test complete CRUD operations for entities"""
    storage = Neo4jGraphStorage(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test_password"
    )
    
    try:
        await storage.connect()
        
        # Create entity
        entity = Entity(name="Test Entity", type="CONCEPT")
        entity_id = await storage.create_entity(entity)
        assert entity_id == entity.id
        
        # Read entity
        retrieved_entity = await storage.get_entity(entity_id)
        assert retrieved_entity is not None
        assert retrieved_entity.name == "Test Entity"
        
        # Update entity
        retrieved_entity.summary = "Updated summary"
        success = await storage.update_entity(retrieved_entity)
        assert success
        
        # Delete entity
        success = await storage.delete_entity(entity_id)
        assert success
        
    finally:
        await storage.disconnect()
```

## Success Criteria

- [ ] Package structure created and properly configured
- [ ] Core data models implemented with validation
- [ ] Base storage interface defined
- [ ] Neo4j storage implementation complete
- [ ] All CRUD operations working correctly
- [ ] Bulk operations implemented for performance
- [ ] Unit tests achieving >90% coverage
- [ ] Integration tests passing
- [ ] Documentation complete

## Next Steps

After completing this task:
1. Proceed to [Task 1.3: NLP Pipeline Foundation](./task-1.3-nlp-pipeline-foundation.md)
2. Begin implementing entity and relation extraction
3. Start planning integration with existing document processing pipeline

---

**Status**: ⏳ Not Started  
**Assignee**: TBD  
**Last Updated**: December 2024