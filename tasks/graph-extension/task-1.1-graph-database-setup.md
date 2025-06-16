# Task 1.1: Graph Database Setup

**Phase**: 1 - Foundation Infrastructure  
**Priority**: Critical  
**Total Estimated Time**: 5-8 days  
**Dependencies**: None

## Overview

This task establishes the graph database foundation for the graph-augmented RAG system. It includes database selection, Docker integration, and schema design.

## Subtasks

### Task 1.1.1: Graph Database Selection & Setup
**Priority**: Critical  
**Estimated Time**: 3-5 days  
**Dependencies**: None

#### Implementation Steps

1. **Research & Selection**
   - Evaluate Neo4j vs ArangoDB vs TigerGraph
   - Consider licensing, performance, and Python integration
   - Document decision rationale

2. **Docker Integration**
   - Create `docker-compose.graph.yml`
   - Add graph database service configuration
   - Configure persistent volumes for graph data
   - Set up authentication and security

3. **Connection Library**
   - Install appropriate Python driver (neo4j, python-arango)
   - Create connection configuration in `morag-core`
   - Implement connection pooling and retry logic

#### Code Examples

**Docker Compose Configuration**:
```yaml
# docker-compose.graph.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15-community
    container_name: morag-neo4j
    environment:
      - NEO4J_AUTH=neo4j/morag_password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    networks:
      - morag-network

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:

networks:
  morag-network:
    external: true
```

**Connection Configuration**:
```python
# morag-core/src/morag_core/config/graph_config.py
from pydantic import BaseSettings
from typing import Optional

class GraphDatabaseConfig(BaseSettings):
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "morag_password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60
    
    class Config:
        env_prefix = "GRAPH_DB_"
```

**Connection Manager**:
```python
# morag-core/src/morag_core/services/graph_connection.py
from neo4j import GraphDatabase, Driver
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

class GraphConnectionManager:
    def __init__(self, config: GraphDatabaseConfig):
        self.config = config
        self._driver: Optional[Driver] = None
    
    async def connect(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout
            )
        return self._driver
    
    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None
    
    @asynccontextmanager
    async def session(self, database: Optional[str] = None):
        driver = await self.connect()
        session = driver.session(database=database or self.config.database)
        try:
            yield session
        finally:
            await session.close()
```

#### Deliverables
- [ ] Graph database running in Docker
- [ ] Connection configuration
- [ ] Basic connectivity tests
- [ ] Documentation of database selection rationale

---

### Task 1.1.2: Graph Schema Design
**Priority**: Critical  
**Estimated Time**: 2-3 days  
**Dependencies**: 1.1.1

#### Implementation Steps

1. **Entity Schema**
   - Define entity node structure
   - Create unique constraints and indexes
   - Optimize for query performance

2. **Relationship Schema**
   - Define relationship types and properties
   - Create relationship indexes
   - Design for traversal efficiency

3. **Document Linkage Schema**
   - Link entities to source documents
   - Track provenance and confidence
   - Enable source attribution

#### Code Examples

**Entity Schema (Cypher)**:
```cypher
// Create constraints for entities
CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT entity_name_type FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE;

// Create indexes for performance
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);
CREATE INDEX entity_embedding FOR (e:Entity) ON (e.embedding);
CREATE INDEX entity_created_at FOR (e:Entity) ON (e.created_at);

// Full-text search index
CREATE FULLTEXT INDEX entity_search FOR (e:Entity) ON EACH [e.name, e.summary, e.aliases];
```

**Relationship Schema (Cypher)**:
```cypher
// Create constraints for relationships
CREATE CONSTRAINT rel_id FOR ()-[r:RELATES_TO]-() REQUIRE r.id IS UNIQUE;

// Create indexes for relationships
CREATE INDEX rel_type FOR ()-[r:RELATES_TO]-() ON (r.relation_type);
CREATE INDEX rel_confidence FOR ()-[r:RELATES_TO]-() ON (r.confidence);
CREATE INDEX rel_created_at FOR ()-[r:RELATES_TO]-() ON (r.created_at);
```

**Document Linkage Schema (Cypher)**:
```cypher
// Create constraints for documents
CREATE CONSTRAINT doc_id FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT doc_source FOR (d:Document) REQUIRE d.source_path IS UNIQUE;

// Create indexes for documents
CREATE INDEX doc_source FOR (d:Document) ON (d.source_path);
CREATE INDEX doc_type FOR (d:Document) ON (d.document_type);
CREATE INDEX doc_created_at FOR (d:Document) ON (d.created_at);

// Chunk linkage
CREATE CONSTRAINT chunk_id FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE INDEX chunk_doc_id FOR (c:Chunk) ON (c.document_id);
CREATE INDEX chunk_position FOR (c:Chunk) ON (c.position);
```

**Schema Initialization Script**:
```python
# scripts/init_graph_schema.py
import asyncio
from morag_core.services.graph_connection import GraphConnectionManager
from morag_core.config.graph_config import GraphDatabaseConfig

class GraphSchemaInitializer:
    def __init__(self, connection_manager: GraphConnectionManager):
        self.connection_manager = connection_manager
    
    async def initialize_schema(self):
        """Initialize the complete graph schema"""
        schema_queries = [
            # Entity constraints and indexes
            "CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_name FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type FOR (e:Entity) ON (e.type)",
            "CREATE FULLTEXT INDEX entity_search FOR (e:Entity) ON EACH [e.name, e.summary, e.aliases]",
            
            # Relationship constraints and indexes
            "CREATE CONSTRAINT rel_id FOR ()-[r:RELATES_TO]-() REQUIRE r.id IS UNIQUE",
            "CREATE INDEX rel_type FOR ()-[r:RELATES_TO]-() ON (r.relation_type)",
            "CREATE INDEX rel_confidence FOR ()-[r:RELATES_TO]-() ON (r.confidence)",
            
            # Document constraints and indexes
            "CREATE CONSTRAINT doc_id FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX doc_source FOR (d:Document) ON (d.source_path)",
            "CREATE INDEX doc_type FOR (d:Document) ON (d.document_type)",
            
            # Chunk constraints and indexes
            "CREATE CONSTRAINT chunk_id FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX chunk_doc_id FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX chunk_position FOR (c:Chunk) ON (c.position)"
        ]
        
        async with self.connection_manager.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                    print(f"✓ Executed: {query}")
                except Exception as e:
                    print(f"⚠ Warning for '{query}': {e}")

async def main():
    config = GraphDatabaseConfig()
    connection_manager = GraphConnectionManager(config)
    initializer = GraphSchemaInitializer(connection_manager)
    
    try:
        await initializer.initialize_schema()
        print("Graph schema initialization completed successfully!")
    finally:
        await connection_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Deliverables
- [ ] Graph schema definition
- [ ] Index optimization
- [ ] Schema migration scripts
- [ ] Performance benchmarks for schema queries

## Testing Requirements

### Unit Tests
```python
# tests/test_graph_database_setup.py
import pytest
import asyncio
from morag_core.services.graph_connection import GraphConnectionManager
from morag_core.config.graph_config import GraphDatabaseConfig

@pytest.mark.asyncio
async def test_graph_connection():
    """Test basic graph database connectivity"""
    config = GraphDatabaseConfig()
    manager = GraphConnectionManager(config)
    
    try:
        async with manager.session() as session:
            result = await session.run("RETURN 'Hello, Graph!' as message")
            record = await result.single()
            assert record["message"] == "Hello, Graph!"
    finally:
        await manager.close()

@pytest.mark.asyncio
async def test_schema_constraints():
    """Test that schema constraints are properly created"""
    config = GraphDatabaseConfig()
    manager = GraphConnectionManager(config)
    
    try:
        async with manager.session() as session:
            # Check entity constraint
            result = await session.run(
                "SHOW CONSTRAINTS WHERE name CONTAINS 'entity_id'"
            )
            constraints = [record async for record in result]
            assert len(constraints) > 0
    finally:
        await manager.close()
```

### Integration Tests
```python
# tests/integration/test_graph_database_integration.py
import pytest
import asyncio
from morag_core.services.graph_connection import GraphConnectionManager

@pytest.mark.asyncio
async def test_crud_operations():
    """Test basic CRUD operations on the graph database"""
    # Test entity creation, reading, updating, deletion
    pass

@pytest.mark.asyncio
async def test_performance_benchmarks():
    """Test query performance with indexes"""
    # Benchmark query performance
    pass
```

## Success Criteria

- [ ] Graph database is running and accessible via Docker
- [ ] Connection pooling and retry logic working correctly
- [ ] All schema constraints and indexes created successfully
- [ ] Basic CRUD operations perform within acceptable time limits
- [ ] Unit and integration tests passing
- [ ] Documentation complete and up-to-date

## Next Steps

After completing this task:
1. Proceed to [Task 1.2: Core Graph Package Creation](./task-1.2-core-graph-package.md)
2. Begin implementing the graph data models and storage interfaces
3. Start planning the NLP pipeline integration

---

**Status**: ⏳ Not Started  
**Assignee**: TBD  
**Last Updated**: December 2024