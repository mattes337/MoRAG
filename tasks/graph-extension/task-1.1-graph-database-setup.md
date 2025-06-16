# Task 1.1: Neo4J Database Setup

**Phase**: 1 - Foundation Infrastructure  
**Priority**: Critical  
**Total Estimated Time**: 5-7 days  
**Dependencies**: None

## Overview

This task establishes the Neo4J graph database foundation for the graph-augmented RAG system. Neo4J has been selected as the graph database for its mature Python ecosystem, excellent performance, and robust Cypher query language.

## Subtasks

### Task 1.1.1: Neo4J Database Setup
**Priority**: Critical  
**Estimated Time**: 2-3 days  
**Dependencies**: None

#### Implementation Steps

1. **Neo4J Decision Rationale**
   - Neo4J Community Edition provides excellent performance for our use case
   - Mature Python driver (neo4j>=5.15.0) with async support
   - Rich Cypher query language for complex graph operations
   - Strong ecosystem and documentation
   - APOC plugin support for advanced graph algorithms

2. **Docker Integration**
   - Create `docker-compose.graph.yml`
   - Add graph database service configuration
   - Configure persistent volumes for graph data
   - Set up authentication and security

3. **Neo4J Python Driver Setup**
   - Install neo4j>=5.15.0 Python driver
   - Create connection configuration in `morag-core`
   - Implement connection pooling and retry logic
   - Configure async session management

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
- [ ] Neo4J database running in Docker with APOC plugin
- [ ] Neo4J connection configuration and pooling
- [ ] Basic connectivity and health check tests
- [ ] Neo4J-specific configuration documentation

---

### Task 1.1.2: Neo4J Dynamic Schema with LLM-Based Relation Discovery
**Priority**: Critical  
**Estimated Time**: 3-4 days  
**Dependencies**: 1.1.1

#### Implementation Steps

1. **Minimal Base Schema**
   - Define minimal entity and relationship structure
   - Create flexible schema that supports dynamic relation types
   - Optimize for extensibility and performance

2. **LLM-Based Relation Discovery**
   - Implement LLM service for dynamic relation extraction
   - Create relation type registry for discovered relationships
   - Design confidence scoring and validation system

3. **Dynamic Schema Evolution**
   - Implement schema migration for new relation types
   - Create relation type management system
   - Enable real-time schema updates

4. **Document Linkage Schema**
   - Link entities to source documents
   - Track provenance and confidence
   - Enable source attribution

#### Code Examples

**Minimal Base Schema (Cypher)**:
```cypher
// Create constraints for entities
CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT entity_name FOR (e:Entity) REQUIRE e.name IS UNIQUE;

// Create indexes for performance
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);
CREATE INDEX entity_embedding FOR (e:Entity) ON (e.embedding);
CREATE INDEX entity_created_at FOR (e:Entity) ON (e.created_at);
CREATE INDEX entity_confidence FOR (e:Entity) ON (e.confidence);

// Full-text search index
CREATE FULLTEXT INDEX entity_search FOR (e:Entity) ON EACH [e.name, e.summary, e.aliases];

// Dynamic relationship constraints (flexible schema)
CREATE CONSTRAINT rel_id FOR ()-[r]-() REQUIRE r.id IS UNIQUE;

// Indexes for dynamic relationships
CREATE INDEX rel_type FOR ()-[r]-() ON (r.relation_type);
CREATE INDEX rel_confidence FOR ()-[r]-() ON (r.confidence);
CREATE INDEX rel_created_at FOR ()-[r]-() ON (r.created_at);
CREATE INDEX rel_source_chunk FOR ()-[r]-() ON (r.source_chunk_id);

// Relation type registry
CREATE CONSTRAINT relation_type_id FOR (rt:RelationType) REQUIRE rt.id IS UNIQUE;
CREATE CONSTRAINT relation_type_name FOR (rt:RelationType) REQUIRE rt.name IS UNIQUE;
CREATE INDEX relation_type_domain FOR (rt:RelationType) ON (rt.domain);
CREATE INDEX relation_type_created FOR (rt:RelationType) ON (rt.created_at);
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

**Dynamic Schema Initialization Script**:
```python
# scripts/init_graph_schema.py
import asyncio
from typing import List, Dict, Any
from morag_core.services.graph_connection import GraphConnectionManager
from morag_core.config.graph_config import GraphDatabaseConfig

class DynamicGraphSchemaInitializer:
    def __init__(self, connection_manager: GraphConnectionManager):
        self.connection_manager = connection_manager
    
    async def initialize_schema(self):
        """Initialize the dynamic graph schema with minimal base structure"""
        schema_queries = [
            # Entity constraints and indexes
            "CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX entity_type FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_embedding FOR (e:Entity) ON (e.embedding)",
            "CREATE INDEX entity_created_at FOR (e:Entity) ON (e.created_at)",
            "CREATE INDEX entity_confidence FOR (e:Entity) ON (e.confidence)",
            "CREATE FULLTEXT INDEX entity_search FOR (e:Entity) ON EACH [e.name, e.summary, e.aliases]",
            
            # Dynamic relationship constraints (flexible schema)
            "CREATE CONSTRAINT rel_id FOR ()-[r]-() REQUIRE r.id IS UNIQUE",
            "CREATE INDEX rel_type FOR ()-[r]-() ON (r.relation_type)",
            "CREATE INDEX rel_confidence FOR ()-[r]-() ON (r.confidence)",
            "CREATE INDEX rel_created_at FOR ()-[r]-() ON (r.created_at)",
            "CREATE INDEX rel_source_chunk FOR ()-[r]-() ON (r.source_chunk_id)",
            
            # Relation type registry
            "CREATE CONSTRAINT relation_type_id FOR (rt:RelationType) REQUIRE rt.id IS UNIQUE",
            "CREATE CONSTRAINT relation_type_name FOR (rt:RelationType) REQUIRE rt.name IS UNIQUE",
            "CREATE INDEX relation_type_domain FOR (rt:RelationType) ON (rt.domain)",
            "CREATE INDEX relation_type_created FOR (rt:RelationType) ON (rt.created_at)",
            
            # Document constraints and indexes
            "CREATE CONSTRAINT doc_id FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT doc_source FOR (d:Document) REQUIRE d.source_path IS UNIQUE",
            "CREATE INDEX doc_source FOR (d:Document) ON (d.source_path)",
            "CREATE INDEX doc_type FOR (d:Document) ON (d.document_type)",
            "CREATE INDEX doc_created_at FOR (d:Document) ON (d.created_at)",
            
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
    
    async def register_relation_type(self, relation_name: str, description: str, 
                                   domain: str = "general", 
                                   properties: Dict[str, Any] = None) -> str:
        """Register a new relation type discovered by LLM"""
        properties = properties or {}
        
        query = """
        MERGE (rt:RelationType {name: $name})
        ON CREATE SET 
            rt.id = randomUUID(),
            rt.description = $description,
            rt.domain = $domain,
            rt.properties = $properties,
            rt.created_at = datetime(),
            rt.usage_count = 0
        ON MATCH SET 
            rt.usage_count = rt.usage_count + 1,
            rt.last_used = datetime()
        RETURN rt.id as relation_type_id
        """
        
        async with self.connection_manager.session() as session:
            result = await session.run(query, {
                "name": relation_name,
                "description": description,
                "domain": domain,
                "properties": properties
            })
            record = await result.single()
            return record["relation_type_id"]
    
    async def get_existing_relation_types(self) -> List[Dict[str, Any]]:
        """Get all existing relation types for LLM context"""
        query = """
        MATCH (rt:RelationType)
        RETURN rt.name as name, rt.description as description, 
               rt.domain as domain, rt.usage_count as usage_count
        ORDER BY rt.usage_count DESC, rt.created_at ASC
        """
        
        async with self.connection_manager.session() as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

async def main():
    config = GraphDatabaseConfig()
    connection_manager = GraphConnectionManager(config)
    initializer = DynamicGraphSchemaInitializer(connection_manager)
    
    try:
        await initializer.initialize_schema()
        print("Dynamic graph schema initialization completed successfully!")
        
        # Initialize with some common relation types
        common_relations = [
            ("RELATED_TO", "General relationship between entities", "general"),
            ("PART_OF", "Entity is part of another entity", "structural"),
            ("LOCATED_IN", "Entity is located within another entity", "spatial"),
            ("CAUSED_BY", "Entity is caused by another entity", "causal"),
            ("ENABLES", "Entity enables or facilitates another entity", "functional")
        ]
        
        for name, desc, domain in common_relations:
            await initializer.register_relation_type(name, desc, domain)
            print(f"✓ Registered relation type: {name}")
            
    finally:
        await connection_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Task 1.1.3: LLM-Based Relation Discovery Service
**Priority**: Critical  
**Estimated Time**: 2-3 days  
**Dependencies**: 1.1.2

#### Implementation Steps

1. **LLM Integration**
   - Create LLM service client
   - Design relation extraction prompts
   - Implement response parsing

2. **Relation Type Management**
   - Create relation type registry service
   - Implement dynamic relation creation
   - Design relation validation workflow

3. **Schema Evolution**
   - Implement schema update mechanisms
   - Create relation type migration tools
   - Design caching for performance

#### Code Examples

**LLM Relation Discovery Service**:
```python
# morag-graph/src/morag_graph/services/llm_relation_discovery.py
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
import uuid
from datetime import datetime

from morag_core.services.llm_client import LLMClient
from morag_core.config.llm_config import LLMConfig
from morag_graph.storage.relation_registry import RelationTypeRegistry

class LLMRelationDiscoveryService:
    """Service for discovering relations between entities using LLM"""
    
    def __init__(self, llm_client: LLMClient, relation_registry: RelationTypeRegistry):
        self.llm_client = llm_client
        self.relation_registry = relation_registry
        
    async def discover_relations(self, text_chunk: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover relations between entities in a text chunk using LLM"""
        if not entities or len(entities) < 2:
            return []
            
        # Get existing relation types for context
        existing_relations = await self.relation_registry.get_all_relation_types()
        
        # Prepare prompt for LLM
        prompt = self._create_relation_discovery_prompt(text_chunk, entities, existing_relations)
        
        # Call LLM
        response = await self.llm_client.generate_text(prompt)
        
        # Parse relations from LLM response
        discovered_relations = self._parse_relations_from_llm(response, entities)
        
        # Register any new relation types
        await self._register_new_relation_types(discovered_relations)
        
        return discovered_relations
    
    def _create_relation_discovery_prompt(self, text_chunk: str, 
                                         entities: List[Dict[str, Any]],
                                         existing_relations: List[Dict[str, Any]]) -> str:
        """Create prompt for relation discovery"""
        entity_names = [e["name"] for e in entities]
        relation_types = [r["name"] for r in existing_relations]
        relation_descriptions = {r["name"]: r["description"] for r in existing_relations}
        
        prompt = f"""Analyze the following text and identify relationships between the entities.

Text: "{text_chunk}"

Entities: {', '.join(entity_names)}

Existing relationship types:
{json.dumps(relation_descriptions, indent=2)}

Instructions:
1. Identify all relationships between the entities in the text.
2. For each relationship, use an existing relationship type if appropriate.
3. If no existing relationship type fits, create a new one with a clear name and description.
4. Format your response as a JSON array of relationships with the following structure:
   [{{"source": "EntityName1", "target": "EntityName2", "relation_type": "RELATION_NAME", "description": "Description of this specific relationship", "is_new_relation": false, "new_relation_description": null}}]
5. For new relation types, set is_new_relation to true and provide a description in new_relation_description.

Response:"""
        
        return prompt
    
    def _parse_relations_from_llm(self, llm_response: str, 
                                entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse relations from LLM response"""
        try:
            # Extract JSON from response (handle potential text before/after JSON)
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1
            if json_start == -1 or json_end == 0:
                return []
                
            json_str = llm_response[json_start:json_end]
            relations_data = json.loads(json_str)
            
            # Validate and format relations
            entity_map = {e["name"]: e["id"] for e in entities}
            validated_relations = []
            
            for rel in relations_data:
                if rel["source"] in entity_map and rel["target"] in entity_map:
                    validated_relations.append({
                        "id": str(uuid.uuid4()),
                        "source_id": entity_map[rel["source"]],
                        "source_name": rel["source"],
                        "target_id": entity_map[rel["target"]],
                        "target_name": rel["target"],
                        "relation_type": rel["relation_type"],
                        "description": rel["description"],
                        "is_new_relation": rel.get("is_new_relation", False),
                        "new_relation_description": rel.get("new_relation_description"),
                        "confidence": 0.85,  # Default confidence for LLM-discovered relations
                        "created_at": datetime.now().isoformat()
                    })
            
            return validated_relations
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []
    
    async def _register_new_relation_types(self, relations: List[Dict[str, Any]]):
        """Register any new relation types discovered by LLM"""
        for relation in relations:
            if relation.get("is_new_relation", False) and relation.get("new_relation_description"):
                await self.relation_registry.register_relation_type(
                    relation["relation_type"],
                    relation["new_relation_description"],
                    "llm_discovered"
                )
```

**Relation Type Registry Service**:
```python
# morag-graph/src/morag_graph/storage/relation_registry.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from morag_core.services.graph_connection import GraphConnectionManager

class RelationTypeRegistry:
    """Service for managing relation types in the graph database"""
    
    def __init__(self, connection_manager: GraphConnectionManager):
        self.connection_manager = connection_manager
        self._cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes cache TTL
    
    async def register_relation_type(self, name: str, description: str, 
                                   domain: str = "general", 
                                   properties: Dict[str, Any] = None) -> str:
        """Register a new relation type or update an existing one"""
        properties = properties or {}
        
        query = """
        MERGE (rt:RelationType {name: $name})
        ON CREATE SET 
            rt.id = randomUUID(),
            rt.description = $description,
            rt.domain = $domain,
            rt.properties = $properties,
            rt.created_at = datetime(),
            rt.usage_count = 0,
            rt.source = 'llm'
        ON MATCH SET 
            rt.usage_count = rt.usage_count + 1,
            rt.last_used = datetime()
        RETURN rt.id as relation_type_id
        """
        
        async with self.connection_manager.session() as session:
            result = await session.run(query, {
                "name": name,
                "description": description,
                "domain": domain,
                "properties": properties
            })
            record = await result.single()
            
            # Invalidate cache
            self._cache = {}
            self._cache_timestamp = None
            
            return record["relation_type_id"]
    
    async def get_all_relation_types(self) -> List[Dict[str, Any]]:
        """Get all registered relation types"""
        # Check cache first
        now = datetime.now().timestamp()
        if self._cache and self._cache_timestamp and (now - self._cache_timestamp < self._cache_ttl):
            return list(self._cache.values())
        
        query = """
        MATCH (rt:RelationType)
        RETURN rt.id as id, rt.name as name, rt.description as description, 
               rt.domain as domain, rt.usage_count as usage_count,
               rt.created_at as created_at, rt.source as source
        ORDER BY rt.usage_count DESC, rt.created_at ASC
        """
        
        async with self.connection_manager.session() as session:
            result = await session.run(query)
            relations = [dict(record) async for record in result]
            
            # Update cache
            self._cache = {r["id"]: r for r in relations}
            self._cache_timestamp = now
            
            return relations
    
    async def get_relation_type_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a relation type by name"""
        # Try cache first
        if self._cache:
            for rel in self._cache.values():
                if rel["name"] == name:
                    return rel
        
        query = """
        MATCH (rt:RelationType {name: $name})
        RETURN rt.id as id, rt.name as name, rt.description as description, 
               rt.domain as domain, rt.usage_count as usage_count,
               rt.created_at as created_at, rt.source as source
        """
        
        async with self.connection_manager.session() as session:
            result = await session.run(query, {"name": name})
            record = await result.single()
            return dict(record) if record else None
```

#### Deliverables
- [ ] Dynamic graph schema implementation
- [ ] LLM relation discovery service
- [ ] Relation type registry service
- [ ] Schema evolution mechanisms
- [ ] Performance benchmarks for dynamic schema

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