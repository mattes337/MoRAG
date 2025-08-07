# Neo4j Graph Operations

## Entity Operations

### Entity Storage with Deduplication
```cypher
-- MERGE strategy for entity deduplication by normalized name
MERGE (e:Person {normalized_name: $normalized_name})
ON CREATE SET
    e.id = $id,
    e.name = $name,
    e.type = $type,
    e.confidence = $confidence,
    e.metadata = $metadata,
    e.created_at = datetime(),
    e.updated_at = datetime()
ON MATCH SET
    e.name = CASE
        WHEN $confidence > coalesce(e.confidence, 0.0) THEN $name
        ELSE e.name
    END,
    e.type = CASE
        WHEN $confidence > coalesce(e.confidence, 0.0) THEN $type
        ELSE e.type
    END,
    e.confidence = CASE
        WHEN $confidence > coalesce(e.confidence, 0.0) THEN $confidence
        ELSE e.confidence
    END,
    e.metadata = $metadata,
    e.updated_at = datetime()
RETURN e.id as id, e.type as final_type, e.name as final_name
```

### Dynamic Entity Labels
```python
def get_neo4j_label(entity_type):
    """Convert entity type to Neo4j label (title case)."""
    # Remove prefixes and normalize
    clean_type = entity_type.replace("EntityType_", "").replace("_", " ")
    
    # Convert to title case
    return clean_type.title().replace(" ", "")

# Examples:
# "PERSON" -> "Person"
# "ORGANIZATION" -> "Organization" 
# "MEDICAL_CONDITION" -> "MedicalCondition"
```

### Entity Retrieval Patterns
```cypher
-- Get entity by ID
MATCH (e {id: $entity_id})
RETURN e

-- Get entities by type
MATCH (e:Person)
WHERE e.confidence >= $min_confidence
RETURN e
ORDER BY e.confidence DESC

-- Get entities by name pattern
MATCH (e)
WHERE e.name CONTAINS $search_term
RETURN e, labels(e) as entity_labels
ORDER BY e.confidence DESC
```

## Relationship Operations

### Dynamic Relationship Creation
```cypher
-- Create relationship with dynamic type
MATCH (source {id: $source_id}), (target {id: $target_id})
MERGE (source)-[r:PARTNERS_WITH {id: $relation_id}]->(target)
SET r.type = $relation_type,
    r.confidence = $confidence,
    r.metadata = $metadata,
    r.created_at = coalesce(r.created_at, datetime()),
    r.updated_at = datetime()
RETURN r.id as id
```

### Relationship Type Normalization
```python
def normalize_relation_type(relation_type):
    """Normalize relation type to Neo4j format."""
    # Convert to uppercase and replace spaces/hyphens with underscores
    normalized = relation_type.upper().replace(" ", "_").replace("-", "_")
    
    # Remove common prefixes
    normalized = normalized.replace("RELATIONTYPE_", "")
    
    # Ensure singular form for consistency
    singular_mappings = {
        "TREATS": "TREATS",
        "CAUSES": "CAUSES", 
        "PREVENTS": "PREVENTS",
        "PARTNERS_WITH": "PARTNERS_WITH",
        "LOCATED_IN": "LOCATED_IN"
    }
    
    return singular_mappings.get(normalized, normalized)
```

### Relationship Queries
```cypher
-- Get all relationships for an entity
MATCH (e {id: $entity_id})-[r]-(connected)
RETURN e, r, connected, type(r) as relationship_type
ORDER BY r.confidence DESC

-- Get specific relationship types
MATCH (e {id: $entity_id})-[r:TREATS|CAUSES|PREVENTS]-(connected)
RETURN e, r, connected
ORDER BY r.confidence DESC

-- Get relationship paths
MATCH path = (source {id: $source_id})-[r*1..3]-(target {id: $target_id})
RETURN path, length(path) as path_length
ORDER BY path_length
```

## Graph Traversal Patterns

### Neighbor Discovery
```cypher
-- Get direct neighbors
MATCH (start:Entity {id: $entity_id})-[r]-(neighbor:Entity)
WHERE start <> neighbor
RETURN DISTINCT neighbor, r, type(r) as relation_type
ORDER BY neighbor.name

-- Get neighbors within distance
MATCH (start:Entity {id: $entity_id})
MATCH path = (start)-[r*1..3]-(neighbor:Entity)
WHERE start <> neighbor
RETURN DISTINCT neighbor, length(path) as distance
ORDER BY distance, neighbor.name
```

### Path Finding
```cypher
-- Shortest path between entities
MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
MATCH path = shortestPath((source)-[r*1..5]-(target))
RETURN path,
       length(path) as path_length,
       [node in nodes(path) | {id: node.id, name: node.name, type: node.type}] as entities,
       [rel in relationships(path) | {type: rel.type, confidence: rel.confidence}] as relationships
ORDER BY path_length

-- All paths between entities (limited)
MATCH path = (source:Entity {id: $source_id})-[r*1..3]-(target:Entity {id: $target_id})
WHERE source <> target
RETURN [node in nodes(path) | node.id] as path_ids,
       length(path) as path_length
ORDER BY path_length
LIMIT 10
```

### Breadth-First Exploration
```cypher
-- BFS traversal with relationship filtering
MATCH (start:Entity {id: $entity_id})
CALL apoc.path.expandConfig(start, {
    relationshipFilter: "TREATS|CAUSES|PREVENTS",
    labelFilter: "+Entity",
    minLevel: 1,
    maxLevel: 3,
    bfs: true,
    uniqueness: "NODE_GLOBAL"
}) YIELD path
RETURN path, length(path) as depth
ORDER BY depth
```

## Fact Operations

### Fact Storage
```cypher
-- Store fact with relationships to entities
CREATE (f:Fact {
    id: $fact_id,
    content: $content,
    subject: $subject,
    predicate: $predicate,
    object: $object,
    confidence: $confidence,
    fact_type: $fact_type,
    domain: $domain,
    created_at: datetime()
})

-- Link fact to source chunk
MATCH (f:Fact {id: $fact_id})
MATCH (chunk:DocumentChunk {id: $chunk_id})
MERGE (chunk)-[:CONTAINS_FACT]->(f)

-- Link fact to entities
MATCH (f:Fact {id: $fact_id})
MATCH (e:Entity {id: $entity_id})
MERGE (f)-[:MENTIONS]->(e)
```

### Fact Retrieval
```cypher
-- Get facts for entity
MATCH (e:Entity {id: $entity_id})<-[:MENTIONS]-(f:Fact)
RETURN f, e
ORDER BY f.confidence DESC

-- Get facts by domain
MATCH (f:Fact)
WHERE f.domain = $domain AND f.confidence >= $min_confidence
RETURN f
ORDER BY f.confidence DESC

-- Get facts with source attribution
MATCH (chunk:DocumentChunk)-[:CONTAINS_FACT]->(f:Fact)-[:MENTIONS]->(e:Entity)
WHERE e.id = $entity_id
RETURN f, chunk, e
ORDER BY f.confidence DESC
```

## Document and Chunk Operations

### Document Storage
```cypher
-- Store document with metadata
CREATE (d:Document {
    id: $document_id,
    title: $title,
    content_type: $content_type,
    file_path: $file_path,
    metadata: $metadata,
    created_at: datetime()
})

-- Store document chunk
CREATE (chunk:DocumentChunk {
    id: $chunk_id,
    document_id: $document_id,
    chunk_index: $chunk_index,
    text: $text,
    start_position: $start_position,
    end_position: $end_position,
    metadata: $metadata,
    created_at: datetime()
})

-- Link chunk to document
MATCH (d:Document {id: $document_id})
MATCH (chunk:DocumentChunk {document_id: $document_id})
MERGE (d)-[:HAS_CHUNK]->(chunk)
```

### Chunk Retrieval
```cypher
-- Get chunks for document
MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(chunk:DocumentChunk)
RETURN chunk
ORDER BY chunk.chunk_index

-- Get chunks containing entity mentions
MATCH (chunk:DocumentChunk)-[:CONTAINS_FACT]->(f:Fact)-[:MENTIONS]->(e:Entity)
WHERE e.id = $entity_id
RETURN DISTINCT chunk, f, e
ORDER BY chunk.chunk_index
```

## Performance Optimization

### Index Creation
```cypher
-- Entity indexes
CREATE INDEX entity_id_index FOR (e:Entity) ON (e.id);
CREATE INDEX entity_name_index FOR (e:Entity) ON (e.name);
CREATE INDEX entity_normalized_name_index FOR (e:Entity) ON (e.normalized_name);
CREATE INDEX entity_type_index FOR (e:Entity) ON (e.type);

-- Fact indexes
CREATE INDEX fact_id_index FOR (f:Fact) ON (f.id);
CREATE INDEX fact_domain_index FOR (f:Fact) ON (f.domain);
CREATE INDEX fact_confidence_index FOR (f:Fact) ON (f.confidence);

-- Document indexes
CREATE INDEX document_id_index FOR (d:Document) ON (d.id);
CREATE INDEX chunk_id_index FOR (c:DocumentChunk) ON (c.id);
CREATE INDEX chunk_document_id_index FOR (c:DocumentChunk) ON (c.document_id);
```

### Query Optimization
```cypher
-- Use EXPLAIN to analyze query performance
EXPLAIN MATCH (e:Entity {id: $entity_id})-[r]-(connected)
RETURN e, r, connected

-- Use PROFILE for detailed execution statistics
PROFILE MATCH (e:Entity)-[r:TREATS]-(target)
WHERE e.confidence > 0.8
RETURN e, target
ORDER BY e.confidence DESC
LIMIT 10
```

### Batch Operations
```cypher
-- Batch entity creation
UNWIND $entities as entity
MERGE (e:Entity {normalized_name: entity.normalized_name})
ON CREATE SET e += entity, e.created_at = datetime()
ON MATCH SET e += entity, e.updated_at = datetime()

-- Batch relationship creation
UNWIND $relationships as rel
MATCH (source {id: rel.source_id})
MATCH (target {id: rel.target_id})
MERGE (source)-[r:RELATIONSHIP {id: rel.id}]->(target)
SET r += rel.properties
```

## Graph Analytics

### Entity Statistics
```cypher
-- Entity count by type
MATCH (e:Entity)
RETURN labels(e)[0] as entity_type, count(e) as count
ORDER BY count DESC

-- Entity confidence distribution
MATCH (e:Entity)
RETURN 
    CASE 
        WHEN e.confidence >= 0.9 THEN "High (0.9+)"
        WHEN e.confidence >= 0.7 THEN "Medium (0.7-0.9)"
        WHEN e.confidence >= 0.5 THEN "Low (0.5-0.7)"
        ELSE "Very Low (<0.5)"
    END as confidence_range,
    count(e) as count
ORDER BY confidence_range
```

### Relationship Statistics
```cypher
-- Relationship count by type
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC

-- Most connected entities
MATCH (e:Entity)-[r]-()
RETURN e.name, e.type, count(r) as connection_count
ORDER BY connection_count DESC
LIMIT 10
```

### Graph Connectivity
```cypher
-- Connected components
CALL gds.wcc.stream('entity-graph')
YIELD nodeId, componentId
RETURN componentId, count(nodeId) as component_size
ORDER BY component_size DESC

-- PageRank for entity importance
CALL gds.pageRank.stream('entity-graph')
YIELD nodeId, score
MATCH (e:Entity) WHERE id(e) = nodeId
RETURN e.name, e.type, score
ORDER BY score DESC
LIMIT 10
```

## Error Handling

### Constraint Violations
```cypher
-- Handle duplicate entity creation
MERGE (e:Entity {id: $entity_id})
ON CREATE SET e += $properties
ON MATCH SET e += $properties
RETURN e

-- Handle missing entities in relationships
MATCH (source {id: $source_id}), (target {id: $target_id})
WITH source, target
WHERE source IS NOT NULL AND target IS NOT NULL
MERGE (source)-[r:RELATIONSHIP]->(target)
RETURN r
```

### Transaction Management
```python
async def execute_graph_transaction(operations):
    async with neo4j_driver.session() as session:
        async with session.begin_transaction() as tx:
            try:
                results = []
                for operation in operations:
                    result = await tx.run(operation.query, operation.parameters)
                    results.append(result)
                
                await tx.commit()
                return results
            
            except Exception as e:
                await tx.rollback()
                logger.error(f"Transaction failed: {e}")
                raise
```
