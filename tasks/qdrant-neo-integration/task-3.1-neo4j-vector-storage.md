# Task 3.1: Neo4j Vector Storage

## Overview

This task focuses on implementing selective vector storage in Neo4j to complement the primary vector storage in Qdrant. The goal is to store specific types of embeddings directly in Neo4j nodes for enhanced graph-based similarity operations and hybrid queries.

## Objectives

- Add vector embedding fields to Neo4j Entity and Relation nodes
- Implement vector similarity operations in Neo4j
- Create vector indexing for efficient similarity searches
- Establish synchronization between Neo4j and Qdrant vectors
- Enable graph-aware vector operations

## Implementation Plan

### 1. Neo4j Schema Extensions

#### Entity Vector Storage

```cypher
-- Add vector embedding field to Entity nodes
MATCH (e:Entity)
SET e.embedding_vector = null,
    e.embedding_model = null,
    e.embedding_created_at = null,
    e.embedding_dimensions = null

-- Create vector index for Entity embeddings
CREATE VECTOR INDEX entity_embedding_index
FOR (e:Entity)
ON (e.embedding_vector)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}

-- Create text index for entity names (for hybrid search)
CREATE FULLTEXT INDEX entity_name_index
FOR (e:Entity)
ON EACH [e.name, e.description]
```

#### Relation Vector Storage

```cypher
-- Add vector embedding field to Relation nodes
MATCH ()-[r:RELATION]->()
SET r.embedding_vector = null,
    r.context_embedding = null,
    r.embedding_model = null,
    r.embedding_created_at = null

-- Create vector index for Relation embeddings
CREATE VECTOR INDEX relation_embedding_index
FOR ()-[r:RELATION]-()
ON (r.embedding_vector)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}

-- Create vector index for context embeddings
CREATE VECTOR INDEX relation_context_index
FOR ()-[r:RELATION]-()
ON (r.context_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

#### Document-Level Vector Storage

```cypher
-- Add document-level embedding for clustering
MATCH (d:Document)
SET d.summary_embedding = null,
    d.topic_embedding = null,
    d.embedding_model = null,
    d.embedding_created_at = null

-- Create vector index for document embeddings
CREATE VECTOR INDEX document_embedding_index
FOR (d:Document)
ON (d.summary_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

### 2. Vector Storage Service

```python
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class VectorMetadata:
    """Metadata for stored vectors"""
    model_name: str
    dimensions: int
    created_at: datetime
    similarity_function: str = "cosine"
    vector_type: str = "dense"  # dense, sparse, hybrid

class Neo4jVectorStorage:
    """Service for managing vector storage in Neo4j"""
    
    def __init__(self, neo4j_storage, embedding_service):
        self.neo4j_storage = neo4j_storage
        self.embedding_service = embedding_service
        self.logger = logging.getLogger(__name__)
        
        # Vector configuration
        self.default_dimensions = 384
        self.default_model = "text-embedding-004"
        self.similarity_function = "cosine"
    
    async def store_entity_embedding(
        self, 
        entity_id: str, 
        text: str, 
        metadata: Optional[VectorMetadata] = None
    ) -> bool:
        """Store embedding vector for an entity"""
        try:
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(text)
            
            if metadata is None:
                metadata = VectorMetadata(
                    model_name=self.default_model,
                    dimensions=len(embedding),
                    created_at=datetime.utcnow()
                )
            
            # Store in Neo4j
            query = """
            MATCH (e:Entity {id: $entity_id})
            SET e.embedding_vector = $embedding,
                e.embedding_model = $model_name,
                e.embedding_dimensions = $dimensions,
                e.embedding_created_at = $created_at
            RETURN e.id as entity_id
            """
            
            result = await self.neo4j_storage.execute_query(query, {
                "entity_id": entity_id,
                "embedding": embedding,
                "model_name": metadata.model_name,
                "dimensions": metadata.dimensions,
                "created_at": metadata.created_at.isoformat()
            })
            
            if result and len(result) > 0:
                self.logger.info(f"Stored embedding for entity {entity_id}")
                return True
            else:
                self.logger.warning(f"Entity {entity_id} not found for embedding storage")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing entity embedding for {entity_id}: {e}")
            return False
    
    async def store_relation_embedding(
        self, 
        relation_id: str, 
        relation_text: str,
        context_text: Optional[str] = None,
        metadata: Optional[VectorMetadata] = None
    ) -> bool:
        """Store embedding vectors for a relation"""
        try:
            # Generate relation embedding
            relation_embedding = await self.embedding_service.generate_embedding(relation_text)
            
            # Generate context embedding if provided
            context_embedding = None
            if context_text:
                context_embedding = await self.embedding_service.generate_embedding(context_text)
            
            if metadata is None:
                metadata = VectorMetadata(
                    model_name=self.default_model,
                    dimensions=len(relation_embedding),
                    created_at=datetime.utcnow()
                )
            
            # Store in Neo4j
            query = """
            MATCH ()-[r:RELATION {id: $relation_id}]->()
            SET r.embedding_vector = $relation_embedding,
                r.context_embedding = $context_embedding,
                r.embedding_model = $model_name,
                r.embedding_created_at = $created_at
            RETURN r.id as relation_id
            """
            
            result = await self.neo4j_storage.execute_query(query, {
                "relation_id": relation_id,
                "relation_embedding": relation_embedding,
                "context_embedding": context_embedding,
                "model_name": metadata.model_name,
                "created_at": metadata.created_at.isoformat()
            })
            
            if result and len(result) > 0:
                self.logger.info(f"Stored embedding for relation {relation_id}")
                return True
            else:
                self.logger.warning(f"Relation {relation_id} not found for embedding storage")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing relation embedding for {relation_id}: {e}")
            return False
    
    async def store_document_embedding(
        self, 
        document_id: str, 
        summary_text: str,
        topic_text: Optional[str] = None,
        metadata: Optional[VectorMetadata] = None
    ) -> bool:
        """Store document-level embeddings"""
        try:
            # Generate summary embedding
            summary_embedding = await self.embedding_service.generate_embedding(summary_text)
            
            # Generate topic embedding if provided
            topic_embedding = None
            if topic_text:
                topic_embedding = await self.embedding_service.generate_embedding(topic_text)
            
            if metadata is None:
                metadata = VectorMetadata(
                    model_name=self.default_model,
                    dimensions=len(summary_embedding),
                    created_at=datetime.utcnow()
                )
            
            # Store in Neo4j
            query = """
            MATCH (d:Document {id: $document_id})
            SET d.summary_embedding = $summary_embedding,
                d.topic_embedding = $topic_embedding,
                d.embedding_model = $model_name,
                d.embedding_created_at = $created_at
            RETURN d.id as document_id
            """
            
            result = await self.neo4j_storage.execute_query(query, {
                "document_id": document_id,
                "summary_embedding": summary_embedding,
                "topic_embedding": topic_embedding,
                "model_name": metadata.model_name,
                "created_at": metadata.created_at.isoformat()
            })
            
            if result and len(result) > 0:
                self.logger.info(f"Stored embedding for document {document_id}")
                return True
            else:
                self.logger.warning(f"Document {document_id} not found for embedding storage")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing document embedding for {document_id}: {e}")
            return False
    
    async def find_similar_entities(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Find entities similar to query embedding"""
        try:
            # Build type filter
            type_filter = ""
            params = {
                "query_embedding": query_embedding,
                "limit": limit,
                "threshold": similarity_threshold
            }
            
            if entity_types:
                type_filter = "AND e.type IN $entity_types"
                params["entity_types"] = entity_types
            
            # Vector similarity search
            query = f"""
            CALL db.index.vector.queryNodes('entity_embedding_index', $limit, $query_embedding)
            YIELD node as e, score
            WHERE score >= $threshold {type_filter}
            RETURN e.id as entity_id, e.name as name, e.type as type, 
                   score, e.embedding_created_at as embedding_date
            ORDER BY score DESC
            """
            
            results = await self.neo4j_storage.execute_query(query, params)
            
            return [
                {
                    "entity_id": result["entity_id"],
                    "name": result["name"],
                    "type": result["type"],
                    "similarity_score": result["score"],
                    "embedding_date": result["embedding_date"]
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error finding similar entities: {e}")
            return []
    
    async def find_similar_relations(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        use_context: bool = False,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Find relations similar to query embedding"""
        try:
            # Choose embedding field
            embedding_field = "context_embedding" if use_context else "embedding_vector"
            index_name = "relation_context_index" if use_context else "relation_embedding_index"
            
            # Build type filter
            type_filter = ""
            params = {
                "query_embedding": query_embedding,
                "limit": limit
            }
            
            if relation_types:
                type_filter = "AND type(r) IN $relation_types"
                params["relation_types"] = relation_types
            
            # Vector similarity search
            query = f"""
            CALL db.index.vector.queryRelationships('{index_name}', $limit, $query_embedding)
            YIELD relationship as r, score
            WHERE r.{embedding_field} IS NOT NULL {type_filter}
            MATCH (source)-[r]->(target)
            RETURN r.id as relation_id, type(r) as relation_type,
                   source.id as source_id, source.name as source_name,
                   target.id as target_id, target.name as target_name,
                   score, r.embedding_created_at as embedding_date
            ORDER BY score DESC
            """
            
            results = await self.neo4j_storage.execute_query(query, params)
            
            return [
                {
                    "relation_id": result["relation_id"],
                    "relation_type": result["relation_type"],
                    "source": {
                        "id": result["source_id"],
                        "name": result["source_name"]
                    },
                    "target": {
                        "id": result["target_id"],
                        "name": result["target_name"]
                    },
                    "similarity_score": result["score"],
                    "embedding_date": result["embedding_date"]
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error finding similar relations: {e}")
            return []
    
    async def find_similar_documents(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        use_topic: bool = False
    ) -> List[Dict]:
        """Find documents similar to query embedding"""
        try:
            # Choose embedding field
            embedding_field = "topic_embedding" if use_topic else "summary_embedding"
            
            # Vector similarity search
            query = f"""
            CALL db.index.vector.queryNodes('document_embedding_index', $limit, $query_embedding)
            YIELD node as d, score
            WHERE d.{embedding_field} IS NOT NULL
            RETURN d.id as document_id, d.source_file as source_file,
                   d.title as title, score, d.embedding_created_at as embedding_date
            ORDER BY score DESC
            """
            
            params = {
                "query_embedding": query_embedding,
                "limit": limit
            }
            
            results = await self.neo4j_storage.execute_query(query, params)
            
            return [
                {
                    "document_id": result["document_id"],
                    "source_file": result["source_file"],
                    "title": result["title"],
                    "similarity_score": result["score"],
                    "embedding_date": result["embedding_date"]
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error finding similar documents: {e}")
            return []
    
    async def get_entity_neighbors_by_similarity(
        self, 
        entity_id: str, 
        similarity_threshold: float = 0.8,
        max_hops: int = 2
    ) -> List[Dict]:
        """Find entity neighbors using both graph structure and vector similarity"""
        try:
            # First get the entity's embedding
            entity_query = """
            MATCH (e:Entity {id: $entity_id})
            RETURN e.embedding_vector as embedding
            """
            
            entity_result = await self.neo4j_storage.execute_query(
                entity_query, {"entity_id": entity_id}
            )
            
            if not entity_result or not entity_result[0].get("embedding"):
                self.logger.warning(f"No embedding found for entity {entity_id}")
                return []
            
            query_embedding = entity_result[0]["embedding"]
            
            # Find similar entities within graph neighborhood
            neighbor_query = f"""
            MATCH (source:Entity {{id: $entity_id}})-[*1..{max_hops}]-(neighbor:Entity)
            WHERE neighbor.embedding_vector IS NOT NULL
            WITH neighbor, 
                 gds.similarity.cosine(source.embedding_vector, neighbor.embedding_vector) as similarity
            WHERE similarity >= $threshold
            RETURN neighbor.id as neighbor_id, neighbor.name as name, 
                   neighbor.type as type, similarity
            ORDER BY similarity DESC
            LIMIT 20
            """
            
            results = await self.neo4j_storage.execute_query(neighbor_query, {
                "entity_id": entity_id,
                "threshold": similarity_threshold
            })
            
            return [
                {
                    "entity_id": result["neighbor_id"],
                    "name": result["name"],
                    "type": result["type"],
                    "similarity_score": result["similarity"]
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error finding similar neighbors for {entity_id}: {e}")
            return []
    
    async def batch_store_entity_embeddings(
        self, 
        entity_embeddings: List[Tuple[str, List[float]]],
        metadata: Optional[VectorMetadata] = None
    ) -> Dict[str, bool]:
        """Store multiple entity embeddings in batch"""
        if metadata is None:
            metadata = VectorMetadata(
                model_name=self.default_model,
                dimensions=self.default_dimensions,
                created_at=datetime.utcnow()
            )
        
        results = {}
        batch_size = 100
        
        for i in range(0, len(entity_embeddings), batch_size):
            batch = entity_embeddings[i:i + batch_size]
            
            # Prepare batch data
            batch_data = [
                {
                    "entity_id": entity_id,
                    "embedding": embedding,
                    "model_name": metadata.model_name,
                    "dimensions": metadata.dimensions,
                    "created_at": metadata.created_at.isoformat()
                }
                for entity_id, embedding in batch
            ]
            
            # Batch update query
            query = """
            UNWIND $batch as item
            MATCH (e:Entity {id: item.entity_id})
            SET e.embedding_vector = item.embedding,
                e.embedding_model = item.model_name,
                e.embedding_dimensions = item.dimensions,
                e.embedding_created_at = item.created_at
            RETURN e.id as entity_id
            """
            
            try:
                batch_results = await self.neo4j_storage.execute_query(
                    query, {"batch": batch_data}
                )
                
                # Track success for each entity in batch
                successful_ids = {result["entity_id"] for result in batch_results}
                
                for entity_id, _ in batch:
                    results[entity_id] = entity_id in successful_ids
                    
                self.logger.info(f"Processed batch of {len(batch)} entity embeddings")
                
            except Exception as e:
                self.logger.error(f"Error in batch embedding storage: {e}")
                for entity_id, _ in batch:
                    results[entity_id] = False
        
        return results
    
    async def get_vector_statistics(self) -> Dict:
        """Get statistics about stored vectors"""
        try:
            stats_query = """
            MATCH (e:Entity)
            WITH count(e) as total_entities,
                 count(e.embedding_vector) as entities_with_embeddings
            
            MATCH ()-[r:RELATION]-()
            WITH total_entities, entities_with_embeddings,
                 count(r) as total_relations,
                 count(r.embedding_vector) as relations_with_embeddings
            
            MATCH (d:Document)
            WITH total_entities, entities_with_embeddings,
                 total_relations, relations_with_embeddings,
                 count(d) as total_documents,
                 count(d.summary_embedding) as documents_with_embeddings
            
            RETURN total_entities, entities_with_embeddings,
                   total_relations, relations_with_embeddings,
                   total_documents, documents_with_embeddings
            """
            
            result = await self.neo4j_storage.execute_query(stats_query)
            
            if result and len(result) > 0:
                stats = result[0]
                return {
                    "entities": {
                        "total": stats["total_entities"],
                        "with_embeddings": stats["entities_with_embeddings"],
                        "coverage_percentage": (
                            stats["entities_with_embeddings"] / stats["total_entities"] * 100
                            if stats["total_entities"] > 0 else 0
                        )
                    },
                    "relations": {
                        "total": stats["total_relations"],
                        "with_embeddings": stats["relations_with_embeddings"],
                        "coverage_percentage": (
                            stats["relations_with_embeddings"] / stats["total_relations"] * 100
                            if stats["total_relations"] > 0 else 0
                        )
                    },
                    "documents": {
                        "total": stats["total_documents"],
                        "with_embeddings": stats["documents_with_embeddings"],
                        "coverage_percentage": (
                            stats["documents_with_embeddings"] / stats["total_documents"] * 100
                            if stats["total_documents"] > 0 else 0
                        )
                    }
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting vector statistics: {e}")
            return {}
```

### 3. Vector Synchronization Service

```python
class VectorSynchronizationService:
    """Service for synchronizing vectors between Neo4j and Qdrant"""
    
    def __init__(self, neo4j_vector_storage, qdrant_storage, id_mapping_service):
        self.neo4j_vectors = neo4j_vector_storage
        self.qdrant_storage = qdrant_storage
        self.id_mapping = id_mapping_service
        self.logger = logging.getLogger(__name__)
    
    async def sync_entity_embeddings_to_neo4j(self, entity_ids: Optional[List[str]] = None) -> Dict:
        """Sync entity embeddings from Qdrant metadata to Neo4j"""
        try:
            # Get entities that need embeddings
            if entity_ids is None:
                # Get all entities without embeddings
                query = """
                MATCH (e:Entity)
                WHERE e.embedding_vector IS NULL
                RETURN e.id as entity_id, e.name as name, e.type as type
                LIMIT 1000
                """
                
                result = await self.neo4j_vectors.neo4j_storage.execute_query(query)
                entity_ids = [r["entity_id"] for r in result]
            
            # Generate embeddings for entities
            embeddings_to_store = []
            
            for entity_id in entity_ids:
                # Get entity details
                entity_query = """
                MATCH (e:Entity {id: $entity_id})
                RETURN e.name as name, e.type as type, e.description as description
                """
                
                entity_result = await self.neo4j_vectors.neo4j_storage.execute_query(
                    entity_query, {"entity_id": entity_id}
                )
                
                if entity_result and len(entity_result) > 0:
                    entity_data = entity_result[0]
                    
                    # Create text for embedding
                    entity_text = f"{entity_data['name']} ({entity_data['type']})"
                    if entity_data.get('description'):
                        entity_text += f": {entity_data['description']}"
                    
                    # Generate embedding
                    embedding = await self.neo4j_vectors.embedding_service.generate_embedding(entity_text)
                    embeddings_to_store.append((entity_id, embedding))
            
            # Store embeddings in batch
            results = await self.neo4j_vectors.batch_store_entity_embeddings(embeddings_to_store)
            
            successful_count = sum(1 for success in results.values() if success)
            
            self.logger.info(f"Synced {successful_count}/{len(entity_ids)} entity embeddings to Neo4j")
            
            return {
                "total_processed": len(entity_ids),
                "successful": successful_count,
                "failed": len(entity_ids) - successful_count,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error syncing entity embeddings: {e}")
            return {"error": str(e)}
    
    async def sync_document_embeddings(self, document_ids: Optional[List[str]] = None) -> Dict:
        """Sync document-level embeddings"""
        try:
            # Get documents that need embeddings
            if document_ids is None:
                query = """
                MATCH (d:Document)
                WHERE d.summary_embedding IS NULL
                RETURN d.id as document_id, d.title as title, d.source_file as source_file
                LIMIT 100
                """
                
                result = await self.neo4j_vectors.neo4j_storage.execute_query(query)
                document_ids = [r["document_id"] for r in result]
            
            successful_count = 0
            
            for document_id in document_ids:
                # Get document chunks for summary
                chunks_query = """
                MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(chunk:DocumentChunk)
                RETURN chunk.text as text
                ORDER BY chunk.chunk_index
                LIMIT 10
                """
                
                chunks_result = await self.neo4j_vectors.neo4j_storage.execute_query(
                    chunks_query, {"document_id": document_id}
                )
                
                if chunks_result:
                    # Create summary text from first few chunks
                    summary_text = " ".join([chunk["text"][:200] for chunk in chunks_result[:3]])
                    
                    # Store document embedding
                    success = await self.neo4j_vectors.store_document_embedding(
                        document_id=document_id,
                        summary_text=summary_text
                    )
                    
                    if success:
                        successful_count += 1
            
            self.logger.info(f"Synced {successful_count}/{len(document_ids)} document embeddings")
            
            return {
                "total_processed": len(document_ids),
                "successful": successful_count,
                "failed": len(document_ids) - successful_count
            }
            
        except Exception as e:
            self.logger.error(f"Error syncing document embeddings: {e}")
            return {"error": str(e)}
    
    async def validate_vector_consistency(self) -> Dict:
        """Validate consistency between Neo4j and Qdrant vectors"""
        try:
            inconsistencies = []
            
            # Check entity embeddings
            entity_query = """
            MATCH (e:Entity)
            WHERE e.embedding_vector IS NOT NULL
            RETURN e.id as entity_id, e.embedding_vector as embedding
            LIMIT 100
            """
            
            entities = await self.neo4j_vectors.neo4j_storage.execute_query(entity_query)
            
            for entity in entities:
                entity_id = entity["entity_id"]
                neo4j_embedding = entity["embedding"]
                
                # Check if entity is referenced in Qdrant
                qdrant_refs = await self.id_mapping._check_qdrant_entity_references(entity_id)
                
                if not qdrant_refs:
                    inconsistencies.append({
                        "type": "entity",
                        "id": entity_id,
                        "issue": "Entity has Neo4j embedding but no Qdrant references"
                    })
            
            # Check document embeddings
            doc_query = """
            MATCH (d:Document)
            WHERE d.summary_embedding IS NOT NULL
            RETURN d.id as document_id, d.summary_embedding as embedding
            LIMIT 50
            """
            
            documents = await self.neo4j_vectors.neo4j_storage.execute_query(doc_query)
            
            for doc in documents:
                document_id = doc["document_id"]
                
                # Check if document has chunks in Qdrant
                qdrant_chunks = await self.qdrant_storage.client.scroll(
                    collection_name="morag_vectors",
                    scroll_filter={
                        "must": [{"key": "document_id", "match": {"value": document_id}}]
                    },
                    limit=1
                )
                
                if not qdrant_chunks[0]:  # No chunks found
                    inconsistencies.append({
                        "type": "document",
                        "id": document_id,
                        "issue": "Document has Neo4j embedding but no Qdrant chunks"
                    })
            
            return {
                "total_inconsistencies": len(inconsistencies),
                "inconsistencies": inconsistencies,
                "validation_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error validating vector consistency: {e}")
            return {"error": str(e)}
```

## Testing Strategy

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestNeo4jVectorStorage:
    @pytest.fixture
    async def vector_storage(self):
        neo4j_storage = AsyncMock()
        embedding_service = AsyncMock()
        
        return Neo4jVectorStorage(neo4j_storage, embedding_service)
    
    async def test_store_entity_embedding(self, vector_storage):
        """Test storing entity embedding"""
        # Mock embedding generation
        test_embedding = [0.1, 0.2, 0.3] * 128  # 384-dim vector
        vector_storage.embedding_service.generate_embedding.return_value = test_embedding
        
        # Mock Neo4j response
        vector_storage.neo4j_storage.execute_query.return_value = [{"entity_id": "entity_123"}]
        
        # Test storage
        success = await vector_storage.store_entity_embedding(
            entity_id="entity_123",
            text="John Doe (PERSON)"
        )
        
        assert success is True
        vector_storage.embedding_service.generate_embedding.assert_called_once_with("John Doe (PERSON)")
        vector_storage.neo4j_storage.execute_query.assert_called_once()
    
    async def test_find_similar_entities(self, vector_storage):
        """Test finding similar entities"""
        query_embedding = [0.1] * 384
        
        # Mock Neo4j response
        vector_storage.neo4j_storage.execute_query.return_value = [
            {
                "entity_id": "entity_1",
                "name": "John Doe",
                "type": "PERSON",
                "score": 0.95,
                "embedding_date": "2024-01-01T00:00:00"
            },
            {
                "entity_id": "entity_2",
                "name": "Jane Smith",
                "type": "PERSON",
                "score": 0.87,
                "embedding_date": "2024-01-01T00:00:00"
            }
        ]
        
        results = await vector_storage.find_similar_entities(
            query_embedding=query_embedding,
            limit=10,
            entity_types=["PERSON"]
        )
        
        assert len(results) == 2
        assert results[0]["entity_id"] == "entity_1"
        assert results[0]["similarity_score"] == 0.95
        assert results[1]["entity_id"] == "entity_2"
        assert results[1]["similarity_score"] == 0.87
    
    async def test_batch_store_entity_embeddings(self, vector_storage):
        """Test batch storage of entity embeddings"""
        entity_embeddings = [
            ("entity_1", [0.1] * 384),
            ("entity_2", [0.2] * 384),
            ("entity_3", [0.3] * 384)
        ]
        
        # Mock Neo4j batch response
        vector_storage.neo4j_storage.execute_query.return_value = [
            {"entity_id": "entity_1"},
            {"entity_id": "entity_2"},
            {"entity_id": "entity_3"}
        ]
        
        results = await vector_storage.batch_store_entity_embeddings(entity_embeddings)
        
        assert len(results) == 3
        assert all(results.values())  # All should be successful
        assert results["entity_1"] is True
        assert results["entity_2"] is True
        assert results["entity_3"] is True

class TestVectorSynchronizationService:
    @pytest.fixture
    async def sync_service(self):
        neo4j_vector_storage = AsyncMock()
        qdrant_storage = AsyncMock()
        id_mapping_service = AsyncMock()
        
        return VectorSynchronizationService(
            neo4j_vector_storage, qdrant_storage, id_mapping_service
        )
    
    async def test_sync_entity_embeddings_to_neo4j(self, sync_service):
        """Test syncing entity embeddings to Neo4j"""
        # Mock entities without embeddings
        sync_service.neo4j_vectors.neo4j_storage.execute_query.side_effect = [
            [  # First call: entities without embeddings
                {"entity_id": "entity_1", "name": "John Doe", "type": "PERSON"},
                {"entity_id": "entity_2", "name": "ACME Corp", "type": "ORGANIZATION"}
            ],
            [  # Second call: entity details for entity_1
                {"name": "John Doe", "type": "PERSON", "description": None}
            ],
            [  # Third call: entity details for entity_2
                {"name": "ACME Corp", "type": "ORGANIZATION", "description": "Technology company"}
            ]
        ]
        
        # Mock batch storage
        sync_service.neo4j_vectors.batch_store_entity_embeddings.return_value = {
            "entity_1": True,
            "entity_2": True
        }
        
        result = await sync_service.sync_entity_embeddings_to_neo4j()
        
        assert result["total_processed"] == 2
        assert result["successful"] == 2
        assert result["failed"] == 0
```

### Integration Tests

```python
class TestNeo4jVectorIntegration:
    """Integration tests with real Neo4j instance"""
    
    @pytest.fixture
    async def integration_setup(self):
        # Set up real Neo4j connection
        neo4j_storage = Neo4jStorage(test_config)
        embedding_service = EmbeddingService(test_config)
        
        vector_storage = Neo4jVectorStorage(neo4j_storage, embedding_service)
        
        # Clean up test data
        await self._cleanup_test_data(neo4j_storage)
        
        yield vector_storage
        
        # Clean up after tests
        await self._cleanup_test_data(neo4j_storage)
    
    async def test_end_to_end_vector_operations(self, integration_setup):
        """Test complete vector storage and retrieval workflow"""
        vector_storage = integration_setup
        
        # Create test entity
        entity_id = "test_entity_123"
        await vector_storage.neo4j_storage.execute_query(
            "CREATE (e:Entity {id: $id, name: $name, type: $type})",
            {"id": entity_id, "name": "Test Person", "type": "PERSON"}
        )
        
        # Store embedding
        success = await vector_storage.store_entity_embedding(
            entity_id=entity_id,
            text="Test Person (PERSON)"
        )
        assert success is True
        
        # Verify embedding was stored
        result = await vector_storage.neo4j_storage.execute_query(
            "MATCH (e:Entity {id: $id}) RETURN e.embedding_vector as embedding",
            {"id": entity_id}
        )
        
        assert result and len(result) > 0
        assert result[0]["embedding"] is not None
        assert len(result[0]["embedding"]) == 384  # Expected dimensions
        
        # Test similarity search
        query_embedding = result[0]["embedding"]  # Use same embedding for exact match
        similar_entities = await vector_storage.find_similar_entities(
            query_embedding=query_embedding,
            limit=5
        )
        
        assert len(similar_entities) >= 1
        assert similar_entities[0]["entity_id"] == entity_id
        assert similar_entities[0]["similarity_score"] >= 0.99  # Should be very similar
    
    async def _cleanup_test_data(self, neo4j_storage):
        """Clean up test data"""
        await neo4j_storage.execute_query(
            "MATCH (n) WHERE n.id STARTS WITH 'test_' DELETE n"
        )
```

## Performance Considerations

### Optimization Strategies

1. **Vector Indexing**:
   - Use appropriate vector index configurations
   - Monitor index performance and rebuild when necessary
   - Consider index warming strategies

2. **Batch Operations**:
   - Process embeddings in batches to reduce overhead
   - Use parallel processing for independent operations
   - Implement progress tracking for long-running operations

3. **Memory Management**:
   - Stream large result sets instead of loading all in memory
   - Use appropriate vector dimensions (384 vs 1536)
   - Implement vector compression if needed

### Performance Targets

- **Vector Storage**: < 100ms per entity/relation embedding
- **Similarity Search**: < 200ms for top-10 results
- **Batch Operations**: Process 1000+ embeddings per minute
- **Index Performance**: < 500ms for vector index queries

## Success Criteria

- [ ] Vector storage implemented for entities, relations, and documents
- [ ] Vector similarity search working with acceptable performance
- [ ] Batch operations for efficient bulk processing
- [ ] Integration with existing Neo4j schema
- [ ] Synchronization service for vector consistency
- [ ] Comprehensive test coverage (>90%)
- [ ] Performance benchmarks meeting targets
- [ ] Monitoring and statistics collection

## Risk Assessment

**Medium Risk**: Vector index performance and memory usage

**Mitigation Strategies**:
- Monitor vector index performance continuously
- Implement vector dimension optimization
- Use streaming for large result sets
- Implement circuit breakers for resource protection

## Rollback Plan

1. **Remove vector indexes** to free up resources
2. **Clear vector fields** from nodes and relationships
3. **Restore previous schema** without vector extensions
4. **Monitor system performance** after rollback

## Next Steps

- **Task 3.2**: Selective Vector Strategy
- **Task 3.3**: Embedding Synchronization Pipeline
- **Integration**: Incorporate into hybrid retrieval system

## Dependencies

- **Task 2.3**: ID Mapping Utilities (completed)
- Neo4j 5.0+ with vector index support
- Embedding service for vector generation
- Sufficient memory for vector storage and indexing

## Estimated Time

**5-6 days**

## Status

- [ ] Not Started
- [ ] In Progress
- [ ] Testing
- [ ] Completed
- [ ] Verified