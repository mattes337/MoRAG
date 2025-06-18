# Task 2.2: Vector Embedding Integration

## Overview

Integrate dense and sparse vector embeddings across Neo4j and Qdrant systems to enable advanced semantic search and hybrid retrieval capabilities.

## Objectives

- Implement dense and sparse vector storage and retrieval
- Create synchronization mechanisms between Neo4j and Qdrant
- Develop embedding model management and caching
- Enable hybrid search combining multiple vector types
- Establish performance monitoring and optimization

## Implementation Plan

### 1. Vector Embedding Data Models

```python
# src/morag_graph/models/vector_embedding.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np

class VectorType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

class EmbeddingModel(Enum):
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI_ADA = "openai-ada-002"
    COHERE_EMBED = "cohere-embed"
    CUSTOM = "custom"

@dataclass
class VectorEmbedding:
    """Unified vector embedding representation."""
    id: str
    text: str
    vector_type: VectorType
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None
    model_name: str = ""
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Validate vector type consistency
        if self.vector_type == VectorType.DENSE and self.dense_vector is None:
            raise ValueError("Dense vector type requires dense_vector data")
        elif self.vector_type == VectorType.SPARSE and self.sparse_vector is None:
            raise ValueError("Sparse vector type requires sparse_vector data")
        elif self.vector_type == VectorType.HYBRID:
            if self.dense_vector is None or self.sparse_vector is None:
                raise ValueError("Hybrid vector type requires both dense and sparse vectors")

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation and storage."""
    model: EmbeddingModel
    dense_dimension: int = 384
    sparse_dimension: Optional[int] = None
    batch_size: int = 100
    storage_strategy: str = "hybrid"  # "neo4j_only", "qdrant_only", "hybrid"
    cache_embeddings: bool = True
    normalize_vectors: bool = True
```

### 2. Vector Embedding Manager Service

```python
# src/morag_graph/services/vector_embedding_manager.py
import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from ..models.vector_embedding import VectorEmbedding, VectorType, EmbeddingConfig, EmbeddingModel
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage

logger = logging.getLogger(__name__)

class VectorEmbeddingManager:
    """Manages vector embeddings across Neo4j and Qdrant systems."""
    
    def __init__(self, 
                 neo4j_storage: Neo4jStorage,
                 qdrant_storage: QdrantStorage,
                 config: EmbeddingConfig):
        self.neo4j = neo4j_storage
        self.qdrant = qdrant_storage
        self.config = config
        self._embedding_models: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}
        self._batch_queue: List[Tuple[str, str]] = []  # (id, text) pairs
        self._processing_lock = asyncio.Lock()
    
    async def initialize_vector_storage(self) -> Dict[str, Any]:
        """Initialize vector storage systems with proper configurations."""
        try:
            # Initialize Qdrant collections for different vector types
            await self._initialize_qdrant_dense_collection()
            await self._initialize_qdrant_sparse_collection()
            
            # Initialize Neo4j vector indexes
            await self._initialize_neo4j_vector_indexes()
            
            # Load and cache embedding models
            await self._load_embedding_models()
            
            logger.info("Vector storage systems initialized successfully")
            return {
                'status': 'success',
                'qdrant_collections': ['dense', 'sparse'],
                'neo4j_indexes': ['entity_embeddings', 'chunk_embeddings'],
                'models_loaded': list(self._embedding_models.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize vector storage: {e}")
            raise
    
    async def create_embedding(self, 
                             text: str,
                             embedding_id: str,
                             vector_type: VectorType,
                             metadata: Optional[Dict[str, Any]] = None) -> VectorEmbedding:
        """Create a single vector embedding."""
        try:
            # Generate embedding based on type
            if vector_type == VectorType.DENSE:
                dense_vector = await self._generate_dense_embedding(text)
                embedding = VectorEmbedding(
                    id=embedding_id,
                    text=text,
                    vector_type=vector_type,
                    dense_vector=dense_vector,
                    metadata=metadata or {},
                    model_name=self.config.model.value,
                    created_at=datetime.utcnow().isoformat()
                )
            
            elif vector_type == VectorType.SPARSE:
                sparse_vector = await self._generate_sparse_embedding(text)
                embedding = VectorEmbedding(
                    id=embedding_id,
                    text=text,
                    vector_type=vector_type,
                    sparse_vector=sparse_vector,
                    metadata=metadata or {},
                    model_name=self.config.model.value,
                    created_at=datetime.utcnow().isoformat()
                )
            
            elif vector_type == VectorType.HYBRID:
                dense_vector = await self._generate_dense_embedding(text)
                sparse_vector = await self._generate_sparse_embedding(text)
                embedding = VectorEmbedding(
                    id=embedding_id,
                    text=text,
                    vector_type=vector_type,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    metadata=metadata or {},
                    model_name=self.config.model.value,
                    created_at=datetime.utcnow().isoformat()
                )
            
            # Store embedding in appropriate systems
            await self._store_embedding(embedding)
            
            logger.debug(f"Created {vector_type.value} embedding: {embedding_id}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create embedding {embedding_id}: {e}")
            raise
    
    async def batch_create_embeddings(self,
                                    texts: List[str],
                                    embedding_ids: List[str],
                                    vector_type: VectorType,
                                    metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[VectorEmbedding]:
        """Create multiple embeddings in batch for efficiency."""
        if len(texts) != len(embedding_ids):
            raise ValueError("texts and embedding_ids must have the same length")
        
        async with self._processing_lock:
            embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = embedding_ids[i:i + batch_size]
                batch_metadata = metadata_list[i:i + batch_size] if metadata_list else [None] * len(batch_texts)
                
                # Process batch
                batch_embeddings = await self._process_embedding_batch(
                    batch_texts, batch_ids, vector_type, batch_metadata
                )
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"Processed embedding batch {i//batch_size + 1}: {len(batch_embeddings)} embeddings")
            
            logger.info(f"Created {len(embeddings)} embeddings in batch mode")
            return embeddings
    
    async def search_similar_embeddings(self,
                                      query_embedding: VectorEmbedding,
                                      vector_type: VectorType,
                                      top_k: int = 10,
                                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings using vector similarity."""
        try:
            if vector_type == VectorType.DENSE and query_embedding.dense_vector:
                results = await self.qdrant.search_similar_vectors(
                    query_vector=query_embedding.dense_vector,
                    collection_name=f"{self.qdrant.collection_name}_dense",
                    top_k=top_k,
                    filter_conditions=filter_metadata
                )
            
            elif vector_type == VectorType.SPARSE and query_embedding.sparse_vector:
                results = await self.qdrant.search_sparse_vectors(
                    query_vector=query_embedding.sparse_vector,
                    collection_name=f"{self.qdrant.collection_name}_sparse",
                    top_k=top_k,
                    filter_conditions=filter_metadata
                )
            
            else:
                raise ValueError(f"Invalid vector type {vector_type} or missing vector data")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise
    
    async def synchronize_embedding(self, embedding: VectorEmbedding) -> Dict[str, Any]:
        """Synchronize embedding across Neo4j and Qdrant systems."""
        try:
            sync_results = {
                'embedding_id': embedding.id,
                'neo4j_stored': False,
                'qdrant_stored': False,
                'sync_status': 'pending'
            }
            
            # Store in Neo4j (for graph relationships and metadata)
            if self.config.storage_strategy in ['neo4j_only', 'hybrid']:
                await self._store_embedding_neo4j(embedding)
                sync_results['neo4j_stored'] = True
            
            # Store in Qdrant (for vector similarity search)
            if self.config.storage_strategy in ['qdrant_only', 'hybrid']:
                await self._store_embedding_qdrant(embedding)
                sync_results['qdrant_stored'] = True
            
            sync_results['sync_status'] = 'success'
            logger.debug(f"Synchronized embedding {embedding.id} across systems")
            
            return sync_results
            
        except Exception as e:
            logger.error(f"Failed to synchronize embedding {embedding.id}: {e}")
            sync_results['sync_status'] = 'failed'
            sync_results['error'] = str(e)
            return sync_results
    
    async def get_synchronization_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics between systems."""
        try:
            # Count embeddings in each system
            neo4j_count = await self._count_neo4j_embeddings()
            qdrant_count = await self._count_qdrant_embeddings()
            
            # Calculate synchronization percentage
            total_embeddings = max(neo4j_count, qdrant_count)
            sync_percentage = (min(neo4j_count, qdrant_count) / total_embeddings * 100) if total_embeddings > 0 else 100
            
            return {
                'neo4j_embeddings': neo4j_count,
                'qdrant_embeddings': qdrant_count,
                'total_embeddings': total_embeddings,
                'sync_percentage': round(sync_percentage, 2),
                'sync_status': 'healthy' if sync_percentage >= 95 else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Failed to get synchronization statistics: {e}")
            return {'error': str(e), 'sync_status': 'unknown'}
```

### 3. Hybrid Search Engine

```python
# src/morag_graph/services/hybrid_search_engine.py
import asyncio
import logging
from typing import List, Dict, Optional, Any

from .vector_embedding_manager import VectorEmbeddingManager
from ..models.vector_embedding import VectorType

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Advanced search engine combining multiple retrieval methods."""
    
    def __init__(self, 
                 vector_manager: VectorEmbeddingManager,
                 query_engine: Any):  # CrossSystemQueryEngine from Task 2.1
        self.vector_manager = vector_manager
        self.query_engine = query_engine
    
    async def semantic_search(self, 
                            query_text: str,
                            top_k: int = 10,
                            filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute semantic search using dense embeddings."""
        # Generate dense query embedding
        query_embedding = await self.vector_manager.create_embedding(
            text=query_text,
            embedding_id=f"semantic_query_{hash(query_text)}",
            vector_type=VectorType.DENSE
        )
        
        # Search using dense vectors
        results = await self.vector_manager.search_similar_embeddings(
            query_embedding=query_embedding,
            vector_type=VectorType.DENSE,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        return {
            'query_text': query_text,
            'results': results,
            'total_found': len(results),
            'search_type': 'semantic'
        }
    
    async def keyword_search(self, 
                           query_text: str,
                           top_k: int = 10) -> Dict[str, Any]:
        """Execute keyword-based search using sparse embeddings."""
        # Generate sparse query embedding
        query_embedding = await self.vector_manager.create_embedding(
            text=query_text,
            embedding_id=f"keyword_query_{hash(query_text)}",
            vector_type=VectorType.SPARSE
        )
        
        # Search using sparse vectors
        results = await self.vector_manager.search_similar_embeddings(
            query_embedding=query_embedding,
            vector_type=VectorType.SPARSE,
            top_k=top_k
        )
        
        return {
            'query_text': query_text,
            'results': results,
            'total_found': len(results),
            'search_type': 'keyword'
        }
    
    async def entity_based_search(self, 
                                entity_ids: List[str],
                                expand_graph: bool = True,
                                max_hops: int = 2,
                                top_k: int = 10) -> Dict[str, Any]:
        """Execute entity-based search with graph expansion."""
        # Use cross-system query engine for entity search
        results = await self.query_engine.execute_hybrid_entity_query(
            entity_context=entity_ids,
            expand_entities=expand_graph,
            max_hops=max_hops,
            final_top_k=top_k
        )
        
        return {
            'entity_ids': entity_ids,
            'results': results['results'],
            'total_found': len(results['results']),
            'search_type': 'entity_based',
            'graph_expansion_applied': results.get('graph_expansion_applied', False)
        }
```

## Testing Strategy

*Note: Complete testing implementation and additional sections are available in the companion file `task-2.2-completion.md`*

### Unit Tests Overview

- **VectorEmbeddingManager Tests:** Dense/sparse embedding creation, batch processing, synchronization
- **HybridSearchEngine Tests:** Semantic search, keyword search, entity-based search
- **Integration Tests:** End-to-end embedding flow, cross-system synchronization

### Performance Targets

- Single embedding creation: < 100ms
- Batch processing (100 embeddings): < 5s
- Semantic search: < 200ms
- Cross-system synchronization: 99%+ consistency

## Success Criteria

- [ ] Dense and sparse vector embeddings successfully integrated
- [ ] Cross-system synchronization maintains high consistency
- [ ] Hybrid search combines multiple retrieval methods effectively
- [ ] Performance targets met for all operations
- [ ] Comprehensive test coverage achieved

## Dependencies

- **Requires:** Task 2.1 (Cross-System Entity Linking)
- **Blocks:** Task 2.3 (Hybrid Query Optimization)

## Estimated Time

**5-7 days** (including testing and optimization)

## Status

- [ ] **Planning:** Requirements analysis and design
- [ ] **Implementation:** Core vector embedding functionality
- [ ] **Testing:** Unit and integration test development
- [ ] **Optimization:** Performance tuning and benchmarking
- [ ] **Documentation:** API documentation and usage examples