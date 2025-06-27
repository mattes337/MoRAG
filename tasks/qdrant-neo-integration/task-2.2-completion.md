# Task 2.2: Vector Embedding Integration - Completion

## Remaining HybridSearchEngine Methods

```python
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

### Unit Tests

```python
# tests/test_vector_embedding_integration.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.morag_graph.services.vector_embedding_manager import (
    VectorEmbeddingManager, VectorEmbedding, VectorType, EmbeddingConfig, EmbeddingModel
)
from src.morag_graph.services.hybrid_search_engine import HybridSearchEngine

class TestVectorEmbeddingManager:
    @pytest.fixture
    async def embedding_manager(self):
        neo4j_storage = Mock()
        qdrant_storage = Mock()
        config = EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS,
            dense_dimension=384,
            storage_strategy="hybrid"
        )
        
        manager = VectorEmbeddingManager(neo4j_storage, qdrant_storage, config)
        await manager.initialize_vector_storage()
        return manager
    
    @pytest.mark.asyncio
    async def test_create_dense_embedding(self, embedding_manager):
        """Test dense embedding creation."""
        embedding = await embedding_manager.create_embedding(
            text="Test document content",
            embedding_id="test_embedding_1",
            vector_type=VectorType.DENSE
        )
        
        assert embedding.id == "test_embedding_1"
        assert embedding.vector_type == VectorType.DENSE
        assert embedding.dense_vector is not None
        assert len(embedding.dense_vector) == 384
        assert embedding.sparse_vector is None
    
    @pytest.mark.asyncio
    async def test_create_sparse_embedding(self, embedding_manager):
        """Test sparse embedding creation."""
        embedding = await embedding_manager.create_embedding(
            text="Test document with keywords",
            embedding_id="test_embedding_2",
            vector_type=VectorType.SPARSE
        )
        
        assert embedding.id == "test_embedding_2"
        assert embedding.vector_type == VectorType.SPARSE
        assert embedding.sparse_vector is not None
        assert isinstance(embedding.sparse_vector, dict)
        assert embedding.dense_vector is None
    
    @pytest.mark.asyncio
    async def test_batch_create_embeddings(self, embedding_manager):
        """Test batch embedding creation."""
        texts = ["Document 1", "Document 2", "Document 3"]
        embedding_ids = ["batch_1", "batch_2", "batch_3"]
        
        embeddings = await embedding_manager.batch_create_embeddings(
            texts=texts,
            embedding_ids=embedding_ids,
            vector_type=VectorType.DENSE
        )
        
        assert len(embeddings) == 3
        for i, embedding in enumerate(embeddings):
            assert embedding.id == embedding_ids[i]
            assert embedding.vector_type == VectorType.DENSE
    
    @pytest.mark.asyncio
    async def test_synchronize_embeddings(self, embedding_manager):
        """Test embedding synchronization between systems."""
        embedding = VectorEmbedding(
            id="sync_test",
            text="Synchronization test",
            vector_type=VectorType.DENSE,
            dense_vector=[0.1] * 384
        )
        
        result = await embedding_manager.synchronize_embedding(embedding)
        
        assert result['neo4j_stored'] is True
        assert result['qdrant_stored'] is True
        assert result['sync_status'] == 'success'

class TestHybridSearchEngine:
    @pytest.fixture
    async def search_engine(self):
        vector_manager = Mock(spec=VectorEmbeddingManager)
        query_engine = Mock()
        
        engine = HybridSearchEngine(vector_manager, query_engine)
        return engine
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, search_engine):
        """Test semantic search functionality."""
        # Mock vector manager response
        mock_embedding = Mock()
        mock_embedding.dense_vector = [0.1] * 384
        search_engine.vector_manager.create_embedding.return_value = mock_embedding
        
        mock_results = [
            {'id': 'doc1', 'score': 0.95, 'text': 'Relevant document 1'},
            {'id': 'doc2', 'score': 0.87, 'text': 'Relevant document 2'}
        ]
        search_engine.vector_manager.search_similar_embeddings.return_value = mock_results
        
        results = await search_engine.semantic_search(
            query_text="test query",
            top_k=5
        )
        
        assert results['query_text'] == "test query"
        assert results['total_found'] == 2
        assert results['search_type'] == 'semantic'
        assert len(results['results']) == 2
    
    @pytest.mark.asyncio
    async def test_keyword_search(self, search_engine):
        """Test keyword search functionality."""
        # Mock sparse embedding creation and search
        mock_sparse_embedding = Mock()
        mock_sparse_embedding.sparse_vector = {'keyword': 0.8, 'test': 0.6}
        search_engine.vector_manager.create_embedding.return_value = mock_sparse_embedding
        
        mock_results = [
            {'id': 'doc3', 'score': 0.92, 'text': 'Keyword match document'}
        ]
        search_engine.vector_manager.search_similar_embeddings.return_value = mock_results
        
        results = await search_engine.keyword_search(
            query_text="keyword test",
            top_k=10
        )
        
        assert results['query_text'] == "keyword test"
        assert results['search_type'] == 'keyword'
        assert len(results['results']) == 1
```

### Integration Tests

```python
# tests/integration/test_vector_integration_flow.py
import pytest
import asyncio
from src.morag_graph.services.vector_embedding_manager import VectorEmbeddingManager
from src.morag_graph.services.hybrid_search_engine import HybridSearchEngine
from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.storage.qdrant_storage import QdrantStorage

@pytest.mark.integration
class TestVectorIntegrationFlow:
    @pytest.fixture(scope="class")
    async def integration_setup(self):
        """Setup integration test environment."""
        # Initialize real storage instances for integration testing
        neo4j_storage = Neo4jStorage(test_config)
        qdrant_storage = QdrantStorage(test_config)
        
        vector_manager = VectorEmbeddingManager(
            neo4j_storage=neo4j_storage,
            qdrant_storage=qdrant_storage,
            config=test_embedding_config
        )
        
        await vector_manager.initialize_vector_storage()
        
        search_engine = HybridSearchEngine(
            vector_manager=vector_manager,
            query_engine=test_query_engine
        )
        
        return {
            'vector_manager': vector_manager,
            'search_engine': search_engine,
            'neo4j_storage': neo4j_storage,
            'qdrant_storage': qdrant_storage
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_embedding_flow(self, integration_setup):
        """Test complete embedding creation and search flow."""
        components = await integration_setup
        vector_manager = components['vector_manager']
        search_engine = components['search_engine']
        
        # Create test embeddings
        test_texts = [
            "Machine learning algorithms for data analysis",
            "Natural language processing techniques",
            "Computer vision and image recognition"
        ]
        
        embedding_ids = [f"integration_test_{i}" for i in range(len(test_texts))]
        
        # Create dense embeddings
        dense_embeddings = await vector_manager.batch_create_embeddings(
            texts=test_texts,
            embedding_ids=embedding_ids,
            vector_type=VectorType.DENSE
        )
        
        # Verify embeddings were created
        assert len(dense_embeddings) == 3
        
        # Test semantic search
        search_results = await search_engine.semantic_search(
            query_text="machine learning data",
            top_k=5
        )
        
        assert search_results['total_found'] > 0
        assert search_results['search_type'] == 'semantic'
        
        # Verify top result is relevant
        top_result = search_results['results'][0]
        assert 'machine learning' in top_result['text'].lower() or top_result['score'] > 0.7
    
    @pytest.mark.asyncio
    async def test_cross_system_synchronization(self, integration_setup):
        """Test synchronization between Neo4j and Qdrant."""
        components = await integration_setup
        vector_manager = components['vector_manager']
        
        # Create embedding with synchronization
        embedding = await vector_manager.create_embedding(
            text="Synchronization test document",
            embedding_id="sync_integration_test",
            vector_type=VectorType.DENSE
        )
        
        # Verify embedding exists in both systems
        neo4j_exists = await components['neo4j_storage'].embedding_exists(embedding.id)
        qdrant_exists = await components['qdrant_storage'].embedding_exists(embedding.id)
        
        assert neo4j_exists is True
        assert qdrant_exists is True
        
        # Test synchronization status
        sync_stats = await vector_manager.get_synchronization_statistics()
        assert sync_stats['total_embeddings'] > 0
        assert sync_stats['sync_percentage'] >= 95.0  # Allow for minor sync delays
```

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**
   - Process embeddings in configurable batch sizes (default: 100)
   - Use async/await for concurrent operations
   - Implement connection pooling for both Neo4j and Qdrant

2. **Caching Strategy**
   - Cache frequently accessed embeddings in Redis
   - Implement LRU cache for embedding models
   - Cache search results for common queries

3. **Index Optimization**
   - Use appropriate vector indexes in both systems
   - Configure HNSW parameters for optimal performance
   - Monitor index rebuild requirements

4. **Memory Management**
   - Stream large embedding batches
   - Implement garbage collection for temporary embeddings
   - Monitor memory usage during batch operations

### Performance Benchmarks

```python
# Performance targets for Task 2.2
PERFORMANCE_TARGETS = {
    'embedding_creation': {
        'single_embedding': '< 100ms',
        'batch_100_embeddings': '< 5s',
        'batch_1000_embeddings': '< 30s'
    },
    'search_performance': {
        'semantic_search': '< 200ms',
        'keyword_search': '< 150ms',
        'hybrid_search': '< 300ms'
    },
    'synchronization': {
        'single_sync': '< 50ms',
        'batch_sync_100': '< 2s',
        'full_system_sync': '< 5min'
    },
    'memory_usage': {
        'max_memory_per_embedding': '< 2MB',
        'batch_memory_overhead': '< 20%'
    }
}
```

## Success Criteria

- [ ] Dense and sparse vector embeddings successfully created and stored
- [ ] Cross-system synchronization maintains 99%+ consistency
- [ ] Semantic search returns relevant results with >0.7 average score
- [ ] Keyword search handles sparse vector queries effectively
- [ ] Hybrid search combines multiple retrieval methods
- [ ] Performance targets met for all operations
- [ ] All unit and integration tests pass
- [ ] Memory usage remains within acceptable limits
- [ ] Error handling covers all failure scenarios
- [ ] Monitoring and logging provide adequate visibility

## Risk Assessment

**High Risk:**
- Vector dimension mismatches between systems
- Synchronization failures during high load
- Memory exhaustion with large embedding batches

**Medium Risk:**
- Performance degradation with large vector collections
- Model compatibility issues across different embedding types
- Network latency affecting cross-system operations

**Low Risk:**
- Minor configuration differences
- Logging and monitoring setup
- Documentation completeness

## Rollback Plan

1. **Immediate Rollback:**
   - Disable vector embedding integration
   - Revert to previous search mechanisms
   - Preserve existing data integrity

2. **Data Recovery:**
   - Backup vector collections before deployment
   - Implement rollback scripts for both Neo4j and Qdrant
   - Verify data consistency after rollback

3. **Service Restoration:**
   - Restart services with previous configuration
   - Validate all search functionality
   - Monitor system performance post-rollback

## Next Steps

- **Task 2.3:** Hybrid Query Optimization
- **Task 3.1:** Performance Monitoring Integration
- **Task 3.2:** Advanced Caching Strategies

## Dependencies

- **Requires:** Task 2.1 (Cross-System Entity Linking)
- **Blocks:** Task 2.3 (Hybrid Query Optimization)
- **Related:** Task 1.3 (Entity ID Integration)

## Estimated Time

**5-7 days** (including testing and optimization)

## Status

- [ ] **Planning:** Requirements analysis and design
- [ ] **Implementation:** Core vector embedding functionality
- [ ] **Testing:** Unit and integration test development
- [ ] **Optimization:** Performance tuning and benchmarking
- [ ] **Documentation:** API documentation and usage examples
- [ ] **Review:** Code review and quality assurance
- [ ] **Deployment:** Production deployment and monitoring