# Task 2: Optimize Search Endpoint Embedding Strategy

## Problem Analysis

### Current Issues
1. **Inefficient batch embedding for single queries**: The search endpoint currently uses batch embedding APIs for single query text, which is unnecessary overhead.

2. **Performance overhead**: Batch embedding adds complexity and latency when only one query needs to be embedded.

3. **Resource waste**: Batch embedding infrastructure is designed for multiple texts but search only needs single text embedding.

4. **Code complexity**: Using batch APIs for single queries makes the code more complex than necessary.

### Current Implementation Analysis
From `packages/morag-services/src/morag_services/services.py` line 720:
```python
query_embedding = await self._gemini_embedding_service.generate_embedding(
    query,
    task_type="retrieval_query"
)
```

The current implementation actually uses single embedding, but let me verify if there are any batch calls in the search path.

From analysis, the search endpoint appears to already use single embedding calls, but we need to investigate:
1. Whether there are any hidden batch calls in the embedding service
2. If the embedding service is optimized for single queries
3. Whether the task_type parameter is being used efficiently

### Investigation Required
1. **Check embedding service implementation** for any batch processing in single calls
2. **Analyze performance metrics** of current search embedding
3. **Verify task_type usage** and optimization opportunities
4. **Review embedding caching** possibilities

## Solution Approach

### 1. Single Query Optimization
- Ensure search endpoint uses the most efficient single embedding method
- Remove any unnecessary batch processing overhead
- Optimize for the specific use case of search queries

### 2. Embedding Service Streamlining
- Create dedicated search embedding method if needed
- Optimize task_type handling for search queries
- Implement caching for repeated search queries

### 3. Performance Monitoring
- Add performance metrics for search embedding
- Compare single vs batch embedding performance
- Monitor embedding latency and throughput

### 4. Code Simplification
- Remove unnecessary complexity in search embedding path
- Streamline the embedding generation process
- Improve error handling for search-specific scenarios

## Implementation Plan

### Phase 1: Current State Analysis
1. **Audit embedding service calls** in search endpoint
   - Trace all embedding calls from search to service
   - Identify any batch processing in the call chain
   - Measure current performance metrics

2. **Review embedding service implementation**
   - Check `generate_embedding()` vs `embed_batch()` usage
   - Analyze internal implementation for efficiency
   - Identify optimization opportunities

### Phase 2: Optimization Implementation
1. **Create search-optimized embedding method**
   - Implement `generate_search_embedding()` if beneficial
   - Optimize for single query use case
   - Add search-specific caching

2. **Remove batch processing overhead**
   - Eliminate any unnecessary batch processing
   - Streamline the embedding generation path
   - Optimize task_type handling

### Phase 3: Performance Enhancement
1. **Implement embedding caching**
   - Cache frequent search queries
   - Add TTL-based cache invalidation
   - Monitor cache hit rates

2. **Add performance monitoring**
   - Track embedding generation time
   - Monitor API call efficiency
   - Add performance metrics to logs

### Phase 4: Testing and Validation
1. **Performance testing**
   - Benchmark search embedding performance
   - Compare before/after optimization
   - Test with various query types and sizes

2. **Load testing**
   - Test concurrent search requests
   - Validate embedding service scalability
   - Monitor resource usage

## Technical Implementation

### Current Search Flow Analysis
```python
# Current flow in search_similar()
query_embedding = await self._gemini_embedding_service.generate_embedding(
    query,
    task_type="retrieval_query"
)
```

### Proposed Optimizations

#### 1. Search-Specific Embedding Method
```python
async def generate_search_embedding(
    self,
    query: str,
    cache_key: Optional[str] = None
) -> List[float]:
    """Generate embedding optimized for search queries."""
    
    # Check cache first
    if cache_key and self._embedding_cache:
        cached = await self._embedding_cache.get(cache_key)
        if cached:
            return cached
    
    # Generate embedding with search-optimized parameters
    embedding = await self._generate_embedding_optimized(
        query,
        task_type="retrieval_query",
        optimize_for_search=True
    )
    
    # Cache result
    if cache_key and self._embedding_cache:
        await self._embedding_cache.set(cache_key, embedding, ttl=3600)
    
    return embedding
```

#### 2. Embedding Cache Implementation
```python
class EmbeddingCache:
    """Simple in-memory cache for search embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._access_times = {}
        self._max_size = max_size
    
    async def get(self, key: str) -> Optional[List[float]]:
        """Get cached embedding."""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    async def set(self, key: str, embedding: List[float], ttl: int = 3600):
        """Cache embedding with TTL."""
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        self._cache[key] = embedding
        self._access_times[key] = time.time()
```

#### 3. Performance Monitoring
```python
async def search_similar_with_metrics(
    self,
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search with performance monitoring."""
    
    start_time = time.time()
    
    # Generate embedding with timing
    embedding_start = time.time()
    query_embedding = await self._generate_search_embedding_optimized(query)
    embedding_time = time.time() - embedding_start
    
    # Perform search with timing
    search_start = time.time()
    results = await self._vector_storage.search_similar(
        query_vector=query_embedding,
        limit=limit,
        filters=filters
    )
    search_time = time.time() - search_start
    
    total_time = time.time() - start_time
    
    # Log performance metrics
    logger.info("Search performance metrics",
               embedding_time=embedding_time,
               search_time=search_time,
               total_time=total_time,
               query_length=len(query),
               results_count=len(results))
    
    return results
```

## Files to Modify

### Core Search Implementation
1. `packages/morag-services/src/morag_services/services.py`
   - Optimize `search_similar()` method
   - Add search-specific embedding generation
   - Implement performance monitoring

2. `packages/morag-services/src/morag_services/embedding.py`
   - Add search-optimized embedding method
   - Implement embedding caching
   - Remove unnecessary batch processing overhead

### Embedding Service
1. `packages/morag-embedding/src/morag_embedding/service.py`
   - Optimize single embedding generation
   - Add search-specific optimizations
   - Implement caching layer

### Configuration
1. `packages/morag-core/src/morag_core/config.py`
   - Add search embedding configuration options
   - Add caching configuration
   - Add performance monitoring settings

### API Layer
1. `packages/morag/src/morag/server.py`
   - Update search endpoint to use optimized embedding
   - Add performance metrics to response
   - Implement query preprocessing

## Testing Strategy

### Performance Tests
1. **Embedding generation benchmarks**
   - Measure single embedding performance
   - Compare optimized vs current implementation
   - Test with various query lengths

2. **Search endpoint benchmarks**
   - Measure end-to-end search performance
   - Test concurrent search requests
   - Monitor resource usage

### Functional Tests
1. **Search accuracy tests**
   - Verify optimization doesn't affect search quality
   - Test with various query types
   - Validate embedding consistency

2. **Cache functionality tests**
   - Test cache hit/miss scenarios
   - Validate TTL expiration
   - Test cache eviction policies

### Load Tests
1. **Concurrent search tests**
   - Test multiple simultaneous searches
   - Monitor embedding service performance
   - Validate cache effectiveness

2. **Stress tests**
   - Test with high query volumes
   - Monitor memory usage
   - Test cache performance under load

## Success Criteria

### Performance Improvements
1. ✅ Search embedding generation time reduced by at least 20%
2. ✅ Reduced API calls to embedding service
3. ✅ Improved cache hit rate for repeated queries
4. ✅ Better resource utilization

### Quality Maintenance
1. ✅ Search accuracy maintained or improved
2. ✅ Consistent embedding generation
3. ✅ Reliable cache behavior
4. ✅ Proper error handling

### Monitoring and Observability
1. ✅ Performance metrics available in logs
2. ✅ Cache statistics monitoring
3. ✅ Embedding service health monitoring
4. ✅ Search endpoint performance tracking

## Configuration Options

### Environment Variables
```env
# Search embedding optimization
SEARCH_EMBEDDING_CACHE_ENABLED=true
SEARCH_EMBEDDING_CACHE_SIZE=1000
SEARCH_EMBEDDING_CACHE_TTL=3600

# Performance monitoring
SEARCH_PERFORMANCE_MONITORING=true
SEARCH_METRICS_LOG_LEVEL=info

# Embedding optimization
EMBEDDING_SINGLE_QUERY_OPTIMIZATION=true
EMBEDDING_SEARCH_TASK_TYPE=retrieval_query
```

### Configuration Class Updates
```python
class Settings(BaseSettings):
    # Search embedding settings
    search_embedding_cache_enabled: bool = True
    search_embedding_cache_size: int = 1000
    search_embedding_cache_ttl: int = 3600
    
    # Performance monitoring
    search_performance_monitoring: bool = True
    search_metrics_log_level: str = "info"
    
    # Embedding optimization
    embedding_single_query_optimization: bool = True
    embedding_search_task_type: str = "retrieval_query"
```

## Risk Mitigation

### Performance Risks
1. **Cache memory usage**: Large cache might consume too much memory
   - Mitigation: Implement LRU eviction and configurable cache size
   - Monitoring: Track cache memory usage

2. **Cache invalidation**: Stale embeddings might affect search quality
   - Mitigation: Implement TTL-based expiration
   - Validation: Monitor search quality metrics

### Implementation Risks
1. **Breaking changes**: Optimization might break existing functionality
   - Mitigation: Maintain backward compatibility
   - Testing: Comprehensive regression testing

2. **Complexity increase**: Caching adds complexity
   - Mitigation: Simple, well-tested cache implementation
   - Documentation: Clear cache behavior documentation

## Implementation Timeline

### Week 1: Analysis and Design
- [ ] Audit current search embedding implementation
- [ ] Identify optimization opportunities
- [ ] Design cache architecture

### Week 2: Core Optimization
- [ ] Implement search-specific embedding method
- [ ] Add embedding caching
- [ ] Optimize single query path

### Week 3: Performance Monitoring
- [ ] Add performance metrics
- [ ] Implement monitoring and logging
- [ ] Create performance benchmarks

### Week 4: Testing and Validation
- [ ] Performance testing and optimization
- [ ] Load testing and validation
- [ ] Documentation and deployment

## Next Steps

1. **Audit current implementation**: Analyze the actual embedding calls in search
2. **Identify bottlenecks**: Find specific areas for optimization
3. **Implement caching**: Start with simple in-memory cache
4. **Add monitoring**: Implement performance metrics and logging

This task focuses on optimizing the search endpoint's embedding strategy for better performance and efficiency.
