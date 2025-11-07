# Quick Win 4: Enhanced Caching Strategy

## Overview

**Priority**: ⚡ **Next Sprint** (1 week, Medium Impact, High ROI)
**Source**: Multiple papers emphasizing performance optimization
**Expected Impact**: 40-50% reduction in query latency for common patterns

## Problem Statement

MoRAG currently has basic caching implementation but lacks:
- Intelligent cache policies based on query patterns
- Entity neighborhood caching for graph traversal
- Query similarity matching for cache hits
- Cache warming for frequently accessed combinations
- Adaptive cache sizing and eviction policies

This results in unnecessary recomputation of expensive operations like graph traversal, entity extraction, and LLM calls.

## Solution Overview

Implement a multi-layered intelligent caching system that caches entity neighborhoods, query results, and intermediate computations with smart policies for cache warming, similarity matching, and adaptive management.

## Technical Implementation

### 1. Multi-Layer Cache System

Create `packages/morag-core/src/morag_core/caching/cache_manager.py`:

```python
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import json
import asyncio
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    tags: Set[str] = None

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0

class CacheLayer(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Set[str] = None) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self, tags: Optional[Set[str]] = None) -> int:
        pass

class MemoryCache(CacheLayer):
    def __init__(self, max_size_mb: int = 100, max_entries: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.size_bytes = 0

    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]

            # Check TTL
            if entry.ttl_seconds and (datetime.now() - entry.created_at).seconds > entry.ttl_seconds:
                await self.delete(key)
                self.stats.misses += 1
                return None

            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            # Move to end (LRU)
            self.cache.move_to_end(key)

            self.stats.hits += 1
            self._update_hit_rate()
            return entry.value

        self.stats.misses += 1
        self._update_hit_rate()
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Set[str] = None) -> bool:
        # Calculate size
        size = self._estimate_size(value)

        # Check if we need to evict
        while (self.size_bytes + size > self.max_size_bytes or
               len(self.cache) >= self.max_entries) and self.cache:
            await self._evict_lru()

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            size_bytes=size,
            ttl_seconds=ttl,
            tags=tags or set()
        )

        # Remove existing if present
        if key in self.cache:
            old_entry = self.cache[key]
            self.size_bytes -= old_entry.size_bytes

        self.cache[key] = entry
        self.size_bytes += size

        self.stats.entry_count = len(self.cache)
        self.stats.size_bytes = self.size_bytes

        return True

    async def delete(self, key: str) -> bool:
        if key in self.cache:
            entry = self.cache.pop(key)
            self.size_bytes -= entry.size_bytes
            self.stats.entry_count = len(self.cache)
            self.stats.size_bytes = self.size_bytes
            return True
        return False

    async def clear(self, tags: Optional[Set[str]] = None) -> int:
        if tags is None:
            count = len(self.cache)
            self.cache.clear()
            self.size_bytes = 0
            self.stats.entry_count = 0
            self.stats.size_bytes = 0
            return count

        # Clear by tags
        to_remove = []
        for key, entry in self.cache.items():
            if entry.tags and entry.tags & tags:
                to_remove.append(key)

        for key in to_remove:
            await self.delete(key)

        return len(to_remove)

    async def _evict_lru(self):
        if self.cache:
            key, entry = self.cache.popitem(last=False)  # Remove oldest
            self.size_bytes -= entry.size_bytes
            self.stats.evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(json.dumps(value, default=str).encode('utf-8'))
        except:
            return 1024  # Default estimate

    def _update_hit_rate(self):
        total = self.stats.hits + self.stats.misses
        self.stats.hit_rate = self.stats.hits / total if total > 0 else 0.0

class IntelligentCacheManager:
    def __init__(self):
        self.layers = {
            'memory': MemoryCache(max_size_mb=200, max_entries=2000),
            'entity_neighborhoods': MemoryCache(max_size_mb=100, max_entries=500),
            'query_results': MemoryCache(max_size_mb=150, max_entries=300),
            'embeddings': MemoryCache(max_size_mb=50, max_entries=1000)
        }

        self.query_patterns = defaultdict(int)
        self.entity_access_patterns = defaultdict(int)
        self.similarity_threshold = 0.85

    async def get_entity_neighborhood(self, entity: str, collection: str, hops: int = 1) -> Optional[Dict[str, Any]]:
        """Get cached entity neighborhood."""
        cache_key = self._entity_neighborhood_key(entity, collection, hops)

        result = await self.layers['entity_neighborhoods'].get(cache_key)
        if result:
            self.entity_access_patterns[entity] += 1

        return result

    async def set_entity_neighborhood(self, entity: str, collection: str, hops: int, neighborhood: Dict[str, Any], ttl: int = 3600):
        """Cache entity neighborhood."""
        cache_key = self._entity_neighborhood_key(entity, collection, hops)
        tags = {'entity', f'collection:{collection}', f'entity:{entity}'}

        await self.layers['entity_neighborhoods'].set(
            cache_key, neighborhood, ttl=ttl, tags=tags
        )

    async def get_query_result(self, query: str, collection: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached query result with similarity matching."""
        # Try exact match first
        exact_key = self._query_key(query, collection, context)
        result = await self.layers['query_results'].get(exact_key)

        if result:
            self.query_patterns[query] += 1
            return result

        # Try similarity matching
        similar_result = await self._find_similar_query_result(query, collection, context)
        if similar_result:
            self.query_patterns[query] += 1
            return similar_result

        return None

    async def set_query_result(self, query: str, collection: str, result: Dict[str, Any],
                             context: Dict[str, Any] = None, ttl: int = 1800):
        """Cache query result."""
        cache_key = self._query_key(query, collection, context)
        tags = {'query', f'collection:{collection}'}

        # Add query pattern tags
        query_type = self._classify_query_type(query)
        tags.add(f'query_type:{query_type}')

        await self.layers['query_results'].set(
            cache_key, result, ttl=ttl, tags=tags
        )

        self.query_patterns[query] += 1

    async def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding."""
        cache_key = self._embedding_key(text, model)
        return await self.layers['embeddings'].get(cache_key)

    async def set_embedding(self, text: str, model: str, embedding: List[float], ttl: int = 86400):
        """Cache embedding."""
        cache_key = self._embedding_key(text, model)
        tags = {'embedding', f'model:{model}'}

        await self.layers['embeddings'].set(
            cache_key, embedding, ttl=ttl, tags=tags
        )

    async def warm_cache(self, collection: str):
        """Warm cache with frequently accessed data."""
        # Warm popular entities
        popular_entities = sorted(
            self.entity_access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        for entity, _ in popular_entities:
            # Pre-load entity neighborhoods if not cached
            cache_key = self._entity_neighborhood_key(entity, collection, 1)
            if not await self.layers['entity_neighborhoods'].get(cache_key):
                # Would trigger actual neighborhood loading
                pass

        # Warm popular query patterns
        popular_queries = sorted(
            self.query_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Could pre-compute results for common query patterns

    async def invalidate_collection(self, collection: str):
        """Invalidate all cache entries for a collection."""
        tags = {f'collection:{collection}'}

        for layer in self.layers.values():
            await layer.clear(tags)

    async def invalidate_entity(self, entity: str):
        """Invalidate cache entries for a specific entity."""
        tags = {f'entity:{entity}'}

        for layer in self.layers.values():
            await layer.clear(tags)

    def get_cache_stats(self) -> Dict[str, CacheStats]:
        """Get cache statistics for all layers."""
        return {name: layer.stats for name, layer in self.layers.items()}

    async def _find_similar_query_result(self, query: str, collection: str,
                                       context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Find cached result for similar query."""
        # This would require storing query embeddings and doing similarity search
        # For now, implement simple keyword-based similarity

        query_words = set(query.lower().split())

        # Check recent queries for similarity
        for cached_query in list(self.query_patterns.keys())[-50:]:  # Check last 50 queries
            cached_words = set(cached_query.lower().split())

            # Calculate Jaccard similarity
            intersection = len(query_words & cached_words)
            union = len(query_words | cached_words)
            similarity = intersection / union if union > 0 else 0

            if similarity >= self.similarity_threshold:
                cache_key = self._query_key(cached_query, collection, context)
                result = await self.layers['query_results'].get(cache_key)
                if result:
                    return result

        return None

    def _entity_neighborhood_key(self, entity: str, collection: str, hops: int) -> str:
        """Generate cache key for entity neighborhood."""
        key_data = f"entity_neighborhood:{entity}:{collection}:{hops}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _query_key(self, query: str, collection: str, context: Dict[str, Any] = None) -> str:
        """Generate cache key for query."""
        context_str = json.dumps(context or {}, sort_keys=True)
        key_data = f"query:{query}:{collection}:{context_str}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _embedding_key(self, text: str, model: str) -> str:
        """Generate cache key for embedding."""
        key_data = f"embedding:{text}:{model}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for cache tagging."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['who', 'what', 'when', 'where']):
            return 'factual'
        elif any(word in query_lower for word in ['why', 'how', 'explain']):
            return 'analytical'
        elif any(word in query_lower for word in ['summarize', 'overview']):
            return 'summary'
        else:
            return 'general'
```

### 2. Cache-Aware Services

Update services to use intelligent caching:

```python
# packages/morag-graph/src/morag_graph/neo4j_service.py

class Neo4jService:
    def __init__(self, cache_manager: IntelligentCacheManager):
        # ... existing initialization
        self.cache = cache_manager

    async def get_entity_neighborhood(self, entity: str, collection: str, hops: int = 1) -> Dict[str, Any]:
        """Get entity neighborhood with caching."""

        # Try cache first
        cached_result = await self.cache.get_entity_neighborhood(entity, collection, hops)
        if cached_result:
            return cached_result

        # Compute neighborhood
        neighborhood = await self._compute_entity_neighborhood(entity, collection, hops)

        # Cache result
        await self.cache.set_entity_neighborhood(entity, collection, hops, neighborhood)

        return neighborhood

    async def _compute_entity_neighborhood(self, entity: str, collection: str, hops: int) -> Dict[str, Any]:
        """Compute entity neighborhood from database."""
        # ... existing neighborhood computation logic
        pass
```

```python
# packages/morag-embedding/src/morag_embedding/embedding_service.py

class EmbeddingService:
    def __init__(self, cache_manager: IntelligentCacheManager):
        # ... existing initialization
        self.cache = cache_manager

    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Get embedding with caching."""
        model = model or self.default_model

        # Try cache first
        cached_embedding = await self.cache.get_embedding(text, model)
        if cached_embedding:
            return cached_embedding

        # Compute embedding
        embedding = await self._compute_embedding(text, model)

        # Cache result
        await self.cache.set_embedding(text, model, embedding)

        return embedding
```

### 3. Cache Warming Service

Create `packages/morag-core/src/morag_core/caching/cache_warmer.py`:

```python
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from .cache_manager import IntelligentCacheManager

class CacheWarmer:
    def __init__(self, cache_manager: IntelligentCacheManager):
        self.cache = cache_manager
        self.warming_interval = 3600  # 1 hour
        self.is_running = False

    async def start_background_warming(self):
        """Start background cache warming."""
        self.is_running = True

        while self.is_running:
            try:
                await self._warm_popular_entities()
                await self._warm_common_queries()
                await self._cleanup_expired_entries()

                await asyncio.sleep(self.warming_interval)
            except Exception as e:
                print(f"Cache warming error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    def stop_background_warming(self):
        """Stop background cache warming."""
        self.is_running = False

    async def _warm_popular_entities(self):
        """Warm cache with popular entities."""
        # Get popular entities from access patterns
        popular_entities = sorted(
            self.cache.entity_access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]  # Top 50 entities

        for entity, access_count in popular_entities:
            # Check if entity neighborhood is cached
            # If not, trigger loading (would need access to graph service)
            pass

    async def _warm_common_queries(self):
        """Warm cache with common query patterns."""
        # Analyze query patterns and pre-compute common variations
        common_patterns = self._identify_common_patterns()

        for pattern in common_patterns:
            # Pre-compute results for common query variations
            pass

    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        # This would be handled by TTL in the cache layers
        pass

    def _identify_common_patterns(self) -> List[str]:
        """Identify common query patterns for warming."""
        # Analyze query_patterns to find common templates
        patterns = []

        # Group similar queries
        query_groups = {}
        for query in self.cache.query_patterns.keys():
            # Extract pattern (e.g., "What is X?" -> "What is {entity}?")
            pattern = self._extract_pattern(query)
            if pattern not in query_groups:
                query_groups[pattern] = []
            query_groups[pattern].append(query)

        # Return patterns with high frequency
        for pattern, queries in query_groups.items():
            if len(queries) >= 3:  # Pattern appears 3+ times
                patterns.append(pattern)

        return patterns

    def _extract_pattern(self, query: str) -> str:
        """Extract pattern from query."""
        # Simple pattern extraction - could be more sophisticated
        import re

        # Replace potential entity names with placeholder
        pattern = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', '{entity}', query)
        return pattern
```

## Integration Points

### 1. Update GraphTraversalAgent

```python
# packages/morag-reasoning/src/morag_reasoning/graph_traversal_agent.py

class GraphTraversalAgent:
    def __init__(self, cache_manager: IntelligentCacheManager, ...):
        # ... existing initialization
        self.cache = cache_manager

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query with intelligent caching."""
        collection = context.get('collection_name', 'default')

        # Try cache first
        cached_result = await self.cache.get_query_result(query, collection, context)
        if cached_result:
            cached_result['from_cache'] = True
            return cached_result

        # Process query normally
        result = await self._process_query_uncached(query, context)

        # Cache result
        await self.cache.set_query_result(query, collection, result, context)

        result['from_cache'] = False
        return result
```

### 2. Add Cache Management API

```python
# packages/morag-services/src/morag_services/api/cache_endpoints.py

from fastapi import APIRouter, HTTPException
from ..cache_manager import IntelligentCacheManager

router = APIRouter(prefix="/cache", tags=["cache"])

@router.get("/stats")
async def get_cache_stats():
    """Get cache statistics."""
    stats = cache_manager.get_cache_stats()
    return stats

@router.post("/warm/{collection_name}")
async def warm_cache(collection_name: str):
    """Warm cache for collection."""
    await cache_manager.warm_cache(collection_name)
    return {"message": f"Cache warming initiated for {collection_name}"}

@router.delete("/invalidate/collection/{collection_name}")
async def invalidate_collection(collection_name: str):
    """Invalidate cache for collection."""
    await cache_manager.invalidate_collection(collection_name)
    return {"message": f"Cache invalidated for {collection_name}"}

@router.delete("/invalidate/entity/{entity_name}")
async def invalidate_entity(entity_name: str):
    """Invalidate cache for entity."""
    await cache_manager.invalidate_entity(entity_name)
    return {"message": f"Cache invalidated for entity {entity_name}"}
```

## Configuration

```yaml
# caching.yml
caching:
  enabled: true

  memory_cache:
    max_size_mb: 200
    max_entries: 2000

  entity_neighborhoods:
    max_size_mb: 100
    max_entries: 500
    ttl_seconds: 3600

  query_results:
    max_size_mb: 150
    max_entries: 300
    ttl_seconds: 1800
    similarity_threshold: 0.85

  embeddings:
    max_size_mb: 50
    max_entries: 1000
    ttl_seconds: 86400

  warming:
    enabled: true
    interval_seconds: 3600
    popular_entities_count: 50
    common_patterns_threshold: 3
```

## Testing Strategy

```python
# tests/unit/test_caching.py
import pytest
from morag_core.caching.cache_manager import IntelligentCacheManager

class TestCaching:
    def setup_method(self):
        self.cache = IntelligentCacheManager()

    @pytest.mark.asyncio
    async def test_entity_neighborhood_caching(self):
        # Test entity neighborhood caching
        entity = "Tesla"
        collection = "test"
        neighborhood = {"entities": ["Elon Musk", "SpaceX"], "relations": []}

        # Cache neighborhood
        await self.cache.set_entity_neighborhood(entity, collection, 1, neighborhood)

        # Retrieve from cache
        cached = await self.cache.get_entity_neighborhood(entity, collection, 1)
        assert cached == neighborhood

    @pytest.mark.asyncio
    async def test_query_similarity_matching(self):
        # Test query similarity matching
        pass

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        # Test LRU eviction
        pass
```

## Monitoring

```python
cache_monitoring = {
    'hit_rates': {
        'entity_neighborhoods': 0.0,
        'query_results': 0.0,
        'embeddings': 0.0
    },
    'cache_sizes': {
        'memory_usage_mb': 0,
        'entry_counts': 0
    },
    'performance_impact': {
        'avg_query_time_cached': 0.0,
        'avg_query_time_uncached': 0.0,
        'cache_lookup_time': 0.0
    },
    'warming_effectiveness': {
        'entities_warmed': 0,
        'warm_cache_hit_rate': 0.0
    }
}
```

## Success Metrics

- **Latency Reduction**: 40-50% reduction in query processing time
- **Cache Hit Rate**: >70% for entity neighborhoods, >60% for queries
- **Memory Efficiency**: Optimal memory usage with minimal evictions
- **Warming Effectiveness**: >80% hit rate for warmed entities

## Future Enhancements

1. **Distributed Caching**: Redis/Memcached integration
2. **Predictive Warming**: ML-based cache warming
3. **Adaptive Policies**: Dynamic TTL and eviction policies
4. **Cross-Session Learning**: Persistent query pattern analysis
