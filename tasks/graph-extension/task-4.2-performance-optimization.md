# Task 4.2: Performance Optimization

**Phase**: 4 - Advanced Features  
**Priority**: High  
**Estimated Time**: 8-10 days total  
**Dependencies**: Task 4.1 (Multi-Hop Reasoning)

## Overview

This task implements comprehensive performance optimization strategies for the graph-augmented RAG system. It focuses on caching mechanisms, parallel processing, and system-wide optimizations to ensure the system can handle production workloads efficiently while maintaining response quality.

## Subtasks

### 4.2.1: Caching Strategy
**Estimated Time**: 4-5 days  
**Priority**: High

#### Implementation Steps

1. **Multi-Level Caching System**
   ```python
   # src/morag_cache/graph_cache.py
   from typing import Dict, Any, Optional, List, Union, Tuple
   from dataclasses import dataclass, field
   from datetime import datetime, timedelta
   import asyncio
   import hashlib
   import json
   import logging
   from abc import ABC, abstractmethod
   import redis.asyncio as redis
   from morag_graph.models import GraphPath, Entity, Relation
   
   @dataclass
   class CacheEntry:
       key: str
       value: Any
       created_at: datetime
       last_accessed: datetime
       access_count: int = 0
       ttl: Optional[timedelta] = None
       size_bytes: int = 0
       tags: List[str] = field(default_factory=list)
   
   @dataclass
   class CacheStats:
       hits: int = 0
       misses: int = 0
       evictions: int = 0
       total_size: int = 0
       entry_count: int = 0
       
       @property
       def hit_rate(self) -> float:
           total = self.hits + self.misses
           return self.hits / total if total > 0 else 0.0
   
   class CacheBackend(ABC):
       @abstractmethod
       async def get(self, key: str) -> Optional[Any]:
           pass
       
       @abstractmethod
       async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
           pass
       
       @abstractmethod
       async def delete(self, key: str) -> bool:
           pass
       
       @abstractmethod
       async def clear(self) -> bool:
           pass
   
   class MemoryCache(CacheBackend):
       def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
           self.max_size = max_size
           self.max_memory_bytes = max_memory_mb * 1024 * 1024
           self.cache: Dict[str, CacheEntry] = {}
           self.stats = CacheStats()
           self.logger = logging.getLogger(__name__)
       
       async def get(self, key: str) -> Optional[Any]:
           if key in self.cache:
               entry = self.cache[key]
               
               # Check TTL
               if entry.ttl and datetime.now() - entry.created_at > entry.ttl:
                   await self.delete(key)
                   self.stats.misses += 1
                   return None
               
               # Update access info
               entry.last_accessed = datetime.now()
               entry.access_count += 1
               self.stats.hits += 1
               return entry.value
           
           self.stats.misses += 1
           return None
       
       async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
           try:
               # Calculate size
               size_bytes = len(json.dumps(value, default=str).encode('utf-8'))
               
               # Check if we need to evict
               await self._ensure_capacity(size_bytes)
               
               # Create entry
               entry = CacheEntry(
                   key=key,
                   value=value,
                   created_at=datetime.now(),
                   last_accessed=datetime.now(),
                   ttl=timedelta(seconds=ttl) if ttl else None,
                   size_bytes=size_bytes
               )
               
               self.cache[key] = entry
               self.stats.total_size += size_bytes
               self.stats.entry_count += 1
               
               return True
           
           except Exception as e:
               self.logger.error(f"Error setting cache entry: {str(e)}")
               return False
       
       async def delete(self, key: str) -> bool:
           if key in self.cache:
               entry = self.cache.pop(key)
               self.stats.total_size -= entry.size_bytes
               self.stats.entry_count -= 1
               return True
           return False
       
       async def clear(self) -> bool:
           self.cache.clear()
           self.stats = CacheStats()
           return True
       
       async def _ensure_capacity(self, new_entry_size: int):
           """Ensure cache has capacity for new entry."""
           # Check memory limit
           while (self.stats.total_size + new_entry_size > self.max_memory_bytes or 
                  len(self.cache) >= self.max_size):
               await self._evict_lru()
       
       async def _evict_lru(self):
           """Evict least recently used entry."""
           if not self.cache:
               return
           
           # Find LRU entry
           lru_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k].last_accessed)
           
           await self.delete(lru_key)
           self.stats.evictions += 1
   
   class RedisCache(CacheBackend):
       def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "morag:"):
           self.redis_url = redis_url
           self.prefix = prefix
           self.redis_client: Optional[redis.Redis] = None
           self.logger = logging.getLogger(__name__)
       
       async def _get_client(self) -> redis.Redis:
           if self.redis_client is None:
               self.redis_client = redis.from_url(self.redis_url)
           return self.redis_client
       
       def _make_key(self, key: str) -> str:
           return f"{self.prefix}{key}"
       
       async def get(self, key: str) -> Optional[Any]:
           try:
               client = await self._get_client()
               data = await client.get(self._make_key(key))
               if data:
                   return json.loads(data)
               return None
           except Exception as e:
               self.logger.error(f"Redis get error: {str(e)}")
               return None
       
       async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
           try:
               client = await self._get_client()
               data = json.dumps(value, default=str)
               await client.set(self._make_key(key), data, ex=ttl)
               return True
           except Exception as e:
               self.logger.error(f"Redis set error: {str(e)}")
               return False
       
       async def delete(self, key: str) -> bool:
           try:
               client = await self._get_client()
               result = await client.delete(self._make_key(key))
               return result > 0
           except Exception as e:
               self.logger.error(f"Redis delete error: {str(e)}")
               return False
       
       async def clear(self) -> bool:
           try:
               client = await self._get_client()
               keys = await client.keys(f"{self.prefix}*")
               if keys:
                   await client.delete(*keys)
               return True
           except Exception as e:
               self.logger.error(f"Redis clear error: {str(e)}")
               return False
   
   class GraphCache:
       def __init__(
           self, 
           l1_cache: Optional[CacheBackend] = None,
           l2_cache: Optional[CacheBackend] = None,
           default_ttl: int = 3600  # 1 hour
       ):
           self.l1_cache = l1_cache or MemoryCache()
           self.l2_cache = l2_cache
           self.default_ttl = default_ttl
           self.logger = logging.getLogger(__name__)
           
           # Cache categories with different TTLs
           self.cache_configs = {
               "entity": {"ttl": 7200, "prefix": "ent:"},      # 2 hours
               "relation": {"ttl": 7200, "prefix": "rel:"},    # 2 hours
               "path": {"ttl": 1800, "prefix": "path:"},       # 30 minutes
               "query": {"ttl": 900, "prefix": "query:"},      # 15 minutes
               "vector": {"ttl": 3600, "prefix": "vec:"},      # 1 hour
               "graph_stats": {"ttl": 14400, "prefix": "stats:"}  # 4 hours
           }
       
       def _make_cache_key(self, category: str, identifier: str) -> str:
           """Create a cache key with category prefix."""
           config = self.cache_configs.get(category, {"prefix": "misc:"})
           prefix = config["prefix"]
           
           # Hash long identifiers
           if len(identifier) > 100:
               identifier = hashlib.md5(identifier.encode()).hexdigest()
           
           return f"{prefix}{identifier}"
       
       async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
           """Get cached entity information."""
           key = self._make_cache_key("entity", entity_id)
           return await self._get_with_fallback(key)
       
       async def set_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> bool:
           """Cache entity information."""
           key = self._make_cache_key("entity", entity_id)
           ttl = self.cache_configs["entity"]["ttl"]
           return await self._set_with_fallback(key, entity_data, ttl)
       
       async def get_relations(self, entity_id: str, relation_type: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
           """Get cached relations for an entity."""
           cache_id = f"{entity_id}:{relation_type}" if relation_type else entity_id
           key = self._make_cache_key("relation", cache_id)
           return await self._get_with_fallback(key)
       
       async def set_relations(self, entity_id: str, relations: List[Dict[str, Any]], relation_type: Optional[str] = None) -> bool:
           """Cache relations for an entity."""
           cache_id = f"{entity_id}:{relation_type}" if relation_type else entity_id
           key = self._make_cache_key("relation", cache_id)
           ttl = self.cache_configs["relation"]["ttl"]
           return await self._set_with_fallback(key, relations, ttl)
       
       async def get_paths(self, start_entity: str, end_entity: str, max_depth: int) -> Optional[List[GraphPath]]:
           """Get cached paths between entities."""
           path_id = f"{start_entity}->{end_entity}:d{max_depth}"
           key = self._make_cache_key("path", path_id)
           
           cached_data = await self._get_with_fallback(key)
           if cached_data:
               # Convert back to GraphPath objects
               return [GraphPath.from_dict(path_data) for path_data in cached_data]
           return None
       
       async def set_paths(self, start_entity: str, end_entity: str, max_depth: int, paths: List[GraphPath]) -> bool:
           """Cache paths between entities."""
           path_id = f"{start_entity}->{end_entity}:d{max_depth}"
           key = self._make_cache_key("path", path_id)
           ttl = self.cache_configs["path"]["ttl"]
           
           # Convert GraphPath objects to serializable format
           serializable_paths = [path.to_dict() for path in paths]
           return await self._set_with_fallback(key, serializable_paths, ttl)
       
       async def get_query_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
           """Get cached query result."""
           key = self._make_cache_key("query", query_hash)
           return await self._get_with_fallback(key)
       
       async def set_query_result(self, query_hash: str, result: Dict[str, Any]) -> bool:
           """Cache query result."""
           key = self._make_cache_key("query", query_hash)
           ttl = self.cache_configs["query"]["ttl"]
           return await self._set_with_fallback(key, result, ttl)
       
       async def get_vector_embeddings(self, text_hash: str) -> Optional[List[float]]:
           """Get cached vector embeddings."""
           key = self._make_cache_key("vector", text_hash)
           return await self._get_with_fallback(key)
       
       async def set_vector_embeddings(self, text_hash: str, embeddings: List[float]) -> bool:
           """Cache vector embeddings."""
           key = self._make_cache_key("vector", text_hash)
           ttl = self.cache_configs["vector"]["ttl"]
           return await self._set_with_fallback(key, embeddings, ttl)
       
       async def invalidate_entity(self, entity_id: str):
           """Invalidate all cache entries related to an entity."""
           # Invalidate entity cache
           entity_key = self._make_cache_key("entity", entity_id)
           await self.l1_cache.delete(entity_key)
           if self.l2_cache:
               await self.l2_cache.delete(entity_key)
           
           # Invalidate relation cache
           relation_key = self._make_cache_key("relation", entity_id)
           await self.l1_cache.delete(relation_key)
           if self.l2_cache:
               await self.l2_cache.delete(relation_key)
           
           self.logger.info(f"Invalidated cache for entity: {entity_id}")
       
       async def invalidate_by_pattern(self, pattern: str):
           """Invalidate cache entries matching a pattern."""
           # This is a simplified implementation
           # In production, you might want more sophisticated pattern matching
           if hasattr(self.l1_cache, 'cache'):
               keys_to_delete = [key for key in self.l1_cache.cache.keys() if pattern in key]
               for key in keys_to_delete:
                   await self.l1_cache.delete(key)
           
           self.logger.info(f"Invalidated cache entries matching pattern: {pattern}")
       
       async def _get_with_fallback(self, key: str) -> Optional[Any]:
           """Get from L1 cache, fallback to L2 if miss."""
           # Try L1 cache first
           result = await self.l1_cache.get(key)
           if result is not None:
               return result
           
           # Try L2 cache if available
           if self.l2_cache:
               result = await self.l2_cache.get(key)
               if result is not None:
                   # Populate L1 cache
                   await self.l1_cache.set(key, result, self.default_ttl)
                   return result
           
           return None
       
       async def _set_with_fallback(self, key: str, value: Any, ttl: int) -> bool:
           """Set in both L1 and L2 caches."""
           l1_success = await self.l1_cache.set(key, value, ttl)
           l2_success = True
           
           if self.l2_cache:
               l2_success = await self.l2_cache.set(key, value, ttl)
           
           return l1_success and l2_success
       
       async def get_stats(self) -> Dict[str, Any]:
           """Get cache statistics."""
           stats = {"l1_cache": {}}
           
           if hasattr(self.l1_cache, 'stats'):
               l1_stats = self.l1_cache.stats
               stats["l1_cache"] = {
                   "hits": l1_stats.hits,
                   "misses": l1_stats.misses,
                   "hit_rate": l1_stats.hit_rate,
                   "evictions": l1_stats.evictions,
                   "entry_count": l1_stats.entry_count,
                   "total_size_mb": l1_stats.total_size / (1024 * 1024)
               }
           
           return stats
   ```

#### Deliverables
- Multi-level caching system (L1: Memory, L2: Redis)
- Category-specific cache configurations
- Cache invalidation strategies
- Performance monitoring and statistics

### 4.2.2: Parallel Processing
**Estimated Time**: 4-5 days  
**Priority**: High

#### Implementation Steps

1. **Parallel Graph Operations**
   ```python
   # src/morag_parallel/graph_processor.py
   import asyncio
   import concurrent.futures
   from typing import List, Dict, Any, Optional, Callable, Awaitable
   from dataclasses import dataclass
   import logging
   import time
   from contextlib import asynccontextmanager
   
   @dataclass
   class ProcessingTask:
       task_id: str
       task_type: str
       data: Any
       priority: int = 1
       created_at: float = None
       
       def __post_init__(self):
           if self.created_at is None:
               self.created_at = time.time()
   
   @dataclass
   class ProcessingResult:
       task_id: str
       success: bool
       result: Any = None
       error: Optional[str] = None
       processing_time: float = 0.0
   
   class ResourceManager:
       def __init__(self, max_cpu_workers: int = 4, max_memory_mb: int = 2048):
           self.max_cpu_workers = max_cpu_workers
           self.max_memory_mb = max_memory_mb
           self.current_memory_usage = 0
           self.active_tasks = 0
           self.logger = logging.getLogger(__name__)
       
       @asynccontextmanager
       async def acquire_resources(self, estimated_memory_mb: int = 100):
           """Context manager for resource acquisition."""
           # Wait for available resources
           while (self.active_tasks >= self.max_cpu_workers or 
                  self.current_memory_usage + estimated_memory_mb > self.max_memory_mb):
               await asyncio.sleep(0.1)
           
           # Acquire resources
           self.active_tasks += 1
           self.current_memory_usage += estimated_memory_mb
           
           try:
               yield
           finally:
               # Release resources
               self.active_tasks -= 1
               self.current_memory_usage -= estimated_memory_mb
       
       def get_resource_status(self) -> Dict[str, Any]:
           return {
               "active_tasks": self.active_tasks,
               "max_workers": self.max_cpu_workers,
               "memory_usage_mb": self.current_memory_usage,
               "max_memory_mb": self.max_memory_mb,
               "cpu_utilization": self.active_tasks / self.max_cpu_workers,
               "memory_utilization": self.current_memory_usage / self.max_memory_mb
           }
   
   class ParallelGraphProcessor:
       def __init__(
           self, 
           graph_engine,
           max_workers: int = 8,
           batch_size: int = 50,
           max_memory_mb: int = 2048
       ):
           self.graph_engine = graph_engine
           self.max_workers = max_workers
           self.batch_size = batch_size
           self.resource_manager = ResourceManager(max_workers, max_memory_mb)
           self.task_queue = asyncio.Queue()
           self.result_queue = asyncio.Queue()
           self.workers = []
           self.running = False
           self.logger = logging.getLogger(__name__)
       
       async def start(self):
           """Start the parallel processing workers."""
           if self.running:
               return
           
           self.running = True
           self.workers = [
               asyncio.create_task(self._worker(f"worker-{i}"))
               for i in range(self.max_workers)
           ]
           
           self.logger.info(f"Started {self.max_workers} parallel workers")
       
       async def stop(self):
           """Stop all workers gracefully."""
           self.running = False
           
           # Cancel all workers
           for worker in self.workers:
               worker.cancel()
           
           # Wait for workers to finish
           await asyncio.gather(*self.workers, return_exceptions=True)
           self.workers.clear()
           
           self.logger.info("Stopped all parallel workers")
       
       async def _worker(self, worker_id: str):
           """Worker coroutine that processes tasks from the queue."""
           self.logger.info(f"Worker {worker_id} started")
           
           while self.running:
               try:
                   # Get task from queue with timeout
                   task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                   
                   # Process task with resource management
                   async with self.resource_manager.acquire_resources():
                       result = await self._process_task(task, worker_id)
                       await self.result_queue.put(result)
                   
                   self.task_queue.task_done()
               
               except asyncio.TimeoutError:
                   continue  # No task available, continue loop
               except Exception as e:
                   self.logger.error(f"Worker {worker_id} error: {str(e)}")
           
           self.logger.info(f"Worker {worker_id} stopped")
       
       async def _process_task(self, task: ProcessingTask, worker_id: str) -> ProcessingResult:
           """Process a single task."""
           start_time = time.time()
           
           try:
               if task.task_type == "entity_expansion":
                   result = await self._process_entity_expansion(task.data)
               elif task.task_type == "path_finding":
                   result = await self._process_path_finding(task.data)
               elif task.task_type == "relation_extraction":
                   result = await self._process_relation_extraction(task.data)
               elif task.task_type == "vector_search":
                   result = await self._process_vector_search(task.data)
               else:
                   raise ValueError(f"Unknown task type: {task.task_type}")
               
               processing_time = time.time() - start_time
               
               return ProcessingResult(
                   task_id=task.task_id,
                   success=True,
                   result=result,
                   processing_time=processing_time
               )
           
           except Exception as e:
               processing_time = time.time() - start_time
               self.logger.error(f"Task {task.task_id} failed in worker {worker_id}: {str(e)}")
               
               return ProcessingResult(
                   task_id=task.task_id,
                   success=False,
                   error=str(e),
                   processing_time=processing_time
               )
       
       async def process_batch(
           self, 
           tasks: List[ProcessingTask],
           timeout: Optional[float] = None
       ) -> List[ProcessingResult]:
           """Process a batch of tasks in parallel."""
           if not self.running:
               await self.start()
           
           # Add tasks to queue
           for task in tasks:
               await self.task_queue.put(task)
           
           # Collect results
           results = []
           start_time = time.time()
           
           while len(results) < len(tasks):
               try:
                   # Check timeout
                   if timeout and (time.time() - start_time) > timeout:
                   self.logger.warning(f"Batch processing timeout after {timeout}s")
                   break
               
               # Get result with timeout
               result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
               results.append(result)
               
           except asyncio.TimeoutError:
               continue
           
           self.logger.info(f"Processed batch of {len(results)}/{len(tasks)} tasks")
           return results
       
       async def process_entities_parallel(
           self, 
           entity_ids: List[str],
           expansion_depth: int = 2
       ) -> Dict[str, Any]:
           """Process multiple entities in parallel for expansion."""
           tasks = [
               ProcessingTask(
                   task_id=f"entity-{entity_id}",
                   task_type="entity_expansion",
                   data={"entity_id": entity_id, "depth": expansion_depth}
               )
               for entity_id in entity_ids
           ]
           
           results = await self.process_batch(tasks)
           
           # Combine results
           combined_entities = {}
           combined_relations = []
           
           for result in results:
               if result.success and result.result:
                   entities = result.result.get("entities", {})
                   relations = result.result.get("relations", [])
                   
                   combined_entities.update(entities)
                   combined_relations.extend(relations)
           
           return {
               "entities": combined_entities,
               "relations": combined_relations,
               "processing_stats": {
                   "total_tasks": len(tasks),
                   "successful_tasks": sum(1 for r in results if r.success),
                   "failed_tasks": sum(1 for r in results if not r.success),
                   "total_time": sum(r.processing_time for r in results)
               }
           }
       
       async def find_paths_parallel(
           self, 
           path_queries: List[Dict[str, Any]]
       ) -> List[Dict[str, Any]]:
           """Find multiple paths in parallel."""
           tasks = [
               ProcessingTask(
                   task_id=f"path-{i}",
                   task_type="path_finding",
                   data=query
               )
               for i, query in enumerate(path_queries)
           ]
           
           results = await self.process_batch(tasks)
           
           # Return results maintaining order
           path_results = []
           for result in results:
               if result.success:
                   path_results.append(result.result)
               else:
                   path_results.append({"error": result.error, "paths": []})
           
           return path_results
       
       async def _process_entity_expansion(self, data: Dict[str, Any]) -> Dict[str, Any]:
           """Process entity expansion task."""
           entity_id = data["entity_id"]
           depth = data.get("depth", 2)
           
           # Get entity details
           entity_info = await self.graph_engine.get_entity_details(entity_id)
           if not entity_info:
               return {"entities": {}, "relations": []}
           
           # Get related entities and relations
           relations = await self.graph_engine.get_entity_relations(
               entity_id, max_depth=depth
           )
           
           # Collect all related entities
           related_entities = {}
           for relation in relations:
               for entity_key in ["subject", "object"]:
                   if entity_key in relation:
                       related_id = relation[entity_key]
                       if related_id not in related_entities:
                           related_info = await self.graph_engine.get_entity_details(related_id)
                           if related_info:
                               related_entities[related_id] = related_info
           
           return {
               "entities": {entity_id: entity_info, **related_entities},
               "relations": relations
           }
       
       async def _process_path_finding(self, data: Dict[str, Any]) -> Dict[str, Any]:
           """Process path finding task."""
           start_entity = data["start_entity"]
           end_entity = data.get("end_entity")
           max_depth = data.get("max_depth", 4)
           algorithm = data.get("algorithm", "bfs")
           
           if end_entity:
               # Find specific path
               paths = await self.graph_engine.find_paths(
                   start_entity, end_entity, max_depth, algorithm
               )
           else:
               # General traversal
               result = await self.graph_engine.traverse(
                   start_entity, algorithm, max_depth
               )
               paths = result.get("paths", [])
           
           return {
               "start_entity": start_entity,
               "end_entity": end_entity,
               "paths": paths,
               "path_count": len(paths)
           }
       
       async def _process_relation_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
           """Process relation extraction task."""
           # This would integrate with the relation extraction pipeline
           # For now, return placeholder
           return {"relations": [], "entities": {}}
       
       async def _process_vector_search(self, data: Dict[str, Any]) -> Dict[str, Any]:
           """Process vector search task."""
           # This would integrate with vector search
           # For now, return placeholder
           return {"results": [], "scores": []}
       
       async def get_performance_stats(self) -> Dict[str, Any]:
           """Get performance statistics."""
           resource_status = self.resource_manager.get_resource_status()
           
           return {
               "workers": {
                   "total": self.max_workers,
                   "active": len([w for w in self.workers if not w.done()]),
                   "running": self.running
               },
               "queues": {
                   "task_queue_size": self.task_queue.qsize(),
                   "result_queue_size": self.result_queue.qsize()
               },
               "resources": resource_status,
               "batch_size": self.batch_size
           }
   ```

#### Deliverables
- Parallel graph processing system
- Resource management and throttling
- Batch processing capabilities
- Performance monitoring and statistics

## Testing Requirements

### Unit Tests
```python
# tests/test_performance_optimization.py
import pytest
import asyncio
from morag_cache.graph_cache import GraphCache, MemoryCache, RedisCache
from morag_parallel.graph_processor import ParallelGraphProcessor, ProcessingTask

class TestGraphCache:
    @pytest.mark.asyncio
    async def test_memory_cache_operations(self):
        cache = MemoryCache(max_size=100)
        
        # Test set and get
        await cache.set("test_key", {"data": "test_value"}, ttl=60)
        result = await cache.get("test_key")
        assert result == {"data": "test_value"}
        
        # Test cache miss
        result = await cache.get("nonexistent_key")
        assert result is None
        
        # Test cache stats
        assert hasattr(cache, 'stats')
        assert cache.stats.hits > 0
        assert cache.stats.misses > 0
    
    @pytest.mark.asyncio
    async def test_graph_cache_entity_operations(self):
        graph_cache = GraphCache()
        
        entity_data = {
            "id": "test_entity",
            "type": "ORG",
            "properties": {"name": "Test Organization"}
        }
        
        # Test entity caching
        await graph_cache.set_entity("test_entity", entity_data)
        cached_entity = await graph_cache.get_entity("test_entity")
        assert cached_entity == entity_data
        
        # Test cache invalidation
        await graph_cache.invalidate_entity("test_entity")
        cached_entity = await graph_cache.get_entity("test_entity")
        assert cached_entity is None

class TestParallelGraphProcessor:
    @pytest.mark.asyncio
    async def test_parallel_processing_setup(self, mock_graph_engine):
        processor = ParallelGraphProcessor(
            graph_engine=mock_graph_engine,
            max_workers=4,
            batch_size=10
        )
        
        # Test start/stop
        await processor.start()
        assert processor.running
        assert len(processor.workers) == 4
        
        await processor.stop()
        assert not processor.running
        assert len(processor.workers) == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_graph_engine):
        processor = ParallelGraphProcessor(
            graph_engine=mock_graph_engine,
            max_workers=2
        )
        
        await processor.start()
        
        # Create test tasks
        tasks = [
            ProcessingTask(
                task_id=f"task-{i}",
                task_type="entity_expansion",
                data={"entity_id": f"entity-{i}", "depth": 1}
            )
            for i in range(5)
        ]
        
        # Process batch
        results = await processor.process_batch(tasks, timeout=10.0)
        
        assert len(results) == 5
        assert all(isinstance(result.task_id, str) for result in results)
        
        await processor.stop()
```

### Performance Tests
```python
# tests/performance/test_cache_performance.py
import pytest
import time
import asyncio
from morag_cache.graph_cache import GraphCache

class TestCachePerformance:
    @pytest.mark.asyncio
    async def test_cache_throughput(self):
        """Test cache operations per second."""
        cache = GraphCache()
        
        # Warm up
        for i in range(100):
            await cache.set_entity(f"entity-{i}", {"id": f"entity-{i}"})
        
        # Measure read performance
        start_time = time.time()
        for i in range(1000):
            await cache.get_entity(f"entity-{i % 100}")
        read_time = time.time() - start_time
        
        read_ops_per_sec = 1000 / read_time
        assert read_ops_per_sec > 1000  # Should handle 1000+ reads/sec
        
        # Measure write performance
        start_time = time.time()
        for i in range(500):
            await cache.set_entity(f"new-entity-{i}", {"id": f"new-entity-{i}"})
        write_time = time.time() - start_time
        
        write_ops_per_sec = 500 / write_time
        assert write_ops_per_sec > 500  # Should handle 500+ writes/sec

class TestParallelPerformance:
    @pytest.mark.asyncio
    async def test_parallel_speedup(self, mock_graph_engine):
        """Test that parallel processing provides speedup."""
        from morag_parallel.graph_processor import ParallelGraphProcessor, ProcessingTask
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for i in range(20):
            # Simulate processing time
            await asyncio.sleep(0.1)
            sequential_results.append(f"result-{i}")
        sequential_time = time.time() - start_time
        
        # Parallel processing
        processor = ParallelGraphProcessor(
            graph_engine=mock_graph_engine,
            max_workers=4
        )
        
        await processor.start()
        
        tasks = [
            ProcessingTask(
                task_id=f"task-{i}",
                task_type="entity_expansion",
                data={"entity_id": f"entity-{i}"}
            )
            for i in range(20)
        ]
        
        start_time = time.time()
        parallel_results = await processor.process_batch(tasks, timeout=30.0)
        parallel_time = time.time() - start_time
        
        await processor.stop()
        
        # Parallel should be significantly faster
        speedup = sequential_time / parallel_time
        assert speedup > 2.0  # Should be at least 2x faster
        assert len(parallel_results) == 20
```

## Success Criteria

- [ ] Multi-level caching system reduces database queries by 70%+
- [ ] Cache hit rate > 80% for frequently accessed entities
- [ ] Parallel processing achieves 3x+ speedup for batch operations
- [ ] Memory usage stays within configured limits
- [ ] Cache invalidation works correctly for data consistency
- [ ] System handles 1000+ concurrent requests
- [ ] Performance targets met across all optimization areas
- [ ] Unit test coverage > 90%
- [ ] Performance tests validate optimization effectiveness

## Performance Targets

- **Cache Operations**: > 1000 reads/sec, > 500 writes/sec
- **Parallel Processing**: 3x+ speedup for batch operations
- **Memory Usage**: < 2GB for large workloads
- **Response Time**: < 100ms for cached queries
- **Throughput**: > 1000 concurrent requests
- **Cache Hit Rate**: > 80% for entities, > 70% for paths

## Next Steps

After completing this task:
1. Proceed to **Task 4.3**: Monitoring & Analytics
2. Implement comprehensive performance monitoring
3. Fine-tune optimization parameters based on production data

## Dependencies

**Requires**:
- Task 4.1: Multi-Hop Reasoning
- Redis for L2 caching
- Async processing capabilities

**Enables**:
- Task 4.3: Monitoring & Analytics
- Production-ready performance
- Scalable system architecture