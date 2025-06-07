# Task 6: Priority Queue Implementation

## Objective
Implement a sophisticated priority queue system that routes tasks to appropriate workers based on classification results, with dynamic load balancing, failover mechanisms, and real-time task redistribution.

## Current State Analysis

### Existing Queue System
- Basic Celery queues from Task 1 (gpu_priority, cpu_standard, gpu_fallback)
- Static queue assignment based on content type
- No dynamic load balancing or task redistribution
- No priority handling within queues

### Required Enhancements
- Dynamic task routing based on classification results
- Priority handling within queues
- Real-time load balancing across workers
- Automatic failover when workers become unavailable
- Task redistribution for optimal performance

## Implementation Plan

### Step 1: Enhanced Queue Management Models

#### 1.1 Create Queue Management Models
**File**: `packages/morag-core/src/morag_core/models/queue_management.py`

```python
"""Queue management models for priority-based task routing."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class QueuePriority(int, Enum):
    URGENT = 10      # Emergency tasks
    HIGH = 8         # GPU-required tasks
    NORMAL = 5       # Standard priority
    LOW = 3          # Background tasks
    BATCH = 1        # Batch processing

class QueueType(str, Enum):
    GPU_PRIORITY = "gpu_priority"
    GPU_FALLBACK = "gpu_fallback"
    CPU_STANDARD = "cpu_standard"
    CPU_BATCH = "cpu_batch"

class TaskQueueEntry(BaseModel):
    """Entry in a task queue with priority and metadata."""
    task_id: str
    queue_type: QueueType
    priority: QueuePriority
    worker_id: Optional[str] = None  # Assigned worker
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_duration: float  # minutes
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
class QueueStats(BaseModel):
    """Statistics for a specific queue."""
    queue_type: QueueType
    total_tasks: int
    pending_tasks: int
    processing_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_wait_time: float  # minutes
    average_processing_time: float  # minutes
    worker_count: int
    
class LoadBalancingDecision(BaseModel):
    """Decision about load balancing and task redistribution."""
    action: str  # "assign", "redistribute", "hold", "fallback"
    source_queue: QueueType
    target_queue: Optional[QueueType] = None
    worker_id: Optional[str] = None
    reason: str
    confidence: float  # 0.0 to 1.0
    estimated_improvement: float = 0.0  # Expected time savings in minutes
```

### Step 2: Priority Queue Manager

#### 2.1 Create Priority Queue Manager
**File**: `packages/morag/src/morag/services/priority_queue_manager.py`

```python
"""Priority queue manager for intelligent task routing and load balancing."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import structlog
from redis import Redis
from celery import Celery

from morag_core.models.queue_management import (
    QueueType, QueuePriority, TaskQueueEntry, QueueStats, LoadBalancingDecision
)
from morag_core.models.task_classification import TaskClassificationResult
from morag_core.models.worker import WorkerInfo
from morag.services.worker_registry import WorkerRegistry
from morag.services.task_classifier import IntelligentTaskClassifier

logger = structlog.get_logger(__name__)

class PriorityQueueManager:
    """Manages priority queues and intelligent task routing."""
    
    def __init__(self, celery_app: Celery, redis_client: Redis, 
                 worker_registry: WorkerRegistry, task_classifier: IntelligentTaskClassifier):
        self.celery_app = celery_app
        self.redis = redis_client
        self.worker_registry = worker_registry
        self.task_classifier = task_classifier
        
        # Queue configuration
        self.queue_configs = {
            QueueType.GPU_PRIORITY: {
                'max_tasks_per_worker': 2,
                'priority_weight': 1.0,
                'timeout_minutes': 120
            },
            QueueType.GPU_FALLBACK: {
                'max_tasks_per_worker': 3,
                'priority_weight': 0.7,
                'timeout_minutes': 180
            },
            QueueType.CPU_STANDARD: {
                'max_tasks_per_worker': 4,
                'priority_weight': 0.5,
                'timeout_minutes': 60
            },
            QueueType.CPU_BATCH: {
                'max_tasks_per_worker': 8,
                'priority_weight': 0.2,
                'timeout_minutes': 300
            }
        }
        
        # Internal state
        self.queue_entries: Dict[str, TaskQueueEntry] = {}
        self.worker_assignments: Dict[str, Set[str]] = {}  # worker_id -> task_ids
        self._load_balancing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the priority queue manager."""
        logger.info("Starting priority queue manager")
        
        # Load existing queue state from Redis
        await self._load_queue_state()
        
        # Start background tasks
        self._load_balancing_task = asyncio.create_task(self._load_balancing_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Priority queue manager started")
    
    async def stop(self):
        """Stop the priority queue manager."""
        logger.info("Stopping priority queue manager")
        
        # Cancel background tasks
        if self._load_balancing_task:
            self._load_balancing_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Save queue state to Redis
        await self._save_queue_state()
        
        logger.info("Priority queue manager stopped")
    
    async def submit_task(self, classification_result: TaskClassificationResult,
                         task_args: List, task_kwargs: Dict) -> str:
        """Submit a task to the appropriate priority queue."""
        try:
            routing = classification_result.routing_decision
            characteristics = classification_result.characteristics
            
            # Create queue entry
            queue_entry = TaskQueueEntry(
                task_id=classification_result.task_id,
                queue_type=QueueType(routing.selected_queue),
                priority=self._determine_priority(characteristics, routing),
                worker_id=routing.selected_worker_id,
                estimated_duration=routing.estimated_completion_time,
                resource_requirements={
                    'memory_gb': characteristics.memory_requirement_gb,
                    'disk_gb': characteristics.disk_space_requirement_gb,
                    'gpu_required': any(req.value.startswith('gpu') 
                                      for req in characteristics.resource_requirements)
                }
            )
            
            # Store queue entry
            self.queue_entries[classification_result.task_id] = queue_entry
            
            # Make load balancing decision
            lb_decision = await self._make_load_balancing_decision(queue_entry)
            
            # Apply load balancing decision
            final_queue, final_worker = await self._apply_load_balancing_decision(
                queue_entry, lb_decision
            )
            
            # Submit to Celery with appropriate routing
            celery_options = {
                'queue': final_queue.value,
                'priority': queue_entry.priority.value,
                'routing_key': final_queue.value
            }
            
            if final_worker:
                celery_options['routing_key'] = f"{final_queue.value}.{final_worker}"
            
            # Submit the actual task
            task = self.celery_app.send_task(
                'morag.worker.process_file_task',  # or appropriate task name
                args=task_args,
                kwargs=task_kwargs,
                **celery_options
            )
            
            # Update tracking
            if final_worker:
                if final_worker not in self.worker_assignments:
                    self.worker_assignments[final_worker] = set()
                self.worker_assignments[final_worker].add(classification_result.task_id)
            
            logger.info("Task submitted to priority queue",
                       task_id=classification_result.task_id,
                       queue=final_queue.value,
                       priority=queue_entry.priority.value,
                       worker=final_worker,
                       celery_task_id=task.id)
            
            return task.id
            
        except Exception as e:
            logger.error("Failed to submit task to priority queue",
                        task_id=classification_result.task_id,
                        error=str(e))
            raise
    
    async def task_completed(self, task_id: str, success: bool):
        """Mark a task as completed and update tracking."""
        try:
            if task_id not in self.queue_entries:
                logger.warning("Completed task not found in queue entries", task_id=task_id)
                return
            
            queue_entry = self.queue_entries[task_id]
            
            # Update worker assignments
            if queue_entry.worker_id:
                worker_tasks = self.worker_assignments.get(queue_entry.worker_id, set())
                worker_tasks.discard(task_id)
                if not worker_tasks:
                    self.worker_assignments.pop(queue_entry.worker_id, None)
            
            # Remove from queue entries
            del self.queue_entries[task_id]
            
            # Update statistics
            await self._update_queue_stats(queue_entry.queue_type, success)
            
            logger.info("Task completion tracked",
                       task_id=task_id,
                       success=success,
                       queue=queue_entry.queue_type.value,
                       worker=queue_entry.worker_id)
            
        except Exception as e:
            logger.error("Failed to track task completion",
                        task_id=task_id,
                        error=str(e))
    
    async def get_queue_stats(self) -> Dict[QueueType, QueueStats]:
        """Get statistics for all queues."""
        stats = {}
        
        for queue_type in QueueType:
            queue_tasks = [entry for entry in self.queue_entries.values() 
                          if entry.queue_type == queue_type]
            
            # Get worker count for this queue
            available_workers = self.worker_registry.get_available_workers()
            queue_workers = self._get_workers_for_queue(queue_type, available_workers)
            
            stats[queue_type] = QueueStats(
                queue_type=queue_type,
                total_tasks=len(queue_tasks),
                pending_tasks=len([t for t in queue_tasks if not t.worker_id]),
                processing_tasks=len([t for t in queue_tasks if t.worker_id]),
                completed_tasks=0,  # Would be tracked separately
                failed_tasks=0,     # Would be tracked separately
                average_wait_time=0.0,  # Would be calculated from historical data
                average_processing_time=0.0,  # Would be calculated from historical data
                worker_count=len(queue_workers)
            )
        
        return stats
    
    def _determine_priority(self, characteristics, routing) -> QueuePriority:
        """Determine task priority based on characteristics and routing."""
        # High priority for GPU-required tasks
        if any(req.value == 'gpu_required' for req in characteristics.resource_requirements):
            return QueuePriority.HIGH
        
        # Normal priority for GPU-preferred tasks
        if any(req.value == 'gpu_preferred' for req in characteristics.resource_requirements):
            return QueuePriority.NORMAL
        
        # Lower priority for CPU-only tasks
        if characteristics.complexity.value == 'low':
            return QueuePriority.LOW
        
        return QueuePriority.NORMAL
    
    async def _make_load_balancing_decision(self, queue_entry: TaskQueueEntry) -> LoadBalancingDecision:
        """Make a load balancing decision for a task."""
        try:
            # Get available workers
            available_workers = self.worker_registry.get_available_workers()
            queue_workers = self._get_workers_for_queue(queue_entry.queue_type, available_workers)
            
            if not queue_workers:
                # No workers available - use fallback
                fallback_queue = self._get_fallback_queue(queue_entry.queue_type)
                return LoadBalancingDecision(
                    action="fallback",
                    source_queue=queue_entry.queue_type,
                    target_queue=fallback_queue,
                    reason="No workers available for preferred queue",
                    confidence=0.8
                )
            
            # Check if assigned worker is still optimal
            if queue_entry.worker_id:
                assigned_worker = next((w for w in queue_workers 
                                      if w.registration.worker_id == queue_entry.worker_id), None)
                
                if assigned_worker and assigned_worker.is_available:
                    # Check current load
                    current_load = len(self.worker_assignments.get(queue_entry.worker_id, set()))
                    max_load = self.queue_configs[queue_entry.queue_type]['max_tasks_per_worker']
                    
                    if current_load < max_load:
                        return LoadBalancingDecision(
                            action="assign",
                            source_queue=queue_entry.queue_type,
                            worker_id=queue_entry.worker_id,
                            reason="Assigned worker is available and not overloaded",
                            confidence=0.9
                        )
            
            # Find best available worker
            best_worker = self._find_best_worker(queue_entry, queue_workers)
            
            if best_worker:
                return LoadBalancingDecision(
                    action="assign",
                    source_queue=queue_entry.queue_type,
                    worker_id=best_worker.registration.worker_id,
                    reason=f"Best available worker with load: {best_worker.metrics.active_tasks}",
                    confidence=0.8
                )
            
            # Hold in queue if no good options
            return LoadBalancingDecision(
                action="hold",
                source_queue=queue_entry.queue_type,
                reason="No suitable workers available - holding in queue",
                confidence=0.6
            )
            
        except Exception as e:
            logger.error("Load balancing decision failed", error=str(e))
            return LoadBalancingDecision(
                action="hold",
                source_queue=queue_entry.queue_type,
                reason=f"Error in load balancing: {str(e)}",
                confidence=0.1
            )
    
    async def _apply_load_balancing_decision(self, queue_entry: TaskQueueEntry,
                                           decision: LoadBalancingDecision) -> tuple:
        """Apply a load balancing decision and return final queue and worker."""
        if decision.action == "assign":
            queue_entry.worker_id = decision.worker_id
            return queue_entry.queue_type, decision.worker_id
        
        elif decision.action == "fallback":
            queue_entry.queue_type = decision.target_queue
            queue_entry.worker_id = None
            return decision.target_queue, None
        
        elif decision.action == "redistribute":
            queue_entry.queue_type = decision.target_queue
            queue_entry.worker_id = decision.worker_id
            return decision.target_queue, decision.worker_id
        
        else:  # hold
            return queue_entry.queue_type, queue_entry.worker_id
    
    def _get_workers_for_queue(self, queue_type: QueueType, 
                              available_workers: List[WorkerInfo]) -> List[WorkerInfo]:
        """Get workers that can handle tasks from a specific queue."""
        suitable_workers = []
        
        for worker in available_workers:
            if queue_type in [QueueType.GPU_PRIORITY, QueueType.GPU_FALLBACK]:
                if worker.registration.capability.value in ['gpu', 'hybrid']:
                    suitable_workers.append(worker)
            else:  # CPU queues
                if worker.registration.capability.value in ['cpu', 'hybrid']:
                    suitable_workers.append(worker)
        
        return suitable_workers
    
    def _find_best_worker(self, queue_entry: TaskQueueEntry, 
                         workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        """Find the best worker for a task based on current load and capabilities."""
        if not workers:
            return None
        
        # Score workers based on load and capability
        scored_workers = []
        
        for worker in workers:
            current_load = len(self.worker_assignments.get(worker.registration.worker_id, set()))
            max_load = self.queue_configs[queue_entry.queue_type]['max_tasks_per_worker']
            
            if current_load >= max_load:
                continue  # Worker is at capacity
            
            # Calculate score (lower is better)
            load_score = current_load / max_load
            capability_score = 0.0
            
            # Prefer GPU workers for GPU queues
            if queue_entry.queue_type in [QueueType.GPU_PRIORITY, QueueType.GPU_FALLBACK]:
                if worker.registration.capability.value == 'gpu':
                    capability_score = 0.0
                elif worker.registration.capability.value == 'hybrid':
                    capability_score = 0.2
                else:
                    capability_score = 1.0
            
            total_score = load_score + capability_score
            scored_workers.append((worker, total_score))
        
        if not scored_workers:
            return None
        
        # Return worker with lowest score
        scored_workers.sort(key=lambda x: x[1])
        return scored_workers[0][0]
    
    def _get_fallback_queue(self, queue_type: QueueType) -> QueueType:
        """Get fallback queue for a given queue type."""
        fallback_map = {
            QueueType.GPU_PRIORITY: QueueType.GPU_FALLBACK,
            QueueType.GPU_FALLBACK: QueueType.CPU_STANDARD,
            QueueType.CPU_STANDARD: QueueType.CPU_BATCH,
            QueueType.CPU_BATCH: QueueType.CPU_BATCH  # No fallback
        }
        return fallback_map.get(queue_type, QueueType.CPU_STANDARD)
    
    async def _load_balancing_loop(self):
        """Periodic load balancing and task redistribution."""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                await self._perform_load_balancing()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Load balancing loop error", error=str(e))
    
    async def _monitoring_loop(self):
        """Periodic monitoring and statistics updates."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._update_monitoring_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
    
    async def _perform_load_balancing(self):
        """Perform load balancing across workers and queues."""
        # Implementation for periodic load balancing
        pass
    
    async def _update_monitoring_stats(self):
        """Update monitoring statistics."""
        # Implementation for statistics updates
        pass
    
    async def _load_queue_state(self):
        """Load queue state from Redis."""
        # Implementation for loading persistent state
        pass
    
    async def _save_queue_state(self):
        """Save queue state to Redis."""
        # Implementation for saving persistent state
        pass
    
    async def _update_queue_stats(self, queue_type: QueueType, success: bool):
        """Update queue statistics."""
        # Implementation for statistics tracking
        pass
```

## Testing Requirements

### Unit Tests
1. **Priority Queue Manager Tests**
   - Test task submission and routing
   - Test load balancing decisions
   - Test worker assignment logic
   - Test fallback queue selection

2. **Queue Statistics Tests**
   - Test queue stats calculation
   - Test worker load tracking
   - Test task completion tracking

### Integration Tests
1. **End-to-End Queue Tests**
   - Test complete task routing workflow
   - Test load balancing with multiple workers
   - Test failover scenarios
   - Test priority handling

### Test Files to Create
- `tests/test_priority_queue_manager.py`
- `tests/test_load_balancing.py`
- `tests/integration/test_queue_routing_e2e.py`

## Dependencies
- **Existing**: Task classification from Task 5
- **Existing**: Worker registry from Task 2
- **Existing**: Celery queue infrastructure

## Success Criteria
1. Tasks are routed to optimal workers based on classification
2. Load balancing distributes tasks evenly across available workers
3. Failover mechanisms work when workers become unavailable
4. Priority handling ensures urgent tasks are processed first
5. Queue statistics provide visibility into system performance
6. System handles dynamic worker availability changes

## Next Steps
After completing this task:
1. Proceed to Task 7: Failover Mechanisms
2. Test priority queue with various load scenarios
3. Validate load balancing and failover behavior

---

**Dependencies**: Task 5 (Task Classification), Task 2 (Worker Registry)
**Estimated Time**: 3-4 days
**Risk Level**: Medium (complex queue management logic)
