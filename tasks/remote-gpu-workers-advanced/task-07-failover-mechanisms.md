# Task 7: Failover Mechanisms

## Objective
Implement robust failover mechanisms that handle GPU worker unavailability, network failures, task timeouts, and automatic task redistribution to ensure system reliability and continuous operation.

## Current State Analysis

### Existing Failover Capabilities
- Basic Celery task retry mechanisms
- Queue fallback from Task 6 (GPU → CPU queues)
- Worker health monitoring from Task 2
- No automatic task redistribution on worker failure

### Required Enhancements
- Intelligent failover strategies based on failure type
- Automatic task redistribution when workers fail
- Network failure detection and recovery
- Task timeout handling with graceful degradation
- Circuit breaker patterns for unreliable workers

## Implementation Plan

### Step 1: Failover Strategy Models

#### 1.1 Create Failover Models
**File**: `packages/morag-core/src/morag_core/models/failover.py`

```python
"""Failover strategy models for robust task handling."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class FailureType(str, Enum):
    WORKER_OFFLINE = "worker_offline"
    WORKER_OVERLOADED = "worker_overloaded"
    NETWORK_FAILURE = "network_failure"
    TASK_TIMEOUT = "task_timeout"
    PROCESSING_ERROR = "processing_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    AUTHENTICATION_FAILURE = "authentication_failure"

class FailoverStrategy(str, Enum):
    RETRY_SAME_WORKER = "retry_same_worker"
    REASSIGN_SAME_QUEUE = "reassign_same_queue"
    FALLBACK_QUEUE = "fallback_queue"
    DEGRADE_QUALITY = "degrade_quality"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL_INTERVENTION = "manual_intervention"

class FailoverAction(BaseModel):
    """Action to take during failover."""
    strategy: FailoverStrategy
    target_worker_id: Optional[str] = None
    target_queue: Optional[str] = None
    retry_delay_seconds: int = 0
    max_retries: int = 3
    quality_degradation: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

class FailureEvent(BaseModel):
    """Record of a failure event."""
    event_id: str = Field(default_factory=lambda: f"fail_{int(datetime.utcnow().timestamp())}")
    task_id: str
    worker_id: Optional[str]
    failure_type: FailureType
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_strategy: Optional[FailoverStrategy] = None

class WorkerReliabilityScore(BaseModel):
    """Reliability score for a worker based on historical performance."""
    worker_id: str
    success_rate: float  # 0.0 to 1.0
    average_response_time: float  # seconds
    failure_count: int
    total_tasks: int
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[datetime] = None

class FailoverDecision(BaseModel):
    """Decision made by the failover system."""
    task_id: str
    failure_event: FailureEvent
    selected_action: FailoverAction
    alternative_actions: List[FailoverAction] = Field(default_factory=list)
    confidence: float  # 0.0 to 1.0
    estimated_delay: float  # minutes
    decision_metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Step 2: Failover Manager

#### 2.1 Create Failover Manager
**File**: `packages/morag/src/morag/services/failover_manager.py`

```python
"""Failover manager for handling task and worker failures."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import structlog
from redis import Redis

from morag_core.models.failover import (
    FailureType, FailoverStrategy, FailoverAction, FailureEvent,
    WorkerReliabilityScore, FailoverDecision
)
from morag_core.models.queue_management import QueueType, TaskQueueEntry
from morag.services.worker_registry import WorkerRegistry
from morag.services.priority_queue_manager import PriorityQueueManager

logger = structlog.get_logger(__name__)

class FailoverManager:
    """Manages failover strategies and task redistribution."""
    
    def __init__(self, redis_client: Redis, worker_registry: WorkerRegistry,
                 queue_manager: PriorityQueueManager):
        self.redis = redis_client
        self.worker_registry = worker_registry
        self.queue_manager = queue_manager
        
        # Failover configuration
        self.config = {
            'max_retries_per_task': 3,
            'circuit_breaker_threshold': 5,  # failures before circuit breaker
            'circuit_breaker_timeout': 300,  # seconds
            'worker_timeout_threshold': 180,  # seconds
            'network_retry_delay': 30,  # seconds
            'quality_degradation_enabled': True
        }
        
        # Internal state
        self.failure_events: Dict[str, FailureEvent] = {}
        self.worker_reliability: Dict[str, WorkerReliabilityScore] = {}
        self.active_circuit_breakers: Set[str] = set()
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the failover manager."""
        logger.info("Starting failover manager")
        
        # Load historical reliability data
        await self._load_reliability_data()
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Failover manager started")
    
    async def stop(self):
        """Stop the failover manager."""
        logger.info("Stopping failover manager")
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Save reliability data
        await self._save_reliability_data()
        
        logger.info("Failover manager stopped")
    
    async def handle_failure(self, task_id: str, worker_id: Optional[str],
                           failure_type: FailureType, error_message: str,
                           context: Dict = None) -> FailoverDecision:
        """Handle a task or worker failure."""
        try:
            # Create failure event
            failure_event = FailureEvent(
                task_id=task_id,
                worker_id=worker_id,
                failure_type=failure_type,
                error_message=error_message,
                context=context or {}
            )
            
            # Store failure event
            self.failure_events[failure_event.event_id] = failure_event
            
            # Update worker reliability if applicable
            if worker_id:
                await self._update_worker_reliability(worker_id, False)
            
            # Determine failover strategy
            failover_decision = await self._determine_failover_strategy(failure_event)
            
            # Execute failover action
            await self._execute_failover_action(failover_decision)
            
            logger.info("Failure handled",
                       task_id=task_id,
                       worker_id=worker_id,
                       failure_type=failure_type.value,
                       strategy=failover_decision.selected_action.strategy.value)
            
            return failover_decision
            
        except Exception as e:
            logger.error("Failed to handle failure",
                        task_id=task_id,
                        worker_id=worker_id,
                        error=str(e))
            raise
    
    async def handle_task_timeout(self, task_id: str, worker_id: str,
                                timeout_seconds: int) -> FailoverDecision:
        """Handle a task timeout."""
        return await self.handle_failure(
            task_id=task_id,
            worker_id=worker_id,
            failure_type=FailureType.TASK_TIMEOUT,
            error_message=f"Task timed out after {timeout_seconds} seconds",
            context={'timeout_seconds': timeout_seconds}
        )
    
    async def handle_worker_offline(self, worker_id: str, 
                                  affected_tasks: List[str]) -> List[FailoverDecision]:
        """Handle a worker going offline with multiple affected tasks."""
        decisions = []
        
        # Mark worker as unreliable
        await self._update_worker_reliability(worker_id, False)
        
        # Handle each affected task
        for task_id in affected_tasks:
            decision = await self.handle_failure(
                task_id=task_id,
                worker_id=worker_id,
                failure_type=FailureType.WORKER_OFFLINE,
                error_message=f"Worker {worker_id} went offline",
                context={'affected_tasks_count': len(affected_tasks)}
            )
            decisions.append(decision)
        
        logger.warning("Worker offline handled",
                      worker_id=worker_id,
                      affected_tasks=len(affected_tasks))
        
        return decisions
    
    async def check_circuit_breaker(self, worker_id: str) -> bool:
        """Check if circuit breaker is active for a worker."""
        if worker_id not in self.worker_reliability:
            return False
        
        reliability = self.worker_reliability[worker_id]
        
        if reliability.circuit_breaker_active:
            # Check if circuit breaker should be reset
            if (reliability.circuit_breaker_until and 
                datetime.utcnow() > reliability.circuit_breaker_until):
                
                reliability.circuit_breaker_active = False
                reliability.circuit_breaker_until = None
                reliability.consecutive_failures = 0
                self.active_circuit_breakers.discard(worker_id)
                
                logger.info("Circuit breaker reset", worker_id=worker_id)
                return False
            
            return True
        
        return False
    
    async def get_worker_reliability(self, worker_id: str) -> Optional[WorkerReliabilityScore]:
        """Get reliability score for a worker."""
        return self.worker_reliability.get(worker_id)
    
    async def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics for monitoring."""
        total_failures = len(self.failure_events)
        failure_by_type = {}
        recent_failures = 0
        
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for event in self.failure_events.values():
            # Count by type
            failure_type = event.failure_type.value
            failure_by_type[failure_type] = failure_by_type.get(failure_type, 0) + 1
            
            # Count recent failures
            if event.occurred_at > cutoff_time:
                recent_failures += 1
        
        return {
            'total_failures': total_failures,
            'recent_failures_24h': recent_failures,
            'failure_by_type': failure_by_type,
            'active_circuit_breakers': len(self.active_circuit_breakers),
            'workers_with_reliability_data': len(self.worker_reliability)
        }
    
    async def _determine_failover_strategy(self, failure_event: FailureEvent) -> FailoverDecision:
        """Determine the best failover strategy for a failure."""
        task_id = failure_event.task_id
        worker_id = failure_event.worker_id
        failure_type = failure_event.failure_type
        
        # Get task information
        # Note: This would integrate with the queue manager to get task details
        
        # Determine strategy based on failure type and context
        if failure_type == FailureType.WORKER_OFFLINE:
            if worker_id and await self.check_circuit_breaker(worker_id):
                action = FailoverAction(
                    strategy=FailoverStrategy.FALLBACK_QUEUE,
                    reason="Worker has active circuit breaker"
                )
            else:
                action = FailoverAction(
                    strategy=FailoverStrategy.REASSIGN_SAME_QUEUE,
                    reason="Worker offline - reassign to another worker"
                )
        
        elif failure_type == FailureType.TASK_TIMEOUT:
            # Check worker reliability
            if worker_id and worker_id in self.worker_reliability:
                reliability = self.worker_reliability[worker_id]
                if reliability.consecutive_failures >= 3:
                    action = FailoverAction(
                        strategy=FailoverStrategy.CIRCUIT_BREAKER,
                        reason="Multiple consecutive failures - activating circuit breaker"
                    )
                else:
                    action = FailoverAction(
                        strategy=FailoverStrategy.RETRY_SAME_WORKER,
                        retry_delay_seconds=60,
                        reason="Timeout - retry with delay"
                    )
            else:
                action = FailoverAction(
                    strategy=FailoverStrategy.REASSIGN_SAME_QUEUE,
                    reason="Timeout - reassign to different worker"
                )
        
        elif failure_type == FailureType.NETWORK_FAILURE:
            action = FailoverAction(
                strategy=FailoverStrategy.RETRY_SAME_WORKER,
                retry_delay_seconds=self.config['network_retry_delay'],
                max_retries=2,
                reason="Network failure - retry with delay"
            )
        
        elif failure_type == FailureType.RESOURCE_EXHAUSTION:
            if self.config['quality_degradation_enabled']:
                action = FailoverAction(
                    strategy=FailoverStrategy.DEGRADE_QUALITY,
                    quality_degradation={
                        'disable_diarization': True,
                        'reduce_quality': True
                    },
                    reason="Resource exhaustion - degrade quality and retry"
                )
            else:
                action = FailoverAction(
                    strategy=FailoverStrategy.FALLBACK_QUEUE,
                    reason="Resource exhaustion - fallback to CPU queue"
                )
        
        else:
            # Default strategy
            action = FailoverAction(
                strategy=FailoverStrategy.REASSIGN_SAME_QUEUE,
                reason=f"Default strategy for {failure_type.value}"
            )
        
        # Create decision
        decision = FailoverDecision(
            task_id=task_id,
            failure_event=failure_event,
            selected_action=action,
            confidence=0.8,  # Would be calculated based on various factors
            estimated_delay=action.retry_delay_seconds / 60.0,
            decision_metadata={
                'failure_count': len([e for e in self.failure_events.values() 
                                    if e.task_id == task_id]),
                'worker_reliability': (self.worker_reliability.get(worker_id).success_rate 
                                     if worker_id and worker_id in self.worker_reliability else None)
            }
        )
        
        return decision
    
    async def _execute_failover_action(self, decision: FailoverDecision):
        """Execute a failover action."""
        action = decision.selected_action
        task_id = decision.task_id
        
        try:
            if action.strategy == FailoverStrategy.RETRY_SAME_WORKER:
                # Schedule retry with delay
                if action.retry_delay_seconds > 0:
                    await asyncio.sleep(action.retry_delay_seconds)
                
                # Resubmit task to same worker
                # This would integrate with the queue manager
                logger.info("Retrying task on same worker",
                           task_id=task_id,
                           delay=action.retry_delay_seconds)
            
            elif action.strategy == FailoverStrategy.REASSIGN_SAME_QUEUE:
                # Find alternative worker in same queue
                # This would integrate with the queue manager
                logger.info("Reassigning task to different worker",
                           task_id=task_id)
            
            elif action.strategy == FailoverStrategy.FALLBACK_QUEUE:
                # Move task to fallback queue
                # This would integrate with the queue manager
                logger.info("Moving task to fallback queue",
                           task_id=task_id)
            
            elif action.strategy == FailoverStrategy.CIRCUIT_BREAKER:
                # Activate circuit breaker for worker
                worker_id = decision.failure_event.worker_id
                if worker_id:
                    await self._activate_circuit_breaker(worker_id)
                
                # Reassign task
                logger.info("Circuit breaker activated, reassigning task",
                           task_id=task_id,
                           worker_id=worker_id)
            
            elif action.strategy == FailoverStrategy.DEGRADE_QUALITY:
                # Resubmit with degraded quality settings
                logger.info("Retrying task with degraded quality",
                           task_id=task_id,
                           degradation=action.quality_degradation)
            
            # Mark failure event as resolved
            decision.failure_event.resolved = True
            decision.failure_event.resolved_at = datetime.utcnow()
            decision.failure_event.resolution_strategy = action.strategy
            
        except Exception as e:
            logger.error("Failed to execute failover action",
                        task_id=task_id,
                        strategy=action.strategy.value,
                        error=str(e))
            raise
    
    async def _update_worker_reliability(self, worker_id: str, success: bool):
        """Update worker reliability score."""
        if worker_id not in self.worker_reliability:
            self.worker_reliability[worker_id] = WorkerReliabilityScore(
                worker_id=worker_id,
                success_rate=1.0 if success else 0.0,
                average_response_time=0.0,
                failure_count=0 if success else 1,
                total_tasks=1,
                consecutive_failures=0 if success else 1
            )
        else:
            reliability = self.worker_reliability[worker_id]
            reliability.total_tasks += 1
            
            if success:
                reliability.consecutive_failures = 0
                # Update success rate
                success_count = int(reliability.success_rate * (reliability.total_tasks - 1)) + 1
                reliability.success_rate = success_count / reliability.total_tasks
            else:
                reliability.failure_count += 1
                reliability.consecutive_failures += 1
                reliability.last_failure = datetime.utcnow()
                
                # Update success rate
                success_count = int(reliability.success_rate * (reliability.total_tasks - 1))
                reliability.success_rate = success_count / reliability.total_tasks
                
                # Check if circuit breaker should be activated
                if (reliability.consecutive_failures >= self.config['circuit_breaker_threshold'] and
                    not reliability.circuit_breaker_active):
                    await self._activate_circuit_breaker(worker_id)
    
    async def _activate_circuit_breaker(self, worker_id: str):
        """Activate circuit breaker for a worker."""
        if worker_id not in self.worker_reliability:
            return
        
        reliability = self.worker_reliability[worker_id]
        reliability.circuit_breaker_active = True
        reliability.circuit_breaker_until = (datetime.utcnow() + 
                                            timedelta(seconds=self.config['circuit_breaker_timeout']))
        
        self.active_circuit_breakers.add(worker_id)
        
        logger.warning("Circuit breaker activated",
                      worker_id=worker_id,
                      consecutive_failures=reliability.consecutive_failures,
                      timeout_until=reliability.circuit_breaker_until)
    
    async def _monitoring_loop(self):
        """Periodic monitoring and cleanup."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up old failure events
                await self._cleanup_old_events()
                
                # Check circuit breakers
                await self._check_circuit_breakers()
                
                # Update statistics
                await self._update_statistics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
    
    async def _cleanup_old_events(self):
        """Clean up old failure events."""
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        old_events = [event_id for event_id, event in self.failure_events.items()
                     if event.occurred_at < cutoff_time]
        
        for event_id in old_events:
            del self.failure_events[event_id]
        
        if old_events:
            logger.debug("Cleaned up old failure events", count=len(old_events))
    
    async def _check_circuit_breakers(self):
        """Check and reset expired circuit breakers."""
        current_time = datetime.utcnow()
        
        for worker_id in list(self.active_circuit_breakers):
            reliability = self.worker_reliability.get(worker_id)
            if (reliability and reliability.circuit_breaker_until and
                current_time > reliability.circuit_breaker_until):
                
                reliability.circuit_breaker_active = False
                reliability.circuit_breaker_until = None
                reliability.consecutive_failures = 0
                self.active_circuit_breakers.discard(worker_id)
                
                logger.info("Circuit breaker automatically reset", worker_id=worker_id)
    
    async def _update_statistics(self):
        """Update failure statistics."""
        # Implementation for statistics updates
        pass
    
    async def _load_reliability_data(self):
        """Load worker reliability data from Redis."""
        # Implementation for loading persistent reliability data
        pass
    
    async def _save_reliability_data(self):
        """Save worker reliability data to Redis."""
        # Implementation for saving persistent reliability data
        pass
```

## Testing Requirements

### Unit Tests
1. **Failover Manager Tests**
   - Test failure event handling
   - Test failover strategy determination
   - Test circuit breaker activation/reset
   - Test worker reliability tracking

2. **Failover Strategy Tests**
   - Test different failure type handling
   - Test quality degradation logic
   - Test retry mechanisms
   - Test queue fallback logic

### Integration Tests
1. **End-to-End Failover Tests**
   - Test worker failure scenarios
   - Test network failure recovery
   - Test task timeout handling
   - Test circuit breaker behavior

### Test Files to Create
- `tests/test_failover_manager.py`
- `tests/test_failover_strategies.py`
- `tests/integration/test_failover_scenarios.py`

## Dependencies
- **Existing**: Worker registry from Task 2
- **Existing**: Priority queue manager from Task 6
- **Existing**: Redis for persistence

## Success Criteria
1. System handles worker failures gracefully with automatic task redistribution
2. Circuit breaker prevents repeated failures on unreliable workers
3. Network failures are detected and handled with appropriate retry logic
4. Task timeouts trigger intelligent failover strategies
5. Quality degradation provides fallback when resources are insufficient
6. Failure statistics provide visibility into system reliability

## Next Steps
After completing this task:
1. Proceed to Task 8: Remote Worker Package
2. Test failover mechanisms with simulated failures
3. Validate circuit breaker and retry logic

---

**Dependencies**: Task 6 (Priority Queue), Task 2 (Worker Registry)
**Estimated Time**: 4-5 days
**Risk Level**: High (critical system reliability component)
