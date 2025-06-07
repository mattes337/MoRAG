# Task 2: Worker Registration System

## Objective
Implement a worker registration system that allows workers to register their capabilities, location, and status with the main server, enabling dynamic task routing and worker management.

## Current State Analysis

### Existing Worker System
- Workers start independently without server coordination
- No central registry of worker capabilities or status
- No way to track which workers are available for specific task types
- Workers are identified only by Celery worker names

### Required Capabilities
- Worker capability registration (GPU, CPU, hybrid)
- Health monitoring and heartbeat system
- Dynamic worker discovery and status tracking
- Worker metadata (location, hardware specs, load)

## Implementation Plan

### Step 1: Worker Registry Data Model

#### 1.1 Create Worker Registry Models
**File**: `packages/morag-core/src/morag_core/models/worker.py`

```python
"""Worker registration and management models."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid

class WorkerCapability(str, Enum):
    CPU = "cpu"
    GPU = "gpu" 
    HYBRID = "hybrid"

class WorkerStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class HardwareInfo(BaseModel):
    """Hardware information for a worker."""
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    gpu_models: List[str] = Field(default_factory=list)
    disk_space_gb: float = 0.0

class WorkerLocation(BaseModel):
    """Worker location and network information."""
    hostname: str
    ip_address: str
    port: int = 8000
    is_remote: bool = True
    network_latency_ms: Optional[float] = None

class WorkerMetrics(BaseModel):
    """Current worker performance metrics."""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration_seconds: float = 0.0

class WorkerRegistration(BaseModel):
    """Worker registration information."""
    worker_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    capability: WorkerCapability
    hardware: HardwareInfo
    location: WorkerLocation
    max_concurrent_tasks: int = 1
    supported_content_types: List[str] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)
    
class WorkerInfo(BaseModel):
    """Complete worker information including status and metrics."""
    registration: WorkerRegistration
    status: WorkerStatus = WorkerStatus.OFFLINE
    metrics: WorkerMetrics = Field(default_factory=WorkerMetrics)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return (self.status == WorkerStatus.ONLINE and 
                self.metrics.active_tasks < self.registration.max_concurrent_tasks)
    
    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy (recent heartbeat)."""
        time_since_heartbeat = datetime.utcnow() - self.last_heartbeat
        return time_since_heartbeat.total_seconds() < 60  # 1 minute threshold

class WorkerRegistrationRequest(BaseModel):
    """Request to register a new worker."""
    registration: WorkerRegistration
    auth_token: Optional[str] = None

class WorkerHeartbeat(BaseModel):
    """Worker heartbeat with current status and metrics."""
    worker_id: str
    status: WorkerStatus
    metrics: WorkerMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### Step 2: Worker Registry Service

#### 2.1 Create Worker Registry Service
**File**: `packages/morag/src/morag/services/worker_registry.py`

```python
"""Worker registry service for managing worker registration and status."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import structlog
from redis import Redis

from morag_core.models.worker import (
    WorkerInfo, WorkerRegistration, WorkerHeartbeat, 
    WorkerCapability, WorkerStatus
)

logger = structlog.get_logger(__name__)

class WorkerRegistry:
    """Central registry for managing workers."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.workers: Dict[str, WorkerInfo] = {}
        self._heartbeat_timeout = 60  # seconds
        self._cleanup_interval = 30   # seconds
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the worker registry service."""
        logger.info("Starting worker registry service")
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_workers())
    
    async def stop(self):
        """Stop the worker registry service."""
        logger.info("Stopping worker registry service")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def register_worker(self, registration: WorkerRegistration) -> bool:
        """Register a new worker or update existing registration."""
        try:
            worker_info = WorkerInfo(
                registration=registration,
                status=WorkerStatus.ONLINE,
                registered_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            )
            
            self.workers[registration.worker_id] = worker_info
            
            # Store in Redis for persistence
            await self._store_worker_in_redis(worker_info)
            
            logger.info("Worker registered successfully",
                       worker_id=registration.worker_id,
                       name=registration.name,
                       capability=registration.capability,
                       location=f"{registration.location.hostname}:{registration.location.port}")
            
            return True
            
        except Exception as e:
            logger.error("Failed to register worker",
                        worker_id=registration.worker_id,
                        error=str(e))
            return False
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker."""
        try:
            if worker_id in self.workers:
                worker_info = self.workers[worker_id]
                worker_info.status = WorkerStatus.OFFLINE
                
                # Remove from active workers but keep in Redis for history
                del self.workers[worker_id]
                
                logger.info("Worker unregistered",
                           worker_id=worker_id,
                           name=worker_info.registration.name)
                return True
            else:
                logger.warning("Attempted to unregister unknown worker",
                              worker_id=worker_id)
                return False
                
        except Exception as e:
            logger.error("Failed to unregister worker",
                        worker_id=worker_id,
                        error=str(e))
            return False
    
    async def update_heartbeat(self, heartbeat: WorkerHeartbeat) -> bool:
        """Update worker heartbeat and metrics."""
        try:
            worker_id = heartbeat.worker_id
            
            if worker_id not in self.workers:
                logger.warning("Heartbeat from unknown worker",
                              worker_id=worker_id)
                return False
            
            worker_info = self.workers[worker_id]
            worker_info.status = heartbeat.status
            worker_info.metrics = heartbeat.metrics
            worker_info.last_heartbeat = heartbeat.timestamp
            
            # Update Redis
            await self._store_worker_in_redis(worker_info)
            
            logger.debug("Worker heartbeat updated",
                        worker_id=worker_id,
                        status=heartbeat.status.value,
                        active_tasks=heartbeat.metrics.active_tasks)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update worker heartbeat",
                        worker_id=heartbeat.worker_id,
                        error=str(e))
            return False
    
    def get_available_workers(self, capability: Optional[WorkerCapability] = None,
                            content_type: Optional[str] = None) -> List[WorkerInfo]:
        """Get list of available workers matching criteria."""
        available_workers = []
        
        for worker_info in self.workers.values():
            if not worker_info.is_available or not worker_info.is_healthy:
                continue
            
            # Filter by capability
            if capability and worker_info.registration.capability != capability:
                if not (capability == WorkerCapability.GPU and 
                       worker_info.registration.capability == WorkerCapability.HYBRID):
                    continue
            
            # Filter by content type
            if content_type and worker_info.registration.supported_content_types:
                if content_type not in worker_info.registration.supported_content_types:
                    continue
            
            available_workers.append(worker_info)
        
        # Sort by load (fewer active tasks first)
        available_workers.sort(key=lambda w: w.metrics.active_tasks)
        
        return available_workers
    
    def get_worker_info(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get information about a specific worker."""
        return self.workers.get(worker_id)
    
    def get_all_workers(self) -> List[WorkerInfo]:
        """Get information about all registered workers."""
        return list(self.workers.values())
    
    def get_worker_stats(self) -> Dict[str, int]:
        """Get statistics about registered workers."""
        stats = {
            'total': len(self.workers),
            'online': 0,
            'offline': 0,
            'busy': 0,
            'gpu_workers': 0,
            'cpu_workers': 0,
            'hybrid_workers': 0,
            'available': 0
        }
        
        for worker_info in self.workers.values():
            if worker_info.status == WorkerStatus.ONLINE:
                stats['online'] += 1
            elif worker_info.status == WorkerStatus.OFFLINE:
                stats['offline'] += 1
            elif worker_info.status == WorkerStatus.BUSY:
                stats['busy'] += 1
            
            if worker_info.registration.capability == WorkerCapability.GPU:
                stats['gpu_workers'] += 1
            elif worker_info.registration.capability == WorkerCapability.CPU:
                stats['cpu_workers'] += 1
            elif worker_info.registration.capability == WorkerCapability.HYBRID:
                stats['hybrid_workers'] += 1
            
            if worker_info.is_available:
                stats['available'] += 1
        
        return stats
    
    async def _store_worker_in_redis(self, worker_info: WorkerInfo):
        """Store worker information in Redis."""
        try:
            key = f"worker:{worker_info.registration.worker_id}"
            data = worker_info.model_dump_json()
            self.redis.setex(key, 3600, data)  # 1 hour TTL
        except Exception as e:
            logger.error("Failed to store worker in Redis",
                        worker_id=worker_info.registration.worker_id,
                        error=str(e))
    
    async def _cleanup_stale_workers(self):
        """Periodically clean up stale workers."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                current_time = datetime.utcnow()
                stale_workers = []
                
                for worker_id, worker_info in self.workers.items():
                    time_since_heartbeat = current_time - worker_info.last_heartbeat
                    
                    if time_since_heartbeat.total_seconds() > self._heartbeat_timeout:
                        stale_workers.append(worker_id)
                
                for worker_id in stale_workers:
                    worker_info = self.workers[worker_id]
                    worker_info.status = WorkerStatus.OFFLINE
                    
                    logger.warning("Worker marked as stale due to missing heartbeat",
                                  worker_id=worker_id,
                                  name=worker_info.registration.name,
                                  last_heartbeat=worker_info.last_heartbeat)
                    
                    del self.workers[worker_id]
                
                if stale_workers:
                    logger.info("Cleaned up stale workers", count=len(stale_workers))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in worker cleanup task", error=str(e))
```

### Step 3: Worker Registration API Endpoints

#### 3.1 Add Worker Management Endpoints
**File**: `packages/morag/src/morag/server.py`

Add new endpoints for worker management:

```python
from morag.services.worker_registry import WorkerRegistry
from morag_core.models.worker import (
    WorkerRegistrationRequest, WorkerHeartbeat, WorkerInfo
)

# Initialize worker registry
worker_registry = WorkerRegistry(redis_client)

@app.post("/api/v1/workers/register", response_model=Dict[str, Any])
async def register_worker(request: WorkerRegistrationRequest):
    """Register a new worker with the system."""
    try:
        success = await worker_registry.register_worker(request.registration)
        
        if success:
            return {
                "success": True,
                "message": "Worker registered successfully",
                "worker_id": request.registration.worker_id
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register worker")
            
    except Exception as e:
        logger.error("Worker registration failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str, heartbeat: WorkerHeartbeat):
    """Update worker heartbeat and status."""
    try:
        if heartbeat.worker_id != worker_id:
            raise HTTPException(status_code=400, detail="Worker ID mismatch")
        
        success = await worker_registry.update_heartbeat(heartbeat)
        
        if success:
            return {"success": True, "message": "Heartbeat updated"}
        else:
            raise HTTPException(status_code=404, detail="Worker not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Heartbeat update failed", worker_id=worker_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/workers/{worker_id}")
async def unregister_worker(worker_id: str):
    """Unregister a worker from the system."""
    try:
        success = await worker_registry.unregister_worker(worker_id)
        
        if success:
            return {"success": True, "message": "Worker unregistered"}
        else:
            raise HTTPException(status_code=404, detail="Worker not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Worker unregistration failed", worker_id=worker_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workers", response_model=List[WorkerInfo])
async def list_workers():
    """Get list of all registered workers."""
    return worker_registry.get_all_workers()

@app.get("/api/v1/workers/stats")
async def get_worker_stats():
    """Get worker statistics."""
    return worker_registry.get_worker_stats()

@app.get("/api/v1/workers/{worker_id}", response_model=WorkerInfo)
async def get_worker_info(worker_id: str):
    """Get information about a specific worker."""
    worker_info = worker_registry.get_worker_info(worker_id)
    if not worker_info:
        raise HTTPException(status_code=404, detail="Worker not found")
    return worker_info
```

### Step 4: Worker Client Implementation

#### 4.1 Create Worker Client
**File**: `packages/morag/src/morag/worker_client.py`

```python
"""Worker client for registering with the main server."""

import asyncio
import platform
import psutil
import socket
from datetime import datetime
from typing import Optional
import aiohttp
import structlog

from morag_core.models.worker import (
    WorkerRegistration, WorkerHeartbeat, WorkerCapability, 
    WorkerStatus, HardwareInfo, WorkerLocation, WorkerMetrics
)
from morag_core.utils.device import get_device_info

logger = structlog.get_logger(__name__)

class WorkerClient:
    """Client for worker registration and heartbeat management."""
    
    def __init__(self, server_url: str, worker_name: str, 
                 capability: WorkerCapability, auth_token: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.worker_name = worker_name
        self.capability = capability
        self.auth_token = auth_token
        self.worker_id: Optional[str] = None
        self.registration: Optional[WorkerRegistration] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = 30  # seconds
        
    async def register(self) -> bool:
        """Register this worker with the server."""
        try:
            # Gather hardware information
            hardware = await self._get_hardware_info()
            location = await self._get_location_info()
            
            # Create registration
            self.registration = WorkerRegistration(
                name=self.worker_name,
                capability=self.capability,
                hardware=hardware,
                location=location,
                supported_content_types=self._get_supported_content_types()
            )
            
            # Send registration request
            async with aiohttp.ClientSession() as session:
                url = f"{self.server_url}/api/v1/workers/register"
                data = {
                    "registration": self.registration.model_dump(),
                    "auth_token": self.auth_token
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.worker_id = result["worker_id"]
                        
                        logger.info("Worker registered successfully",
                                   worker_id=self.worker_id,
                                   server_url=self.server_url)
                        
                        # Start heartbeat
                        await self.start_heartbeat()
                        return True
                    else:
                        error_text = await response.text()
                        logger.error("Worker registration failed",
                                    status=response.status,
                                    error=error_text)
                        return False
                        
        except Exception as e:
            logger.error("Worker registration error", error=str(e))
            return False
    
    async def start_heartbeat(self):
        """Start sending periodic heartbeats."""
        if self._heartbeat_task:
            return
            
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat started", interval=self._heartbeat_interval)
    
    async def stop_heartbeat(self):
        """Stop sending heartbeats."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("Heartbeat stopped")
    
    async def unregister(self):
        """Unregister this worker from the server."""
        try:
            await self.stop_heartbeat()
            
            if self.worker_id:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.server_url}/api/v1/workers/{self.worker_id}"
                    async with session.delete(url) as response:
                        if response.status == 200:
                            logger.info("Worker unregistered successfully")
                        else:
                            logger.warning("Worker unregistration failed",
                                          status=response.status)
                            
        except Exception as e:
            logger.error("Worker unregistration error", error=str(e))
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to the server."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                await self._send_heartbeat()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _send_heartbeat(self):
        """Send a single heartbeat to the server."""
        if not self.worker_id:
            return
            
        try:
            metrics = await self._get_current_metrics()
            heartbeat = WorkerHeartbeat(
                worker_id=self.worker_id,
                status=WorkerStatus.ONLINE,
                metrics=metrics
            )
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.server_url}/api/v1/workers/{self.worker_id}/heartbeat"
                data = heartbeat.model_dump()
                
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        logger.warning("Heartbeat failed",
                                      status=response.status,
                                      worker_id=self.worker_id)
                        
        except Exception as e:
            logger.error("Failed to send heartbeat", error=str(e))
    
    async def _get_hardware_info(self) -> HardwareInfo:
        """Get current hardware information."""
        device_info = get_device_info()
        
        return HardwareInfo(
            cpu_cores=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_count=device_info.get('gpu_count', 0),
            gpu_memory_gb=device_info.get('gpu_memory_gb', 0.0),
            gpu_models=device_info.get('gpu_models', []),
            disk_space_gb=psutil.disk_usage('/').total / (1024**3)
        )
    
    async def _get_location_info(self) -> WorkerLocation:
        """Get worker location information."""
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "127.0.0.1"
            
        return WorkerLocation(
            hostname=hostname,
            ip_address=ip_address,
            is_remote=True  # Assume remote unless proven otherwise
        )
    
    async def _get_current_metrics(self) -> WorkerMetrics:
        """Get current worker performance metrics."""
        return WorkerMetrics(
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            memory_usage_percent=psutil.virtual_memory().percent,
            # GPU metrics would be added here if available
            active_tasks=0,  # This would be tracked by the worker
            completed_tasks=0,  # This would be tracked by the worker
            failed_tasks=0  # This would be tracked by the worker
        )
    
    def _get_supported_content_types(self) -> list:
        """Get list of content types this worker can handle."""
        if self.capability == WorkerCapability.GPU:
            return ['audio', 'video', 'image']
        elif self.capability == WorkerCapability.CPU:
            return ['document', 'web', 'text']
        else:  # HYBRID
            return ['audio', 'video', 'image', 'document', 'web', 'text']
```

## Testing Requirements

### Unit Tests
1. **Worker Registry Tests**
   - Test worker registration and unregistration
   - Test heartbeat updates and stale worker cleanup
   - Test worker filtering and availability checks

2. **Worker Client Tests**
   - Test registration process
   - Test heartbeat functionality
   - Test hardware detection

### Integration Tests
1. **API Endpoint Tests**
   - Test worker registration API
   - Test heartbeat API
   - Test worker listing and stats APIs

2. **End-to-End Tests**
   - Test complete worker lifecycle (register → heartbeat → unregister)
   - Test multiple workers with different capabilities
   - Test worker failover scenarios

### Test Files to Create
- `tests/test_worker_registry.py`
- `tests/test_worker_client.py`
- `tests/integration/test_worker_registration_api.py`

## Dependencies
- **New**: `aiohttp` for worker client HTTP requests
- **New**: `psutil` for hardware monitoring
- **Existing**: Redis for worker data persistence

## Success Criteria
1. Workers can successfully register with the server
2. Worker heartbeats are processed and tracked correctly
3. Stale workers are automatically cleaned up
4. Worker capabilities and status are accurately reported
5. API endpoints provide comprehensive worker management
6. System handles multiple workers with different capabilities

## Next Steps
After completing this task:
1. Proceed to Task 3: File Transfer Service
2. Test worker registration with simulated remote workers
3. Validate heartbeat and cleanup mechanisms

---

**Dependencies**: Task 1 (Queue Architecture)
**Estimated Time**: 3-4 days
**Risk Level**: Medium (new service integration)
