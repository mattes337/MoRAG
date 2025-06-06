# Task 4: Worker Communication Protocol

## Objective
Implement a robust communication protocol between the main server and remote workers, including heartbeat monitoring, health checking, connection management, and network interruption handling.

## Current State Analysis

### Existing Communication
- Workers communicate via Redis/Celery task queue
- No direct communication channel between server and workers
- No real-time status updates or health monitoring
- No mechanism for server to directly contact workers

### Remote Worker Requirements
- Real-time communication for status updates
- Health monitoring and failure detection
- Network interruption recovery
- Task progress reporting
- Emergency task cancellation

## Implementation Plan

### Step 1: Communication Models

#### 1.1 Create Communication Models
**File**: `packages/morag-core/src/morag_core/models/communication.py`

```python
"""Communication models for worker-server interaction."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import uuid

class MessageType(str, Enum):
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    TASK_CANCEL = "task_cancel"
    WORKER_SHUTDOWN = "worker_shutdown"
    SERVER_COMMAND = "server_command"

class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class WorkerMessage(BaseModel):
    """Message from worker to server."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None  # For request-response correlation

class ServerMessage(BaseModel):
    """Message from server to worker."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_worker_id: str
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    expires_at: Optional[datetime] = None

class TaskProgressUpdate(BaseModel):
    """Task progress update from worker."""
    task_id: str
    worker_id: str
    progress_percent: float
    stage: str
    message: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkerHealthStatus(BaseModel):
    """Worker health status information."""
    worker_id: str
    is_healthy: bool
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    active_tasks: int
    queue_size: int
    last_task_completion: Optional[datetime] = None
    error_count: int = 0
    warnings: List[str] = Field(default_factory=list)

class ConnectionStatus(BaseModel):
    """Connection status between worker and server."""
    worker_id: str
    is_connected: bool
    connection_quality: float  # 0.0 to 1.0
    latency_ms: float
    last_successful_ping: datetime
    reconnection_attempts: int = 0
    connection_errors: List[str] = Field(default_factory=list)
```

### Step 2: WebSocket Communication Service

#### 2.1 Create WebSocket Communication Service
**File**: `packages/morag/src/morag/services/communication.py`

```python
"""WebSocket-based communication service for worker-server interaction."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, Callable, Any
import websockets
from websockets.server import WebSocketServerProtocol
import structlog

from morag_core.models.communication import (
    WorkerMessage, ServerMessage, MessageType, MessagePriority,
    TaskProgressUpdate, WorkerHealthStatus, ConnectionStatus
)

logger = structlog.get_logger(__name__)

class CommunicationService:
    """WebSocket-based communication service."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.server = None
        self.connected_workers: Dict[str, WebSocketServerProtocol] = {}
        self.worker_connections: Dict[str, ConnectionStatus] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10   # seconds
        self._ping_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the WebSocket communication server."""
        logger.info("Starting communication service", host=self.host, port=self.port)
        
        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout
        )
        
        # Start ping monitoring task
        self._ping_task = asyncio.create_task(self._monitor_connections())
        
        logger.info("Communication service started")
    
    async def stop(self):
        """Stop the WebSocket communication server."""
        logger.info("Stopping communication service")
        
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all worker connections
        for websocket in self.connected_workers.values():
            await websocket.close()
        
        self.connected_workers.clear()
        self.worker_connections.clear()
        
        logger.info("Communication service stopped")
    
    def register_message_handler(self, message_type: MessageType, 
                                handler: Callable[[WorkerMessage], None]):
        """Register a handler for specific message types."""
        self.message_handlers[message_type] = handler
        logger.debug("Message handler registered", message_type=message_type.value)
    
    async def send_message_to_worker(self, worker_id: str, message: ServerMessage) -> bool:
        """Send a message to a specific worker."""
        try:
            if worker_id not in self.connected_workers:
                logger.warning("Worker not connected", worker_id=worker_id)
                return False
            
            websocket = self.connected_workers[worker_id]
            message_data = message.model_dump_json()
            
            await websocket.send(message_data)
            
            logger.debug("Message sent to worker",
                        worker_id=worker_id,
                        message_type=message.message_type.value,
                        message_id=message.message_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to send message to worker",
                        worker_id=worker_id,
                        error=str(e))
            return False
    
    async def broadcast_message(self, message: ServerMessage, 
                              exclude_workers: Set[str] = None) -> int:
        """Broadcast a message to all connected workers."""
        exclude_workers = exclude_workers or set()
        sent_count = 0
        
        for worker_id in list(self.connected_workers.keys()):
            if worker_id not in exclude_workers:
                if await self.send_message_to_worker(worker_id, message):
                    sent_count += 1
        
        logger.info("Message broadcasted",
                   message_type=message.message_type.value,
                   sent_count=sent_count,
                   total_workers=len(self.connected_workers))
        
        return sent_count
    
    def get_connected_workers(self) -> List[str]:
        """Get list of currently connected worker IDs."""
        return list(self.connected_workers.keys())
    
    def get_worker_connection_status(self, worker_id: str) -> Optional[ConnectionStatus]:
        """Get connection status for a specific worker."""
        return self.worker_connections.get(worker_id)
    
    def is_worker_connected(self, worker_id: str) -> bool:
        """Check if a worker is currently connected."""
        return worker_id in self.connected_workers
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection."""
        worker_id = None
        
        try:
            # Wait for worker identification
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30)
            auth_data = json.loads(auth_message)
            
            worker_id = auth_data.get('worker_id')
            auth_token = auth_data.get('auth_token')
            
            if not worker_id or not self._verify_worker_auth(worker_id, auth_token):
                await websocket.close(code=4001, reason="Authentication failed")
                return
            
            # Register worker connection
            self.connected_workers[worker_id] = websocket
            self.worker_connections[worker_id] = ConnectionStatus(
                worker_id=worker_id,
                is_connected=True,
                connection_quality=1.0,
                latency_ms=0.0,
                last_successful_ping=datetime.utcnow()
            )
            
            logger.info("Worker connected", worker_id=worker_id)
            
            # Send connection confirmation
            confirmation = ServerMessage(
                target_worker_id=worker_id,
                message_type=MessageType.SERVER_COMMAND,
                data={"command": "connection_confirmed"}
            )
            await self.send_message_to_worker(worker_id, confirmation)
            
            # Handle messages from this worker
            async for message_data in websocket:
                try:
                    message_dict = json.loads(message_data)
                    message = WorkerMessage(**message_dict)
                    
                    # Update connection status
                    if worker_id in self.worker_connections:
                        self.worker_connections[worker_id].last_successful_ping = datetime.utcnow()
                    
                    # Handle the message
                    await self._handle_worker_message(message)
                    
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received from worker", worker_id=worker_id)
                except Exception as e:
                    logger.error("Error handling worker message",
                               worker_id=worker_id,
                               error=str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Worker disconnected", worker_id=worker_id)
        except asyncio.TimeoutError:
            logger.warning("Worker authentication timeout")
        except Exception as e:
            logger.error("Connection handling error", error=str(e))
        finally:
            # Clean up worker connection
            if worker_id:
                self.connected_workers.pop(worker_id, None)
                if worker_id in self.worker_connections:
                    self.worker_connections[worker_id].is_connected = False
    
    async def _handle_worker_message(self, message: WorkerMessage):
        """Handle a message received from a worker."""
        logger.debug("Message received from worker",
                    worker_id=message.worker_id,
                    message_type=message.message_type.value,
                    message_id=message.message_id)
        
        # Call registered handler if available
        if message.message_type in self.message_handlers:
            try:
                await self.message_handlers[message.message_type](message)
            except Exception as e:
                logger.error("Message handler error",
                           message_type=message.message_type.value,
                           error=str(e))
        else:
            logger.warning("No handler for message type",
                          message_type=message.message_type.value)
    
    def _verify_worker_auth(self, worker_id: str, auth_token: str) -> bool:
        """Verify worker authentication token."""
        # TODO: Implement proper JWT token verification
        # For now, just check if token is provided
        return bool(auth_token)
    
    async def _monitor_connections(self):
        """Monitor worker connections and update status."""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                current_time = datetime.utcnow()
                
                for worker_id, connection_status in list(self.worker_connections.items()):
                    if not connection_status.is_connected:
                        continue
                    
                    # Check if worker is still responsive
                    time_since_ping = current_time - connection_status.last_successful_ping
                    
                    if time_since_ping.total_seconds() > (self.ping_interval * 2):
                        # Worker seems unresponsive
                        logger.warning("Worker appears unresponsive",
                                     worker_id=worker_id,
                                     last_ping=connection_status.last_successful_ping)
                        
                        connection_status.connection_quality = max(0.0, 
                            connection_status.connection_quality - 0.2)
                        
                        if connection_status.connection_quality <= 0.1:
                            # Mark as disconnected
                            connection_status.is_connected = False
                            self.connected_workers.pop(worker_id, None)
                            logger.warning("Worker marked as disconnected due to unresponsiveness",
                                         worker_id=worker_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in connection monitoring", error=str(e))
```

### Step 3: Worker Communication Client

#### 3.1 Create Worker Communication Client
**File**: `packages/morag/src/morag/worker_communication.py`

```python
"""Worker-side communication client for server interaction."""

import asyncio
import json
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import websockets
from websockets.client import WebSocketClientProtocol
import structlog

from morag_core.models.communication import (
    WorkerMessage, ServerMessage, MessageType, MessagePriority,
    TaskProgressUpdate, WorkerHealthStatus
)

logger = structlog.get_logger(__name__)

class WorkerCommunicationClient:
    """Client for worker-server WebSocket communication."""
    
    def __init__(self, server_url: str, worker_id: str, auth_token: str):
        self.server_url = server_url
        self.worker_id = worker_id
        self.auth_token = auth_token
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.is_connected = False
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = 30  # seconds
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10
        
    async def connect(self) -> bool:
        """Connect to the server."""
        try:
            logger.info("Connecting to server", server_url=self.server_url)
            
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Send authentication
            auth_message = {
                "worker_id": self.worker_id,
                "auth_token": self.auth_token
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for confirmation
            response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
            response_data = json.loads(response)
            
            if response_data.get('data', {}).get('command') == 'connection_confirmed':
                self.is_connected = True
                logger.info("Connected to server successfully")
                
                # Start message handling tasks
                self._receive_task = asyncio.create_task(self._receive_messages())
                self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
                
                return True
            else:
                logger.error("Server connection confirmation failed")
                return False
                
        except Exception as e:
            logger.error("Failed to connect to server", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        logger.info("Disconnecting from server")
        
        self.is_connected = False
        
        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        # Send shutdown message
        if self.websocket:
            try:
                shutdown_message = WorkerMessage(
                    worker_id=self.worker_id,
                    message_type=MessageType.WORKER_SHUTDOWN
                )
                await self.send_message(shutdown_message)
                await self.websocket.close()
            except Exception:
                pass
        
        logger.info("Disconnected from server")
    
    async def send_message(self, message: WorkerMessage) -> bool:
        """Send a message to the server."""
        try:
            if not self.is_connected or not self.websocket:
                logger.warning("Cannot send message - not connected")
                return False
            
            message_data = message.model_dump_json()
            await self.websocket.send(message_data)
            
            logger.debug("Message sent to server",
                        message_type=message.message_type.value,
                        message_id=message.message_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to send message to server", error=str(e))
            return False
    
    async def send_task_progress(self, task_id: str, progress_percent: float,
                               stage: str, message: str = None) -> bool:
        """Send task progress update to server."""
        progress_update = TaskProgressUpdate(
            task_id=task_id,
            worker_id=self.worker_id,
            progress_percent=progress_percent,
            stage=stage,
            message=message
        )
        
        message = WorkerMessage(
            worker_id=self.worker_id,
            message_type=MessageType.TASK_PROGRESS,
            data=progress_update.model_dump()
        )
        
        return await self.send_message(message)
    
    async def send_health_status(self, health_status: WorkerHealthStatus) -> bool:
        """Send health status update to server."""
        message = WorkerMessage(
            worker_id=self.worker_id,
            message_type=MessageType.STATUS_UPDATE,
            data=health_status.model_dump()
        )
        
        return await self.send_message(message)
    
    def register_message_handler(self, message_type: MessageType,
                               handler: Callable[[ServerMessage], None]):
        """Register a handler for server messages."""
        self.message_handlers[message_type] = handler
    
    async def _receive_messages(self):
        """Receive and handle messages from server."""
        try:
            async for message_data in self.websocket:
                try:
                    message_dict = json.loads(message_data)
                    message = ServerMessage(**message_dict)
                    
                    logger.debug("Message received from server",
                               message_type=message.message_type.value,
                               message_id=message.message_id)
                    
                    # Handle the message
                    if message.message_type in self.message_handlers:
                        await self.message_handlers[message.message_type](message)
                    
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received from server")
                except Exception as e:
                    logger.error("Error handling server message", error=str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Server connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error("Error in message receiving", error=str(e))
            self.is_connected = False
    
    async def _send_heartbeats(self):
        """Send periodic heartbeats to server."""
        while self.is_connected:
            try:
                heartbeat = WorkerMessage(
                    worker_id=self.worker_id,
                    message_type=MessageType.HEARTBEAT,
                    data={"timestamp": datetime.utcnow().isoformat()}
                )
                
                await self.send_message(heartbeat)
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error sending heartbeat", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retry
```

## Testing Requirements

### Unit Tests
1. **Communication Service Tests**
   - Test WebSocket server startup/shutdown
   - Test worker connection handling
   - Test message routing and broadcasting
   - Test connection monitoring

2. **Worker Client Tests**
   - Test server connection/disconnection
   - Test message sending/receiving
   - Test heartbeat functionality
   - Test reconnection logic

### Integration Tests
1. **End-to-End Communication Tests**
   - Test worker-server message exchange
   - Test connection failure and recovery
   - Test multiple workers communication
   - Test message priority handling

### Test Files to Create
- `tests/test_communication_service.py`
- `tests/test_worker_communication.py`
- `tests/integration/test_worker_server_communication.py`

## Dependencies
- **New**: `websockets` for WebSocket communication
- **Existing**: JSON for message serialization

## Success Criteria
1. Workers can establish WebSocket connections with server
2. Real-time bidirectional communication works reliably
3. Connection failures are detected and handled gracefully
4. Message routing and broadcasting function correctly
5. Heartbeat monitoring maintains connection health
6. System handles network interruptions with automatic recovery

## Next Steps
After completing this task:
1. Proceed to Task 5: Task Classification System
2. Test communication with simulated remote workers
3. Validate connection recovery and error handling

---

**Dependencies**: Task 3 (File Transfer Service)
**Estimated Time**: 3-4 days
**Risk Level**: Medium (network communication complexity)
