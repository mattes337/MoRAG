# Task 8: Remote Worker Package

## Objective
Create a standalone remote worker package that can run on separate machines with GPU capabilities, handle file transfers, communicate with the main server, and process tasks independently.

## Current State Analysis

### Existing Worker System
- Workers run on same machine as server
- Shared volume access for file handling
- Local Celery worker processes
- No remote deployment capabilities

### Remote Worker Requirements
- Standalone deployment on remote machines
- GPU acceleration support
- Secure communication with main server
- File transfer capabilities
- Independent task processing
- Health monitoring and reporting

## Implementation Plan

### Step 1: Remote Worker Package Structure

#### 1.1 Create Remote Worker Package
**Directory Structure**:
```
packages/morag-remote-worker/
├── src/
│   └── morag_remote_worker/
│       ├── __init__.py
│       ├── worker.py
│       ├── client.py
│       ├── processor.py
│       ├── file_handler.py
│       └── config.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── docker-compose.yml
```

#### 1.2 Remote Worker Configuration
**File**: `packages/morag-remote-worker/src/morag_remote_worker/config.py`

```python
"""Configuration for remote worker."""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field

class RemoteWorkerConfig(BaseSettings):
    """Configuration for remote worker."""
    
    # Worker Identity
    worker_id: str = Field(default_factory=lambda: f"remote-worker-{os.getpid()}")
    worker_name: str = "Remote GPU Worker"
    worker_capability: str = "gpu"  # gpu, cpu, hybrid
    
    # Server Connection
    server_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8001"
    auth_token: str = ""
    
    # Redis/Celery Configuration
    redis_url: str = "redis://localhost:6379/0"
    celery_queues: List[str] = Field(default_factory=lambda: ["gpu_priority", "gpu_fallback"])
    
    # File Transfer
    transfer_encryption_key: str = "default-key-change-in-production"
    max_file_size_gb: float = 10.0
    temp_dir: str = "/tmp/morag-remote-worker"
    cleanup_interval_minutes: int = 60
    
    # Processing Configuration
    max_concurrent_tasks: int = 2
    task_timeout_minutes: int = 120
    enable_gpu: bool = True
    gpu_memory_limit_gb: Optional[float] = None
    
    # Health Monitoring
    heartbeat_interval_seconds: int = 30
    health_check_interval_seconds: int = 60
    max_consecutive_failures: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Audio Processing
    whisper_model_size: str = "base"
    enable_diarization: bool = True
    enable_topic_segmentation: bool = True
    
    # Video Processing
    enable_gpu_acceleration: bool = True
    thumbnail_quality: int = 85
    
    # Environment-specific overrides
    class Config:
        env_prefix = "MORAG_REMOTE_"
        env_file = ".env"
```

### Step 2: Remote Worker Core

#### 2.1 Create Remote Worker Main Class
**File**: `packages/morag-remote-worker/src/morag_remote_worker/worker.py`

```python
"""Remote worker implementation."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
import structlog
from celery import Celery

from morag_remote_worker.config import RemoteWorkerConfig
from morag_remote_worker.client import RemoteWorkerClient
from morag_remote_worker.processor import RemoteTaskProcessor
from morag_remote_worker.file_handler import RemoteFileHandler

logger = structlog.get_logger(__name__)

class RemoteWorker:
    """Remote worker that processes tasks from the main MoRAG server."""
    
    def __init__(self, config: RemoteWorkerConfig):
        self.config = config
        self.running = False
        
        # Initialize components
        self.client = RemoteWorkerClient(config)
        self.processor = RemoteTaskProcessor(config)
        self.file_handler = RemoteFileHandler(config)
        
        # Celery app for task processing
        self.celery_app = self._create_celery_app()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start(self):
        """Start the remote worker."""
        try:
            logger.info("Starting remote worker",
                       worker_id=self.config.worker_id,
                       server_url=self.config.server_url)
            
            # Create temp directory
            temp_dir = Path(self.config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Start components
            await self.file_handler.start()
            await self.processor.start()
            await self.client.start()
            
            # Register with server
            success = await self.client.register_worker()
            if not success:
                raise RuntimeError("Failed to register with server")
            
            # Start Celery worker
            self._start_celery_worker()
            
            self.running = True
            logger.info("Remote worker started successfully")
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error("Failed to start remote worker", error=str(e))
            raise
    
    async def stop(self):
        """Stop the remote worker."""
        logger.info("Stopping remote worker")
        self.running = False
        
        try:
            # Unregister from server
            await self.client.unregister_worker()
            
            # Stop components
            await self.client.stop()
            await self.processor.stop()
            await self.file_handler.stop()
            
            # Stop Celery worker
            self._stop_celery_worker()
            
            logger.info("Remote worker stopped")
            
        except Exception as e:
            logger.error("Error stopping remote worker", error=str(e))
    
    def _create_celery_app(self) -> Celery:
        """Create and configure Celery app."""
        app = Celery(f'remote_worker_{self.config.worker_id}')
        
        app.conf.update(
            broker_url=self.config.redis_url,
            result_backend=self.config.redis_url,
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=self.config.task_timeout_minutes * 60,
            task_soft_time_limit=self.config.task_timeout_minutes * 60 - 300,  # 5 min buffer
            worker_prefetch_multiplier=1,
            worker_max_tasks_per_child=100,
            broker_connection_retry_on_startup=True,
        )
        
        # Register task handlers
        self._register_task_handlers(app)
        
        return app
    
    def _register_task_handlers(self, app: Celery):
        """Register Celery task handlers."""
        
        @app.task(bind=True, name='remote.process_audio')
        async def process_audio_task(self, file_path: str, options: dict):
            return await self.processor.process_audio(file_path, options)
        
        @app.task(bind=True, name='remote.process_video')
        async def process_video_task(self, file_path: str, options: dict):
            return await self.processor.process_video(file_path, options)
        
        @app.task(bind=True, name='remote.process_image')
        async def process_image_task(self, file_path: str, options: dict):
            return await self.processor.process_image(file_path, options)
        
        @app.task(bind=True, name='remote.process_file')
        async def process_file_task(self, file_path: str, content_type: str, options: dict):
            return await self.processor.process_file(file_path, content_type, options)
    
    def _start_celery_worker(self):
        """Start Celery worker process."""
        # This would start Celery worker in a separate process/thread
        # For now, we'll use a simplified approach
        logger.info("Starting Celery worker",
                   queues=self.config.celery_queues,
                   concurrency=self.config.max_concurrent_tasks)
    
    def _stop_celery_worker(self):
        """Stop Celery worker process."""
        logger.info("Stopping Celery worker")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal", signal=signum)
        self.running = False

async def main():
    """Main entry point for remote worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoRAG Remote Worker")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--worker-id", help="Worker ID")
    parser.add_argument("--server-url", help="Server URL")
    parser.add_argument("--capability", choices=['gpu', 'cpu', 'hybrid'], 
                       default='gpu', help="Worker capability")
    
    args = parser.parse_args()
    
    # Load configuration
    config = RemoteWorkerConfig()
    
    if args.worker_id:
        config.worker_id = args.worker_id
    if args.server_url:
        config.server_url = args.server_url
    if args.capability:
        config.worker_capability = args.capability
    
    # Setup logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, config.log_level.upper(), structlog.stdlib.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        cache_logger_on_first_use=True,
    )
    
    # Create and start worker
    worker = RemoteWorker(config)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Worker failed", error=str(e))
        sys.exit(1)
    finally:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Remote Worker Client

#### 2.2 Create Remote Worker Client
**File**: `packages/morag-remote-worker/src/morag_remote_worker/client.py`

```python
"""Remote worker client for server communication."""

import asyncio
import platform
import psutil
from typing import Optional
import structlog
import aiohttp

from morag_core.models.worker import (
    WorkerRegistration, WorkerCapability, HardwareInfo, WorkerLocation
)
from morag.worker_communication import WorkerCommunicationClient
from morag_remote_worker.config import RemoteWorkerConfig

logger = structlog.get_logger(__name__)

class RemoteWorkerClient:
    """Client for remote worker server communication."""
    
    def __init__(self, config: RemoteWorkerConfig):
        self.config = config
        self.communication_client: Optional[WorkerCommunicationClient] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Start the client."""
        logger.info("Starting remote worker client")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Create WebSocket communication client
        self.communication_client = WorkerCommunicationClient(
            server_url=self.config.websocket_url,
            worker_id=self.config.worker_id,
            auth_token=self.config.auth_token
        )
        
        # Connect to server
        success = await self.communication_client.connect()
        if not success:
            raise RuntimeError("Failed to connect to server")
        
        logger.info("Remote worker client started")
    
    async def stop(self):
        """Stop the client."""
        logger.info("Stopping remote worker client")
        
        if self.communication_client:
            await self.communication_client.disconnect()
        
        if self.session:
            await self.session.close()
        
        logger.info("Remote worker client stopped")
    
    async def register_worker(self) -> bool:
        """Register this worker with the server."""
        try:
            # Gather hardware information
            hardware_info = await self._get_hardware_info()
            location_info = await self._get_location_info()
            
            # Create registration
            registration = WorkerRegistration(
                worker_id=self.config.worker_id,
                name=self.config.worker_name,
                capability=WorkerCapability(self.config.worker_capability),
                hardware=hardware_info,
                location=location_info,
                max_concurrent_tasks=self.config.max_concurrent_tasks,
                supported_content_types=self._get_supported_content_types()
            )
            
            # Send registration request
            url = f"{self.config.server_url}/api/v1/workers/register"
            data = {
                "registration": registration.model_dump(),
                "auth_token": self.config.auth_token
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("Worker registered successfully",
                               worker_id=result.get("worker_id"))
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
    
    async def unregister_worker(self) -> bool:
        """Unregister this worker from the server."""
        try:
            url = f"{self.config.server_url}/api/v1/workers/{self.config.worker_id}"
            
            async with self.session.delete(url) as response:
                if response.status == 200:
                    logger.info("Worker unregistered successfully")
                    return True
                else:
                    logger.warning("Worker unregistration failed",
                                  status=response.status)
                    return False
                    
        except Exception as e:
            logger.error("Worker unregistration error", error=str(e))
            return False
    
    async def download_file(self, transfer_id: str, auth_token: str, 
                          local_path: str) -> bool:
        """Download a file from the server."""
        try:
            url = f"{self.config.server_url}/api/v1/transfers/{transfer_id}/download"
            params = {"token": auth_token}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    with open(local_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    logger.info("File downloaded successfully",
                               transfer_id=transfer_id,
                               local_path=local_path)
                    return True
                else:
                    logger.error("File download failed",
                                transfer_id=transfer_id,
                                status=response.status)
                    return False
                    
        except Exception as e:
            logger.error("File download error",
                        transfer_id=transfer_id,
                        error=str(e))
            return False
    
    async def upload_file(self, transfer_id: str, auth_token: str,
                         file_path: str) -> bool:
        """Upload a file to the server."""
        try:
            url = f"{self.config.server_url}/api/v1/transfers/{transfer_id}/upload"
            params = {"token": auth_token}
            
            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=file_path)
                
                async with self.session.post(url, params=params, data=data) as response:
                    if response.status == 200:
                        logger.info("File uploaded successfully",
                                   transfer_id=transfer_id,
                                   file_path=file_path)
                        return True
                    else:
                        logger.error("File upload failed",
                                    transfer_id=transfer_id,
                                    status=response.status)
                        return False
                        
        except Exception as e:
            logger.error("File upload error",
                        transfer_id=transfer_id,
                        error=str(e))
            return False
    
    async def _get_hardware_info(self) -> HardwareInfo:
        """Get hardware information for this worker."""
        # Get GPU information
        gpu_count = 0
        gpu_memory_gb = 0.0
        gpu_models = []
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_models.append(torch.cuda.get_device_name(i))
                    gpu_memory_gb += torch.cuda.get_device_properties(i).total_memory / (1024**3)
        except ImportError:
            logger.warning("PyTorch not available - no GPU detection")
        
        return HardwareInfo(
            cpu_cores=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            gpu_models=gpu_models,
            disk_space_gb=psutil.disk_usage('/').total / (1024**3)
        )
    
    async def _get_location_info(self) -> WorkerLocation:
        """Get location information for this worker."""
        import socket
        
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "127.0.0.1"
        
        return WorkerLocation(
            hostname=hostname,
            ip_address=ip_address,
            is_remote=True
        )
    
    def _get_supported_content_types(self) -> list:
        """Get list of content types this worker can handle."""
        if self.config.worker_capability == 'gpu':
            return ['audio', 'video', 'image']
        elif self.config.worker_capability == 'cpu':
            return ['document', 'web', 'text']
        else:  # hybrid
            return ['audio', 'video', 'image', 'document', 'web', 'text']
```

### Step 4: Remote Task Processor

#### 2.3 Create Remote Task Processor
**File**: `packages/morag-remote-worker/src/morag_remote_worker/processor.py`

```python
"""Remote task processor for handling different content types."""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

from morag_audio import AudioProcessor, AudioConfig
from morag_video import VideoProcessor, VideoConfig
from morag_image import ImageProcessor, ImageConfig
from morag_core.models import ProcessingResult
from morag_remote_worker.config import RemoteWorkerConfig

logger = structlog.get_logger(__name__)

class RemoteTaskProcessor:
    """Processes tasks on remote worker."""
    
    def __init__(self, config: RemoteWorkerConfig):
        self.config = config
        self.audio_processor: Optional[AudioProcessor] = None
        self.video_processor: Optional[VideoProcessor] = None
        self.image_processor: Optional[ImageProcessor] = None
        
    async def start(self):
        """Start the task processor."""
        logger.info("Starting remote task processor")
        
        # Initialize processors based on worker capability
        if self.config.worker_capability in ['gpu', 'hybrid']:
            await self._initialize_gpu_processors()
        
        if self.config.worker_capability in ['cpu', 'hybrid']:
            await self._initialize_cpu_processors()
        
        logger.info("Remote task processor started")
    
    async def stop(self):
        """Stop the task processor."""
        logger.info("Stopping remote task processor")
        
        # Cleanup processors
        if self.audio_processor:
            # Cleanup if needed
            pass
        
        logger.info("Remote task processor stopped")
    
    async def process_file(self, file_path: str, content_type: str, 
                          options: Dict[str, Any]) -> ProcessingResult:
        """Process a file based on its content type."""
        try:
            logger.info("Processing file",
                       file_path=file_path,
                       content_type=content_type)
            
            if content_type == 'audio':
                return await self.process_audio(file_path, options)
            elif content_type == 'video':
                return await self.process_video(file_path, options)
            elif content_type == 'image':
                return await self.process_image(file_path, options)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
                
        except Exception as e:
            logger.error("File processing failed",
                        file_path=file_path,
                        content_type=content_type,
                        error=str(e))
            raise
    
    async def process_audio(self, file_path: str, options: Dict[str, Any]) -> ProcessingResult:
        """Process audio file."""
        if not self.audio_processor:
            raise RuntimeError("Audio processor not initialized")
        
        try:
            result = await self.audio_processor.process_file(Path(file_path))
            
            logger.info("Audio processing completed",
                       file_path=file_path,
                       duration=result.processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Audio processing failed",
                        file_path=file_path,
                        error=str(e))
            raise
    
    async def process_video(self, file_path: str, options: Dict[str, Any]) -> ProcessingResult:
        """Process video file."""
        if not self.video_processor:
            raise RuntimeError("Video processor not initialized")
        
        try:
            result = await self.video_processor.process_file(Path(file_path))
            
            logger.info("Video processing completed",
                       file_path=file_path,
                       duration=result.processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Video processing failed",
                        file_path=file_path,
                        error=str(e))
            raise
    
    async def process_image(self, file_path: str, options: Dict[str, Any]) -> ProcessingResult:
        """Process image file."""
        if not self.image_processor:
            raise RuntimeError("Image processor not initialized")
        
        try:
            result = await self.image_processor.process_file(Path(file_path))
            
            logger.info("Image processing completed",
                       file_path=file_path,
                       duration=result.processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Image processing failed",
                        file_path=file_path,
                        error=str(e))
            raise
    
    async def _initialize_gpu_processors(self):
        """Initialize GPU-accelerated processors."""
        logger.info("Initializing GPU processors")
        
        # Audio processor with GPU acceleration
        audio_config = AudioConfig(
            model_size=self.config.whisper_model_size,
            device="cuda" if self.config.enable_gpu else "cpu",
            enable_diarization=self.config.enable_diarization,
            enable_topic_segmentation=self.config.enable_topic_segmentation
        )
        self.audio_processor = AudioProcessor(audio_config)
        
        # Video processor with GPU acceleration
        video_config = VideoConfig(
            enable_gpu_acceleration=self.config.enable_gpu_acceleration,
            extract_audio=True,
            generate_thumbnails=True,
            thumbnail_quality=self.config.thumbnail_quality
        )
        self.video_processor = VideoProcessor(video_config)
        
        # Image processor
        image_config = ImageConfig(
            enable_gpu=self.config.enable_gpu
        )
        self.image_processor = ImageProcessor(image_config)
    
    async def _initialize_cpu_processors(self):
        """Initialize CPU-only processors."""
        logger.info("Initializing CPU processors")
        
        # For CPU workers, we would initialize document processors, etc.
        # This is a placeholder for CPU-specific processors
        pass
```

## Testing Requirements

### Unit Tests
1. **Remote Worker Tests**
   - Test worker startup/shutdown
   - Test server registration/unregistration
   - Test task processing
   - Test file transfer integration

2. **Remote Client Tests**
   - Test server communication
   - Test file upload/download
   - Test WebSocket connection
   - Test hardware detection

### Integration Tests
1. **End-to-End Remote Worker Tests**
   - Test complete remote task processing workflow
   - Test worker failure and recovery
   - Test file transfer with large files
   - Test GPU acceleration functionality

### Test Files to Create
- `tests/test_remote_worker.py`
- `tests/test_remote_client.py`
- `tests/integration/test_remote_worker_e2e.py`

## Dependencies
- **New**: All MoRAG processing packages (audio, video, image)
- **Existing**: Worker communication from Task 4
- **Existing**: File transfer from Task 3

## Success Criteria
1. Remote worker can be deployed on separate machines
2. Worker successfully registers with main server
3. GPU acceleration works for audio/video processing
4. File transfers work reliably for large files
5. Worker handles task failures gracefully
6. Health monitoring and reporting function correctly

## Next Steps
After completing this task:
1. Proceed to Task 9: Authentication & Security
2. Test remote worker deployment on separate machine
3. Validate GPU acceleration and performance improvements

---

**Dependencies**: Task 4 (Worker Communication), Task 3 (File Transfer)
**Estimated Time**: 5-6 days
**Risk Level**: High (complex standalone deployment)
