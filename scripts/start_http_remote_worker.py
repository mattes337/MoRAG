#!/usr/bin/env python3
"""
HTTP-based Remote Worker for MoRAG - No Redis Required

This script starts a remote worker that connects directly to the MoRAG server
via HTTP API calls, eliminating the need for Redis/Celery infrastructure.

Usage:
    python scripts/start_http_remote_worker.py --server-url http://main-server:8000 --api-key your-api-key

Features:
- Direct HTTP communication with main server
- No Redis dependency
- Automatic task polling
- GPU/CPU processing support
- File transfer via HTTP
- Configurable polling intervals
"""

import argparse
import asyncio
import aiohttp
import json
import logging
import os
import sys
import time
import tempfile
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from aiohttp import web

# Add the packages to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))

import structlog
from morag.api import MoRAGAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for HTTP remote worker."""
    server_url: str
    api_key: str
    worker_id: str
    worker_type: str = "gpu"  # gpu, cpu
    poll_interval: int = 5  # seconds
    max_concurrent_tasks: int = 1
    timeout: int = 300  # seconds
    health_port: int = 8080  # health check port


class HTTPRemoteWorker:
    """HTTP-based remote worker that processes tasks without Redis."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.morag_api: Optional[MoRAGAPI] = None
        self.running = False
        self.active_tasks = 0
        self.health_app: Optional[web.Application] = None
        self.health_runner: Optional[web.AppRunner] = None
        self.start_time = time.time()
        
    async def start(self):
        """Start the remote worker."""
        logger.info("Starting HTTP Remote Worker", 
                   worker_id=self.config.worker_id,
                   server_url=self.config.server_url,
                   worker_type=self.config.worker_type)
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Initialize MoRAG API
        self.morag_api = MoRAGAPI()
        
        # Register worker with server
        await self.register_worker()
        
        self.running = True
        
        # Start health check server
        await self.start_health_server()

        # Start main polling loop
        try:
            await self.poll_loop()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()
    
    async def register_worker(self):
        """Register this worker with the main server."""
        registration_data = {
            "worker_id": self.config.worker_id,
            "worker_type": self.config.worker_type,
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "capabilities": ["audio", "video", "document", "image", "web", "youtube"]
        }
        
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        try:
            async with self.session.post(
                f"{self.config.server_url}/api/v1/workers/register",
                json=registration_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info("Worker registered successfully")
                else:
                    logger.error("Failed to register worker", status=response.status)
                    raise Exception(f"Registration failed: {response.status}")
        except Exception as e:
            logger.error("Worker registration failed", error=str(e))
            raise
    
    async def poll_loop(self):
        """Main polling loop to get tasks from server."""
        while self.running:
            try:
                if self.active_tasks < self.config.max_concurrent_tasks:
                    task = await self.get_next_task()
                    if task:
                        # Process task in background
                        asyncio.create_task(self.process_task(task))
                
                await asyncio.sleep(self.config.poll_interval)
                
            except Exception as e:
                logger.error("Error in poll loop", error=str(e))
                await asyncio.sleep(self.config.poll_interval)
    
    async def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get next available task from server."""
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        try:
            async with self.session.get(
                f"{self.config.server_url}/api/v1/workers/{self.config.worker_id}/tasks/next",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 204:
                    # No tasks available
                    return None
                else:
                    logger.warning("Failed to get task", status=response.status)
                    return None
        except Exception as e:
            logger.error("Error getting next task", error=str(e))
            return None
    
    async def process_task(self, task: Dict[str, Any]):
        """Process a single task."""
        task_id = task.get("task_id")
        task_type = task.get("task_type")
        
        logger.info("Processing task", task_id=task_id, task_type=task_type)
        self.active_tasks += 1
        
        try:
            # Download input files if needed
            input_files = await self.download_task_files(task)
            
            # Process based on task type
            result = await self.execute_task(task, input_files)
            
            # Upload result files if any
            await self.upload_result_files(task_id, result)
            
            # Report success
            await self.report_task_result(task_id, "completed", result)
            
        except Exception as e:
            logger.error("Task processing failed", task_id=task_id, error=str(e))
            await self.report_task_result(task_id, "failed", {"error": str(e)})
        
        finally:
            self.active_tasks -= 1
    
    async def download_task_files(self, task: Dict[str, Any]) -> Dict[str, str]:
        """Download input files for the task."""
        files = {}
        
        for file_info in task.get("input_files", []):
            file_url = file_info["url"]
            file_name = file_info["name"]
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}")
            
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            
            async with self.session.get(file_url, headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(8192):
                        temp_file.write(chunk)
                    temp_file.close()
                    files[file_info["key"]] = temp_file.name
                else:
                    logger.error("Failed to download file", url=file_url, status=response.status)
        
        return files
    
    async def execute_task(self, task: Dict[str, Any], input_files: Dict[str, str]) -> Dict[str, Any]:
        """Execute the actual task processing."""
        task_type = task["task_type"]
        params = task.get("parameters", {})
        
        # Map task types to MoRAG API methods
        if task_type == "process_audio":
            file_path = input_files.get("audio_file")
            if file_path:
                result = await self.morag_api.process_audio_file(file_path, **params)
                return {"content": result.content, "metadata": result.metadata}
        
        elif task_type == "process_video":
            file_path = input_files.get("video_file")
            if file_path:
                result = await self.morag_api.process_video_file(file_path, **params)
                return {"content": result.content, "metadata": result.metadata}
        
        elif task_type == "process_document":
            file_path = input_files.get("document_file")
            if file_path:
                result = await self.morag_api.process_document(file_path, **params)
                return {"content": result.content, "metadata": result.metadata}
        
        elif task_type == "process_image":
            file_path = input_files.get("image_file")
            if file_path:
                result = await self.morag_api.process_image(file_path, **params)
                return {"content": result.content, "metadata": result.metadata}
        
        elif task_type == "process_web":
            url = params.get("url")
            if url:
                result = await self.morag_api.process_web_page(url, **params)
                return {"content": result.content, "metadata": result.metadata}
        
        elif task_type == "process_youtube":
            url = params.get("url")
            if url:
                result = await self.morag_api.process_youtube_video(url, **params)
                return {"content": result.content, "metadata": result.metadata}
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def upload_result_files(self, task_id: str, result: Dict[str, Any]):
        """Upload any result files to the server."""
        # For now, results are returned as JSON
        # Future: Support file uploads for generated content
        pass
    
    async def report_task_result(self, task_id: str, status: str, result: Dict[str, Any]):
        """Report task completion to server."""
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        report_data = {
            "task_id": task_id,
            "worker_id": self.config.worker_id,
            "status": status,
            "result": result,
            "completed_at": time.time()
        }
        
        try:
            async with self.session.post(
                f"{self.config.server_url}/api/v1/workers/tasks/{task_id}/result",
                json=report_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info("Task result reported", task_id=task_id, status=status)
                else:
                    logger.error("Failed to report result", task_id=task_id, status=response.status)
        except Exception as e:
            logger.error("Error reporting task result", task_id=task_id, error=str(e))
    
    async def shutdown(self):
        """Shutdown the worker gracefully."""
        logger.info("Shutting down worker")
        self.running = False

        if self.health_runner:
            await self.health_runner.cleanup()

        if self.session:
            await self.session.close()

    async def start_health_server(self):
        """Start health check HTTP server."""
        self.health_app = web.Application()
        self.health_app.router.add_get('/health', self.health_check)
        self.health_app.router.add_get('/status', self.status_check)

        self.health_runner = web.AppRunner(self.health_app)
        await self.health_runner.setup()

        site = web.TCPSite(self.health_runner, '0.0.0.0', self.config.health_port)
        await site.start()

        logger.info("Health check server started", port=self.config.health_port)

    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy" if self.running else "shutting_down",
            "worker_id": self.config.worker_id,
            "worker_type": self.config.worker_type,
            "uptime": time.time() - self.start_time,
            "active_tasks": self.active_tasks
        })

    async def status_check(self, request):
        """Detailed status endpoint."""
        return web.json_response({
            "worker_id": self.config.worker_id,
            "worker_type": self.config.worker_type,
            "server_url": self.config.server_url,
            "running": self.running,
            "active_tasks": self.active_tasks,
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "poll_interval": self.config.poll_interval,
            "uptime": time.time() - self.start_time,
            "start_time": self.start_time
        })


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HTTP Remote Worker for MoRAG")
    parser.add_argument("--server-url", required=True, help="Main server URL")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--worker-id", help="Worker ID (auto-generated if not provided)")
    parser.add_argument("--worker-type", default="gpu", choices=["gpu", "cpu"], help="Worker type")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    parser.add_argument("--health-port", type=int, default=8080, help="Health check port")
    
    args = parser.parse_args()
    
    # Generate worker ID if not provided
    worker_id = args.worker_id or f"http-worker-{int(time.time())}"
    
    config = WorkerConfig(
        server_url=args.server_url.rstrip('/'),
        api_key=args.api_key,
        worker_id=worker_id,
        worker_type=args.worker_type,
        poll_interval=args.poll_interval,
        max_concurrent_tasks=args.max_concurrent,
        timeout=args.timeout,
        health_port=args.health_port
    )
    
    # Handle shutdown signals
    worker = HTTPRemoteWorker(config)
    
    def signal_handler(signum, frame):
        logger.info("Received signal, shutting down...")
        worker.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start worker
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error("Worker failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
