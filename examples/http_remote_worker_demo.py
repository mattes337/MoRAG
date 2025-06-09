#!/usr/bin/env python3
"""
HTTP Remote Worker Demo

This example demonstrates how to set up and use HTTP-based remote workers
that don't require Redis infrastructure.

Features demonstrated:
- Starting HTTP remote workers
- Processing tasks without Redis
- Direct HTTP communication with main server
- Worker health monitoring
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path

# Configuration
MAIN_SERVER_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Replace with actual API key


async def demo_http_worker_setup():
    """Demonstrate HTTP worker setup and usage."""
    print("🚀 HTTP Remote Worker Demo")
    print("=" * 50)
    
    # 1. Check server connectivity
    print("\n1. Testing server connectivity...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{MAIN_SERVER_URL}/health") as response:
                if response.status == 200:
                    print("✅ Main server is reachable")
                else:
                    print(f"❌ Server returned status: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Cannot reach server: {e}")
            return
    
    # 2. Show worker configuration options
    print("\n2. Worker Configuration Options:")
    print("   - GPU Worker: High-performance processing")
    print("   - CPU Worker: Compatibility fallback")
    print("   - Polling interval: 3-10 seconds recommended")
    print("   - Concurrent tasks: 1-2 for GPU, 1 for CPU")
    
    # 3. Demonstrate worker startup commands
    print("\n3. Worker Startup Commands:")
    
    print("\n   🖥️  Python Script (Direct):")
    print(f"   python scripts/start_http_remote_worker.py \\")
    print(f"       --server-url {MAIN_SERVER_URL} \\")
    print(f"       --api-key {API_KEY} \\")
    print(f"       --worker-type gpu \\")
    print(f"       --poll-interval 5 \\")
    print(f"       --max-concurrent 1")
    
    print("\n   🐧 Linux/macOS Script:")
    print(f"   ./scripts/start-http-worker.sh \\")
    print(f"       --server-url {MAIN_SERVER_URL} \\")
    print(f"       --api-key {API_KEY}")
    
    print("\n   🪟 Windows Script:")
    print(f"   scripts\\start-http-worker.bat ^")
    print(f"       --server-url {MAIN_SERVER_URL} ^")
    print(f"       --api-key {API_KEY}")
    
    print("\n   🐳 Docker (Single Worker):")
    print(f"   docker run -e MORAG_SERVER_URL={MAIN_SERVER_URL} \\")
    print(f"              -e MORAG_API_KEY={API_KEY} \\")
    print(f"              -e WORKER_TYPE=gpu \\")
    print(f"              morag-http-worker:latest")
    
    print("\n   🐳 Docker Compose (Multiple Workers):")
    print(f"   docker-compose -f docker-compose.workers.yml up")
    
    # 4. Show environment file setup
    print("\n4. Environment File Setup:")
    print("   cp configs/http-worker.env.example .env")
    print("   # Edit .env with your settings:")
    print(f"   MORAG_SERVER_URL={MAIN_SERVER_URL}")
    print(f"   MORAG_API_KEY={API_KEY}")
    print("   WORKER_TYPE=gpu")
    print("   POLL_INTERVAL=5")
    print("   MAX_CONCURRENT_TASKS=1")
    
    # 5. Demonstrate health check
    print("\n5. Worker Health Monitoring:")
    print("   # Check worker health")
    print("   curl http://worker-host:8080/health")
    print("   # Get detailed status")
    print("   curl http://worker-host:8080/status")
    
    # 6. Show task processing flow
    print("\n6. Task Processing Flow:")
    print("   📥 Worker polls server for tasks")
    print("   📋 Server assigns task to worker")
    print("   📁 Worker downloads input files")
    print("   ⚙️  Worker processes with local MoRAG")
    print("   📤 Worker uploads results to server")
    print("   💾 Server stores in vector database")
    
    # 7. Performance comparison
    print("\n7. Performance Comparison:")
    print("   Redis Workers:")
    print("   ✅ Push-based (instant task assignment)")
    print("   ❌ Requires Redis infrastructure")
    print("   ❌ Complex setup and debugging")
    print("   ")
    print("   HTTP Workers:")
    print("   ✅ No Redis dependency")
    print("   ✅ Simple setup and debugging")
    print("   ✅ Better isolation")
    print("   ⚠️  Poll-based (small latency)")
    
    # 8. Scaling recommendations
    print("\n8. Scaling Recommendations:")
    print("   Small Setup (1-2 workers):")
    print("   - Use Python scripts directly")
    print("   - Single GPU worker + CPU fallback")
    print("   ")
    print("   Medium Setup (3-10 workers):")
    print("   - Use Docker Compose")
    print("   - Multiple GPU workers")
    print("   - Load balancing by user")
    print("   ")
    print("   Large Setup (10+ workers):")
    print("   - Use Kubernetes")
    print("   - Auto-scaling based on queue length")
    print("   - Geographic distribution")


async def demo_worker_api_calls():
    """Demonstrate the API calls that workers make."""
    print("\n" + "=" * 50)
    print("🔌 Worker API Communication Demo")
    print("=" * 50)
    
    # Note: This is just for demonstration - actual worker handles this automatically
    
    print("\n1. Worker Registration:")
    registration_data = {
        "worker_id": "demo-worker-123",
        "worker_type": "gpu",
        "max_concurrent_tasks": 1,
        "capabilities": ["audio", "video", "document", "image", "web", "youtube"]
    }
    print(f"   POST /api/v1/workers/register")
    print(f"   {json.dumps(registration_data, indent=6)}")
    
    print("\n2. Task Polling:")
    print(f"   GET /api/v1/workers/demo-worker-123/tasks/next")
    print(f"   Authorization: Bearer {API_KEY}")
    
    print("\n3. Task Response Example:")
    task_example = {
        "task_id": "task-456",
        "task_type": "process_audio",
        "parameters": {
            "enable_diarization": True,
            "language": "en"
        },
        "input_files": [
            {
                "key": "audio_file",
                "name": "recording.mp3",
                "url": "/api/v1/files/download/abc123"
            }
        ]
    }
    print(f"   {json.dumps(task_example, indent=6)}")
    
    print("\n4. Result Submission:")
    result_data = {
        "task_id": "task-456",
        "worker_id": "demo-worker-123",
        "status": "completed",
        "result": {
            "content": "Transcribed audio content...",
            "metadata": {
                "duration": 120.5,
                "speakers": ["Speaker 1", "Speaker 2"],
                "language": "en"
            }
        },
        "completed_at": time.time()
    }
    print(f"   POST /api/v1/workers/tasks/task-456/result")
    print(f"   {json.dumps(result_data, indent=6)}")


def demo_configuration_examples():
    """Show different configuration examples."""
    print("\n" + "=" * 50)
    print("⚙️  Configuration Examples")
    print("=" * 50)
    
    print("\n1. High-Performance GPU Worker:")
    print("   WORKER_TYPE=gpu")
    print("   MAX_CONCURRENT_TASKS=2")
    print("   POLL_INTERVAL=3")
    print("   MORAG_FORCE_CPU=false")
    print("   CUDA_VISIBLE_DEVICES=0")
    print("   OMP_NUM_THREADS=8")
    
    print("\n2. CPU-Only Compatibility Worker:")
    print("   WORKER_TYPE=cpu")
    print("   MAX_CONCURRENT_TASKS=1")
    print("   POLL_INTERVAL=10")
    print("   MORAG_FORCE_CPU=true")
    print("   OMP_NUM_THREADS=4")
    print("   PYTORCH_DISABLE_NNPACK=1")
    print("   PYTORCH_DISABLE_AVX=1")
    
    print("\n3. Multi-Worker Setup:")
    print("   # Worker 1 (Primary GPU)")
    print("   WORKER_TYPE=gpu")
    print("   CUDA_VISIBLE_DEVICES=0")
    print("   HEALTH_CHECK_PORT=8080")
    print("   ")
    print("   # Worker 2 (Secondary GPU)")
    print("   WORKER_TYPE=gpu")
    print("   CUDA_VISIBLE_DEVICES=1")
    print("   HEALTH_CHECK_PORT=8081")
    print("   ")
    print("   # Worker 3 (CPU Fallback)")
    print("   WORKER_TYPE=cpu")
    print("   MORAG_FORCE_CPU=true")
    print("   HEALTH_CHECK_PORT=8082")


def main():
    """Main demo function."""
    print("HTTP Remote Workers Demo")
    print("This demo shows how to set up workers without Redis")
    print()
    
    # Run async demos
    asyncio.run(demo_http_worker_setup())
    asyncio.run(demo_worker_api_calls())
    
    # Run sync demos
    demo_configuration_examples()
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Copy configs/http-worker.env.example to .env")
    print("2. Edit .env with your server URL and API key")
    print("3. Run: python scripts/start_http_remote_worker.py")
    print("4. Or use Docker: docker-compose -f docker-compose.workers.yml up")
    print("5. Monitor with: curl http://localhost:8080/health")
    print("=" * 50)


if __name__ == "__main__":
    main()
