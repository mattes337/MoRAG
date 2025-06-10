# Task 4: Remote Converter Tool Development

## Overview

Develop a standalone remote conversion application that polls the MoRAG API for pending remote jobs, processes them using existing MoRAG conversion code, and submits results back to the API. This tool will run on external machines with better processing capabilities.

## Objectives

1. Create standalone remote converter application
2. Implement job polling and claiming mechanism
3. Integrate with existing MoRAG processing components
4. Add secure file transfer capabilities
5. Implement robust error handling and retry logic
6. Add monitoring and logging for remote operations

## Technical Requirements

### 1. Remote Converter Application Structure

**File**: `tools/remote-converter/remote_converter.py`

```python
#!/usr/bin/env python3
"""
MoRAG Remote Converter Tool

Standalone application that polls for remote conversion jobs and processes them
using existing MoRAG components.
"""

import asyncio
import os
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import structlog
import requests
from dotenv import load_dotenv

# Add MoRAG packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from morag_audio import AudioProcessor
from morag_video import VideoProcessor
from morag_document import DocumentProcessor
from morag_image import ImageProcessor
from morag_web import WebProcessor
from morag_youtube import YouTubeProcessor
from morag_core.models import ProcessingResult

logger = structlog.get_logger(__name__)

class RemoteConverter:
    """Remote conversion worker that processes MoRAG jobs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.worker_id = config['worker_id']
        self.api_base_url = config['api_base_url']
        self.api_key = config.get('api_key')
        self.content_types = config['content_types']
        self.poll_interval = config.get('poll_interval', 10)
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 2)
        self.running = False
        self.active_jobs = {}
        
        # Initialize processors
        self.processors = {
            'audio': AudioProcessor(),
            'video': VideoProcessor(),
            'document': DocumentProcessor(),
            'image': ImageProcessor(),
            'web': WebProcessor(),
            'youtube': YouTubeProcessor()
        }
        
        logger.info("Remote converter initialized",
                   worker_id=self.worker_id,
                   content_types=self.content_types,
                   api_base_url=self.api_base_url)
    
    def start(self):
        """Start the remote converter worker."""
        self.running = True
        logger.info("Starting remote converter worker", worker_id=self.worker_id)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start main processing loop
        asyncio.run(self._main_loop())
    
    def stop(self):
        """Stop the remote converter worker."""
        self.running = False
        logger.info("Stopping remote converter worker", worker_id=self.worker_id)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal", signal=signum)
        self.stop()
    
    async def _main_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Check if we can accept more jobs
                if len(self.active_jobs) < self.max_concurrent_jobs:
                    # Poll for new jobs
                    job = await self._poll_for_job()
                    if job:
                        # Process job in background
                        task = asyncio.create_task(self._process_job(job))
                        self.active_jobs[job['job_id']] = task
                
                # Clean up completed jobs
                await self._cleanup_completed_jobs()
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error("Error in main loop", error=str(e))
                await asyncio.sleep(self.poll_interval)
        
        # Wait for active jobs to complete
        if self.active_jobs:
            logger.info("Waiting for active jobs to complete", count=len(self.active_jobs))
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
    
    async def _poll_for_job(self) -> Optional[Dict[str, Any]]:
        """Poll the API for available jobs."""
        try:
            params = {
                'worker_id': self.worker_id,
                'content_types': ','.join(self.content_types),
                'max_jobs': 1
            }

            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            response = requests.get(
                f"{self.api_base_url}/api/v1/remote-jobs/poll",
                params=params,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                job_data = response.json()
                if job_data.get('job_id'):
                    logger.info("Received job from API",
                               job_id=job_data['job_id'],
                               content_type=job_data.get('content_type'))
                    return job_data
            elif response.status_code != 204:  # 204 = No jobs available
                logger.warning("Unexpected response from poll endpoint",
                             status_code=response.status_code,
                             response=response.text)

            return None

        except Exception as e:
            logger.error("Error polling for jobs", error=str(e))
            return None
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single job."""
        job_id = job['job_id']
        content_type = job['content_type']
        source_file_url = job['source_file_url']
        task_options = job.get('task_options', {})
        
        try:
            logger.info("Starting job processing",
                       job_id=job_id,
                       content_type=content_type)
            
            # Download source file
            source_file_path = await self._download_source_file(source_file_url, job_id)
            if not source_file_path:
                await self._submit_job_result(job_id, {
                    'success': False,
                    'error_message': 'Failed to download source file'
                })
                return
            
            # Process the file
            start_time = time.time()
            result = await self._process_file(source_file_path, content_type, task_options)
            processing_time = time.time() - start_time
            
            # Submit result
            if result and result.success:
                await self._submit_job_result(job_id, {
                    'success': True,
                    'content': result.text_content,
                    'metadata': result.metadata,
                    'processing_time': processing_time
                })
                logger.info("Job completed successfully",
                           job_id=job_id,
                           processing_time=processing_time)
            else:
                error_message = result.error_message if result else "Processing failed"
                await self._submit_job_result(job_id, {
                    'success': False,
                    'error_message': error_message,
                    'processing_time': processing_time
                })
                logger.error("Job processing failed",
                           job_id=job_id,
                           error=error_message)
            
            # Clean up source file
            try:
                os.unlink(source_file_path)
            except Exception as e:
                logger.warning("Failed to clean up source file",
                             file_path=source_file_path,
                             error=str(e))
            
        except Exception as e:
            logger.error("Exception processing job", job_id=job_id, error=str(e))
            await self._submit_job_result(job_id, {
                'success': False,
                'error_message': f"Processing exception: {str(e)}"
            })
        finally:
            # Remove from active jobs
            self.active_jobs.pop(job_id, None)
    
    async def _download_source_file(self, source_file_url: str, job_id: str) -> Optional[str]:
        """Download source file from API."""
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.get(
                f"{self.api_base_url}{source_file_url}",
                headers=headers,
                stream=True,
                timeout=300  # 5 minute timeout for download
            )
            
            if response.status_code == 200:
                # Create temp directory for this job
                temp_dir = Path(f"/tmp/morag_remote_{job_id}")
                temp_dir.mkdir(exist_ok=True)
                
                # Determine file extension from content-disposition or URL
                filename = f"source_file_{job_id}"
                if 'content-disposition' in response.headers:
                    import re
                    cd = response.headers['content-disposition']
                    match = re.search(r'filename="?([^"]+)"?', cd)
                    if match:
                        filename = match.group(1)
                
                file_path = temp_dir / filename
                
                # Download file
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info("Source file downloaded",
                           job_id=job_id,
                           file_path=str(file_path),
                           file_size=file_path.stat().st_size)
                
                return str(file_path)
            else:
                logger.error("Failed to download source file",
                           job_id=job_id,
                           status_code=response.status_code)
                return None
                
        except Exception as e:
            logger.error("Exception downloading source file",
                        job_id=job_id,
                        error=str(e))
            return None
    
    async def _process_file(self, file_path: str, content_type: str, options: Dict[str, Any]) -> Optional[ProcessingResult]:
        """Process file using appropriate MoRAG processor."""
        try:
            processor = self.processors.get(content_type)
            if not processor:
                logger.error("No processor available for content type", content_type=content_type)
                return ProcessingResult(
                    success=False,
                    text_content="",
                    metadata={},
                    processing_time=0.0,
                    error_message=f"No processor available for content type: {content_type}"
                )
            
            # Process the file
            if content_type == 'audio':
                result = await processor.process_audio(file_path, options)
            elif content_type == 'video':
                result = await processor.process_video(file_path, options)
            elif content_type == 'document':
                result = await processor.process_document(file_path, options)
            elif content_type == 'image':
                result = await processor.process_image(file_path, options)
            elif content_type == 'web':
                # For web content, file_path would contain the URL
                with open(file_path, 'r') as f:
                    url = f.read().strip()
                result = await processor.process_url(url, options)
            elif content_type == 'youtube':
                # For YouTube content, file_path would contain the URL
                with open(file_path, 'r') as f:
                    url = f.read().strip()
                result = await processor.process_youtube_video(url, options)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            return result
            
        except Exception as e:
            logger.error("Exception processing file",
                        file_path=file_path,
                        content_type=content_type,
                        error=str(e))
            return ProcessingResult(
                success=False,
                text_content="",
                metadata={},
                processing_time=0.0,
                error_message=str(e)
            )
    
    async def _submit_job_result(self, job_id: str, result: Dict[str, Any]):
        """Submit job result to API."""
        try:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.put(
                f"{self.api_base_url}/api/v1/remote-jobs/{job_id}/result",
                json=result,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Job result submitted successfully", job_id=job_id)
            else:
                logger.error("Failed to submit job result",
                           job_id=job_id,
                           status_code=response.status_code,
                           response=response.text)
                
        except Exception as e:
            logger.error("Exception submitting job result",
                        job_id=job_id,
                        error=str(e))
    
    async def _cleanup_completed_jobs(self):
        """Clean up completed job tasks."""
        completed_jobs = []
        for job_id, task in self.active_jobs.items():
            if task.done():
                completed_jobs.append(job_id)
                try:
                    await task  # This will raise any exceptions that occurred
                except Exception as e:
                    logger.error("Job task completed with exception",
                               job_id=job_id,
                               error=str(e))
        
        for job_id in completed_jobs:
            self.active_jobs.pop(job_id, None)
```

### 2. Configuration Management

**File**: `tools/remote-converter/config.py`

```python
import os
from typing import Dict, Any, List
from pathlib import Path
import yaml
import structlog

logger = structlog.get_logger(__name__)

class RemoteConverterConfig:
    """Configuration management for remote converter."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "remote_converter_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables."""
        config = {}
        
        # Load from config file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
                logger.info("Loaded configuration from file", file=self.config_file)
            except Exception as e:
                logger.warning("Failed to load config file", file=self.config_file, error=str(e))
        
        # Override with environment variables
        env_config = {
            'worker_id': os.getenv('MORAG_WORKER_ID', f'remote-worker-{os.getpid()}'),
            'api_base_url': os.getenv('MORAG_API_BASE_URL', 'http://localhost:8000'),
            'api_key': os.getenv('MORAG_API_KEY'),
            'content_types': os.getenv('MORAG_WORKER_CONTENT_TYPES', 'audio,video').split(','),
            'poll_interval': int(os.getenv('MORAG_WORKER_POLL_INTERVAL', '10')),
            'max_concurrent_jobs': int(os.getenv('MORAG_WORKER_MAX_CONCURRENT_JOBS', '2')),
            'log_level': os.getenv('MORAG_LOG_LEVEL', 'INFO'),
            'temp_dir': os.getenv('MORAG_TEMP_DIR', '/tmp/morag_remote')
        }
        
        # Remove None values
        env_config = {k: v for k, v in env_config.items() if v is not None}
        config.update(env_config)
        
        return config
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_fields = ['worker_id', 'api_base_url', 'content_types']
        
        for field in required_fields:
            if not self.config.get(field):
                logger.error("Missing required configuration", field=field)
                return False
        
        # Validate content types
        valid_types = ['audio', 'video', 'document', 'image', 'web', 'youtube']
        for content_type in self.config['content_types']:
            if content_type not in valid_types:
                logger.error("Invalid content type", content_type=content_type, valid_types=valid_types)
                return False
        
        return True
    
    def create_sample_config(self, file_path: str = None):
        """Create a sample configuration file."""
        if not file_path:
            file_path = "remote_converter_config.yaml.example"
        
        sample_config = {
            'worker_id': 'gpu-worker-01',
            'api_base_url': 'https://api.morag.com',
            'api_key': 'your-api-key-here',
            'content_types': ['audio', 'video'],
            'poll_interval': 10,
            'max_concurrent_jobs': 2,
            'log_level': 'INFO',
            'temp_dir': '/tmp/morag_remote'
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False)
        
        logger.info("Sample configuration created", file=file_path)
```

### 3. Command Line Interface

**File**: `tools/remote-converter/cli.py`

```python
#!/usr/bin/env python3
"""Command line interface for MoRAG Remote Converter."""

import argparse
import sys
import os
from pathlib import Path
import structlog
from dotenv import load_dotenv

from config import RemoteConverterConfig
from remote_converter import RemoteConverter

def setup_logging(log_level: str):
    """Set up structured logging."""
    structlog.configure(
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
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MoRAG Remote Converter")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--worker-id', help='Unique worker identifier')
    parser.add_argument('--api-url', help='MoRAG API base URL')
    parser.add_argument('--api-key', help='API authentication key')
    parser.add_argument('--content-types', help='Comma-separated list of content types to process')
    parser.add_argument('--poll-interval', type=int, help='Polling interval in seconds')
    parser.add_argument('--max-jobs', type=int, help='Maximum concurrent jobs')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    parser.add_argument('--test-connection', action='store_true', help='Test API connection and exit')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create sample config if requested
    if args.create_config:
        config_manager = RemoteConverterConfig()
        config_manager.create_sample_config()
        print("Sample configuration created: remote_converter_config.yaml.example")
        return
    
    # Load configuration
    config_manager = RemoteConverterConfig(args.config)
    
    # Override with command line arguments
    if args.worker_id:
        config_manager.config['worker_id'] = args.worker_id
    if args.api_url:
        config_manager.config['api_base_url'] = args.api_url
    if args.api_key:
        config_manager.config['api_key'] = args.api_key
    if args.content_types:
        config_manager.config['content_types'] = args.content_types.split(',')
    if args.poll_interval:
        config_manager.config['poll_interval'] = args.poll_interval
    if args.max_jobs:
        config_manager.config['max_concurrent_jobs'] = args.max_jobs
    if args.log_level:
        config_manager.config['log_level'] = args.log_level
    
    # Validate configuration
    if not config_manager.validate():
        print("Configuration validation failed. Use --create-config to create a sample configuration.")
        sys.exit(1)
    
    # Set up logging
    setup_logging(config_manager.get('log_level', 'INFO'))
    logger = structlog.get_logger(__name__)
    
    # Test connection if requested
    if args.test_connection:
        import requests
        try:
            response = requests.get(
                f"{config_manager.get('api_base_url')}/docs",
                timeout=10
            )
            if response.status_code == 200:
                print("✓ API connection successful")
                sys.exit(0)
            else:
                print(f"✗ API connection failed: HTTP {response.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"✗ API connection failed: {e}")
            sys.exit(1)
    
    # Create and start remote converter
    try:
        converter = RemoteConverter(config_manager.config)
        converter.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down")
    except Exception as e:
        logger.error("Remote converter failed", error=str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
```

### 4. Installation and Setup Scripts

**File**: `tools/remote-converter/install.sh`

```bash
#!/bin/bash
# Installation script for MoRAG Remote Converter

set -e

echo "Installing MoRAG Remote Converter..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip

# Install MoRAG packages
echo "Installing MoRAG packages..."
pip install -e ../../packages/morag-core
pip install -e ../../packages/morag-audio
pip install -e ../../packages/morag-video
pip install -e ../../packages/morag-document
pip install -e ../../packages/morag-image
pip install -e ../../packages/morag-web
pip install -e ../../packages/morag-youtube

# Install additional dependencies
pip install requests pyyaml python-dotenv structlog

# Create configuration file
echo "Creating configuration file..."
python3 cli.py --create-config

# Create systemd service file
echo "Creating systemd service file..."
cat > morag-remote-converter.service << EOF
[Unit]
Description=MoRAG Remote Converter
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python cli.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit remote_converter_config.yaml.example and save as remote_converter_config.yaml"
echo "2. Set environment variables or update config file with your API details"
echo "3. Test connection: python3 cli.py --test-connection"
echo "4. Start converter: python3 cli.py"
echo ""
echo "To install as system service:"
echo "sudo cp morag-remote-converter.service /etc/systemd/system/"
echo "sudo systemctl enable morag-remote-converter"
echo "sudo systemctl start morag-remote-converter"
```

## Implementation Steps

1. **Create Application Structure** (Day 1)
   - Set up remote converter directory
   - Create main application class
   - Implement basic polling mechanism

2. **Add File Processing** (Day 1-2)
   - Integrate MoRAG processors
   - Add file download functionality
   - Implement result submission

3. **Configuration and CLI** (Day 2)
   - Create configuration management
   - Add command line interface
   - Create installation scripts

4. **Error Handling and Monitoring** (Day 3)
   - Add robust error handling
   - Implement retry mechanisms
   - Add comprehensive logging

5. **Testing and Documentation** (Day 3-4)
   - Test with various content types
   - Create setup documentation
   - Performance testing

## Testing Requirements

### Integration Tests

**File**: `tools/remote-converter/tests/test_remote_converter.py`

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from remote_converter import RemoteConverter

class TestRemoteConverter:
    def test_initialization(self):
        # Test converter initialization
        pass
    
    def test_job_polling(self):
        # Test job polling mechanism
        pass
    
    def test_file_processing(self):
        # Test file processing with different content types
        pass
    
    def test_error_handling(self):
        # Test error handling and recovery
        pass
    
    def test_concurrent_processing(self):
        # Test concurrent job processing
        pass
```

## Success Criteria

1. Remote converter successfully polls for and processes jobs
2. All MoRAG content types are supported
3. Error handling and retry mechanisms work correctly
4. Concurrent job processing works reliably
5. Installation and setup process is straightforward

## Dependencies

- Remote job API endpoints (Task 1)
- Database schema for job tracking (Task 2)
- Worker modifications for job creation (Task 3)
- All existing MoRAG processing packages

## Next Steps

After completing this task:
1. Proceed to [Task 5: Job Lifecycle Management](./05-job-lifecycle-management.md)
2. Test remote converter with real jobs
3. Set up monitoring and alerting
4. Begin integration testing with complete system
