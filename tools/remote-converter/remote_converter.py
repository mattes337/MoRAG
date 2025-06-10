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
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import structlog
import requests
from dotenv import load_dotenv

# Add MoRAG packages to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-audio" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-video" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-document" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-image" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-web" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-youtube" / "src"))

try:
    # Import processors directly from their modules to avoid __init__.py issues
    from morag_audio.processor import AudioProcessor, AudioProcessingResult
    from morag_video.processor import VideoProcessor, VideoProcessingResult
    from morag_document.processor import DocumentProcessor
    from morag_image.processor import ImageProcessor, ImageProcessingResult
    from morag_web.processor import WebProcessor, WebScrapingResult as WebProcessingResult
    from morag_youtube.processor import YouTubeProcessor, YouTubeDownloadResult as YouTubeProcessingResult
    # Import base result types
    from morag_core.interfaces.processor import ProcessingResult as BaseProcessingResult
except ImportError as e:
    print(f"Error importing MoRAG packages: {e}")
    print("Please ensure MoRAG packages are installed:")
    print("pip install -e packages/morag-core")
    print("pip install -e packages/morag-audio")
    print("pip install -e packages/morag-video")
    sys.exit(1)

logger = structlog.get_logger(__name__)


@dataclass
class ProcessingResult:
    """Unified processing result for remote converter."""
    success: bool
    text_content: str
    metadata: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None


def convert_to_unified_result(result: Any, processing_time: float = None) -> ProcessingResult:
    """Convert processor-specific result to unified ProcessingResult."""
    if isinstance(result, AudioProcessingResult):
        return ProcessingResult(
            success=result.success,
            text_content=result.transcript,
            metadata=result.metadata,
            processing_time=processing_time or result.processing_time,
            error_message=result.error_message
        )
    elif isinstance(result, VideoProcessingResult):
        # VideoProcessingResult doesn't have transcript directly, extract from audio processing result
        text_content = ""
        if hasattr(result, 'audio_processing_result') and result.audio_processing_result:
            text_content = getattr(result.audio_processing_result, 'transcript', "")

        return ProcessingResult(
            success=True,  # VideoProcessingResult doesn't have success field
            text_content=text_content,
            metadata=result.metadata.__dict__ if hasattr(result.metadata, '__dict__') else {},
            processing_time=processing_time or result.processing_time,
            error_message=None
        )
    elif isinstance(result, BaseProcessingResult):
        # Handle BaseProcessingResult from processors that use the interface
        text_content = ""

        # Extract text content from metadata or document
        if hasattr(result, 'document') and result.document:
            text_content = getattr(result.document, 'content', "")
        elif hasattr(result, 'metadata') and result.metadata:
            # For image processing, combine caption and extracted text from metadata
            metadata = result.metadata
            if 'caption' in metadata and metadata['caption']:
                text_content += f"Caption: {metadata['caption']}\n"
            if 'extracted_text' in metadata and metadata['extracted_text']:
                text_content += f"Extracted Text: {metadata['extracted_text']}"

        return ProcessingResult(
            success=result.success,
            text_content=text_content.strip(),
            metadata=result.metadata,
            processing_time=processing_time or result.processing_time,
            error_message=result.error_message
        )
    elif isinstance(result, ImageProcessingResult):
        # Specific ImageProcessingResult (when called directly)
        text_content = ""
        if hasattr(result, 'caption') and result.caption:
            text_content += f"Caption: {result.caption}\n"
        if hasattr(result, 'extracted_text') and result.extracted_text:
            text_content += f"Extracted Text: {result.extracted_text}"

        return ProcessingResult(
            success=True,  # ImageProcessingResult doesn't have success field
            text_content=text_content.strip(),
            metadata=result.metadata.__dict__ if hasattr(result.metadata, '__dict__') else result.metadata,
            processing_time=processing_time or result.processing_time,
            error_message=None
        )
    elif isinstance(result, WebProcessingResult):
        # WebScrapingResult has content field with WebContent
        text_content = ""
        if hasattr(result, 'content') and result.content:
            text_content = getattr(result.content, 'markdown_content', "") or getattr(result.content, 'content', "")

        return ProcessingResult(
            success=result.success,
            text_content=text_content,
            metadata=result.metadata,
            processing_time=processing_time or result.processing_time,
            error_message=result.error_message
        )
    elif isinstance(result, YouTubeProcessingResult):
        # YouTubeDownloadResult - extract metadata as text content
        text_content = ""
        if hasattr(result, 'metadata') and result.metadata:
            metadata = result.metadata
            text_content = f"Title: {getattr(metadata, 'title', 'Unknown')}\n"
            text_content += f"Description: {getattr(metadata, 'description', 'No description')}\n"
            text_content += f"Uploader: {getattr(metadata, 'uploader', 'Unknown')}\n"
            text_content += f"Duration: {getattr(metadata, 'duration', 0)} seconds"

        return ProcessingResult(
            success=result.success,
            text_content=text_content,
            metadata=result.metadata.__dict__ if hasattr(result.metadata, '__dict__') else {},
            processing_time=processing_time or result.processing_time,
            error_message=getattr(result, 'error_message', None)
        )
    else:
        # Fallback for unknown result types
        return ProcessingResult(
            success=False,
            text_content="",
            metadata={},
            processing_time=processing_time or 0.0,
            error_message=f"Unknown result type: {type(result)}"
        )


class RemoteConverter:
    """Remote conversion worker that processes MoRAG jobs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.worker_id = config['worker_id']
        self.api_base_url = config['api_base_url'].rstrip('/')
        self.api_key = config.get('api_key')
        self.content_types = config['content_types']
        self.poll_interval = config.get('poll_interval', 10)
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 2)
        self.temp_dir = config.get('temp_dir', '/tmp/morag_remote')
        self.running = False
        self.active_jobs = {}
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize processors
        self.processors = {}
        try:
            if 'audio' in self.content_types:
                self.processors['audio'] = AudioProcessor()
            if 'video' in self.content_types:
                self.processors['video'] = VideoProcessor()
            if 'document' in self.content_types:
                self.processors['document'] = DocumentProcessor()
            if 'image' in self.content_types:
                self.processors['image'] = ImageProcessor()
            if 'web' in self.content_types:
                self.processors['web'] = WebProcessor()
            if 'youtube' in self.content_types:
                self.processors['youtube'] = YouTubeProcessor()
        except Exception as e:
            logger.error("Failed to initialize processors", error=str(e))
            raise
        
        logger.info("Remote converter initialized",
                   worker_id=self.worker_id,
                   content_types=self.content_types,
                   api_base_url=self.api_base_url,
                   temp_dir=self.temp_dir)
    
    def start(self):
        """Start the remote converter."""
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Starting remote converter", worker_id=self.worker_id)
        
        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error("Remote converter failed", error=str(e))
            raise
        finally:
            self._cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal", signal=signum)
        self.running = False
    
    async def _main_loop(self):
        """Main processing loop."""
        logger.info("Remote converter main loop started")
        
        while self.running:
            try:
                # Clean up completed jobs
                await self._cleanup_completed_jobs()
                
                # Check if we can take more jobs
                if len(self.active_jobs) < self.max_concurrent_jobs:
                    # Poll for new jobs
                    job = await self._poll_for_job()
                    if job:
                        # Start processing job asynchronously
                        task = asyncio.create_task(self._process_job(job))
                        self.active_jobs[job['job_id']] = task
                        logger.info("Started processing job", 
                                   job_id=job['job_id'],
                                   active_jobs=len(self.active_jobs))
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error("Error in main loop", error=str(e))
                await asyncio.sleep(self.poll_interval)
        
        # Wait for active jobs to complete
        if self.active_jobs:
            logger.info("Waiting for active jobs to complete", count=len(self.active_jobs))
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
    
    async def _cleanup_completed_jobs(self):
        """Remove completed jobs from active jobs list."""
        completed_jobs = []
        for job_id, task in self.active_jobs.items():
            if task.done():
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            del self.active_jobs[job_id]
    
    async def _poll_for_job(self) -> Optional[Dict[str, Any]]:
        """Poll the API for available jobs."""
        try:
            url = f"{self.api_base_url}/api/v1/remote-jobs/poll"
            params = {
                'worker_id': self.worker_id,
                'content_types': ','.join(self.content_types),
                'max_jobs': 1
            }
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('job_id'):
                    logger.info("Job received from API", 
                               job_id=data['job_id'],
                               content_type=data.get('content_type'))
                    return data
            elif response.status_code != 204:  # 204 = no jobs available
                logger.warning("Failed to poll for jobs", 
                             status_code=response.status_code,
                             response=response.text)
            
            return None
            
        except Exception as e:
            logger.error("Exception polling for jobs", error=str(e))
            return None
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single job."""
        job_id = job['job_id']
        content_type = job['content_type']
        task_options = job.get('task_options', {})
        
        logger.info("Processing job", 
                   job_id=job_id,
                   content_type=content_type)
        
        start_time = time.time()
        
        try:
            # Download source file
            file_path = await self._download_source_file(job_id, job.get('source_file_url'))
            if not file_path:
                await self._submit_error_result(job_id, "Failed to download source file")
                return
            
            # Process the file
            result = await self._process_file(file_path, content_type, task_options)
            
            # Submit result
            if result and result.success:
                await self._submit_success_result(job_id, result, time.time() - start_time)
            else:
                error_msg = result.error_message if result else "Processing failed"
                await self._submit_error_result(job_id, error_msg)
            
        except Exception as e:
            logger.error("Exception processing job", job_id=job_id, error=str(e))
            await self._submit_error_result(job_id, f"Processing exception: {str(e)}")
        
        finally:
            # Clean up temporary files
            try:
                if 'file_path' in locals() and file_path and os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file", error=str(e))
    
    async def _download_source_file(self, job_id: str, source_file_url: str) -> Optional[str]:
        """Download source file for processing."""
        try:
            if not source_file_url:
                logger.error("No source file URL provided", job_id=job_id)
                return None

            url = f"{self.api_base_url}{source_file_url}"
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            response = requests.get(url, headers=headers, timeout=300, stream=True)

            if response.status_code == 200:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    dir=self.temp_dir,
                    delete=False,
                    suffix=f"_{job_id}"
                )

                # Download file in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)

                temp_file.close()

                logger.info("Source file downloaded",
                           job_id=job_id,
                           file_path=temp_file.name,
                           file_size=os.path.getsize(temp_file.name))

                return temp_file.name
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
        start_time = time.time()

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

            # Process the file based on content type using correct method names
            if content_type == 'audio':
                # AudioProcessor has process() method
                result = await processor.process(file_path)
            elif content_type == 'video':
                # VideoProcessor has process() method
                result = await processor.process(file_path)
            elif content_type == 'document':
                # DocumentProcessor has process_file() method
                result = await processor.process_file(file_path)
            elif content_type == 'image':
                # ImageProcessor has process() method
                result = await processor.process(file_path)
            elif content_type == 'web':
                # For web content, file_path would contain the URL
                with open(file_path, 'r') as f:
                    url = f.read().strip()
                # WebProcessor has process_url() method
                result = await processor.process_url(url)
            elif content_type == 'youtube':
                # For YouTube content, file_path would contain the URL
                with open(file_path, 'r') as f:
                    url = f.read().strip()
                # YouTubeProcessor has process_url() method
                result = await processor.process_url(url)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

            # Convert processor-specific result to unified format
            processing_time = time.time() - start_time
            return convert_to_unified_result(result, processing_time)

        except Exception as e:
            logger.error("Exception processing file",
                        file_path=file_path,
                        content_type=content_type,
                        error=str(e))
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                text_content="",
                metadata={},
                processing_time=processing_time,
                error_message=f"Processing exception: {str(e)}"
            )

    async def _submit_success_result(self, job_id: str, result: ProcessingResult, processing_time: float):
        """Submit successful processing result."""
        try:
            url = f"{self.api_base_url}/api/v1/remote-jobs/{job_id}/result"

            payload = {
                'success': True,
                'content': result.text_content or "",
                'metadata': result.metadata or {},
                'processing_time': processing_time
            }

            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            response = requests.put(url, json=payload, headers=headers, timeout=60)

            if response.status_code == 200:
                logger.info("Successfully submitted result", job_id=job_id)
            else:
                logger.error("Failed to submit result",
                           job_id=job_id,
                           status_code=response.status_code,
                           response=response.text)

        except Exception as e:
            logger.error("Exception submitting result", job_id=job_id, error=str(e))

    async def _submit_error_result(self, job_id: str, error_message: str):
        """Submit error result."""
        try:
            url = f"{self.api_base_url}/api/v1/remote-jobs/{job_id}/result"

            payload = {
                'success': False,
                'error_message': error_message
            }

            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            response = requests.put(url, json=payload, headers=headers, timeout=60)

            if response.status_code == 200:
                logger.info("Successfully submitted error result", job_id=job_id)
            else:
                logger.error("Failed to submit error result",
                           job_id=job_id,
                           status_code=response.status_code)

        except Exception as e:
            logger.error("Exception submitting error result", job_id=job_id, error=str(e))

    def test_connection(self) -> bool:
        """Test connection to the MoRAG API."""
        try:
            url = f"{self.api_base_url}/health"
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                logger.info("API connection test successful", api_url=self.api_base_url)
                return True
            else:
                logger.error("API connection test failed",
                           api_url=self.api_base_url,
                           status_code=response.status_code)
                return False

        except Exception as e:
            logger.error("API connection test exception",
                        api_url=self.api_base_url,
                        error=str(e))
            return False

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up remote converter")

        # Clean up temp directory
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning("Failed to clean up temp directory", error=str(e))
