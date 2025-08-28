"""External YouTube transcription service client."""

import asyncio
import base64
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import aiohttp
import aiofiles
from urllib.parse import urlparse, parse_qs
import re

import structlog
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class YouTubeExternalServiceError(ProcessingError):
    """Exception raised when external YouTube service fails."""
    pass


class YouTubeExternalService:
    """Client for external YouTube transcription service."""
    
    def __init__(self, service_url: Optional[str] = None, timeout: int = 300):
        """Initialize the external service client.
        
        Args:
            service_url: Base URL of the external service (defaults to env var or localhost)
            timeout: Request timeout in seconds (default: 5 minutes)
        """
        self.service_url = service_url or os.getenv("YOUTUBE_SERVICE_URL", "http://localhost:8000")
        self.timeout = timeout
        self.endpoint = f"{self.service_url}/v1/youtube/transcribe"
        
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats.
        
        Args:
            url: YouTube URL or video ID
            
        Returns:
            Video ID string
            
        Raises:
            YouTubeExternalServiceError: If URL format is invalid
        """
        # If it's already just a video ID (11 characters, alphanumeric + underscore/dash)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
            
        # Parse different YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/.*[?&]v=([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
                
        raise YouTubeExternalServiceError(f"Invalid YouTube URL or video ID: {url}")
    
    async def transcribe_video(
        self, 
        url: str, 
        download_video: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Transcribe a YouTube video using the external service.
        
        Args:
            url: YouTube video URL or video ID
            download_video: Whether to download the video file
            output_dir: Directory to save downloaded video (if download_video=True)
            
        Returns:
            Dictionary containing transcription results and metadata
            
        Raises:
            YouTubeExternalServiceError: If the service request fails
        """
        try:
            # Validate URL and extract video ID for logging
            video_id = self._extract_video_id(url)
            logger.info("Starting external YouTube transcription", 
                       video_id=video_id, 
                       download_video=download_video)
            
            # Prepare request payload
            payload = {
                "url": url,
                "download_video": download_video
            }
            
            # Make request with extended timeout
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug("Sending request to external service", 
                           endpoint=self.endpoint, 
                           payload=payload)
                
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 503:
                        raise YouTubeExternalServiceError("YouTube Modal service is not available")
                    elif response.status == 429:
                        raise YouTubeExternalServiceError("API error: Rate limit exceeded")
                    elif response.status != 200:
                        error_text = await response.text()
                        raise YouTubeExternalServiceError(
                            f"External service request failed with status {response.status}: {error_text}"
                        )
                    
                    result = await response.json()
            
            # Check for errors in response
            if "error" in result:
                raise YouTubeExternalServiceError(f"External service error: {result['error']}")
            
            # Handle video download if requested
            output_file = None
            if download_video and result.get("success") and "video_download" in result:
                video_data = result["video_download"]
                if video_data.get("downloaded") and "content_base64" in video_data:
                    output_file = await self._save_video_file(
                        video_data["content_base64"],
                        video_id,
                        video_data.get("format", "mp4"),
                        output_dir
                    )
                    logger.info("Video file saved", output_file=str(output_file))
            
            # Process and return results
            processed_result = self._process_service_response(result, output_file)
            
            logger.info("External YouTube transcription completed successfully", 
                       video_id=video_id,
                       has_transcript=bool(processed_result.get("transcript")),
                       output_file=str(output_file) if output_file else None)
            
            return processed_result
            
        except aiohttp.ClientError as e:
            raise YouTubeExternalServiceError(f"Network error calling external service: {str(e)}")
        except asyncio.TimeoutError:
            raise YouTubeExternalServiceError(f"External service request timeout after {self.timeout} seconds")
        except Exception as e:
            if isinstance(e, YouTubeExternalServiceError):
                raise
            raise YouTubeExternalServiceError(f"Unexpected error: {str(e)}")

    async def _save_video_file(
        self,
        base64_content: str,
        video_id: str,
        format_info: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Save base64-encoded video content to file.

        Args:
            base64_content: Base64-encoded video content
            video_id: YouTube video ID
            format_info: Format information (e.g., "720p mp4")
            output_dir: Directory to save file

        Returns:
            Path to saved video file
        """
        if output_dir is None:
            output_dir = Path.cwd() / "youtube_downloads"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract extension from format info or default to mp4
        ext = "mp4"
        if "mp4" in format_info.lower():
            ext = "mp4"
        elif "webm" in format_info.lower():
            ext = "webm"
        elif "mkv" in format_info.lower():
            ext = "mkv"

        # Create filename
        filename = f"{video_id}.{ext}"
        output_file = output_dir / filename

        # Decode and save video content
        try:
            video_bytes = base64.b64decode(base64_content)
            async with aiofiles.open(output_file, 'wb') as f:
                await f.write(video_bytes)

            logger.debug("Video file saved successfully",
                        output_file=str(output_file),
                        size_bytes=len(video_bytes))

            return output_file

        except Exception as e:
            raise YouTubeExternalServiceError(f"Failed to save video file: {str(e)}")

    def _process_service_response(self, response: Dict[str, Any], output_file: Optional[Path]) -> Dict[str, Any]:
        """Process the external service response into our expected format.

        Args:
            response: Raw response from external service
            output_file: Path to downloaded video file (if any)

        Returns:
            Processed response dictionary
        """
        result = {
            "success": response.get("success", False),
            "metadata": response.get("metadata", {}),
            "transcript": response.get("transcript", {}),
            "transcript_languages": response.get("transcript_languages", []),
            "formats": response.get("formats", []),
            "total_formats": response.get("total_formats", 0)
        }

        # Add output file information if video was downloaded
        if output_file:
            result["output_file"] = str(output_file)
            result["video_download"] = {
                "downloaded": True,
                "file_path": str(output_file),
                "size_bytes": output_file.stat().st_size if output_file.exists() else 0
            }

        return result

    async def health_check(self) -> Dict[str, Any]:
        """Check if the external service is available.

        Returns:
            Dictionary with health status information
        """
        try:
            health_url = f"{self.service_url}/health"
            timeout = aiohttp.ClientTimeout(total=10)  # Short timeout for health check

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return {
                            "status": "healthy",
                            "youtube_modal_connected": health_data.get("youtube_modal_connected", False),
                            "service_url": self.service_url
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"Health check failed with status {response.status}",
                            "service_url": self.service_url
                        }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_url": self.service_url
            }
