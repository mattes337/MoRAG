"""Utility functions for MoRAG services."""

import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.parse import urlparse
import structlog

logger = structlog.get_logger(__name__)


class ServiceUtilities:
    """Utility functions for content type detection and service operations."""

    @staticmethod
    def is_document(path_or_url: str) -> bool:
        """Check if path/URL is a document."""
        path = Path(path_or_url)
        document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', 
                              '.odt', '.pages', '.epub', '.mobi', '.djvu',
                              '.xls', '.xlsx', '.csv', '.ppt', '.pptx'}
        return path.suffix.lower() in document_extensions

    @staticmethod
    def is_audio(path_or_url: str) -> bool:
        """Check if path/URL is audio."""
        path = Path(path_or_url)
        audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma', 
                           '.flac', '.aiff', '.au', '.ra', '.3gp', '.amr'}
        return path.suffix.lower() in audio_extensions

    @staticmethod
    def is_video(path_or_url: str) -> bool:
        """Check if path/URL is video."""
        path = Path(path_or_url)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
                           '.webm', '.m4v', '.3gp', '.ogv', '.ts', '.mts'}
        return path.suffix.lower() in video_extensions

    @staticmethod
    def is_image(path_or_url: str) -> bool:
        """Check if path/URL is an image."""
        path = Path(path_or_url)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', 
                           '.tif', '.webp', '.svg', '.ico', '.heic', '.heif'}
        return path.suffix.lower() in image_extensions

    @staticmethod
    def is_web(path_or_url: str) -> bool:
        """Check if this is a web URL (but not YouTube)."""
        return path_or_url.startswith(('http://', 'https://')) and not ServiceUtilities.is_youtube(path_or_url)

    @staticmethod
    def is_youtube(path_or_url: str) -> bool:
        """Check if this is a YouTube URL."""
        youtube_patterns = [r'youtube\.com', r'youtu\.be', r'youtube-nocookie\.com']
        return any(re.search(pattern, path_or_url) for pattern in youtube_patterns)

    @staticmethod
    def detect_content_type(path_or_url: str) -> str:
        """Detect content type based on path or URL.
        
        Args:
            path_or_url: Path to file or URL
            
        Returns:
            Detected content type string
        """
        # Check in order of specificity
        if ServiceUtilities.is_youtube(path_or_url):
            return "youtube"
        elif ServiceUtilities.is_web(path_or_url):
            return "web"
        elif ServiceUtilities.is_document(path_or_url):
            return "document"
        elif ServiceUtilities.is_audio(path_or_url):
            return "audio"
        elif ServiceUtilities.is_video(path_or_url):
            return "video"
        elif ServiceUtilities.is_image(path_or_url):
            return "image"
        else:
            return "unknown"

    @staticmethod
    def validate_path_or_url(path_or_url: str) -> bool:
        """Validate if path exists or URL is well-formed.
        
        Args:
            path_or_url: Path or URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if it's a URL
            if path_or_url.startswith(('http://', 'https://')):
                parsed = urlparse(path_or_url)
                return bool(parsed.netloc and parsed.scheme)
            
            # Check if it's a local path
            path = Path(path_or_url)
            return path.exists() and path.is_file()
            
        except Exception as e:
            logger.debug("Path/URL validation failed", path=path_or_url, error=str(e))
            return False

    @staticmethod
    def get_file_info(path_or_url: str) -> Dict[str, Any]:
        """Get basic file information.
        
        Args:
            path_or_url: Path or URL
            
        Returns:
            Dictionary with file information
        """
        info = {
            "path_or_url": path_or_url,
            "content_type": ServiceUtilities.detect_content_type(path_or_url),
            "is_url": path_or_url.startswith(('http://', 'https://')),
            "is_local_file": False,
            "exists": False,
            "size": None,
            "extension": None,
            "name": None
        }
        
        try:
            if info["is_url"]:
                parsed = urlparse(path_or_url)
                info["name"] = Path(parsed.path).name or parsed.netloc
                info["exists"] = True  # Assume URLs exist (will be validated during processing)
            else:
                path = Path(path_or_url)
                info["is_local_file"] = True
                info["exists"] = path.exists() and path.is_file()
                
                if info["exists"]:
                    info["size"] = path.stat().st_size
                    info["extension"] = path.suffix.lower()
                    info["name"] = path.name
                else:
                    # Still extract name and extension even if file doesn't exist
                    info["extension"] = path.suffix.lower()
                    info["name"] = path.name
                    
        except Exception as e:
            logger.debug("Failed to get file info", path=path_or_url, error=str(e))
        
        return info

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            max_name_length = 255 - len(ext)
            sanitized = name[:max_name_length] + ext
        
        return sanitized.strip()

    @staticmethod
    def estimate_processing_time(content_type: str, size_bytes: Optional[int] = None) -> float:
        """Estimate processing time based on content type and size.
        
        Args:
            content_type: Type of content
            size_bytes: Size in bytes (if known)
            
        Returns:
            Estimated processing time in seconds
        """
        base_times = {
            "document": 2.0,  # Base time for documents
            "audio": 10.0,    # Audio processing takes longer
            "video": 30.0,    # Video processing takes longest
            "image": 3.0,     # Image OCR processing
            "web": 5.0,       # Web scraping and processing
            "youtube": 20.0,  # YouTube download + processing
            "unknown": 5.0    # Default estimate
        }
        
        base_time = base_times.get(content_type, base_times["unknown"])
        
        if size_bytes is None:
            return base_time
        
        # Adjust based on file size (rough estimates)
        size_mb = size_bytes / (1024 * 1024)
        
        if content_type == "document":
            # ~0.5 seconds per MB for documents
            return base_time + (size_mb * 0.5)
        elif content_type in ["audio", "video"]:
            # ~2 seconds per MB for media files
            return base_time + (size_mb * 2.0)
        elif content_type == "image":
            # ~1 second per MB for images
            return base_time + (size_mb * 1.0)
        else:
            # Default scaling
            return base_time + (size_mb * 0.1)

    @staticmethod
    def format_processing_stats(stats: Dict[str, Any]) -> str:
        """Format processing statistics for logging.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Formatted statistics string
        """
        formatted_parts = []
        
        if "processing_time" in stats:
            formatted_parts.append(f"time: {stats['processing_time']:.2f}s")
        
        if "content_length" in stats:
            formatted_parts.append(f"length: {stats['content_length']:,} chars")
        
        if "chunk_count" in stats:
            formatted_parts.append(f"chunks: {stats['chunk_count']}")
        
        if "error_count" in stats:
            formatted_parts.append(f"errors: {stats['error_count']}")
        
        return ", ".join(formatted_parts)


__all__ = ["ServiceUtilities"]