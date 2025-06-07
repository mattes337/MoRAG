"""YouTube processing tasks for MoRAG."""

from typing import Dict, Any, List, Optional
import structlog
from pathlib import Path

from morag_core.interfaces.task import BaseTask

from .processor import YouTubeProcessor, YouTubeConfig, YouTubeDownloadResult

logger = structlog.get_logger(__name__)


class YouTubeProcessingTask(BaseTask):
    """Base class for YouTube processing tasks."""
    
    def __init__(self):
        super().__init__()
        self.youtube_processor = YouTubeProcessor()


class ProcessYouTubeVideoTask(YouTubeProcessingTask):
    """Task for processing a single YouTube video."""
    
    async def execute(
        self,
        url: str,
        config: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        store_embeddings: bool = True
    ) -> Dict[str, Any]:
        """Process YouTube video with download and metadata extraction."""
        
        logger.info("Starting YouTube video processing task",
                   task_id=task_id,
                   url=url)
        
        try:
            await self.update_status("PROCESSING", {"stage": "youtube_download"})
            
            # Parse configuration
            youtube_config = YouTubeConfig()
            if config:
                for key, value in config.items():
                    if hasattr(youtube_config, key):
                        setattr(youtube_config, key, value)
            
            # Process YouTube video
            youtube_result = await self.youtube_processor.process_url(url, youtube_config)
            
            result = {
                "metadata": {
                    "id": youtube_result.metadata.id,
                    "title": youtube_result.metadata.title,
                    "description": youtube_result.metadata.description,
                    "uploader": youtube_result.metadata.uploader,
                    "upload_date": youtube_result.metadata.upload_date,
                    "duration": youtube_result.metadata.duration,
                    "view_count": youtube_result.metadata.view_count,
                    "like_count": youtube_result.metadata.like_count,
                    "comment_count": youtube_result.metadata.comment_count,
                    "tags": youtube_result.metadata.tags,
                    "categories": youtube_result.metadata.categories,
                    "channel_id": youtube_result.metadata.channel_id,
                    "playlist_id": youtube_result.metadata.playlist_id,
                    "playlist_title": youtube_result.metadata.playlist_title,
                    "playlist_index": youtube_result.metadata.playlist_index
                },
                "files": {
                    "video_path": str(youtube_result.video_path) if youtube_result.video_path else None,
                    "audio_path": str(youtube_result.audio_path) if youtube_result.audio_path else None,
                    "subtitle_paths": [str(p) for p in youtube_result.subtitle_paths],
                    "thumbnail_paths": [str(p) for p in youtube_result.thumbnail_paths]
                },
                "processing_time": youtube_result.processing_time,
                "file_size": youtube_result.file_size,
                "success": youtube_result.success
            }
            
            await self.update_status("SUCCESS", result)
            
            logger.info("YouTube video processing task completed",
                       task_id=task_id,
                       video_id=youtube_result.metadata.id,
                       title=youtube_result.metadata.title)
            
            # Clean up temporary files
            self.youtube_processor.cleanup(youtube_result)
            
            return result
            
        except Exception as e:
            error_msg = f"YouTube video processing failed: {str(e)}"
            logger.error("YouTube video processing task failed",
                        task_id=task_id,
                        error=str(e))
            
            await self.update_status("FAILURE", {"error": error_msg})
            return {
                "success": False,
                "error": error_msg,
                "url": url
            }


class ProcessYouTubePlaylistTask(YouTubeProcessingTask):
    """Task for processing a YouTube playlist."""
    
    async def execute(
        self,
        url: str,
        config: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        store_embeddings: bool = True
    ) -> Dict[str, Any]:
        """Process YouTube playlist with download and metadata extraction."""
        
        logger.info("Starting YouTube playlist processing task",
                   task_id=task_id,
                   url=url)
        
        try:
            await self.update_status("PROCESSING", {"stage": "playlist_processing"})
            
            # Parse configuration
            youtube_config = YouTubeConfig()
            if config:
                for key, value in config.items():
                    if hasattr(youtube_config, key):
                        setattr(youtube_config, key, value)
            
            # Process YouTube playlist
            playlist_results = await self.youtube_processor.process_playlist(url, youtube_config)
            
            result = {
                "playlist_url": url,
                "total_videos": len(playlist_results),
                "successful_downloads": len([r for r in playlist_results if r.success]),
                "total_processing_time": sum(r.processing_time for r in playlist_results),
                "total_file_size": sum(r.file_size for r in playlist_results),
                "videos": [],
                "success": True
            }
            
            # Process each video result
            for video_result in playlist_results:
                video_data = {
                    "metadata": {
                        "id": video_result.metadata.id,
                        "title": video_result.metadata.title,
                        "uploader": video_result.metadata.uploader,
                        "duration": video_result.metadata.duration,
                        "view_count": video_result.metadata.view_count,
                        "playlist_index": video_result.metadata.playlist_index
                    },
                    "files": {
                        "video_path": str(video_result.video_path) if video_result.video_path else None,
                        "audio_path": str(video_result.audio_path) if video_result.audio_path else None,
                        "subtitle_count": len(video_result.subtitle_paths),
                        "thumbnail_count": len(video_result.thumbnail_paths)
                    },
                    "processing_time": video_result.processing_time,
                    "file_size": video_result.file_size,
                    "success": video_result.success
                }
                result["videos"].append(video_data)
                
                # Clean up temporary files for this video
                self.youtube_processor.cleanup(video_result)
            
            await self.update_status("SUCCESS", result)
            
            logger.info("YouTube playlist processing task completed",
                       task_id=task_id,
                       total_videos=result["total_videos"],
                       successful_downloads=result["successful_downloads"])
            
            return result
            
        except Exception as e:
            error_msg = f"YouTube playlist processing failed: {str(e)}"
            logger.error("YouTube playlist processing task failed",
                        task_id=task_id,
                        error=str(e))
            
            await self.update_status("FAILURE", {"error": error_msg})
            return {
                "success": False,
                "error": error_msg,
                "url": url
            }


class ExtractYouTubeMetadataTask(YouTubeProcessingTask):
    """Task for extracting YouTube metadata without downloading."""
    
    async def execute(
        self,
        url: str,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract YouTube metadata without downloading."""
        
        logger.info("Starting YouTube metadata extraction task",
                   task_id=task_id,
                   url=url)
        
        try:
            await self.update_status("PROCESSING", {"stage": "metadata_extraction"})
            
            # Configure for metadata only
            config = YouTubeConfig(extract_metadata_only=True)
            
            # Extract metadata
            youtube_result = await self.youtube_processor.process_url(url, config)
            
            result = {
                "metadata": {
                    "id": youtube_result.metadata.id,
                    "title": youtube_result.metadata.title,
                    "description": youtube_result.metadata.description,
                    "uploader": youtube_result.metadata.uploader,
                    "upload_date": youtube_result.metadata.upload_date,
                    "duration": youtube_result.metadata.duration,
                    "view_count": youtube_result.metadata.view_count,
                    "like_count": youtube_result.metadata.like_count,
                    "comment_count": youtube_result.metadata.comment_count,
                    "tags": youtube_result.metadata.tags,
                    "categories": youtube_result.metadata.categories,
                    "thumbnail_url": youtube_result.metadata.thumbnail_url,
                    "webpage_url": youtube_result.metadata.webpage_url,
                    "channel_id": youtube_result.metadata.channel_id,
                    "channel_url": youtube_result.metadata.channel_url,
                    "playlist_id": youtube_result.metadata.playlist_id,
                    "playlist_title": youtube_result.metadata.playlist_title,
                    "playlist_index": youtube_result.metadata.playlist_index
                },
                "processing_time": youtube_result.processing_time,
                "success": True
            }
            
            await self.update_status("SUCCESS", result)
            
            logger.info("YouTube metadata extraction task completed",
                       task_id=task_id,
                       video_id=youtube_result.metadata.id,
                       title=youtube_result.metadata.title)
            
            return result
            
        except Exception as e:
            error_msg = f"YouTube metadata extraction failed: {str(e)}"
            logger.error("YouTube metadata extraction task failed",
                        task_id=task_id,
                        error=str(e))
            
            await self.update_status("FAILURE", {"error": error_msg})
            return {
                "success": False,
                "error": error_msg,
                "url": url
            }


# Convenience functions for external use
async def process_youtube_video(
    url: str,
    config: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    store_embeddings: bool = True
) -> Dict[str, Any]:
    """Process a single YouTube video."""
    task = ProcessYouTubeVideoTask()
    return await task.execute(url, config, task_id, store_embeddings)


async def process_youtube_playlist(
    url: str,
    config: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    store_embeddings: bool = True
) -> Dict[str, Any]:
    """Process a YouTube playlist."""
    task = ProcessYouTubePlaylistTask()
    return await task.execute(url, config, task_id, store_embeddings)


async def extract_youtube_metadata(
    url: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """Extract YouTube metadata without downloading."""
    task = ExtractYouTubeMetadataTask()
    return await task.execute(url, task_id)


# For backward compatibility with Celery-based systems
def create_celery_tasks(celery_app):
    """Create Celery task wrappers for YouTube processing."""
    
    @celery_app.task(bind=True)
    def process_youtube_video_celery(self, url: str, config: Optional[Dict[str, Any]] = None):
        """Celery wrapper for YouTube video processing."""
        import asyncio
        return asyncio.run(process_youtube_video(url, config, self.request.id))
    
    @celery_app.task(bind=True)
    def process_youtube_playlist_celery(self, url: str, config: Optional[Dict[str, Any]] = None):
        """Celery wrapper for YouTube playlist processing."""
        import asyncio
        return asyncio.run(process_youtube_playlist(url, config, self.request.id))
    
    @celery_app.task(bind=True)
    def extract_youtube_metadata_celery(self, url: str):
        """Celery wrapper for YouTube metadata extraction."""
        import asyncio
        return asyncio.run(extract_youtube_metadata(url, self.request.id))
    
    return {
        'process_youtube_video': process_youtube_video_celery,
        'process_youtube_playlist': process_youtube_playlist_celery,
        'extract_youtube_metadata': extract_youtube_metadata_celery
    }
