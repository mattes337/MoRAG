"""Celery tasks for YouTube processing."""

from typing import Dict, Any, List, Optional
import structlog
from pathlib import Path

from morag.core.celery_app import celery_app
from morag.tasks.base import ProcessingTask
from morag.processors.youtube import youtube_processor, YouTubeConfig
from morag.services.embedding import gemini_service
from morag.services.storage import qdrant_service

logger = structlog.get_logger()

@celery_app.task(bind=True, base=ProcessingTask)
async def process_youtube_video(
    self,
    url: str,
    task_id: str,
    config: Optional[Dict[str, Any]] = None,
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
        youtube_result = await youtube_processor.process_url(url, youtube_config)
        
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
            "download_time": youtube_result.download_time,
            "file_size": youtube_result.file_size,
            "embeddings_stored": 0
        }
        
        # Store embeddings if requested
        if store_embeddings:
            await self.update_status("PROCESSING", {"stage": "embedding_generation"})
            
            # Combine title, description, and tags for embedding
            combined_text = f"Title: {youtube_result.metadata.title}\n"
            if youtube_result.metadata.description:
                combined_text += f"Description: {youtube_result.metadata.description}\n"
            if youtube_result.metadata.tags:
                combined_text += f"Tags: {', '.join(youtube_result.metadata.tags)}\n"
            
            # Generate embeddings
            embedding = await gemini_service.generate_embedding(combined_text)
            
            # Store in vector database
            metadata = {
                "source_type": "youtube",
                "video_id": youtube_result.metadata.id,
                "url": url,
                "title": youtube_result.metadata.title,
                "uploader": youtube_result.metadata.uploader,
                "upload_date": youtube_result.metadata.upload_date,
                "duration": youtube_result.metadata.duration,
                "view_count": youtube_result.metadata.view_count,
                "channel_id": youtube_result.metadata.channel_id,
                "has_video": youtube_result.video_path is not None,
                "has_audio": youtube_result.audio_path is not None,
                "has_subtitles": len(youtube_result.subtitle_paths) > 0,
                "download_time": youtube_result.download_time,
                "file_size": youtube_result.file_size
            }
            
            await qdrant_service.store_embedding(
                embedding=embedding,
                text=combined_text,
                metadata=metadata,
                collection_name="youtube"
            )
            
            result["embeddings_stored"] = 1
        
        await self.update_status("SUCCESS", result)
        
        logger.info("YouTube video processing task completed",
                   task_id=task_id,
                   video_id=youtube_result.metadata.id,
                   title=youtube_result.metadata.title,
                   embeddings_stored=result["embeddings_stored"])
        
        # Clean up temporary files
        youtube_processor.cleanup_temp_files(youtube_result.temp_files)
        
        return result
        
    except Exception as e:
        error_msg = f"YouTube video processing failed: {str(e)}"
        logger.error("YouTube video processing task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise

@celery_app.task(bind=True, base=ProcessingTask)
async def process_youtube_playlist(
    self,
    url: str,
    task_id: str,
    config: Optional[Dict[str, Any]] = None,
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
        playlist_results = await youtube_processor.process_playlist(url, youtube_config)
        
        result = {
            "playlist_url": url,
            "total_videos": len(playlist_results),
            "successful_downloads": len([r for r in playlist_results if r.video_path or r.audio_path]),
            "total_download_time": sum(r.download_time for r in playlist_results),
            "total_file_size": sum(r.file_size for r in playlist_results),
            "videos": [],
            "embeddings_stored": 0
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
                "download_time": video_result.download_time,
                "file_size": video_result.file_size
            }
            result["videos"].append(video_data)
            
            # Store embeddings if requested
            if store_embeddings:
                combined_text = f"Title: {video_result.metadata.title}\n"
                if video_result.metadata.description:
                    combined_text += f"Description: {video_result.metadata.description}\n"
                if video_result.metadata.tags:
                    combined_text += f"Tags: {', '.join(video_result.metadata.tags)}\n"
                
                # Generate embeddings
                embedding = await gemini_service.generate_embedding(combined_text)
                
                # Store in vector database
                metadata = {
                    "source_type": "youtube_playlist",
                    "video_id": video_result.metadata.id,
                    "playlist_url": url,
                    "playlist_id": video_result.metadata.playlist_id,
                    "playlist_title": video_result.metadata.playlist_title,
                    "playlist_index": video_result.metadata.playlist_index,
                    "title": video_result.metadata.title,
                    "uploader": video_result.metadata.uploader,
                    "duration": video_result.metadata.duration,
                    "view_count": video_result.metadata.view_count,
                    "channel_id": video_result.metadata.channel_id
                }
                
                await qdrant_service.store_embedding(
                    embedding=embedding,
                    text=combined_text,
                    metadata=metadata,
                    collection_name="youtube"
                )
                
                result["embeddings_stored"] += 1
            
            # Clean up temporary files for this video
            youtube_processor.cleanup_temp_files(video_result.temp_files)
        
        await self.update_status("SUCCESS", result)
        
        logger.info("YouTube playlist processing task completed",
                   task_id=task_id,
                   total_videos=result["total_videos"],
                   successful_downloads=result["successful_downloads"],
                   embeddings_stored=result["embeddings_stored"])
        
        return result
        
    except Exception as e:
        error_msg = f"YouTube playlist processing failed: {str(e)}"
        logger.error("YouTube playlist processing task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise

@celery_app.task(bind=True, base=ProcessingTask)
async def extract_youtube_metadata(
    self,
    url: str,
    task_id: str
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
        youtube_result = await youtube_processor.process_url(url, config)
        
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
            "extraction_time": youtube_result.download_time
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
        raise

@celery_app.task(bind=True, base=ProcessingTask)
async def download_youtube_audio(
    self,
    url: str,
    task_id: str,
    quality: str = "192"
) -> Dict[str, Any]:
    """Download YouTube audio only."""
    
    logger.info("Starting YouTube audio download task",
               task_id=task_id,
               url=url,
               quality=quality)
    
    try:
        await self.update_status("PROCESSING", {"stage": "audio_download"})
        
        # Configure for audio only
        config = YouTubeConfig(
            extract_audio=True,
            download_subtitles=False,
            download_thumbnails=False
        )
        
        # Download audio
        youtube_result = await youtube_processor.process_url(url, config)
        
        result = {
            "metadata": {
                "id": youtube_result.metadata.id,
                "title": youtube_result.metadata.title,
                "uploader": youtube_result.metadata.uploader,
                "duration": youtube_result.metadata.duration
            },
            "audio_path": str(youtube_result.audio_path) if youtube_result.audio_path else None,
            "download_time": youtube_result.download_time,
            "file_size": youtube_result.file_size
        }
        
        await self.update_status("SUCCESS", result)
        
        logger.info("YouTube audio download task completed",
                   task_id=task_id,
                   video_id=youtube_result.metadata.id,
                   audio_downloaded=youtube_result.audio_path is not None)
        
        # Note: Don't clean up temp files here as they may be needed by caller
        
        return result
        
    except Exception as e:
        error_msg = f"YouTube audio download failed: {str(e)}"
        logger.error("YouTube audio download task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise
