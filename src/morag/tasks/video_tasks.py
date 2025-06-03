"""Celery tasks for video processing."""

from typing import Dict, Any, List, Optional
import structlog
import asyncio
from pathlib import Path

from morag.core.celery_app import celery_app
from morag.tasks.base import ProcessingTask
from morag.processors.video import video_processor, VideoConfig
from morag.services.ffmpeg_service import ffmpeg_service
from morag.tasks.audio_tasks import process_audio_file
from morag.services.embedding import gemini_service
from morag.services.storage import qdrant_service

logger = structlog.get_logger()

async def _process_video_file_impl(
    self,
    file_path: str,
    task_id: str,
    config: Optional[Dict[str, Any]] = None,
    process_audio: bool = True
) -> Dict[str, Any]:
    """Process video file with audio extraction and thumbnail generation."""

    logger.info("Starting video processing task",
               task_id=task_id,
               file_path=file_path)

    try:
        self.update_status("PROCESSING", {"stage": "video_analysis"})

        # Parse configuration
        video_config = VideoConfig()
        if config:
            for key, value in config.items():
                if hasattr(video_config, key):
                    setattr(video_config, key, value)

        # Process video
        video_result = await video_processor.process_video(
            Path(file_path),
            video_config
        )

        result = {
            "video_metadata": {
                "duration": video_result.metadata.duration,
                "width": video_result.metadata.width,
                "height": video_result.metadata.height,
                "fps": video_result.metadata.fps,
                "codec": video_result.metadata.codec,
                "format": video_result.metadata.format,
                "has_audio": video_result.metadata.has_audio,
                "file_size": video_result.metadata.file_size
            },
            "thumbnails": [str(path) for path in video_result.thumbnails],
            "keyframes": [str(path) for path in video_result.keyframes],
            "processing_time": video_result.processing_time,
            "audio_processing_result": None
        }

        # Process extracted audio if available and requested
        if process_audio and video_result.audio_path and video_result.metadata.has_audio:
            self.update_status("PROCESSING", {"stage": "audio_processing"})

            logger.info("Processing extracted audio",
                       task_id=task_id,
                       audio_path=str(video_result.audio_path))

            # Process audio using existing audio pipeline
            # Note: process_audio_file is now synchronous, so we call it directly
            audio_result = process_audio_file(
                str(video_result.audio_path),
                f"{task_id}_audio",
                config={"model_size": "base"},
                use_enhanced_summary=True
            )

            result["audio_processing_result"] = audio_result

        self.update_status("SUCCESS", result)

        logger.info("Video processing task completed",
                   task_id=task_id,
                   has_audio=video_result.metadata.has_audio,
                   thumbnails_count=len(video_result.thumbnails),
                   keyframes_count=len(video_result.keyframes))

        # Clean up temporary files
        video_processor.cleanup_temp_files(video_result.temp_files)

        return result

    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        logger.error("Video processing task failed",
                    task_id=task_id,
                    error=str(e))

        self.update_status("FAILURE", {"error": error_msg})
        raise


@celery_app.task(bind=True, base=ProcessingTask)
def process_video_file(
    self,
    file_path: str,
    task_id: str,
    config: Optional[Dict[str, Any]] = None,
    process_audio: bool = True
) -> Dict[str, Any]:
    """Process video file with audio extraction and thumbnail generation."""
    return asyncio.run(_process_video_file_impl(self, file_path, task_id, config, process_audio))

async def _extract_video_audio_impl(
    self,
    file_path: str,
    task_id: str,
    audio_format: str = "wav"
) -> Dict[str, Any]:
    """Extract audio from video file."""

    logger.info("Starting video audio extraction",
               task_id=task_id,
               file_path=file_path,
               audio_format=audio_format)

    try:
        self.update_status("PROCESSING", {"stage": "audio_extraction"})

        # Extract audio using FFmpeg service
        audio_path = await ffmpeg_service.extract_audio(
            Path(file_path),
            output_format=audio_format
        )

        result = {
            "audio_path": str(audio_path),
            "audio_format": audio_format,
            "file_size": audio_path.stat().st_size
        }

        self.update_status("SUCCESS", result)

        logger.info("Video audio extraction completed",
                   task_id=task_id,
                   audio_path=str(audio_path))

        return result

    except Exception as e:
        error_msg = f"Video audio extraction failed: {str(e)}"
        logger.error("Video audio extraction task failed",
                    task_id=task_id,
                    error=str(e))

        self.update_status("FAILURE", {"error": error_msg})
        raise


@celery_app.task(bind=True, base=ProcessingTask)
def extract_video_audio(
    self,
    file_path: str,
    task_id: str,
    audio_format: str = "wav"
) -> Dict[str, Any]:
    """Extract audio from video file."""
    return asyncio.run(_extract_video_audio_impl(self, file_path, task_id, audio_format))

async def _generate_video_thumbnails_impl(
    self,
    file_path: str,
    task_id: str,
    count: int = 5,
    size: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Generate thumbnails from video file."""

    logger.info("Starting video thumbnail generation",
               task_id=task_id,
               file_path=file_path,
               count=count)

    try:
        self.update_status("PROCESSING", {"stage": "thumbnail_generation"})

        # Set default size if not provided
        if size is None:
            size = [320, 240]

        # Generate thumbnails using FFmpeg service
        thumbnails = await ffmpeg_service.generate_thumbnails(
            Path(file_path),
            count=count,
            size=(size[0], size[1])
        )

        result = {
            "thumbnails": [str(path) for path in thumbnails],
            "count": len(thumbnails),
            "size": size
        }

        self.update_status("SUCCESS", result)

        logger.info("Video thumbnail generation completed",
                   task_id=task_id,
                   thumbnails_count=len(thumbnails))

        return result

    except Exception as e:
        error_msg = f"Video thumbnail generation failed: {str(e)}"
        logger.error("Video thumbnail generation task failed",
                    task_id=task_id,
                    error=str(e))

        self.update_status("FAILURE", {"error": error_msg})
        raise


@celery_app.task(bind=True, base=ProcessingTask)
def generate_video_thumbnails(
    self,
    file_path: str,
    task_id: str,
    count: int = 5,
    size: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Generate thumbnails from video file."""
    return asyncio.run(_generate_video_thumbnails_impl(self, file_path, task_id, count, size))
