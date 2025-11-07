"""Video processor wrapper for stage processing."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import structlog

from .interface import ProcessorResult, StageProcessor

logger = structlog.get_logger(__name__)

# Import core exceptions
try:
    from morag_core.exceptions import ProcessingError
except ImportError:

    class ProcessingError(Exception):  # type: ignore
        pass


class VideoStageProcessor(StageProcessor):
    """Stage processor for video content using morag_video package."""

    def __init__(self):
        """Initialize video stage processor."""
        self._video_processor = None
        self._services = None

    def _get_video_processor(self):
        """Get or create video processor instance."""
        if self._video_processor is None:
            try:
                from morag_video import VideoConfig, VideoProcessor

                config = VideoConfig(
                    extract_audio=True,
                    generate_thumbnails=False,
                    extract_keyframes=False,
                )
                self._video_processor = VideoProcessor(config)
            except ImportError as e:
                raise ProcessingError(f"Video processor not available: {e}")
        return self._video_processor

    def _get_services(self):
        """Get MoRAG services for video processing."""
        if self._services is None:
            try:
                from morag_services import MoRAGServices

                self._services = MoRAGServices()
            except ImportError as e:
                raise ProcessingError(f"MoRAG services not available: {e}")
        return self._services

    def supports_content_type(self, content_type: str) -> bool:
        """Check if this processor supports the given content type."""
        return content_type.upper() == "VIDEO"

    async def process(
        self, input_file: Path, output_file: Path, config: Dict[str, Any]
    ) -> ProcessorResult:
        """Process video file to markdown."""
        logger.info("Processing video file", input_file=str(input_file))

        try:
            # Try using morag_video processor first
            try:
                processor = self._get_video_processor()
                result = await processor.process_video(input_file)

                # Extract transcript from audio processing result
                transcript = ""
                if result.audio_processing_result and hasattr(
                    result.audio_processing_result, "transcript"
                ):
                    transcript = result.audio_processing_result.transcript

                metadata = {
                    "title": result.metadata.title or input_file.stem,
                    "source": str(input_file),
                    "type": "video",
                    "duration": result.metadata.duration,
                    "format": result.metadata.format,
                    "resolution": f"{result.metadata.width}x{result.metadata.height}"
                    if result.metadata.width
                    else None,
                    "fps": result.metadata.fps,
                    "created_at": datetime.now().isoformat(),
                }

                content = f"\n# Video Analysis\n\n"
                if transcript:
                    content += f"## Transcript\n\n{transcript}\n\n"

                content += f"## Video Information\n\n"
                content += f"- **Duration**: {result.metadata.duration}\n"
                content += f"- **Format**: {result.metadata.format}\n"
                if result.metadata.width:
                    content += f"- **Resolution**: {result.metadata.width}x{result.metadata.height}\n"
                if result.metadata.fps:
                    content += f"- **FPS**: {result.metadata.fps}\n"

                markdown_content = self.create_markdown_with_metadata(content, metadata)
                output_file.write_text(markdown_content, encoding="utf-8")

                return ProcessorResult(
                    content=content,
                    metadata=metadata,
                    metrics={
                        "duration": result.metadata.duration,
                        "transcript_length": len(transcript),
                        "has_timestamps": config.get("include_timestamps", True),
                    },
                    final_output_file=output_file,
                )

            except Exception as video_error:
                logger.warning(
                    "morag_video processor failed, trying MoRAG services",
                    error=str(video_error),
                )

                # Fallback to MoRAG services
                services = self._get_services()

                # Prepare options for video service
                options = {
                    "include_timestamps": config.get("include_timestamps", True),
                    "speaker_diarization": config.get("speaker_diarization", True),
                    "topic_segmentation": config.get("topic_segmentation", True),
                    "extract_thumbnails": config.get("extract_thumbnails", False),
                }

                # Use video service
                result = await services.process_video(str(input_file), options)

                metadata = {
                    "title": result.metadata.get("title") or input_file.stem,
                    "source": str(input_file),
                    "type": "video",
                    "duration": result.metadata.get("duration"),
                    "language": result.metadata.get("language"),
                    "created_at": datetime.now().isoformat(),
                    **result.metadata,
                }

                content = f"\n# Video Analysis\n\n"
                if result.text_content:
                    content += f"## Transcript\n\n{result.text_content}\n\n"

                markdown_content = self.create_markdown_with_metadata(content, metadata)
                output_file.write_text(markdown_content, encoding="utf-8")

                return ProcessorResult(
                    content=content,
                    metadata=metadata,
                    metrics={
                        "duration": result.metadata.get("duration"),
                        "transcript_length": len(result.text_content or ""),
                        "has_timestamps": config.get("include_timestamps", True),
                    },
                    final_output_file=output_file,
                )

        except Exception as e:
            logger.error(
                "Video processing failed", input_file=str(input_file), error=str(e)
            )
            raise ProcessingError(f"Video processing failed for {input_file}: {e}")
