"""Audio processor wrapper for stage processing."""

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


class AudioStageProcessor(StageProcessor):
    """Stage processor for audio content using morag_audio package."""

    def __init__(self):
        """Initialize audio stage processor."""
        self._audio_processor = None
        self._services = None

    def _get_audio_processor(self):
        """Get or create audio processor instance."""
        if self._audio_processor is None:
            try:
                from morag_audio import AudioConfig, AudioProcessor

                config = AudioConfig(
                    transcribe=True, speaker_diarization=True, noise_reduction=True
                )
                self._audio_processor = AudioProcessor(config)
            except ImportError as e:
                raise ProcessingError(f"Audio processor not available: {e}")
        return self._audio_processor

    def _get_services(self):
        """Get MoRAG services for audio processing."""
        if self._services is None:
            try:
                from morag_services import MoRAGServices

                self._services = MoRAGServices()
            except ImportError as e:
                raise ProcessingError(f"MoRAG services not available: {e}")
        return self._services

    def supports_content_type(self, content_type: str) -> bool:
        """Check if this processor supports the given content type."""
        return content_type.upper() == "AUDIO"

    async def process(
        self, input_file: Path, output_file: Path, config: Dict[str, Any]
    ) -> ProcessorResult:
        """Process audio file to markdown."""
        logger.info("Processing audio file", input_file=str(input_file))

        try:
            # Try using morag_audio processor first
            try:
                processor = self._get_audio_processor()
                result = await processor.process_audio(input_file)

                metadata = {
                    "title": result.metadata.title or input_file.stem,
                    "source": str(input_file),
                    "type": "audio",
                    "duration": result.metadata.duration,
                    "format": result.metadata.format,
                    "sample_rate": result.metadata.sample_rate,
                    "channels": result.metadata.channels,
                    "language": result.metadata.language,
                    "created_at": datetime.now().isoformat(),
                }

                content = f"\n# Audio Analysis\n\n"
                if result.transcript:
                    content += f"## Transcript\n\n{result.transcript}\n\n"

                if result.speakers and len(result.speakers) > 1:
                    content += f"## Speaker Information\n\n"
                    for i, speaker in enumerate(result.speakers):
                        content += f"- **Speaker {i+1}**: {speaker.get('duration', 'Unknown duration')}\n"
                    content += "\n"

                content += f"## Audio Information\n\n"
                content += f"- **Duration**: {result.metadata.duration}\n"
                content += f"- **Format**: {result.metadata.format}\n"
                content += f"- **Sample Rate**: {result.metadata.sample_rate}\n"
                content += f"- **Channels**: {result.metadata.channels}\n"
                if result.metadata.language:
                    content += f"- **Language**: {result.metadata.language}\n"

                markdown_content = self.create_markdown_with_metadata(content, metadata)
                output_file.write_text(markdown_content, encoding="utf-8")

                return ProcessorResult(
                    content=content,
                    metadata=metadata,
                    metrics={
                        "duration": result.metadata.duration,
                        "transcript_length": len(result.transcript or ""),
                        "has_timestamps": config.get("include_timestamps", True),
                    },
                    final_output_file=output_file,
                )

            except Exception as audio_error:
                logger.warning(
                    "morag_audio processor failed, trying MoRAG services",
                    error=str(audio_error),
                )

                # Fallback to MoRAG services
                services = self._get_services()

                # Prepare options for audio service
                options = {
                    "include_timestamps": config.get("include_timestamps", True),
                    "speaker_diarization": config.get("speaker_diarization", True),
                    "noise_reduction": config.get("noise_reduction", True),
                    "language": config.get("language", "auto"),
                }

                # Use audio service
                result = await services.process_audio(str(input_file), options)

                metadata = {
                    "title": result.metadata.get("title") or input_file.stem,
                    "source": str(input_file),
                    "type": "audio",
                    "duration": result.metadata.get("duration"),
                    "language": result.metadata.get("language"),
                    "created_at": datetime.now().isoformat(),
                    **result.metadata,
                }

                content = f"\n# Audio Analysis\n\n"
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
                "Audio processing failed", input_file=str(input_file), error=str(e)
            )
            raise ProcessingError(f"Audio processing failed for {input_file}: {e}")
