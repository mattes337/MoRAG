"""Audio service module for MoRAG."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import structlog

from morag_core.exceptions import ProcessingError as ServiceError
from morag_core.utils import ensure_directory as ensure_directory_exists
from morag_embedding import GeminiEmbeddingService

from morag_audio.processor import AudioProcessor, AudioConfig, AudioProcessingResult, AudioProcessingError
from morag_audio.converters import AudioConverter, AudioConversionOptions

logger = structlog.get_logger(__name__)


class AudioServiceError(ServiceError):
    """Error raised by the AudioService."""
    pass


class AudioService:
    """High-level service for audio processing and conversion.
    
    This service provides a unified interface for processing audio files,
    generating transcriptions, and converting to markdown format.
    """

    def __init__(self, 
                 config: Optional[AudioConfig] = None,
                 embedding_service: Optional[GeminiEmbeddingService] = None,
                 output_dir: Optional[Union[str, Path]] = None):
        """Initialize the audio service.
        
        Args:
            config: Configuration for audio processing
            embedding_service: Optional embedding service for enhanced features
            output_dir: Directory to store processed files
        """
        self.config = config or AudioConfig()
        self.processor = AudioProcessor(self.config)
        self.converter = AudioConverter()
        self.embedding_service = embedding_service
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            ensure_directory_exists(self.output_dir)
        else:
            self.output_dir = None
        
        logger.info("Audio service initialized",
                   enable_diarization=self.config.enable_diarization,
                   enable_topic_segmentation=self.config.enable_topic_segmentation,
                   has_embedding_service=self.embedding_service is not None)

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status information
        """
        try:
            # Check if processor is available
            if not self.processor:
                return {
                    "status": "unhealthy",
                    "error": "Audio processor not initialized"
                }

            # Check if transcriber is available
            if not hasattr(self.processor, 'transcriber') or not self.processor.transcriber:
                return {
                    "status": "unhealthy",
                    "error": "Whisper transcriber not initialized"
                }

            return {
                "status": "healthy",
                "processor": "ready",
                "transcriber": "ready",
                "diarization_enabled": self.config.enable_diarization,
                "topic_segmentation_enabled": self.config.enable_topic_segmentation,
                "embedding_service": self.embedding_service is not None
            }

        except Exception as e:
            logger.error("Audio service health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def process_file(self,
                         file_path: Union[str, Path],
                         save_output: bool = True,
                         output_format: str = "markdown",
                         progress_callback: callable = None) -> Dict[str, Any]:
        """Process an audio file and optionally save the results.
        
        Args:
            file_path: Path to the audio file
            save_output: Whether to save the output files
            output_format: Format to save the output (markdown, json, txt)
            
        Returns:
            Dictionary containing processing results and output file paths
            
        Raises:
            AudioServiceError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AudioServiceError(f"File not found: {file_path}")
        
        logger.info("Processing audio file", 
                   file_path=str(file_path),
                   save_output=save_output,
                   output_format=output_format)
        
        try:
            # Process the audio file
            result = await self.processor.process(file_path, progress_callback)
            
            # Generate embeddings if embedding service is available
            if self.embedding_service and result.success:
                try:
                    # Generate embeddings for the full transcript
                    transcript_embedding = await self.embedding_service.embed_text(result.transcript)
                    result.metadata["transcript_embedding"] = transcript_embedding.tolist()
                    
                    # Generate embeddings for each segment
                    segment_texts = [segment.text for segment in result.segments]
                    segment_embeddings = await self.embedding_service.embed_batch(segment_texts)
                    
                    # Add embeddings to segments
                    for i, segment in enumerate(result.segments):
                        segment.embedding = segment_embeddings[i].tolist()
                        
                    logger.info("Generated embeddings for transcript and segments",
                              transcript_length=len(result.transcript),
                              segment_count=len(result.segments))
                except Exception as e:
                    logger.warning("Failed to generate embeddings", error=str(e))
            
            # Convert to requested format
            formatted_content = None
            if result.success:
                conversion_options = AudioConversionOptions(
                    include_timestamps=True,
                    include_speakers=self.config.enable_diarization,
                    include_topics=self.config.enable_topic_segmentation,
                    group_by_topic=self.config.enable_topic_segmentation,
                    group_by_speaker=False  # Use per-line timestamps as per new format
                )

                if output_format == "markdown":
                    markdown_result = await self.converter.convert_to_markdown(result, conversion_options)
                    formatted_content = markdown_result.content
                elif output_format == "json":
                    formatted_content = await self.converter.convert_to_json(result, conversion_options)
            
            # Save output files if requested
            output_files = {}
            if save_output and result.success and self.output_dir:
                output_files = await self._save_output_files(file_path, result, formatted_content if output_format == "markdown" else None, output_format)

            processing_time = time.time() - start_time

            # Prepare response
            response = {
                "success": result.success,
                "processing_time": processing_time,
                "metadata": result.metadata,
                "output_files": output_files
            }

            if not save_output or not self.output_dir:
                if output_format == "markdown" and formatted_content:
                    response["content"] = formatted_content
                elif output_format == "json" and formatted_content:
                    response["content"] = formatted_content
                elif output_format == "txt":
                    response["content"] = result.transcript
            
            logger.info("Audio processing completed",
                       file_path=str(file_path),
                       processing_time=processing_time,
                       success=result.success)
            
            return response
            
        except Exception as e:
            logger.error("Audio processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            
            processing_time = time.time() - start_time
            return {
                "success": False,
                "processing_time": processing_time,
                "error": str(e),
                "output_files": {}
            }
    
    async def _save_output_files(self,
                               file_path: Path,
                               result: AudioProcessingResult,
                               markdown_content: Optional[str],
                               output_format: str) -> Dict[str, str]:
        """Save output files and return their paths."""
        if not self.output_dir:
            return {}

        output_files = {}
        base_name = file_path.stem

        # Sanitize the base name for directory creation
        # Remove problematic characters and limit length
        sanitized_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if len(sanitized_name) > 100:  # Limit directory name length
            sanitized_name = sanitized_name[:100].strip()
        if not sanitized_name:  # Fallback if name becomes empty
            sanitized_name = f"audio_{hash(base_name) % 10000}"

        # Create a subdirectory for this file
        file_output_dir = self.output_dir / sanitized_name
        try:
            ensure_directory_exists(file_output_dir)
            logger.debug("Created output directory", output_dir=str(file_output_dir))
        except Exception as e:
            logger.error("Failed to create output directory",
                        output_dir=str(file_output_dir),
                        error=str(e))
            raise AudioServiceError(f"Failed to create output directory: {str(e)}")

        # Save transcript as text
        if result.transcript:
            transcript_path = file_output_dir / f"{sanitized_name}_transcript.txt"
            try:
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(result.transcript)
                output_files["transcript"] = str(transcript_path)
                logger.debug("Saved transcript", path=str(transcript_path))
            except Exception as e:
                logger.error("Failed to save transcript",
                           path=str(transcript_path),
                           error=str(e))
                raise AudioServiceError(f"Failed to save transcript: {str(e)}")

        # Save segments as JSON if available
        if result.segments:
            import json
            segments_path = file_output_dir / f"{sanitized_name}_segments.json"
            segments_data = []

            for segment in result.segments:
                segment_data = {
                    "start": getattr(segment, 'start', 0),
                    "end": getattr(segment, 'end', 0),
                    "text": getattr(segment, 'text', ''),
                    "speaker": getattr(segment, 'speaker', None),
                    "confidence": getattr(segment, 'confidence', None),
                    "topic_id": getattr(segment, 'topic_id', None),
                    "topic_label": getattr(segment, 'topic_label', None)
                }
                segments_data.append(segment_data)

            try:
                segments_path.write_text(json.dumps(segments_data, indent=2, ensure_ascii=False), encoding='utf-8')
                output_files["segments"] = str(segments_path)
                logger.info("Saved segments file",
                           segments_count=len(segments_data),
                           segments_path=str(segments_path))
            except Exception as e:
                logger.error("Failed to save segments",
                           path=str(segments_path),
                           error=str(e))

        # Save markdown if available
        if markdown_content:
            markdown_path = file_output_dir / f"{sanitized_name}.md"
            try:
                with open(markdown_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                output_files["markdown"] = str(markdown_path)
                logger.debug("Saved markdown", path=str(markdown_path))
            except Exception as e:
                logger.error("Failed to save markdown",
                           path=str(markdown_path),
                           error=str(e))

        # Save metadata as JSON
        import json
        metadata_path = file_output_dir / f"{sanitized_name}_metadata.json"
        metadata_dict = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "processing_time": result.processing_time,
            "transcript_length": len(result.transcript) if result.transcript else 0,
            "segments_count": len(result.segments) if result.segments else 0,
            "has_speaker_diarization": any(getattr(seg, 'speaker', None) for seg in result.segments) if result.segments else False,
            "has_topic_segmentation": any(getattr(seg, 'topic_id', None) is not None for seg in result.segments) if result.segments else False,
            "metadata": result.metadata
        }
        try:
            metadata_path.write_text(json.dumps(metadata_dict, indent=2, ensure_ascii=False), encoding='utf-8')
            output_files["metadata"] = str(metadata_path)
            logger.debug("Saved metadata", path=str(metadata_path))
        except Exception as e:
            logger.error("Failed to save metadata",
                       path=str(metadata_path),
                       error=str(e))

        
        logger.info("Saved output files", 
                   output_dir=str(file_output_dir),
                   files=list(output_files.keys()))
        
        return output_files