"""Video service module for MoRAG."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import structlog

from morag_core.exceptions import ProcessingError as ServiceError
from morag_core.utils import ensure_directory as ensure_directory_exists
from morag_embedding import EmbeddingService

from morag_video.processor import VideoProcessor, VideoConfig, VideoProcessingResult, VideoProcessingError
from morag_video.converters import VideoConverter, VideoConversionOptions

logger = structlog.get_logger(__name__)


class VideoServiceError(ServiceError):
    """Error raised by the VideoService."""
    pass


class VideoService:
    """High-level service for video processing and conversion.
    
    This service provides a unified interface for processing video files,
    generating transcriptions, extracting keyframes, and converting to markdown format.
    """

    def __init__(self, 
                 config: Optional[VideoConfig] = None,
                 embedding_service: Optional[EmbeddingService] = None,
                 output_dir: Optional[Union[str, Path]] = None):
        """Initialize the video service.
        
        Args:
            config: Configuration for video processing
            embedding_service: Optional embedding service for enhanced features
            output_dir: Directory to store processed files
        """
        self.config = config or VideoConfig()
        self.processor = VideoProcessor(self.config)
        self.converter = VideoConverter()
        self.embedding_service = embedding_service
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            ensure_directory_exists(self.output_dir)
        else:
            self.output_dir = None
        
        logger.info("Video service initialized",
                   extract_audio=self.config.extract_audio,
                   generate_thumbnails=self.config.generate_thumbnails,
                   extract_keyframes=self.config.extract_keyframes,
                   enable_enhanced_audio=self.config.enable_enhanced_audio,
                   enable_ocr=self.config.enable_ocr,
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
                    "error": "Video processor not initialized"
                }

            return {
                "status": "healthy",
                "processor": "ready",
                "enhanced_audio_enabled": self.config.enable_enhanced_audio,
                "ocr_enabled": self.config.enable_ocr,
                "extract_audio": self.config.extract_audio,
                "extract_keyframes": self.config.extract_keyframes,
                "generate_thumbnails": self.config.generate_thumbnails,
                "embedding_service": self.embedding_service is not None
            }

        except Exception as e:
            logger.error("Video service health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def process_file(self,
                         file_path: Union[str, Path],
                         save_output: bool = True,
                         output_format: str = "markdown",
                         progress_callback: callable = None) -> Dict[str, Any]:
        """Process a video file and optionally save the results.
        
        Args:
            file_path: Path to the video file
            save_output: Whether to save the output files
            output_format: Format to save the output (markdown, json, txt)
            
        Returns:
            Dictionary containing processing results and output file paths
            
        Raises:
            VideoServiceError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise VideoServiceError(f"Video file not found: {file_path}")
        
        try:
            # Process the video file
            logger.info("Starting video processing", file_path=str(file_path))
            if progress_callback:
                progress_callback(0.1, "Initializing video processing")
            result = await self.processor.process_video(file_path, progress_callback)
            
            # Generate embeddings if embedding service is available
            if self.embedding_service and result.audio_processing_result and result.audio_processing_result.transcript:
                try:
                    logger.info("Generating embeddings for transcript")
                    transcript_embedding = await self.embedding_service.embed_text(
                        result.audio_processing_result.transcript
                    )
                    
                    # Add embeddings to result
                    result.audio_processing_result.embeddings = {
                        "transcript": transcript_embedding
                    }
                    
                    # Generate embeddings for segments if available
                    if result.audio_processing_result.segments:
                        logger.info("Generating embeddings for segments")
                        segment_texts = [seg.text for seg in result.audio_processing_result.segments]
                        segment_embeddings = await self.embedding_service.embed_batch(segment_texts)
                        
                        # Add segment embeddings to result
                        result.audio_processing_result.embeddings["segments"] = segment_embeddings
                        
                except Exception as e:
                    logger.warning("Failed to generate embeddings", error=str(e))
            
            # Convert to requested format
            formatted_content = None
            if output_format == "markdown":
                try:
                    logger.info("Converting video processing result to markdown")
                    conversion_options = VideoConversionOptions(
                        include_metadata=True,
                        include_thumbnails=True,
                        include_keyframes=True,
                        include_transcript=True,
                        group_by_speaker=self.config.enable_speaker_diarization,
                        group_by_topic=self.config.enable_topic_segmentation
                    )
                    formatted_content = await self.converter.convert_to_markdown(result, conversion_options)
                except Exception as e:
                    logger.warning("Failed to convert to markdown", error=str(e))
            elif output_format == "json":
                try:
                    logger.info("Converting video processing result to JSON")
                    formatted_content = await self._convert_to_json(result)
                except Exception as e:
                    logger.warning("Failed to convert to JSON", error=str(e))
            
            # Save output files if requested
            output_files = {}
            if save_output and self.output_dir:
                output_files = await self._save_output_files(
                    file_path,
                    result,
                    formatted_content if output_format == "markdown" else None,
                    output_format
                )

            # Prepare response
            processing_time = time.time() - start_time
            response = {
                "success": True,
                "processing_time": processing_time,
                "content": formatted_content or "",  # Include formatted content
                "metadata": {
                    "duration": result.metadata.duration,
                    "resolution": f"{result.metadata.width}x{result.metadata.height}",
                    "fps": result.metadata.fps,
                    "format": result.metadata.format,
                    "has_audio": result.metadata.has_audio
                },
                "thumbnails": [str(path) for path in result.thumbnails],
                "keyframes": [str(path) for path in result.keyframes],
                "output_files": output_files
            }
            
            # Add audio processing results if available
            if result.audio_processing_result:
                response["audio"] = {
                    "transcript_length": len(result.audio_processing_result.transcript) if result.audio_processing_result.transcript else 0,
                    "language": result.audio_processing_result.metadata.get("language", "unknown"),
                    "segments_count": len(result.audio_processing_result.segments) if result.audio_processing_result.segments else 0,
                    "has_speaker_diarization": result.audio_processing_result.metadata.get("has_speaker_info", False),
                    "has_topic_segmentation": result.audio_processing_result.metadata.get("has_topic_info", False),
                    "has_embeddings": hasattr(result.audio_processing_result, "embeddings")
                }
            
            # Add OCR results if available
            if result.ocr_results:
                response["ocr"] = {
                    "images_processed": len(result.ocr_results),
                    "has_text": any(text.strip() for text in result.ocr_results.values())
                }
            
            logger.info("Video processing completed", 
                       file_path=str(file_path),
                       processing_time=processing_time,
                       thumbnails_count=len(result.thumbnails),
                       keyframes_count=len(result.keyframes))
            
            # Clean up temporary files
            self.processor.cleanup_temp_files(result.temp_files)
            
            return response
            
        except Exception as e:
            logger.error("Video processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            if isinstance(e, VideoServiceError):
                raise
            raise VideoServiceError(f"Video processing failed: {str(e)}")

    async def convert_result_to_markdown(self, json_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a JSON processing result to markdown format.

        Args:
            json_result: The JSON result from video processing

        Returns:
            Dictionary with markdown content
        """
        try:
            # Extract the video processing result from the JSON
            if not json_result.get("success", False):
                return json_result

            content = json_result.get("content", "")
            if not content:
                return json_result

            # If content is already markdown, return as-is
            if isinstance(content, str):
                return json_result

            # Convert to markdown using the existing converter
            logger.info("Converting video processing result to markdown")

            # Import here to avoid circular imports
            from morag_video.converters.video import VideoConversionOptions
            from morag_video.processor import VideoProcessingResult

            # If content is a VideoProcessingResult, convert it
            if isinstance(content, VideoProcessingResult):
                options = VideoConversionOptions()
                markdown_content = await self.converter.convert_to_markdown(content, options)
            elif isinstance(content, dict):
                # Try to reconstruct VideoProcessingResult from dict
                # For now, just return the original result
                logger.warning("Cannot convert dict content to markdown, returning original")
                return json_result
            else:
                logger.warning(f"Unknown content type: {type(content)}, returning original")
                return json_result

            # Return the same structure but with markdown content
            markdown_result = json_result.copy()
            markdown_result["content"] = markdown_content

            return markdown_result

        except Exception as e:
            logger.error("Failed to convert result to markdown", error=str(e))
            # Return original result if conversion fails
            return json_result

    async def _save_output_files(self,
                               file_path: Path, 
                               result: VideoProcessingResult,
                               markdown_content: Optional[str],
                               output_format: str) -> Dict[str, str]:
        """Save output files and return their paths."""
        if not self.output_dir:
            return {}
        
        output_files = {}
        base_name = file_path.stem
        
        # Create a subdirectory for this file
        file_output_dir = self.output_dir / base_name
        ensure_directory_exists(file_output_dir)
        
        # Save transcript as text if available
        if result.audio_processing_result and result.audio_processing_result.transcript:
            transcript_path = file_output_dir / f"{base_name}_transcript.txt"
            transcript_path.write_text(result.audio_processing_result.transcript)
            output_files["transcript"] = str(transcript_path)
        
        # Save markdown if available
        if markdown_content:
            markdown_path = file_output_dir / f"{base_name}.md"
            markdown_path.write_text(markdown_content)
            output_files["markdown"] = str(markdown_path)
        
        # Save segments as JSON if available
        if result.audio_processing_result and result.audio_processing_result.segments:
            import json
            segments_path = file_output_dir / f"{base_name}_segments.json"
            segments_data = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "speaker": segment.speaker if hasattr(segment, "speaker") else None,
                    "topic": segment.topic if hasattr(segment, "topic") else None
                }
                for segment in result.audio_processing_result.segments
            ]
            segments_path.write_text(json.dumps(segments_data, indent=2))
            output_files["segments"] = str(segments_path)
        
        # Save metadata as JSON
        import json
        metadata_path = file_output_dir / f"{base_name}_metadata.json"
        metadata_dict = {
            "duration": result.metadata.duration,
            "width": result.metadata.width,
            "height": result.metadata.height,
            "fps": result.metadata.fps,
            "codec": result.metadata.codec,
            "bitrate": result.metadata.bitrate,
            "file_size": result.metadata.file_size,
            "format": result.metadata.format,
            "has_audio": result.metadata.has_audio,
            "audio_codec": result.metadata.audio_codec,
            "creation_time": result.metadata.creation_time
        }
        metadata_path.write_text(json.dumps(metadata_dict, indent=2))
        output_files["metadata"] = str(metadata_path)
        
        # Save thumbnails
        if result.thumbnails:
            thumbnail_dir = file_output_dir / "thumbnails"
            ensure_directory_exists(thumbnail_dir)
            
            for i, thumbnail_path in enumerate(result.thumbnails):
                ext = thumbnail_path.suffix
                dest_path = thumbnail_dir / f"thumbnail_{i}{ext}"
                import shutil
                shutil.copy2(thumbnail_path, dest_path)
            
            output_files["thumbnails_dir"] = str(thumbnail_dir)
        
        # Save keyframes
        if result.keyframes:
            keyframe_dir = file_output_dir / "keyframes"
            ensure_directory_exists(keyframe_dir)
            
            for i, keyframe_path in enumerate(result.keyframes):
                ext = keyframe_path.suffix
                dest_path = keyframe_dir / f"keyframe_{i}{ext}"
                import shutil
                shutil.copy2(keyframe_path, dest_path)
            
            output_files["keyframes_dir"] = str(keyframe_dir)
        
        # Save OCR results if available
        if result.ocr_results:
            ocr_path = file_output_dir / f"{base_name}_ocr.json"
            import json
            ocr_path.write_text(json.dumps(result.ocr_results, indent=2))
            output_files["ocr"] = str(ocr_path)
        
        logger.info("Output files saved", 
                   output_dir=str(file_output_dir),
                   files=list(output_files.keys()))
        
        return output_files

    async def _convert_to_json(self, result: VideoProcessingResult) -> Dict[str, Any]:
        """Convert video processing result to structured JSON.

        Args:
            result: Video processing result

        Returns:
            Dictionary with structured JSON data
        """
        try:
            # Extract title from file path if available
            title = ""
            filename = ""
            if hasattr(result, 'file_path') and result.file_path:
                filename = Path(result.file_path).name
                title = Path(result.file_path).stem

            # Build topics from audio processing result
            topics = []
            if result.audio_processing_result and result.audio_processing_result.segments:
                # Check if topic segmentation was performed and topics are available in metadata
                topics_metadata = result.audio_processing_result.metadata.get("topics", [])

                if topics_metadata:
                    # Group segments by topic using the topic information from metadata
                    for topic_info in topics_metadata:
                        topic_data = {
                            "timestamp": int(topic_info.get("start_time", 0)),
                            "title": topic_info.get("title", f"Topic {topic_info.get('id', '')}"),
                            "summary": topic_info.get("summary", ""),
                            "sentences": []
                        }

                        # Find segments that belong to this topic
                        topic_id = topic_info.get("id", "").replace("TOPIC_", "") if isinstance(topic_info.get("id"), str) else topic_info.get("id")
                        if isinstance(topic_id, str) and topic_id.isdigit():
                            topic_id = int(topic_id)

                        for segment in result.audio_processing_result.segments:
                            if hasattr(segment, 'topic_id') and segment.topic_id == topic_id:
                                sentence = {
                                    "timestamp": int(segment.start),
                                    "speaker": getattr(segment, 'speaker', 1) if hasattr(segment, 'speaker') else 1,
                                    "text": segment.text
                                }
                                topic_data["sentences"].append(sentence)

                        topics.append(topic_data)
                else:
                    # Single topic with all segments (no topic segmentation)
                    topic_data = {
                        "timestamp": int(result.audio_processing_result.segments[0].start) if result.audio_processing_result.segments else 0,
                        "title": "Full Transcript",
                        "sentences": []
                    }

                    for segment in result.audio_processing_result.segments:
                        sentence = {
                            "timestamp": int(segment.start),
                            "speaker": getattr(segment, 'speaker', 1) if hasattr(segment, 'speaker') else 1,
                            "text": segment.text
                        }
                        topic_data["sentences"].append(sentence)

                    topics.append(topic_data)

            # Build metadata
            metadata = {
                "duration": result.metadata.duration,
                "resolution": f"{result.metadata.width}x{result.metadata.height}",
                "fps": result.metadata.fps,
                "format": result.metadata.format,
                "has_audio": result.metadata.has_audio,
                "thumbnails_count": len(result.thumbnails),
                "keyframes_count": len(result.keyframes)
            }

            # Add audio processing metadata if available
            if result.audio_processing_result:
                metadata.update({
                    "transcript_length": len(result.audio_processing_result.transcript) if result.audio_processing_result.transcript else 0,
                    "segments_count": len(result.audio_processing_result.segments) if result.audio_processing_result.segments else 0,
                    "has_speaker_diarization": result.audio_processing_result.metadata.get("has_speaker_info", False),
                    "has_topic_segmentation": result.audio_processing_result.metadata.get("has_topic_info", False)
                })

            return {
                "title": title,
                "filename": filename,
                "metadata": metadata,
                "topics": topics
            }

        except Exception as e:
            logger.error("Failed to convert video result to JSON", error=str(e))
            return {
                "title": "",
                "filename": "",
                "metadata": {},
                "topics": [],
                "error": str(e)
            }