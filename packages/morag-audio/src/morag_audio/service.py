"""Audio service module for MoRAG."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import structlog

from morag_core.errors import ServiceError
from morag_core.utils import ensure_directory_exists
from morag_embedding import EmbeddingService

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
                 embedding_service: Optional[EmbeddingService] = None,
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
    
    async def process_file(self, 
                         file_path: Union[str, Path], 
                         save_output: bool = True,
                         output_format: str = "markdown") -> Dict[str, Any]:
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
            result = await self.processor.process(file_path)
            
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
            
            # Convert to markdown if requested
            markdown_content = None
            if result.success and output_format == "markdown":
                conversion_options = AudioConversionOptions(
                    include_timestamps=True,
                    include_speakers=self.config.enable_diarization,
                    include_topics=self.config.enable_topic_segmentation
                )
                markdown_result = await self.converter.convert_to_markdown(result, conversion_options)
                markdown_content = markdown_result.content
            
            # Save output files if requested
            output_files = {}
            if save_output and result.success and self.output_dir:
                output_files = await self._save_output_files(file_path, result, markdown_content, output_format)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "success": result.success,
                "processing_time": processing_time,
                "metadata": result.metadata,
                "output_files": output_files
            }
            
            if not save_output or not self.output_dir:
                if output_format == "markdown" and markdown_content:
                    response["content"] = markdown_content
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
        
        # Create a subdirectory for this file
        file_output_dir = self.output_dir / base_name
        ensure_directory_exists(file_output_dir)
        
        # Save transcript as text
        if result.transcript:
            transcript_path = file_output_dir / f"{base_name}_transcript.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(result.transcript)
            output_files["transcript"] = str(transcript_path)
        
        # Save markdown if available
        if markdown_content:
            markdown_path = file_output_dir / f"{base_name}.md"
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            output_files["markdown"] = str(markdown_path)
        
        # Save segments as JSON
        if result.segments:
            import json
            
            # Convert segments to serializable format
            segments_data = []
            for segment in result.segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": segment.confidence
                }
                
                if segment.speaker:
                    segment_dict["speaker"] = segment.speaker
                    
                if segment.topic_id is not None:
                    segment_dict["topic_id"] = segment.topic_id
                    
                if hasattr(segment, "embedding"):
                    segment_dict["embedding"] = segment.embedding
                    
                segments_data.append(segment_dict)
            
            segments_path = file_output_dir / f"{base_name}_segments.json"
            with open(segments_path, "w", encoding="utf-8") as f:
                json.dump(segments_data, f, indent=2)
            output_files["segments"] = str(segments_path)
        
        # Save metadata
        if result.metadata:
            import json
            metadata_path = file_output_dir / f"{base_name}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                # Filter out non-serializable objects
                serializable_metadata = {}
                for key, value in result.metadata.items():
                    try:
                        json.dumps({key: value})
                        serializable_metadata[key] = value
                    except (TypeError, OverflowError):
                        serializable_metadata[key] = str(value)
                        
                json.dump(serializable_metadata, f, indent=2)
            output_files["metadata"] = str(metadata_path)
        
        logger.info("Saved output files", 
                   output_dir=str(file_output_dir),
                   files=list(output_files.keys()))
        
        return output_files