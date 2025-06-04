"""Video processing with FFmpeg for audio extraction and thumbnail generation."""

import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import asyncio
import structlog
import time

# Import ffmpeg-python components
from ffmpeg._probe import probe as ffmpeg_probe
from ffmpeg._run import input as ffmpeg_input, output as ffmpeg_output, run as ffmpeg_run, Error as FFmpegError
from ffmpeg._filters import filter as ffmpeg_filter
from PIL import Image
import cv2

from morag.core.config import settings
from morag.core.exceptions import ProcessingError, ExternalServiceError

logger = structlog.get_logger()

@dataclass
class VideoConfig:
    """Configuration for video processing."""
    extract_audio: bool = True
    generate_thumbnails: bool = True
    thumbnail_count: int = 5
    extract_keyframes: bool = False
    max_keyframes: int = 10
    audio_format: str = "mp3"
    thumbnail_size: Tuple[int, int] = (320, 240)
    thumbnail_format: str = "jpg"
    keyframe_threshold: float = 0.3  # Scene change threshold
    # Enhanced audio processing options
    enable_enhanced_audio: bool = True
    enable_speaker_diarization: bool = True
    enable_topic_segmentation: bool = True
    audio_model_size: str = "base"

@dataclass
class VideoMetadata:
    """Video metadata information."""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: Optional[int]
    file_size: int
    format: str
    has_audio: bool
    audio_codec: Optional[str]
    creation_time: Optional[str]

@dataclass
class VideoProcessingResult:
    """Result of video processing operation."""
    audio_path: Optional[Path]
    thumbnails: List[Path]
    keyframes: List[Path]
    metadata: VideoMetadata
    processing_time: float
    temp_files: List[Path] = field(default_factory=list)
    # Enhanced audio processing results
    audio_processing_result: Optional[Any] = None  # AudioProcessingResult from audio processor

class VideoProcessor:
    """Video processing service using FFmpeg."""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_video"
        self.temp_dir.mkdir(exist_ok=True)
        # Import audio processor here to avoid circular imports
        self._audio_processor = None

    def _get_audio_processor(self):
        """Get audio processor instance, importing only when needed."""
        if self._audio_processor is None:
            from morag.processors.audio import AudioProcessor
            self._audio_processor = AudioProcessor()
        return self._audio_processor

    async def process_video(
        self,
        file_path: Path,
        config: VideoConfig
    ) -> VideoProcessingResult:
        """Process video file with audio extraction and thumbnail generation."""
        start_time = time.time()
        
        try:
            logger.info("Starting video processing", file_path=str(file_path))
            
            # Validate input file
            if not file_path.exists():
                raise ProcessingError(f"Video file not found: {file_path}")
            
            # Extract metadata
            metadata = await self._extract_metadata(file_path)
            
            # Initialize result
            result = VideoProcessingResult(
                audio_path=None,
                thumbnails=[],
                keyframes=[],
                metadata=metadata,
                processing_time=0.0,
                temp_files=[]
            )
            
            # Extract audio if requested
            if config.extract_audio and metadata.has_audio:
                audio_path = await self._extract_audio(file_path, config.audio_format)
                result.audio_path = audio_path
                result.temp_files.append(audio_path)

                # Process extracted audio with enhanced features if enabled
                if config.enable_enhanced_audio and audio_path:
                    try:
                        logger.info("Starting enhanced audio processing for video",
                                   audio_path=str(audio_path),
                                   enable_diarization=config.enable_speaker_diarization,
                                   enable_topic_segmentation=config.enable_topic_segmentation)

                        audio_processor = self._get_audio_processor()

                        # Import AudioConfig here to avoid circular imports
                        from morag.processors.audio import AudioConfig

                        audio_config = AudioConfig(
                            model_size=config.audio_model_size,
                            language=None,  # Auto-detect
                            device="auto"  # Auto-detect best available device (GPU/CPU)
                        )

                        # Process audio with enhanced features
                        audio_result = await audio_processor.process_audio_file(
                            audio_path,
                            config=audio_config,
                            enable_diarization=config.enable_speaker_diarization,
                            enable_topic_segmentation=config.enable_topic_segmentation
                        )

                        result.audio_processing_result = audio_result

                        logger.info("Enhanced audio processing completed for video",
                                   transcription_length=len(audio_result.text),
                                   language=audio_result.language,
                                   speakers_detected=audio_result.speaker_diarization.total_speakers if audio_result.speaker_diarization else 0,
                                   topics_detected=audio_result.topic_segmentation.total_topics if audio_result.topic_segmentation else 0)

                    except Exception as e:
                        logger.warning("Enhanced audio processing failed for video",
                                     audio_path=str(audio_path),
                                     error=str(e))
                        # Continue without enhanced audio processing
                        result.audio_processing_result = None
            
            # Generate thumbnails if requested
            if config.generate_thumbnails:
                thumbnails = await self._generate_thumbnails(
                    file_path, 
                    config.thumbnail_count,
                    config.thumbnail_size,
                    config.thumbnail_format
                )
                result.thumbnails = thumbnails
                result.temp_files.extend(thumbnails)
            
            # Extract keyframes if requested
            if config.extract_keyframes:
                keyframes = await self._extract_keyframes(
                    file_path,
                    config.max_keyframes,
                    config.keyframe_threshold,
                    config.thumbnail_size,
                    config.thumbnail_format
                )
                result.keyframes = keyframes
                result.temp_files.extend(keyframes)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info("Video processing completed",
                       file_path=str(file_path),
                       processing_time=processing_time,
                       audio_extracted=result.audio_path is not None,
                       thumbnails_count=len(result.thumbnails),
                       keyframes_count=len(result.keyframes))
            
            return result
            
        except Exception as e:
            logger.error("Video processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            raise ProcessingError(f"Video processing failed: {str(e)}")
    
    async def _extract_metadata(self, file_path: Path) -> VideoMetadata:
        """Extract comprehensive video metadata."""
        try:
            # Use ffmpeg_probe to get metadata
            probe = await asyncio.to_thread(ffmpeg_probe, str(file_path))
            
            # Find video stream
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ProcessingError("No video stream found in file")
            
            # Find audio stream
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            # Extract format information
            format_info = probe.get('format', {})
            
            metadata = VideoMetadata(
                duration=float(format_info.get('duration', 0)),
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                fps=eval(video_stream.get('r_frame_rate', '0/1')),  # Convert fraction to float
                codec=video_stream.get('codec_name', 'unknown'),
                bitrate=int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
                file_size=int(format_info.get('size', 0)),
                format=format_info.get('format_name', 'unknown'),
                has_audio=audio_stream is not None,
                audio_codec=audio_stream.get('codec_name') if audio_stream else None,
                creation_time=format_info.get('tags', {}).get('creation_time')
            )
            
            logger.debug("Video metadata extracted",
                        duration=metadata.duration,
                        resolution=f"{metadata.width}x{metadata.height}",
                        fps=metadata.fps,
                        has_audio=metadata.has_audio)
            
            return metadata
            
        except Exception as e:
            logger.error("Metadata extraction failed", 
                        file_path=str(file_path),
                        error=str(e))
            raise ExternalServiceError(f"Metadata extraction failed: {str(e)}", "ffmpeg")
    
    async def _extract_audio(self, file_path: Path, audio_format: str = "mp3") -> Path:
        """Extract audio track from video file with minimal processing overhead."""
        try:
            # Create output path
            audio_path = self.temp_dir / f"audio_{int(time.time())}_{file_path.stem}.{audio_format}"

            logger.debug("Extracting audio from video",
                        video_path=str(file_path),
                        audio_path=str(audio_path),
                        format=audio_format)

            # Get video metadata to determine best extraction strategy
            metadata = await self._extract_metadata(file_path)

            # Determine codec strategy for minimal processing time
            if audio_format == "mp3":
                # Try to copy stream if source is already MP3, otherwise use fast encoding
                if metadata.audio_codec and "mp3" in metadata.audio_codec.lower():
                    codec = "copy"  # Stream copy - fastest, no re-encoding
                    logger.debug("Using stream copy for MP3 extraction (minimal overhead)")
                else:
                    codec = "libmp3lame"  # Fast MP3 encoding
                    logger.debug("Using MP3 encoding for audio extraction")
            elif audio_format == "wav":
                codec = "pcm_s16le"  # Uncompressed WAV
                logger.warning("Using uncompressed WAV format - will result in large files")
            elif audio_format == "aac":
                # Try to copy stream if source is already AAC
                if metadata.audio_codec and "aac" in metadata.audio_codec.lower():
                    codec = "copy"  # Stream copy - fastest
                    logger.debug("Using stream copy for AAC extraction (minimal overhead)")
                else:
                    codec = "aac"  # Fast AAC encoding
                    logger.debug("Using AAC encoding for audio extraction")
            else:
                codec = "libmp3lame"  # Default to MP3 for unknown formats
                logger.debug(f"Unknown format {audio_format}, defaulting to MP3 encoding")

            # Extract audio using ffmpeg with optimized settings and error capture
            def run_ffmpeg_extraction():
                try:
                    # Build output parameters, excluding None values
                    output_params = {"acodec": codec}

                    # Add quality settings for non-copy modes
                    if codec != "copy":
                        if audio_format == "mp3":
                            output_params["audio_bitrate"] = "128k"
                        elif audio_format == "aac":
                            output_params["audio_bitrate"] = "128k"
                        # For WAV (pcm_s16le), don't set bitrate as it's uncompressed

                    ffmpeg_run(
                        ffmpeg_output(
                            ffmpeg_input(str(file_path)),
                            str(audio_path),
                            **output_params
                        ),
                        overwrite_output=True,
                        quiet=False,  # Enable stderr capture
                        capture_stderr=True
                    )
                except FFmpegError as e:
                    # Capture stderr for detailed error information
                    stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
                    raise ProcessingError(f"FFmpeg audio extraction error: {stderr_output}")
                except Exception as e:
                    raise ProcessingError(f"Audio extraction error: {str(e)}")

            await asyncio.to_thread(run_ffmpeg_extraction)

            if not audio_path.exists():
                raise ProcessingError("Audio extraction failed - output file not created")
            
            logger.debug("Audio extraction completed",
                        audio_path=str(audio_path),
                        file_size=audio_path.stat().st_size)
            
            return audio_path
            
        except Exception as e:
            logger.error("Audio extraction failed",
                        file_path=str(file_path),
                        error=str(e))
            raise ExternalServiceError(f"Audio extraction failed: {str(e)}", "ffmpeg")
    
    async def _generate_thumbnails(
        self,
        file_path: Path,
        count: int,
        size: Tuple[int, int],
        format: str = "jpg"
    ) -> List[Path]:
        """Generate thumbnails at evenly spaced intervals."""
        try:
            thumbnails = []

            logger.debug("Starting thumbnail generation",
                        file_path=str(file_path),
                        count=count,
                        size=size,
                        format=format)

            # Get video duration for timestamp calculation
            try:
                probe = await asyncio.to_thread(ffmpeg_probe, str(file_path))
                duration = float(probe['format']['duration'])
                logger.debug("Video duration extracted", duration=duration)
            except Exception as e:
                logger.error("Failed to probe video for duration",
                           file_path=str(file_path),
                           error=str(e))
                raise ProcessingError(f"Failed to probe video: {str(e)}")

            # Validate duration
            if duration <= 0:
                raise ProcessingError(f"Invalid video duration: {duration}")

            # Calculate timestamps for thumbnails
            if count == 1:
                timestamps = [duration / 2]  # Middle of video
            else:
                # Ensure we don't go beyond video duration
                timestamps = [min(i * duration / (count - 1), duration - 1) for i in range(count)]

            logger.debug("Calculated timestamps", timestamps=timestamps)

            for i, timestamp in enumerate(timestamps):
                thumbnail_path = self.temp_dir / f"thumb_{int(time.time())}_{i}.{format}"

                logger.debug("Generating thumbnail",
                           index=i,
                           timestamp=timestamp,
                           output_path=str(thumbnail_path))

                try:
                    # Generate thumbnail using ffmpeg with better error handling
                    def generate_thumbnail():
                        try:
                            # Build ffmpeg command using internal functions
                            input_stream = ffmpeg_input(str(file_path), ss=timestamp)
                            scaled_stream = ffmpeg_filter(input_stream, 'scale', size[0], size[1])
                            output_stream = ffmpeg_output(scaled_stream, str(thumbnail_path), vframes=1)

                            # Run with error capture
                            ffmpeg_run(output_stream, overwrite_output=True, quiet=False)

                        except FFmpegError as e:
                            # Capture stderr for detailed error information
                            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
                            raise ProcessingError(f"FFmpeg error: {stderr_output}")
                        except Exception as e:
                            raise ProcessingError(f"Thumbnail generation error: {str(e)}")

                    await asyncio.to_thread(generate_thumbnail)

                    # Verify thumbnail was created
                    if thumbnail_path.exists() and thumbnail_path.stat().st_size > 0:
                        thumbnails.append(thumbnail_path)
                        logger.debug("Thumbnail generated successfully",
                                   thumbnail_path=str(thumbnail_path),
                                   timestamp=timestamp,
                                   file_size=thumbnail_path.stat().st_size)
                    else:
                        logger.warning("Thumbnail file not created or empty",
                                     thumbnail_path=str(thumbnail_path),
                                     timestamp=timestamp)

                        # Try alternative approaches
                        # 1. Try with different timestamp
                        if timestamp > 1:
                            alt_timestamp = max(0, timestamp - 1)
                            logger.debug("Retrying with alternative timestamp",
                                       original_timestamp=timestamp,
                                       alt_timestamp=alt_timestamp)

                            alt_thumbnail_path = self.temp_dir / f"thumb_{int(time.time())}_{i}_alt.{format}"

                            def generate_alt_thumbnail():
                                try:
                                    input_stream = ffmpeg_input(str(file_path), ss=alt_timestamp)
                                    scaled_stream = ffmpeg_filter(input_stream, 'scale', size[0], size[1])
                                    output_stream = ffmpeg_output(scaled_stream, str(alt_thumbnail_path), vframes=1)
                                    ffmpeg_run(output_stream, overwrite_output=True, quiet=False)
                                except Exception as e:
                                    logger.warning("Alternative timestamp thumbnail generation failed", error=str(e))

                            await asyncio.to_thread(generate_alt_thumbnail)

                            if alt_thumbnail_path.exists() and alt_thumbnail_path.stat().st_size > 0:
                                thumbnails.append(alt_thumbnail_path)
                                logger.debug("Alternative timestamp thumbnail generated successfully",
                                           thumbnail_path=str(alt_thumbnail_path))
                                continue

                        # 2. Try without scaling (original size)
                        logger.debug("Retrying without scaling", timestamp=timestamp)
                        no_scale_path = self.temp_dir / f"thumb_{int(time.time())}_{i}_noscale.{format}"

                        def generate_no_scale_thumbnail():
                            try:
                                input_stream = ffmpeg_input(str(file_path), ss=timestamp)
                                output_stream = ffmpeg_output(input_stream, str(no_scale_path), vframes=1)
                                ffmpeg_run(output_stream, overwrite_output=True, quiet=False)
                            except Exception as e:
                                logger.warning("No-scale thumbnail generation failed", error=str(e))

                        await asyncio.to_thread(generate_no_scale_thumbnail)

                        if no_scale_path.exists() and no_scale_path.stat().st_size > 0:
                            thumbnails.append(no_scale_path)
                            logger.debug("No-scale thumbnail generated successfully",
                                       thumbnail_path=str(no_scale_path))

                except Exception as e:
                    logger.warning("Failed to generate thumbnail",
                                 index=i,
                                 timestamp=timestamp,
                                 error=str(e))
                    # Continue with next thumbnail instead of failing completely
                    continue

            logger.info("Thumbnail generation completed",
                       count=len(thumbnails),
                       requested=count,
                       success_rate=f"{len(thumbnails)}/{count}")

            return thumbnails

        except Exception as e:
            logger.error("Thumbnail generation failed",
                        file_path=str(file_path),
                        error=str(e))
            raise ExternalServiceError(f"Thumbnail generation failed: {str(e)}", "ffmpeg")

    async def _extract_keyframes(
        self,
        file_path: Path,
        max_frames: int,
        threshold: float,
        size: Tuple[int, int],
        format: str = "jpg"
    ) -> List[Path]:
        """Extract keyframes based on scene changes."""
        try:
            keyframes = []

            logger.debug("Extracting keyframes",
                        file_path=str(file_path),
                        max_frames=max_frames,
                        threshold=threshold)

            # Use OpenCV for scene change detection
            cap = cv2.VideoCapture(str(file_path))

            if not cap.isOpened():
                raise ProcessingError("Could not open video file with OpenCV")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            prev_frame = None
            frame_idx = 0
            keyframe_timestamps = []

            # Sample frames at regular intervals to detect scene changes
            sample_interval = max(1, frame_count // (max_frames * 10))  # Sample more than needed

            while len(keyframe_timestamps) < max_frames and frame_idx < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    # Calculate histogram difference
                    hist_diff = cv2.compareHist(
                        cv2.calcHist([prev_frame], [0], None, [256], [0, 256]),
                        cv2.calcHist([gray], [0], None, [256], [0, 256]),
                        cv2.HISTCMP_CORREL
                    )

                    # If difference is significant, it's a scene change
                    if hist_diff < (1 - threshold):
                        timestamp = frame_idx / fps
                        keyframe_timestamps.append(timestamp)
                        logger.debug("Keyframe detected",
                                   frame_idx=frame_idx,
                                   timestamp=timestamp,
                                   hist_diff=hist_diff)

                prev_frame = gray
                frame_idx += sample_interval

            cap.release()

            # Generate keyframe images using ffmpeg
            for i, timestamp in enumerate(keyframe_timestamps):
                keyframe_path = self.temp_dir / f"keyframe_{int(time.time())}_{i}.{format}"

                try:
                    def generate_keyframe():
                        try:
                            # Build ffmpeg command using internal functions
                            input_stream = ffmpeg_input(str(file_path), ss=timestamp)
                            scaled_stream = ffmpeg_filter(input_stream, 'scale', size[0], size[1])
                            output_stream = ffmpeg_output(scaled_stream, str(keyframe_path), vframes=1)
                            ffmpeg_run(output_stream, overwrite_output=True, quiet=False)

                        except FFmpegError as e:
                            # Capture stderr for detailed error information
                            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
                            raise ProcessingError(f"FFmpeg keyframe error: {stderr_output}")
                        except Exception as e:
                            raise ProcessingError(f"Keyframe generation error: {str(e)}")

                    await asyncio.to_thread(generate_keyframe)

                    # Verify keyframe was created
                    if keyframe_path.exists() and keyframe_path.stat().st_size > 0:
                        keyframes.append(keyframe_path)
                        logger.debug("Keyframe generated successfully",
                                   keyframe_path=str(keyframe_path),
                                   timestamp=timestamp)
                    else:
                        logger.warning("Keyframe file not created or empty",
                                     keyframe_path=str(keyframe_path),
                                     timestamp=timestamp)

                except Exception as e:
                    logger.warning("Failed to generate keyframe",
                                 index=i,
                                 timestamp=timestamp,
                                 error=str(e))
                    # Continue with next keyframe instead of failing completely
                    continue

            logger.info("Keyframes extracted",
                       count=len(keyframes),
                       max_frames=max_frames)

            return keyframes

        except Exception as e:
            logger.error("Keyframe extraction failed",
                        file_path=str(file_path),
                        error=str(e))
            raise ExternalServiceError(f"Keyframe extraction failed: {str(e)}", "ffmpeg")

    def cleanup_temp_files(self, temp_files: List[Path]):
        """Clean up temporary files."""
        for file_path in temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug("Temporary file cleaned up", file_path=str(file_path))
            except Exception as e:
                logger.warning("Failed to clean up temporary file",
                             file_path=str(file_path),
                             error=str(e))

# Global instance
video_processor = VideoProcessor()
