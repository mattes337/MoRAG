"""Video processor module for MoRAG."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import asyncio
import structlog
import tempfile

# Import ffmpeg-python components
from ffmpeg._probe import probe as ffmpeg_probe
from ffmpeg._run import input as ffmpeg_input, output as ffmpeg_output, run as ffmpeg_run, Error as FFmpegError
from ffmpeg._filters import filter as ffmpeg_filter
from PIL import Image
import cv2
import numpy as np

from morag_core.errors import ProcessingError, ExternalServiceError

logger = structlog.get_logger(__name__)


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
    enable_speaker_diarization: bool = False
    enable_topic_segmentation: bool = False
    audio_model_size: str = "base"
    # OCR options
    enable_ocr: bool = False
    ocr_engine: str = "tesseract"  # tesseract or easyocr


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
    # OCR results
    ocr_results: Optional[Dict[str, Any]] = None


class VideoProcessingError(ProcessingError):
    """Error raised during video processing."""
    pass


class VideoProcessor:
    """Video processing service using FFmpeg."""

    def __init__(self, config: Optional[VideoConfig] = None):
        """Initialize video processor.
        
        Args:
            config: Configuration for video processing
        """
        self.config = config or VideoConfig()
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_video"
        self.temp_dir.mkdir(exist_ok=True)
        # Import audio processor here to avoid circular imports
        self._audio_processor = None
        self._ocr_engine = None

    def _get_audio_processor(self):
        """Get audio processor instance, importing only when needed."""
        if self._audio_processor is None:
            try:
                from morag_audio import AudioProcessor, AudioConfig
                
                # Create audio config based on video config
                audio_config = AudioConfig(
                    model_size=self.config.audio_model_size,
                    enable_diarization=self.config.enable_speaker_diarization,
                    enable_topic_segmentation=self.config.enable_topic_segmentation,
                    device="auto"  # Auto-detect best available device
                )
                
                self._audio_processor = AudioProcessor(audio_config)
                logger.info("Initialized AudioProcessor for video processing",
                           model_size=self.config.audio_model_size,
                           enable_diarization=self.config.enable_speaker_diarization,
                           enable_topic_segmentation=self.config.enable_topic_segmentation)
            except ImportError as e:
                logger.warning("Could not import AudioProcessor", error=str(e))
                raise VideoProcessingError(f"AudioProcessor not available: {str(e)}")
        return self._audio_processor

    def _get_ocr_engine(self):
        """Get OCR engine instance, importing only when needed."""
        if self._ocr_engine is None and self.config.enable_ocr:
            try:
                if self.config.ocr_engine == "tesseract":
                    import pytesseract
                    self._ocr_engine = pytesseract
                    logger.info("Initialized Tesseract OCR engine")
                elif self.config.ocr_engine == "easyocr":
                    import easyocr
                    self._ocr_engine = easyocr.Reader(['en'])
                    logger.info("Initialized EasyOCR engine")
                else:
                    logger.warning(f"Unknown OCR engine: {self.config.ocr_engine}")
                    raise VideoProcessingError(f"Unknown OCR engine: {self.config.ocr_engine}")
            except ImportError as e:
                logger.warning(f"Could not import {self.config.ocr_engine}", error=str(e))
                raise VideoProcessingError(f"OCR engine not available: {str(e)}")
        return self._ocr_engine

    async def process_video(self, file_path: Union[str, Path]) -> VideoProcessingResult:
        """Process video file with audio extraction and thumbnail generation.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            VideoProcessingResult containing processing results
            
        Raises:
            VideoProcessingError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            logger.info("Starting video processing", file_path=str(file_path))
            
            # Validate input file
            if not file_path.exists():
                raise VideoProcessingError(f"Video file not found: {file_path}")
            
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
            if self.config.extract_audio and metadata.has_audio:
                audio_path = await self._extract_audio(file_path, self.config.audio_format)
                result.audio_path = audio_path
                result.temp_files.append(audio_path)

                # Process extracted audio with enhanced features if enabled
                if self.config.enable_enhanced_audio and audio_path:
                    try:
                        logger.info("Starting enhanced audio processing for video",
                                   audio_path=str(audio_path),
                                   enable_diarization=self.config.enable_speaker_diarization,
                                   enable_topic_segmentation=self.config.enable_topic_segmentation)

                        audio_processor = self._get_audio_processor()
                        audio_result = await audio_processor.process(audio_path)
                        result.audio_processing_result = audio_result

                        logger.info("Enhanced audio processing completed for video",
                                   transcription_length=len(audio_result.transcript) if audio_result.transcript else 0,
                                   language=audio_result.language,
                                   speakers_detected=len(audio_result.speaker_segments) if audio_result.speaker_segments else 0,
                                   topics_detected=len(audio_result.topic_segments) if audio_result.topic_segments else 0)

                    except Exception as e:
                        logger.warning("Enhanced audio processing failed for video",
                                     audio_path=str(audio_path),
                                     error=str(e))
                        # Continue without enhanced audio processing
                        result.audio_processing_result = None
            
            # Generate thumbnails if requested
            if self.config.generate_thumbnails:
                thumbnails = await self._generate_thumbnails(
                    file_path, 
                    self.config.thumbnail_count,
                    self.config.thumbnail_size,
                    self.config.thumbnail_format
                )
                result.thumbnails = thumbnails
                result.temp_files.extend(thumbnails)
            
            # Extract keyframes if requested
            if self.config.extract_keyframes:
                keyframes = await self._extract_keyframes(
                    file_path,
                    self.config.max_keyframes,
                    self.config.keyframe_threshold,
                    self.config.thumbnail_size,
                    self.config.thumbnail_format
                )
                result.keyframes = keyframes
                result.temp_files.extend(keyframes)
                
                # Perform OCR on keyframes if requested
                if self.config.enable_ocr and keyframes:
                    try:
                        ocr_results = await self._perform_ocr(keyframes)
                        result.ocr_results = ocr_results
                    except Exception as e:
                        logger.warning("OCR processing failed",
                                     error=str(e))
                        result.ocr_results = None
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info("Video processing completed",
                       file_path=str(file_path),
                       processing_time=processing_time,
                       audio_extracted=result.audio_path is not None,
                       thumbnails_count=len(result.thumbnails),
                       keyframes_count=len(result.keyframes),
                       ocr_performed=result.ocr_results is not None)
            
            return result
            
        except Exception as e:
            logger.error("Video processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            if isinstance(e, VideoProcessingError):
                raise
            raise VideoProcessingError(f"Video processing failed: {str(e)}")
    
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
                raise VideoProcessingError("No video stream found in file")
            
            # Find audio stream
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            # Extract format information
            format_info = probe.get('format', {})
            
            # Calculate fps from frame rate fraction
            fps_fraction = video_stream.get('r_frame_rate', '0/1')
            try:
                if '/' in fps_fraction:
                    num, den = map(int, fps_fraction.split('/'))
                    fps = num / den if den != 0 else 0
                else:
                    fps = float(fps_fraction)
            except (ValueError, ZeroDivisionError):
                fps = 0
            
            metadata = VideoMetadata(
                duration=float(format_info.get('duration', 0)),
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                fps=fps,
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
                    raise VideoProcessingError(f"FFmpeg audio extraction error: {stderr_output}")
                except Exception as e:
                    raise VideoProcessingError(f"Audio extraction error: {str(e)}")

            await asyncio.to_thread(run_ffmpeg_extraction)

            if not audio_path.exists():
                raise VideoProcessingError("Audio extraction failed - output file not created")
            
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
                raise VideoProcessingError(f"Failed to probe video: {str(e)}")

            # Validate duration
            if duration <= 0:
                raise VideoProcessingError(f"Invalid video duration: {duration}")

            # Calculate timestamps for thumbnails
            if count == 1:
                timestamps = [duration / 2]  # Middle of video
            else:
                # Ensure we don't go beyond video duration
                timestamps = [min(i * duration / (count - 1), duration - 1) for i in range(count)]

            logger.debug("Calculated timestamps", timestamps=timestamps)

            for i, timestamp in enumerate(timestamps):
                # Use unique timestamp-based filename to avoid pattern conflicts
                unique_id = int(time.time() * 1000000)  # Microsecond precision
                thumbnail_path = self.temp_dir / f"thumb_{unique_id}_{i}.{format}"

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
                            # Use -update option to write single image and avoid pattern warnings
                            output_stream = ffmpeg_output(scaled_stream, str(thumbnail_path),
                                                        vframes=1, update=1)
                            ffmpeg_run(output_stream, overwrite_output=True, quiet=True, capture_stderr=True)

                        except FFmpegError as e:
                            # Capture stderr for detailed error information
                            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
                            # Filter out pattern warnings which are not actual errors
                            if "image sequence pattern" not in stderr_output.lower():
                                raise VideoProcessingError(f"FFmpeg error: {stderr_output}")
                            else:
                                logger.debug("FFmpeg pattern warning (non-critical)", stderr=stderr_output)
                        except Exception as e:
                            raise VideoProcessingError(f"Thumbnail generation error: {str(e)}")

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

            # Try to use PySceneDetect if available for better scene detection
            try:
                import scenedetect
                from scenedetect import VideoManager, SceneManager
                from scenedetect.detectors import ContentDetector
                
                logger.info("Using PySceneDetect for keyframe extraction")
                
                # Create video manager and scene manager
                video_manager = VideoManager([str(file_path)])
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=threshold))
                
                # Improve performance with downscale and limited analysis
                video_manager.set_downscale_factor()
                
                # Start video manager
                video_manager.start()
                
                # Detect scenes
                scene_manager.detect_scenes(frame_source=video_manager)
                
                # Get scene list and frame timestamps
                scene_list = scene_manager.get_scene_list()
                
                # Get frame rate for timestamp calculation
                fps = video_manager.get_framerate()
                
                # Extract keyframes at scene changes
                for i, scene in enumerate(scene_list[:max_frames]):
                    # Get timestamp of first frame in scene
                    frame_num = scene[0]
                    timestamp = frame_num / fps
                    
                    # Generate keyframe using ffmpeg
                    unique_id = int(time.time() * 1000000)  # Microsecond precision
                    keyframe_path = self.temp_dir / f"keyframe_{unique_id}_{i}.{format}"
                    
                    def generate_keyframe():
                        try:
                            input_stream = ffmpeg_input(str(file_path), ss=timestamp)
                            scaled_stream = ffmpeg_filter(input_stream, 'scale', size[0], size[1])
                            output_stream = ffmpeg_output(scaled_stream, str(keyframe_path), vframes=1)
                            ffmpeg_run(output_stream, overwrite_output=True, quiet=True)
                        except Exception as e:
                            logger.warning("Keyframe generation failed", error=str(e))
                    
                    await asyncio.to_thread(generate_keyframe)
                    
                    if keyframe_path.exists() and keyframe_path.stat().st_size > 0:
                        keyframes.append(keyframe_path)
                        logger.debug("Keyframe generated successfully",
                                   keyframe_path=str(keyframe_path),
                                   timestamp=timestamp)
                
                # If we got keyframes, return them
                if keyframes:
                    return keyframes
                
                # Otherwise fall back to OpenCV method
                logger.warning("PySceneDetect didn't find any keyframes, falling back to OpenCV")
                
            except ImportError:
                logger.info("PySceneDetect not available, using OpenCV for keyframe extraction")
            except Exception as e:
                logger.warning("PySceneDetect failed, falling back to OpenCV", error=str(e))

            # Use OpenCV for scene change detection
            cap = cv2.VideoCapture(str(file_path))

            if not cap.isOpened():
                raise VideoProcessingError("Could not open video file with OpenCV")

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
                # Use unique timestamp-based filename to avoid pattern conflicts
                unique_id = int(time.time() * 1000000)  # Microsecond precision
                keyframe_path = self.temp_dir / f"keyframe_{unique_id}_{i}.{format}"

                try:
                    def generate_keyframe():
                        try:
                            # Build ffmpeg command using internal functions
                            input_stream = ffmpeg_input(str(file_path), ss=timestamp)
                            scaled_stream = ffmpeg_filter(input_stream, 'scale', size[0], size[1])
                            # Use -update option to write single image and avoid pattern warnings
                            output_stream = ffmpeg_output(scaled_stream, str(keyframe_path),
                                                        vframes=1, update=1)
                            ffmpeg_run(output_stream, overwrite_output=True, quiet=True, capture_stderr=True)

                        except FFmpegError as e:
                            # Capture stderr for detailed error information
                            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
                            # Filter out pattern warnings which are not actual errors
                            if "image sequence pattern" not in stderr_output.lower():
                                raise VideoProcessingError(f"FFmpeg keyframe error: {stderr_output}")
                            else:
                                logger.debug("FFmpeg pattern warning (non-critical)", stderr=stderr_output)
                        except Exception as e:
                            raise VideoProcessingError(f"Keyframe generation error: {str(e)}")

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

    async def _perform_ocr(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Perform OCR on a list of images."""
        try:
            ocr_engine = self._get_ocr_engine()
            if not ocr_engine:
                raise VideoProcessingError("OCR engine not available")
                
            results = {}
            
            for image_path in image_paths:
                try:
                    if self.config.ocr_engine == "tesseract":
                        # Use pytesseract
                        image = Image.open(image_path)
                        text = ocr_engine.image_to_string(image)
                        results[str(image_path)] = text
                    elif self.config.ocr_engine == "easyocr":
                        # Use easyocr
                        result = ocr_engine.readtext(str(image_path))
                        # Extract text from result
                        text = "\n".join([item[1] for item in result])
                        results[str(image_path)] = text
                except Exception as e:
                    logger.warning(f"OCR failed for {image_path}", error=str(e))
                    results[str(image_path)] = f"OCR failed: {str(e)}"
            
            return results
        except Exception as e:
            logger.error("OCR processing failed", error=str(e))
            raise VideoProcessingError(f"OCR processing failed: {str(e)}")

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