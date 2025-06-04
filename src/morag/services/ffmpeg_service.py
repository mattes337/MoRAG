"""FFmpeg service for video processing operations."""

import asyncio
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import structlog
import time

import ffmpeg

from morag.core.exceptions import ExternalServiceError, ProcessingError

logger = structlog.get_logger()

class FFmpegService:
    """Service for FFmpeg operations."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_ffmpeg"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def extract_audio(
        self,
        video_path: Path,
        output_format: str = "mp3",
        sample_rate: int = 16000,
        channels: int = 1,
        optimize_for_speed: bool = True
    ) -> Path:
        """Extract audio from video file."""
        try:
            output_path = self.temp_dir / f"audio_{int(time.time())}_{video_path.stem}.{output_format}"
            
            logger.debug("Extracting audio with FFmpeg",
                        video_path=str(video_path),
                        output_path=str(output_path),
                        format=output_format,
                        optimize_for_speed=optimize_for_speed)

            # Configure audio codec based on format with speed optimization
            if optimize_for_speed:
                try:
                    # Get metadata to check source audio codec
                    metadata = await self.extract_metadata(video_path)
                    source_codec = metadata.get('audio_codec', '').lower()

                    # Use stream copy if source and target are compatible
                    if output_format.lower() == "mp3" and "mp3" in source_codec:
                        codec = "copy"
                        use_copy = True
                        logger.debug("Using stream copy for MP3 extraction (minimal overhead)")
                    elif output_format.lower() == "aac" and "aac" in source_codec:
                        codec = "copy"
                        use_copy = True
                        logger.debug("Using stream copy for AAC extraction (minimal overhead)")
                    else:
                        use_copy = False
                        if output_format.lower() == "wav":
                            codec = "pcm_s16le"
                            logger.warning("Using uncompressed WAV format - will result in large files")
                        elif output_format.lower() == "mp3":
                            codec = "libmp3lame"
                        elif output_format.lower() == "aac":
                            codec = "aac"
                        elif output_format.lower() == "flac":
                            codec = "flac"
                        else:
                            codec = "libmp3lame"  # Default to MP3
                        logger.debug(f"Using {codec} encoding for audio extraction")
                except Exception as e:
                    logger.warning(f"Could not optimize audio extraction: {e}, using standard method")
                    use_copy = False
                    if output_format.lower() == "wav":
                        codec = "pcm_s16le"
                    elif output_format.lower() == "mp3":
                        codec = "libmp3lame"
                    elif output_format.lower() == "flac":
                        codec = "flac"
                    else:
                        codec = "libmp3lame"  # Default to MP3
            else:
                # Standard codec selection without optimization
                use_copy = False
                if output_format.lower() == "wav":
                    codec = "pcm_s16le"
                elif output_format.lower() == "mp3":
                    codec = "libmp3lame"
                elif output_format.lower() == "flac":
                    codec = "flac"
                else:
                    codec = "libmp3lame"  # Default to MP3
            
            # Extract audio using ffmpeg with appropriate parameters
            if use_copy:
                # When copying stream, don't modify audio parameters
                await asyncio.to_thread(
                    lambda: (
                        ffmpeg
                        .input(str(video_path))
                        .output(str(output_path), acodec=codec)
                        .overwrite_output()
                        .run(quiet=True, capture_stderr=True)
                    )
                )
            else:
                # When encoding, apply audio parameters and quality settings
                output_params = {
                    'acodec': codec,
                    'ac': channels,
                    'ar': sample_rate
                }

                # Add quality settings for specific codecs
                if codec == "libmp3lame":
                    output_params['audio_bitrate'] = "128k"  # Fast encoding
                elif codec == "aac":
                    output_params['audio_bitrate'] = "128k"

                await asyncio.to_thread(
                    lambda: (
                        ffmpeg
                        .input(str(video_path))
                        .output(str(output_path), **output_params)
                        .overwrite_output()
                        .run(quiet=True, capture_stderr=True)
                    )
                )
            
            if not output_path.exists():
                raise ProcessingError("Audio extraction failed - output file not created")
            
            logger.info("Audio extracted successfully",
                       output_path=str(output_path),
                       file_size=output_path.stat().st_size)
            
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg audio extraction failed: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error("FFmpeg audio extraction failed",
                        video_path=str(video_path),
                        error=error_msg)
            raise ExternalServiceError(error_msg, "ffmpeg")
        except Exception as e:
            logger.error("Audio extraction failed",
                        video_path=str(video_path),
                        error=str(e))
            raise ProcessingError(f"Audio extraction failed: {str(e)}")
    
    async def generate_thumbnails(
        self, 
        video_path: Path, 
        count: int = 5,
        size: Tuple[int, int] = (320, 240),
        format: str = "jpg"
    ) -> List[Path]:
        """Generate thumbnails from video at evenly spaced intervals."""
        try:
            thumbnails = []
            
            # Get video duration
            probe = await self.probe_video(video_path)
            duration = float(probe['format']['duration'])
            
            # Calculate timestamps
            if count == 1:
                timestamps = [duration / 2]
            else:
                timestamps = [i * duration / (count - 1) for i in range(count)]
            
            for i, timestamp in enumerate(timestamps):
                # Use unique timestamp-based filename to avoid pattern conflicts
                unique_id = int(time.time() * 1000000)  # Microsecond precision
                thumbnail_path = self.temp_dir / f"thumb_{unique_id}_{i}.{format}"

                await asyncio.to_thread(
                    lambda ts=timestamp, path=thumbnail_path: (
                        ffmpeg
                        .input(str(video_path), ss=ts)
                        .filter('scale', size[0], size[1])
                        .output(str(path), vframes=1, **{'update': 1})  # Use update flag
                        .overwrite_output()
                        .run(quiet=True, capture_stderr=True)
                    )
                )
                
                if thumbnail_path.exists():
                    thumbnails.append(thumbnail_path)
                    logger.debug("Thumbnail generated",
                                thumbnail_path=str(thumbnail_path),
                                timestamp=timestamp)
            
            logger.info("Thumbnails generated",
                       count=len(thumbnails),
                       requested=count)
            
            return thumbnails
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg thumbnail generation failed: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error("FFmpeg thumbnail generation failed",
                        video_path=str(video_path),
                        error=error_msg)
            raise ExternalServiceError(error_msg, "ffmpeg")
        except Exception as e:
            logger.error("Thumbnail generation failed",
                        video_path=str(video_path),
                        error=str(e))
            raise ProcessingError(f"Thumbnail generation failed: {str(e)}")
    
    async def extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract comprehensive video metadata."""
        try:
            probe = await self.probe_video(video_path)
            
            # Find video and audio streams
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            format_info = probe.get('format', {})
            
            metadata = {
                'duration': float(format_info.get('duration', 0)),
                'file_size': int(format_info.get('size', 0)),
                'format_name': format_info.get('format_name', 'unknown'),
                'bit_rate': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
                'streams': len(probe.get('streams', [])),
                'creation_time': format_info.get('tags', {}).get('creation_time'),
            }
            
            if video_stream:
                metadata.update({
                    'video_codec': video_stream.get('codec_name', 'unknown'),
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': self._parse_frame_rate(video_stream.get('r_frame_rate', '0/1')),
                    'video_bitrate': int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else None,
                    'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                })
            
            if audio_stream:
                metadata.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'audio_bitrate': int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None,
                })
            
            metadata['has_audio'] = audio_stream is not None
            metadata['has_video'] = video_stream is not None
            
            return metadata
            
        except Exception as e:
            logger.error("Metadata extraction failed",
                        video_path=str(video_path),
                        error=str(e))
            raise ProcessingError(f"Metadata extraction failed: {str(e)}")
    
    async def probe_video(self, video_path: Path) -> Dict[str, Any]:
        """Probe video file for metadata."""
        try:
            probe = await asyncio.to_thread(ffmpeg.probe, str(video_path))
            return probe
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg probe failed: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error("FFmpeg probe failed",
                        video_path=str(video_path),
                        error=error_msg)
            raise ExternalServiceError(error_msg, "ffmpeg")
    
    async def extract_keyframes(
        self, 
        video_path: Path, 
        max_frames: int = 10,
        size: Tuple[int, int] = (320, 240),
        format: str = "jpg"
    ) -> List[Path]:
        """Extract keyframes using FFmpeg scene detection."""
        try:
            keyframes = []
            
            # Use FFmpeg's scene detection filter with unique naming
            unique_id = int(time.time() * 1000000)  # Microsecond precision
            temp_pattern = self.temp_dir / f"keyframe_{unique_id}_%03d.{format}"

            await asyncio.to_thread(
                lambda: (
                    ffmpeg
                    .input(str(video_path))
                    .filter('select', f'gt(scene,0.3)')
                    .filter('scale', size[0], size[1])
                    .output(str(temp_pattern), vsync='vfr', vframes=max_frames, **{'update': 1})
                    .overwrite_output()
                    .run(quiet=True, capture_stderr=True)
                )
            )

            # Collect generated keyframes
            for i in range(max_frames):
                keyframe_path = self.temp_dir / f"keyframe_{unique_id}_{i:03d}.{format}"
                if keyframe_path.exists():
                    keyframes.append(keyframe_path)
            
            logger.info("Keyframes extracted using FFmpeg",
                       count=len(keyframes),
                       max_frames=max_frames)
            
            return keyframes
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg keyframe extraction failed: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error("FFmpeg keyframe extraction failed",
                        video_path=str(video_path),
                        error=error_msg)
            raise ExternalServiceError(error_msg, "ffmpeg")
        except Exception as e:
            logger.error("Keyframe extraction failed",
                        video_path=str(video_path),
                        error=str(e))
            raise ProcessingError(f"Keyframe extraction failed: {str(e)}")
    
    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """Parse frame rate string (e.g., '30/1') to float."""
        try:
            if '/' in frame_rate_str:
                numerator, denominator = frame_rate_str.split('/')
                return float(numerator) / float(denominator)
            return float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            return 0.0

# Global instance
ffmpeg_service = FFmpegService()
