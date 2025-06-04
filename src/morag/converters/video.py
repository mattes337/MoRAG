"""Video to Markdown converter using existing MoRAG video processor."""

import time
from pathlib import Path
from typing import Union, List
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.video import video_processor, VideoConfig

logger = structlog.get_logger(__name__)


class VideoConverter(BaseConverter):
    """Video to Markdown converter using MoRAG's video processor."""
    
    def __init__(self):
        super().__init__("MoRAG Video Converter")
        self.supported_formats = ['video', 'mp4', 'avi', 'mov', 'mkv']
        self.quality_validator = ConversionQualityValidator()
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert video to structured markdown with transcription and keyframes.
        
        Args:
            file_path: Path to video file
            options: Conversion options
            
        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        await self.validate_input(file_path)
        
        logger.info(
            "Starting video conversion",
            file_path=str(file_path),
            extract_keyframes=options.format_options.get('extract_keyframes', True),
            include_audio=options.format_options.get('include_audio', True)
        )
        
        try:
            # Create video configuration from conversion options
            video_config = VideoConfig(
                extract_audio=options.format_options.get('include_audio', True),
                generate_thumbnails=options.format_options.get('generate_thumbnails', True),
                thumbnail_count=options.format_options.get('thumbnail_count', 5),
                extract_keyframes=options.format_options.get('extract_keyframes', True),
                max_keyframes=options.format_options.get('max_keyframes', 10),
                audio_format=options.format_options.get('audio_format', 'wav'),
                thumbnail_size=tuple(options.format_options.get('thumbnail_size', [320, 240])),
                thumbnail_format=options.format_options.get('thumbnail_format', 'jpg'),
                keyframe_threshold=options.format_options.get('keyframe_threshold', 0.3),
                # Enhanced audio processing options
                enable_enhanced_audio=options.format_options.get('enable_enhanced_audio', True),
                enable_speaker_diarization=options.format_options.get('enable_speaker_diarization', True),
                enable_topic_segmentation=options.format_options.get('enable_topic_segmentation', True),
                audio_model_size=options.format_options.get('audio_model_size', 'base')
            )

            # Use existing MoRAG video processor
            video_result = await video_processor.process_video(file_path, video_config)

            # Convert to structured markdown
            markdown_content = await self._create_structured_markdown(video_result, options, file_path)
            
            # Calculate quality score
            # Convert VideoMetadata to dictionary for quality validation
            metadata_dict = self._enhance_metadata(video_result.metadata, file_path)
            quality_score = self.quality_validator.validate_conversion(str(file_path), ConversionResult(
                content=markdown_content,
                metadata=metadata_dict
            ))
            
            processing_time = time.time() - start_time
            
            result = ConversionResult(
                content=markdown_content,
                metadata=metadata_dict,  # Use the already converted metadata
                quality_score=quality_score,
                processing_time=processing_time,
                success=True,
                original_format='video',
                converter_used=self.name
            )
            
            logger.info(
                "Video conversion completed",
                processing_time=processing_time,
                quality_score=quality_score.overall_score,
                word_count=result.word_count,
                duration=video_result.metadata.duration
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Video conversion failed: {str(e)}"
            
            logger.error(
                "Video conversion failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time=processing_time
            )
            
            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                original_format='video',
                converter_used=self.name
            )
    
    async def _create_structured_markdown(self, video_result, options: ConversionOptions, file_path: Path) -> str:
        """Create structured markdown from video processing result.

        Args:
            video_result: Video processing result from MoRAG processor
            options: Conversion options
            file_path: Path to the original video file

        Returns:
            Structured markdown content
        """
        sections = []

        # Document header - use file path since metadata doesn't have filename
        title = file_path.stem
        sections.append(f"# Video Analysis: {title}")
        sections.append("")

        # Metadata section
        if options.include_metadata:
            sections.append("## Video Information")
            sections.append("")

            duration = video_result.metadata.duration
            duration_str = f"{duration:.1f} seconds" if duration < 60 else f"{duration/60:.1f} minutes"

            # Create resolution string
            resolution = f"{video_result.metadata.width}x{video_result.metadata.height}"

            metadata_items = [
                ("**Duration**", duration_str),
                ("**Resolution**", resolution),
                ("**Frame Rate**", f"{video_result.metadata.fps:.1f} fps"),
                ("**Format**", video_result.metadata.format),
                ("**Codec**", video_result.metadata.codec),
                ("**File Size**", f"{video_result.metadata.file_size / (1024*1024):.1f} MB"),
                ("**Has Audio**", "Yes" if video_result.metadata.has_audio else "No"),
                ("**Processing Method**", "MoRAG Video Processor")
            ]

            if video_result.metadata.has_audio and video_result.metadata.audio_codec:
                metadata_items.append(("**Audio Codec**", video_result.metadata.audio_codec))

            for label, value in metadata_items:
                if value and value != 'Unknown':
                    sections.append(f"{label}: {value}")

            sections.append("")
        
        # Summary section (if available)
        if hasattr(video_result, 'summary') and video_result.summary:
            sections.append("## Summary")
            sections.append("")
            sections.append(video_result.summary)
            sections.append("")
        
        # Audio transcript section with enhanced processing results
        if video_result.audio_path and video_result.metadata.has_audio:
            if video_result.audio_processing_result:
                # Use enhanced audio processing results
                audio_result = video_result.audio_processing_result
                sections.extend(self._create_enhanced_audio_markdown(audio_result, video_result.metadata.duration))
            else:
                # Fallback for basic audio extraction
                duration = video_result.metadata.duration
                duration_str = self._format_timestamp(duration)
                sections.append(f"# Audio Content [00:00 - {duration_str}]")
                sections.append("")
                sections.append("*Audio track extracted but enhanced processing not available.*")
                sections.append(f"*Audio file: {video_result.audio_path.name}*")
                sections.append("")
        elif not video_result.metadata.has_audio:
            sections.append("# Audio Content")
            sections.append("")
            sections.append("*No audio track detected in this video.*")
            sections.append("")
        
        # Keyframes section
        if hasattr(video_result, 'keyframes') and video_result.keyframes and options.extract_images:
            sections.append("## Visual Timeline")
            sections.append("")

            for i, keyframe_path in enumerate(video_result.keyframes, 1):
                # keyframe_path is a Path object, not a dictionary
                # Extract timestamp from filename if possible, otherwise use index
                timestamp = i * (video_result.metadata.duration / len(video_result.keyframes)) if video_result.metadata.duration > 0 else i * 10
                timestamp_str = self._format_timestamp(timestamp)

                sections.append(f"### Keyframe {i} ({timestamp_str})")
                sections.append(f"**File**: {keyframe_path.name}")

                # Note: keyframes are just image files, no additional metadata available
                sections.append("*Keyframe extracted from video for visual reference*")
                sections.append("")
        
        # Scene analysis (if available)
        if hasattr(video_result, 'scenes') and video_result.scenes:
            sections.append("## Scene Analysis")
            sections.append("")

            for i, scene in enumerate(video_result.scenes, 1):
                # Check if scene is a dictionary or has the expected structure
                if isinstance(scene, dict):
                    start_time = self._format_timestamp(scene.get('start_time', 0))
                    end_time = self._format_timestamp(scene.get('end_time', 0))

                    sections.append(f"### Scene {i} ({start_time} - {end_time})")

                    if scene.get('description'):
                        sections.append(f"**Description**: {scene['description']}")

                    if scene.get('activity'):
                        sections.append(f"**Activity**: {scene['activity']}")
                else:
                    # Fallback for unexpected scene format
                    sections.append(f"### Scene {i}")
                    sections.append(f"**Content**: {str(scene)}")

                sections.append("")

        return "\n".join(sections)

    def _create_enhanced_audio_markdown(self, audio_result, video_duration: float) -> List[str]:
        """Create enhanced audio markdown with topic segmentation and speaker diarization.

        Args:
            audio_result: AudioProcessingResult from enhanced audio processing
            video_duration: Total video duration in seconds

        Returns:
            List of markdown lines for audio content
        """
        sections = []

        # Check if we have topic segmentation results
        if audio_result.topic_segmentation and audio_result.topic_segmentation.topics:
            # Create conversational format with topic headers and speaker dialogue
            topic_counter = 1
            for topic in audio_result.topic_segmentation.topics:
                # Calculate proper timestamp - use topic index if start_time is invalid
                start_seconds = 0
                if topic.start_time is not None and topic.start_time >= 0:
                    start_seconds = int(topic.start_time)
                    logger.debug("Using topic start time",
                               topic_id=topic.topic_id,
                               start_time=topic.start_time,
                               start_seconds=start_seconds)
                else:
                    # Fallback: estimate based on topic position and video duration
                    estimated_start = (topic_counter - 1) * (video_duration / len(audio_result.topic_segmentation.topics))
                    start_seconds = int(estimated_start)
                    logger.debug("Using estimated start time",
                               topic_id=topic.topic_id,
                               estimated_start=estimated_start,
                               start_seconds=start_seconds)

                # Generate proper topic title
                topic_title = f"Discussion Topic {topic_counter}"
                if topic.title and topic.title != f"Topic {topic.topic_id.split('_')[-1]}":
                    topic_title = topic.title

                sections.append(f"# {topic_title} [{start_seconds}]")
                sections.append("")

                # Create speaker dialogue for this topic with improved deduplication
                topic_dialogue = self._create_topic_dialogue(topic, audio_result)

                # Enhanced deduplication - track both full text and normalized text
                added_texts = set()
                added_normalized = set()

                for dialogue_line in topic_dialogue:
                    dialogue_clean = dialogue_line.strip()
                    if not dialogue_clean:
                        continue

                    # Extract just the text part (after speaker label)
                    if ": " in dialogue_clean:
                        text_part = dialogue_clean.split(": ", 1)[1]
                        normalized_text = text_part.lower().strip()
                    else:
                        normalized_text = dialogue_clean.lower().strip()

                    # Check for duplicates using both exact and normalized matching
                    if (dialogue_clean not in added_texts and
                        normalized_text not in added_normalized and
                        len(normalized_text) > 3):  # Skip very short texts

                        sections.append(dialogue_line)
                        added_texts.add(dialogue_clean)
                        added_normalized.add(normalized_text)
                    else:
                        logger.debug("Skipping duplicate dialogue line",
                                   line=dialogue_clean[:50],
                                   reason="duplicate_text")

                sections.append("")
                topic_counter += 1
        else:
            # Fallback to simple transcript format
            duration_str = self._format_timestamp(video_duration)
            sections.append(f"# Audio Transcript [00:00 - {duration_str}]")
            sections.append("")

            if audio_result.text:
                sections.append(audio_result.text)
            else:
                sections.append("*No transcript available.*")
            sections.append("")

            # Add speaker information if available
            if audio_result.speaker_diarization:
                sections.append("## Speaker Information")
                sections.append("")
                sections.append(f"**Total Speakers Detected**: {audio_result.speaker_diarization.total_speakers}")
                sections.append(f"**Total Duration**: {self._format_timestamp(audio_result.speaker_diarization.total_duration)}")
                sections.append("")

        return sections

    def _create_topic_dialogue(self, topic, audio_result) -> List[str]:
        """Create speaker dialogue for a topic segment.

        Args:
            topic: TopicSegment with timing and content information
            audio_result: AudioProcessingResult with speaker and transcript data

        Returns:
            List of markdown lines for topic dialogue
        """
        dialogue_lines = []

        if not topic.sentences:
            return ["*No dialogue available for this topic.*"]

        # Clean and deduplicate sentences first
        unique_sentences = []
        seen_sentences = set()

        for sentence in topic.sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and clean_sentence not in seen_sentences:
                unique_sentences.append(clean_sentence)
                seen_sentences.add(clean_sentence)

        if not unique_sentences:
            return ["*No dialogue available for this topic.*"]

        # Try to map sentences to speakers using timing and transcript information
        if (audio_result.speaker_diarization and
            audio_result.speaker_diarization.segments and
            hasattr(audio_result, 'segments') and audio_result.segments):

            # Create a mapping of text to speaker based on transcript segments
            text_to_speaker = {}

            for transcript_seg in audio_result.segments:
                if hasattr(transcript_seg, 'text') and hasattr(transcript_seg, 'start_time'):
                    # Find the speaker who was talking during this transcript segment
                    for speaker_seg in audio_result.speaker_diarization.segments:
                        # Check if transcript segment overlaps with speaker segment
                        if (speaker_seg.start_time <= transcript_seg.start_time <= speaker_seg.end_time or
                            speaker_seg.start_time <= transcript_seg.end_time <= speaker_seg.end_time):

                            # Map this text to the speaker
                            text_to_speaker[transcript_seg.text.strip().lower()] = speaker_seg.speaker_id
                            break

            # Now map sentences to speakers
            for sentence in unique_sentences:
                speaker_id = "SPEAKER_00"  # Default fallback

                # Try to find exact or partial match in transcript
                sentence_lower = sentence.lower()
                best_match_speaker = None
                best_match_score = 0

                for transcript_text, mapped_speaker in text_to_speaker.items():
                    # Check for exact match or high overlap
                    if sentence_lower == transcript_text:
                        speaker_id = mapped_speaker
                        break
                    elif sentence_lower in transcript_text or transcript_text in sentence_lower:
                        # Calculate overlap score
                        overlap = len(set(sentence_lower.split()) & set(transcript_text.split()))
                        if overlap > best_match_score:
                            best_match_score = overlap
                            best_match_speaker = mapped_speaker

                # Use best match if found
                if best_match_speaker and best_match_score > 1:
                    speaker_id = best_match_speaker

                dialogue_lines.append(f"{speaker_id}: {sentence}")

        elif (audio_result.speaker_diarization and
              audio_result.speaker_diarization.segments):

            # Fallback: alternate between available speakers
            all_speakers = list(set(seg.speaker_id for seg in audio_result.speaker_diarization.segments))
            if all_speakers:
                for i, sentence in enumerate(unique_sentences):
                    speaker_id = all_speakers[i % len(all_speakers)]
                    dialogue_lines.append(f"{speaker_id}: {sentence}")
            else:
                # No speakers found, use default
                for sentence in unique_sentences:
                    dialogue_lines.append(f"SPEAKER_00: {sentence}")
        else:
            # No speaker diarization available, use simple format
            for sentence in unique_sentences:
                dialogue_lines.append(f"SPEAKER_00: {sentence}")

        return dialogue_lines

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _enhance_metadata(self, original_metadata, file_path: Path) -> dict:
        """Enhance metadata with additional information.

        Args:
            original_metadata: Original VideoMetadata from processor
            file_path: Path to original file

        Returns:
            Enhanced metadata dictionary
        """
        # Convert VideoMetadata dataclass to dictionary
        enhanced = {
            'duration': original_metadata.duration,
            'width': original_metadata.width,
            'height': original_metadata.height,
            'fps': original_metadata.fps,
            'codec': original_metadata.codec,
            'bitrate': original_metadata.bitrate,
            'file_size': original_metadata.file_size,
            'format': original_metadata.format,
            'has_audio': original_metadata.has_audio,
            'audio_codec': original_metadata.audio_codec,
            'creation_time': original_metadata.creation_time
        }

        # Add file information
        enhanced.update({
            'original_filename': file_path.name,
            'original_file_size': file_path.stat().st_size,
            'conversion_format': 'video_to_markdown',
            'converter_version': '1.0.0',
            'file_extension': file_path.suffix.lower()
        })

        return enhanced
