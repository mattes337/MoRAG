"""Integration tests for video-audio processing pipeline."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import time

from morag.processors.video import VideoProcessor, VideoConfig, VideoProcessingResult, VideoMetadata
from morag.processors.audio import AudioProcessingResult, AudioTranscriptSegment
from morag.services.speaker_diarization import DiarizationResult, SpeakerInfo, SpeakerSegment
from morag.services.topic_segmentation import TopicSegmentationResult, TopicSegment
from morag.tasks.video_tasks import _process_video_file_impl
from morag.converters.video import VideoConverter
from morag.converters.base import ConversionOptions


class TestVideoAudioIntegration:
    """Test the integrated video-audio processing pipeline."""
    
    @pytest.fixture
    def mock_video_file(self):
        """Create a mock video file path."""
        return Path("/tmp/test_video.mp4")
    
    @pytest.fixture
    def mock_audio_result(self):
        """Create a mock enhanced audio processing result."""
        # Mock transcript segments
        segments = [
            AudioTranscriptSegment(
                text="Hello, welcome to our discussion.",
                start_time=0.0,
                end_time=3.0,
                confidence=0.95,
                speaker_id="SPEAKER_00",
                language="en"
            ),
            AudioTranscriptSegment(
                text="Thank you for having me.",
                start_time=3.5,
                end_time=6.0,
                confidence=0.92,
                speaker_id="SPEAKER_01",
                language="en"
            ),
            AudioTranscriptSegment(
                text="Let's talk about the main topic.",
                start_time=6.5,
                end_time=9.0,
                confidence=0.88,
                speaker_id="SPEAKER_00",
                language="en"
            )
        ]
        
        # Mock speaker diarization
        speakers = [
            SpeakerInfo("SPEAKER_00", 6.0, 2, 6.0, [0.95, 0.88], 0.0, 9.0),
            SpeakerInfo("SPEAKER_01", 2.5, 1, 2.5, [0.92], 3.5, 6.0)
        ]
        
        speaker_segments = [
            SpeakerSegment("SPEAKER_00", 0.0, 3.0, 3.0, 0.95),
            SpeakerSegment("SPEAKER_01", 3.5, 6.0, 2.5, 0.92),
            SpeakerSegment("SPEAKER_00", 6.5, 9.0, 2.5, 0.88)
        ]
        
        diarization_result = DiarizationResult(
            speakers=speakers,
            segments=speaker_segments,
            total_speakers=2,
            total_duration=9.0,
            speaker_overlap_time=0.0,
            processing_time=1.5,
            model_used="pyannote/speaker-diarization-3.1",
            confidence_threshold=0.7
        )
        
        # Mock topic segmentation
        topics = [
            TopicSegment(
                topic_id="topic_1",
                title="Introduction",
                summary="Welcome and initial greetings",
                sentences=["Hello, welcome to our discussion.", "Thank you for having me."],
                start_time=0.0,
                end_time=6.0,
                duration=6.0,
                confidence=0.9,
                keywords=["welcome", "discussion", "thank"],
                speaker_distribution={"SPEAKER_00": 50.0, "SPEAKER_01": 50.0}
            ),
            TopicSegment(
                topic_id="topic_2",
                title="Main Discussion",
                summary="Transition to the main topic",
                sentences=["Let's talk about the main topic."],
                start_time=6.5,
                end_time=9.0,
                duration=2.5,
                confidence=0.85,
                keywords=["talk", "main", "topic"],
                speaker_distribution={"SPEAKER_00": 100.0}
            )
        ]
        
        topic_result = TopicSegmentationResult(
            topics=topics,
            total_topics=2,
            processing_time=0.8,
            model_used="all-MiniLM-L6-v2",
            similarity_threshold=0.7,
            segmentation_method="semantic_embedding"
        )
        
        return AudioProcessingResult(
            text="Hello, welcome to our discussion. Thank you for having me. Let's talk about the main topic.",
            language="en",
            confidence=0.92,
            duration=9.0,
            segments=segments,
            metadata={"sample_rate": 16000, "channels": 1},
            processing_time=2.3,
            model_used="base",
            speaker_diarization=diarization_result,
            topic_segmentation=topic_result
        )
    
    @pytest.fixture
    def mock_video_metadata(self):
        """Create mock video metadata."""
        return VideoMetadata(
            duration=120.0,
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
            bitrate=5000000,
            file_size=50000000,
            format="mp4",
            has_audio=True,
            audio_codec="aac",
            creation_time=None
        )
    
    @pytest.mark.asyncio
    async def test_video_processor_enhanced_audio_integration(
        self, 
        mock_video_file, 
        mock_video_metadata, 
        mock_audio_result
    ):
        """Test that VideoProcessor correctly integrates enhanced audio processing."""
        
        with patch('morag.processors.video.ffmpeg_probe') as mock_probe, \
             patch('morag.processors.video.ffmpeg_run') as mock_ffmpeg, \
             patch('morag.processors.video.asyncio.to_thread') as mock_thread:
            
            # Mock ffmpeg probe response
            mock_probe.return_value = {
                'streams': [
                    {'codec_type': 'video', 'width': 1920, 'height': 1080, 'r_frame_rate': '30/1', 'codec_name': 'h264'},
                    {'codec_type': 'audio', 'codec_name': 'aac'}
                ],
                'format': {
                    'duration': '120.0',
                    'bit_rate': '5000000',
                    'size': '50000000',
                    'format_name': 'mp4'
                }
            }
            
            # Mock ffmpeg execution - handle both single function and function with args
            def mock_thread_side_effect(func, *args, **kwargs):
                if args or kwargs:
                    return func(*args, **kwargs)
                else:
                    return func()
            mock_thread.side_effect = mock_thread_side_effect
            
            # Mock audio processor
            mock_audio_processor = Mock()
            mock_audio_processor.process_audio_file = AsyncMock(return_value=mock_audio_result)
            
            # Create video processor and inject mock audio processor
            processor = VideoProcessor()
            processor._audio_processor = mock_audio_processor
            
            # Create config with enhanced audio enabled
            config = VideoConfig(
                extract_audio=True,
                enable_enhanced_audio=True,
                enable_speaker_diarization=True,
                enable_topic_segmentation=True,
                audio_model_size="base"
            )
            
            # Mock file existence
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'stat') as mock_stat:
                
                mock_stat.return_value.st_size = 1024
                
                # Process video
                result = await processor.process_video(mock_video_file, config)
                
                # Verify enhanced audio processing was called
                mock_audio_processor.process_audio_file.assert_called_once()
                
                # Verify result contains enhanced audio processing
                assert result.audio_processing_result is not None
                assert result.audio_processing_result == mock_audio_result
                assert result.audio_processing_result.speaker_diarization is not None
                assert result.audio_processing_result.topic_segmentation is not None
                assert result.audio_processing_result.speaker_diarization.total_speakers == 2
                assert result.audio_processing_result.topic_segmentation.total_topics == 2
    
    @pytest.mark.asyncio
    async def test_video_task_enhanced_integration(
        self, 
        mock_video_file, 
        mock_video_metadata, 
        mock_audio_result
    ):
        """Test that video task correctly handles enhanced audio processing results."""
        
        # Mock video processing result with enhanced audio
        mock_video_result = VideoProcessingResult(
            audio_path=Path("/tmp/extracted_audio.wav"),
            thumbnails=[],
            keyframes=[],
            metadata=mock_video_metadata,
            processing_time=5.0,
            temp_files=[],
            audio_processing_result=mock_audio_result
        )
        
        with patch('morag.tasks.video_tasks.video_processor') as mock_processor:
            mock_processor.process_video = AsyncMock(return_value=mock_video_result)
            mock_processor.cleanup_temp_files = Mock()
            
            # Create mock task
            mock_task = Mock()
            mock_task.update_status = Mock()
            
            # Process video file
            result = await _process_video_file_impl(
                mock_task,
                str(mock_video_file),
                "test_task_id",
                config={"enable_enhanced_audio": True},
                process_audio=True
            )
            
            # Verify enhanced audio results are included
            assert "audio_processing_result" in result
            assert result["audio_processing_result"] is not None
            
            audio_result = result["audio_processing_result"]
            assert audio_result["text"] == mock_audio_result.text
            assert audio_result["language"] == mock_audio_result.language
            assert audio_result["speaker_diarization"]["total_speakers"] == 2
            assert audio_result["topic_segmentation"]["total_topics"] == 2
    
    @pytest.mark.asyncio
    async def test_video_converter_enhanced_markdown(
        self, 
        mock_video_file, 
        mock_video_metadata, 
        mock_audio_result
    ):
        """Test that video converter creates enhanced markdown with topic headers and speaker dialogue."""
        
        # Mock video processing result with enhanced audio
        mock_video_result = VideoProcessingResult(
            audio_path=Path("/tmp/extracted_audio.wav"),
            thumbnails=[],
            keyframes=[],
            metadata=mock_video_metadata,
            processing_time=5.0,
            temp_files=[],
            audio_processing_result=mock_audio_result
        )
        
        with patch('morag.converters.video.video_processor') as mock_processor:
            mock_processor.process_video = AsyncMock(return_value=mock_video_result)
            
            # Create converter
            converter = VideoConverter()
            
            # Create conversion options
            options = ConversionOptions(
                include_metadata=True,
                format_options={
                    'include_audio': True,
                    'enable_enhanced_audio': True,
                    'enable_speaker_diarization': True,
                    'enable_topic_segmentation': True
                }
            )
            
            # Mock file existence for converter
            with patch.object(Path, 'exists', return_value=True):
                # Convert video
                result = await converter.convert(mock_video_file, options)
            
            # Verify conversion was successful
            assert result.success
            assert result.content
            
            # Verify enhanced audio content is included
            content = result.content
            assert "# Introduction [00:00 - 00:06]" in content
            assert "# Main Discussion [00:06 - 00:09]" in content
            assert "**SPEAKER_00**: Hello, welcome to our discussion." in content
            assert "**SPEAKER_01**: Thank you for having me." in content
            assert "Welcome and initial greetings" in content  # Topic summary
