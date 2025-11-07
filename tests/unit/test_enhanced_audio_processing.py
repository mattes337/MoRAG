"""Tests for enhanced audio processing with speaker diarization and topic segmentation."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from morag_audio import AudioProcessor, AudioConfig, AudioProcessingResult, AudioTranscriptSegment
from morag_audio.services import (
    EnhancedSpeakerDiarization, DiarizationResult, SpeakerInfo, SpeakerSegment
)
from morag_audio.services import (
    EnhancedTopicSegmentation, TopicSegmentationResult, TopicSegment
)
from morag_audio import AudioConverter
from morag_core.interfaces.converter import ConversionOptions


class TestEnhancedSpeakerDiarization:
    """Test enhanced speaker diarization functionality."""

    @pytest.fixture
    def speaker_diarization_service(self):
        """Create speaker diarization service instance."""
        return EnhancedSpeakerDiarization()

    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create a minimal WAV file (just headers)
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            yield Path(f.name)
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_fallback_diarization_single_speaker(self, speaker_diarization_service, mock_audio_file):
        """Test fallback diarization for short audio (single speaker)."""
        with patch('morag.services.speaker_diarization.PYANNOTE_AVAILABLE', False):
            service = EnhancedSpeakerDiarization()

            result = await service.diarize_audio(mock_audio_file)

            assert isinstance(result, DiarizationResult)
            assert result.total_speakers == 1
            assert len(result.speakers) == 1
            assert len(result.segments) == 1
            assert result.speakers[0].speaker_id == "SPEAKER_00"
            assert result.model_used == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_diarization_multiple_speakers(self, speaker_diarization_service):
        """Test fallback diarization for long audio (multiple speakers)."""
        with patch('morag.services.speaker_diarization.PYANNOTE_AVAILABLE', False):
            service = EnhancedSpeakerDiarization()

            # Mock pydub to return long audio
            with patch('pydub.AudioSegment.from_file') as mock_audio:
                mock_audio_instance = Mock()
                mock_audio_instance.__len__ = Mock(return_value=120000)  # 2 minutes in ms
                mock_audio.return_value = mock_audio_instance

                result = await service.diarize_audio("fake_path.wav")

                assert result.total_speakers == 2
                assert len(result.speakers) == 2
                assert len(result.segments) == 2
                assert result.speakers[0].speaker_id == "SPEAKER_00"
                assert result.speakers[1].speaker_id == "SPEAKER_01"

    @pytest.mark.asyncio
    async def test_diarization_with_pyannote_mock(self, speaker_diarization_service):
        """Test diarization with mocked pyannote pipeline."""
        if not hasattr(speaker_diarization_service, 'pipeline') or not speaker_diarization_service.pipeline:
            pytest.skip("Pyannote not available")

        # Mock the pipeline result
        mock_annotation = Mock()
        mock_segments = [
            (Mock(start=0.0, end=5.0), None, "SPEAKER_00"),
            (Mock(start=5.0, end=10.0), None, "SPEAKER_01"),
            (Mock(start=10.0, end=15.0), None, "SPEAKER_00"),
        ]
        mock_annotation.itertracks.return_value = mock_segments

        with patch.object(speaker_diarization_service, '_run_diarization', return_value=mock_annotation):
            result = await speaker_diarization_service.diarize_audio("fake_path.wav")

            assert result.total_speakers == 2
            assert len(result.segments) == 3
            assert result.total_duration == 15.0


class TestEnhancedTopicSegmentation:
    """Test enhanced topic segmentation functionality."""

    @pytest.fixture
    def topic_segmentation_service(self):
        """Create topic segmentation service instance."""
        return EnhancedTopicSegmentation()

    @pytest.mark.asyncio
    async def test_fallback_segmentation_short_text(self, topic_segmentation_service):
        """Test fallback segmentation for short text."""
        with patch('morag.services.topic_segmentation.TOPIC_SEGMENTATION_AVAILABLE', False):
            service = EnhancedTopicSegmentation()

            text = "This is a short text. It has only two sentences."
            result = await service.segment_topics(text)

            assert isinstance(result, TopicSegmentationResult)
            assert result.total_topics == 1
            assert len(result.topics) == 1
            assert result.topics[0].title == "Main Content"
            assert result.segmentation_method == "single_topic"

    @pytest.mark.asyncio
    async def test_fallback_segmentation_long_text(self, topic_segmentation_service):
        """Test fallback segmentation for longer text."""
        with patch('morag.services.topic_segmentation.TOPIC_SEGMENTATION_AVAILABLE', False):
            service = EnhancedTopicSegmentation()

            # Create text with many sentences
            sentences = [f"This is sentence number {i}." for i in range(20)]
            text = " ".join(sentences)

            result = await service.segment_topics(text)

            assert result.total_topics > 1
            assert result.total_topics <= 3  # Should split into 2-3 topics
            assert result.segmentation_method == "simple_split"

    @pytest.mark.asyncio
    async def test_topic_segmentation_with_speaker_segments(self, topic_segmentation_service):
        """Test topic segmentation with speaker segment integration."""
        if not topic_segmentation_service.model_loaded:
            pytest.skip("Topic segmentation models not available")

        text = "First topic about technology. AI is changing the world. " \
               "Second topic about nature. Trees are important for environment. " \
               "Third topic about cooking. Recipes make food delicious."

        # Mock speaker segments
        speaker_segments = [
            Mock(speaker_id="SPEAKER_00", start_time=0.0, end_time=10.0),
            Mock(speaker_id="SPEAKER_01", start_time=10.0, end_time=20.0),
            Mock(speaker_id="SPEAKER_00", start_time=20.0, end_time=30.0),
        ]

        with patch.object(topic_segmentation_service, '_generate_embeddings') as mock_embeddings:
            # Mock embeddings to force topic boundaries
            mock_embeddings.return_value = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]]

            result = await topic_segmentation_service.segment_topics(
                text, speaker_segments=speaker_segments
            )

            assert result.total_topics >= 1
            # Check that speaker distribution is calculated
            for topic in result.topics:
                if topic.speaker_distribution:
                    assert isinstance(topic.speaker_distribution, dict)


class TestEnhancedAudioProcessor:
    """Test enhanced audio processor with integrated features."""

    @pytest.fixture
    def audio_processor(self):
        """Create audio processor instance."""
        config = AudioConfig(model_size="tiny")  # Use smallest model for testing
        return AudioProcessor(config)

    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            yield Path(f.name)
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_enhanced_audio_processing_disabled_features(self, audio_processor, mock_audio_file):
        """Test audio processing with enhanced features disabled."""
        with patch.object(audio_processor, '_transcribe_audio') as mock_transcribe:
            # Mock transcription result
            mock_result = AudioProcessingResult(
                text="Test transcription",
                language="en",
                confidence=0.9,
                duration=10.0,
                segments=[AudioTranscriptSegment("Test transcription", 0.0, 10.0, 0.9)],
                metadata={},
                processing_time=1.0,
                model_used="tiny"
            )
            mock_transcribe.return_value = mock_result

            result = await audio_processor.process_audio_file(
                mock_audio_file,
                enable_diarization=False,
                enable_topic_segmentation=False
            )

            assert isinstance(result, AudioProcessingResult)
            assert result.text == "Test transcription"
            assert result.speaker_diarization is None
            assert result.topic_segmentation is None

    @pytest.mark.asyncio
    async def test_enhanced_audio_processing_with_features(self, audio_processor, mock_audio_file):
        """Test audio processing with enhanced features enabled."""
        with patch.object(audio_processor, '_transcribe_audio') as mock_transcribe, \
             patch('morag.services.speaker_diarization.speaker_diarization_service') as mock_diarization, \
             patch('morag.services.topic_segmentation.topic_segmentation_service') as mock_segmentation:

            # Mock transcription result
            mock_result = AudioProcessingResult(
                text="Test transcription with multiple topics",
                language="en",
                confidence=0.9,
                duration=10.0,
                segments=[AudioTranscriptSegment("Test transcription with multiple topics", 0.0, 10.0, 0.9)],
                metadata={},
                processing_time=1.0,
                model_used="tiny"
            )
            mock_transcribe.return_value = mock_result

            # Mock diarization result
            mock_diarization_result = DiarizationResult(
                speakers=[SpeakerInfo("SPEAKER_00", 10.0, 1, 10.0, [0.9], 0.0, 10.0)],
                segments=[SpeakerSegment("SPEAKER_00", 0.0, 10.0, 10.0, 0.9)],
                total_speakers=1,
                total_duration=10.0,
                speaker_overlap_time=0.0,
                processing_time=0.5,
                model_used="test",
                confidence_threshold=0.5
            )
            mock_diarization.diarize_audio = AsyncMock(return_value=mock_diarization_result)

            # Mock segmentation result
            mock_segmentation_result = TopicSegmentationResult(
                topics=[TopicSegment("topic_1", "Test Topic", "Summary", ["Test transcription"])],
                total_topics=1,
                processing_time=0.3,
                model_used="test",
                similarity_threshold=0.7,
                segmentation_method="test"
            )
            mock_segmentation.segment_topics = AsyncMock(return_value=mock_segmentation_result)

            result = await audio_processor.process_audio_file(
                mock_audio_file,
                enable_diarization=True,
                enable_topic_segmentation=True
            )

            assert isinstance(result, AudioProcessingResult)
            assert result.text == "Test transcription with multiple topics"
            assert result.speaker_diarization is not None
            assert result.topic_segmentation is not None
            assert result.speaker_diarization.total_speakers == 1
            assert result.topic_segmentation.total_topics == 1


class TestEnhancedAudioConverter:
    """Test enhanced audio converter with integrated features."""

    @pytest.fixture
    def audio_converter(self):
        """Create audio converter instance."""
        return AudioConverter()

    @pytest.fixture
    def conversion_options(self):
        """Create conversion options."""
        return ConversionOptions(
            include_metadata=True,
            format_options={
                'enable_diarization': True,
                'enable_topic_segmentation': True,
                'include_timestamps': True,
                'include_speaker_info': True,
                'include_topic_info': True
            }
        )

    @pytest.mark.asyncio
    async def test_enhanced_audio_conversion(self, audio_converter, conversion_options):
        """Test enhanced audio conversion with all features."""
        with patch('morag.processors.audio.audio_processor') as mock_processor:
            # Mock enhanced audio processing result
            mock_diarization = DiarizationResult(
                speakers=[SpeakerInfo("SPEAKER_00", 30.0, 2, 15.0, [0.9, 0.8], 0.0, 30.0)],
                segments=[
                    SpeakerSegment("SPEAKER_00", 0.0, 15.0, 15.0, 0.9),
                    SpeakerSegment("SPEAKER_00", 15.0, 30.0, 15.0, 0.8)
                ],
                total_speakers=1,
                total_duration=30.0,
                speaker_overlap_time=0.0,
                processing_time=1.0,
                model_used="test",
                confidence_threshold=0.5
            )

            mock_segmentation = TopicSegmentationResult(
                topics=[
                    TopicSegment("topic_1", "Technology", "About AI", ["AI is changing the world"]),
                    TopicSegment("topic_2", "Nature", "About environment", ["Trees are important"])
                ],
                total_topics=2,
                processing_time=0.5,
                model_used="test",
                similarity_threshold=0.7,
                segmentation_method="semantic"
            )

            mock_result = AudioProcessingResult(
                text="AI is changing the world. Trees are important for environment.",
                language="en",
                confidence=0.85,
                duration=30.0,
                segments=[
                    AudioTranscriptSegment("AI is changing the world.", 0.0, 15.0, 0.9),
                    AudioTranscriptSegment("Trees are important for environment.", 15.0, 30.0, 0.8)
                ],
                metadata={'filename': 'test.wav', 'duration': 30.0},
                processing_time=2.0,
                model_used="tiny",
                speaker_diarization=mock_diarization,
                topic_segmentation=mock_segmentation
            )

            mock_processor.process_audio_file = AsyncMock(return_value=mock_result)

            # Test conversion
            result = await audio_converter.convert("test.wav", conversion_options)

            assert result.success
            assert "# Audio Transcription: test" in result.content
            assert "## Speakers" in result.content
            assert "## Topics" in result.content
            assert "Technology" in result.content
            assert "Nature" in result.content
            assert result.metadata.get('num_speakers') == 1
            assert result.metadata.get('num_topics') == 2

    def test_topic_dialogue_creation(self, audio_converter):
        """Test the topic dialogue creation method directly."""
        # Create test data
        topic = {
            'topic': 'Test Topic',
            'sentences': [
                'Hello, welcome to our discussion.',
                'Today we will talk about AI.',
                'AI is very important.'
            ]
        }

        speaker_segments = [
            {'speaker': 'SPEAKER_00', 'start_time': 0.0, 'end_time': 10.0},
            {'speaker': 'SPEAKER_01', 'start_time': 10.0, 'end_time': 20.0}
        ]

        transcript_segments = [
            Mock(text='Hello, welcome to our discussion.', start_time=0.0, end_time=5.0),
            Mock(text='Today we will talk about AI.', start_time=5.0, end_time=10.0),
            Mock(text='AI is very important.', start_time=15.0, end_time=20.0)
        ]

        # Test dialogue creation
        dialogue = audio_converter._create_topic_dialogue(
            topic, speaker_segments, transcript_segments
        )

        assert len(dialogue) == 3
        assert all('speaker' in entry and 'text' in entry for entry in dialogue)

        # Check that speakers are assigned
        speakers_found = set(entry['speaker'] for entry in dialogue)
        assert len(speakers_found) >= 1  # At least one speaker should be assigned

        # Check text content
        texts = [entry['text'] for entry in dialogue]
        assert 'Hello, welcome to our discussion.' in texts
        assert 'Today we will talk about AI.' in texts
        assert 'AI is very important.' in texts

    @pytest.mark.asyncio
    async def test_conversational_markdown_format(self, audio_converter):
        """Test that the markdown format includes conversational structure."""
        # Create mock enhanced result
        enhanced_result = Mock()
        enhanced_result.transcript = "Hello there. How are you? I'm fine, thanks."
        enhanced_result.summary = None  # No summary for this test
        enhanced_result.metadata = {
            'filename': 'test.wav',
            'duration': 30.0,
            'diarization_used': True,
            'topic_segmentation_used': True,
            'num_speakers': 2,
            'num_topics': 1
        }
        enhanced_result.segments = [
            Mock(text='Hello there.', start_time=0.0, end_time=2.0),
            Mock(text='How are you?', start_time=2.0, end_time=4.0),
            Mock(text="I'm fine, thanks.", start_time=4.0, end_time=6.0)
        ]
        enhanced_result.speakers = [
            {'id': 'SPEAKER_00', 'total_speaking_time': 15.0, 'segments_count': 2},
            {'id': 'SPEAKER_01', 'total_speaking_time': 15.0, 'segments_count': 1}
        ]
        enhanced_result.speaker_segments = [
            {'speaker': 'SPEAKER_00', 'start_time': 0.0, 'end_time': 4.0},
            {'speaker': 'SPEAKER_01', 'start_time': 4.0, 'end_time': 6.0}
        ]
        enhanced_result.topics = [
            {
                'topic': 'Greeting',
                'sentences': ['Hello there.', 'How are you?', "I'm fine, thanks."]
            }
        ]

        # Mock options
        options = Mock()
        options.include_metadata = True
        options.format_options = {
            'include_topic_info': True,
            'include_speaker_info': True,
            'include_timestamps': True
        }

        # Test markdown creation
        markdown = await audio_converter._create_enhanced_structured_markdown(enhanced_result, options)

        # Check basic structure
        assert "# Audio Transcription:" in markdown
        # Topics should now be main headers with timestamps, not under "## Topics"
        assert "# Greeting" in markdown or "# Topic" in markdown  # Topic as main header

        # Check for speaker dialogue format (should contain speaker IDs)
        assert "SPEAKER_" in markdown or "Speaker_" in markdown


if __name__ == "__main__":
    pytest.main([__file__])
