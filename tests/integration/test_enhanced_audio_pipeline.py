"""Integration tests for enhanced audio processing pipeline."""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os
import wave
import struct
import numpy as np

from morag_audio import AudioProcessor, AudioConfig
from morag_audio.services import speaker_diarization_service
from morag_audio.services import topic_segmentation_service
from morag_audio import AudioConverter
from morag_core.interfaces.converter import ConversionOptions
from morag_core.config import settings


class TestEnhancedAudioPipeline:
    """Integration tests for the complete enhanced audio processing pipeline."""

    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample WAV file for testing."""
        # Create a simple sine wave audio file
        sample_rate = 16000
        duration = 10  # 10 seconds
        frequency = 440  # A4 note

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = np.sin(2 * np.pi * frequency * t)

        # Add some variation to simulate speech-like patterns
        envelope = np.exp(-t / 5)  # Decay envelope
        wave_data = wave_data * envelope * 0.3

        # Convert to 16-bit integers
        wave_data = (wave_data * 32767).astype(np.int16)

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())

            yield Path(f.name)

        # Cleanup
        os.unlink(f.name)

    @pytest.fixture
    def audio_processor(self):
        """Create audio processor with test configuration."""
        config = AudioConfig(
            model_size="tiny",  # Use smallest model for faster testing
            device="cpu",
            compute_type="int8"
        )
        return AudioProcessor(config)

    @pytest.fixture
    def audio_converter(self):
        """Create audio converter instance."""
        return AudioConverter()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_basic_audio_processing_pipeline(self, audio_processor, sample_audio_file):
        """Test basic audio processing without enhanced features."""
        result = await audio_processor.process_audio_file(
            sample_audio_file,
            enable_diarization=False,
            enable_topic_segmentation=False
        )

        assert result is not None
        assert isinstance(result.text, str)
        assert result.language is not None
        assert result.duration > 0
        assert len(result.segments) >= 0
        assert result.speaker_diarization is None
        assert result.topic_segmentation is None
        assert result.processing_time > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_speaker_diarization_service(self, sample_audio_file):
        """Test speaker diarization service independently."""
        result = await speaker_diarization_service.diarize_audio(sample_audio_file)

        assert result is not None
        assert result.total_speakers >= 1
        assert len(result.speakers) == result.total_speakers
        assert len(result.segments) >= 1
        assert result.total_duration > 0
        assert result.processing_time >= 0

        # Check speaker information
        for speaker in result.speakers:
            assert speaker.speaker_id is not None
            assert speaker.total_speaking_time >= 0
            assert speaker.segment_count >= 1
            assert speaker.first_appearance >= 0
            assert speaker.last_appearance >= speaker.first_appearance

        # Check segments
        for segment in result.segments:
            assert segment.speaker_id is not None
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time
            assert segment.duration == segment.end_time - segment.start_time

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_topic_segmentation_service(self):
        """Test topic segmentation service independently."""
        # Create sample text with multiple topics
        text = """
        Artificial intelligence is revolutionizing many industries. Machine learning algorithms
        can process vast amounts of data and identify patterns that humans might miss. Deep
        learning networks are particularly effective for image and speech recognition tasks.

        Climate change is one of the most pressing issues of our time. Rising global temperatures
        are causing ice caps to melt and sea levels to rise. Renewable energy sources like solar
        and wind power are becoming increasingly important for reducing carbon emissions.

        Cooking is both an art and a science. Understanding the chemical reactions that occur
        during cooking can help create better flavors and textures. Proper temperature control
        and timing are crucial for achieving the desired results in any recipe.
        """

        result = await topic_segmentation_service.segment_topics(text)

        assert result is not None
        assert result.total_topics >= 1
        assert len(result.topics) == result.total_topics
        assert result.processing_time >= 0

        # Check topic information
        for topic in result.topics:
            assert topic.topic_id is not None
            assert topic.title is not None
            assert len(topic.sentences) >= 1
            assert topic.confidence >= 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_enhanced_audio_processing_pipeline(self, audio_processor, sample_audio_file):
        """Test complete enhanced audio processing pipeline."""
        # Test with enhanced features enabled
        result = await audio_processor.process_audio_file(
            sample_audio_file,
            enable_diarization=True,
            enable_topic_segmentation=True
        )

        assert result is not None
        assert isinstance(result.text, str)
        assert result.language is not None
        assert result.duration > 0
        assert result.processing_time > 0

        # Check that enhanced features were attempted (may fallback if models not available)
        # Speaker diarization should at least provide fallback result
        if result.speaker_diarization:
            assert result.speaker_diarization.total_speakers >= 1
            assert len(result.speaker_diarization.speakers) >= 1
            assert len(result.speaker_diarization.segments) >= 1

        # Topic segmentation should work if text is available
        if result.text and result.topic_segmentation:
            assert result.topic_segmentation.total_topics >= 1
            assert len(result.topic_segmentation.topics) >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_audio_converter_integration(self, audio_converter, sample_audio_file):
        """Test complete audio converter with enhanced features."""
        options = ConversionOptions(
            include_metadata=True,
            format_options={
                'enable_diarization': True,
                'enable_topic_segmentation': True,
                'include_timestamps': True,
                'include_speaker_info': True,
                'include_topic_info': True
            }
        )

        result = await audio_converter.convert(sample_audio_file, options)

        assert result is not None
        assert result.success
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert result.processing_time > 0

        # Check markdown structure - updated for new format
        content = result.content
        assert "# Audio Transcription:" in content
        assert "## Audio Information" in content
        # Should have topic headers instead of transcript/processing sections
        assert "# Topic" in content or "# Main Content" in content

        # Check metadata
        assert 'diarization_used' in result.metadata
        assert 'topic_segmentation_used' in result.metadata

        # Should have speaker labels in content
        assert "Speaker_00:" in content or "SPEAKER_" in content

        if result.metadata.get('topic_segmentation_used'):
            # Topics are now main headers, not under a "## Topics" section
            assert result.metadata.get('num_topics', 0) >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_configuration_integration(self, audio_processor, sample_audio_file):
        """Test that configuration settings are properly used."""
        # Test with settings-based configuration
        original_diarization = settings.enable_speaker_diarization
        original_segmentation = settings.enable_topic_segmentation

        try:
            # Temporarily modify settings
            settings.enable_speaker_diarization = True
            settings.enable_topic_segmentation = True

            result = await audio_processor.process_audio_file(sample_audio_file)

            # Should use settings defaults
            assert result is not None

            # Test explicit override
            result_disabled = await audio_processor.process_audio_file(
                sample_audio_file,
                enable_diarization=False,
                enable_topic_segmentation=False
            )

            assert result_disabled.speaker_diarization is None
            assert result_disabled.topic_segmentation is None

        finally:
            # Restore original settings
            settings.enable_speaker_diarization = original_diarization
            settings.enable_topic_segmentation = original_segmentation

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling_integration(self, audio_processor):
        """Test error handling in the enhanced pipeline."""
        # Test with non-existent file
        with pytest.raises(Exception):
            await audio_processor.process_audio_file("non_existent_file.wav")

        # Test with invalid file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'invalid audio data')
            invalid_file = Path(f.name)

        try:
            # Should handle gracefully and potentially provide fallback
            result = await audio_processor.process_audio_file(invalid_file)
            # If it doesn't raise an exception, it should provide some result
            if result:
                assert isinstance(result.text, str)
        except Exception:
            # Exception is acceptable for invalid files
            pass
        finally:
            os.unlink(invalid_file)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_performance_benchmarks(self, audio_processor, sample_audio_file):
        """Test performance benchmarks for the enhanced pipeline."""
        # Measure processing time for different configurations

        # Basic processing
        start_time = asyncio.get_event_loop().time()
        result_basic = await audio_processor.process_audio_file(
            sample_audio_file,
            enable_diarization=False,
            enable_topic_segmentation=False
        )
        basic_time = asyncio.get_event_loop().time() - start_time

        # Enhanced processing
        start_time = asyncio.get_event_loop().time()
        result_enhanced = await audio_processor.process_audio_file(
            sample_audio_file,
            enable_diarization=True,
            enable_topic_segmentation=True
        )
        enhanced_time = asyncio.get_event_loop().time() - start_time

        # Basic assertions
        assert result_basic.processing_time > 0
        assert result_enhanced.processing_time > 0
        assert basic_time > 0
        assert enhanced_time > 0

        # Enhanced processing should take longer (but not excessively)
        # Allow for some variance in timing
        assert enhanced_time >= basic_time * 0.8  # At least 80% of basic time

        print(f"Basic processing time: {basic_time:.2f}s")
        print(f"Enhanced processing time: {enhanced_time:.2f}s")
        print(f"Overhead: {((enhanced_time - basic_time) / basic_time * 100):.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
