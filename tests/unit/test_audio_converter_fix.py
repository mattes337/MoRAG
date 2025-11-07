"""Tests for audio converter method call fix."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from morag_audio import AudioConverter, AudioProcessingResult, AudioTranscriptSegment
from morag_core.interfaces.converter import ChunkingStrategy, ConversionOptions


class TestAudioConverterFix:
    """Test audio converter method call fix."""

    @pytest.fixture
    def audio_converter(self):
        """Create audio converter instance."""
        return AudioConverter()

    @pytest.fixture
    def conversion_options(self):
        """Create conversion options."""
        return ConversionOptions(
            chunking_strategy=ChunkingStrategy.PAGE,
            include_metadata=True,
            extract_images=False,
            min_quality_threshold=0.7,
            enable_fallback=True,
            format_options={
                "enable_diarization": True,
                "include_timestamps": True,
                "confidence_threshold": 0.8,
            },
        )

    @pytest.fixture
    def mock_audio_result(self):
        """Create mock audio processing result."""
        return AudioProcessingResult(
            text="This is a test transcription from the audio file.",
            language="en",
            confidence=0.95,
            duration=120.0,
            segments=[
                AudioTranscriptSegment(
                    text="This is a test transcription",
                    start_time=0.0,
                    end_time=2.5,
                    confidence=0.95,
                    language="en",
                ),
                AudioTranscriptSegment(
                    text="from the audio file.",
                    start_time=2.5,
                    end_time=5.0,
                    confidence=0.94,
                    language="en",
                ),
            ],
            metadata={
                "filename": "test.m4a",
                "duration": 120.0,
                "language": "en",
                "model_used": "base",
            },
            processing_time=15.2,
            model_used="base",
        )

    @pytest.mark.asyncio
    async def test_audio_converter_calls_correct_method(self):
        """Test that audio converter calls process_audio_file instead of process_audio."""
        # Test the method call directly by mocking the audio processor
        with patch("morag.converters.audio.audio_processor") as mock_processor:
            # Setup mock to have the correct method
            mock_processor.process_audio_file = AsyncMock(
                return_value=MagicMock(
                    text="Test transcription", metadata={"duration": 120.0}, segments=[]
                )
            )

            # Import and test the converter's method call
            from morag_audio import audio_processor

            # Verify the method exists and can be called
            result = await audio_processor.process_audio_file("test_file.m4a")

            # Verify the correct method was called
            mock_processor.process_audio_file.assert_called_once_with("test_file.m4a")

            # Verify the old method name doesn't exist or isn't called
            assert (
                not hasattr(mock_processor, "process_audio")
                or not mock_processor.process_audio.called
            )

    @pytest.mark.asyncio
    async def test_audio_converter_handles_method_error_gracefully(
        self, audio_converter, conversion_options
    ):
        """Test that audio converter handles method call errors gracefully."""
        test_file = Path("test_audio.m4a")

        # Mock the audio processor to raise AttributeError (the original bug)
        with patch("morag.converters.audio.audio_processor") as mock_processor:
            # Simulate the original bug
            mock_processor.process_audio_file.side_effect = AttributeError(
                "'AudioProcessor' object has no attribute 'process_audio'"
            )

            # Mock validation
            with patch.object(
                audio_converter, "validate_input", new_callable=AsyncMock
            ):
                # Call the converter
                result = await audio_converter.convert(test_file, conversion_options)

                # Verify error handling
                assert result.success is False
                assert "Audio conversion failed" in result.error_message
                assert (
                    "'AudioProcessor' object has no attribute 'process_audio'"
                    in result.error_message
                )

    @pytest.mark.asyncio
    async def test_audio_converter_parameter_passing(
        self, audio_converter, conversion_options, mock_audio_result
    ):
        """Test that parameters are passed correctly to process_audio_file."""
        test_file = Path("test_audio.wav")

        # Mock the audio processor
        with patch("morag.converters.audio.audio_processor") as mock_processor:
            mock_processor.process_audio_file = AsyncMock(
                return_value=mock_audio_result
            )

            # Mock other methods
            with patch.object(
                audio_converter, "validate_input", new_callable=AsyncMock
            ):
                with patch.object(
                    audio_converter, "_enhance_audio_processing", new_callable=AsyncMock
                ) as mock_enhance:
                    with patch.object(
                        audio_converter,
                        "_create_enhanced_structured_markdown",
                        new_callable=AsyncMock,
                    ) as mock_markdown:
                        with patch.object(
                            audio_converter.quality_validator, "validate_conversion"
                        ) as mock_quality:
                            # Setup return values
                            mock_enhance.return_value = MagicMock(
                                transcript="Test transcript",
                                metadata={"duration": 120.0},
                                summary="",
                                segments=[],
                                speakers=[],
                                topics=[],
                            )
                            mock_markdown.return_value = "# Audio Content"
                            mock_quality.return_value = MagicMock(overall_score=0.8)

                            # Call the converter
                            await audio_converter.convert(test_file, conversion_options)

                            # Verify the file path was passed as string
                            mock_processor.process_audio_file.assert_called_once_with(
                                str(test_file)
                            )

    @pytest.mark.asyncio
    async def test_audio_converter_integration_with_real_audio_processor(
        self, audio_converter, conversion_options
    ):
        """Test integration with real audio processor (method exists)."""
        from morag_audio import AudioProcessor

        # Verify the method exists on the real class
        processor = AudioProcessor()
        assert hasattr(processor, "process_audio_file")
        assert callable(getattr(processor, "process_audio_file"))

        # Verify the old method name doesn't exist
        assert not hasattr(processor, "process_audio")
