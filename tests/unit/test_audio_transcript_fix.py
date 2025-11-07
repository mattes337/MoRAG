#!/usr/bin/env python3
"""
Test for Audio Transcript Missing Fix

This test ensures that the enhanced audio markdown conversion includes
the transcript section and properly formats the content.
"""

import pytest

from morag_audio import AudioConverter
from morag_services import ConversionOptions, ChunkingStrategy
from morag_audio import AudioProcessingResult, AudioTranscriptSegment


class TestAudioTranscriptFix:
    """Test audio transcript missing fix."""

    @pytest.fixture
    def audio_converter(self):
        """Create audio converter instance."""
        return AudioConverter()

    @pytest.fixture
    def mock_audio_result(self):
        """Create mock audio processing result with segments."""
        return AudioProcessingResult(
            text="Dies ist ein Testtext, bitte transcribieren.",
            language="de",
            confidence=0.95,
            duration=5.29,
            segments=[
                AudioTranscriptSegment(
                    text="Dies ist ein Testtext, bitte transcribieren.",
                    start_time=0.0,
                    end_time=5.29,
                    confidence=0.95,
                    language="de"
                )
            ],
            metadata={
                'filename': 'recording.m4a',
                'duration': 5.29,
                'language': 'de',
                'model_used': 'base'
            },
            processing_time=2.0,
            model_used="base"
        )

    @pytest.fixture
    def conversion_options(self):
        """Create conversion options."""
        return ConversionOptions(
            chunking_strategy=ChunkingStrategy.PAGE,
            include_metadata=False,
            extract_images=False,
            format_options={
                'enable_diarization': True,
                'include_timestamps': True,
                'confidence_threshold': 0.8
            }
        )

    @pytest.mark.asyncio
    async def test_enhanced_markdown_includes_transcript(
        self,
        audio_converter,
        mock_audio_result,
        conversion_options
    ):
        """Test that enhanced markdown includes transcript section."""
        # Create enhanced result
        enhanced_result = type('EnhancedAudioResult', (), {
            'transcript': mock_audio_result.text,
            'metadata': mock_audio_result.metadata.copy(),
            'summary': '',
            'segments': mock_audio_result.segments,
            'speakers': [],
            'topics': []
        })()

        # Generate markdown
        markdown = await audio_converter._create_enhanced_structured_markdown(
            enhanced_result,
            conversion_options
        )

        # Verify transcript section is present
        assert "## Transcript" in markdown
        assert "Dies ist ein Testtext, bitte transcribieren." in markdown
        assert "**[00:00 - 00:05]**" in markdown
        assert "## Processing Details" in markdown

    @pytest.mark.asyncio
    async def test_enhanced_markdown_with_multiple_segments(
        self,
        audio_converter,
        conversion_options
    ):
        """Test enhanced markdown with multiple audio segments."""
        # Create result with multiple segments
        enhanced_result = type('EnhancedAudioResult', (), {
            'transcript': "Hello world. This is a test.",
            'metadata': {'filename': 'test.wav', 'model_used': 'base'},
            'summary': '',
            'segments': [
                AudioTranscriptSegment(
                    text="Hello world.",
                    start_time=0.0,
                    end_time=2.0,
                    confidence=0.95,
                    language="en"
                ),
                AudioTranscriptSegment(
                    text="This is a test.",
                    start_time=2.0,
                    end_time=4.0,
                    confidence=0.90,
                    language="en"
                )
            ],
            'speakers': [],
            'topics': []
        })()

        # Generate markdown
        markdown = await audio_converter._create_enhanced_structured_markdown(
            enhanced_result,
            conversion_options
        )

        # Verify both segments are present
        assert "Hello world." in markdown
        assert "This is a test." in markdown
        assert "**[00:00 - 00:02]**" in markdown
        assert "**[00:02 - 00:04]**" in markdown

    @pytest.mark.asyncio
    async def test_enhanced_markdown_without_segments(
        self,
        audio_converter,
        conversion_options
    ):
        """Test enhanced markdown fallback when no segments available."""
        # Create result without segments
        enhanced_result = type('EnhancedAudioResult', (), {
            'transcript': "Simple transcript text.",
            'metadata': {'filename': 'test.wav', 'model_used': 'base'},
            'summary': '',
            'segments': [],
            'speakers': [],
            'topics': []
        })()

        # Generate markdown
        markdown = await audio_converter._create_enhanced_structured_markdown(
            enhanced_result,
            conversion_options
        )

        # Verify fallback transcript is used - now should be in topic format
        assert "# Topic" in markdown or "# Main Content" in markdown
        assert "Simple transcript text." in markdown
        # Should not have timestamps when no segments
        assert "**[" not in markdown

    @pytest.mark.asyncio
    async def test_enhanced_markdown_with_topics(
        self,
        audio_converter,
        mock_audio_result,
        conversion_options
    ):
        """Test enhanced markdown with topic segmentation."""
        # Create enhanced result with topics
        enhanced_result = type('EnhancedAudioResult', (), {
            'transcript': mock_audio_result.text,
            'metadata': mock_audio_result.metadata.copy(),
            'summary': '',
            'segments': mock_audio_result.segments,
            'speakers': [],
            'topics': [
                {
                    'topic': 'Test Topic',
                    'sentences': ['Dies ist ein Testtext.', 'Bitte transcribieren.']
                }
            ]
        })()

        # Enable topic info in options
        conversion_options.format_options['include_topic_info'] = True

        # Generate markdown
        markdown = await audio_converter._create_enhanced_structured_markdown(
            enhanced_result,
            conversion_options
        )

        # Verify topics section is present
        assert "## Topics" in markdown
        assert "### Test Topic" in markdown
        assert "- Dies ist ein Testtext." in markdown
        assert "- Bitte transcribieren." in markdown

    def test_format_timestamp(self, audio_converter):
        """Test timestamp formatting."""
        # Test various timestamp values
        assert audio_converter._format_timestamp(0.0) == "00:00"
        assert audio_converter._format_timestamp(5.29) == "00:05"
        assert audio_converter._format_timestamp(65.5) == "01:05"
        assert audio_converter._format_timestamp(125.8) == "02:05"

    @pytest.mark.asyncio
    async def test_regular_markdown_still_works(
        self,
        audio_converter,
        mock_audio_result,
        conversion_options
    ):
        """Test that regular markdown creation still works correctly."""
        # Generate regular markdown
        markdown = await audio_converter._create_structured_markdown(
            mock_audio_result,
            conversion_options
        )

        # Verify transcript section is now in topic format
        assert "# Main Content" in markdown
        assert "Dies ist ein Testtext, bitte transcribieren." in markdown
        # Should have speaker labels now
        assert "Speaker_00:" in markdown
