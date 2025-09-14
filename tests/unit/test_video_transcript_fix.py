"""Test video transcription format fixes."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from morag_video import VideoConverter
from morag_core.interfaces.converter import ConversionOptions
from morag_audio import AudioProcessingResult, AudioTranscriptSegment
from morag_audio.services import DiarizationResult, SpeakerSegment, SpeakerInfo
from morag_audio.services import TopicSegmentationResult, TopicSegment


class TestVideoTranscriptFix:
    """Test video transcription format fixes."""

    @pytest.fixture
    def video_converter(self):
        """Create a video converter instance."""
        return VideoConverter()

    @pytest.fixture
    def mock_audio_result(self):
        """Create a mock audio result with proper structure."""
        # Mock transcript segments with timing
        segments = [
            AudioTranscriptSegment(
                text="Also die Auswertung des Fukuda-Testes",
                start_time=0.0,
                end_time=3.0,
                confidence=0.95,
                speaker_id=None,
                language="de"
            ),
            AudioTranscriptSegment(
                text="Der Fukuda-Lauftest hat verschiedene Möglichkeiten",
                start_time=3.5,
                end_time=7.0,
                confidence=0.92,
                speaker_id=None,
                language="de"
            ),
            AudioTranscriptSegment(
                text="Zunächst testen wir immer mit geöffnetem Mund",
                start_time=7.5,
                end_time=11.0,
                confidence=0.88,
                speaker_id=None,
                language="de"
            ),
            AudioTranscriptSegment(
                text="Das ist der Grund, warum viele Patienten zum Physiotherapeuten gehen",
                start_time=45.0,
                end_time=50.0,
                confidence=0.90,
                speaker_id=None,
                language="de"
            ),
            AudioTranscriptSegment(
                text="Genauso ist es, wenn der Patient zubeißt",
                start_time=50.5,
                end_time=54.0,
                confidence=0.87,
                speaker_id=None,
                language="de"
            )
        ]

        # Mock speaker diarization with proper speaker IDs
        speakers = [
            SpeakerInfo("SPEAKER_00", 35.0, 8, 4.375, [0.95, 0.92, 0.88, 0.90, 0.87, 0.85, 0.89, 0.91], 0.0, 80.0),
            SpeakerInfo("SPEAKER_01", 15.0, 3, 5.0, [0.88, 0.92, 0.85], 45.0, 60.0)
        ]

        speaker_segments = [
            SpeakerSegment("SPEAKER_00", 0.0, 40.0, 40.0, 0.95),
            SpeakerSegment("SPEAKER_01", 40.5, 55.0, 14.5, 0.92),
            SpeakerSegment("SPEAKER_00", 55.5, 80.0, 24.5, 0.88)
        ]

        diarization_result = DiarizationResult(
            speakers=speakers,
            segments=speaker_segments,
            total_speakers=2,
            total_duration=80.0,
            speaker_overlap_time=0.0,
            processing_time=2.5,
            model_used="pyannote/speaker-diarization-3.1",
            confidence_threshold=0.5
        )

        # Mock topic segmentation with proper timestamps
        topics = [
            TopicSegment(
                topic_id="topic_1",
                title="Erklärung zum Fukuda-Test",
                summary="",  # No summary as requested
                sentences=[
                    "Also die Auswertung des Fukuda-Testes",
                    "Der Fukuda-Lauftest hat verschiedene Möglichkeiten",
                    "Zunächst testen wir immer mit geöffnetem Mund"
                ],
                start_time=0.0,
                end_time=40.0,
                duration=40.0,
                confidence=0.9,
                keywords=["fukuda", "test", "auswertung"],
                speaker_distribution={"SPEAKER_00": 100.0}
            ),
            TopicSegment(
                topic_id="topic_2",
                title="Physiotherapie und Behandlung",
                summary="",  # No summary as requested
                sentences=[
                    "Das ist der Grund, warum viele Patienten zum Physiotherapeuten gehen",
                    "Genauso ist es, wenn der Patient zubeißt"
                ],
                start_time=45.0,
                end_time=80.0,
                duration=35.0,
                confidence=0.85,
                keywords=["physiotherapeut", "patient", "behandlung"],
                speaker_distribution={"SPEAKER_01": 60.0, "SPEAKER_00": 40.0}
            )
        ]

        topic_result = TopicSegmentationResult(
            topics=topics,
            total_topics=2,
            processing_time=1.2,
            model_used="all-MiniLM-L6-v2",
            similarity_threshold=0.7,
            segmentation_method="semantic_embedding"
        )

        return AudioProcessingResult(
            text="Also die Auswertung des Fukuda-Testes. Der Fukuda-Lauftest hat verschiedene Möglichkeiten. Zunächst testen wir immer mit geöffnetem Mund. Das ist der Grund, warum viele Patienten zum Physiotherapeuten gehen. Genauso ist es, wenn der Patient zubeißt.",
            language="de",
            confidence=0.90,
            duration=80.0,
            segments=segments,
            metadata={
                'filename': '13_Erklärung_zum_Fukuda-Test.mp4',
                'model_used': 'large-v3',
                'processor_used': 'Enhanced Audio Processor'
            },
            processing_time=3.5,
            model_used='large-v3',
            speaker_diarization=diarization_result,
            topic_segmentation=topic_result
        )

    def test_topic_timestamp_format(self, video_converter, mock_audio_result):
        """Test that topic timestamps are in correct [seconds] format."""
        markdown_lines = video_converter._create_enhanced_audio_markdown(
            mock_audio_result, 80.0
        )

        markdown_content = "\n".join(markdown_lines)

        # Check for proper timestamp format [0] and [45] - using actual topic titles
        assert "# Erklärung zum Fukuda-Test [0]" in markdown_content
        assert "# Physiotherapie und Behandlung [45]" in markdown_content

        # Should not have [0] for all topics
        assert markdown_content.count("[0]") == 1

    def test_speaker_labeling_consistency(self, video_converter, mock_audio_result):
        """Test that speaker labels are consistent and properly formatted."""
        markdown_lines = video_converter._create_enhanced_audio_markdown(
            mock_audio_result, 80.0
        )
        
        markdown_content = "\n".join(markdown_lines)
        
        # Check for proper speaker format
        assert "SPEAKER_00:" in markdown_content
        assert "SPEAKER_01:" in markdown_content
        
        # Should not have generic Speaker_00 format
        assert "Speaker_00:" not in markdown_content

    def test_no_text_repetition(self, video_converter, mock_audio_result):
        """Test that there's no repetitive text in the output."""
        markdown_lines = video_converter._create_enhanced_audio_markdown(
            mock_audio_result, 80.0
        )
        
        markdown_content = "\n".join(markdown_lines)
        
        # Check that sentences don't repeat
        lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
        
        # Count occurrences of each line
        line_counts = {}
        for line in lines:
            if ": " in line:  # Only check dialogue lines
                text_part = line.split(": ", 1)[1]
                line_counts[text_part] = line_counts.get(text_part, 0) + 1
        
        # No text should appear more than once
        repeated_texts = [text for text, count in line_counts.items() if count > 1]
        assert len(repeated_texts) == 0, f"Found repeated texts: {repeated_texts}"

    def test_proper_topic_structure(self, video_converter, mock_audio_result):
        """Test that topics have proper structure without unwanted headers."""
        markdown_lines = video_converter._create_enhanced_audio_markdown(
            mock_audio_result, 80.0
        )

        markdown_content = "\n".join(markdown_lines)

        # Should not have these unwanted headers
        assert "## Speakers" not in markdown_content
        assert "## transcript" not in markdown_content
        assert "## processing details" not in markdown_content

        # Should have proper topic headers - using actual topic titles
        assert "# Erklärung zum Fukuda-Test" in markdown_content
        assert "# Physiotherapie und Behandlung" in markdown_content

    def test_dialogue_format(self, video_converter, mock_audio_result):
        """Test that dialogue follows the correct format."""
        markdown_lines = video_converter._create_enhanced_audio_markdown(
            mock_audio_result, 80.0
        )
        
        markdown_content = "\n".join(markdown_lines)
        
        # Check for proper dialogue format
        dialogue_lines = [line for line in markdown_content.split('\n') if ': ' in line]
        
        for line in dialogue_lines:
            # Each dialogue line should have format "SPEAKER_XX: text"
            assert line.startswith('SPEAKER_'), f"Invalid dialogue format: {line}"
            assert ': ' in line, f"Missing colon separator: {line}"
            
            # Text part should not be empty
            text_part = line.split(': ', 1)[1].strip()
            assert len(text_part) > 0, f"Empty text in dialogue: {line}"
