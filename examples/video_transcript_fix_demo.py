#!/usr/bin/env python3
"""
Demo script showing the video transcription format fixes.

This script demonstrates the improvements made to video transcription output:
1. Proper timestamp format [seconds] instead of [0] for all topics
2. Consistent speaker labeling (SPEAKER_00, SPEAKER_01)
3. No text repetition
4. Clean topic structure without unwanted headers
5. Proper dialogue format

Run this script to see the before/after comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from morag_video import VideoConverter
from morag_audio import AudioProcessingResult, AudioTranscriptSegment
from morag_audio.services import DiarizationResult, SpeakerSegment, SpeakerInfo
from morag_audio.services import TopicSegmentationResult, TopicSegment


def create_mock_audio_result():
    """Create a mock audio result similar to the German video example."""
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
        SpeakerInfo("SPEAKER_00", 35.0, 8, 4.375, [0.95, 0.92, 0.88, 0.90], 0.0, 80.0),
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
            summary="",
            sentences=[
                "Also die Auswertung des Fukuda-Testes",
                "Der Fukuda-Lauftest hat verschiedene Möglichkeiten"
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
            summary="",
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
        text="Also die Auswertung des Fukuda-Testes. Der Fukuda-Lauftest hat verschiedene Möglichkeiten. Das ist der Grund, warum viele Patienten zum Physiotherapeuten gehen. Genauso ist es, wenn der Patient zubeißt.",
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


def main():
    """Demonstrate the video transcription format fixes."""
    print("🎥 Video Transcription Format Fixes Demo")
    print("=" * 50)
    
    # Create mock data
    audio_result = create_mock_audio_result()
    video_converter = VideoConverter()
    
    # Generate the improved markdown
    print("\n✅ IMPROVED OUTPUT (After Fixes):")
    print("-" * 30)
    
    markdown_lines = video_converter._create_enhanced_audio_markdown(audio_result, 80.0)
    improved_output = "\n".join(markdown_lines)
    
    print(improved_output)
    
    print("\n🔍 KEY IMPROVEMENTS:")
    print("-" * 20)
    
    # Check improvements
    improvements = []
    
    # 1. Timestamp format
    if "[0]" in improved_output and "[45]" in improved_output:
        improvements.append("✅ Proper timestamps: [0] and [45] instead of [0] for all topics")
    
    # 2. Speaker format
    if "SPEAKER_00:" in improved_output and "SPEAKER_01:" in improved_output:
        improvements.append("✅ Consistent speaker format: SPEAKER_00, SPEAKER_01")
    
    # 3. No unwanted headers
    unwanted_headers = ["## Speakers", "## transcript", "## processing details"]
    if not any(header in improved_output for header in unwanted_headers):
        improvements.append("✅ Clean structure: No unwanted headers")
    
    # 4. Meaningful topic titles
    if "Erklärung zum Fukuda-Test" in improved_output and "Physiotherapie und Behandlung" in improved_output:
        improvements.append("✅ Meaningful topic titles: Uses actual topic names when available")
    
    # 5. No text repetition
    lines = [line.strip() for line in improved_output.split('\n') if ': ' in line]
    texts = [line.split(': ', 1)[1] for line in lines]
    unique_texts = set(texts)
    if len(texts) == len(unique_texts):
        improvements.append("✅ No text repetition: Each sentence appears only once")
    
    for improvement in improvements:
        print(improvement)
    
    print(f"\n📊 STATISTICS:")
    print(f"   • Total lines: {len(improved_output.split(chr(10)))}")
    print(f"   • Dialogue lines: {len([l for l in improved_output.split(chr(10)) if ': ' in l])}")
    print(f"   • Topics: {improved_output.count('# ')}")
    print(f"   • Speakers detected: {len(set([l.split(':')[0] for l in improved_output.split(chr(10)) if ': ' in l]))}")
    
    print("\n🎯 SUMMARY:")
    print("The video transcription format has been significantly improved with:")
    print("• Proper timestamp formatting in [seconds] format")
    print("• Consistent SPEAKER_XX labeling")
    print("• Elimination of text repetition")
    print("• Clean topic structure")
    print("• Better speaker-to-text mapping")


if __name__ == "__main__":
    main()
