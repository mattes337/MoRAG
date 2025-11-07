#!/usr/bin/env python3
"""Test script to validate audio/video transcription fixes.

This script tests:
1. Timestamp calculation fixes
2. Text repetition prevention
3. Enhanced logging and debugging
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import structlog
from morag_core.config import settings
from morag_audio import AudioProcessor, AudioConfig
from morag_video import VideoProcessor, VideoConfig
from morag_audio import AudioConverter
from morag_video import VideoConverter
from morag_core.models import ConversionOptions, ChunkingStrategy
from morag_audio.services import EnhancedTopicSegmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = structlog.get_logger(__name__)


class TranscriptionFixValidator:
    """Validate transcription fixes."""

    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.audio_converter = AudioConverter()
        self.video_converter = VideoConverter()
        self.topic_segmentation = EnhancedTopicSegmentation()

    async def test_timestamp_fixes(self, audio_file: Path):
        """Test timestamp calculation fixes."""
        logger.info("🕐 Testing timestamp calculation fixes")

        try:
            # Test with enhanced audio processing
            config = AudioConfig(
                model_size="base",  # Use smaller model for faster testing
                enable_diarization=True,
                enable_topic_segmentation=True
            )

            result = await self.audio_processor.process_audio_file(
                audio_file,
                config,
                enable_diarization=True,
                enable_topic_segmentation=True
            )

            # Check if we have topic segmentation results
            if result.topic_segmentation and result.topic_segmentation.topics:
                logger.info("✅ Topic segmentation completed",
                           topics_count=len(result.topic_segmentation.topics))

                # Check timestamps for each topic
                timestamp_issues = 0
                for i, topic in enumerate(result.topic_segmentation.topics):
                    if topic.start_time is None or topic.start_time == 0:
                        timestamp_issues += 1
                        logger.warning(f"⚠️  Topic {i+1} has timestamp issue",
                                     topic_id=topic.topic_id,
                                     start_time=topic.start_time,
                                     end_time=topic.end_time)
                    else:
                        logger.info(f"✅ Topic {i+1} has valid timestamp",
                                   topic_id=topic.topic_id,
                                   start_time=topic.start_time,
                                   end_time=topic.end_time)

                if timestamp_issues == 0:
                    logger.info("🎉 All topics have valid timestamps!")
                    return True
                else:
                    logger.warning(f"⚠️  {timestamp_issues} topics have timestamp issues")
                    return False
            else:
                logger.warning("❌ No topic segmentation results found")
                return False

        except Exception as e:
            logger.error("❌ Timestamp test failed", error=str(e), exc_info=True)
            return False

    async def test_repetition_fixes(self, audio_file: Path):
        """Test text repetition prevention."""
        logger.info("🔄 Testing text repetition prevention")

        try:
            # Test audio conversion
            options = ConversionOptions(
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                include_metadata=True,
                extract_images=False
            )

            conversion_result = await self.audio_converter.convert(audio_file, options)

            if not conversion_result.success:
                logger.error("❌ Audio conversion failed",
                           error=conversion_result.error_message)
                return False

            # Analyze content for repetition
            content = conversion_result.content
            lines = content.split('\n')

            # Check for consecutive repeated lines
            consecutive_repeats = 0
            last_line = None

            for line in lines:
                line_clean = line.strip()
                if line_clean and line_clean == last_line:
                    consecutive_repeats += 1
                    if consecutive_repeats > 2:  # Allow some repetition but not excessive
                        logger.warning("⚠️  Excessive repetition detected",
                                     line=line_clean[:50],
                                     consecutive_count=consecutive_repeats)
                else:
                    consecutive_repeats = 0
                last_line = line_clean

            # Check for patterns at the end
            if len(lines) > 10:
                last_10_lines = [line.strip() for line in lines[-10:] if line.strip()]
                unique_last_lines = set(last_10_lines)

                if len(unique_last_lines) < len(last_10_lines) / 2:
                    logger.warning("⚠️  Repetitive pattern detected at end",
                                 total_lines=len(last_10_lines),
                                 unique_lines=len(unique_last_lines))
                    return False

            logger.info("✅ No excessive repetition detected")
            return True

        except Exception as e:
            logger.error("❌ Repetition test failed", error=str(e), exc_info=True)
            return False

    async def test_video_processing(self, video_file: Path):
        """Test video processing with fixes."""
        logger.info("🎬 Testing video processing fixes")

        try:
            # Test video conversion
            options = ConversionOptions(
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                include_metadata=True,
                extract_images=False,
                format_options={
                    'include_audio': True,
                    'enable_enhanced_audio': True,
                    'enable_speaker_diarization': True,
                    'enable_topic_segmentation': True
                }
            )

            conversion_result = await self.video_converter.convert(video_file, options)

            if not conversion_result.success:
                logger.error("❌ Video conversion failed",
                           error=conversion_result.error_message)
                return False

            # Check for timestamp format in content
            content = conversion_result.content
            timestamp_pattern_found = False

            for line in content.split('\n'):
                if '[' in line and ']' in line and line.startswith('#'):
                    # Found a topic header with timestamp
                    timestamp_pattern_found = True
                    logger.info("✅ Found timestamp pattern", line=line.strip())
                    break

            if not timestamp_pattern_found:
                logger.warning("⚠️  No timestamp patterns found in video output")
                return False

            logger.info("✅ Video processing completed successfully")
            return True

        except Exception as e:
            logger.error("❌ Video test failed", error=str(e), exc_info=True)
            return False

    def analyze_content_quality(self, content: str) -> Dict[str, Any]:
        """Analyze the quality of generated content."""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        # Count different types of content
        topic_headers = sum(1 for line in lines if line.startswith('#') and '[' in line)
        speaker_lines = sum(1 for line in lines if ':' in line and any(speaker in line for speaker in ['Speaker_', 'SPEAKER_']))

        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'topic_headers': topic_headers,
            'speaker_lines': speaker_lines,
            'avg_line_length': sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
        }


async def main():
    """Main test function."""
    logger.info("🧪 Starting transcription fixes validation")

    # Look for test files
    test_files = {
        'audio': [
            Path("test_audio.wav"),
            Path("test_audio.mp3"),
            Path("sample.wav"),
            Path("sample.mp3"),
            Path("audio.wav"),
            Path("audio.mp3")
        ],
        'video': [
            Path("test_video.mp4"),
            Path("test_video.avi"),
            Path("sample.mp4"),
            Path("sample.avi"),
            Path("video.mp4"),
            Path("video.avi")
        ]
    }

    audio_file = None
    video_file = None

    for test_file in test_files['audio']:
        if test_file.exists():
            audio_file = test_file
            break

    for test_file in test_files['video']:
        if test_file.exists():
            video_file = test_file
            break

    if not audio_file:
        logger.warning("⚠️  No audio test file found. Audio tests will be skipped.")

    if not video_file:
        logger.warning("⚠️  No video test file found. Video tests will be skipped.")

    if not audio_file and not video_file:
        logger.error("❌ No test files found. Please provide test audio or video files.")
        return False

    validator = TranscriptionFixValidator()
    test_results = []

    # Test audio processing
    if audio_file:
        logger.info(f"🎵 Testing with audio file: {audio_file}")

        timestamp_result = await validator.test_timestamp_fixes(audio_file)
        test_results.append(('Timestamp Fixes', timestamp_result))

        repetition_result = await validator.test_repetition_fixes(audio_file)
        test_results.append(('Repetition Prevention', repetition_result))

    # Test video processing
    if video_file:
        logger.info(f"🎬 Testing with video file: {video_file}")

        video_result = await validator.test_video_processing(video_file)
        test_results.append(('Video Processing', video_result))

    # Summary
    logger.info("=" * 60)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed_tests += 1

    logger.info(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Transcription fixes are working correctly.")
        return True
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
