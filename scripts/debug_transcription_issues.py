#!/usr/bin/env python3
"""Debug script to identify and fix audio/video transcription issues.

This script will:
1. Test timestamp calculation and identify why they show as 0
2. Test text processing to find repetition loops
3. Add extensive logging to trace the issues
4. Test with a sample audio file to reproduce the problems
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog
from morag.core.config import settings
from morag.processors.audio import AudioProcessor, AudioConfig
from morag.processors.video import VideoProcessor, VideoConfig
from morag.converters.audio import AudioConverter
from morag.converters.video import VideoConverter
from morag.converters.config import ConversionOptions, ChunkingStrategy
from morag.services.whisper_service import whisper_service
from morag.services.speaker_diarization import speaker_diarization_service
from morag.services.topic_segmentation import EnhancedTopicSegmentation

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_transcription.log')
    ]
)

logger = structlog.get_logger(__name__)


class TranscriptionDebugger:
    """Debug transcription issues with detailed logging."""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.audio_converter = AudioConverter()
        self.video_converter = VideoConverter()
        self.topic_segmentation = EnhancedTopicSegmentation()
        
    async def debug_timestamp_calculation(self, audio_file: Path):
        """Debug timestamp calculation issues."""
        logger.info("=== DEBUGGING TIMESTAMP CALCULATION ===")
        
        try:
            # Step 1: Test basic transcription
            logger.info("Step 1: Testing basic Whisper transcription")
            config = AudioConfig(model_size="base")  # Use smaller model for faster testing
            
            transcription_result = await whisper_service.transcribe_audio(audio_file, config)
            logger.info("Basic transcription completed", 
                       segments_count=len(transcription_result.segments),
                       duration=transcription_result.duration)
            
            # Log segment details
            for i, segment in enumerate(transcription_result.segments[:5]):  # First 5 segments
                logger.info(f"Segment {i}", 
                           text=segment.text[:50] + "..." if len(segment.text) > 50 else segment.text,
                           start_time=segment.start_time,
                           end_time=segment.end_time)
            
            # Step 2: Test speaker diarization
            logger.info("Step 2: Testing speaker diarization")
            diarization_result = await speaker_diarization_service.diarize_audio(audio_file)
            logger.info("Speaker diarization completed",
                       total_speakers=diarization_result.total_speakers,
                       segments_count=len(diarization_result.segments))
            
            # Log speaker segment details
            for i, segment in enumerate(diarization_result.segments[:5]):  # First 5 segments
                logger.info(f"Speaker segment {i}",
                           speaker_id=segment.speaker_id,
                           start_time=segment.start_time,
                           end_time=segment.end_time,
                           duration=segment.duration)
            
            # Step 3: Test topic segmentation
            logger.info("Step 3: Testing topic segmentation")
            topic_result = await self.topic_segmentation.segment_topics(
                transcription_result.text,
                diarization_result.segments,
                transcription_result.segments
            )
            logger.info("Topic segmentation completed",
                       total_topics=topic_result.total_topics)
            
            # Log topic details with timestamp calculation
            for i, topic in enumerate(topic_result.topics):
                logger.info(f"Topic {i}",
                           title=topic.title,
                           start_time=topic.start_time,
                           end_time=topic.end_time,
                           duration=topic.duration,
                           sentences_count=len(topic.sentences))
                
                # Debug timestamp calculation for this topic
                await self._debug_topic_timestamp_calculation(
                    topic, transcription_result.segments, diarization_result.segments
                )
            
            return {
                'transcription': transcription_result,
                'diarization': diarization_result,
                'topics': topic_result
            }
            
        except Exception as e:
            logger.error("Timestamp debugging failed", error=str(e), exc_info=True)
            raise
    
    async def _debug_topic_timestamp_calculation(self, topic, transcript_segments, speaker_segments):
        """Debug timestamp calculation for a specific topic."""
        logger.info(f"=== DEBUGGING TOPIC TIMESTAMP: {topic.title} ===")
        
        # Check if topic has sentences
        if not topic.sentences:
            logger.warning("Topic has no sentences")
            return
        
        logger.info("Topic sentences", count=len(topic.sentences))
        for i, sentence in enumerate(topic.sentences[:3]):  # First 3 sentences
            logger.info(f"Sentence {i}", text=sentence[:100] + "..." if len(sentence) > 100 else sentence)
        
        # Try to match sentences with transcript segments
        logger.info("Attempting to match sentences with transcript segments")
        matches_found = 0
        
        for sentence in topic.sentences[:3]:  # Debug first 3 sentences
            sentence_clean = sentence.strip().lower()
            logger.info(f"Looking for matches for sentence", sentence=sentence_clean[:50])
            
            for j, segment in enumerate(transcript_segments):
                if hasattr(segment, 'text') and hasattr(segment, 'start_time'):
                    segment_text = segment.text.strip().lower()
                    
                    # Check for matches
                    if (sentence_clean in segment_text or 
                        segment_text in sentence_clean):
                        logger.info(f"MATCH FOUND",
                                   sentence_part=sentence_clean[:30],
                                   segment_text=segment_text[:30],
                                   segment_start=segment.start_time,
                                   segment_end=segment.end_time)
                        matches_found += 1
                        break
            else:
                logger.warning(f"NO MATCH FOUND for sentence", sentence=sentence_clean[:50])
        
        logger.info("Timestamp matching summary", 
                   matches_found=matches_found,
                   total_sentences=len(topic.sentences[:3]))
    
    async def debug_text_repetition(self, audio_file: Path):
        """Debug text repetition issues."""
        logger.info("=== DEBUGGING TEXT REPETITION ===")
        
        try:
            # Test full audio processing pipeline
            logger.info("Testing full audio processing pipeline")
            
            config = AudioConfig(
                model_size="base",
                enable_diarization=True,
                enable_topic_segmentation=True
            )
            
            result = await self.audio_processor.process_audio_file(
                audio_file, 
                config,
                enable_diarization=True,
                enable_topic_segmentation=True
            )
            
            logger.info("Audio processing completed",
                       text_length=len(result.text),
                       segments_count=len(result.segments) if result.segments else 0)
            
            # Check for text repetition patterns
            await self._analyze_text_repetition(result.text)
            
            # Test conversion to markdown
            logger.info("Testing conversion to markdown")
            options = ConversionOptions(
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                include_metadata=True,
                extract_images=False
            )
            
            conversion_result = await self.audio_converter.convert(audio_file, options)
            
            logger.info("Conversion completed",
                       content_length=len(conversion_result.content),
                       success=conversion_result.success)
            
            # Analyze markdown content for repetition
            await self._analyze_text_repetition(conversion_result.content, "markdown")
            
            return {
                'processing_result': result,
                'conversion_result': conversion_result
            }
            
        except Exception as e:
            logger.error("Text repetition debugging failed", error=str(e), exc_info=True)
            raise
    
    async def _analyze_text_repetition(self, text: str, text_type: str = "text"):
        """Analyze text for repetition patterns."""
        logger.info(f"=== ANALYZING {text_type.upper()} REPETITION ===")
        
        if not text:
            logger.warning(f"No {text_type} to analyze")
            return
        
        # Split into lines
        lines = text.split('\n')
        logger.info(f"Total lines in {text_type}", count=len(lines))
        
        # Check for repeated lines
        line_counts = {}
        for line in lines:
            line_clean = line.strip()
            if line_clean:
                line_counts[line_clean] = line_counts.get(line_clean, 0) + 1
        
        # Find repeated lines
        repeated_lines = {line: count for line, count in line_counts.items() if count > 1}
        
        if repeated_lines:
            logger.warning(f"REPEATED LINES FOUND in {text_type}", count=len(repeated_lines))
            for line, count in list(repeated_lines.items())[:5]:  # Show first 5
                logger.warning(f"Repeated {count} times", line=line[:100])
        else:
            logger.info(f"No repeated lines found in {text_type}")
        
        # Check for repetition at the end
        last_lines = lines[-10:] if len(lines) > 10 else lines
        logger.info(f"Last {len(last_lines)} lines of {text_type}")
        for i, line in enumerate(last_lines):
            logger.info(f"Line {len(lines) - len(last_lines) + i}", content=line[:100])
        
        # Check for patterns in the last part
        if len(lines) > 20:
            last_part = '\n'.join(lines[-20:])
            await self._check_repetition_patterns(last_part, f"last 20 lines of {text_type}")

    async def _check_repetition_patterns(self, text: str, description: str):
        """Check for repetition patterns in text."""
        logger.info(f"=== CHECKING REPETITION PATTERNS: {description} ===")

        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < 2:
            logger.info("Not enough sentences to check patterns")
            return

        # Check for consecutive repeated sentences
        consecutive_repeats = 0
        for i in range(1, len(sentences)):
            if sentences[i] == sentences[i-1]:
                consecutive_repeats += 1
                logger.warning(f"Consecutive repeat found",
                             sentence=sentences[i][:100],
                             position=i)

        if consecutive_repeats > 0:
            logger.warning(f"CONSECUTIVE REPETITION DETECTED",
                          count=consecutive_repeats,
                          description=description)
        else:
            logger.info(f"No consecutive repetition found in {description}")


async def main():
    """Main debugging function."""
    logger.info("Starting transcription debugging session")
    
    # Check if we have a test audio file
    test_files = [
        Path("test_audio.wav"),
        Path("test_audio.mp3"),
        Path("sample.wav"),
        Path("sample.mp3"),
        Path("audio.wav"),
        Path("audio.mp3")
    ]
    
    audio_file = None
    for test_file in test_files:
        if test_file.exists():
            audio_file = test_file
            break
    
    if not audio_file:
        logger.error("No test audio file found. Please provide one of: " + 
                    ", ".join(str(f) for f in test_files))
        return
    
    logger.info("Using test audio file", file=str(audio_file))
    
    debugger = TranscriptionDebugger()
    
    try:
        # Debug timestamp issues
        logger.info("=" * 60)
        timestamp_results = await debugger.debug_timestamp_calculation(audio_file)
        
        # Debug text repetition issues
        logger.info("=" * 60)
        repetition_results = await debugger.debug_text_repetition(audio_file)
        
        logger.info("Debugging session completed successfully")
        
    except Exception as e:
        logger.error("Debugging session failed", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
