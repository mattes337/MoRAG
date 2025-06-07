#!/usr/bin/env python3
"""
Demo script for enhanced audio processing with speaker diarization and topic segmentation.

This script demonstrates the enhanced audio processing capabilities of MoRAG,
including speaker diarization and topic segmentation features.
"""

import asyncio
import sys
import time
from pathlib import Path
import argparse
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_audio import AudioProcessor, AudioConfig
from morag_audio.services import speaker_diarization_service
from morag_audio.services import topic_segmentation_service
from morag_audio import AudioConverter
from morag_core.interfaces.converter import ConversionOptions
from morag_core.config import settings


def print_banner():
    """Print demo banner."""
    print("=" * 80)
    print("🎵 MoRAG Enhanced Audio Processing Demo")
    print("   Speaker Diarization + Topic Segmentation")
    print("=" * 80)
    print()


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 60}")
    print(f"📋 {title}")
    print('─' * 60)


async def demo_basic_audio_processing(audio_file: Path):
    """Demonstrate basic audio processing."""
    print_section("Basic Audio Processing")
    
    config = AudioConfig(
        model_size="base",  # Use base model for better quality
        device="cpu",
        compute_type="int8"
    )
    
    processor = AudioProcessor(config)
    
    print(f"🎯 Processing: {audio_file.name}")
    print(f"📁 File size: {audio_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    start_time = time.time()
    result = await processor.process_audio_file(
        audio_file,
        enable_diarization=False,
        enable_topic_segmentation=False
    )
    processing_time = time.time() - start_time
    
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"🗣️  Language detected: {result.language}")
    print(f"⏰ Duration: {result.duration:.2f} seconds")
    print(f"📊 Confidence: {result.confidence:.2f}")
    print(f"📝 Text length: {len(result.text)} characters")
    print(f"🔢 Segments: {len(result.segments)}")
    
    # Show first 200 characters of transcript
    preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
    print(f"📄 Transcript preview: {preview}")
    
    return result


async def demo_speaker_diarization(audio_file: Path):
    """Demonstrate speaker diarization."""
    print_section("Speaker Diarization")
    
    print(f"🎯 Analyzing speakers in: {audio_file.name}")
    
    start_time = time.time()
    result = await speaker_diarization_service.diarize_audio(audio_file)
    processing_time = time.time() - start_time
    
    print(f"⏱️  Diarization time: {processing_time:.2f} seconds")
    print(f"👥 Speakers detected: {result.total_speakers}")
    print(f"⏰ Total duration: {result.total_duration:.2f} seconds")
    print(f"🔄 Speaker overlap time: {result.speaker_overlap_time:.2f} seconds")
    print(f"🤖 Model used: {result.model_used}")
    
    print("\n👥 Speaker Details:")
    for i, speaker in enumerate(result.speakers, 1):
        speaking_time = speaker.total_speaking_time
        percentage = (speaking_time / result.total_duration) * 100 if result.total_duration > 0 else 0
        
        print(f"  Speaker {i} ({speaker.speaker_id}):")
        print(f"    🕐 Speaking time: {speaking_time:.2f}s ({percentage:.1f}%)")
        print(f"    📊 Segments: {speaker.segment_count}")
        print(f"    ⏰ Avg segment: {speaker.average_segment_duration:.2f}s")
        print(f"    🎬 First/Last: {speaker.first_appearance:.1f}s - {speaker.last_appearance:.1f}s")
    
    print(f"\n🎬 Speaker Timeline (first 10 segments):")
    for i, segment in enumerate(result.segments[:10]):
        print(f"  {segment.start_time:6.1f}s - {segment.end_time:6.1f}s: {segment.speaker_id}")
    
    if len(result.segments) > 10:
        print(f"  ... and {len(result.segments) - 10} more segments")
    
    return result


async def demo_topic_segmentation(text: str):
    """Demonstrate topic segmentation."""
    print_section("Topic Segmentation")
    
    print(f"🎯 Analyzing topics in text ({len(text)} characters)")
    
    start_time = time.time()
    result = await topic_segmentation_service.segment_topics(text)
    processing_time = time.time() - start_time
    
    print(f"⏱️  Segmentation time: {processing_time:.2f} seconds")
    print(f"📚 Topics detected: {result.total_topics}")
    print(f"🤖 Model used: {result.model_used}")
    print(f"🎯 Method: {result.segmentation_method}")
    print(f"📊 Similarity threshold: {result.similarity_threshold}")
    
    print("\n📚 Topic Details:")
    for i, topic in enumerate(result.topics, 1):
        print(f"  Topic {i}: {topic.title}")
        print(f"    📝 Summary: {topic.summary[:100]}{'...' if len(topic.summary) > 100 else ''}")
        print(f"    📄 Sentences: {len(topic.sentences)}")
        print(f"    📊 Confidence: {topic.confidence:.2f}")
        
        if topic.keywords:
            keywords = ", ".join(topic.keywords[:5])
            print(f"    🏷️  Keywords: {keywords}")
        
        if topic.start_time is not None:
            print(f"    ⏰ Timing: {topic.start_time:.1f}s - {topic.end_time:.1f}s")
        
        # Show first sentence of topic
        if topic.sentences:
            first_sentence = topic.sentences[0][:100] + "..." if len(topic.sentences[0]) > 100 else topic.sentences[0]
            print(f"    📖 First sentence: {first_sentence}")
        print()
    
    return result


async def demo_enhanced_processing(audio_file: Path):
    """Demonstrate complete enhanced processing."""
    print_section("Enhanced Audio Processing (Complete Pipeline)")
    
    config = AudioConfig(
        model_size="base",
        device="cpu",
        compute_type="int8"
    )
    
    processor = AudioProcessor(config)
    
    print(f"🎯 Enhanced processing: {audio_file.name}")
    print("🔧 Features enabled: Speaker Diarization + Topic Segmentation")
    
    start_time = time.time()
    result = await processor.process_audio_file(
        audio_file,
        enable_diarization=True,
        enable_topic_segmentation=True
    )
    processing_time = time.time() - start_time
    
    print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
    print(f"🗣️  Language: {result.language}")
    print(f"⏰ Duration: {result.duration:.2f} seconds")
    print(f"📊 Confidence: {result.confidence:.2f}")
    
    # Speaker diarization results
    if result.speaker_diarization:
        diarization = result.speaker_diarization
        print(f"👥 Speakers: {diarization.total_speakers}")
        print(f"🎬 Speaker segments: {len(diarization.segments)}")
    else:
        print("👥 Speaker diarization: Not available")
    
    # Topic segmentation results
    if result.topic_segmentation:
        segmentation = result.topic_segmentation
        print(f"📚 Topics: {segmentation.total_topics}")
        print(f"🎯 Segmentation method: {segmentation.segmentation_method}")
    else:
        print("📚 Topic segmentation: Not available")
    
    return result


async def demo_markdown_conversion(audio_file: Path):
    """Demonstrate markdown conversion with enhanced features."""
    print_section("Enhanced Markdown Conversion")
    
    converter = AudioConverter()
    
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
    
    print(f"🎯 Converting to markdown: {audio_file.name}")
    
    start_time = time.time()
    result = await converter.convert(audio_file, options)
    processing_time = time.time() - start_time
    
    print(f"⏱️  Conversion time: {processing_time:.2f} seconds")
    print(f"✅ Success: {result.success}")
    print(f"📄 Content length: {len(result.content)} characters")
    print(f"📊 Quality score: {result.quality_score.overall_score:.2f}")
    
    if result.success:
        # Save markdown to file
        output_file = audio_file.parent / f"{audio_file.stem}_enhanced.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.content)
        
        print(f"💾 Markdown saved to: {output_file}")
        
        # Show preview
        lines = result.content.split('\n')
        preview_lines = lines[:20]
        print("\n📄 Markdown Preview (first 20 lines):")
        print("─" * 40)
        for line in preview_lines:
            print(line)
        if len(lines) > 20:
            print(f"... and {len(lines) - 20} more lines")
        print("─" * 40)
    
    return result


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Enhanced Audio Processing Demo")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--basic-only", action="store_true", help="Run only basic processing")
    parser.add_argument("--no-markdown", action="store_true", help="Skip markdown conversion")
    parser.add_argument("--output-json", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    audio_file = Path(args.audio_file)
    
    if not audio_file.exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        return 1
    
    print_banner()
    print(f"🎵 Audio file: {audio_file}")
    print(f"📁 File size: {audio_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    results = {}
    
    try:
        # Basic processing
        basic_result = await demo_basic_audio_processing(audio_file)
        results['basic'] = {
            'text': basic_result.text,
            'language': basic_result.language,
            'duration': basic_result.duration,
            'confidence': basic_result.confidence,
            'segments_count': len(basic_result.segments)
        }
        
        if not args.basic_only:
            # Speaker diarization
            speaker_result = await demo_speaker_diarization(audio_file)
            results['speakers'] = {
                'total_speakers': speaker_result.total_speakers,
                'total_duration': speaker_result.total_duration,
                'model_used': speaker_result.model_used,
                'speakers': [
                    {
                        'id': s.speaker_id,
                        'speaking_time': s.total_speaking_time,
                        'segments': s.segment_count
                    }
                    for s in speaker_result.speakers
                ]
            }
            
            # Topic segmentation
            if basic_result.text:
                topic_result = await demo_topic_segmentation(basic_result.text)
                results['topics'] = {
                    'total_topics': topic_result.total_topics,
                    'method': topic_result.segmentation_method,
                    'model_used': topic_result.model_used,
                    'topics': [
                        {
                            'title': t.title,
                            'summary': t.summary,
                            'sentences_count': len(t.sentences),
                            'keywords': t.keywords
                        }
                        for t in topic_result.topics
                    ]
                }
            
            # Enhanced processing
            enhanced_result = await demo_enhanced_processing(audio_file)
            
            # Markdown conversion
            if not args.no_markdown:
                markdown_result = await demo_markdown_conversion(audio_file)
                results['markdown'] = {
                    'success': markdown_result.success,
                    'content_length': len(markdown_result.content),
                    'quality_score': markdown_result.quality_score.overall_score
                }
        
        # Save results to JSON if requested
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output_json}")
        
        print_section("Demo Complete")
        print("✅ All demonstrations completed successfully!")
        print("🎉 Enhanced audio processing features are working correctly.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
