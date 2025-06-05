#!/usr/bin/env python3
"""
MoRAG Audio Processing Test Script

Usage: python test-audio.py <audio_file>

Examples:
    python test-audio.py my-audio.mp3
    python test-audio.py recording.wav
    python test-audio.py video.mp4  # Extract audio from video
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_audio import AudioProcessor
    from morag_services import ServiceConfig, ContentType
    from morag_core.models import ProcessingConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-audio")
    sys.exit(1)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def print_result(key: str, value: str, indent: int = 0):
    """Print a formatted key-value result."""
    spaces = "  " * indent
    print(f"{spaces}📋 {key}: {value}")


async def test_audio_processing(audio_file: Path) -> bool:
    """Test audio processing functionality."""
    print_header("MoRAG Audio Processing Test")
    
    if not audio_file.exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        return False
    
    print_result("Input File", str(audio_file))
    print_result("File Size", f"{audio_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Initialize configuration
        config = ServiceConfig()
        print_result("Configuration", "✅ Loaded successfully")

        # Initialize audio processor
        processor = AudioProcessor(config)
        print_result("Audio Processor", "✅ Initialized successfully")

        # Create processing configuration
        processing_config = ProcessingConfig(
            max_file_size=100 * 1024 * 1024,  # 100MB
            timeout=300.0,
            extract_metadata=True
        )
        print_result("Processing Config", "✅ Created successfully")
        
        print_section("Processing Audio File")
        print("🔄 Starting audio processing...")
        
        # Process the audio file
        result = await processor.process_file(audio_file, processing_config)
        
        if result.success:
            print("✅ Audio processing completed successfully!")
            
            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Content Type", result.content_type)
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            
            if result.metadata:
                print_section("Metadata")
                for key, value in result.metadata.items():
                    if isinstance(value, (dict, list)):
                        print_result(key, json.dumps(value, indent=2))
                    else:
                        print_result(key, str(value))
            
            if result.content:
                print_section("Content Preview")
                content_preview = result.content[:500] + "..." if len(result.content) > 500 else result.content
                print(f"📄 Content ({len(result.content)} characters):")
                print(content_preview)
            
            if result.summary:
                print_section("Summary")
                print(f"📝 {result.summary}")
            
            # Save results to file
            output_file = audio_file.parent / f"{audio_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'success': result.success,
                    'content_type': result.content_type,
                    'processing_time': result.processing_time,
                    'metadata': result.metadata,
                    'content': result.content,
                    'summary': result.summary,
                    'error': result.error
                }, f, indent=2, ensure_ascii=False)
            
            print_section("Output")
            print_result("Results saved to", str(output_file))
            
            return True
            
        else:
            print("❌ Audio processing failed!")
            print_result("Error", result.error or "Unknown error")
            return False
            
    except Exception as e:
        print(f"❌ Error during audio processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test-audio.py <audio_file>")
        print()
        print("Examples:")
        print("  python test-audio.py my-audio.mp3")
        print("  python test-audio.py recording.wav")
        print("  python test-audio.py video.mp4  # Extract audio from video")
        sys.exit(1)
    
    audio_file = Path(sys.argv[1])
    
    try:
        success = asyncio.run(test_audio_processing(audio_file))
        if success:
            print("\n🎉 Audio processing test completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Audio processing test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
