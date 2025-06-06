#!/usr/bin/env python3
"""
MoRAG Audio Processing Test Script

Supports both processing (immediate results) and ingestion (background + storage) modes.

Usage:
    python test-audio.py <audio_file> [options]

Processing Mode (immediate results):
    python test-audio.py my-audio.mp3
    python test-audio.py recording.wav
    python test-audio.py video.mp4  # Extract audio from video

Ingestion Mode (background processing + storage):
    python test-audio.py my-audio.mp3 --ingest
    python test-audio.py recording.wav --ingest --webhook-url https://my-app.com/webhook
    python test-audio.py audio.mp3 --ingest --metadata '{"category": "meeting", "priority": 1}'

Options:
    --ingest                    Enable ingestion mode (background processing + storage)
    --webhook-url URL          Webhook URL for completion notifications (ingestion mode only)
    --metadata JSON            Additional metadata as JSON string (ingestion mode only)
    --model-size SIZE          Whisper model size: tiny, base, small, medium, large (default: base)
    --enable-diarization       Enable speaker diarization
    --enable-topics            Enable topic segmentation
    --help                     Show this help message
"""

import sys
import asyncio
import json
import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any
import requests

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_audio import AudioProcessor, AudioConfig
    from morag_core.models import ProcessingConfig
    from morag_services import QdrantVectorStorage, GeminiEmbeddingService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-audio")
    print("  pip install -e packages/morag-services")
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


async def test_audio_processing(audio_file: Path, model_size: str = "base",
                               enable_diarization: bool = False, enable_topics: bool = False) -> bool:
    """Test audio processing functionality."""
    print_header("MoRAG Audio Processing Test")

    if not audio_file.exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        return False

    print_result("Input File", str(audio_file))
    print_result("File Size", f"{audio_file.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        # Initialize audio configuration
        config = AudioConfig(
            model_size=model_size,
            device="auto",  # Auto-detect best available device
            enable_diarization=enable_diarization,
            enable_topic_segmentation=enable_topics,
            vad_filter=True,
            word_timestamps=True,
            include_metadata=True
        )
        print_result("Audio Configuration", "✅ Created successfully")
        print_result("Model Size", model_size)
        print_result("Speaker Diarization", "✅ Enabled" if enable_diarization else "❌ Disabled")
        print_result("Topic Segmentation", "✅ Enabled" if enable_topics else "❌ Disabled")

        # Initialize audio processor
        processor = AudioProcessor(config)
        print_result("Audio Processor", "✅ Initialized successfully")

        print_section("Processing Audio File")
        print("🔄 Starting audio processing...")

        # Process the audio file
        result = await processor.process(audio_file)

        if result.success:
            print("✅ Audio processing completed successfully!")

            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            print_result("Transcript Length", f"{len(result.transcript)} characters")
            print_result("Segments Count", f"{len(result.segments)}")

            if result.metadata:
                print_section("Metadata")
                for key, value in result.metadata.items():
                    if isinstance(value, (dict, list)):
                        print_result(key, json.dumps(value, indent=2))
                    else:
                        print_result(key, str(value))

            if result.transcript:
                print_section("Transcript Preview")
                transcript_preview = result.transcript[:500] + "..." if len(result.transcript) > 500 else result.transcript
                print(f"📄 Transcript ({len(result.transcript)} characters):")
                print(transcript_preview)

            if result.segments:
                print_section("Segments Preview (first 3)")
                for i, segment in enumerate(result.segments[:3]):
                    print(f"  Segment {i+1}: [{segment.start:.2f}s - {segment.end:.2f}s]")
                    print(f"    Text: {segment.text[:100]}{'...' if len(segment.text) > 100 else ''}")
                    if segment.speaker:
                        print(f"    Speaker: {segment.speaker}")
                    if segment.confidence:
                        print(f"    Confidence: {segment.confidence:.3f}")

            # Save results to file
            output_file = audio_file.parent / f"{audio_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'processing',
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'transcript': result.transcript,
                    'segments': [
                        {
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text,
                            'speaker': seg.speaker,
                            'confidence': seg.confidence,
                            'topic_id': seg.topic_id,
                            'topic_label': seg.topic_label
                        } for seg in result.segments
                    ],
                    'metadata': result.metadata,
                    'file_path': result.file_path,
                    'error_message': result.error_message
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Results saved to", str(output_file))

            return True

        else:
            print("❌ Audio processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

    except Exception as e:
        print(f"❌ Error during audio processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def store_content_in_vector_db(
    content: str,
    metadata: Dict[str, Any],
    collection_name: str = "morag_vectors"
) -> list:
    """Store processed content in vector database."""
    if not content.strip():
        print("⚠️  Warning: Empty content provided for vector storage")
        return []

    try:
        # Initialize services with environment configuration
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name_env = os.getenv('QDRANT_COLLECTION_NAME', 'morag_vectors')

        vector_storage = QdrantVectorStorage(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            collection_name=collection_name_env
        )

        # Get API key from environment (prefer GEMINI_API_KEY for consistency)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for vector storage")

        embedding_service = GeminiEmbeddingService(api_key=api_key)

        # Connect to vector storage
        await vector_storage.connect()

        # Create document chunks for better retrieval
        chunk_size = 1000  # characters
        chunks = []

        if len(content) <= chunk_size:
            chunks = [content]
        else:
            # Split into overlapping chunks
            overlap = 200
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)

        # Generate embeddings for each chunk
        embeddings = []
        chunk_metadata = []

        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding_result = await embedding_service.generate_embedding_with_result(
                chunk,
                task_type="retrieval_document"
            )

            embeddings.append(embedding_result.embedding)

            # Prepare metadata for this chunk
            chunk_meta = {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "text": chunk,  # Store the actual text for retrieval
                "text_length": len(chunk)
            }
            chunk_metadata.append(chunk_meta)

        # Store vectors in Qdrant
        point_ids = await vector_storage.store_vectors(
            embeddings,
            chunk_metadata,
            collection_name
        )

        print_result("Vector Storage", f"✅ Stored {len(chunks)} chunks with {len(point_ids)} vectors")

        return point_ids

    except Exception as e:
        print(f"❌ Error storing content in vector database: {e}")
        raise


async def test_audio_ingestion(audio_file: Path, webhook_url: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None,
                              model_size: str = "base",
                              enable_diarization: bool = False,
                              enable_topics: bool = False) -> bool:
    """Test audio ingestion functionality using direct processing."""
    print_header("MoRAG Audio Ingestion Test (Direct Processing)")

    if not audio_file.exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        return False

    print_result("Input File", str(audio_file))
    print_result("File Size", f"{audio_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")
    print_result("Model Size", model_size)
    print_result("Speaker Diarization", "✅ Enabled" if enable_diarization else "❌ Disabled")
    print_result("Topic Segmentation", "✅ Enabled" if enable_topics else "❌ Disabled")

    try:
        print_section("Processing Audio")
        print("🔄 Starting audio processing and ingestion...")

        # Initialize audio configuration
        config = AudioConfig(
            model_size=model_size,
            device="auto",  # Auto-detect best available device
            enable_diarization=enable_diarization,
            enable_topic_segmentation=enable_topics,
            vad_filter=True,
            word_timestamps=True,
            include_metadata=True
        )

        # Initialize audio processor
        processor = AudioProcessor(config)

        # Process the audio file
        result = await processor.process(audio_file)

        if not result.success:
            print("❌ Audio processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

        print("✅ Audio processing completed successfully!")
        print_result("Processing Time", f"{result.processing_time:.2f} seconds")
        print_result("Transcript Length", f"{len(result.transcript)} characters")

        # Prepare metadata for vector storage
        vector_metadata = {
            "source_type": "audio",
            "source_path": str(audio_file),
            "processing_time": result.processing_time,
            "model_size": model_size,
            "enable_diarization": enable_diarization,
            "enable_topics": enable_topics,
            "transcript_length": len(result.transcript),
            "segments_count": len(result.segments),
            **(result.metadata or {}),
            **(metadata or {})
        }

        print_section("Storing in Vector Database")
        print("🔄 Storing transcript in vector database...")

        # Store transcript in vector database
        point_ids = await store_content_in_vector_db(
            result.transcript,
            vector_metadata
        )

        print("✅ Audio ingestion completed successfully!")

        print_section("Ingestion Results")
        print_result("Status", "✅ Success")
        print_result("Chunks Processed", str(len(point_ids)))
        print_result("Vector Points Created", str(len(point_ids)))
        print_result("Transcript Length", str(len(result.transcript)))

        # Save ingestion result
        output_file = audio_file.parent / f"{audio_file.stem}_ingest_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'direct_ingestion',
                'success': True,
                'processing_time': result.processing_time,
                'chunks_processed': len(point_ids),
                'vector_point_ids': point_ids,
                'transcript_length': len(result.transcript),
                'segments_count': len(result.segments),
                'model_size': model_size,
                'enable_diarization': enable_diarization,
                'enable_topics': enable_topics,
                'webhook_url': webhook_url,
                'metadata': vector_metadata,
                'file_path': str(audio_file)
            }, f, indent=2, ensure_ascii=False)

        print_section("Output")
        print_result("Ingestion result saved to", str(output_file))
        print_result("Vector Points", f"{len(point_ids)} chunks stored in Qdrant")

        return True

    except Exception as e:
        print(f"❌ Error during audio ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Audio Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-audio.py my-audio.mp3
    python test-audio.py recording.wav --model-size large --enable-diarization
    python test-audio.py video.mp4 --enable-topics

  Ingestion Mode (background processing + storage):
    python test-audio.py my-audio.mp3 --ingest
    python test-audio.py recording.wav --ingest --webhook-url https://my-app.com/webhook
    python test-audio.py audio.mp3 --ingest --metadata '{"category": "meeting"}'
        """
    )

    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--model-size', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size (default: base)')
    parser.add_argument('--enable-diarization', action='store_true',
                       help='Enable speaker diarization')
    parser.add_argument('--enable-topics', action='store_true',
                       help='Enable topic segmentation')

    args = parser.parse_args()

    audio_file = Path(args.audio_file)

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    try:
        if args.ingest:
            # Ingestion mode
            success = asyncio.run(test_audio_ingestion(
                audio_file,
                webhook_url=args.webhook_url,
                metadata=metadata,
                model_size=args.model_size,
                enable_diarization=args.enable_diarization,
                enable_topics=args.enable_topics
            ))
            if success:
                print("\n🎉 Audio ingestion test completed successfully!")
                print("💡 Use the task ID to monitor progress and retrieve results.")
                sys.exit(0)
            else:
                print("\n💥 Audio ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
            success = asyncio.run(test_audio_processing(
                audio_file,
                model_size=args.model_size,
                enable_diarization=args.enable_diarization,
                enable_topics=args.enable_topics
            ))
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
