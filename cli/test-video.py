#!/usr/bin/env python3
"""
MoRAG Video Processing Test Script

Supports both processing (immediate results) and ingestion (background + storage) modes.

Usage:
    python test-video.py <video_file> [options]

Processing Mode (immediate results):
    python test-video.py my-video.mp4
    python test-video.py recording.avi --thumbnails
    python test-video.py presentation.mov --enable-ocr

Ingestion Mode (background processing + storage):
    python test-video.py my-video.mp4 --ingest
    python test-video.py recording.avi --ingest --metadata '{"type": "meeting"}'
    python test-video.py presentation.mov --ingest --webhook-url https://my-app.com/webhook

Options:
    --ingest                    Enable ingestion mode (background processing + storage)
    --webhook-url URL          Webhook URL for completion notifications (ingestion mode only)
    --metadata JSON            Additional metadata as JSON string (ingestion mode only)
    --thumbnails               Generate thumbnails (opt-in, default: false)
    --thumbnail-count N        Number of thumbnails to generate (default: 3)
    --enable-ocr               Enable OCR on video frames
    --help                     Show this help message
"""

import sys
import os
import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import requests

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path, override=True)

try:
    from morag_video import VideoProcessor, VideoConfig
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-video")
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
    print(f"{spaces}[INFO] {key}: {value}")


async def test_video_processing(video_file: Path, generate_thumbnails: bool = False,
                               thumbnail_count: int = 3, enable_ocr: bool = False, language: Optional[str] = None) -> bool:
    """Test video processing functionality."""
    print_header("MoRAG Video Processing Test")

    if not video_file.exists():
        print(f"[FAIL] Error: Video file not found: {video_file}")
        return False

    print_result("Input File", str(video_file))
    print_result("File Size", f"{video_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", video_file.suffix.lower())

    try:
        # Initialize video configuration with FULL FEATURES ENABLED
        config = VideoConfig(
            extract_audio=True,
            generate_thumbnails=generate_thumbnails,
            thumbnail_count=thumbnail_count,
            extract_keyframes=False,  # Disable for faster processing
            enable_enhanced_audio=True,
            enable_speaker_diarization=True,  # ENABLE for proper speaker info
            enable_topic_segmentation=True,  # ENABLE for topic headers
            audio_model_size="base",  # Use base model for faster processing
            enable_ocr=enable_ocr,
            language=language  # Pass language for consistent processing
        )
        print_result("Video Configuration", "[OK] Created successfully")
        print_result("Generate Thumbnails", "[OK] Enabled" if generate_thumbnails else "[FAIL] Disabled")
        print_result("Thumbnail Count", str(thumbnail_count) if generate_thumbnails else "N/A")
        print_result("OCR Enabled", "[OK] Enabled" if enable_ocr else "[FAIL] Disabled")
        print_result("Speaker Diarization", "[OK] Enabled")
        print_result("Topic Segmentation", "[OK] Enabled")

        # Use VideoService instead of VideoProcessor for proper markdown output
        from morag_video import VideoService
        service = VideoService(config=config, output_dir=video_file.parent)
        print_result("Video Service", "[OK] Initialized successfully")

        print_section("Processing Video File")
        print("[PROCESSING] Starting video processing...")
        print("   This may take a while for large videos...")

        # Process the video file using VideoService for proper markdown output
        service_result = await service.process_file(
            file_path=video_file,
            save_output=True,  # Save both JSON and markdown files
            output_format="markdown"
        )

        print("[OK] Video processing completed successfully!")

        print_section("Processing Results")
        print_result("Status", "[OK] Success")
        print_result("Processing Time", f"{service_result.get('processing_time', 0):.2f} seconds")

        # Extract metadata from service result
        result_data = service_result.get('result', {})
        metadata = result_data.get('metadata', {})

        print_section("Video Metadata")
        print_result("Duration", f"{metadata.get('duration', 0):.2f} seconds")
        print_result("Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}")
        print_result("FPS", f"{metadata.get('fps', 0):.2f}")
        print_result("Codec", metadata.get('codec', 'Unknown'))
        print_result("Format", metadata.get('format', 'Unknown'))
        print_result("Has Audio", "[OK] Yes" if metadata.get('has_audio', False) else "[FAIL] No")
        print_result("File Size", f"{metadata.get('file_size', 0) / 1024 / 1024:.2f} MB")

        # Check for audio processing results
        audio_result = result_data.get('audio_processing_result')
        if audio_result:
            print_section("Audio Processing")
            print_result("Audio Extracted", "[OK] Yes")
            print_result("Transcript Length", f"{len(audio_result.get('transcript', ''))}")
            print_result("Segments Count", f"{len(audio_result.get('segments', []))}")

            # Check for speaker diarization
            if audio_result.get('metadata', {}).get('has_speaker_info'):
                print_result("Speaker Diarization", "[OK] Enabled")
                speakers = audio_result.get('metadata', {}).get('speakers', [])
                print_result("Speakers Detected", f"{len(speakers)}")

            # Check for topic segmentation
            if audio_result.get('metadata', {}).get('has_topic_info'):
                print_result("Topic Segmentation", "[OK] Enabled")
                topics = audio_result.get('metadata', {}).get('topics', [])
                print_result("Topics Detected", f"{len(topics)}")

            transcript = audio_result.get('transcript', '')
            if transcript:
                print_section("Transcript Preview")
                transcript_preview = transcript[:500] + "..." if len(transcript) > 500 else transcript
                print(f"📄 Transcript ({len(transcript)} characters):")
                print(transcript_preview)

        # Check for thumbnails
        thumbnails = result_data.get('thumbnails', [])
        if thumbnails:
            print_section("Thumbnails")
            print_result("Thumbnails Generated", f"{len(thumbnails)}")
            for i, thumb in enumerate(thumbnails):
                print_result(f"Thumbnail {i+1}", str(thumb))

        # Check for keyframes
        keyframes = result_data.get('keyframes', [])
        if keyframes:
            print_section("Keyframes")
            print_result("Keyframes Generated", f"{len(keyframes)}")
            for i, frame in enumerate(keyframes):
                print_result(f"Keyframe {i+1}", str(frame))

        # Check for OCR results
        ocr_results = result_data.get('ocr_results')
        if ocr_results:
            print_section("OCR Results")
            print_result("OCR Performed", "[OK] Yes")
            print_result("OCR Data", json.dumps(ocr_results, indent=2))

        # Save comprehensive results to JSON file with all metadata
        output_file = video_file.parent / f"{video_file.stem}_test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'processing',
                'processing_time': service_result.get('processing_time', 0),
                'generate_thumbnails': generate_thumbnails,
                'thumbnail_count': thumbnail_count,
                'enable_ocr': enable_ocr,
                'enable_speaker_diarization': True,
                'enable_topic_segmentation': True,
                'language': language,
                'metadata': metadata,
                'audio_processing_result': audio_result,
                'thumbnails': thumbnails,
                'keyframes': keyframes,
                'ocr_results': ocr_results,
                'output_files': service_result.get('output_files', {})
            }, f, indent=2, ensure_ascii=False)

        # Create intermediate markdown file for ingestion
        if service_result.get('content'):
            intermediate_file = video_file.parent / f"{video_file.stem}_intermediate.md"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                f.write(service_result['content'])
            print_result("Intermediate Markdown File", str(intermediate_file))

        print_section("Output Files")
        print_result("JSON Results", str(output_file))

        # Show all output files created by VideoService
        output_files = service_result.get('output_files', {})
        for file_type, file_path in output_files.items():
            print_result(f"{file_type.title()} File", str(file_path))

        return True

    except Exception as e:
        print(f"[FAIL] Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_video_ingestion(
    video_file: Path,
    webhook_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_qdrant: bool = True,
    use_neo4j: bool = True,
    qdrant_collection_name: Optional[str] = None,
    neo4j_database_name: Optional[str] = None,
    language: Optional[str] = None
) -> bool:
    """
    Test video ingestion using the proper ingestion coordinator.

    Args:
        video_file: Path to video file
        webhook_url: Optional webhook URL for completion notifications
        metadata: Optional metadata dictionary
        use_qdrant: Whether to use Qdrant vector database
        use_neo4j: Whether to use Neo4j graph database
        language: Language code for processing (e.g., 'en', 'de', 'fr')

    Returns:
        True if ingestion was successful, False otherwise
    """
    print_header("MoRAG Video Ingestion Test")

    if not video_file.exists():
        print(f"[FAIL] Error: Video file not found: {video_file}")
        return False

    print_result("Input File", str(video_file))
    print_result("File Size", f"{video_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", video_file.suffix.lower())
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")
    print_result("Language", language or "Auto-detect")
    print_result("Use Qdrant", "[OK] Yes" if use_qdrant else "[FAIL] No")
    print_result("Use Neo4j", "[OK] Yes" if use_neo4j else "[FAIL] No")

    try:
        from morag.api import MoRAGAPI
        from morag.ingestion_coordinator import IngestionCoordinator, DatabaseConfig, DatabaseType
        import uuid

        print_section("Processing Video File")
        print("[PROCESSING] Starting video processing...")
        print("   This may take a while for large videos...")

        # Use VideoService for proper processing with all features enabled
        from morag_video import VideoService, VideoConfig

        # Configure video processing with all features enabled
        video_config = VideoConfig(
            extract_audio=True,
            generate_thumbnails=False,  # Disable to prevent hanging
            extract_keyframes=False,    # Disable for faster processing
            enable_enhanced_audio=True,
            enable_speaker_diarization=True,  # ENABLE for proper speaker info
            enable_topic_segmentation=True,  # ENABLE for topic headers
            enable_ocr=False,           # Disable for faster processing
            language=language
        )

        video_service = VideoService(config=video_config, output_dir=video_file.parent)

        # Process the video file using VideoService
        service_result = await video_service.process_file(
            file_path=video_file,
            save_output=True,
            output_format="markdown"
        )

        # Create intermediate file with transcription for resumable processing
        if service_result.get('content'):
            intermediate_file = video_file.parent / f"{video_file.stem}_intermediate.md"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                f.write(service_result['content'])
            print_result("Intermediate File Created", str(intermediate_file))

            # Create a ProcessingResult for ingestion
            from morag_core.models.config import ProcessingResult
            result = ProcessingResult(
                success=True,
                task_id="video-processing",
                source_type="video",
                content=service_result['content'],
                metadata=service_result.get('result', {}).get('metadata', {}),
                processing_time=service_result.get('processing_time', 0)
            )
            result.text_content = service_result['content']  # Add text_content attribute

        if service_result.get('content'):
            print("[OK] Video processing completed successfully!")

            print_section("Ingesting to Databases")
            print("📊 Starting comprehensive ingestion...")

            # Configure databases based on flags
            database_configs = []
            if use_qdrant:
                collection_name = qdrant_collection_name or os.getenv('MORAG_QDRANT_COLLECTION', 'morag_videos')
                # Use environment variables for Qdrant connection
                qdrant_url = os.getenv('QDRANT_URL')
                if qdrant_url:
                    # Parse URL to get hostname and port
                    from urllib.parse import urlparse
                    parsed = urlparse(qdrant_url)
                    hostname = qdrant_url  # Use full URL as hostname for proper parsing
                    port = parsed.port or (443 if parsed.scheme == 'https' else 6333)
                else:
                    # Fallback to individual host/port settings
                    hostname = os.getenv('QDRANT_HOST', 'localhost')
                    port = int(os.getenv('QDRANT_PORT', '6333'))

                database_configs.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    hostname=hostname,
                    port=port,
                    database_name=collection_name
                ))
            if use_neo4j:
                db_name = neo4j_database_name or os.getenv('NEO4J_DATABASE', 'neo4j')
                database_configs.append(DatabaseConfig(
                    type=DatabaseType.NEO4J,
                    hostname=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                    password=os.getenv('NEO4J_PASSWORD', 'password'),
                    database_name=db_name
                ))

            # Initialize ingestion coordinator
            coordinator = IngestionCoordinator()

            # Use processing result metadata which contains comprehensive document metadata
            ingestion_metadata = result.metadata or {}
            # Merge with any CLI-provided metadata
            if metadata:
                ingestion_metadata.update(metadata)

            # Perform comprehensive ingestion (let coordinator generate proper document ID)
            ingestion_result = await coordinator.ingest_content(
                content=service_result['content'],
                source_path=str(video_file),
                content_type='video',
                metadata=ingestion_metadata,
                processing_result=result,
                databases=database_configs,
                document_id=None,  # Let coordinator generate proper unified ID
                replace_existing=False,
                language=language  # Pass language for consistent extraction
            )

            print("[OK] Content ingested successfully!")

            print_section("Ingestion Results")
            print_result("Status", "[OK] Success")
            print_result("Ingestion ID", ingestion_result['ingestion_id'])
            print_result("Document ID", ingestion_result['source_info']['document_id'])
            print_result("Content Length", f"{ingestion_result['processing_result']['content_length']} characters")
            print_result("Processing Time", f"{ingestion_result['processing_time']:.2f} seconds")
            print_result("Chunks Created", str(ingestion_result['embeddings_data']['chunk_count']))
            print_result("Entities Extracted", str(ingestion_result['graph_data']['entities_count']))
            print_result("Relations Extracted", str(ingestion_result['graph_data']['relations_count']))

            # Show database results
            if 'database_results' in ingestion_result:
                for db_type, db_result in ingestion_result['database_results'].items():
                    if db_result.get('success'):
                        print_result(f"{db_type.title()} Storage", "[OK] Success")
                        if db_type == 'qdrant' and 'points_stored' in db_result:
                            print_result(f"  Points Stored", str(db_result['points_stored']))
                        elif db_type == 'neo4j':
                            if 'chunks_stored' in db_result:
                                print_result(f"  Chunks Stored", str(db_result['chunks_stored']))
                            if 'entities_stored' in db_result:
                                print_result(f"  Entities Stored", str(db_result['entities_stored']))
                            if 'relations_stored' in db_result:
                                print_result(f"  Relations Stored", str(db_result['relations_stored']))
                    else:
                        print_result(f"{db_type.title()} Storage", f"[FAIL] Failed: {db_result.get('error', 'Unknown error')}")

            if webhook_url:
                print_result("Webhook URL", f"Would notify: {webhook_url}")

            print_section("Output Files")
            # The ingestion coordinator automatically creates the files
            result_file = video_file.parent / f"{video_file.stem}.ingest_result.json"
            data_file = video_file.parent / f"{video_file.stem}.ingest_data.json"

            if result_file.exists():
                print_result("Ingest Result File", str(result_file))
            if data_file.exists():
                print_result("Ingest Data File", str(data_file))

            return True
        else:
            print("[FAIL] Video processing failed!")
            print_result("Error", "No content generated from video processing")
            return False

    except Exception as e:
        print(f"[FAIL] Error during video ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False




def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Video Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-video.py my-video.mp4
    python test-video.py recording.avi --thumbnails --thumbnail-count 5
    python test-video.py presentation.mov --enable-ocr

  Ingestion Mode (background processing + storage):
    python test-video.py my-video.mp4 --ingest
    python test-video.py recording.avi --ingest --metadata '{"type": "meeting"}'
    python test-video.py presentation.mov --ingest --webhook-url https://my-app.com/webhook

  Resume from Process Result:
    python test-video.py my-video.mp4 --use-process-result my-video.process_result.json

  Resume from Ingestion Data:
    python test-video.py my-video.mp4 --use-ingestion-data my-video.ingest_data.json

  Resume from Intermediate File:
    python test-video.py my-video.mp4 --use-intermediate my-video_intermediate.md --ingest --language de

Note: Video processing may take several minutes for large files.
        """
    )

    parser.add_argument('video_file', help='Path to video file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (ingestion mode only)')
    parser.add_argument('--qdrant-collection', help='Qdrant collection name (default: from environment or morag_videos)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (ingestion mode only)')
    parser.add_argument('--neo4j-database', help='Neo4j database name (default: from environment or neo4j)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--thumbnails', action='store_true',
                       help='Generate thumbnails (opt-in, default: false)')
    parser.add_argument('--thumbnail-count', type=int, default=3,
                       help='Number of thumbnails to generate (default: 3)')
    parser.add_argument('--enable-ocr', action='store_true',
                       help='Enable OCR on video frames')
    parser.add_argument('--language', required=True, help='Language code for processing (e.g., en, de, fr) - MANDATORY for consistent processing')
    parser.add_argument('--use-process-result', help='Skip processing and use existing process result file (e.g., my-file.process_result.json)')
    parser.add_argument('--use-ingestion-data', help='Skip processing and ingestion calculation, use existing ingestion data file (e.g., my-file.ingest_data.json)')
    parser.add_argument('--use-intermediate', help='Skip video processing and use existing intermediate file (e.g., my-file_intermediate.md)')

    args = parser.parse_args()

    video_file = Path(args.video_file)

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"[FAIL] Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    # Handle intermediate file argument
    if args.use_intermediate:
        async def handle_intermediate_file():
            intermediate_file = Path(args.use_intermediate)
            if not intermediate_file.exists():
                print(f"[FAIL] Error: Intermediate file not found: {intermediate_file}")
                return False

            try:
                with open(intermediate_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                print(f"[OK] Using existing intermediate file: {intermediate_file}")
                print("💡 Skipping video processing, using transcription from intermediate file...")

                # Create a mock processing result with the intermediate content
                from morag_core.models.config import ProcessingResult
                mock_result = ProcessingResult(
                    success=True,
                    task_id="intermediate-resume",
                    source_type="video",
                    content=text_content,
                    metadata=metadata or {}
                )
                mock_result.text_content = text_content  # Add text_content attribute

                if args.ingest:
                    # Continue with ingestion using the intermediate content
                    from morag.ingestion_coordinator import IngestionCoordinator, DatabaseConfig, DatabaseType

                    print_section("Ingesting to Databases")
                    print("📊 Starting comprehensive ingestion from intermediate file...")

                    # Configure databases based on flags
                    database_configs = []
                    if args.qdrant:
                        collection_name = args.qdrant_collection or os.getenv('MORAG_QDRANT_COLLECTION', 'morag_videos')
                        qdrant_url = os.getenv('QDRANT_URL')
                        if qdrant_url:
                            from urllib.parse import urlparse
                            parsed = urlparse(qdrant_url)
                            hostname = qdrant_url
                            port = parsed.port or (443 if parsed.scheme == 'https' else 6333)
                        else:
                            hostname = os.getenv('QDRANT_HOST', 'localhost')
                            port = int(os.getenv('QDRANT_PORT', '6333'))

                        database_configs.append(DatabaseConfig(
                            type=DatabaseType.QDRANT,
                            hostname=hostname,
                            port=port,
                            database_name=collection_name
                        ))
                    if args.neo4j:
                        db_name = args.neo4j_database or os.getenv('NEO4J_DATABASE', 'neo4j')
                        database_configs.append(DatabaseConfig(
                            type=DatabaseType.NEO4J,
                            hostname=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                            username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                            password=os.getenv('NEO4J_PASSWORD', 'password'),
                            database_name=db_name
                        ))

                    # Initialize ingestion coordinator
                    coordinator = IngestionCoordinator()

                    # Use CLI-provided metadata
                    ingestion_metadata = metadata or {}

                    # Perform comprehensive ingestion
                    ingestion_result = await coordinator.ingest_content(
                        content=text_content,
                        source_path=str(video_file),
                        content_type='video',
                        metadata=ingestion_metadata,
                        processing_result=mock_result,
                        databases=database_configs,
                        document_id=None,
                        replace_existing=False,
                        language=args.language
                    )

                    print("[OK] Content ingested successfully from intermediate file!")
                    print(f"\n[SUCCESS] Video ingestion from intermediate file completed successfully!")
                    return True
                else:
                    print("[OK] Intermediate file loaded successfully!")
                    print(f"Content length: {len(text_content)} characters")
                    return True

            except Exception as e:
                print(f"[FAIL] Error reading intermediate file: {e}")
                return False

        # Run the async function
        success = asyncio.run(handle_intermediate_file())
        sys.exit(0 if success else 1)

    # Handle resume arguments
    from resume_utils import handle_resume_arguments
    handle_resume_arguments(args, str(video_file), 'video', metadata)

    try:
        if args.ingest:
            # Ingestion mode
            success = asyncio.run(test_video_ingestion(
                video_file,
                webhook_url=args.webhook_url,
                metadata=metadata,
                use_qdrant=args.qdrant,
                use_neo4j=args.neo4j,
                qdrant_collection_name=args.qdrant_collection,
                neo4j_database_name=args.neo4j_database,
                language=args.language
            ))
            if success:
                print("\n[SUCCESS] Video ingestion test completed successfully!")
                print("💡 Check the .ingest_result.json and .ingest_data.json files for details.")
                sys.exit(0)
            else:
                print("\n[ERROR] Video ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
            success = asyncio.run(test_video_processing(
                video_file,
                generate_thumbnails=args.thumbnails,
                thumbnail_count=args.thumbnail_count,
                enable_ocr=args.enable_ocr,
                language=args.language
            ))
            if success:
                print("\n[SUCCESS] Video processing test completed successfully!")
                sys.exit(0)
            else:
                print("\n[ERROR] Video processing test failed!")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n[STOP]  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
