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

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

try:
    from morag_audio import AudioProcessor, AudioConfig
    from morag_core.models import ProcessingConfig
    from morag_services import QdrantVectorStorage, GeminiEmbeddingService
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
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
    print(f"{spaces}[INFO] {key}: {value}")


async def test_audio_processing(audio_file: Path, model_size: str = "base",
                               enable_diarization: bool = False, enable_topics: bool = False) -> bool:
    """Test audio processing functionality."""
    print_header("MoRAG Audio Processing Test")

    if not audio_file.exists():
        print(f"[FAIL] Error: Audio file not found: {audio_file}")
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
        print_result("Audio Configuration", "[OK] Created successfully")
        print_result("Model Size", model_size)
        print_result("Speaker Diarization", "[OK] Enabled" if enable_diarization else "[FAIL] Disabled")
        print_result("Topic Segmentation", "[OK] Enabled" if enable_topics else "[FAIL] Disabled")

        # Initialize audio processor
        processor = AudioProcessor(config)
        print_result("Audio Processor", "[OK] Initialized successfully")

        print_section("Processing Audio File")
        print("[PROCESSING] Starting audio processing...")

        # Process the audio file
        result = await processor.process(audio_file)

        if result.success:
            print("[OK] Audio processing completed successfully!")

            print_section("Processing Results")
            print_result("Status", "[OK] Success")
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
            print("[FAIL] Audio processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

    except Exception as e:
        print(f"[FAIL] Error during audio processing: {e}")
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
        print("[WARN]  Warning: Empty content provided for vector storage")
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

        print_result("Vector Storage", f"[OK] Stored {len(chunks)} chunks with {len(point_ids)} vectors")

        return point_ids

    except Exception as e:
        print(f"[FAIL] Error storing content in vector database: {e}")
        raise


async def test_audio_ingestion(
    audio_file: Path,
    webhook_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_size: str = "base",
    enable_diarization: bool = False,
    enable_topics: bool = False,
    use_qdrant: bool = True,
    use_neo4j: bool = False,
    qdrant_collection_name: Optional[str] = None,
    neo4j_database_name: Optional[str] = None
) -> bool:
    """Test audio ingestion functionality with graph extraction and dual database storage."""
    print_header("MoRAG Audio Ingestion Test")

    if not audio_file.exists():
        print(f"[FAIL] Error: Audio file not found: {audio_file}")
        return False

    print_result("Input File", str(audio_file))
    print_result("File Size", f"{audio_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("Webhook URL", webhook_url or "Not provided")
    print_result("Custom Metadata", json.dumps(metadata, indent=2) if metadata else "None")
    print_result("Qdrant Storage", "[OK] Enabled" if use_qdrant else "[FAIL] Disabled")
    print_result("Neo4j Storage", "[OK] Enabled" if use_neo4j else "[FAIL] Disabled")

    import time
    start_time = time.time()
    
    try:
        print("[PROCESSING] Starting audio processing and ingestion...")

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
            print("[FAIL] Audio processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

        print("[OK] Audio processing completed successfully!")
        processing_time = time.time() - start_time
        print_result("Processing Time", f"{result.processing_time:.2f} seconds")
        print_result("Transcript Length", f"{len(result.transcript)} characters")

        # Extract entities and relations
        print_section("Graph Extraction")
        print("[PROCESSING] Extracting entities and relations...")
        
        try:
            from graph_extraction import extract_entities_and_relations
            entities, relations = await extract_entities_and_relations(
                text=result.transcript,
                doc_id=f"audio_{audio_file.stem}",
                context=f"Audio transcription from {audio_file.name}"
            )
            print_result("Entities Extracted", f"{len(entities)}")
            print_result("Relations Extracted", f"{len(relations)}")
        except Exception as e:
            print(f"[WARN] Warning: Graph extraction failed: {e}")
            entities, relations = [], []

        # Create intermediate files
        print_section("Creating Intermediate Files")
        
        # Prepare segments data
        segments_data = []
        for segment in result.segments:
            segments_data.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'speaker': getattr(segment, 'speaker', None),
                'confidence': getattr(segment, 'confidence', None)
            })
        
        # Create processing metadata
        from common_schema import ContentType, ProcessingMode
        from morag_core.utils import create_processing_metadata, get_output_paths
        from morag_core.intermediate import IntermediateJSON
        from morag_core.markdown import MarkdownGenerator
        
        proc_metadata = create_processing_metadata(
            content_type=ContentType.AUDIO,
            source_path=str(audio_file),
            processing_time=processing_time,
            mode=ProcessingMode.INGESTION,
            model_info={
                'model_size': model_size,
                'enable_diarization': enable_diarization,
                'enable_topics': enable_topics
            },
            options={
                'vad_filter': True,
                'word_timestamps': True,
                'include_metadata': True,
                'use_qdrant': use_qdrant,
                'use_neo4j': use_neo4j
            }
        )
        
        # Create intermediate JSON
        intermediate = IntermediateJSON(
            content_type=ContentType.AUDIO.value,
            source_path=str(audio_file),
            title=f"Audio Transcription: {audio_file.name}",
            text_content=result.transcript,
            metadata=proc_metadata,
            entities=entities,
            relations=relations,
            segments=segments_data,
            custom_metadata=metadata
        )
        
        # Get output paths
        output_paths = get_output_paths(audio_file, ProcessingMode.INGESTION)
        
        # Save intermediate JSON
        intermediate.to_json(output_paths['intermediate_json'])
        print_result("Intermediate JSON", str(output_paths['intermediate_json']))
        
        # Generate and save intermediate markdown
        MarkdownGenerator.save_markdown(intermediate, output_paths['intermediate_md'])
        print_result("Intermediate Markdown", str(output_paths['intermediate_md']))

        # Database ingestion
        print_section("Database Ingestion")
        
        ingestion_results = {'qdrant': None, 'neo4j': None}
        
        if use_qdrant or use_neo4j:
            print("[PROCESSING] Starting database ingestion...")
            
            # Prepare metadata for ingestion
            ingestion_metadata = {
                "source_type": "audio",
                "source_path": str(audio_file),
                "processing_time": result.processing_time,
                "model_size": model_size,
                "enable_diarization": enable_diarization,
                "enable_topics": enable_topics,
                "transcript_length": len(result.transcript),
                "segments_count": len(result.segments),
                "entities_count": len(entities),
                "relations_count": len(relations),
                **(result.metadata or {}),
                **(metadata or {})
            }
            
            try:
                from graph_extraction import extract_and_ingest

                # Prepare database configurations
                qdrant_config = None
                neo4j_config = None

                if use_qdrant:
                    collection_name = qdrant_collection_name or os.getenv('MORAG_QDRANT_COLLECTION', 'morag_audio')
                    qdrant_config = {
                        'host': 'localhost',
                        'port': 6333,
                        'collection_name': collection_name
                    }

                if use_neo4j:
                    db_name = neo4j_database_name or os.getenv('NEO4J_DATABASE', 'neo4j')
                    neo4j_config = {
                        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
                        'password': os.getenv('NEO4J_PASSWORD', 'password'),
                        'database': db_name
                    }

                graph_results = await extract_and_ingest(
                    text_content=result.transcript,
                    doc_id=f"audio_{audio_file.stem}",
                    context=f"Audio transcription from {audio_file.name}",
                    use_qdrant=use_qdrant,
                    use_neo4j=use_neo4j,
                    qdrant_config=qdrant_config,
                    neo4j_config=neo4j_config,
                    metadata=ingestion_metadata
                )
                
                ingestion_results = graph_results.get('ingestion', {})
                
                if use_qdrant and ingestion_results.get('qdrant'):
                    if ingestion_results['qdrant'].get('success'):
                        print_result("Qdrant Ingestion", "[OK] Success")
                        print_result("Qdrant Chunks", str(ingestion_results['qdrant'].get('chunks_count', 0)))
                    else:
                        print_result("Qdrant Ingestion", f"[FAIL] Failed: {ingestion_results['qdrant'].get('error', 'Unknown error')}")
                
                if use_neo4j and ingestion_results.get('neo4j'):
                    if ingestion_results['neo4j'].get('success'):
                        print_result("Neo4j Ingestion", "[OK] Success")
                        print_result("Neo4j Entities", str(ingestion_results['neo4j'].get('entities_stored', 0)))
                        print_result("Neo4j Relations", str(ingestion_results['neo4j'].get('relations_stored', 0)))
                    else:
                        print_result("Neo4j Ingestion", f"[FAIL] Failed: {ingestion_results['neo4j'].get('error', 'Unknown error')}")
                        
            except Exception as e:
                print(f"[WARN] Warning: Database ingestion failed: {e}")
                ingestion_results = {'error': str(e)}
        
        print("[OK] Audio ingestion completed successfully!")

        print_section("Ingestion Results")
        print_result("Status", "[OK] Success")
        print_result("Total Processing Time", f"{processing_time:.2f} seconds")
        print_result("Transcript Length", str(len(result.transcript)))
        print_result("Segments Count", str(len(result.segments)))
        print_result("Entities Extracted", str(len(entities)))
        print_result("Relations Extracted", str(len(relations)))

        # Save ingestion result
        with open(output_paths['result_json'], 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'ingestion',
                'success': True,
                'processing_time': processing_time,
                'transcript_length': len(result.transcript),
                'segments_count': len(result.segments),
                'entities_count': len(entities),
                'relations_count': len(relations),
                'model_size': model_size,
                'enable_diarization': enable_diarization,
                'enable_topics': enable_topics,
                'use_qdrant': use_qdrant,
                'use_neo4j': use_neo4j,
                'webhook_url': webhook_url,
                'ingestion_results': ingestion_results,
                'metadata': ingestion_metadata,
                'file_path': str(audio_file)
            }, f, indent=2, ensure_ascii=False)

        print_section("Output Files")
        print_result("Intermediate JSON", str(output_paths['intermediate_json']))
        print_result("Intermediate Markdown", str(output_paths['intermediate_md']))
        print_result("Ingestion Result", str(output_paths['result_json']))

        return True

    except Exception as e:
        print(f"[FAIL] Error during audio ingestion: {e}")
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

  Resume from Process Result:
    python test-audio.py my-audio.mp3 --use-process-result my-audio.process_result.json

  Resume from Ingestion Data:
    python test-audio.py my-audio.mp3 --use-ingestion-data my-audio.ingest_data.json
        """
    )

    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (ingestion mode only)')
    parser.add_argument('--qdrant-collection', help='Qdrant collection name (default: from environment or morag_audio)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (ingestion mode only)')
    parser.add_argument('--neo4j-database', help='Neo4j database name (default: from environment or neo4j)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--model-size', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size (default: base)')
    parser.add_argument('--enable-diarization', action='store_true',
                       help='Enable speaker diarization')
    parser.add_argument('--enable-topics', action='store_true',
                       help='Enable topic segmentation')
    parser.add_argument('--language', required=True, help='Language code for processing (e.g., en, de, fr) - MANDATORY for consistent processing')
    parser.add_argument('--use-process-result', help='Skip processing and use existing process result file (e.g., my-file.process_result.json)')
    parser.add_argument('--use-ingestion-data', help='Skip processing and ingestion calculation, use existing ingestion data file (e.g., my-file.ingest_data.json)')

    args = parser.parse_args()

    audio_file = Path(args.audio_file)

    # Extract database configuration arguments
    qdrant_collection_name = args.qdrant_collection
    neo4j_database_name = args.neo4j_database

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"[FAIL] Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    # Handle resume arguments
    from resume_utils import handle_resume_arguments
    handle_resume_arguments(args, str(audio_file), 'audio', metadata)

    try:
        if args.ingest:
            # Ingestion mode
            success = asyncio.run(test_audio_ingestion(
                audio_file,
                webhook_url=args.webhook_url,
                metadata=metadata,
                model_size=args.model_size,
                enable_diarization=args.enable_diarization,
                enable_topics=args.enable_topics,
                use_qdrant=args.qdrant,
                use_neo4j=args.neo4j,
                qdrant_collection_name=args.qdrant_collection,
                neo4j_database_name=args.neo4j_database
            ))
            if success:
                print("\n[SUCCESS] Audio ingestion test completed successfully!")
                print("💡 Use the task ID to monitor progress and retrieve results.")
                sys.exit(0)
            else:
                print("\n[ERROR] Audio ingestion test failed!")
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
                print("\n[SUCCESS] Audio processing test completed successfully!")
                sys.exit(0)
            else:
                print("\n[ERROR] Audio processing test failed!")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n[STOP]  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
