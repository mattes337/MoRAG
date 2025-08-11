#!/usr/bin/env python3
"""
MoRAG Folder Processing CLI Script

Recursively processes all files in a folder for ingestion.
Supports automatic file type detection and skips files that have already been processed.

Usage:
    python test-folder.py <folder_path> [options]

Examples:
    # Process all files in folder with Qdrant storage
    python test-folder.py /path/to/documents --ingest --qdrant
    
    # Process with both Qdrant and Neo4j
    python test-folder.py /path/to/documents --ingest --qdrant --neo4j
    
    # Process with custom metadata
    python test-folder.py /path/to/documents --ingest --qdrant --metadata '{"category": "research"}'
    
    # Dry run to see what files would be processed
    python test-folder.py /path/to/documents --dry-run

    # Force reprocess files even if output files exist
    python test-folder.py /path/to/documents --ingest --qdrant --force-reprocess

    # Process with limited concurrency
    python test-folder.py /path/to/documents --ingest --qdrant --max-concurrent 1

Features:
    - Automatic file type detection (documents, images, audio, video)
    - Recursive folder processing (can be disabled with --no-recursive)
    - Skip already processed files (based on ingest_result.json existence)
    - Resume from existing ingest_data.json files
    - Configurable concurrency (default: 3 files at once)
    - Dry run mode to preview what would be processed
    - Force reprocess option to override skip logic
    - Support for custom metadata and language specification
    - Comprehensive error handling and progress reporting
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add the CLI directory to the path for imports
cli_dir = Path(__file__).parent
sys.path.insert(0, str(cli_dir))

# Load environment variables from .env file
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

from morag.api import MoRAGAPI
from morag_graph.models.database_config import DatabaseConfig, DatabaseType


class ServiceResultWrapper:
    """Wrapper to convert service dictionary results to object-like interface."""

    def __init__(self, result_dict):
        self.success = result_dict.get('success', True)
        self.processing_time = result_dict.get('processing_time', 0.0)
        self.metadata = result_dict.get('metadata', {})
        self.error_message = result_dict.get('error', None)
        self.content = result_dict.get('content', '')
        self.text_content = result_dict.get('content', '')
        # Handle document-specific results
        if 'document' in result_dict:
            self.document = result_dict['document']


def get_supported_extensions() -> Set[str]:
    """Get all supported file extensions for ingestion."""
    return {
        # Documents
        '.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.pptx', '.ppt',
        '.xlsx', '.xls', '.csv', '.html', '.htm',
        # Audio
        '.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac',
        # Video
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'
    }


def is_intermediate_file(file_path: Path) -> bool:
    """Check if a file is an intermediate file that should be skipped."""
    stem = file_path.stem
    return stem.endswith('_intermediate')


def detect_content_type(file_path: Path) -> Optional[str]:
    """Detect content type from file extension."""
    # Skip intermediate files - they should not be processed as regular documents
    if is_intermediate_file(file_path):
        return None

    suffix = file_path.suffix.lower()

    # Document types
    if suffix in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.html', '.htm']:
        return 'document'
    # Audio types
    elif suffix in ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac']:
        return 'audio'
    # Video types
    elif suffix in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']:
        return 'video'
    # Image types
    elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']:
        return 'image'

    return None


def get_output_file_paths(file_path: Path) -> Dict[str, Path]:
    """Get the expected output file paths for a given input file."""
    stem = file_path.stem
    parent = file_path.parent

    return {
        'ingest_result': parent / f"{stem}.ingest_result.json",
        'ingest_data': parent / f"{stem}.ingest_data.json",
        'processing_result': parent / f"{stem}_processing_result.json",
        'intermediate_json': parent / f"{stem}_intermediate.json",
        'intermediate_md': parent / f"{stem}_intermediate.md"
    }


def get_intermediate_file_path(file_path: Path) -> Optional[Path]:
    """Get the intermediate markdown file path for a given input file."""
    stem = file_path.stem
    parent = file_path.parent
    intermediate_md = parent / f"{stem}_intermediate.md"

    if intermediate_md.exists():
        return intermediate_md
    return None


def should_skip_file(file_path: Path, force_reprocess: bool = False) -> tuple[bool, str]:
    """Check if a file should be skipped based on existing output files."""
    if force_reprocess:
        return False, "Force reprocess enabled"

    output_paths = get_output_file_paths(file_path)

    # Skip if ingest_result exists (complete processing)
    if output_paths['ingest_result'].exists():
        return True, f"Ingest result exists: {output_paths['ingest_result']}"

    return False, "No existing output files found"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_file_info(file_path: Path) -> Dict[str, any]:
    """Get basic file information."""
    try:
        stat = file_path.stat()
        return {
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'modified': stat.st_mtime,
            'extension': file_path.suffix.lower()
        }
    except Exception:
        return {
            'size': 0,
            'size_formatted': '0 B',
            'modified': 0,
            'extension': file_path.suffix.lower()
        }


def find_processable_files(folder_path: Path, recursive: bool = True) -> List[Path]:
    """Find all processable files in the folder, excluding intermediate files."""
    supported_extensions = get_supported_extensions()
    processable_files = []

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for file_path in folder_path.glob(pattern):
        if (file_path.is_file() and
            file_path.suffix.lower() in supported_extensions and
            not is_intermediate_file(file_path)):
            processable_files.append(file_path)

    return sorted(processable_files)


async def process_single_file(
    file_path: Path,
    api: MoRAGAPI,
    database_configs: List[DatabaseConfig],
    metadata: Optional[Dict] = None,
    language: Optional[str] = None,
    dry_run: bool = False,
    force_reprocess: bool = False
) -> Dict[str, any]:
    """Process a single file for ingestion."""
    
    # Check if we should skip this file
    should_skip, skip_reason = should_skip_file(file_path, force_reprocess)
    
    if should_skip:
        return {
            'file': str(file_path),
            'status': 'skipped',
            'reason': skip_reason,
            'content_type': detect_content_type(file_path)
        }
    
    # Detect content type
    content_type = detect_content_type(file_path)
    if not content_type:
        return {
            'file': str(file_path),
            'status': 'unsupported',
            'reason': f"Unsupported file type: {file_path.suffix}",
            'content_type': None
        }
    
    if dry_run:
        return {
            'file': str(file_path),
            'status': 'would_process',
            'content_type': content_type,
            'reason': 'Dry run mode'
        }
    
    # Initialize variables for error handling
    original_file_path = file_path
    original_content_type = content_type

    try:
        file_info = get_file_info(file_path)
        print(f"[PROCESSING] Processing: {file_path}")
        print(f"   Content type: {content_type}")
        print(f"   File size: {file_info['size_formatted']}")

        # Auto-detect resume files using centralized functionality
        from resume_utils import auto_detect_resume_files
        detected_files = auto_detect_resume_files(str(file_path))

        # Check if ingest_data file exists (can skip processing and go straight to ingestion)
        if detected_files['ingestion_data']:
            ingest_data_path = detected_files['ingestion_data']
            print(f"   [AUTO-DETECT] Using existing ingest_data: {ingest_data_path}")
            try:
                # Load existing ingest_data and perform database writes
                with open(ingest_data_path, 'r', encoding='utf-8') as f:
                    ingest_data = json.load(f)

                # Use the ingestion coordinator to write to databases
                from morag.ingestion_coordinator import IngestionCoordinator
                coordinator = IngestionCoordinator()

                # Extract data from ingest_data file
                document_id = ingest_data.get('document_id')
                vector_data = ingest_data.get('vector_data', {})
                graph_data = ingest_data.get('graph_data', {})

                # Prepare embeddings data
                embeddings_data = {
                    'chunks': [chunk['chunk_text'] for chunk in vector_data.get('chunks', [])],
                    'embeddings': [chunk['embedding'] for chunk in vector_data.get('chunks', [])],
                    'chunk_metadata': [chunk['metadata'] for chunk in vector_data.get('chunks', [])]
                }

                # Write to databases
                await coordinator._write_to_databases(
                    database_configs, embeddings_data, graph_data, document_id, True
                )

                print(f"   [SUCCESS] Database ingestion completed from existing data")
                return {
                    'file': str(file_path),
                    'status': 'success_from_data',
                    'content_type': content_type,
                    'reason': 'Used existing ingest_data file'
                }
            except Exception as e:
                print(f"   [WARNING] Failed to use existing ingest_data, will reprocess: {e}")
                # Fall through to normal processing

        # Check if process_result file exists (can skip processing but need to do ingestion)
        elif detected_files['process_result']:
            process_result_path = detected_files['process_result']
            print(f"   [AUTO-DETECT] Using existing process_result: {process_result_path}")
            try:
                # Load existing process_result and continue with ingestion
                with open(process_result_path, 'r', encoding='utf-8') as f:
                    process_result_data = json.load(f)

                # Use the ingestion coordinator to perform full ingestion
                from morag.ingestion_coordinator import IngestionCoordinator
                from morag_core.models.config import ProcessingResult

                coordinator = IngestionCoordinator()

                # Create ProcessingResult object from the data
                result = ProcessingResult(
                    success=process_result_data.get('success', True),
                    task_id=process_result_data.get('task_id', 'folder-resume'),
                    source_type=content_type,
                    content=process_result_data.get('content', ''),
                    metadata=process_result_data.get('metadata', {}),
                    processing_time=process_result_data.get('processing_time', 0.0)
                )

                # Perform comprehensive ingestion
                ingestion_result = await coordinator.ingest_content(
                    content=result.content,
                    source_path=str(file_path),
                    content_type=content_type,
                    metadata=metadata or {},
                    processing_result=result,
                    databases=database_configs,
                    document_id=None,
                    replace_existing=False,
                    language=language
                )

                print(f"   [SUCCESS] Content ingested successfully from process result!")
                return {
                    'file': str(file_path),
                    'status': 'success_from_process_result',
                    'content_type': content_type,
                    'reason': 'Used existing process_result file'
                }
            except Exception as e:
                print(f"   [WARNING] Failed to use existing process_result, will reprocess: {e}")
                # Fall through to normal processing

        # Check for intermediate file for audio/video content types and create mock result

        if content_type in ['audio', 'video']:
            intermediate_file = get_intermediate_file_path(file_path)
            if intermediate_file:
                print(f"   [INFO] Found intermediate file: {intermediate_file}")
                print(f"   [INFO] Using intermediate file instead of processing original {content_type}")

                # Read the intermediate file content
                try:
                    with open(intermediate_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()

                    # Create a mock processing result with the intermediate content (like test-video.py)
                    from morag_core.models.config import ProcessingResult
                    result = ProcessingResult(
                        success=True,
                        task_id="intermediate-resume",
                        source_type=content_type,
                        content=text_content,
                        metadata=metadata or {},
                        processing_time=0.0
                    )

                    # Add text_content attribute for compatibility with ingestion logic
                    result.text_content = text_content

                    print(f"   [SUCCESS] Loaded content from intermediate file ({len(text_content)} chars)")

                except Exception as e:
                    print(f"   [WARNING] Failed to read intermediate file: {e}")
                    print(f"   [INFO] Falling back to processing original {content_type} file")
                    # Fall through to normal processing
                    result = None
            else:
                result = None
        else:
            result = None

        # If we don't have a result from intermediate file, process normally
        if result is None:
            if content_type == 'document':
                from morag_document import DocumentProcessor
                processor = DocumentProcessor()
                result = await processor.process_file(file_path)
            elif content_type == 'image':
                from morag_image import ImageService
                processor = ImageService()
                service_result = await processor.process_image(file_path)
                result = ServiceResultWrapper(service_result)
            elif content_type == 'audio':
                from morag_audio import AudioService
                from morag_audio.processor import AudioConfig

                # Configure audio processing for consistent language handling
                audio_config = AudioConfig(
                    enable_enhanced_audio=True,
                    enable_speaker_diarization=True,
                    enable_topic_segmentation=True,
                    language=language
                )

                processor = AudioService(config=audio_config)
                service_result = await processor.process_file(file_path, save_output=False)
                result = ServiceResultWrapper(service_result)
            elif content_type == 'video':
                from morag_video import VideoService
                from morag_video.processor import VideoConfig

                # Configure video processing to disable thumbnail generation for faster processing
                video_config = VideoConfig(
                    extract_audio=True,
                    generate_thumbnails=False,  # Disable to prevent hanging
                    extract_keyframes=False,    # Disable for faster processing
                    enable_enhanced_audio=True,
                    enable_speaker_diarization=True,
                    enable_topic_segmentation=True,
                    enable_ocr=False,           # Disable for faster processing
                    language=language
                )

                processor = VideoService(config=video_config)
                service_result = await processor.process_file(file_path, save_output=False)
                result = ServiceResultWrapper(service_result)
            else:
                # Use the general API for other types
                result = await api.process_file(str(file_path), content_type)

        if not result.success:
            raise Exception(f"Processing failed: {result.error_message or 'Unknown error'}")

        # If databases are configured, perform ingestion
        if database_configs:
            from morag.ingestion_coordinator import IngestionCoordinator
            coordinator = IngestionCoordinator()

            # Prepare enhanced metadata (use original file path for source_path)
            enhanced_metadata = {
                'source_type': original_content_type,
                'source_path': str(original_file_path),
                'processing_time': result.processing_time,
                **(metadata or {}),
                **(result.metadata or {})
            }

            # Extract content from result (handle different ProcessingResult types)
            content = ""
            if hasattr(result, 'text_content') and result.text_content:
                content = result.text_content
            elif hasattr(result, 'content') and result.content:
                content = result.content
            elif hasattr(result, 'document') and result.document:
                if hasattr(result.document, 'raw_text'):
                    content = result.document.raw_text
                elif hasattr(result.document, 'content'):
                    content = result.document.content

            if not content:
                raise Exception("No content extracted from processing result")

            # Perform comprehensive ingestion (use original file path and content type)
            ingestion_result = await coordinator.ingest_content(
                content=content,
                source_path=str(original_file_path),
                content_type=original_content_type,
                metadata=enhanced_metadata,
                processing_result=result,
                databases=database_configs,
                chunk_size=4000,  # Default chunk size
                chunk_overlap=200,  # Default overlap
                document_id=None,  # Let coordinator generate unified ID
                replace_existing=True,
                language=language
            )

        print(f"   [SUCCESS] Success: {original_file_path}")
        return {
            'file': str(original_file_path),
            'status': 'success',
            'content_type': original_content_type,
            'task_id': getattr(result, 'task_id', None),
            'processing_time': getattr(result, 'processing_time', None)
        }

    except Exception as e:
        print(f"   [ERROR] Error processing {original_file_path}: {e}")
        return {
            'file': str(original_file_path),
            'status': 'error',
            'content_type': original_content_type,
            'error': str(e)
        }


async def process_folder(
    folder_path: Path,
    database_configs: List[DatabaseConfig],
    metadata: Optional[Dict] = None,
    language: Optional[str] = None,
    recursive: bool = True,
    dry_run: bool = False,
    max_concurrent: int = 3,
    force_reprocess: bool = False
) -> Dict[str, any]:
    """Process all files in a folder."""
    
    print(f"[INFO] Scanning folder: {folder_path}")
    print(f"   Recursive: {recursive}")
    print(f"   Dry run: {dry_run}")
    
    # Find all processable files
    files = find_processable_files(folder_path, recursive)
    print(f"   Found {len(files)} processable files")
    
    if not files:
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'results': []
        }
    
    # Initialize API
    api = MoRAGAPI()
    
    # Process files with limited concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_path):
        async with semaphore:
            return await process_single_file(
                file_path, api, database_configs, metadata, language, dry_run, force_reprocess
            )
    
    # Process all files
    print(f"\n[PROCESSING] Processing {len(files)} files...")
    if not dry_run:
        total_size = sum(get_file_info(f)['size'] for f in files)
        print(f"   Total size: {format_file_size(total_size)}")
        print(f"   Concurrency: {max_concurrent} files at once")

    results = await asyncio.gather(*[process_with_semaphore(f) for f in files])
    
    # Summarize results
    summary = {
        'total_files': len(files),
        'processed': len([r for r in results if r['status'] in ['success', 'success_from_data']]),
        'skipped': len([r for r in results if r['status'] == 'skipped']),
        'errors': len([r for r in results if r['status'] == 'error']),
        'unsupported': len([r for r in results if r['status'] == 'unsupported']),
        'would_process': len([r for r in results if r['status'] == 'would_process']),
        'results': results
    }
    
    return summary


def setup_database_configs(args) -> List[DatabaseConfig]:
    """Set up database configurations based on CLI arguments."""
    database_configs = []
    
    if args.qdrant:
        qdrant_collection = (
            args.qdrant_collection or
            os.getenv('QDRANT_COLLECTION', 'morag_folder')
        )
        database_configs.append(DatabaseConfig(
            type=DatabaseType.QDRANT,
            hostname=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', '6333')),
            database_name=qdrant_collection
        ))
    
    if args.neo4j:
        neo4j_database = (
            args.neo4j_database or
            os.getenv('NEO4J_DATABASE', 'neo4j')
        )
        database_configs.append(DatabaseConfig(
            type=DatabaseType.NEO4J,
            hostname=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=os.getenv('NEO4J_USERNAME', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'password'),
            database_name=neo4j_database
        ))
    
    return database_configs


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process all files in a folder for MoRAG ingestion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('folder', help='Path to folder to process')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (required for database storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (requires --ingest)')
    parser.add_argument('--qdrant-collection', help='Qdrant collection name (default: from environment or morag_folder)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (requires --ingest)')
    parser.add_argument('--neo4j-database', help='Neo4j database name (default: from environment or neo4j)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string')
    parser.add_argument('--language', help='Language code for processing (auto-detect if not specified)')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Process files recursively (default: True)')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                       help='Do not process files recursively')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what files would be processed without actually processing them')
    parser.add_argument('--max-concurrent', type=int, default=3,
                       help='Maximum number of files to process concurrently (default: 3)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of files even if output files exist')
    
    args = parser.parse_args()
    
    # Validate arguments
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"[ERROR] Error: Folder does not exist: {folder_path}")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"[ERROR] Error: Path is not a directory: {folder_path}")
        sys.exit(1)

    if not args.ingest and (args.qdrant or args.neo4j):
        print("[ERROR] Error: --qdrant and --neo4j require --ingest flag")
        sys.exit(1)
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error: Invalid JSON in metadata: {e}")
            sys.exit(1)
    
    # Set up database configurations
    database_configs = setup_database_configs(args)
    
    if args.ingest and not database_configs:
        print("[WARNING] Warning: Ingestion mode enabled but no databases configured")
        print("   Use --qdrant and/or --neo4j to specify databases")
    
    # Run the processing
    try:
        summary = asyncio.run(process_folder(
            folder_path=folder_path,
            database_configs=database_configs,
            metadata=metadata,
            language=args.language,
            recursive=args.recursive,
            dry_run=args.dry_run,
            max_concurrent=args.max_concurrent,
            force_reprocess=args.force_reprocess
        ))
        
        # Print summary
        print(f"\n[SUMMARY] Processing Summary:")
        print(f"   Total files found: {summary['total_files']}")
        if args.dry_run:
            print(f"   Would process: {summary['would_process']}")
        else:
            print(f"   Successfully processed: {summary['processed']}")
        print(f"   Skipped (already processed): {summary['skipped']}")
        print(f"   Errors: {summary['errors']}")
        print(f"   Unsupported file types: {summary['unsupported']}")

        # Show errors if any
        errors = [r for r in summary['results'] if r['status'] == 'error']
        if errors:
            print(f"\n[ERRORS] Errors encountered:")
            for error in errors:
                print(f"   {error['file']}: {error['error']}")
        
        # Exit with error code if there were errors
        if summary['errors'] > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[WARNING] Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
