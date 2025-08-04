#!/usr/bin/env python3
"""
MoRAG Markdown Folder Ingestion Script

Recursively processes all markdown files in a folder and ingests them into Neo4j.
Optimized for German language processing and Neo4j graph database storage.

Usage:
    python ingest-markdown-folder.py <folder_path> [options]

Examples:
    # Ingest all markdown files with German language
    python ingest-markdown-folder.py /path/to/markdown/files --language de
    
    # Ingest with custom Neo4j database
    python ingest-markdown-folder.py /path/to/markdown/files --language de --neo4j-database my_graph
    
    # Dry run to see what files would be processed
    python ingest-markdown-folder.py /path/to/markdown/files --dry-run
    
    # Force reprocess files even if already processed
    python ingest-markdown-folder.py /path/to/markdown/files --language de --force-reprocess

Features:
    - Processes only markdown files (.md, .markdown)
    - Recursive folder processing
    - Skip already processed files (based on ingest_result.json existence)
    - Resume from existing ingest_data.json files
    - Configurable concurrency (default: 3 files at once)
    - Dry run mode to preview what would be processed
    - Force reprocess option to override skip logic
    - German language processing by default
    - Neo4j graph database storage with entity/relation extraction
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


def get_markdown_extensions() -> Set[str]:
    """Get supported markdown file extensions."""
    return {'.md', '.markdown'}


def is_intermediate_file(file_path: Path) -> bool:
    """Check if a file is an intermediate file that should be skipped."""
    stem = file_path.stem
    return stem.endswith('_intermediate')


def find_markdown_files(folder_path: Path, recursive: bool = True) -> List[Path]:
    """Find all markdown files in the folder."""
    markdown_extensions = get_markdown_extensions()
    files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in folder_path.glob(pattern):
        if (file_path.is_file() and 
            file_path.suffix.lower() in markdown_extensions and
            not is_intermediate_file(file_path)):
            files.append(file_path)
    
    return sorted(files)


def should_skip_file(file_path: Path, force_reprocess: bool = False) -> tuple[bool, str]:
    """Check if a file should be skipped based on existing output files."""
    if force_reprocess:
        return False, "force reprocess enabled"
    
    # Check for ingest_result.json (indicates completed ingestion)
    ingest_result_path = file_path.with_suffix(file_path.suffix + '_ingest_result.json')
    if ingest_result_path.exists():
        return True, "already ingested (ingest_result.json exists)"
    
    return False, "needs processing"


async def process_single_markdown_file(
    file_path: Path,
    database_configs: List[DatabaseConfig],
    metadata: Optional[Dict] = None,
    language: str = "de",
    force_reprocess: bool = False
) -> Dict[str, any]:
    """Process a single markdown file."""
    
    print(f"[INFO] Processing: {file_path}")
    
    # Check if we should skip this file
    should_skip, skip_reason = should_skip_file(file_path, force_reprocess)
    if should_skip:
        print(f"   [SKIP] Skipping: {skip_reason}")
        return {
            'file': str(file_path),
            'status': 'skipped',
            'reason': skip_reason
        }
    
    try:
        # Initialize MoRAG API
        api = MoRAGAPI()

        # Initialize ingestion coordinator separately
        from morag.ingestion_coordinator import IngestionCoordinator
        coordinator = IngestionCoordinator()
        
        # Enhanced metadata for markdown files
        enhanced_metadata = {
            'source_type': 'markdown',
            'file_extension': file_path.suffix,
            'file_size': file_path.stat().st_size,
            'language': language,
            **(metadata or {})
        }
        
        # Check for existing ingest_data.json file (intermediate processing result)
        ingest_data_path = file_path.with_suffix(file_path.suffix + '_ingest_data.json')
        if ingest_data_path.exists() and not force_reprocess:
            print(f"   [RESUME] Found existing ingest_data.json, resuming from ingestion step")
            try:
                # Load existing ingest data
                with open(ingest_data_path, 'r', encoding='utf-8') as f:
                    ingest_data = json.load(f)
                
                # Create a mock processing result from the ingest data
                result = ServiceResultWrapper({
                    'success': True,
                    'content': ingest_data.get('content', ''),
                    'metadata': ingest_data.get('metadata', {}),
                    'processing_time': 0.0
                })
                
                # Perform comprehensive ingestion
                ingestion_result = await coordinator.ingest_content(
                    content=result.content,
                    source_path=str(file_path),
                    content_type='text/markdown',
                    metadata=enhanced_metadata,
                    processing_result=result,
                    databases=database_configs,
                    document_id=None,
                    replace_existing=False,
                    language=language
                )

                print(f"   [SUCCESS] Content ingested successfully from ingest_data!")
                return {
                    'file': str(file_path),
                    'status': 'success_from_ingest_data',
                    'reason': 'Used existing ingest_data file'
                }
            except Exception as e:
                print(f"   [WARNING] Failed to use existing ingest_data, will reprocess: {e}")
                # Fall through to normal processing

        # Process the markdown file
        print(f"   [PROCESS] Processing markdown file...")
        result = await api.process_document(
            file_path=str(file_path),
            options={
                'metadata': enhanced_metadata
            }
        )
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Unknown processing error')
            print(f"   [ERROR] Processing failed: {error_msg}")
            return {
                'file': str(file_path),
                'status': 'error',
                'error': error_msg
            }

        if database_configs:
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

            # Perform comprehensive ingestion
            ingestion_result = await coordinator.ingest_content(
                content=content,
                source_path=str(file_path),
                content_type='text/markdown',
                metadata=enhanced_metadata,
                processing_result=result,
                databases=database_configs,
                chunk_size=4000,  # Default chunk size
                chunk_overlap=200,  # Default overlap
                document_id=None,  # Let coordinator generate unified ID
                replace_existing=True,
                language=language
            )

        print(f"   [SUCCESS] Success: {file_path}")
        return {
            'file': str(file_path),
            'status': 'success',
            'task_id': getattr(result, 'task_id', None),
            'processing_time': getattr(result, 'processing_time', None)
        }
        
    except Exception as e:
        print(f"   [ERROR] Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'file': str(file_path),
            'status': 'error',
            'error': str(e)
        }


async def process_markdown_folder(
    folder_path: Path,
    database_configs: List[DatabaseConfig],
    metadata: Optional[Dict] = None,
    language: str = "de",
    recursive: bool = True,
    dry_run: bool = False,
    max_concurrent: int = 3,
    force_reprocess: bool = False
) -> Dict[str, any]:
    """Process all markdown files in a folder."""

    print(f"[INFO] Scanning folder for markdown files: {folder_path}")
    print(f"   Language: {language}")
    print(f"   Recursive: {recursive}")
    print(f"   Dry run: {dry_run}")

    # Find all markdown files
    files = find_markdown_files(folder_path, recursive)
    print(f"   Found {len(files)} markdown files")

    if not files:
        return {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'results': []
        }

    if dry_run:
        print(f"\n[DRY RUN] Would process the following files:")
        would_process = 0
        for file_path in files:
            should_skip, skip_reason = should_skip_file(file_path, force_reprocess)
            if should_skip:
                print(f"   [SKIP] {file_path} - {skip_reason}")
            else:
                print(f"   [PROCESS] {file_path}")
                would_process += 1

        return {
            'total_files': len(files),
            'would_process': would_process,
            'skipped': len(files) - would_process,
            'errors': 0,
            'results': []
        }

    # Process files with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(file_path):
        async with semaphore:
            return await process_single_markdown_file(
                file_path, database_configs, metadata, language, force_reprocess
            )

    print(f"\n[INFO] Processing {len(files)} markdown files with max concurrency: {max_concurrent}")

    # Process all files
    tasks = [process_with_semaphore(file_path) for file_path in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                'file': str(files[i]),
                'status': 'error',
                'error': str(result)
            })
        else:
            processed_results.append(result)

    # Calculate summary statistics
    total_files = len(files)
    processed = len([r for r in processed_results if r['status'] in ['success', 'success_from_ingest_data']])
    skipped = len([r for r in processed_results if r['status'] == 'skipped'])
    errors = len([r for r in processed_results if r['status'] == 'error'])

    return {
        'total_files': total_files,
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'results': processed_results
    }


def setup_neo4j_config(args) -> List[DatabaseConfig]:
    """Set up Neo4j database configuration."""
    database_configs = []

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
        description='Ingest all markdown files in a folder into Neo4j graph database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('folder', help='Path to folder containing markdown files')
    parser.add_argument('--language', default='de', help='Language code for processing (default: de)')
    parser.add_argument('--neo4j-database', help='Neo4j database name (default: from environment or neo4j)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string')
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

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    # Set up Neo4j database configuration
    database_configs = setup_neo4j_config(args)

    print(f"[INFO] MoRAG Markdown Folder Ingestion")
    print(f"   Target folder: {folder_path}")
    print(f"   Language: {args.language}")
    print(f"   Neo4j database: {database_configs[0].database_name}")
    print(f"   Neo4j URI: {database_configs[0].hostname}")

    # Run the processing
    try:
        summary = asyncio.run(process_markdown_folder(
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
        print(f"   Total markdown files found: {summary['total_files']}")
        if args.dry_run:
            print(f"   Would process: {summary['would_process']}")
        else:
            print(f"   Successfully processed: {summary['processed']}")
        print(f"   Skipped (already processed): {summary['skipped']}")
        print(f"   Errors: {summary['errors']}")

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
