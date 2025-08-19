#!/usr/bin/env python3
"""
MoRAG Document Processing Test Script

Supports both processing (immediate results) and ingestion (background + storage) modes.

Usage:
    python test-document.py <document_file> [options]

Processing Mode (immediate results):
    python test-document.py my-document.pdf
    python test-document.py presentation.pptx
    python test-document.py spreadsheet.xlsx
    python test-document.py document.docx

Ingestion Mode (background processing + storage):
    python test-document.py my-document.pdf --ingest
    python test-document.py document.docx --ingest --metadata '{"category": "research", "priority": 1}'
    python test-document.py presentation.pptx --ingest --chunking-strategy chapter

Options:
    --ingest                    Enable ingestion mode (background processing + storage)
    --webhook-url URL          Webhook URL for completion notifications (ingestion mode only)
    --metadata JSON            Additional metadata as JSON string (ingestion mode only)
    --chunking-strategy STRATEGY  Chunking strategy: paragraph, sentence, page, chapter (default: paragraph)
    --chunk-size SIZE          Maximum chunk size in characters (default: 1000)
    --chunk-overlap SIZE       Overlap between chunks in characters (default: 200)
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
    from morag_document import DocumentProcessor, DocumentService
    from morag_core.interfaces.processor import ProcessingConfig
    from morag_core.interfaces.service import ServiceConfig
    from morag_services import QdrantVectorStorage
    from morag_embedding import GeminiEmbeddingService
    from morag_core.models import Document, DocumentChunk
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-document")
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
    try:
        print(f"{spaces}[INFO] {key}: {value}")
    except UnicodeEncodeError:
        # Fallback for terminals that don't support Unicode
        print(f"{spaces}[INFO] {key}: {value}")


async def test_document_processing(document_file: Path, chunking_strategy: str = "paragraph",
                                  chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
    """Test document processing functionality."""
    print_header("MoRAG Document Processing Test")

    if not document_file.exists():
        print(f"[FAIL] Error: Document file not found: {document_file}")
        return False

    print_result("Input File", str(document_file))
    print_result("File Size", f"{document_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", document_file.suffix.lower())

    try:
        # Use DocumentService for proper markdown output
        config = ServiceConfig()
        service = DocumentService(config=config, output_dir=document_file.parent)
        await service.initialize()
        print_result("Document Service", "[OK] Initialized successfully")
        print_result("Chunking Strategy", chunking_strategy)
        print_result("Chunk Size", f"{chunk_size} characters")
        print_result("Chunk Overlap", f"{chunk_overlap} characters")

        print_section("Processing Document File")
        print("[PROCESSING] Starting document processing...")
        print("   This may take a while for large documents...")

        # Process the document file using DocumentService
        service_result = await service.process_file(
            file_path=document_file,
            save_output=True,
            output_format="markdown"
        )

        if service_result.get('success', False):
            try:
                print("[OK] Document processing completed successfully!")
            except UnicodeEncodeError:
                print("[SUCCESS] Document processing completed successfully!")

            # Extract result data from service result
            result_data = service_result.get('result', {})

            print_section("Processing Results")
            try:
                print_result("Status", "[OK] Success")
            except UnicodeEncodeError:
                print_result("Status", "[SUCCESS]")
            print_result("Processing Time", f"{service_result.get('processing_time', 0):.2f} seconds")

            # Check for document processing results
            if result_data:
                content = result_data.get('content', '')
                chunks = result_data.get('chunks', [])
                print_result("Content Length", f"{len(content)} characters")
                print_result("Chunks Count", f"{len(chunks)}")

            metadata = result_data.get('metadata', {})
            if metadata:
                print_section("Metadata")
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        print_result(key, json.dumps(value, indent=2))
                    else:
                        print_result(key, str(value))

            content = result_data.get('content', '')
            chunks = result_data.get('chunks', [])

            if content or chunks:
                print_section("Document Information")
                print_result("Content Available", "[OK] Yes" if content else "[INFO] No")
                print_result("Chunks Available", "[OK] Yes" if chunks else "[INFO] No")

                if content:
                    print_section("Content Preview")
                    content_preview = content[:500] + "..." if len(content) > 500 else content
                    print(f"📄 Content ({len(content)} characters):")
                    print(content_preview)

                if chunks:
                    print_section("Chunks Preview (first 3)")
                    for i, chunk in enumerate(chunks[:3]):
                        # Handle both dict and object chunks
                        if hasattr(chunk, 'content'):
                            chunk_content = chunk.content
                            chunk_index = getattr(chunk, 'chunk_index', None)
                            chunk_metadata = getattr(chunk, 'metadata', None)
                        else:
                            chunk_content = chunk.get('content', '')
                            chunk_index = chunk.get('chunk_index')
                            chunk_metadata = chunk.get('metadata')

                        print(f"  Chunk {i+1}:")
                        print(f"    Content: {chunk_content[:100]}{'...' if len(chunk_content) > 100 else ''}")
                        if chunk_index is not None:
                            print(f"    Index: {chunk_index}")
                        if chunk_metadata:
                            print(f"    Metadata: {chunk_metadata}")

            # Convert chunks to serializable format
            serializable_chunks = []
            for chunk in chunks:
                if hasattr(chunk, 'content'):
                    # DocumentChunk object
                    chunk_dict = {
                        'content': chunk.content,
                        'chunk_index': getattr(chunk, 'chunk_index', None),
                        'metadata': getattr(chunk, 'metadata', {}),
                        'start_char': getattr(chunk, 'start_char', None),
                        'end_char': getattr(chunk, 'end_char', None)
                    }
                else:
                    # Already a dict
                    chunk_dict = chunk
                serializable_chunks.append(chunk_dict)

            # Save comprehensive results to JSON file
            output_file = document_file.parent / f"{document_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'processing',
                    'success': service_result.get('success', False),
                    'processing_time': service_result.get('processing_time', 0),
                    'chunking_strategy': chunking_strategy,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'content': result_data.get('content', ''),
                    'chunks': serializable_chunks,
                    'metadata': result_data.get('metadata', {}),
                    'output_files': service_result.get('output_files', {})
                }, f, indent=2, ensure_ascii=False)

            print_section("Output Files")
            print_result("JSON Results", str(output_file))

            # Show all output files created by DocumentService
            output_files = service_result.get('output_files', {})
            for file_type, file_path in output_files.items():
                print_result(f"{file_type.title()} File", str(file_path))

            return True

        else:
            try:
                print("[FAIL] Document processing failed!")
            except UnicodeEncodeError:
                print("[ERROR] Document processing failed!")
            print_result("Error", service_result.get('error', 'Unknown error'))
            return False

    except Exception as e:
        try:
            print(f"[FAIL] Error during document processing: {e}")
        except UnicodeEncodeError:
            print(f"[ERROR] Error during document processing: {e}")
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


async def test_document_ingestion(document_file: Path, webhook_url: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 chunking_strategy: str = "paragraph",
                                 chunk_size: int = 1000, chunk_overlap: int = 200,
                                 use_qdrant: bool = True, use_neo4j: bool = True,
                                 qdrant_collection_name: Optional[str] = None,
                                 neo4j_database_name: Optional[str] = None,
                                 language: str = "en") -> bool:
    """Test document ingestion using the proper ingestion coordinator."""
    print_header("MoRAG Document Ingestion Test")

    if not document_file.exists():
        print(f"[FAIL] Error: Document file not found: {document_file}")
        return False

    print_result("Input File", str(document_file))
    print_result("File Size", f"{document_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", document_file.suffix.lower())
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")
    print_result("Chunking Strategy", chunking_strategy)
    print_result("Chunk Size", f"{chunk_size} characters")
    print_result("Chunk Overlap", f"{chunk_overlap} characters")
    print_result("Use Qdrant", "[OK] Yes" if use_qdrant else "[FAIL] No")
    print_result("Use Neo4j", "[OK] Yes" if use_neo4j else "[FAIL] No")

    try:
        from morag.ingestion_coordinator import IngestionCoordinator, DatabaseConfig, DatabaseType
        import uuid

        print_section("Processing Document")
        print("[PROCESSING] Starting document processing...")

        # Initialize document processor
        processor = DocumentProcessor()

        # Process the document file with options
        result = await processor.process_file(
            document_file,
            extract_metadata=True,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not result.success:
            print("[FAIL] Document processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

        print("[OK] Document processing completed successfully!")
        print_result("Processing Time", f"{result.processing_time:.2f} seconds")

        print_section("Ingesting to Databases")
        print("📊 Starting comprehensive ingestion...")

        # Configure databases based on flags
        database_configs = []
        if use_qdrant:
            qdrant_collection = (
                qdrant_collection_name or
                os.getenv('QDRANT_COLLECTION', 'morag_documents')
            )
            database_configs.append(DatabaseConfig(
                type=DatabaseType.QDRANT,
                hostname='localhost',
                port=6333,
                database_name=qdrant_collection
            ))
        if use_neo4j:
            neo4j_database = (
                neo4j_database_name or
                os.getenv('NEO4J_DATABASE', 'neo4j')
            )
            database_configs.append(DatabaseConfig(
                type=DatabaseType.NEO4J,
                hostname=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                password=os.getenv('NEO4J_PASSWORD', 'password'),
                database_name=neo4j_database
            ))

        # Prepare enhanced metadata
        enhanced_metadata = {
            "source_type": "document",
            "source_path": str(document_file),
            "processing_time": result.processing_time,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            **(result.metadata or {}),
            **(metadata or {})
        }

        # Initialize ingestion coordinator
        coordinator = IngestionCoordinator()

        # Perform comprehensive ingestion (let coordinator generate proper document ID)
        ingestion_result = await coordinator.ingest_content(
            content=result.document.raw_text if result.document else result.content,
            source_path=str(document_file),
            content_type='document',
            metadata=enhanced_metadata,
            processing_result=result,
            databases=database_configs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=None,  # Let coordinator generate unified document ID
            replace_existing=False,
            language=language  # Pass language parameter for fact extraction
        )

        print("[OK] Document ingestion completed successfully!")

        print_section("Ingestion Results")
        print_result("Status", "[OK] Success")
        print_result("Ingestion ID", ingestion_result['ingestion_id'])
        print_result("Document ID", ingestion_result['source_info']['document_id'])
        print_result("Content Length", f"{ingestion_result['processing_result']['content_length']} characters")
        print_result("Processing Time", f"{ingestion_result['processing_time']:.2f} seconds")
        print_result("Chunks Created", str(ingestion_result['embeddings_data']['chunk_count']))
        print_result("Facts Extracted", str(ingestion_result['graph_data']['facts_count']))
        print_result("Relationships Extracted", str(ingestion_result['graph_data']['relationships_count']))

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
                        if 'facts_processed' in db_result:
                            print_result(f"  Facts Processed", str(db_result['facts_processed']))
                        if 'relationships_processed' in db_result:
                            print_result(f"  Relationships Processed", str(db_result['relationships_processed']))
                        # Legacy support for old entity/relation fields
                        if 'entities_stored' in db_result:
                            print_result(f"  Entities Stored", str(db_result['entities_stored']))
                        if 'relations_stored' in db_result:
                            print_result(f"  Relations Stored", str(db_result['relations_stored']))
                else:
                    print_result(f"{db_type.title()} Storage", f"[FAIL] Failed: {db_result.get('error', 'Unknown error')}")

        print_section("Output Files")
        # The ingestion coordinator automatically creates the files
        result_file = document_file.parent / f"{document_file.stem}.ingest_result.json"
        data_file = document_file.parent / f"{document_file.stem}.ingest_data.json"

        if result_file.exists():
            print_result("Ingest Result File", str(result_file))
        if data_file.exists():
            print_result("Ingest Data File", str(data_file))

        return True

    except Exception as e:
        print(f"[FAIL] Error during document ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Document Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-document.py my-document.pdf
    python test-document.py presentation.pptx --chunking-strategy chapter
    python test-document.py document.docx --chunk-size 2000

  Ingestion Mode (background processing + storage):
    python test-document.py my-document.pdf --ingest
    python test-document.py document.docx --ingest --metadata '{"category": "research"}'
    python test-document.py presentation.pptx --ingest --webhook-url https://my-app.com/webhook

  Resume from Process Result:
    python test-document.py my-document.pdf --use-process-result my-document.process_result.json

  Resume from Ingestion Data:
    python test-document.py my-document.pdf --use-ingestion-data my-document.ingest_data.json
        """
    )

    parser.add_argument('document_file', help='Path to document file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (ingestion mode only)')
    parser.add_argument('--qdrant-collection', help='Qdrant collection name (default: from environment or morag_documents)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (ingestion mode only)')
    parser.add_argument('--neo4j-database', help='Neo4j database name (default: from environment or neo4j)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--chunking-strategy', choices=['paragraph', 'sentence', 'page', 'chapter'],
                       default='paragraph', help='Chunking strategy (default: paragraph)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Maximum chunk size in characters (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help='Overlap between chunks in characters (default: 200)')
    parser.add_argument('--language', required=True, help='Language code for processing (e.g., en, de, fr) - MANDATORY for consistent processing')
    parser.add_argument('--use-process-result', help='Skip processing and use existing process result file (e.g., my-file.process_result.json)')
    parser.add_argument('--use-ingestion-data', help='Skip processing and ingestion calculation, use existing ingestion data file (e.g., my-file.ingest_data.json)')

    args = parser.parse_args()

    document_file = Path(args.document_file)

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

    # Validate database flags require ingestion mode
    if (args.qdrant or args.neo4j) and not args.ingest:
        print(f"[FAIL] Error: --neo4j and --qdrant flags require --ingest mode")
        print("To store entities and relations in databases, use: --ingest --neo4j --qdrant")
        sys.exit(1)

    # Handle resume arguments
    from resume_utils import handle_resume_arguments
    handle_resume_arguments(args, str(document_file), 'document', metadata)

    try:
        if args.ingest:
            # Ingestion mode
            success = asyncio.run(test_document_ingestion(
                document_file,
                webhook_url=args.webhook_url,
                metadata=metadata,
                chunking_strategy=args.chunking_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                use_qdrant=args.qdrant,
                use_neo4j=args.neo4j,
                qdrant_collection_name=args.qdrant_collection,
                neo4j_database_name=args.neo4j_database,
                language=args.language  # Pass language parameter
            ))
            if success:
                print("\n[SUCCESS] Document ingestion test completed successfully!")
                print("💡 Check the .ingest_result.json and .ingest_data.json files for details.")
                sys.exit(0)
            else:
                print("\n[ERROR] Document ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
            success = asyncio.run(test_document_processing(
                document_file,
                chunking_strategy=args.chunking_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            ))
            if success:
                print("\n[SUCCESS] Document processing test completed successfully!")
                sys.exit(0)
            else:
                print("\n[ERROR] Document processing test failed!")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n[STOP]  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        try:
            print(f"\n[FAIL] Fatal error: {e}")
        except UnicodeEncodeError:
            print(f"\n[FATAL ERROR]: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
