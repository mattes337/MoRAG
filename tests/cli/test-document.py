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
from pathlib import Path
from typing import Optional, Dict, Any
import requests

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_document import DocumentProcessor
    from morag_core.interfaces.processor import ProcessingConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-document")
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


async def test_document_processing(document_file: Path, chunking_strategy: str = "paragraph",
                                  chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
    """Test document processing functionality."""
    print_header("MoRAG Document Processing Test")

    if not document_file.exists():
        print(f"❌ Error: Document file not found: {document_file}")
        return False

    print_result("Input File", str(document_file))
    print_result("File Size", f"{document_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", document_file.suffix.lower())

    try:
        # Initialize document processor (no config needed)
        processor = DocumentProcessor()
        print_result("Document Processor", "✅ Initialized successfully")
        print_result("Chunking Strategy", chunking_strategy)
        print_result("Chunk Size", f"{chunk_size} characters")
        print_result("Chunk Overlap", f"{chunk_overlap} characters")

        print_section("Processing Document File")
        print("🔄 Starting document processing...")

        # Process the document file with options
        result = await processor.process_file(
            document_file,
            extract_metadata=True,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if result.success:
            print("✅ Document processing completed successfully!")

            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")

            if result.metadata:
                print_section("Metadata")
                for key, value in result.metadata.items():
                    if isinstance(value, (dict, list)):
                        print_result(key, json.dumps(value, indent=2))
                    else:
                        print_result(key, str(value))

            if result.document:
                print_section("Document Information")
                doc = result.document
                print_result("Title", doc.metadata.title or "N/A")
                print_result("Author", doc.metadata.author or "N/A")
                print_result("Page Count", str(doc.metadata.page_count or "N/A"))
                print_result("Word Count", str(doc.metadata.word_count or "N/A"))
                print_result("Chunks Count", str(len(doc.chunks)))

                if doc.raw_text:
                    print_section("Content Preview")
                    content_preview = doc.raw_text[:500] + "..." if len(doc.raw_text) > 500 else doc.raw_text
                    print(f"📄 Raw Text ({len(doc.raw_text)} characters):")
                    print(content_preview)

                if doc.chunks:
                    print_section("Chunks Preview (first 3)")
                    for i, chunk in enumerate(doc.chunks[:3]):
                        print(f"  Chunk {i+1}:")
                        print(f"    Content: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")
                        if chunk.page_number:
                            print(f"    Page: {chunk.page_number}")
                        if chunk.metadata:
                            print(f"    Metadata: {chunk.metadata}")

            # Save results to file
            output_file = document_file.parent / f"{document_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'processing',
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'chunking_strategy': chunking_strategy,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'metadata': result.metadata,
                    'document': {
                        'title': result.document.metadata.title if result.document else None,
                        'author': result.document.metadata.author if result.document else None,
                        'page_count': result.document.metadata.page_count if result.document else None,
                        'word_count': result.document.metadata.word_count if result.document else None,
                        'raw_text': result.document.raw_text if result.document else None,
                        'chunks_count': len(result.document.chunks) if result.document else 0,
                        'chunks': [
                            {
                                'content': chunk.content,
                                'page_number': chunk.page_number,
                                'metadata': chunk.metadata
                            } for chunk in result.document.chunks
                        ] if result.document else []
                    },
                    'error_message': result.error_message
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Results saved to", str(output_file))

            return True

        else:
            print("❌ Document processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_document_ingestion(document_file: Path, webhook_url: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Test document ingestion functionality."""
    print_header("MoRAG Document Ingestion Test")

    if not document_file.exists():
        print(f"❌ Error: Document file not found: {document_file}")
        return False

    print_result("Input File", str(document_file))
    print_result("File Size", f"{document_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", document_file.suffix.lower())
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")

    try:
        print_section("Submitting Ingestion Task")
        print("🔄 Starting document ingestion...")

        # Prepare form data
        files = {'file': open(document_file, 'rb')}
        data = {'source_type': 'document'}

        if webhook_url:
            data['webhook_url'] = webhook_url
        if metadata:
            data['metadata'] = json.dumps(metadata)

        # Submit to ingestion API
        response = requests.post(
            'http://localhost:8000/api/v1/ingest/file',
            files=files,
            data=data,
            timeout=30
        )

        files['file'].close()

        if response.status_code == 200:
            result = response.json()
            print("✅ Document ingestion task submitted successfully!")

            print_section("Ingestion Results")
            print_result("Status", "✅ Success")
            print_result("Task ID", result.get('task_id', 'Unknown'))
            print_result("Message", result.get('message', 'Task created'))
            print_result("Estimated Time", f"{result.get('estimated_time', 'Unknown')} seconds")

            # Save ingestion result
            output_file = document_file.parent / f"{document_file.stem}_ingest_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'ingestion',
                    'task_id': result.get('task_id'),
                    'status': result.get('status'),
                    'message': result.get('message'),
                    'estimated_time': result.get('estimated_time'),
                    'webhook_url': webhook_url,
                    'metadata': metadata,
                    'file_path': str(document_file)
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Ingestion result saved to", str(output_file))
            print_result("Monitor task status", f"curl http://localhost:8000/api/v1/status/{result.get('task_id')}")

            return True
        else:
            print("❌ Document ingestion failed!")
            print_result("Status Code", str(response.status_code))
            print_result("Error", response.text)
            return False

    except Exception as e:
        print(f"❌ Error during document ingestion: {e}")
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
        """
    )

    parser.add_argument('document_file', help='Path to document file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--chunking-strategy', choices=['paragraph', 'sentence', 'page', 'chapter'],
                       default='paragraph', help='Chunking strategy (default: paragraph)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Maximum chunk size in characters (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help='Overlap between chunks in characters (default: 200)')

    args = parser.parse_args()

    document_file = Path(args.document_file)

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
            success = asyncio.run(test_document_ingestion(
                document_file,
                webhook_url=args.webhook_url,
                metadata=metadata
            ))
            if success:
                print("\n🎉 Document ingestion test completed successfully!")
                print("💡 Use the task ID to monitor progress and retrieve results.")
                sys.exit(0)
            else:
                print("\n💥 Document ingestion test failed!")
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
                print("\n🎉 Document processing test completed successfully!")
                sys.exit(0)
            else:
                print("\n💥 Document processing test failed!")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
