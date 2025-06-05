#!/usr/bin/env python3
"""
MoRAG Document Processing Test Script

Usage: python test-document.py <document_file>

Examples:
    python test-document.py my-document.pdf
    python test-document.py presentation.pptx
    python test-document.py spreadsheet.xlsx
    python test-document.py document.docx
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


async def test_document_processing(document_file: Path) -> bool:
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

        print_section("Processing Document File")
        print("🔄 Starting document processing...")

        # Process the document file with options
        result = await processor.process_file(
            document_file,
            extract_metadata=True,
            chunking_strategy="paragraph",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        if result.success:
            print("✅ Document processing completed successfully!")
            
            print_section("Processing Results")
            print_result("Status", "✅ Success")
            # Content type not available in ProcessingResult
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
            
            # Summary not available in ProcessingResult
            
            # Save results to file
            output_file = document_file.parent / f"{document_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'success': result.success,
                    'processing_time': result.processing_time,
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


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test-document.py <document_file>")
        print()
        print("Examples:")
        print("  python test-document.py my-document.pdf")
        print("  python test-document.py presentation.pptx")
        print("  python test-document.py spreadsheet.xlsx")
        print("  python test-document.py document.docx")
        sys.exit(1)
    
    document_file = Path(sys.argv[1])
    
    try:
        success = asyncio.run(test_document_processing(document_file))
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
