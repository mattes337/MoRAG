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
    from morag_services import ServiceConfig, ContentType
    from morag_core.models import ProcessingConfig
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
        # Initialize configuration
        config = ServiceConfig()
        print_result("Configuration", "✅ Loaded successfully")

        # Initialize document processor
        processor = DocumentProcessor(config)
        print_result("Document Processor", "✅ Initialized successfully")

        # Create processing configuration
        processing_config = ProcessingConfig(
            max_file_size=100 * 1024 * 1024,  # 100MB
            timeout=300.0,
            extract_metadata=True
        )
        print_result("Processing Config", "✅ Created successfully")
        
        print_section("Processing Document File")
        print("🔄 Starting document processing...")
        
        # Process the document file
        result = await processor.process_file(document_file, processing_config)
        
        if result.success:
            print("✅ Document processing completed successfully!")
            
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
                content_preview = result.content[:1000] + "..." if len(result.content) > 1000 else result.content
                print(f"📄 Content ({len(result.content)} characters):")
                print(content_preview)
                
                # Count pages if available
                page_count = result.content.count("## Page ")
                if page_count > 0:
                    print_result("Pages Detected", str(page_count))
            
            if result.summary:
                print_section("Summary")
                print(f"📝 {result.summary}")
            
            # Save results to file
            output_file = document_file.parent / f"{document_file.stem}_test_result.json"
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
            
            # Also save markdown content
            markdown_file = document_file.parent / f"{document_file.stem}_converted.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            print_section("Output")
            print_result("Results saved to", str(output_file))
            print_result("Markdown saved to", str(markdown_file))
            
            return True
            
        else:
            print("❌ Document processing failed!")
            print_result("Error", result.error or "Unknown error")
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
