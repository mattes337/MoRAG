#!/usr/bin/env python3
"""Debug PDF processing."""

import asyncio
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-document" / "src"))

async def debug_pdf():
    """Debug PDF processing."""
    try:
        from morag_document.converters.pdf import PDFConverter
        from morag_core.interfaces.converter import ConversionOptions
        
        print("✅ Imports successful")
        
        # Initialize PDF converter
        converter = PDFConverter()
        print("✅ PDF converter initialized")
        
        # Test file
        pdf_file = Path("uploads/6e57bcfe_Start research.pdf")
        if not pdf_file.exists():
            print(f"❌ PDF file not found: {pdf_file}")
            return
        
        print(f"✅ PDF file found: {pdf_file}")
        
        # Check if format is supported
        supports = await converter.supports_format("pdf")
        print(f"✅ PDF format supported: {supports}")
        
        # Create options
        options = ConversionOptions()
        print("✅ Conversion options created")
        
        # Try to convert
        print("🔄 Starting PDF conversion...")
        result = await converter.convert(pdf_file, options)
        print("✅ PDF conversion completed!")
        
        print(f"📄 Success: {result.success}")
        print(f"📄 Content length: {len(result.content)}")
        print(f"📄 Document: {result.document}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_pdf())
