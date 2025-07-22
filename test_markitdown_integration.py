#!/usr/bin/env python3
"""Test markitdown integration end-to-end."""

import asyncio
import tempfile
from pathlib import Path
from morag_document.processor import DocumentProcessor

async def test_end_to_end():
    """Test end-to-end document processing with markitdown."""
    processor = DocumentProcessor()
    
    # Create a simple test markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write('# Test Document\n\nThis is a test document with **bold** text and *italic* text.\n\n## Section 2\n\n- Item 1\n- Item 2\n- Item 3')
        test_file = f.name
    
    try:
        # Process the file
        result = await processor.process_file(test_file)
        
        print('Processing result:')
        print(f'  Success: {result.success}')
        print(f'  Processing time: {result.processing_time:.3f}s')
        print(f'  Document chunks: {len(result.document.chunks) if result.document else 0}')
        print(f'  Quality score: {result.metadata.get("quality_score", "N/A")}')
        print(f'  Raw text length: {len(result.document.raw_text) if result.document else 0}')
        
        if result.document and result.document.raw_text:
            print(f'  First 100 chars: {result.document.raw_text[:100]}...')
            
        # Test format detection
        print('\nFormat support verification:')
        test_formats = ['pdf', 'docx', 'xlsx', 'pptx', 'txt', 'jpg', 'mp3', 'zip']
        for fmt in test_formats:
            supported = await processor.supports_format(fmt)
            converter = processor.converters.get(fmt)
            converter_name = converter.__class__.__name__ if converter else 'None'
            print(f'  {fmt}: {supported} ({converter_name})')
            
    finally:
        # Clean up
        Path(test_file).unlink()

if __name__ == '__main__':
    asyncio.run(test_end_to_end())
