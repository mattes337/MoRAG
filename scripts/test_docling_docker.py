#!/usr/bin/env python3
"""
Test script to validate Docling functionality in Docker environment.
This script tests both CPU-only and regular Docling configurations.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the packages to the Python path
sys.path.insert(0, '/app/packages/morag-document/src')
sys.path.insert(0, '/app/packages/morag-core/src')

from morag_document.converters.pdf import PDFConverter
from morag_core.interfaces.document import Document, DocumentMetadata
from morag_core.interfaces.processor import ConversionOptions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_docling_availability():
    """Test if Docling is available and working."""
    logger.info("Testing Docling availability...")
    
    # Create PDF converter
    converter = PDFConverter()
    
    # Check if docling is available
    docling_available = converter._docling_available
    logger.info(f"Docling available: {docling_available}")
    
    # Print environment variables
    force_cpu = os.environ.get('MORAG_FORCE_CPU', 'false')
    disable_docling = os.environ.get('MORAG_DISABLE_DOCLING', 'false')
    logger.info(f"MORAG_FORCE_CPU: {force_cpu}")
    logger.info(f"MORAG_DISABLE_DOCLING: {disable_docling}")
    
    return docling_available

def test_docling_import():
    """Test direct Docling import and basic functionality."""
    logger.info("Testing direct Docling import...")
    
    try:
        import docling
        logger.info(f"Docling version: {docling.__version__ if hasattr(docling, '__version__') else 'unknown'}")
        
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        
        # Test basic initialization
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        
        converter = DocumentConverter()
        logger.info("Docling DocumentConverter initialized successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import Docling: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Docling: {e}")
        return False

def test_pytorch_compatibility():
    """Test PyTorch compatibility in CPU mode."""
    logger.info("Testing PyTorch compatibility...")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
        
        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        logger.info(f"Basic tensor operation successful: {y}")
        
        return True
        
    except Exception as e:
        logger.error(f"PyTorch compatibility test failed: {e}")
        return False

def create_test_pdf():
    """Create a simple test PDF for testing."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        temp_file.close()
        
        # Create PDF content
        c = canvas.Canvas(temp_path, pagesize=letter)
        c.drawString(100, 750, "Test Document")
        c.drawString(100, 700, "This is a test PDF document for Docling validation.")
        c.drawString(100, 650, "It contains simple text content.")
        c.showPage()
        c.save()
        
        logger.info(f"Created test PDF: {temp_path}")
        return temp_path
        
    except ImportError:
        logger.warning("reportlab not available, cannot create test PDF")
        return None
    except Exception as e:
        logger.error(f"Failed to create test PDF: {e}")
        return None

def test_pdf_processing(pdf_path):
    """Test PDF processing with the converter."""
    if not pdf_path or not os.path.exists(pdf_path):
        logger.warning("No test PDF available for processing test")
        return False
    
    logger.info(f"Testing PDF processing with: {pdf_path}")
    
    try:
        # Create converter and document
        converter = PDFConverter()
        document = Document(
            content="",
            metadata=DocumentMetadata(
                source_type="document",
                file_path=pdf_path,
                file_name=os.path.basename(pdf_path)
            )
        )
        
        # Create conversion options
        options = ConversionOptions(
            chunking_strategy="page",
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Test conversion (async function, so we need to handle it properly)
        import asyncio
        
        async def run_conversion():
            return await converter._extract_text(Path(pdf_path), document, options)
        
        # Run the conversion
        result = asyncio.run(run_conversion())
        
        logger.info(f"PDF processing successful")
        logger.info(f"Extracted text length: {len(result.raw_text) if result.raw_text else 0}")
        logger.info(f"Page count: {result.metadata.page_count}")
        logger.info(f"Word count: {result.metadata.word_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return False
    finally:
        # Clean up test file
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
                logger.info("Cleaned up test PDF")
            except Exception as e:
                logger.warning(f"Failed to clean up test PDF: {e}")

def main():
    """Main test function."""
    logger.info("Starting Docling Docker validation tests...")
    
    # Test results
    results = {}
    
    # Test 1: PyTorch compatibility
    results['pytorch'] = test_pytorch_compatibility()
    
    # Test 2: Docling import
    results['docling_import'] = test_docling_import()
    
    # Test 3: Docling availability through converter
    results['docling_availability'] = test_docling_availability()
    
    # Test 4: PDF processing (if possible)
    test_pdf = create_test_pdf()
    results['pdf_processing'] = test_pdf_processing(test_pdf)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("DOCLING DOCKER VALIDATION SUMMARY")
    logger.info("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    logger.info(f"\nOverall Status: {overall_status}")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
