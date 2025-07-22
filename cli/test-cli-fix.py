#!/usr/bin/env python3
"""
Test script to verify CLI functionality and fix common issues.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        import morag_core
        print("✅ morag_core imported successfully")
    except Exception as e:
        print(f"❌ morag_core import failed: {e}")
        return False
    
    try:
        import morag_document
        print("✅ morag_document imported successfully")
    except Exception as e:
        print(f"❌ morag_document import failed: {e}")
        return False
    
    try:
        import morag_audio
        print("✅ morag_audio imported successfully")
    except Exception as e:
        print(f"❌ morag_audio import failed: {e}")
        return False
    
    try:
        import morag_video
        print("✅ morag_video imported successfully")
    except Exception as e:
        print(f"❌ morag_video import failed: {e}")
        return False
    
    try:
        import morag_image
        print("✅ morag_image imported successfully")
    except Exception as e:
        print(f"❌ morag_image import failed: {e}")
        return False
    
    return True

def test_document_processor():
    """Test document processor initialization."""
    print("\nTesting document processor...")
    
    try:
        from morag_document import DocumentProcessor
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized successfully")
        print(f"   Supported formats: {len(processor.converters)} converters")
        return True
    except Exception as e:
        print(f"❌ DocumentProcessor initialization failed: {e}")
        traceback.print_exc()
        return False

def test_markitdown():
    """Test markitdown availability."""
    print("\nTesting markitdown...")
    
    try:
        import markitdown
        print("✅ markitdown imported successfully")
        
        # Test basic functionality
        md = markitdown.MarkItDown()
        print("✅ MarkItDown instance created successfully")
        return True
    except Exception as e:
        print(f"❌ markitdown test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("  CLI Functionality Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test document processor
    if not test_document_processor():
        all_passed = False
    
    # Test markitdown
    if not test_markitdown():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! CLI functionality is working.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
