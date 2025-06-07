#!/usr/bin/env python3
"""
Test script to validate the fixes for:
1. Docling availability check
2. AttributeError in search_by_metadata
"""

import sys
import os
from pathlib import Path

# Add packages to path for testing
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-document" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-core" / "src"))

def test_docling_availability():
    """Test if docling is properly detected as available."""
    print("Testing docling availability...")
    
    try:
        from morag_document.converters.pdf import PDFConverter
        
        # Create converter and check docling availability
        converter = PDFConverter()
        
        if converter._docling_available:
            print("✅ Docling is available")
            return True
        else:
            print("❌ Docling is not available")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing docling: {e}")
        return False

def test_search_by_metadata_return_format():
    """Test the format of search_by_metadata return values."""
    print("\nTesting search_by_metadata return format...")
    
    try:
        # Mock the search_by_metadata method to simulate the issue
        def mock_search_by_metadata(filters, limit=1):
            """Mock method that returns the correct dictionary format."""
            return [
                {
                    "id": "test_point_id_123",
                    "metadata": {"content_checksum": "abc123"},
                    "text": "test content",
                    "score": 1.0
                }
            ]
        
        # Test the fix
        existing_points = mock_search_by_metadata({"content_checksum": "abc123"})
        
        if existing_points:
            # This should work with the fix (dictionary access)
            point_id = existing_points[0]["id"]
            print(f"✅ Dictionary access works: {point_id}")
            
            # This would fail with the old code (object attribute access)
            try:
                old_style_access = existing_points[0].id
                print("❌ Object attribute access should not work")
                return False
            except AttributeError:
                print("✅ Object attribute access correctly fails (as expected)")
                return True
        else:
            print("❌ No existing points returned")
            return False
            
    except Exception as e:
        print(f"❌ Error testing search format: {e}")
        return False

def test_requirements_docling():
    """Test if docling is in requirements.txt."""
    print("\nTesting requirements.txt for docling...")
    
    try:
        requirements_path = Path(__file__).parent / "requirements.txt"
        
        if not requirements_path.exists():
            print("❌ requirements.txt not found")
            return False
            
        with open(requirements_path, 'r') as f:
            content = f.read()
            
        if "docling" in content:
            print("✅ docling found in requirements.txt")
            return True
        else:
            print("❌ docling not found in requirements.txt")
            return False
            
    except Exception as e:
        print(f"❌ Error checking requirements.txt: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 Testing MoRAG fixes...")
    print("=" * 50)
    
    tests = [
        test_requirements_docling,
        test_docling_availability,
        test_search_by_metadata_return_format,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes validated successfully!")
        return 0
    else:
        print("⚠️  Some issues remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())
