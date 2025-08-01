#!/usr/bin/env python3
"""Basic test to verify LangExtract installation and functionality."""

import os
import sys
import textwrap
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

def test_langextract_import():
    """Test that LangExtract can be imported."""
    try:
        import langextract as lx
        print("✓ LangExtract imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import LangExtract: {e}")
        return False

def test_langextract_basic_extraction():
    """Test basic LangExtract functionality."""
    try:
        import langextract as lx
        
        # Define a simple extraction task
        prompt = textwrap.dedent("""
            Extract entities and their types from the text.
            Focus on people, organizations, and locations.
            Use exact text for extractions.
        """).strip()
        
        # Provide a simple example
        examples = [
            lx.data.ExampleData(
                text="John Smith works at Google in Mountain View.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="John Smith",
                        attributes={"role": "employee"}
                    ),
                    lx.data.Extraction(
                        extraction_class="organization",
                        extraction_text="Google",
                        attributes={"type": "company"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Mountain View",
                        attributes={"type": "city"}
                    ),
                ]
            )
        ]
        
        # Test input
        input_text = "Alice Johnson is a researcher at Microsoft in Seattle."
        
        print("Testing basic extraction...")
        print(f"Input: {input_text}")
        
        # Check if API key is available
        api_key = os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠ No API key found. Set LANGEXTRACT_API_KEY or GOOGLE_API_KEY environment variable.")
            print("✓ LangExtract basic setup verified (API key needed for actual extraction)")
            return True
        
        # Run extraction with API key
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash",
            api_key=api_key
        )
        
        print(f"✓ Extraction completed successfully")
        print(f"  Found {len(result.extractions)} extractions")
        for extraction in result.extractions:
            print(f"  - {extraction.extraction_class}: '{extraction.extraction_text}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic extraction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing LangExtract installation and basic functionality...")
    print("=" * 60)
    
    tests = [
        test_langextract_import,
        test_langextract_basic_extraction,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LangExtract is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
