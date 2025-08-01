#!/usr/bin/env python3
"""Isolated test for LangExtract wrapper without any package imports."""

import os
import sys
import asyncio
from pathlib import Path

# Add the specific file to the path without importing the package
wrapper_path = Path(__file__).parent / "packages" / "morag-graph" / "src" / "morag_graph" / "extraction"
sys.path.insert(0, str(wrapper_path))

async def test_langextract_import():
    """Test that LangExtract wrapper can be imported directly."""
    try:
        # Import the standalone module directly
        import langextract_standalone
        
        print("✓ LangExtract standalone module imported successfully")
        
        # Test that classes exist
        assert hasattr(langextract_standalone, 'LangExtractEntityExtractor'), "Should have LangExtractEntityExtractor"
        assert hasattr(langextract_standalone, 'LangExtractRelationExtractor'), "Should have LangExtractRelationExtractor"
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import LangExtract standalone module: {e}")
        return False

async def test_initialization():
    """Test that extractors can be initialized."""
    try:
        import langextract_standalone
        
        # Test entity extractor initialization
        entity_extractor = langextract_standalone.LangExtractEntityExtractor(
            min_confidence=0.7,
            chunk_size=1000,
            dynamic_types=True,
            entity_types={"person": "Human beings", "organization": "Companies"},
            language="en"
        )
        
        # Test relation extractor initialization
        relation_extractor = langextract_standalone.LangExtractRelationExtractor(
            min_confidence=0.7,
            chunk_size=1000,
            dynamic_types=True,
            relation_types={"works_for": "Employment", "located_in": "Location"},
            language="en"
        )
        
        print("✓ Both extractors initialized successfully")
        print(f"  Entity extractor model: {entity_extractor.model_id}")
        print(f"  Relation extractor model: {relation_extractor.model_id}")
        print(f"  API key available: {'Yes' if entity_extractor.api_key else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        return False

async def test_method_availability():
    """Test that required methods are available."""
    try:
        import langextract_standalone
        
        entity_extractor = langextract_standalone.LangExtractEntityExtractor()
        relation_extractor = langextract_standalone.LangExtractRelationExtractor()
        
        # Check entity extractor methods
        required_entity_methods = ['extract', 'get_system_prompt', '__aenter__', '__aexit__']
        for method in required_entity_methods:
            assert hasattr(entity_extractor, method), f"Entity extractor missing {method}"
        
        # Check relation extractor methods
        required_relation_methods = ['extract', 'get_system_prompt', '__aenter__', '__aexit__']
        for method in required_relation_methods:
            assert hasattr(relation_extractor, method), f"Relation extractor missing {method}"
        
        print("✓ All required methods are available")
        return True
        
    except Exception as e:
        print(f"✗ Method availability test failed: {e}")
        return False

async def test_prompt_generation():
    """Test that prompts can be generated."""
    try:
        import langextract_standalone
        
        entity_extractor = langextract_standalone.LangExtractEntityExtractor()
        relation_extractor = langextract_standalone.LangExtractRelationExtractor()
        
        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()
        
        assert isinstance(entity_prompt, str) and len(entity_prompt) > 0, "Entity prompt should be non-empty string"
        assert isinstance(relation_prompt, str) and len(relation_prompt) > 0, "Relation prompt should be non-empty string"
        
        print("✓ Prompts generated successfully")
        print(f"  Entity prompt length: {len(entity_prompt)} characters")
        print(f"  Relation prompt length: {len(relation_prompt)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt generation test failed: {e}")
        return False

async def test_async_context_manager():
    """Test async context manager functionality."""
    try:
        import langextract_standalone
        
        # Test entity extractor context manager
        async with langextract_standalone.LangExtractEntityExtractor() as entity_extractor:
            assert entity_extractor is not None
        
        # Test relation extractor context manager
        async with langextract_standalone.LangExtractRelationExtractor() as relation_extractor:
            assert relation_extractor is not None
        
        print("✓ Async context managers work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Async context manager test failed: {e}")
        return False

async def test_entity_extraction_mock():
    """Test entity extraction with mock data (no API call)."""
    try:
        import langextract_standalone
        
        print("\nTesting Entity Extraction (Mock)...")
        
        extractor = langextract_standalone.LangExtractEntityExtractor()
        
        # Test with empty text
        entities = await extractor.extract("")
        assert entities == [], "Empty text should return empty list"
        
        # Test without API key (should return empty list with warning)
        entities = await extractor.extract("Some text")
        assert entities == [], "No API key should return empty list"
        
        print("✓ Entity extraction mock tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Entity extraction mock test failed: {e}")
        return False

async def test_relation_extraction_mock():
    """Test relation extraction with mock data (no API call)."""
    try:
        import langextract_standalone
        
        print("\nTesting Relation Extraction (Mock)...")
        
        extractor = langextract_standalone.LangExtractRelationExtractor()
        
        # Test with empty text
        relations = await extractor.extract("")
        assert relations == [], "Empty text should return empty list"
        
        # Test without API key (should return empty list with warning)
        relations = await extractor.extract("Some text")
        assert relations == [], "No API key should return empty list"
        
        print("✓ Relation extraction mock tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Relation extraction mock test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Testing LangExtract Wrapper (Isolated)...")
    print("=" * 60)
    
    tests = [
        test_langextract_import,
        test_initialization,
        test_method_availability,
        test_prompt_generation,
        test_async_context_manager,
        test_entity_extraction_mock,
        test_relation_extraction_mock,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print("-" * 40)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LangExtract wrapper is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
