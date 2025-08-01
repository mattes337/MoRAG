#!/usr/bin/env python3
"""Simple test for LangExtract wrapper without circular imports."""

import os
import sys
import asyncio
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

def test_langextract_import():
    """Test that LangExtract wrapper can be imported."""
    try:
        from morag_graph.extraction.langextract_wrapper import (
            LangExtractEntityExtractor,
            LangExtractRelationExtractor
        )
        print("✓ LangExtract wrappers imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import LangExtract wrappers: {e}")
        return False

def test_initialization():
    """Test that extractors can be initialized."""
    try:
        from morag_graph.extraction.langextract_wrapper import (
            LangExtractEntityExtractor,
            LangExtractRelationExtractor
        )
        
        # Test entity extractor initialization
        entity_extractor = LangExtractEntityExtractor(
            min_confidence=0.7,
            chunk_size=1000,
            dynamic_types=True,
            entity_types={"person": "Human beings", "organization": "Companies"},
            language="en"
        )
        
        # Test relation extractor initialization
        relation_extractor = LangExtractRelationExtractor(
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

def test_method_availability():
    """Test that required methods are available."""
    try:
        from morag_graph.extraction.langextract_wrapper import (
            LangExtractEntityExtractor,
            LangExtractRelationExtractor
        )
        
        entity_extractor = LangExtractEntityExtractor()
        relation_extractor = LangExtractRelationExtractor()
        
        # Check entity extractor methods
        required_entity_methods = ['extract', 'get_system_prompt', '__aenter__', '__aexit__']
        for method in required_entity_methods:
            assert hasattr(entity_extractor, method), f"Entity extractor missing {method}"
        
        # Check relation extractor methods
        required_relation_methods = [
            'extract', 'extract_relations', 'extract_with_entities', 
            'extract_from_entity_pairs', 'get_system_prompt', '__aenter__', '__aexit__'
        ]
        for method in required_relation_methods:
            assert hasattr(relation_extractor, method), f"Relation extractor missing {method}"
        
        print("✓ All required methods are available")
        return True
        
    except Exception as e:
        print(f"✗ Method availability test failed: {e}")
        return False

def test_prompt_generation():
    """Test that prompts can be generated."""
    try:
        from morag_graph.extraction.langextract_wrapper import (
            LangExtractEntityExtractor,
            LangExtractRelationExtractor
        )
        
        entity_extractor = LangExtractEntityExtractor()
        relation_extractor = LangExtractRelationExtractor()
        
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
        from morag_graph.extraction.langextract_wrapper import (
            LangExtractEntityExtractor,
            LangExtractRelationExtractor
        )
        
        # Test entity extractor context manager
        async with LangExtractEntityExtractor() as entity_extractor:
            assert entity_extractor is not None
        
        # Test relation extractor context manager
        async with LangExtractRelationExtractor() as relation_extractor:
            assert relation_extractor is not None
        
        print("✓ Async context managers work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Async context manager test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Testing LangExtract Wrapper (Simple Tests)...")
    print("=" * 60)
    
    sync_tests = [
        test_langextract_import,
        test_initialization,
        test_method_availability,
        test_prompt_generation,
    ]
    
    async_tests = [
        test_async_context_manager,
    ]
    
    passed = 0
    total = len(sync_tests) + len(async_tests)
    
    # Run sync tests
    for test in sync_tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print("-" * 40)
    
    # Run async tests
    for test in async_tests:
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
