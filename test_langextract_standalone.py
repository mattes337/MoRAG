#!/usr/bin/env python3
"""Test the standalone LangExtract wrapper implementation."""

import os
import sys
import asyncio
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

async def test_entity_extractor():
    """Test standalone LangExtract entity extractor."""
    try:
        from morag_graph.extraction.langextract_standalone import LangExtractEntityExtractor
        
        print("Testing Standalone LangExtract Entity Extractor...")
        
        # Initialize extractor
        extractor = LangExtractEntityExtractor(
            min_confidence=0.5,
            chunk_size=1000
        )
        
        # Test text
        text = "Dr. Alice Johnson works as a researcher at Microsoft in Seattle, Washington."
        
        print(f"Input text: {text}")
        
        # Check if API key is available
        if not extractor.api_key:
            print("⚠ No API key found. Testing initialization only.")
            print("✓ Standalone LangExtract Entity Extractor initialized successfully")
            return True
        
        # Extract entities
        entities = await extractor.extract(text, source_doc_id="test_doc")
        
        print(f"✓ Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity['type']}: '{entity['name']}' (confidence: {entity['confidence']:.2f})")
            if entity['attributes']:
                print(f"    Attributes: {entity['attributes']}")
        
        # Verify entities are dictionaries with expected keys
        for entity in entities:
            assert 'name' in entity, "Entity should have name"
            assert 'type' in entity, "Entity should have type"
            assert 'confidence' in entity, "Entity should have confidence"
            assert 'attributes' in entity, "Entity should have attributes"
            assert 'source_doc_id' in entity, "Entity should have source_doc_id"
        
        return True
        
    except Exception as e:
        print(f"✗ Entity extractor test failed: {e}")
        return False

async def test_relation_extractor():
    """Test standalone LangExtract relation extractor."""
    try:
        from morag_graph.extraction.langextract_standalone import LangExtractRelationExtractor
        
        print("\nTesting Standalone LangExtract Relation Extractor...")
        
        # Initialize extractor
        extractor = LangExtractRelationExtractor(
            min_confidence=0.5,
            chunk_size=1000
        )
        
        # Test text and mock entities
        text = "Dr. Alice Johnson works as a researcher at Microsoft in Seattle, Washington."
        
        # Create mock entities
        class MockEntity:
            def __init__(self, name, entity_type, entity_id):
                self.name = name
                self.type = entity_type
                self.id = entity_id
        
        entities = [
            MockEntity("Dr. Alice Johnson", "PERSON", "person_1"),
            MockEntity("Microsoft", "ORGANIZATION", "org_1"),
            MockEntity("Seattle", "LOCATION", "loc_1"),
            MockEntity("Washington", "LOCATION", "loc_2"),
        ]
        
        print(f"Input text: {text}")
        print(f"Known entities: {[e.name for e in entities]}")
        
        # Check if API key is available
        if not extractor.api_key:
            print("⚠ No API key found. Testing initialization only.")
            print("✓ Standalone LangExtract Relation Extractor initialized successfully")
            return True
        
        # Extract relations
        relations = await extractor.extract(text, entities=entities, source_doc_id="test_doc")
        
        print(f"✓ Extracted {len(relations)} relations:")
        for relation in relations:
            print(f"  - {relation['type']}: {relation['source_entity_id']} -> {relation['target_entity_id']}")
            print(f"    Context: '{relation['context']}'")
            if relation['attributes']:
                print(f"    Attributes: {relation['attributes']}")
        
        # Verify relations are dictionaries with expected keys
        for relation in relations:
            assert 'source_entity_id' in relation, "Relation should have source_entity_id"
            assert 'target_entity_id' in relation, "Relation should have target_entity_id"
            assert 'type' in relation, "Relation should have type"
            assert 'context' in relation, "Relation should have context"
            assert 'confidence' in relation, "Relation should have confidence"
            assert 'attributes' in relation, "Relation should have attributes"
        
        return True
        
    except Exception as e:
        print(f"✗ Relation extractor test failed: {e}")
        return False

async def test_api_compatibility():
    """Test API compatibility with existing MoRAG interfaces."""
    try:
        from morag_graph.extraction.langextract_standalone import (
            LangExtractEntityExtractor, 
            LangExtractRelationExtractor
        )
        
        print("\nTesting API Compatibility...")
        
        # Test entity extractor methods
        entity_extractor = LangExtractEntityExtractor()
        
        # Check required methods exist
        assert hasattr(entity_extractor, 'extract'), "Entity extractor should have extract method"
        assert hasattr(entity_extractor, 'get_system_prompt'), "Entity extractor should have get_system_prompt method"
        assert hasattr(entity_extractor, '__aenter__'), "Entity extractor should support async context manager"
        assert hasattr(entity_extractor, '__aexit__'), "Entity extractor should support async context manager"
        
        # Test relation extractor methods
        relation_extractor = LangExtractRelationExtractor()
        
        # Check required methods exist
        assert hasattr(relation_extractor, 'extract'), "Relation extractor should have extract method"
        assert hasattr(relation_extractor, 'get_system_prompt'), "Relation extractor should have get_system_prompt method"
        
        # Test initialization parameters (should not raise errors)
        entity_extractor_custom = LangExtractEntityExtractor(
            min_confidence=0.7,
            chunk_size=2000,
            dynamic_types=True,
            entity_types={"person": "Human beings", "organization": "Companies and institutions"},
            language="en",
            enable_normalization=True
        )
        
        relation_extractor_custom = LangExtractRelationExtractor(
            min_confidence=0.7,
            chunk_size=2000,
            dynamic_types=True,
            relation_types={"works_for": "Employment relationship", "located_in": "Location relationship"},
            language="en"
        )
        
        print("✓ All required methods and parameters are available")
        print("✓ API compatibility maintained with existing MoRAG interfaces")
        
        return True
        
    except Exception as e:
        print(f"✗ API compatibility test failed: {e}")
        return False

async def test_prompt_generation():
    """Test that prompts can be generated."""
    try:
        from morag_graph.extraction.langextract_standalone import (
            LangExtractEntityExtractor,
            LangExtractRelationExtractor
        )
        
        print("\nTesting Prompt Generation...")
        
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

async def main():
    """Run all tests."""
    print("Testing Standalone LangExtract Wrapper...")
    print("=" * 60)
    
    tests = [
        test_entity_extractor,
        test_relation_extractor,
        test_api_compatibility,
        test_prompt_generation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print("-" * 40)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Standalone LangExtract wrapper is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
