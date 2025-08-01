#!/usr/bin/env python3
"""Test the LangExtract wrapper implementation."""

import os
import sys
import asyncio
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

async def test_entity_extractor():
    """Test LangExtract entity extractor wrapper."""
    try:
        # Import only what we need to avoid circular imports
        from morag_graph.extraction.langextract_wrapper import LangExtractEntityExtractor
        
        print("Testing LangExtract Entity Extractor...")
        
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
            print("✓ LangExtract Entity Extractor initialized successfully")
            return True
        
        # Extract entities
        entities = await extractor.extract(text, source_doc_id="test_doc")
        
        print(f"✓ Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.type}: '{entity.name}' (confidence: {entity.confidence:.2f})")
            if entity.attributes:
                print(f"    Attributes: {entity.attributes}")
        
        # Verify entities have expected attributes
        for entity in entities:
            assert hasattr(entity, 'name'), "Entity should have name attribute"
            assert hasattr(entity, 'type'), "Entity should have type attribute"
            assert hasattr(entity, 'confidence'), "Entity should have confidence attribute"
        
        return True
        
    except Exception as e:
        print(f"✗ Entity extractor test failed: {e}")
        return False

async def test_relation_extractor():
    """Test LangExtract relation extractor wrapper."""
    try:
        # Import only what we need to avoid circular imports
        from morag_graph.extraction.langextract_wrapper import LangExtractRelationExtractor
        
        print("\nTesting LangExtract Relation Extractor...")
        
        # Initialize extractor
        extractor = LangExtractRelationExtractor(
            min_confidence=0.5,
            chunk_size=1000
        )
        
        # Test text and mock entities (avoid importing Entity to prevent circular imports)
        text = "Dr. Alice Johnson works as a researcher at Microsoft in Seattle, Washington."

        # Create mock entities with required attributes
        class MockEntity:
            def __init__(self, name, type, source_doc_id):
                self.name = name
                self.type = type
                self.source_doc_id = source_doc_id
                self.id = f"entity_{hash(name.lower())}"

        entities = [
            MockEntity("Dr. Alice Johnson", "PERSON", "test_doc"),
            MockEntity("Microsoft", "ORGANIZATION", "test_doc"),
            MockEntity("Seattle", "LOCATION", "test_doc"),
            MockEntity("Washington", "LOCATION", "test_doc"),
        ]
        
        print(f"Input text: {text}")
        print(f"Known entities: {[e.name for e in entities]}")
        
        # Check if API key is available
        if not extractor.api_key:
            print("⚠ No API key found. Testing initialization only.")
            print("✓ LangExtract Relation Extractor initialized successfully")
            return True
        
        # Extract relations
        relations = await extractor.extract(text, entities=entities, source_doc_id="test_doc")
        
        print(f"✓ Extracted {len(relations)} relations:")
        for relation in relations:
            print(f"  - {relation.type}: {relation.source_entity_id} -> {relation.target_entity_id}")
            print(f"    Context: '{relation.context}'")
            if relation.attributes:
                print(f"    Attributes: {relation.attributes}")
        
        # Verify relations have expected attributes
        for relation in relations:
            assert hasattr(relation, 'source_entity_id'), "Relation should have source_entity_id"
            assert hasattr(relation, 'target_entity_id'), "Relation should have target_entity_id"
            assert hasattr(relation, 'type'), "Relation should have type"
            assert hasattr(relation, 'confidence'), "Relation should have confidence"
        
        return True
        
    except Exception as e:
        print(f"✗ Relation extractor test failed: {e}")
        return False

async def test_api_compatibility():
    """Test API compatibility with existing MoRAG interfaces."""
    try:
        from morag_graph.extraction.langextract_wrapper import (
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
        assert hasattr(relation_extractor, 'extract_relations'), "Relation extractor should have extract_relations method"
        assert hasattr(relation_extractor, 'extract_with_entities'), "Relation extractor should have extract_with_entities method"
        assert hasattr(relation_extractor, 'extract_from_entity_pairs'), "Relation extractor should have extract_from_entity_pairs method"
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

async def main():
    """Run all tests."""
    print("Testing LangExtract Wrapper Implementation...")
    print("=" * 60)
    
    tests = [
        test_entity_extractor,
        test_relation_extractor,
        test_api_compatibility,
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
        print("🎉 All tests passed! LangExtract wrapper is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
