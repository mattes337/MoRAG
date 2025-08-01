#!/usr/bin/env python3
"""Test domain-specific examples for LangExtract."""

import os
import sys
import asyncio
from pathlib import Path

# Add the specific file to the path without importing the package
wrapper_path = Path(__file__).parent / "packages" / "morag-graph" / "src" / "morag_graph" / "extraction"
sys.path.insert(0, str(wrapper_path))

async def test_domain_examples_import():
    """Test that domain examples can be imported."""
    try:
        import langextract_examples
        
        print("✓ Domain examples module imported successfully")
        
        # Test that classes exist
        assert hasattr(langextract_examples, 'LangExtractExamples'), "Should have LangExtractExamples"
        assert hasattr(langextract_examples, 'DomainEntityTypes'), "Should have DomainEntityTypes"
        assert hasattr(langextract_examples, 'DomainRelationTypes'), "Should have DomainRelationTypes"
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import domain examples: {e}")
        return False

async def test_entity_examples():
    """Test entity examples for different domains."""
    try:
        import langextract_examples
        
        domains = ["general", "medical", "technical", "legal", "business", "scientific"]
        
        for domain in domains:
            examples = langextract_examples.LangExtractExamples.get_entity_examples(domain)
            print(f"✓ {domain.capitalize()} domain: {len(examples)} entity examples")
            
            # Verify examples have the expected structure
            for example in examples:
                assert hasattr(example, 'text'), f"{domain} example should have text"
                assert hasattr(example, 'extractions'), f"{domain} example should have extractions"
                assert len(example.extractions) > 0, f"{domain} example should have at least one extraction"
                
                for extraction in example.extractions:
                    assert hasattr(extraction, 'extraction_class'), "Extraction should have extraction_class"
                    assert hasattr(extraction, 'extraction_text'), "Extraction should have extraction_text"
                    assert hasattr(extraction, 'attributes'), "Extraction should have attributes"
        
        return True
        
    except Exception as e:
        print(f"✗ Entity examples test failed: {e}")
        return False

async def test_relation_examples():
    """Test relation examples for different domains."""
    try:
        import langextract_examples
        
        domains = ["general", "medical", "technical", "legal", "business", "scientific"]
        
        for domain in domains:
            examples = langextract_examples.LangExtractExamples.get_relation_examples(domain)
            print(f"✓ {domain.capitalize()} domain: {len(examples)} relation examples")
            
            # Verify examples have the expected structure
            for example in examples:
                assert hasattr(example, 'text'), f"{domain} example should have text"
                assert hasattr(example, 'extractions'), f"{domain} example should have extractions"
                assert len(example.extractions) > 0, f"{domain} example should have at least one extraction"
                
                for extraction in example.extractions:
                    assert hasattr(extraction, 'extraction_class'), "Extraction should have extraction_class"
                    assert hasattr(extraction, 'extraction_text'), "Extraction should have extraction_text"
                    assert hasattr(extraction, 'attributes'), "Extraction should have attributes"
                    
                    # Relation extractions should have source and target entities
                    attrs = extraction.attributes
                    if attrs:
                        # Not all relation examples may have these, but many should
                        has_source = 'source_entity' in attrs
                        has_target = 'target_entity' in attrs
                        has_relationship = 'relationship_type' in attrs
                        
                        if has_source or has_target or has_relationship:
                            print(f"    Relation: {attrs.get('relationship_type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Relation examples test failed: {e}")
        return False

async def test_domain_types():
    """Test domain-specific entity and relation types."""
    try:
        import langextract_examples
        
        # Test entity types
        entity_types = langextract_examples.DomainEntityTypes
        domains = ["GENERAL", "MEDICAL", "TECHNICAL", "LEGAL", "BUSINESS", "SCIENTIFIC"]
        
        for domain in domains:
            types_dict = getattr(entity_types, domain)
            assert isinstance(types_dict, dict), f"{domain} entity types should be a dict"
            assert len(types_dict) > 0, f"{domain} should have entity types"
            print(f"✓ {domain.lower().capitalize()} entity types: {len(types_dict)} types")
            
            # Verify structure
            for type_name, description in types_dict.items():
                assert isinstance(type_name, str), "Type name should be string"
                assert isinstance(description, str), "Type description should be string"
                assert len(description) > 0, "Type description should not be empty"
        
        # Test relation types
        relation_types = langextract_examples.DomainRelationTypes
        
        for domain in domains:
            types_dict = getattr(relation_types, domain)
            assert isinstance(types_dict, dict), f"{domain} relation types should be a dict"
            assert len(types_dict) > 0, f"{domain} should have relation types"
            print(f"✓ {domain.lower().capitalize()} relation types: {len(types_dict)} types")
            
            # Verify structure
            for type_name, description in types_dict.items():
                assert isinstance(type_name, str), "Type name should be string"
                assert isinstance(description, str), "Type description should be string"
                assert len(description) > 0, "Type description should not be empty"
        
        return True
        
    except Exception as e:
        print(f"✗ Domain types test failed: {e}")
        return False

async def test_standalone_with_domains():
    """Test standalone wrapper with different domains."""
    try:
        import langextract_standalone
        
        domains = ["general", "medical", "technical", "legal", "business", "scientific"]
        
        for domain in domains:
            # Test entity extractor
            entity_extractor = langextract_standalone.LangExtractEntityExtractor(domain=domain)
            assert entity_extractor.domain == domain, f"Entity extractor should have domain {domain}"
            
            # Test relation extractor
            relation_extractor = langextract_standalone.LangExtractRelationExtractor(domain=domain)
            assert relation_extractor.domain == domain, f"Relation extractor should have domain {domain}"
            
            print(f"✓ {domain.capitalize()} domain extractors initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Standalone domain test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Testing Domain-Specific Examples for LangExtract...")
    print("=" * 60)
    
    tests = [
        test_domain_examples_import,
        test_entity_examples,
        test_relation_examples,
        test_domain_types,
        test_standalone_with_domains,
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
        print("🎉 All tests passed! Domain-specific examples are ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
