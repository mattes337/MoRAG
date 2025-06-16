#!/usr/bin/env python3
"""
Comprehensive test for JSON parsing edge cases in entity/relation extraction.
"""

import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor

def test_entity_extraction_edge_cases():
    """Test entity extraction with various malformed JSON responses."""
    print("Testing Entity Extraction Edge Cases...")
    print("=" * 50)
    
    extractor = EntityExtractor()
    
    test_cases = [
        # Case 1: Perfect JSON
        {
            "name": "Perfect JSON",
            "input": '[{"name": "Apple", "type": "ORG", "context": "tech company", "confidence": 0.9}]',
            "should_work": True
        },
        
        # Case 2: Trailing comma
        {
            "name": "Trailing comma",
            "input": '[{"name": "Apple", "type": "ORG", "context": "tech company", "confidence": 0.9,}]',
            "should_work": True
        },
        
        # Case 3: Unquoted property names
        {
            "name": "Unquoted property names",
            "input": '[{name: "Apple", type: "ORG", context: "tech company", confidence: 0.9}]',
            "should_work": True
        },
        
        # Case 4: Single quotes
        {
            "name": "Single quotes",
            "input": "[{'name': 'Apple', 'type': 'ORG', 'context': 'tech company', 'confidence': 0.9}]",
            "should_work": True
        },
        
        # Case 5: Mixed quotes
        {
            "name": "Mixed quotes",
            "input": '[{"name": \'Apple\', "type": \'ORG\', "context": \'tech company\', "confidence": 0.9}]',
            "should_work": True
        },
        
        # Case 6: With markdown
        {
            "name": "With markdown",
            "input": '```json\n[{"name": "Apple", "type": "ORG", "context": "tech company", "confidence": 0.9}]\n```',
            "should_work": True
        },
        
        # Case 7: With extra text
        {
            "name": "With extra text",
            "input": '[{"name": "Apple", "type": "ORG", "context": "tech company", "confidence": 0.9}]\n\nHere are the extracted entities.',
            "should_work": True
        },
        
        # Case 8: Multiple entities with issues
        {
            "name": "Multiple entities with issues",
            "input": '[{name: "Apple", type: "ORG",}, {"name": "Tim Cook", "type": "PERSON", "confidence": 0.8,}]',
            "should_work": True
        },
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Input: {test_case['input'][:100]}{'...' if len(test_case['input']) > 100 else ''}")
        
        try:
            entities = extractor.parse_response(test_case['input'], "test text")
            if test_case['should_work']:
                print(f"✅ Success: Found {len(entities)} entities")
                for entity in entities:
                    print(f"   - {entity.name} ({entity.type.value})")
            else:
                print(f"⚠️  Unexpected success: {len(entities)} entities")
        except Exception as e:
            if test_case['should_work']:
                print(f"❌ Failed: {e}")
            else:
                print(f"✅ Expected failure: {e}")

def test_relation_extraction_edge_cases():
    """Test relation extraction with various malformed JSON responses."""
    print("\n\nTesting Relation Extraction Edge Cases...")
    print("=" * 50)
    
    extractor = RelationExtractor()
    
    test_cases = [
        # Case 1: Perfect JSON
        {
            "name": "Perfect JSON",
            "input": '[{"source_entity": "Tim Cook", "target_entity": "Apple", "relation_type": "WORKS_FOR", "context": "CEO of Apple", "confidence": 0.9}]',
            "should_work": True
        },
        
        # Case 2: Trailing comma
        {
            "name": "Trailing comma",
            "input": '[{"source_entity": "Tim Cook", "target_entity": "Apple", "relation_type": "WORKS_FOR", "context": "CEO of Apple", "confidence": 0.9,}]',
            "should_work": True
        },
        
        # Case 3: Unquoted property names
        {
            "name": "Unquoted property names",
            "input": '[{source_entity: "Tim Cook", target_entity: "Apple", relation_type: "WORKS_FOR", context: "CEO of Apple", confidence: 0.9}]',
            "should_work": True
        },
        
        # Case 4: Multiple relations with issues
        {
            "name": "Multiple relations with issues",
            "input": '[{source_entity: "Tim Cook", target_entity: "Apple", relation_type: "WORKS_FOR",}, {"source_entity": "Apple", "target_entity": "Cupertino", "relation_type": "LOCATED_IN", "confidence": 0.8,}]',
            "should_work": True
        },
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Input: {test_case['input'][:100]}{'...' if len(test_case['input']) > 100 else ''}")
        
        try:
            relations = extractor.parse_response(test_case['input'], "test text")
            if test_case['should_work']:
                print(f"✅ Success: Found {len(relations)} relations")
                for relation in relations:
                    print(f"   - {relation.source_entity_id} -> {relation.target_entity_id} ({relation.type.value})")
            else:
                print(f"⚠️  Unexpected success: {len(relations)} relations")
        except Exception as e:
            if test_case['should_work']:
                print(f"❌ Failed: {e}")
            else:
                print(f"✅ Expected failure: {e}")

if __name__ == "__main__":
    test_entity_extraction_edge_cases()
    test_relation_extraction_edge_cases()
    print("\n" + "=" * 50)
    print("✅ All edge case tests completed!")