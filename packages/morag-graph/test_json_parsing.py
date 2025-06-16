#!/usr/bin/env python3
"""
Test script to reproduce and fix JSON parsing errors in entity/relation extraction.
"""

import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from morag_graph.extraction.base import BaseExtractor
from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor

def test_malformed_json_cases():
    """Test various malformed JSON cases that might cause parsing errors."""
    
    # Create an entity extractor to test JSON parsing
    extractor = EntityExtractor()
    
    # Test cases that might cause "Expecting property name enclosed in double quotes"
    malformed_cases = [
        # Case 1: Unquoted property names
        '[{name: "Apple", type: "ORG"}]',
        
        # Case 2: Trailing comma
        '[{"name": "Apple", "type": "ORG",}]',
        
        # Case 3: Missing quotes around property names
        '[{"name": "Apple", type: "ORG"}]',
        
        # Case 4: Single quotes instead of double quotes
        "[{'name': 'Apple', 'type': 'ORG'}]",
        
        # Case 5: Mixed quote types
        '[{"name": \'Apple\', "type": "ORG"}]',
        
        # Case 6: Incomplete JSON
        '[{"name": "Apple", "type"',
        
        # Case 7: Extra characters after JSON
        '[{"name": "Apple", "type": "ORG"}] extra text',
        
        # Case 8: JSON with markdown that wasn't stripped
        '```json\n[{"name": "Apple", "type": "ORG"}]\n```\nSome explanation text',
        
        # Case 9: Large JSON with error at specific position (simulating line 236)
        generate_large_malformed_json(),
    ]
    
    print("Testing malformed JSON cases...")
    
    for i, case in enumerate(malformed_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Input: {case[:100]}{'...' if len(case) > 100 else ''}")
        
        try:
            result = extractor.parse_json_response(case)
            print(f"✅ Parsed successfully: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
            if "line 236" in str(e) or "char 6084" in str(e):
                print("🎯 This matches the reported error!")

def generate_large_malformed_json():
    """Generate a large JSON string with an error around line 236."""
    # Create a large valid JSON first
    entities = []
    for i in range(200):
        entities.append({
            "name": f"Entity_{i}",
            "type": "ORG",
            "context": f"This is entity number {i} with some context text",
            "confidence": 0.9
        })
    
    # Convert to JSON string
    json_str = json.dumps(entities, indent=2)
    
    # Introduce an error around line 236 by removing quotes from a property name
    lines = json_str.split('\n')
    if len(lines) > 235:
        # Find a line with a property name and remove quotes
        for i in range(230, min(240, len(lines))):
            if '"name":' in lines[i]:
                lines[i] = lines[i].replace('"name":', 'name:')
                break
    
    return '\n'.join(lines)

def test_improved_json_parsing():
    """Test improved JSON parsing with better error handling."""
    print("\n" + "="*50)
    print("Testing improved JSON parsing...")
    
    # Test the current implementation
    extractor = EntityExtractor()
    
    # Test with a case that should be fixable
    test_case = '[{"name": "Apple", "type": "ORG",}]'  # Trailing comma
    
    try:
        result = extractor.parse_json_response(test_case)
        print(f"✅ Current implementation handled: {result}")
    except Exception as e:
        print(f"❌ Current implementation failed: {e}")
        print("This case should be fixable with improved parsing.")

if __name__ == "__main__":
    test_malformed_json_cases()
    test_improved_json_parsing()