#!/usr/bin/env python3
"""Test entity ID generation with the fixed logic."""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

from morag_graph.utils.id_generation import UnifiedIDGenerator


def test_entity_id_generation():
    """Test the fixed entity ID generation logic."""
    
    print("🧪 Testing Entity ID Generation")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "innere Bilder",
            "entity_type": "CUSTOM",
            "source_doc_id": "",
            "description": "Empty source_doc_id"
        },
        {
            "name": "innere Bilder", 
            "entity_type": "CUSTOM",
            "source_doc_id": None,
            "description": "None source_doc_id"
        },
        {
            "name": "innere Bilder",
            "entity_type": "CUSTOM", 
            "source_doc_id": "doc_test_file_f807af79b2f8ed46",
            "description": "Proper document ID with underscores"
        },
        {
            "name": "innere Bilder",
            "entity_type": "CUSTOM",
            "source_doc_id": "some_document_without_proper_format",
            "description": "Document ID without proper format"
        },
        {
            "name": "innere Bilder",
            "entity_type": "CUSTOM",
            "source_doc_id": "H:\\Drittes Auge\\Constanze Witzel - Open your Third Eye.mp4",
            "description": "File path as source_doc_id"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"  Input:")
        print(f"    name: '{test_case['name']}'")
        print(f"    type: '{test_case['entity_type']}'")
        print(f"    source_doc_id: '{test_case['source_doc_id']}'")
        
        try:
            entity_id = UnifiedIDGenerator.generate_entity_id(
                name=test_case['name'],
                entity_type=test_case['entity_type'],
                source_doc_id=test_case['source_doc_id']
            )
            print(f"  Output: {entity_id}")
            
            # Check if it contains abc123 (old placeholder)
            if 'abc123' in entity_id:
                print(f"  ⚠️  Still contains placeholder!")
            else:
                print(f"  ✅ No placeholder found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    test_entity_id_generation()
