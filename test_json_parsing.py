#!/usr/bin/env python3
"""
Test script to verify JSON parsing in fact extraction.
"""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

def test_json_parsing():
    """Test the JSON parsing logic."""
    print("🧪 Testing JSON Parsing")
    print("=" * 50)
    
    try:
        from morag_graph.extraction.fact_extractor import FactExtractor
        
        # Create a mock fact extractor just to test parsing
        extractor = FactExtractor(
            model_id="gemini-2.0-flash",
            api_key="mock-key",
            domain="test"
        )
        
        # Test cases
        test_cases = [
            # Case 1: Clean JSON array
            '''[
    {
        "subject": "Ginkgo Biloba",
        "object": "brain circulation",
        "approach": "daily intake",
        "solution": "improved concentration",
        "remarks": "120-240mg dosage"
    }
]''',
            
            # Case 2: JSON in markdown code block
            '''```json
[
    {
        "subject": "Ginseng",
        "object": "cognitive function",
        "approach": "standardized extract",
        "solution": "reduced hyperactivity",
        "remarks": "200-400mg daily"
    }
]
```''',
            
            # Case 3: JSON with extra text
            '''Here are the extracted facts:

[
    {
        "subject": "Rhodiola",
        "object": "mental clarity",
        "approach": "adaptogenic properties",
        "solution": "improved attention span",
        "remarks": "300-600mg daily"
    }
]

These facts were extracted from the text.''',
            
            # Case 4: Malformed JSON (this should fail gracefully)
            '''[
    {
        "subject": "Test"
        "object": "missing comma"
    }
]''',
            
            # Case 5: The problematic case that might be causing the error
            '''
    "subject": "Ginkgo Biloba",
    "object": "brain circulation"
'''
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}:")
            print(f"Input: {test_case[:100]}...")
            
            try:
                result = extractor._parse_llm_response(test_case)
                print(f"✅ Parsed {len(result)} candidates")
                
                for j, candidate in enumerate(result):
                    if isinstance(candidate, dict):
                        subject = candidate.get('subject', 'N/A')
                        obj = candidate.get('object', 'N/A')
                        print(f"   Candidate {j+1}: {subject} → {obj}")
                    else:
                        print(f"   Candidate {j+1}: Invalid type {type(candidate)}")
                        
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n🎉 JSON parsing tests completed!")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_json_parsing()
