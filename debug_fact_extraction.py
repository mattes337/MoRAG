#!/usr/bin/env python3
"""
Debug script to test fact extraction and see what the LLM is returning.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

async def debug_fact_extraction():
    """Debug fact extraction to see what's happening."""
    print("🔍 Debugging Fact Extraction")
    print("=" * 50)
    
    try:
        from morag_graph.extraction.fact_extractor import FactExtractor
        
        # Check if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ No GEMINI_API_KEY found. Please set the environment variable.")
            return
        
        print("✅ API key found")
        
        # Initialize fact extractor
        fact_extractor = FactExtractor(
            model_id="gemini-2.0-flash",
            api_key=api_key,
            domain="medical",
            min_confidence=0.7,
            max_facts_per_chunk=5
        )
        
        print("✅ Fact extractor initialized")
        
        # Test text (simple German text about ADHD)
        test_text = """
        Ginkgo Biloba verbessert die Durchblutung des Gehirns und kann die Konzentrationsfähigkeit steigern. 
        Studien zeigen, dass eine tägliche Einnahme von 120-240mg die Aufmerksamkeit bei ADHS-Patienten verbessern kann.
        """
        
        print(f"📝 Test text: {test_text.strip()}")
        print("🔍 Extracting facts...")
        
        # Extract facts with debug logging
        facts = await fact_extractor.extract_facts(
            chunk_text=test_text,
            chunk_id="debug_chunk_1",
            document_id="debug_doc_1",
            context={
                'domain': 'medical',
                'language': 'de',
                'source_file_name': 'debug.txt'
            }
        )
        
        print(f"✅ Extracted {len(facts)} facts")
        
        # Display facts
        for i, fact in enumerate(facts, 1):
            print(f"\n📋 Fact {i}:")
            print(f"   Subject: {fact.subject}")
            print(f"   Object: {fact.object}")
            print(f"   Approach: {fact.approach}")
            print(f"   Solution: {fact.solution}")
            print(f"   Remarks: {fact.remarks}")
            print(f"   Type: {fact.fact_type}")
            print(f"   Confidence: {fact.extraction_confidence:.2f}")
        
        if len(facts) == 0:
            print("⚠️  No facts extracted. This suggests an issue with the LLM response parsing.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_fact_extraction())
