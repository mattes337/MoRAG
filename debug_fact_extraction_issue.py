#!/usr/bin/env python3
"""Debug script to test fact extraction with the actual ADHS content."""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-embedding" / "src"))

from morag_graph.extraction.fact_extractor import FactExtractor
from morag_core.config import settings

async def debug_fact_extraction():
    """Debug fact extraction with actual ADHS content."""
    
    print("🔍 Debugging Fact Extraction Issue")
    print("=" * 50)
    
    try:
        # Initialize fact extractor
        print("📋 Initializing fact extractor...")
        
        fact_extractor = FactExtractor(
            model_id="gemini-2.0-flash-lite",
            api_key=os.getenv("GEMINI_API_KEY"),
            min_confidence=0.7,
            max_facts_per_chunk=10,
            domain="medical"
        )
        
        print("✅ Fact extractor initialized")
        
        # Test with actual ADHS content from the video (medical content about symptoms)
        test_text = """
[12:11 - 12:14] Sie neigen dazu, verschiedene Projekte anzufangen,
[12:14 - 12:17] aber sie dann schnell wieder aufzuhören und nicht zu beenden.
[12:18 - 12:19] Also das ist dieses nicht zu beenden.
[12:19 - 12:22] Das ist dieses Desinteresse, das sich schnell einstellt.
[12:23 - 12:26] Dieser Typ ist auch viel von Frustration geprägt,
[12:26 - 12:29] weil vielleicht etwas, was sie erreichen wollen,
[12:29 - 12:30] was sie dann auch schnell erreichen wollen,
[12:30 - 12:32] weil halt das Interesse dann schnell wieder verschwindet,
[12:33 - 12:35] vielleicht nicht möglich ist und dann wird es nicht erreicht.
[12:35 - 12:37] Die Vorstellung ist nicht realisierbar
[12:37 - 12:40] und dann kommt die Frustration mit ins Spiel bei dem Mischtyp.
[12:40 - 12:46] Und sie zeigen auch generell sehr ausgeprägte emotionale Reaktionen.
[12:46 - 12:48] Das ist auch diese Impulsivität, die rauskommt.
[12:49 - 12:55] Und gerade dieser Mischtyp hat viele Herausforderungen im Alltag.
[12:55 - 12:57] Man sagt aber auch, die haben sehr spezielle Talente.
[12:57 - 12:59] Auch die anderen beiden Typen.
[12:59 - 13:00] Die haben Talente natürlich.
[13:00 - 13:04] Aber dieser Mischtyp, der steht auch noch mal vor einer besonderen Herausforderung,
[13:04 - 13:08] weil eben diese beiden Welten da aufeinanderprallen von dem ADHS.
        """
        
        print(f"📝 Test text length: {len(test_text)} characters")
        print(f"📝 Test text preview: {test_text[:200]}...")
        print("🔍 Extracting facts...")
        
        # Let's also check what prompt is being generated
        from morag_graph.extraction.fact_prompts import FactExtractionPrompts

        prompt = FactExtractionPrompts.create_extraction_prompt(
            chunk_text=test_text,
            domain='medical',
            language='de',
            max_facts=10
        )

        print(f"🔍 Generated prompt length: {len(prompt)}")
        print(f"🔍 Prompt preview (first 500 chars):")
        print(prompt[:500])
        print("...")
        print(f"🔍 Prompt ending (last 500 chars):")
        print(prompt[-500:])

        # Extract facts with debug logging
        facts = await fact_extractor.extract_facts(
            chunk_text=test_text,
            chunk_id="debug_chunk_adhs",
            document_id="debug_doc_adhs",
            context={
                'domain': 'medical',
                'language': 'de',
                'source_file_name': 'adhs_video.mp4'
            }
        )
        
        print(f"✅ Extracted {len(facts)} facts")
        
        if len(facts) == 0:
            print("⚠️  NO FACTS EXTRACTED!")
            print("This suggests the LLM is not responding with valid fact data.")
            print("Let's test with a simpler German medical text...")
            
            # Test with simpler German medical content
            simple_test = """
            ADHS ist eine neurologische Entwicklungsstörung. 
            Die Hauptsymptome sind Unaufmerksamkeit, Hyperaktivität und Impulsivität.
            ADHS betrifft etwa 5% der Kinder weltweit.
            Methylphenidat ist ein häufig verwendetes Medikament zur ADHS-Behandlung.
            Verhaltenstherapie kann bei ADHS-Symptomen helfen.
            """
            
            print(f"\n🔍 Testing with simpler text: {simple_test}")
            
            simple_facts = await fact_extractor.extract_facts(
                chunk_text=simple_test,
                chunk_id="debug_simple_chunk",
                document_id="debug_simple_doc",
                context={
                    'domain': 'medical',
                    'language': 'de',
                    'source_file_name': 'simple_test.txt'
                }
            )
            
            print(f"✅ Simple test extracted {len(simple_facts)} facts")
            
            if len(simple_facts) == 0:
                print("❌ Even simple text yields no facts!")
                print("This indicates a fundamental issue with the fact extraction LLM prompt or response parsing.")
            else:
                print("✅ Simple text works, issue might be with complex timestamped content.")
        
        # Display any facts found
        for i, fact in enumerate(facts, 1):
            print(f"\n📋 Fact {i}:")
            print(f"   Subject: {fact.subject}")
            print(f"   Object: {fact.object}")
            print(f"   Approach: {fact.approach}")
            print(f"   Solution: {fact.solution}")
            print(f"   Remarks: {fact.remarks}")
            print(f"   Type: {fact.fact_type}")
            print(f"   Confidence: {fact.extraction_confidence:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_fact_extraction())
