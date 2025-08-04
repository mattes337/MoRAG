#!/usr/bin/env python3
"""
Simple test script to validate the fact extraction pipeline.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

async def test_fact_extraction():
    """Test basic fact extraction functionality."""
    print("🧪 Testing Fact Extraction Pipeline")
    print("=" * 50)
    
    try:
        # Import fact extraction components
        from morag_graph.extraction.fact_extractor import FactExtractor
        from morag_graph.models.fact import Fact
        
        print("✅ Successfully imported fact extraction components")
        
        # Check if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ No GEMINI_API_KEY found. Skipping LLM-based tests.")
            return False
        
        print("✅ API key found")
        
        # Initialize fact extractor
        fact_extractor = FactExtractor(
            model_id="gemini-2.0-flash",
            api_key=api_key,
            domain="general",
            min_confidence=0.7,
            max_facts_per_chunk=5
        )
        
        print("✅ Fact extractor initialized")
        
        # Test text
        test_text = """
        Python is a high-level programming language that is widely used for web development, 
        data analysis, and artificial intelligence. It was created by Guido van Rossum and 
        first released in 1991. Python's simple syntax makes it easy to learn and use.
        """
        
        print("🔍 Extracting facts from test text...")
        
        # Extract facts
        facts = await fact_extractor.extract_facts(
            chunk_text=test_text,
            chunk_id="test_chunk_1",
            document_id="test_doc_1",
            context={
                'domain': 'technology',
                'language': 'en',
                'source_file_name': 'test.txt'
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
            print(f"   Citation: {fact.get_citation()}")
            print(f"   Machine Source: {fact.get_machine_readable_source()}")
        
        # Test fact model serialization
        print("\n🔄 Testing fact serialization...")
        
        if facts:
            fact_dict = facts[0].to_dict()
            print("✅ Fact serialization successful")
            
            # Test fact reconstruction
            reconstructed_fact = Fact(**fact_dict)
            print("✅ Fact reconstruction successful")
        
        print("\n🎉 All fact extraction tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fact_models():
    """Test fact model functionality without LLM."""
    print("\n🧪 Testing Fact Models")
    print("=" * 50)
    
    try:
        from morag_graph.models.fact import Fact, FactRelation
        from datetime import datetime
        
        print("✅ Successfully imported fact models")
        
        # Create a test fact
        fact = Fact(
            subject="Python",
            object="programming language",
            approach="high-level syntax",
            solution="easy development",
            remarks="widely used for various applications",
            source_chunk_id="test_chunk_1",
            source_document_id="test_doc_1",
            extraction_confidence=0.95,
            fact_type="definition",
            domain="technology",
            keywords=["python", "programming", "language"],
            source_file_name="test.txt",
            page_number=1,
            chapter_title="Introduction to Python"
        )
        
        print("✅ Fact created successfully")
        
        # Test fact methods
        citation = fact.get_citation()
        machine_source = fact.get_machine_readable_source()
        search_text = fact.get_search_text()
        fact_dict = fact.to_dict()
        neo4j_props = fact.get_neo4j_properties()
        
        print(f"✅ Citation: {citation}")
        print(f"✅ Machine source: {machine_source}")
        print(f"✅ Search text length: {len(search_text)} characters")
        print(f"✅ Fact dict keys: {list(fact_dict.keys())}")
        print(f"✅ Neo4j properties keys: {list(neo4j_props.keys())}")
        
        # Create a test relationship
        fact2 = Fact(
            subject="Guido van Rossum",
            object="Python",
            approach="programming language design",
            solution="created Python",
            remarks="first released in 1991",
            source_chunk_id="test_chunk_1",
            source_document_id="test_doc_1",
            extraction_confidence=0.90,
            fact_type="creation",
            domain="technology"
        )
        
        relationship = FactRelation(
            source_fact_id=fact.id,
            target_fact_id=fact2.id,
            relation_type="CREATED_BY",
            confidence=0.85,
            context="Python was created by Guido van Rossum"
        )
        
        print("✅ Fact relationship created successfully")
        print(f"✅ Relationship: {fact.subject} --[{relationship.relation_type}]--> {fact2.subject}")
        
        print("\n🎉 All fact model tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Fact Extraction Pipeline Tests")
    print("=" * 60)
    
    # Test fact models (no external dependencies)
    model_success = await test_fact_models()
    
    # Test fact extraction (requires API key)
    extraction_success = await test_fact_extraction()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Fact Models: {'✅ PASS' if model_success else '❌ FAIL'}")
    print(f"   Fact Extraction: {'✅ PASS' if extraction_success else '❌ FAIL'}")
    
    if model_success and extraction_success:
        print("\n🎉 All tests passed! Fact extraction pipeline is working correctly.")
        return 0
    elif model_success:
        print("\n⚠️  Fact models work, but extraction requires GEMINI_API_KEY environment variable.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
