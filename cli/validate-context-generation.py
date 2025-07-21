#!/usr/bin/env python3
"""
Simple validation script for the context generation functionality.
This script tests the basic functionality without requiring full MoRAG installation.
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
cli_path = project_root / 'cli'
sys.path.insert(0, str(cli_path))

def test_script_imports():
    """Test that the script can be imported without errors."""
    print("🧪 Testing script imports...")
    
    try:
        # Mock the dependencies to avoid import errors
        sys.modules['morag_graph'] = MagicMock()
        sys.modules['morag_graph.ai.entity_agent'] = MagicMock()
        sys.modules['morag_graph.utils.entity_normalizer'] = MagicMock()
        sys.modules['morag.database_factory'] = MagicMock()
        sys.modules['morag_reasoning'] = MagicMock()
        sys.modules['morag_core.ai'] = MagicMock()
        
        # Import the module directly from the file
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_context_generation", cli_path / "test-context-generation.py")
        tcg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tcg)
        
        print("✅ Script imports successfully")
        return tcg
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return None


def test_context_generation_result():
    """Test the ContextGenerationResult class."""
    print("\n🧪 Testing ContextGenerationResult...")
    
    try:
        tcg = test_script_imports()
        if not tcg:
            return False
        
        result = tcg.ContextGenerationResult()
        
        # Test initialization
        assert result.prompt == ""
        assert result.extracted_entities == []
        assert result.graph_entities == []
        assert result.graph_relations == []
        assert result.vector_chunks == []
        assert result.reasoning_paths == []
        assert result.context_score == 0.0
        assert result.final_response == ""
        assert result.processing_steps == []
        assert result.performance_metrics == {}
        assert result.error is None
        assert result.timestamp is not None
        
        print("✅ ContextGenerationResult works correctly")
        return True
        
    except Exception as e:
        print(f"❌ ContextGenerationResult test failed: {e}")
        return False


def test_agentic_context_generator():
    """Test the AgenticContextGenerator class."""
    print("\n🧪 Testing AgenticContextGenerator...")
    
    try:
        tcg = test_script_imports()
        if not tcg:
            return False
        
        # Test initialization
        generator = tcg.AgenticContextGenerator(verbose=False)
        assert generator.neo4j_storage is None
        assert generator.qdrant_storage is None
        assert generator.llm_client is None
        assert generator.verbose is False
        
        # Test with mock dependencies
        mock_neo4j = MagicMock()
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        
        generator = tcg.AgenticContextGenerator(
            neo4j_storage=mock_neo4j,
            qdrant_storage=mock_qdrant,
            llm_client=mock_llm,
            verbose=True
        )
        
        assert generator.neo4j_storage == mock_neo4j
        assert generator.qdrant_storage == mock_qdrant
        assert generator.llm_client == mock_llm
        assert generator.verbose is True
        
        print("✅ AgenticContextGenerator initialization works correctly")
        return True
        
    except Exception as e:
        print(f"❌ AgenticContextGenerator test failed: {e}")
        return False


async def test_entity_extraction():
    """Test entity extraction functionality."""
    print("\n🧪 Testing entity extraction...")
    
    try:
        tcg = test_script_imports()
        if not tcg:
            return False
        
        # Test without LLM client
        generator = tcg.AgenticContextGenerator(llm_client=None)
        entities = await generator._extract_entities_agentic("test prompt")
        assert entities == []
        
        # Test with mock LLM client
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = '''
        [
            {"name": "nutrition", "type": "CONCEPT", "confidence": 0.9, "relevance": "Key topic"},
            {"name": "ADHD", "type": "MEDICAL_CONDITION", "confidence": 0.95, "relevance": "Target condition"}
        ]
        '''
        
        generator = tcg.AgenticContextGenerator(llm_client=mock_llm)
        entities = await generator._extract_entities_agentic("How does nutrition affect ADHD?")
        
        assert len(entities) == 2
        assert entities[0]["name"] == "nutrition"
        assert entities[0]["type"] == "CONCEPT"
        assert entities[1]["name"] == "ADHD"
        assert entities[1]["type"] == "MEDICAL_CONDITION"
        
        print("✅ Entity extraction works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        return False


async def test_vector_search():
    """Test vector search functionality."""
    print("\n🧪 Testing vector search...")
    
    try:
        tcg = test_script_imports()
        if not tcg:
            return False
        
        # Test without Qdrant storage
        generator = tcg.AgenticContextGenerator(qdrant_storage=None)
        chunks = await generator._search_vector_documents("test prompt")
        assert chunks == []
        
        # Test with mock Qdrant storage
        mock_qdrant = AsyncMock()
        mock_qdrant.search_entities.return_value = [
            {
                "content": "Test document content",
                "metadata": {"source": "test.pdf"},
                "score": 0.85
            }
        ]
        
        generator = tcg.AgenticContextGenerator(qdrant_storage=mock_qdrant)
        chunks = await generator._search_vector_documents("test prompt")
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Test document content"
        assert chunks[0]["score"] == 0.85
        
        print("✅ Vector search works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        return False


async def test_context_scoring():
    """Test context scoring functionality."""
    print("\n🧪 Testing context scoring...")
    
    try:
        tcg = test_script_imports()
        if not tcg:
            return False
        
        # Test without LLM client (heuristic scoring)
        generator = tcg.AgenticContextGenerator(llm_client=None)
        result = tcg.ContextGenerationResult()
        result.extracted_entities = [{"name": "test"}] * 3
        result.graph_entities = [{"name": "test"}] * 5
        result.vector_chunks = [{"content": "test"}] * 2
        
        score = await generator._score_context("test prompt", result)
        assert 0.0 <= score <= 1.0
        
        # Test with mock LLM client
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "0.75"
        
        generator = tcg.AgenticContextGenerator(llm_client=mock_llm)
        result = tcg.ContextGenerationResult()
        result.extracted_entities = [{"name": "test"}]
        
        score = await generator._score_context("test prompt", result)
        assert score == 0.75
        
        print("✅ Context scoring works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Context scoring test failed: {e}")
        return False


async def test_response_generation():
    """Test final response generation."""
    print("\n🧪 Testing response generation...")
    
    try:
        tcg = test_script_imports()
        if not tcg:
            return False
        
        # Test without LLM client
        generator = tcg.AgenticContextGenerator(llm_client=None)
        result = tcg.ContextGenerationResult()
        response = await generator._generate_final_response("test prompt", result)
        assert "LLM client not available" in response
        
        # Test with mock LLM client
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "This is a test response."
        
        generator = tcg.AgenticContextGenerator(llm_client=mock_llm)
        result = tcg.ContextGenerationResult()
        result.context_score = 0.8
        result.extracted_entities = [{"name": "test", "type": "CONCEPT", "relevance": "test"}]
        
        response = await generator._generate_final_response("test prompt", result)
        assert response == "This is a test response."
        
        print("✅ Response generation works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Response generation test failed: {e}")
        return False


async def test_full_context_generation():
    """Test the complete context generation flow."""
    print("\n🧪 Testing full context generation flow...")
    
    try:
        tcg = test_script_imports()
        if not tcg:
            return False
        
        # Mock all dependencies
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '[{"name": "test", "type": "CONCEPT", "confidence": 0.9, "relevance": "test"}]',  # Entity extraction
            "0.8",  # Context scoring
            "Final response based on context."  # Final response
        ]
        
        mock_qdrant = AsyncMock()
        mock_qdrant.search_entities.return_value = [
            {"content": "Test content", "metadata": {}, "score": 0.9}
        ]
        
        generator = tcg.AgenticContextGenerator(
            neo4j_storage=None,  # Skip Neo4j for this test
            qdrant_storage=mock_qdrant,
            llm_client=mock_llm,
            verbose=False
        )
        
        result = await generator.generate_context("test prompt")
        
        assert result.prompt == "test prompt"
        assert len(result.extracted_entities) == 1
        assert len(result.vector_chunks) == 1
        assert result.context_score == 0.8
        assert result.final_response == "Final response based on context."
        assert result.error is None
        assert len(result.processing_steps) > 0
        
        print("✅ Full context generation flow works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Full context generation test failed: {e}")
        return False


async def main():
    """Run all validation tests."""
    print("🚀 MoRAG Context Generation Validation")
    print("=" * 50)
    
    tests = [
        ("Script Imports", test_script_imports),
        ("ContextGenerationResult", test_context_generation_result),
        ("AgenticContextGenerator", test_agentic_context_generator),
        ("Entity Extraction", test_entity_extraction),
        ("Vector Search", test_vector_search),
        ("Context Scoring", test_context_scoring),
        ("Response Generation", test_response_generation),
        ("Full Context Generation", test_full_context_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The context generation script is working correctly.")
        print("\n💡 You can now use the script with:")
        print("   python cli/test-context-generation.py --help")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
