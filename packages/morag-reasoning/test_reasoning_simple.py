#!/usr/bin/env python3
"""Simple CLI script to quickly test multi-hop reasoning components.

This script provides a quick way to test individual components of the multi-hop
reasoning system without requiring a full graph database setup.

Usage:
    python test_reasoning_simple.py
    python test_reasoning_simple.py --verbose
    python test_reasoning_simple.py --component path_selection
    python test_reasoning_simple.py --component iterative_retrieval
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from morag_graph.operations import GraphPath
    from morag_reasoning import (
        IterativeRetriever,
        LLMClient,
        LLMConfig,
        PathRelevanceScore,
        PathSelectionAgent,
        ReasoningPathFinder,
        RetrievalContext,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the morag-reasoning package directory")
    print("   and all dependencies are installed.")
    sys.exit(1)


def create_sample_paths() -> List[GraphPath]:
    """Create sample graph paths for testing."""
    # Create mock entities and relations for GraphPath
    from morag_graph.models import Entity, Relation

    # Create sample entities
    apple = Entity(name="Apple Inc.", type="ORGANIZATION")
    steve = Entity(name="Steve Jobs", type="PERSON")
    iphone = Entity(name="iPhone", type="PRODUCT")
    ai_research = Entity(name="AI research", type="CONCEPT")
    stanford = Entity(name="Stanford University", type="ORGANIZATION")
    product_dev = Entity(name="product development", type="CONCEPT")
    innovation = Entity(name="innovation", type="CONCEPT")

    # Create sample relations using string types
    founded_rel = Relation(
        source_entity_id=steve.id, target_entity_id=apple.id, type="FOUNDED"
    )
    created_rel = Relation(
        source_entity_id=steve.id, target_entity_id=iphone.id, type="CREATED_BY"
    )
    conducts_rel = Relation(
        source_entity_id=apple.id, target_entity_id=ai_research.id, type="USES"
    )
    partners_rel = Relation(
        source_entity_id=apple.id, target_entity_id=stanford.id, type="WORKS_WITH"
    )
    led_rel = Relation(
        source_entity_id=steve.id, target_entity_id=product_dev.id, type="LEADS"
    )
    resulted_rel = Relation(
        source_entity_id=product_dev.id, target_entity_id=iphone.id, type="CREATED_BY"
    )
    influences_rel = Relation(
        source_entity_id=ai_research.id, target_entity_id=product_dev.id, type="AFFECTS"
    )
    drives_rel = Relation(
        source_entity_id=product_dev.id, target_entity_id=innovation.id, type="ENABLES"
    )

    return [
        GraphPath(
            entities=[apple, steve, iphone], relations=[founded_rel, created_rel]
        ),
        GraphPath(
            entities=[apple, ai_research, stanford],
            relations=[conducts_rel, partners_rel],
        ),
        GraphPath(
            entities=[steve, product_dev, iphone], relations=[led_rel, resulted_rel]
        ),
        GraphPath(
            entities=[ai_research, product_dev, innovation],
            relations=[influences_rel, drives_rel],
        ),
    ]


async def test_path_selection(verbose: bool = False) -> bool:
    """Test path selection functionality."""
    print("🔍 Testing Path Selection Agent...")

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GEMINI_API_KEY found. Using mock LLM responses.")

        # Create a mock LLM client
        class MockLLMClient:
            async def generate(self, prompt: str, **kwargs):
                return """
                {
                  "selected_paths": [
                    {
                      "path_id": 1,
                      "relevance_score": 8.5,
                      "confidence": 9.0,
                      "reasoning": "This path directly connects Apple to iPhone through Steve Jobs."
                    },
                    {
                      "path_id": 2,
                      "relevance_score": 7.2,
                      "confidence": 8.0,
                      "reasoning": "This path shows Apple's research connections."
                    }
                  ]
                }
                """

        llm_client = MockLLMClient()
    else:
        # Create real LLM client
        llm_config = LLMConfig(
            provider="gemini",
            api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=1000,
        )
        llm_client = LLMClient(llm_config)

    try:
        # Create path selection agent
        agent = PathSelectionAgent(llm_client, max_paths=5)

        # Create sample paths
        sample_paths = create_sample_paths()

        if verbose:
            print(f"   📊 Testing with {len(sample_paths)} sample paths")
            for i, path in enumerate(sample_paths):
                print(f"      Path {i+1}: {' -> '.join(path.entities)}")

        # Test path selection
        query = "How are Apple's founding and product development connected through key people?"
        selected_paths = await agent.select_paths(
            query, sample_paths, strategy="forward_chaining"
        )

        if verbose:
            print(f"   ✅ Selected {len(selected_paths)} paths:")
            for i, path_score in enumerate(selected_paths):
                print(
                    f"      {i+1}. Score: {path_score.relevance_score:.2f}, Confidence: {path_score.confidence:.2f}"
                )
                print(f"         Reasoning: {path_score.reasoning}")
                print(f"         Path: {' -> '.join(path_score.path.entities)}")
        else:
            print(f"   ✅ Selected {len(selected_paths)} relevant paths")

        # Verify results
        assert len(selected_paths) > 0, "No paths were selected"
        assert all(
            isinstance(ps, PathRelevanceScore) for ps in selected_paths
        ), "Invalid path score objects"
        assert all(
            ps.relevance_score >= 0 for ps in selected_paths
        ), "Invalid relevance scores"

        print("   ✅ Path selection test passed!")
        return True

    except Exception as e:
        print(f"   ❌ Path selection test failed: {e}")
        return False


async def test_iterative_retrieval(verbose: bool = False) -> bool:
    """Test iterative retrieval functionality."""
    print("🔄 Testing Iterative Retrieval...")

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GEMINI_API_KEY found. Using mock LLM responses.")

        # Create a mock LLM client
        class MockLLMClient:
            async def generate(self, prompt: str, **kwargs):
                return """
                {
                  "is_sufficient": true,
                  "confidence": 8.5,
                  "reasoning": "Context provides sufficient information about Apple and product development.",
                  "gaps": [],
                  "suggested_queries": []
                }
                """

        llm_client = MockLLMClient()
    else:
        # Create real LLM client
        llm_config = LLMConfig(
            provider="gemini",
            api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=800,
        )
        llm_client = LLMClient(llm_config)

    # Create mock graph engine and vector retriever
    class MockGraphEngine:
        async def get_entity_details(self, entity_name: str):
            return {"type": "ORG", "description": f"Details about {entity_name}"}

        async def get_relations_by_type(self, relation_type: str):
            return [
                {"subject": "Apple", "predicate": relation_type, "object": "iPhone"}
            ]

    class MockVectorRetriever:
        async def search(self, query: str, limit: int = 10):
            return [{"id": "doc1", "content": f"Document about {query}", "score": 0.9}]

    try:
        # Create iterative retriever
        retriever = IterativeRetriever(
            llm_client=llm_client,
            graph_engine=MockGraphEngine(),
            vector_retriever=MockVectorRetriever(),
            max_iterations=3,
            sufficiency_threshold=0.8,
        )

        # Create initial context
        initial_context = RetrievalContext(
            entities={"Apple Inc.": {"type": "ORG"}, "iPhone": {"type": "PRODUCT"}},
            documents=[
                {
                    "id": "doc1",
                    "content": "Apple Inc. is a technology company that develops the iPhone.",
                }
            ],
        )

        if verbose:
            print(
                f"   📊 Initial context: {len(initial_context.entities)} entities, {len(initial_context.documents)} documents"
            )

        # Test context refinement
        query = "What products does Apple develop?"
        refined_context = await retriever.refine_context(query, initial_context)

        if verbose:
            print(
                f"   ✅ Refined context: {len(refined_context.entities)} entities, {len(refined_context.documents)} documents"
            )
            print(
                f"      Iterations used: {refined_context.metadata.get('iterations_used', 0)}"
            )

            final_analysis = refined_context.metadata.get("final_analysis")
            if final_analysis:
                print(f"      Final confidence: {final_analysis.confidence:.2f}")
                print(f"      Context sufficient: {final_analysis.is_sufficient}")
        else:
            print(
                f"   ✅ Context refined in {refined_context.metadata.get('iterations_used', 0)} iterations"
            )

        # Verify results
        assert len(refined_context.entities) >= len(
            initial_context.entities
        ), "Context should not lose entities"
        assert "final_analysis" in refined_context.metadata, "Missing final analysis"
        assert "iterations_used" in refined_context.metadata, "Missing iteration count"

        print("   ✅ Iterative retrieval test passed!")
        return True

    except Exception as e:
        print(f"   ❌ Iterative retrieval test failed: {e}")
        return False


async def test_integration(verbose: bool = False) -> bool:
    """Test integration of all components."""
    print("🔗 Testing Component Integration...")

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GEMINI_API_KEY found. Skipping integration test.")
        return True

    try:
        # Create LLM client
        llm_config = LLMConfig(
            provider="gemini",
            api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=1000,
        )
        llm_client = LLMClient(llm_config)

        # Create mock graph engine
        class MockGraphEngine:
            async def traverse(self, start_entity: str, **kwargs):
                return {"paths": create_sample_paths()}

        # Create mock vector retriever
        class MockVectorRetriever:
            async def search(self, query: str, limit: int = 10):
                return [
                    {"id": "doc1", "content": f"Document about {query}", "score": 0.9}
                ]

        # Initialize all components
        path_selector = PathSelectionAgent(llm_client, max_paths=5)
        path_finder = ReasoningPathFinder(MockGraphEngine(), path_selector)
        iterative_retriever = IterativeRetriever(
            llm_client, MockGraphEngine(), MockVectorRetriever(), max_iterations=2
        )

        # Test end-to-end workflow
        query = "How are Apple's AI research efforts related to their partnership with universities?"
        start_entities = ["Apple Inc.", "AI research"]

        if verbose:
            print(f"   📝 Query: {query}")
            print(f"   🎯 Start entities: {start_entities}")

        # Step 1: Find reasoning paths
        reasoning_paths = await path_finder.find_reasoning_paths(
            query, start_entities, strategy="forward_chaining", max_paths=10
        )

        if verbose:
            print(f"   🔍 Found {len(reasoning_paths)} reasoning paths")

        # Step 2: Create initial context and refine
        initial_context = RetrievalContext(
            entities={
                entity: {"type": "UNKNOWN"}
                for path in reasoning_paths[:2]
                for entity in path.path.entities
            },
            paths=[path_score.path for path_score in reasoning_paths[:3]],
        )

        refined_context = await iterative_retriever.refine_context(
            query, initial_context
        )

        if verbose:
            print(f"   🔄 Context refined: {len(refined_context.entities)} entities")
            print(
                f"      Iterations: {refined_context.metadata.get('iterations_used', 0)}"
            )

        # Verify integration
        assert len(reasoning_paths) > 0, "No reasoning paths found"
        assert len(refined_context.entities) > 0, "No entities in refined context"

        print("   ✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


async def main():
    """Main function to run reasoning tests."""
    parser = argparse.ArgumentParser(
        description="Test multi-hop reasoning components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_reasoning_simple.py                           # Run all tests
  python test_reasoning_simple.py --verbose                 # Run with detailed output
  python test_reasoning_simple.py --component path_selection # Test only path selection
  python test_reasoning_simple.py --component iterative_retrieval # Test only iterative retrieval
""",
    )

    parser.add_argument(
        "--component",
        type=str,
        choices=["path_selection", "iterative_retrieval", "integration"],
        help="Test specific component only",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed test output"
    )

    args = parser.parse_args()

    print("🧠 MoRAG Multi-Hop Reasoning - Component Tests")
    print("=" * 60)

    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY found - using real LLM")
    else:
        print("⚠️  GEMINI_API_KEY not found - using mock responses")

    print("=" * 60)

    success = True

    try:
        if args.component == "path_selection" or not args.component:
            success &= await test_path_selection(args.verbose)
            print()

        if args.component == "iterative_retrieval" or not args.component:
            success &= await test_iterative_retrieval(args.verbose)
            print()

        if args.component == "integration" or not args.component:
            success &= await test_integration(args.verbose)
            print()

        print("=" * 60)
        if success:
            print("✅ All tests passed!")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
