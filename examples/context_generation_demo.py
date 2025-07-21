#!/usr/bin/env python3
"""
Demo script for MoRAG Context Generation Testing

This script demonstrates how to use the context generation testing functionality
to test LLM prompts with agentic AI-driven context gathering.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Import the context generator
try:
    # Add cli directory to path for import
    cli_path = project_root / 'cli'
    sys.path.insert(0, str(cli_path))

    from test_context_generation import AgenticContextGenerator, ContextGenerationResult
    from morag.database_factory import get_default_neo4j_storage, get_default_qdrant_storage
    from morag_reasoning import LLMClient, LLMConfig
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing components: {e}")
    COMPONENTS_AVAILABLE = False


async def demo_context_generation():
    """Demonstrate context generation functionality."""
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Required components not available")
        return
    
    print("🚀 MoRAG Context Generation Demo")
    print("=" * 50)
    
    # Test prompts
    test_prompts = [
        "How does nutrition affect ADHD symptoms?",
        "What are the connections between artificial intelligence and healthcare?",
        "Explain the relationship between exercise and mental health",
        "How do social media algorithms influence user behavior?"
    ]
    
    # Initialize connections (mock for demo)
    neo4j_storage = None
    qdrant_storage = None
    
    try:
        # Try to connect to databases
        print("🔌 Attempting database connections...")
        
        try:
            neo4j_storage = get_default_neo4j_storage()
            if neo4j_storage:
                await neo4j_storage.connect()
                print("✅ Neo4j connected")
        except Exception as e:
            print(f"⚠️  Neo4j not available: {e}")
        
        try:
            qdrant_storage = get_default_qdrant_storage()
            if qdrant_storage:
                await qdrant_storage.connect()
                print("✅ Qdrant connected")
        except Exception as e:
            print(f"⚠️  Qdrant not available: {e}")
        
        # Initialize LLM client
        llm_config = LLMConfig(
            provider="gemini",
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0.1,
            max_tokens=2000
        )
        llm_client = LLMClient(llm_config)
        
        # Initialize context generator
        generator = AgenticContextGenerator(
            neo4j_storage=neo4j_storage,
            qdrant_storage=qdrant_storage,
            llm_client=llm_client,
            verbose=True
        )
        
        # Test each prompt
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*60}")
            print(f"🧪 Test {i}: {prompt}")
            print(f"{'='*60}")
            
            try:
                # Generate context
                result = await generator.generate_context(prompt)
                
                if result.error:
                    print(f"❌ Error: {result.error}")
                    continue
                
                # Display results
                print(f"\n📊 Results Summary:")
                print(f"   Context Score: {result.context_score:.2f}/1.0")
                print(f"   Entities: {len(result.extracted_entities)}")
                print(f"   Graph Entities: {len(result.graph_entities)}")
                print(f"   Graph Relations: {len(result.graph_relations)}")
                print(f"   Vector Chunks: {len(result.vector_chunks)}")
                print(f"   Processing Time: {result.performance_metrics.get('total_time_seconds', 0):.2f}s")
                
                # Show extracted entities
                if result.extracted_entities:
                    print(f"\n🔍 Extracted Entities:")
                    for entity in result.extracted_entities[:3]:
                        print(f"   - {entity.get('name', '')} ({entity.get('type', 'UNKNOWN')})")
                
                # Show sample graph data
                if result.graph_entities:
                    print(f"\n🕸️  Sample Graph Entities:")
                    for entity in result.graph_entities[:3]:
                        print(f"   - {entity.get('name', '')} ({', '.join(entity.get('types', []))})")
                
                if result.graph_relations:
                    print(f"\n🔗 Sample Relations:")
                    for relation in result.graph_relations[:3]:
                        print(f"   - {relation.get('source', '')} → {relation.get('target', '')}")
                
                # Show final response (truncated)
                print(f"\n🎯 Response Preview:")
                response_preview = result.final_response[:200] + "..." if len(result.final_response) > 200 else result.final_response
                print(f"   {response_preview}")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                continue
    
    finally:
        # Clean up connections
        if neo4j_storage:
            try:
                await neo4j_storage.disconnect()
            except:
                pass
        if qdrant_storage:
            try:
                await qdrant_storage.disconnect()
            except:
                pass


async def demo_simple_usage():
    """Demonstrate simple usage without database connections."""
    
    print("\n" + "="*60)
    print("🔧 Simple Usage Demo (LLM Only)")
    print("="*60)
    
    # Initialize with LLM only
    llm_config = LLMConfig(
        provider="gemini",
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash"),
        temperature=0.1,
        max_tokens=1000
    )
    llm_client = LLMClient(llm_config)
    
    generator = AgenticContextGenerator(
        neo4j_storage=None,
        qdrant_storage=None,
        llm_client=llm_client,
        verbose=False
    )
    
    test_prompt = "What are the benefits of regular exercise?"
    
    try:
        result = await generator.generate_context(test_prompt)
        
        print(f"📝 Prompt: {test_prompt}")
        print(f"🔍 Entities Extracted: {len(result.extracted_entities)}")
        
        if result.extracted_entities:
            print("   Entities:")
            for entity in result.extracted_entities:
                print(f"   - {entity.get('name', '')} ({entity.get('type', 'UNKNOWN')})")
        
        print(f"\n🎯 Response:")
        print(f"   {result.final_response[:300]}...")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def print_usage_examples():
    """Print usage examples for the CLI script."""
    
    print("\n" + "="*60)
    print("📖 CLI Usage Examples")
    print("="*60)
    
    examples = [
        "# Basic usage with both databases",
        "python cli/test-context-generation.py --neo4j --qdrant \"How does nutrition affect ADHD?\"",
        "",
        "# Verbose mode with context display",
        "python cli/test-context-generation.py --neo4j --qdrant --verbose --show-context \"AI in healthcare\"",
        "",
        "# Save results to JSON",
        "python cli/test-context-generation.py --neo4j --qdrant --output results.json \"Machine learning\"",
        "",
        "# Use specific model",
        "python cli/test-context-generation.py --neo4j --model gemini-1.5-pro \"Complex query\"",
        "",
        "# Graph operations only",
        "python cli/test-context-generation.py --neo4j \"Entity relationships\"",
        "",
        "# Vector search only", 
        "python cli/test-context-generation.py --qdrant \"Document search\""
    ]
    
    for example in examples:
        print(example)


async def main():
    """Main demo function."""
    
    print("🎭 MoRAG Context Generation Demo")
    print("This demo shows how the agentic AI context generation works")
    print()
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not found in environment")
        print("   Set your API key to run the full demo")
        print_usage_examples()
        return
    
    try:
        # Run the main demo
        await demo_context_generation()
        
        # Run simple demo
        await demo_simple_usage()
        
        # Show usage examples
        print_usage_examples()
        
        print(f"\n✅ Demo completed!")
        print(f"💡 Try the CLI script: python cli/test-context-generation.py --help")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
