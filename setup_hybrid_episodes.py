#!/usr/bin/env python3
"""
Setup script for testing hybrid episode functionality.

This script helps you:
1. Check if API keys are configured
2. Test the hybrid episode functionality with a sample document
3. Provide guidance on setting up the environment
"""

import os
import sys
from pathlib import Path

def check_api_keys():
    """Check if required API keys are configured."""
    print("🔑 Checking API Key Configuration...")
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if gemini_key:
        print(f"✅ GEMINI_API_KEY: Configured (length: {len(gemini_key)})")
        return True
    elif openai_key:
        print(f"✅ OPENAI_API_KEY: Configured (length: {len(openai_key)})")
        return True
    else:
        print("❌ No API keys found!")
        print("\n📋 To use hybrid episodes, you need either:")
        print("   1. GEMINI_API_KEY (recommended for Graphiti)")
        print("   2. OPENAI_API_KEY")
        print("\n🔧 Setup Instructions:")
        print("   1. Get a Gemini API key from: https://makersuite.google.com/app/apikey")
        print("   2. Add to your .env file:")
        print("      GEMINI_API_KEY=your_api_key_here")
        print("   3. Or set as environment variable:")
        print("      set GEMINI_API_KEY=your_api_key_here  (Windows)")
        print("      export GEMINI_API_KEY=your_api_key_here  (Linux/Mac)")
        return False

def check_neo4j_config():
    """Check Neo4j configuration."""
    print("\n🗄️ Checking Neo4j Configuration...")
    
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_pass = os.getenv('NEO4J_PASSWORD', 'password')
    
    print(f"   URI: {neo4j_uri}")
    print(f"   Username: {neo4j_user}")
    print(f"   Password: {'*' * len(neo4j_pass) if neo4j_pass else 'NOT SET'}")
    
    if neo4j_pass == 'password':
        print("⚠️  Using default password. Consider changing for production.")
    
    return True

def test_hybrid_episodes_offline():
    """Test hybrid episode functionality without API calls."""
    print("\n🧪 Testing Hybrid Episode Functionality (Offline Mode)...")
    
    try:
        # Test imports
        from morag_graph.graphiti import (
            DocumentEpisodeMapper, EpisodeStrategy, ContextLevel,
            create_hybrid_episode_mapper
        )
        print("✅ Hybrid episode imports successful")
        
        # Test enum values
        strategies = [s.value for s in EpisodeStrategy]
        contexts = [c.value for c in ContextLevel]
        
        print(f"✅ Episode strategies available: {', '.join(strategies)}")
        print(f"✅ Context levels available: {', '.join(contexts)}")
        
        # Test mapper creation (without connection)
        mapper = DocumentEpisodeMapper(
            config=None,  # No config for offline test
            strategy=EpisodeStrategy.HYBRID,
            context_level=ContextLevel.RICH,
            enable_ai_summarization=False  # Disable for offline test
        )
        print("✅ Hybrid episode mapper created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing hybrid episodes: {e}")
        return False

def show_usage_examples():
    """Show usage examples for hybrid episodes."""
    print("\n📚 Hybrid Episode Usage Examples:")
    print("\n1. Basic hybrid ingestion (recommended):")
    print("   python cli/test-graphiti.py ingest your_document.pdf")
    
    print("\n2. Contextual chunks with rich context:")
    print("   python cli/test-graphiti.py ingest document.pdf \\")
    print("     --episode-strategy contextual_chunks \\")
    print("     --context-level comprehensive")
    
    print("\n3. Document processing with hybrid strategy:")
    print("   python cli/test-document.py document.pdf --graphiti \\")
    print("     --episode-strategy hybrid \\")
    print("     --context-level rich \\")
    print("     --language de")
    
    print("\n4. Fast processing (minimal AI):")
    print("   python cli/test-graphiti.py ingest document.pdf \\")
    print("     --episode-strategy document_only \\")
    print("     --context-level minimal \\")
    print("     --disable-ai-summarization")
    
    print("\n5. Custom episode naming:")
    print("   python cli/test-graphiti.py ingest report.pdf \\")
    print("     --episode-prefix 'Q1_2024_report' \\")
    print("     --context-level rich")

def main():
    """Main setup and test function."""
    print("🚀 MoRAG Hybrid Episode Setup & Test")
    print("=" * 50)
    
    # Check API keys
    api_configured = check_api_keys()
    
    # Check Neo4j
    check_neo4j_config()
    
    # Test functionality
    functionality_ok = test_hybrid_episodes_offline()
    
    # Show examples
    show_usage_examples()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    
    if api_configured and functionality_ok:
        print("✅ Ready to use hybrid episodes!")
        print("\n🎯 Try this command with your document:")
        print("   python cli/test-document.py 'path/to/your/document.pdf' --graphiti --language de")
        
    elif functionality_ok and not api_configured:
        print("⚠️  Hybrid episodes are installed but need API key configuration")
        print("   You can test with --disable-ai-summarization for now")
        print("\n🧪 Test without AI:")
        print("   python cli/test-graphiti.py ingest document.pdf \\")
        print("     --disable-ai-summarization --context-level minimal")
        
    else:
        print("❌ Issues found. Please check the errors above.")
    
    print("\n📖 For more details, see: CLI_HYBRID_EPISODES.md")

if __name__ == "__main__":
    main()
