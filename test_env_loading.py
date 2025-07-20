#!/usr/bin/env python3
"""
Test environment variable loading to debug the Gemini API key issue.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def test_env_loading():
    """Test environment variable loading step by step."""
    
    print("🔍 Testing Environment Variable Loading")
    print("=" * 50)
    
    # Check current working directory
    print(f"📁 Current working directory: {os.getcwd()}")
    
    # Check if .env file exists
    env_file = Path(".env")
    print(f"📄 .env file exists: {env_file.exists()}")
    
    if env_file.exists():
        print(f"📄 .env file path: {env_file.absolute()}")
        
        # Read .env file content
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check if GEMINI_API_KEY is in the file
        if "GEMINI_API_KEY" in content:
            print("✅ GEMINI_API_KEY found in .env file")
            
            # Extract the line with GEMINI_API_KEY
            for line in content.split('\n'):
                if line.startswith('GEMINI_API_KEY'):
                    key_length = len(line.split('=')[1]) if '=' in line else 0
                    print(f"📋 .env file line: GEMINI_API_KEY=***{key_length} chars***")
                    break
        else:
            print("❌ GEMINI_API_KEY not found in .env file")
    
    # Check environment before loading .env
    print(f"\n🔍 GEMINI_API_KEY before load_dotenv(): {os.getenv('GEMINI_API_KEY', 'NOT SET')}")
    
    # Load .env file
    print("\n🔄 Loading .env file...")
    result = load_dotenv(env_file, override=True)
    print(f"📋 load_dotenv() result: {result}")
    
    # Check environment after loading .env
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"✅ GEMINI_API_KEY after load_dotenv(): {gemini_key[:10]}...{gemini_key[-4:]} ({len(gemini_key)} chars)")
    else:
        print("❌ GEMINI_API_KEY still not set after load_dotenv()")
    
    # Test Graphiti config loading
    print("\n🔄 Testing Graphiti config loading...")
    try:
        from morag_graph.graphiti.config import GraphitiConfig
        
        # Try to create config
        config = GraphitiConfig()
        
        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            print(f"✅ Graphiti config loaded API key: {config.openai_api_key[:10]}...{config.openai_api_key[-4:]}")
        else:
            print("❌ Graphiti config did not load API key")
            
    except Exception as e:
        print(f"❌ Error loading Graphiti config: {e}")
    
    # Test direct Graphiti instance creation
    print("\n🔄 Testing Graphiti instance creation...")
    try:
        from morag_graph.graphiti.config import create_graphiti_instance
        
        graphiti = create_graphiti_instance()
        print("✅ Graphiti instance created successfully")
        
    except Exception as e:
        print(f"❌ Error creating Graphiti instance: {e}")

if __name__ == "__main__":
    test_env_loading()
