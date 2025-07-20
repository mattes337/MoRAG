#!/usr/bin/env python3
"""
Fix Neo4j authentication by setting the correct environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key

def fix_neo4j_auth():
    """Fix Neo4j authentication configuration."""
    
    print("🔧 Fixing Neo4j Authentication Configuration")
    print("=" * 50)
    
    # Load current .env file
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Found .env file: {env_file.absolute()}")
    else:
        print(f"⚠️  .env file not found, will create: {env_file.absolute()}")
    
    # Neo4j configuration based on your settings
    neo4j_config = {
        "GRAPHITI_NEO4J_URI": "bolt://localhost:7687",
        "GRAPHITI_NEO4J_USERNAME": "neo4j", 
        "GRAPHITI_NEO4J_PASSWORD": "password",
        "GRAPHITI_NEO4J_DATABASE": "neo4j",  # You mentioned database is "neo4j"
        
        # Also set the alternative variable names for compatibility
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password",
        "NEO4J_DATABASE": "neo4j"
    }
    
    print("\n🔄 Setting Neo4j environment variables...")
    
    # Set each variable in the .env file
    for key, value in neo4j_config.items():
        try:
            set_key(env_file, key, value)
            print(f"✅ Set {key}={value}")
        except Exception as e:
            print(f"❌ Failed to set {key}: {e}")
    
    # Reload environment
    load_dotenv(env_file, override=True)
    
    print("\n🔍 Verifying configuration...")
    
    # Check if variables are set correctly
    for key, expected_value in neo4j_config.items():
        actual_value = os.getenv(key)
        if actual_value == expected_value:
            print(f"✅ {key}: {actual_value}")
        else:
            print(f"❌ {key}: Expected '{expected_value}', got '{actual_value}'")
    
    print("\n🧪 Testing Graphiti configuration...")
    
    try:
        from morag_graph.graphiti.config import GraphitiConfig
        
        # Test configuration creation
        config = GraphitiConfig(
            neo4j_uri=os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("GRAPHITI_NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("GRAPHITI_NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("GRAPHITI_NEO4J_DATABASE", "neo4j"),
            openai_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )
        
        print(f"✅ GraphitiConfig created successfully")
        print(f"   Neo4j URI: {config.neo4j_uri}")
        print(f"   Neo4j Username: {config.neo4j_username}")
        print(f"   Neo4j Password: {'*' * len(config.neo4j_password) if config.neo4j_password else 'NOT SET'}")
        print(f"   Neo4j Database: {config.neo4j_database}")
        
    except Exception as e:
        print(f"❌ Error creating GraphitiConfig: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Neo4j authentication configuration complete!")
    print("\n🚀 Now try running your command again:")
    print('   python cli/test-document.py "D:\\Morag\\lessenig-entgiftung.pdf" --graphiti --language de --episode-strategy hybrid --context-level rich')

if __name__ == "__main__":
    fix_neo4j_auth()
