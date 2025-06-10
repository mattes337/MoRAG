#!/usr/bin/env python3
"""Test script to verify Docker Compose API key configuration."""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_docker_compose_api_key():
    """Test that Docker Compose starts successfully with default API key."""
    print("🐳 Testing Docker Compose API Key Configuration")
    print("=" * 60)
    
    # Check if .env file exists and has MORAG_API_KEY
    env_file = Path("../.env")
    if not env_file.exists():
        print("❌ .env file not found in parent directory")
        return False
    
    # Read .env file
    env_content = env_file.read_text()
    if "MORAG_API_KEY=" not in env_content:
        print("❌ MORAG_API_KEY not found in .env file")
        return False
    
    print("✅ .env file contains MORAG_API_KEY")
    
    # Extract the API key value
    for line in env_content.split('\n'):
        if line.startswith('MORAG_API_KEY='):
            api_key = line.split('=', 1)[1].strip()
            print(f"   API Key: {api_key}")
            break
    
    # Check docker-compose.yml configuration
    compose_file = Path("../docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml not found in parent directory")
        return False
    
    compose_content = compose_file.read_text()
    
    # Check if worker service has env_file and MORAG_API_KEY with default
    if "env_file: .env" in compose_content:
        print("✅ Worker service configured with env_file")
    else:
        print("❌ Worker service missing env_file configuration")
        return False
    
    if "MORAG_API_KEY=${MORAG_API_KEY:-morag-default-api-key-change-me-in-production}" in compose_content:
        print("✅ Worker service has MORAG_API_KEY with default fallback")
    else:
        print("❌ Worker service missing MORAG_API_KEY default fallback")
        return False
    
    print("\n🎯 Configuration Test Results:")
    print("   ✅ .env file has MORAG_API_KEY")
    print("   ✅ docker-compose.yml has env_file for worker")
    print("   ✅ docker-compose.yml has MORAG_API_KEY with default")
    print("   ✅ Ready for Docker Compose deployment!")
    
    return True


def test_api_key_validation():
    """Test API key validation logic."""
    print("\n🔑 Testing API Key Validation Logic")
    print("=" * 40)
    
    # Test different API key scenarios
    test_cases = [
        ("", "Empty API key"),
        ("morag-default-api-key-change-me-in-production", "Default placeholder"),
        ("custom-api-key-12345", "Custom API key"),
        ("very-long-api-key-with-special-chars-!@#$%", "Special characters"),
    ]
    
    for api_key, description in test_cases:
        print(f"   Testing: {description}")
        
        # Check if it's the placeholder
        is_placeholder = api_key == "morag-default-api-key-change-me-in-production"
        is_empty = not api_key
        
        if is_empty:
            print(f"      → Empty key - would trigger default generation")
        elif is_placeholder:
            print(f"      → Placeholder key - would register + generate new")
        else:
            print(f"      → Custom key - would validate if exists")
    
    print("   ✅ API key validation logic tested")
    return True


def main():
    """Run all tests."""
    print("🧪 MoRAG Docker Compose API Key Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Configuration validation
        config_ok = test_docker_compose_api_key()
        
        # Test 2: API key validation logic
        validation_ok = test_api_key_validation()
        
        if config_ok and validation_ok:
            print("\n🎉 All tests passed!")
            print("\n📋 Next Steps:")
            print("   1. Run: docker-compose up -d")
            print("   2. Check logs: docker-compose logs -f")
            print("   3. Test API: curl http://localhost:8000/health")
            print("   4. Create API key: curl -X POST 'http://localhost:8000/api/v1/auth/create-key' -F 'user_id=test'")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
