#!/usr/bin/env python3
"""Test script for default API key initialization."""

import asyncio
import os
import sys
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))

from morag.services.auth_service import APIKeyService
from morag.server import initialize_default_api_key


async def test_default_api_key_initialization():
    """Test the default API key initialization logic."""
    print("🧪 Testing Default API Key Initialization")
    print("=" * 50)
    
    # Test 1: No environment variable set
    print("\n1. Testing with no MORAG_API_KEY environment variable...")
    if 'MORAG_API_KEY' in os.environ:
        del os.environ['MORAG_API_KEY']
    
    api_key_service = APIKeyService()
    await initialize_default_api_key(api_key_service)
    
    # Check if a default key was created
    if api_key_service._api_keys:
        print("✅ Default API key created successfully")
        for key_hash, key_data in api_key_service._api_keys.items():
            print(f"   User ID: {key_data['user_id']}")
            print(f"   Description: {key_data['description']}")
            print(f"   Active: {key_data['active']}")
    else:
        print("❌ No default API key was created")
    
    # Test 2: With placeholder environment variable
    print("\n2. Testing with placeholder MORAG_API_KEY...")
    os.environ['MORAG_API_KEY'] = 'morag-default-api-key-change-me-in-production'
    
    api_key_service2 = APIKeyService()
    await initialize_default_api_key(api_key_service2)
    
    # Check if both keys exist (generated + placeholder)
    print(f"   Total API keys registered: {len(api_key_service2._api_keys)}")
    
    # Test validation of the placeholder key
    user_data = await api_key_service2.validate_api_key('morag-default-api-key-change-me-in-production')
    if user_data:
        print("✅ Placeholder API key is valid")
        print(f"   User ID: {user_data['user_id']}")
    else:
        print("❌ Placeholder API key validation failed")
    
    # Test 3: With custom environment variable
    print("\n3. Testing with custom MORAG_API_KEY...")
    custom_key = "custom-test-api-key-12345"
    os.environ['MORAG_API_KEY'] = custom_key
    
    api_key_service3 = APIKeyService()
    # First create the custom key manually
    await api_key_service3.create_api_key("test_user", "Custom test key")
    
    # Now test initialization
    await initialize_default_api_key(api_key_service3)
    
    print(f"   Total API keys after custom key test: {len(api_key_service3._api_keys)}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_default_api_key_initialization())
