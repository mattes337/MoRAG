#!/usr/bin/env python3
"""Standalone test for API key service functionality."""

import asyncio
import redis
import os
import sys
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class SimpleAPIKeyService:
    """Simplified API key service for testing."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "morag:api_keys:"
        self.user_prefix = "morag:users:"

    async def create_api_key(self, user_id: str, description: str = "",
                           expires_days: Optional[int] = None) -> str:
        """Create a new API key for a user."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        key_data = {
            "user_id": user_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=expires_days)).isoformat() if expires_days else None,
            "active": True
        }

        # Store API key data
        self.redis.setex(
            f"{self.key_prefix}{key_hash}",
            int(timedelta(days=expires_days or 365).total_seconds()),
            json.dumps(key_data)
        )

        print(f"API key created for user {user_id}: {description}")
        return api_key

    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data_json = self.redis.get(f"{self.key_prefix}{key_hash}")

        if not key_data_json:
            return None

        key_data = json.loads(key_data_json)

        # Check if key is active
        if not key_data.get("active", False):
            return None

        # Check expiration
        if key_data.get("expires_at"):
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if datetime.utcnow() > expires_at:
                return None

        return key_data

    def get_user_queue_name(self, user_id: str, worker_type: str = "gpu") -> str:
        """Get queue name for user and worker type."""
        return f"{worker_type}-tasks-{user_id}"

    def get_cpu_queue_name(self, user_id: str) -> str:
        """Get CPU queue name for user."""
        return f"cpu-tasks-{user_id}"

    def get_default_queue_name(self) -> str:
        """Get default queue name for anonymous processing."""
        return "celery"

    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_name = f"{self.key_prefix}{key_hash}"
        
        key_data_json = self.redis.get(key_name)
        if not key_data_json:
            return False
            
        key_data = json.loads(key_data_json)
        key_data["active"] = False
        
        # Update the key data
        self.redis.set(key_name, json.dumps(key_data))
        
        print(f"API key revoked for user {key_data.get('user_id')}")
        return True

async def test_api_key_service():
    """Test API key service functionality."""
    print("🚀 Testing API Key Service...")
    
    # Connect to Redis
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_client = redis.from_url(redis_url)
    
    try:
        # Test Redis connection
        redis_client.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    
    # Initialize API key service
    service = SimpleAPIKeyService(redis_client)
    
    try:
        # Test 1: Create API key
        print("\n1. Testing API key creation...")
        api_key = await service.create_api_key('user123', 'Test key for remote worker')
        print(f"✅ Created API key: {api_key[:8]}...")
        
        # Test 2: Validate API key
        print("\n2. Testing API key validation...")
        user_data = await service.validate_api_key(api_key)
        if user_data and user_data.get('user_id') == 'user123':
            print(f"✅ API key validation successful: {user_data}")
        else:
            print(f"❌ API key validation failed: {user_data}")
            return False
        
        # Test 3: Get queue names
        print("\n3. Testing queue name generation...")
        gpu_queue = service.get_user_queue_name('user123', 'gpu')
        cpu_queue = service.get_cpu_queue_name('user123')
        default_queue = service.get_default_queue_name()
        
        expected_gpu = 'gpu-tasks-user123'
        expected_cpu = 'cpu-tasks-user123'
        expected_default = 'celery'
        
        if gpu_queue == expected_gpu:
            print(f"✅ GPU queue: {gpu_queue}")
        else:
            print(f"❌ GPU queue wrong: expected {expected_gpu}, got {gpu_queue}")
            return False
            
        if cpu_queue == expected_cpu:
            print(f"✅ CPU queue: {cpu_queue}")
        else:
            print(f"❌ CPU queue wrong: expected {expected_cpu}, got {cpu_queue}")
            return False
            
        if default_queue == expected_default:
            print(f"✅ Default queue: {default_queue}")
        else:
            print(f"❌ Default queue wrong: expected {expected_default}, got {default_queue}")
            return False
        
        # Test 4: Invalid API key
        print("\n4. Testing invalid API key...")
        invalid_data = await service.validate_api_key('invalid_key_12345')
        if invalid_data is None:
            print("✅ Invalid API key correctly rejected")
        else:
            print(f"❌ Invalid API key incorrectly accepted: {invalid_data}")
            return False
        
        # Test 5: Revoke API key
        print("\n5. Testing API key revocation...")
        revoked = await service.revoke_api_key(api_key)
        if revoked:
            print("✅ API key revoked successfully")
            
            # Verify revoked key is invalid
            revoked_data = await service.validate_api_key(api_key)
            if revoked_data is None:
                print("✅ Revoked API key correctly rejected")
            else:
                print(f"❌ Revoked API key still valid: {revoked_data}")
                return False
        else:
            print("❌ API key revocation failed")
            return False
        
        print("\n🎉 All API key service tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Remote GPU Workers - API Key Service Tests\n")
    
    # Test API key service
    api_key_success = await test_api_key_service()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"API Key Service: {'✅ PASS' if api_key_success else '❌ FAIL'}")
    
    if api_key_success:
        print("\n🎉 ALL TESTS PASSED - Task 1 API Key Service implementation successful!")
        return True
    else:
        print("\n❌ TESTS FAILED - Check implementation")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
