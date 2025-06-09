#!/usr/bin/env python3
"""Test script for API key service functionality."""

import asyncio
import redis
import os
import sys
from pathlib import Path

# Add the morag package to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))

# Import the auth service directly to avoid full MoRAG dependencies
try:
    from morag.services.auth_service import APIKeyService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This test requires the auth service to be properly installed.")
    sys.exit(1)

async def test_api_key_service():
    """Test API key service functionality."""
    print("Testing API Key Service...")
    
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
    service = APIKeyService(redis_client)
    
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
        
        print(f"✅ GPU queue: {gpu_queue}")
        print(f"✅ CPU queue: {cpu_queue}")
        print(f"✅ Default queue: {default_queue}")
        
        # Test 4: List user API keys
        print("\n4. Testing API key listing...")
        keys = await service.list_user_api_keys('user123')
        if len(keys) >= 1:
            print(f"✅ Found {len(keys)} API key(s) for user123")
        else:
            print(f"❌ Expected at least 1 API key, found {len(keys)}")
            return False
        
        # Test 5: Invalid API key
        print("\n5. Testing invalid API key...")
        invalid_data = await service.validate_api_key('invalid_key_12345')
        if invalid_data is None:
            print("✅ Invalid API key correctly rejected")
        else:
            print(f"❌ Invalid API key incorrectly accepted: {invalid_data}")
            return False
        
        # Test 6: Revoke API key
        print("\n6. Testing API key revocation...")
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

async def test_queue_functionality():
    """Test queue functionality with Celery."""
    print("\n\nTesting Queue Functionality...")

    try:
        # Test queue name generation without importing full worker
        service = APIKeyService(redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0')))

        # Test 1: Get queue names for different scenarios
        print("\n1. Testing queue name generation...")
        gpu_queue = service.get_user_queue_name('user123', 'gpu')
        cpu_queue = service.get_cpu_queue_name('user123')
        default_queue = service.get_default_queue_name()

        expected_gpu = 'gpu-tasks-user123'
        expected_cpu = 'cpu-tasks-user123'
        expected_default = 'celery'

        if gpu_queue == expected_gpu:
            print(f"✅ GPU queue name correct: {gpu_queue}")
        else:
            print(f"❌ GPU queue name wrong: expected {expected_gpu}, got {gpu_queue}")
            return False

        if cpu_queue == expected_cpu:
            print(f"✅ CPU queue name correct: {cpu_queue}")
        else:
            print(f"❌ CPU queue name wrong: expected {expected_cpu}, got {cpu_queue}")
            return False

        if default_queue == expected_default:
            print(f"✅ Default queue name correct: {default_queue}")
        else:
            print(f"❌ Default queue name wrong: expected {expected_default}, got {default_queue}")
            return False

        print("\n🎉 All queue functionality tests passed!")
        return True

    except Exception as e:
        print(f"❌ Queue test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Remote GPU Workers - API Key Service Tests\n")
    
    # Test API key service
    api_key_success = await test_api_key_service()
    
    # Test queue functionality
    queue_success = await test_queue_functionality()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"API Key Service: {'✅ PASS' if api_key_success else '❌ FAIL'}")
    print(f"Queue Functionality: {'✅ PASS' if queue_success else '❌ FAIL'}")
    
    if api_key_success and queue_success:
        print("\n🎉 ALL TESTS PASSED - Task 1 implementation successful!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED - Check implementation")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
