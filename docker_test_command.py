#!/usr/bin/env python3
"""
Simple test command that can be run in Docker to verify the fix.
This is designed to be run as a Docker command to test the environment.
"""

import os
import sys


def main():
    """Simple test that can be run in Docker."""
    print("Docker Environment Test")
    print("=" * 30)
    
    # Test 1: Check critical environment variable
    collection_name = os.getenv('QDRANT_COLLECTION_NAME')
    if collection_name:
        print(f"✅ QDRANT_COLLECTION_NAME: {collection_name}")
    else:
        print("❌ QDRANT_COLLECTION_NAME: NOT SET")
        return False
    
    # Test 2: Try to import settings
    try:
        print("Testing settings import...")
        from morag_core.config import settings
        actual_name = settings.qdrant_collection_name
        print(f"✅ Settings loaded: {actual_name}")
    except Exception as e:
        print(f"❌ Settings failed: {e}")
        return False
    
    # Test 3: Try to import worker
    try:
        print("Testing worker import...")
        import morag.worker
        print("✅ Worker imported successfully")
    except Exception as e:
        print(f"❌ Worker import failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("Docker container environment is working correctly!")
        sys.exit(0)
    else:
        print("Docker container environment has issues!")
        sys.exit(1)
