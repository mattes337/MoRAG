#!/usr/bin/env python3
"""
Debug environment variable loading to find the mismatch.
"""

import os
from dotenv import load_dotenv

def debug_env_vars():
    """Debug environment variable loading."""
    
    print("🔍 Debugging environment variable loading")
    print("=" * 50)
    
    # Check what's in the environment before loading .env
    print("📋 GEMINI_API_KEY before loading .env:")
    print(f"   {os.getenv('GEMINI_API_KEY', 'NOT SET')}")
    
    # Load .env file
    print("\n🔄 Loading .env file...")
    load_dotenv()
    
    # Check what's in the environment after loading .env
    print("📋 GEMINI_API_KEY after loading .env:")
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"   {gemini_key[:10]}...{gemini_key[-4:]}")
        print(f"   Full length: {len(gemini_key)} characters")
    else:
        print("   NOT SET")
    
    # Check if there are multiple .env files
    print("\n🔍 Checking for .env files:")
    env_files = ['.env', '.env.local', '.env.development', '.env.production']
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"   ✅ Found: {env_file}")
        else:
            print(f"   ❌ Not found: {env_file}")
    
    # Read .env file directly
    print("\n📖 Reading .env file directly:")
    try:
        with open('.env', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if 'GEMINI_API_KEY' in line:
                    print(f"   Line {i}: {line.strip()}")
    except Exception as e:
        print(f"   ❌ Error reading .env: {e}")
    
    # Check all environment variables with 'GEMINI' in the name
    print("\n🔍 All GEMINI-related environment variables:")
    for key, value in os.environ.items():
        if 'GEMINI' in key.upper():
            if 'API_KEY' in key.upper():
                print(f"   {key}: {value[:10]}...{value[-4:] if len(value) > 10 else value}")
            else:
                print(f"   {key}: {value}")

if __name__ == "__main__":
    debug_env_vars()
