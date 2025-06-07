#!/usr/bin/env python3
"""
Test script for GPU worker configuration validation.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def check_environment_variables(required_vars: List[str]) -> Dict[str, Any]:
    """Check if required environment variables are set."""
    results = {}
    for var in required_vars:
        value = os.environ.get(var)
        results[var] = {
            'set': value is not None,
            'value': value if value else 'Not set'
        }
    return results

def check_gpu_availability() -> Dict[str, Any]:
    """Check if GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return {
                'available': True,
                'info': result.stdout.strip(),
                'error': None
            }
        else:
            return {
                'available': False,
                'info': None,
                'error': result.stderr.strip()
            }
    except subprocess.TimeoutExpired:
        return {
            'available': False,
            'info': None,
            'error': 'nvidia-smi command timed out'
        }
    except FileNotFoundError:
        return {
            'available': False,
            'info': None,
            'error': 'nvidia-smi not found'
        }

def check_redis_connectivity(redis_url: str) -> Dict[str, Any]:
    """Check Redis connectivity."""
    try:
        result = subprocess.run(['redis-cli', '-u', redis_url, 'ping'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'PONG' in result.stdout:
            return {
                'connected': True,
                'response': result.stdout.strip(),
                'error': None
            }
        else:
            return {
                'connected': False,
                'response': None,
                'error': result.stderr.strip() or 'No PONG response'
            }
    except subprocess.TimeoutExpired:
        return {
            'connected': False,
            'response': None,
            'error': 'Redis connection timed out'
        }
    except FileNotFoundError:
        return {
            'connected': False,
            'response': None,
            'error': 'redis-cli not found'
        }

def check_file_access(temp_dir: str) -> Dict[str, Any]:
    """Check if temp directory is accessible."""
    if not temp_dir:
        return {
            'accessible': False,
            'error': 'TEMP_DIR not specified'
        }
    
    temp_path = Path(temp_dir)
    
    if not temp_path.exists():
        return {
            'accessible': False,
            'error': f'Directory does not exist: {temp_dir}'
        }
    
    if not temp_path.is_dir():
        return {
            'accessible': False,
            'error': f'Path is not a directory: {temp_dir}'
        }
    
    # Test write access
    try:
        test_file = temp_path / 'test_write_access.tmp'
        test_file.write_text('test')
        test_file.unlink()
        return {
            'accessible': True,
            'error': None
        }
    except Exception as e:
        return {
            'accessible': False,
            'error': f'Cannot write to directory: {str(e)}'
        }

def main():
    """Main test function."""
    print("🧪 GPU Worker Configuration Test")
    print("=" * 40)
    
    # Required environment variables
    required_vars = [
        'REDIS_URL',
        'QDRANT_URL',
        'QDRANT_COLLECTION_NAME',
        'GEMINI_API_KEY'
    ]
    
    # Check environment variables
    print("\n📋 Environment Variables:")
    env_results = check_environment_variables(required_vars)
    all_vars_set = True
    
    for var, result in env_results.items():
        status = "✅" if result['set'] else "❌"
        print(f"  {status} {var}: {result['value']}")
        if not result['set']:
            all_vars_set = False
    
    if not all_vars_set:
        print("\n❌ Some required environment variables are missing!")
        return False
    
    # Check GPU availability
    print("\n🔍 GPU Availability:")
    gpu_result = check_gpu_availability()
    if gpu_result['available']:
        print(f"  ✅ GPU detected: {gpu_result['info']}")
    else:
        print(f"  ⚠️  GPU not available: {gpu_result['error']}")
    
    # Check Redis connectivity
    print("\n🔗 Redis Connectivity:")
    redis_url = os.environ.get('REDIS_URL', '')
    if redis_url:
        redis_result = check_redis_connectivity(redis_url)
        if redis_result['connected']:
            print(f"  ✅ Redis connected: {redis_result['response']}")
        else:
            print(f"  ❌ Redis connection failed: {redis_result['error']}")
    else:
        print("  ❌ REDIS_URL not set")
        return False
    
    # Check file access
    print("\n📁 File Access:")
    temp_dir = os.environ.get('TEMP_DIR', '')
    if temp_dir:
        file_result = check_file_access(temp_dir)
        if file_result['accessible']:
            print(f"  ✅ Temp directory accessible: {temp_dir}")
        else:
            print(f"  ❌ Temp directory issue: {file_result['error']}")
    else:
        print("  ⚠️  TEMP_DIR not set (will use system default)")
    
    print("\n🎯 Configuration Test Complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
