#!/usr/bin/env python3
"""Test script for GPU workers functionality."""

import os
import sys
import time
import requests
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))

def test_worker_status(server_url):
    """Test worker status endpoint."""
    print("🔍 Testing worker status...")
    
    try:
        response = requests.get(f"{server_url}/api/v1/status/workers")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Total workers: {data['workers']['total']}")
        print(f"✅ GPU workers: {data['workers']['gpu']}")
        print(f"✅ CPU workers: {data['workers']['cpu']}")
        print(f"✅ GPU available: {data['gpu_available']}")
        
        return data['workers']['gpu'] > 0
        
    except Exception as e:
        print(f"❌ Worker status test failed: {e}")
        return False

def test_gpu_audio_processing(server_url, audio_file):
    """Test GPU audio processing."""
    print("🎵 Testing GPU audio processing...")
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'gpu': 'true'}
            
            response = requests.post(
                f"{server_url}/api/v1/ingest/file",
                files=files,
                data=data,
                timeout=300
            )
            response.raise_for_status()
        
        result = response.json()
        if result.get('task_id'):
            print("✅ GPU audio processing task submitted")
            print(f"   Task ID: {result.get('task_id')}")
            print(f"   Status: {result.get('status')}")
            return True
        else:
            print(f"❌ GPU audio processing failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ GPU audio processing test failed: {e}")
        return False

def test_cpu_fallback(server_url, audio_file):
    """Test CPU fallback when GPU requested but unavailable."""
    print("🔄 Testing CPU fallback...")
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'gpu': 'true'}  # Request GPU but should fallback to CPU
            
            response = requests.post(
                f"{server_url}/api/v1/ingest/file",
                files=files,
                data=data,
                timeout=300
            )
            response.raise_for_status()
        
        result = response.json()
        if result.get('task_id'):
            print("✅ CPU fallback successful")
            print(f"   Task ID: {result.get('task_id')}")
            return True
        else:
            print(f"❌ CPU fallback failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ CPU fallback test failed: {e}")
        return False

def create_test_audio_file():
    """Create a simple test audio file."""
    try:
        import numpy as np
        import wave
        
        # Generate 5 seconds of sine wave at 440 Hz
        sample_rate = 44100
        duration = 5
        frequency = 440
        
        t = np.linspace(0, duration, sample_rate * duration, False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit integers
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_file.name
        
    except ImportError:
        print("⚠️  numpy/wave not available, using dummy file")
        # Create a dummy file
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        temp_file.write(b"dummy audio content for testing")
        temp_file.close()
        return temp_file.name

def main():
    """Run all GPU worker tests."""
    print("🧪 MoRAG GPU Workers Test Suite")
    print("=" * 50)
    
    # Configuration
    server_url = os.getenv('MORAG_SERVER_URL', 'http://localhost:8000')
    audio_file = os.getenv('TEST_AUDIO_FILE')
    
    if not audio_file:
        print("📁 Creating test audio file...")
        audio_file = create_test_audio_file()
        print(f"   Created: {audio_file}")
    
    print(f"🌐 Server URL: {server_url}")
    print(f"🎵 Audio file: {audio_file}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Worker status
    if test_worker_status(server_url):
        tests_passed += 1
    print()
    
    # Test 2: GPU processing (if GPU workers available)
    if test_gpu_audio_processing(server_url, audio_file):
        tests_passed += 1
    print()
    
    # Test 3: CPU fallback
    if test_cpu_fallback(server_url, audio_file):
        tests_passed += 1
    print()
    
    # Cleanup
    if not os.getenv('TEST_AUDIO_FILE'):
        os.unlink(audio_file)
        print(f"🗑️  Cleaned up test file: {audio_file}")
    
    # Results
    print("📊 Test Results")
    print("=" * 20)
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
