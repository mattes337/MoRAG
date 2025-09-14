#!/usr/bin/env python3
"""Test script for MoRAG Remote Converter."""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import RemoteConverterConfig, setup_logging
from remote_converter import RemoteConverter


def test_config_creation():
    """Test configuration creation and validation."""
    print("🧪 Testing configuration creation...")
    
    # Test sample config creation
    config_manager = RemoteConverterConfig()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_config_file = f.name
    
    try:
        success = config_manager.create_sample_config(temp_config_file)
        assert success, "Failed to create sample configuration"
        
        # Test loading the created config
        config_manager2 = RemoteConverterConfig(temp_config_file)
        assert config_manager2.validate_config(), "Configuration validation failed"
        
        print("✅ Configuration creation test passed")
        return True
        
    finally:
        if os.path.exists(temp_config_file):
            os.unlink(temp_config_file)


def test_converter_initialization():
    """Test remote converter initialization."""
    print("🧪 Testing converter initialization...")
    
    config = {
        'worker_id': 'test-worker',
        'api_base_url': 'http://localhost:8000',
        'content_types': ['audio', 'video'],
        'poll_interval': 5,
        'max_concurrent_jobs': 1,
        'temp_dir': tempfile.mkdtemp()
    }
    
    try:
        # Mock the processors to avoid import issues
        with patch('remote_converter.AudioProcessor'), \
             patch('remote_converter.VideoProcessor'), \
             patch('remote_converter.DocumentProcessor'), \
             patch('remote_converter.ImageProcessor'), \
             patch('remote_converter.WebProcessor'), \
             patch('remote_converter.YouTubeProcessor'):
            
            converter = RemoteConverter(config)
            
            assert converter.worker_id == 'test-worker'
            assert converter.api_base_url == 'http://localhost:8000'
            assert converter.content_types == ['audio', 'video']
            assert converter.poll_interval == 5
            assert converter.max_concurrent_jobs == 1
            
            print("✅ Converter initialization test passed")
            return True
            
    except Exception as e:
        print(f"❌ Converter initialization test failed: {e}")
        return False
    
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(config['temp_dir']):
            shutil.rmtree(config['temp_dir'])


def test_connection_testing():
    """Test API connection testing."""
    print("🧪 Testing connection testing...")
    
    config = {
        'worker_id': 'test-worker',
        'api_base_url': 'http://localhost:8000',
        'content_types': ['audio'],
        'poll_interval': 5,
        'max_concurrent_jobs': 1,
        'temp_dir': tempfile.mkdtemp()
    }
    
    try:
        with patch('remote_converter.AudioProcessor'), \
             patch('requests.get') as mock_get:
            
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            converter = RemoteConverter(config)
            result = converter.test_connection()
            
            assert result == True, "Connection test should succeed with mocked response"
            
            # Mock failed response
            mock_response.status_code = 500
            result = converter.test_connection()
            
            assert result == False, "Connection test should fail with error response"
            
            print("✅ Connection testing test passed")
            return True
            
    except Exception as e:
        print(f"❌ Connection testing test failed: {e}")
        return False
    
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(config['temp_dir']):
            shutil.rmtree(config['temp_dir'])


def test_job_polling():
    """Test job polling functionality."""
    print("🧪 Testing job polling...")
    
    config = {
        'worker_id': 'test-worker',
        'api_base_url': 'http://localhost:8000',
        'content_types': ['audio'],
        'poll_interval': 5,
        'max_concurrent_jobs': 1,
        'temp_dir': tempfile.mkdtemp()
    }
    
    try:
        with patch('remote_converter.AudioProcessor'), \
             patch('requests.get') as mock_get:
            
            # Mock job available response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'job_id': 'test-job-123',
                'content_type': 'audio',
                'source_file_url': '/api/v1/remote-jobs/test-job-123/download',
                'task_options': {}
            }
            mock_get.return_value = mock_response
            
            converter = RemoteConverter(config)
            
            # Test polling (this is async, so we need to run it)
            import asyncio
            job = asyncio.run(converter._poll_for_job())
            
            assert job is not None, "Should receive a job"
            assert job['job_id'] == 'test-job-123', "Job ID should match"
            assert job['content_type'] == 'audio', "Content type should match"
            
            print("✅ Job polling test passed")
            return True
            
    except Exception as e:
        print(f"❌ Job polling test failed: {e}")
        return False
    
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(config['temp_dir']):
            shutil.rmtree(config['temp_dir'])


def main():
    """Run all tests."""
    print("🚀 MoRAG Remote Converter Test Suite")
    print("=" * 50)
    
    # Set up logging
    setup_logging('INFO')
    
    tests = [
        test_config_creation,
        test_converter_initialization,
        test_connection_testing,
        test_job_polling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
