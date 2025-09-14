#!/usr/bin/env python3
"""
Test script to validate Qdrant collection name unification.

This script tests that:
1. All components fail fast when QDRANT_COLLECTION_NAME is not provided
2. All components use the same collection name when provided
3. No default values are used anywhere
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_core_config_requires_collection_name():
    """Test that core config fails when collection name is not provided."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove any existing QDRANT_COLLECTION_NAME
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            from packages.morag_core.src.morag_core.config import Settings
            # This should raise ValueError
            settings = Settings()
            assert False, "Expected ValueError for missing QDRANT_COLLECTION_NAME"
        except ValueError as e:
            assert "QDRANT_COLLECTION_NAME environment variable is required" in str(e)
        except ImportError:
            # Skip if package not available
            pytest.skip("morag-core package not available")


def test_core_config_accepts_collection_name():
    """Test that core config works when collection name is provided."""
    with patch.dict(os.environ, {'QDRANT_COLLECTION_NAME': 'test_collection'}):
        try:
            from packages.morag_core.src.morag_core.config import Settings
            settings = Settings()
            assert settings.qdrant_collection_name == 'test_collection'
        except ImportError:
            pytest.skip("morag-core package not available")


def test_qdrant_storage_requires_collection_name():
    """Test that QdrantVectorStorage fails when collection name is not provided."""
    try:
        from packages.morag_services.src.morag_services.storage import QdrantVectorStorage
        
        # This should raise ValueError
        try:
            storage = QdrantVectorStorage(collection_name=None)
            assert False, "Expected ValueError for missing collection_name"
        except ValueError as e:
            assert "collection_name is required" in str(e)
        
        # This should also raise ValueError
        try:
            storage = QdrantVectorStorage(collection_name="")
            assert False, "Expected ValueError for empty collection_name"
        except ValueError as e:
            assert "collection_name is required" in str(e)
            
    except ImportError:
        pytest.skip("morag-services package not available")


def test_qdrant_storage_accepts_collection_name():
    """Test that QdrantVectorStorage works when collection name is provided."""
    try:
        from packages.morag_services.src.morag_services.storage import QdrantVectorStorage
        
        storage = QdrantVectorStorage(collection_name='test_collection')
        assert storage.collection_name == 'test_collection'
        
    except ImportError:
        pytest.skip("morag-services package not available")


def test_services_requires_collection_name():
    """Test that MoRAGServices fails when QDRANT_COLLECTION_NAME is not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove any existing QDRANT_COLLECTION_NAME
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            from packages.morag_services.src.morag_services.services import MoRAGServices
            
            services = MoRAGServices()
            # This should raise ValueError when trying to initialize search services
            try:
                services._initialize_search_services()
                assert False, "Expected ValueError for missing QDRANT_COLLECTION_NAME"
            except ValueError as e:
                assert "QDRANT_COLLECTION_NAME environment variable is required" in str(e)
                
        except ImportError:
            pytest.skip("morag-services package not available")


def test_services_accepts_collection_name():
    """Test that MoRAGServices works when QDRANT_COLLECTION_NAME is set."""
    with patch.dict(os.environ, {'QDRANT_COLLECTION_NAME': 'test_collection'}):
        try:
            from packages.morag_services.src.morag_services.services import MoRAGServices
            
            services = MoRAGServices()
            # Mock the QdrantVectorStorage to avoid actual connection
            with patch('packages.morag_services.src.morag_services.services.QdrantVectorStorage') as mock_storage:
                services._initialize_search_services()
                # Verify that QdrantVectorStorage was called with the correct collection name
                mock_storage.assert_called_once()
                call_kwargs = mock_storage.call_args[1]
                assert call_kwargs['collection_name'] == 'test_collection'
                
        except ImportError:
            pytest.skip("morag-services package not available")


def test_ingest_tasks_requires_collection_name():
    """Test that ingest tasks fail when QDRANT_COLLECTION_NAME is not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove any existing QDRANT_COLLECTION_NAME
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            from packages.morag.src.morag.ingest_tasks import ingest_file_task
            
            # Mock the file and other dependencies
            with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
                temp_file.write(b"test content")
                temp_file.flush()
                
                # This should raise ValueError
                try:
                    # Use asyncio.run to handle the async function
                    import asyncio
                    asyncio.run(ingest_file_task(
                        temp_file.name,
                        source_type='document',
                        task_options={}
                    ))
                    assert False, "Expected ValueError for missing QDRANT_COLLECTION_NAME"
                except ValueError as e:
                    assert "QDRANT_COLLECTION_NAME environment variable is required" in str(e)
                    
        except ImportError:
            pytest.skip("morag package not available")


def test_collection_name_consistency():
    """Test that all components use the same collection name when provided."""
    test_collection = 'unified_test_collection'
    
    with patch.dict(os.environ, {'QDRANT_COLLECTION_NAME': test_collection}):
        # Test core config
        try:
            from packages.morag_core.src.morag_core.config import Settings
            settings = Settings()
            assert settings.qdrant_collection_name == test_collection
        except ImportError:
            pass
        
        # Test storage
        try:
            from packages.morag_services.src.morag_services.storage import QdrantVectorStorage
            storage = QdrantVectorStorage(collection_name=test_collection)
            assert storage.collection_name == test_collection
        except ImportError:
            pass
        
        # Test services initialization
        try:
            from packages.morag_services.src.morag_services.services import MoRAGServices
            services = MoRAGServices()
            with patch('packages.morag_services.src.morag_services.services.QdrantVectorStorage') as mock_storage:
                services._initialize_search_services()
                call_kwargs = mock_storage.call_args[1]
                assert call_kwargs['collection_name'] == test_collection
        except ImportError:
            pass


def test_environment_files_consistency():
    """Test that environment example files use consistent collection names."""
    env_files = [
        '.env.example',
        '.env.prod.example'
    ]
    
    for env_file in env_files:
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                content = f.read()
                # Check that the file contains QDRANT_COLLECTION_NAME=morag_documents
                assert 'QDRANT_COLLECTION_NAME=morag_documents' in content, f"Inconsistent collection name in {env_file}"


if __name__ == "__main__":
    print("Testing Qdrant collection name unification...")
    
    # Run the tests
    test_functions = [
        test_core_config_requires_collection_name,
        test_core_config_accepts_collection_name,
        test_qdrant_storage_requires_collection_name,
        test_qdrant_storage_accepts_collection_name,
        test_services_requires_collection_name,
        test_services_accepts_collection_name,
        test_ingest_tasks_requires_collection_name,
        test_collection_name_consistency,
        test_environment_files_consistency
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✅ {test_func.__name__} PASSED")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⏭️  {test_func.__name__} SKIPPED: {e}")
            skipped += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("✅ All tests passed! Qdrant collection name unification is working correctly.")
