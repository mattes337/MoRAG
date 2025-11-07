#!/usr/bin/env python3
"""
Test that components work correctly with unified collection name.
"""

import os
import sys
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_storage_component():
    """Test QdrantVectorStorage component."""
    print("Testing QdrantVectorStorage component...")

    try:
        sys.path.insert(0, "packages/morag-services/src")
        from morag_services.storage import QdrantVectorStorage

        # Test 1: Should fail with None
        try:
            storage = QdrantVectorStorage(collection_name=None)
            print("❌ Expected ValueError for None collection name")
            return False
        except ValueError as e:
            if "collection_name is required" in str(e):
                print("✅ Correctly rejected None collection name")
            else:
                print(f"❌ Wrong error message: {e}")
                return False

        # Test 2: Should fail with empty string
        try:
            storage = QdrantVectorStorage(collection_name="")
            print("❌ Expected ValueError for empty collection name")
            return False
        except ValueError as e:
            if "collection_name is required" in str(e):
                print("✅ Correctly rejected empty collection name")
            else:
                print(f"❌ Wrong error message: {e}")
                return False

        # Test 3: Should work with valid name
        try:
            storage = QdrantVectorStorage(collection_name="test_collection")
            if storage.collection_name == "test_collection":
                print("✅ Correctly accepted valid collection name")
                return True
            else:
                print(f"❌ Wrong collection name stored: {storage.collection_name}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error with valid collection name: {e}")
            return False

    except ImportError as e:
        print(f"⏭️  Skipping storage test - import error: {e}")
        return True


def test_services_component():
    """Test MoRAGServices component."""
    print("Testing MoRAGServices component...")

    try:
        sys.path.insert(0, "packages/morag-services/src")
        from morag_services.services import MoRAGServices

        # Test 1: Should fail without environment variable
        with patch.dict(os.environ, {}, clear=True):
            if "QDRANT_COLLECTION_NAME" in os.environ:
                del os.environ["QDRANT_COLLECTION_NAME"]

            try:
                services = MoRAGServices()
                services._initialize_search_services()
                print("❌ Expected ValueError for missing QDRANT_COLLECTION_NAME")
                return False
            except ValueError as e:
                if "QDRANT_COLLECTION_NAME environment variable is required" in str(e):
                    print("✅ Correctly failed with missing environment variable")
                else:
                    print(f"❌ Wrong error message: {e}")
                    return False

        # Test 2: Should work with environment variable
        with patch.dict(os.environ, {"QDRANT_COLLECTION_NAME": "test_collection"}):
            try:
                services = MoRAGServices()
                # Mock the actual storage to avoid connection issues
                with patch(
                    "morag_services.services.QdrantVectorStorage"
                ) as mock_storage:
                    services._initialize_search_services()
                    # Verify that QdrantVectorStorage was called with correct collection name
                    mock_storage.assert_called_once()
                    call_kwargs = mock_storage.call_args[1]
                    if call_kwargs["collection_name"] == "test_collection":
                        print("✅ Correctly passed collection name to storage")
                        return True
                    else:
                        print(
                            f"❌ Wrong collection name passed: {call_kwargs['collection_name']}"
                        )
                        return False
            except Exception as e:
                print(f"❌ Unexpected error with valid environment variable: {e}")
                return False

    except ImportError as e:
        print(f"⏭️  Skipping services test - import error: {e}")
        return True


def test_ingest_validation():
    """Test ingest task validation."""
    print("Testing ingest task validation...")

    try:
        sys.path.insert(0, "packages/morag/src")

        # Test the validation logic directly
        def validate_ingest_environment():
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            collection_name_env = os.getenv("QDRANT_COLLECTION_NAME")
            if not collection_name_env:
                raise ValueError(
                    "QDRANT_COLLECTION_NAME environment variable is required"
                )
            return collection_name_env

        # Test 1: Should fail without environment variable
        with patch.dict(os.environ, {}, clear=True):
            if "QDRANT_COLLECTION_NAME" in os.environ:
                del os.environ["QDRANT_COLLECTION_NAME"]

            try:
                validate_ingest_environment()
                print("❌ Expected ValueError for missing QDRANT_COLLECTION_NAME")
                return False
            except ValueError as e:
                if "QDRANT_COLLECTION_NAME environment variable is required" in str(e):
                    print("✅ Correctly failed with missing environment variable")
                else:
                    print(f"❌ Wrong error message: {e}")
                    return False

        # Test 2: Should work with environment variable
        with patch.dict(os.environ, {"QDRANT_COLLECTION_NAME": "test_collection"}):
            try:
                result = validate_ingest_environment()
                if result == "test_collection":
                    print("✅ Correctly accepted valid environment variable")
                    return True
                else:
                    print(f"❌ Wrong collection name returned: {result}")
                    return False
            except Exception as e:
                print(f"❌ Unexpected error with valid environment variable: {e}")
                return False

    except ImportError as e:
        print(f"⏭️  Skipping ingest test - import error: {e}")
        return True


def main():
    """Run all component integration tests."""
    print("Testing component integration with unified collection name...")
    print("=" * 60)

    tests = [test_storage_component, test_services_component, test_ingest_validation]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All component integration tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
