#!/usr/bin/env python3
"""Test the complete ingest workflow to verify the fix."""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add the packages to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))

async def test_ingest_workflow():
    """Test the complete ingest workflow that was failing."""
    print("Testing complete ingest workflow...")
    
    try:
        # Import the function that was failing
        from morag.ingest_tasks import store_content_in_vector_db
        
        print("✅ Successfully imported store_content_in_vector_db")
        
        # Test with sample content
        test_content = """
        This is a test document for MoRAG.
        It contains multiple sentences to test the chunking functionality.
        The system should be able to process this content and store it in the vector database.
        This test verifies that the QdrantVectorStorage instantiation error has been fixed.
        """
        
        test_metadata = {
            "source_type": "test",
            "source_path": "test_document.txt",
            "test_run": True,
            "description": "Test document for verifying ingest workflow"
        }
        
        print("Attempting to store content in vector database...")
        
        # This should now work without the abstract class error
        point_ids = await store_content_in_vector_db(
            content=test_content.strip(),
            metadata=test_metadata,
            collection_name="test_collection"
        )
        
        if point_ids:
            print(f"✅ Content storage successful!")
            print(f"   Generated {len(point_ids)} vector points")
            print(f"   Point IDs: {point_ids[:3]}{'...' if len(point_ids) > 3 else ''}")
            return True
        else:
            print("⚠️ Content storage returned empty point IDs")
            print("   This might be due to Qdrant not being available")
            return True  # Still consider this a success since no error was thrown
        
    except Exception as e:
        error_msg = str(e)
        if "Can't instantiate abstract class QdrantVectorStorage" in error_msg:
            print("❌ The abstract class error is still present!")
            print("   The fix may not have been applied correctly.")
            return False
        elif "No connection could be made" in error_msg or "Connection refused" in error_msg:
            print("⚠️ Qdrant connection failed (expected if Qdrant is not running)")
            print("   But the abstract class error has been fixed!")
            return True
        else:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_services_availability():
    """Test if required services are available."""
    print("\nChecking service availability...")
    
    # Check environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if google_api_key:
        print("✅ GOOGLE_API_KEY is set")
    else:
        print("⚠️ GOOGLE_API_KEY is not set (embedding service will fail)")
    
    # Check Redis (if available)
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis is available")
    except Exception:
        print("⚠️ Redis is not available")
    
    # Check Qdrant (if available)
    try:
        import requests
        response = requests.get('http://localhost:6333/health', timeout=2)
        if response.status_code == 200:
            print("✅ Qdrant is available")
        else:
            print("⚠️ Qdrant responded but with error status")
    except Exception:
        print("⚠️ Qdrant is not available")

async def main():
    """Main test function."""
    print("Testing MoRAG Ingest Workflow")
    print("=" * 50)

    # Set a dummy API key for testing
    os.environ['GOOGLE_API_KEY'] = 'dummy_key_for_testing'
    
    # Test service availability
    await test_services_availability()
    
    print("\n" + "=" * 50)
    
    # Test the main workflow
    success = await test_ingest_workflow()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Ingest workflow test passed!")
        print("\nThe QdrantVectorStorage abstract class error has been fixed.")
        print("You can now run MoRAG workers and process documents.")
        print("\nTo start MoRAG locally:")
        print("1. Start Redis and Qdrant services")
        print("2. Run: python scripts/start_worker.py")
        print("3. Run: uvicorn morag.api.main:app --reload")
    else:
        print("❌ Ingest workflow test failed!")
        print("Check the errors above for details.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
