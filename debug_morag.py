#!/usr/bin/env python3
"""Debug script for MoRAG with extensive logging."""

import asyncio
import sys
import os
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'morag_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag" / "src"))

async def test_imports():
    """Test all critical imports."""
    logger.info("Testing imports...")
    
    try:
        from morag_services.storage import QdrantVectorStorage
        logger.info("OK QdrantVectorStorage import successful")
    except Exception as e:
        logger.error(f"❌ QdrantVectorStorage import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from morag_services.embedding import GeminiEmbeddingService
        logger.info("OK GeminiEmbeddingService import successful")
    except Exception as e:
        logger.error(f"❌ GeminiEmbeddingService import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from morag.ingest_tasks import store_content_in_vector_db
        logger.info("OK ingest_tasks import successful")
    except Exception as e:
        logger.error(f"❌ ingest_tasks import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

async def test_vector_storage():
    """Test vector storage instantiation and methods."""
    logger.info("Testing vector storage...")
    
    try:
        from morag_services.storage import QdrantVectorStorage
        
        # Test instantiation with environment configuration
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'debug_test')

        storage = QdrantVectorStorage(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
        logger.info("OK QdrantVectorStorage instantiated successfully")
        
        # Test health check
        health = await storage.health_check()
        logger.info(f"Health check result: {health}")
        
        if health.get('status') == 'healthy':
            logger.info("OK Qdrant connection successful")

            # Test basic operations
            try:
                await storage.initialize()
                logger.info("OK Storage initialization successful")

                # Test creating collection
                await storage.create_collection("debug_test", 384)
                logger.info("OK Collection creation successful")

            except Exception as e:
                logger.warning(f"WARNING Storage operations failed (may be expected): {e}")
        else:
            logger.warning("WARNING Qdrant not available - this is expected if not running")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector storage test failed: {e}")
        traceback.print_exc()
        return False

async def test_embedding_service():
    """Test embedding service."""
    logger.info("Testing embedding service...")
    
    try:
        from morag_services.embedding import GeminiEmbeddingService
        
        # Check if API key is available (prefer GEMINI_API_KEY for consistency)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("WARNING No GEMINI_API_KEY found - embedding service will fail")
            return True
        
        service = GeminiEmbeddingService(api_key)
        logger.info("OK GeminiEmbeddingService instantiated successfully")

        # Test embedding generation (with a simple text)
        try:
            result = await service.generate_embedding_with_result("test text", task_type="retrieval_document")
            logger.info(f"OK Embedding generation successful (dimension: {len(result.embedding)})")
        except Exception as e:
            logger.warning(f"WARNING Embedding generation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding service test failed: {e}")
        traceback.print_exc()
        return False

async def test_ingest_function():
    """Test the ingest function that was failing."""
    logger.info("Testing ingest function...")
    
    try:
        from morag.ingest_tasks import store_content_in_vector_db
        
        # Test with minimal data
        test_content = "This is a test document for debugging MoRAG."
        test_metadata = {
            "source_type": "debug_test",
            "source_path": "debug_test.txt",
            "test_timestamp": datetime.now().isoformat()
        }
        
        logger.info("Attempting to store content in vector database...")
        
        # This should now work with our fixes
        point_ids = await store_content_in_vector_db(test_content, test_metadata)
        
        if point_ids:
            logger.info(f"OK Content storage successful! Point IDs: {point_ids}")
            return True
        else:
            logger.warning("WARNING Content storage returned empty point IDs")
            return False
        
    except Exception as e:
        logger.error(f"❌ Ingest function test failed: {e}")
        traceback.print_exc()
        return False

async def check_environment():
    """Check environment configuration."""
    logger.info("Checking environment configuration...")

    required_vars = ['GEMINI_API_KEY']
    optional_vars = ['REDIS_URL', 'QDRANT_HOST', 'QDRANT_PORT', 'QDRANT_API_KEY', 'QDRANT_COLLECTION_NAME']
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"OK {var} is set")
        else:
            logger.warning(f"WARNING {var} is not set")

    # Check if API key is set (with backward compatibility)
    gemini_key = os.getenv('GEMINI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')  # For backward compatibility
    if gemini_key or google_key:
        if google_key and not gemini_key:
            logger.warning("WARNING Using deprecated GOOGLE_API_KEY, please use GEMINI_API_KEY instead")
        logger.info("OK API key is available")
    else:
        logger.warning("WARNING No GEMINI_API_KEY found")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys for security
            if 'API_KEY' in var and len(value) > 10:
                masked_value = value[:6] + "..." + value[-4:]
                logger.info(f"OK {var} = {masked_value}")
            else:
                logger.info(f"OK {var} = {value}")
        else:
            logger.info(f"INFO {var} not set (using default)")

async def main():
    """Main debug function."""
    logger.info("=" * 60)
    logger.info("MoRAG Debug Session Started - Testing Abstract Class Fixes")
    logger.info("=" * 60)

    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    # Use real API key from .env if available, otherwise use dummy
    if not os.getenv('GEMINI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
        os.environ['GEMINI_API_KEY'] = 'dummy_key_for_testing'
    
    # Check environment
    await check_environment()
    
    # Test imports
    if not await test_imports():
        logger.error("Import tests failed - stopping")
        return False
    
    # Test vector storage
    if not await test_vector_storage():
        logger.error("Vector storage tests failed")
        return False
    
    # Test embedding service
    if not await test_embedding_service():
        logger.error("Embedding service tests failed")
        return False
    
    # Test ingest function
    if not await test_ingest_function():
        logger.error("Ingest function tests failed")
        return False
    
    logger.info("=" * 60)
    logger.info("SUCCESS All debug tests completed successfully!")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Debug session interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
