#!/usr/bin/env python3
"""Initialize Qdrant database with required collections."""

import asyncio
import os
import structlog
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_services.storage import QdrantVectorStorage

logger = structlog.get_logger()


async def main():
    """Initialize the database."""
    try:
        logger.info("Initializing Qdrant database")

        # Create Qdrant service instance
        qdrant_service = QdrantVectorStorage(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "morag_documents")
        )

        # Connect to Qdrant
        await qdrant_service.connect()

        # Create collection (vector size for text-embedding-004 is 768)
        await qdrant_service.create_collection(
            collection_name=qdrant_service.collection_name,
            vector_size=768,
            force_recreate=False
        )

        # Get collection info
        info = await qdrant_service.get_collection_info()
        logger.info("Database initialized successfully", collection_info=info)

    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        sys.exit(1)
    finally:
        await qdrant_service.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
