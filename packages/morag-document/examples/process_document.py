"""Example script for processing documents."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import morag_document
sys.path.insert(0, str(Path(__file__).parent.parent))

from morag_core.interfaces.converter import ChunkingStrategy
from morag_core.interfaces.service import ServiceConfig
from morag_core.utils.logging import configure_logging
from morag_document.service import DocumentService


async def main():
    """Main function."""
    # Configure logging
    configure_logging()

    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python process_document.py <file_path>")
        return

    # Get file path
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Create service config
    config = ServiceConfig()

    # Enable embedding if API key is available
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        config["enable_embedding"] = True
        config["embedding"] = {
            "api_key": api_key,
            "model": "models/embedding-001",
        }
        print("Embedding service enabled")
    else:
        print("Embedding service disabled (GEMINI_API_KEY not set)")

    # Create service
    service = DocumentService(config)
    await service.initialize()

    try:
        # Process document
        print(f"Processing document: {file_path}")
        result = await service.process_document(
            file_path,
            chunking_strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=1000,
            chunk_overlap=100,
            generate_embeddings=bool(api_key),
        )

        # Print document info
        print("\nDocument Information:")
        print(f"Title: {result.document.metadata.title}")
        print(f"Type: {result.document.metadata.file_type}")
        print(f"Word count: {result.document.metadata.word_count}")
        print(f"Chunks: {len(result.document.chunks)}")
        print(f"Quality score: {result.metadata.get('quality_score', 'N/A')}")

        # Print first few chunks
        print("\nSample Chunks:")
        for i, chunk in enumerate(result.document.chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Section: {chunk.section or 'N/A'}")
            print(f"Content: {chunk.content[:150]}...")
            if chunk.embedding is not None:
                print(f"Embedding: [vector with {len(chunk.embedding)} dimensions]")

        # Generate summary if embedding is enabled
        if api_key:
            print("\nGenerating summary...")
            summary = await service.summarize_document(result.document)
            print("\nSummary:")
            print(summary)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Shutdown service
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
