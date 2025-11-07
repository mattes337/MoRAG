"""Example script for processing text."""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path to import morag_document
sys.path.insert(0, str(Path(__file__).parent.parent))

from morag_core.interfaces.service import ServiceConfig
from morag_core.interfaces.converter import ChunkingStrategy
from morag_core.utils.logging import configure_logging

from morag_document.service import DocumentService


# Sample text for processing
SAMPLE_TEXT = """
# Document Processing with Morag

## Introduction

Morag is a powerful document processing framework that allows you to extract, chunk, and analyze text from various document formats. This example demonstrates how to process raw text using the Morag document service.

## Features

- Text extraction from multiple document formats
- Intelligent chunking strategies
- Metadata extraction
- Embedding generation
- Document summarization

## Usage

The Morag document service provides a simple API for processing documents and text. You can use it to extract text, generate embeddings, and summarize documents.

### Processing Text

To process text, you can use the `process_text` method of the document service. This method takes a text string and returns a processing result with the processed document.

### Chunking Strategies

Morag supports multiple chunking strategies:

1. Paragraph - Split text by paragraphs
2. Sentence - Split text by sentences
3. Fixed - Split text into fixed-size chunks
4. Section - Split text by sections (headings)

### Embedding Generation

If you enable the embedding service, Morag can generate embeddings for each chunk of text. These embeddings can be used for semantic search and other NLP tasks.

## Conclusion

Morag provides a flexible and powerful framework for document processing. You can use it to extract text, generate embeddings, and summarize documents from various formats.
"""


async def main():
    """Main function."""
    # Configure logging
    configure_logging()

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
        # Process text
        print("Processing text...")
        result = await service.process_text(
            SAMPLE_TEXT,
            title="Morag Document Processing Example",
            chunking_strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=500,
            chunk_overlap=50,
            generate_embeddings=bool(api_key),
        )

        # Print document info
        print("\nDocument Information:")
        print(f"Title: {result.document.metadata.title}")
        print(f"Type: {result.document.metadata.file_type}")
        print(f"Word count: {result.document.metadata.word_count}")
        print(f"Chunks: {len(result.document.chunks)}")
        print(f"Quality score: {result.metadata.get('quality_score', 'N/A')}")

        # Print chunks
        print("\nChunks:")
        for i, chunk in enumerate(result.document.chunks):
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk.content[:100]}...")
            if chunk.embedding is not None:
                print(f"Embedding: [vector with {len(chunk.embedding)} dimensions]")

        # Generate summary if embedding is enabled
        if api_key:
            print("\nGenerating summary...")
            summary = await service.summarize_document(result.document)
            print("\nSummary:")
            print(summary)

        # Try different chunking strategy
        print("\n\nProcessing text with SECTION chunking strategy...")
        result = await service.process_text(
            SAMPLE_TEXT,
            title="Morag Document Processing Example",
            chunking_strategy=ChunkingStrategy.SECTION,
            generate_embeddings=False,
        )

        # Print chunks
        print(f"\nChunks with SECTION strategy: {len(result.document.chunks)}")
        for i, chunk in enumerate(result.document.chunks):
            print(f"\nChunk {i+1}:")
            print(f"Section: {chunk.section or 'N/A'}")
            print(f"Content: {chunk.content[:100]}...")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Shutdown service
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
