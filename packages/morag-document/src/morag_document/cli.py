"""Command-line interface for document processing."""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from morag_core.interfaces.converter import ChunkingStrategy
from morag_core.interfaces.service import ServiceConfig
from morag_core.utils.logging import configure_logging

from .service import DocumentService

logger = structlog.get_logger(__name__)


async def process_document(args: argparse.Namespace) -> None:
    """Process document file.

    Args:
        args: Command-line arguments
    """
    # Create service config
    config = ServiceConfig()
    if args.enable_embedding:
        config["enable_embedding"] = True
        config["embedding"] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            "model": args.embedding_model,
        }

    # Create service
    service = DocumentService(config)
    await service.initialize()

    try:
        # Process document
        result = await service.process_document(
            args.file_path,
            chunking_strategy=args.chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            generate_embeddings=args.enable_embedding,
        )

        # Print document info
        print(f"Document: {result.document.metadata.title}")
        print(f"Type: {result.document.metadata.mime_type}")
        print(f"Word count: {result.document.metadata.word_count}")
        print(f"Chunks: {len(result.document.chunks)}")
        print(f"Quality score: {result.metadata.get('quality_score', 'N/A')}")

        # Print chunks if requested
        if args.show_chunks:
            print("\nChunks:")
            for i, chunk in enumerate(result.document.chunks):
                print(f"\nChunk {i+1}:")
                print(f"Section: {chunk.section or 'N/A'}")
                print(f"Content: {chunk.content[:100]}...")
                if chunk.embedding is not None:
                    print(f"Embedding: [vector with {len(chunk.embedding)} dimensions]")

        # Generate summary if requested
        if args.summarize:
            if not args.enable_embedding:
                print("\nError: Summarization requires embedding service to be enabled")
            else:
                summary = await service.summarize_document(
                    result.document, max_length=args.summary_length
                )
                print("\nSummary:")
                print(summary)

        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare output data
            output_data = {
                "metadata": result.document.metadata.to_dict(),
                "chunks": [
                    {
                        "section": chunk.section,
                        "content": chunk.content,
                        "has_embedding": chunk.embedding is not None,
                    }
                    for chunk in result.document.chunks
                ],
                "quality": result.metadata,
            }

            # Add summary if available
            if args.summarize and args.enable_embedding:
                output_data["summary"] = summary

            # Write output
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"\nOutput saved to {output_path}")

    finally:
        # Shutdown service
        await service.shutdown()


def main():
    """Main entry point."""
    # Configure logging
    configure_logging()

    # Create argument parser
    parser = argparse.ArgumentParser(description="Document processing tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process document")
    process_parser.add_argument(
        "file_path", help="Path to document file"
    )
    process_parser.add_argument(
        "--chunking-strategy",
        choices=[s.value for s in ChunkingStrategy],
        default=ChunkingStrategy.PARAGRAPH.value,
        help="Chunking strategy",
    )
    process_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters",
    )
    process_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters",
    )
    process_parser.add_argument(
        "--enable-embedding",
        action="store_true",
        help="Enable embedding generation",
    )
    process_parser.add_argument(
        "--embedding-model",
        default="models/embedding-001",
        help="Embedding model to use",
    )
    process_parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Show document chunks",
    )
    process_parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate document summary",
    )
    process_parser.add_argument(
        "--summary-length",
        type=int,
        default=500,
        help="Maximum summary length",
    )
    process_parser.add_argument(
        "--output",
        help="Output file path (JSON)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run command
    if args.command == "process":
        asyncio.run(process_document(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()