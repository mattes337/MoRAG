"""Command-line interface for MoRAG Services.

This module provides a CLI for accessing MoRAG services.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
import json
import structlog

from .services import MoRAGServices, ServiceConfig, ContentType
from .pipeline import Pipeline

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()

async def process_content(args):
    """Process content based on command-line arguments."""
    # Initialize services
    services = MoRAGServices()
    
    # Process content
    result = await services.process_content(args.path_or_url)
    
    # Print result
    if result.success:
        print(f"Successfully processed {args.path_or_url}")
        print(f"Content type: {result.content_type}")
        
        if result.text_content:
            if args.output:
                # Write text content to file
                output_path = args.output
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.text_content)
                print(f"Text content written to {output_path}")
            else:
                # Print summary of text content
                text_preview = result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content
                print(f"Text content: {text_preview}")
        
        if result.metadata:
            if args.metadata:
                # Write metadata to file
                metadata_path = args.metadata
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(result.metadata, f, indent=2)
                print(f"Metadata written to {metadata_path}")
            else:
                # Print metadata
                print(f"Metadata: {result.metadata}")
        
        if result.extracted_files:
            print(f"Extracted files: {result.extracted_files}")
    else:
        print(f"Error processing {args.path_or_url}: {result.error_message}")
        sys.exit(1)

async def process_batch(args):
    """Process multiple content items in batch."""
    # Initialize services
    config = ServiceConfig(
        max_concurrent_tasks=args.concurrency
    )
    services = MoRAGServices(config)
    
    # Read items from file or use command-line arguments
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]
    else:
        items = args.items
    
    # Filter to only existing files and valid URLs
    valid_items = []
    for item in items:
        if item.startswith("http") or os.path.exists(item):
            valid_items.append(item)
        else:
            logger.warning(f"Skipping non-existent item: {item}")
    
    if not valid_items:
        logger.error("No valid items to process")
        sys.exit(1)
    
    # Process batch
    logger.info(f"Processing {len(valid_items)} items")
    results = await services.process_batch(valid_items)
    
    # Print results summary
    success_count = sum(1 for result in results.values() if result.success)
    print(f"Processed {len(results)} items ({success_count} succeeded, {len(results) - success_count} failed)")
    
    # Write detailed results to file if requested
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            summary = {
                item: {
                    "success": result.success,
                    "content_type": result.content_type,
                    "error_message": result.error_message
                }
                for item, result in results.items()
            }
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")
        
        # Write text content and metadata for each item
        for item, result in results.items():
            if result.success:
                # Create safe filename
                safe_name = Path(item).name if not item.startswith("http") else item.replace("://", "_").replace("/", "_")
                
                # Write text content
                if result.text_content:
                    text_path = output_dir / f"{safe_name}.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(result.text_content)
                
                # Write metadata
                if result.metadata:
                    metadata_path = output_dir / f"{safe_name}.metadata.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(result.metadata, f, indent=2)

async def run_pipeline(args):
    """Run a processing pipeline."""
    # Initialize services
    config = ServiceConfig(
        max_concurrent_tasks=args.concurrency
    )
    services = MoRAGServices(config)
    
    # Create pipeline
    pipeline = Pipeline(services, name="cli_pipeline")
    
    # Configure pipeline steps
    pipeline.process_content()
    
    if args.extract_text:
        pipeline.extract_text()
    
    if args.extract_metadata:
        pipeline.extract_metadata()
    
    if args.generate_embeddings:
        pipeline.generate_embeddings()
    
    # Read items from file or use command-line arguments
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]
    else:
        items = args.items
    
    # Filter to only existing files and valid URLs
    valid_items = []
    for item in items:
        if item.startswith("http") or os.path.exists(item):
            valid_items.append(item)
        else:
            logger.warning(f"Skipping non-existent item: {item}")
    
    if not valid_items:
        logger.error("No valid items to process")
        sys.exit(1)
    
    # Execute pipeline
    logger.info(f"Executing pipeline on {len(valid_items)} items")
    context = await pipeline.process_batch(valid_items)
    
    # Print results summary
    success_count = sum(1 for result in context.results.values() if result.success)
    print(f"Processed {len(context.results)} items ({success_count} succeeded, {len(context.results) - success_count} failed)")
    
    # Write results to output directory if specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            summary = {
                item: {
                    "success": result.success,
                    "content_type": result.content_type,
                    "error_message": result.error_message
                }
                for item, result in context.results.items()
            }
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")
        
        # Write texts
        if hasattr(context, "texts") and context.texts:
            texts_dir = output_dir / "texts"
            texts_dir.mkdir(exist_ok=True)
            for item, text in context.texts.items():
                safe_name = Path(item).name if not item.startswith("http") else item.replace("://", "_").replace("/", "_")
                text_path = texts_dir / f"{safe_name}.txt"
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)
            print(f"Texts written to {texts_dir}")
        
        # Write metadata
        if hasattr(context, "metadata") and context.metadata:
            metadata_dir = output_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            for item, metadata in context.metadata.items():
                safe_name = Path(item).name if not item.startswith("http") else item.replace("://", "_").replace("/", "_")
                metadata_path = metadata_dir / f"{safe_name}.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            print(f"Metadata written to {metadata_dir}")
        
        # Write embeddings
        if hasattr(context, "embeddings") and context.embeddings:
            embeddings_dir = output_dir / "embeddings"
            embeddings_dir.mkdir(exist_ok=True)
            for item, embedding in context.embeddings.items():
                safe_name = Path(item).name if not item.startswith("http") else item.replace("://", "_").replace("/", "_")
                embedding_path = embeddings_dir / f"{safe_name}.json"
                with open(embedding_path, "w", encoding="utf-8") as f:
                    json.dump(embedding, f, indent=2)
            print(f"Embeddings written to {embeddings_dir}")

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="MoRAG Services CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single content item")
    process_parser.add_argument("path_or_url", help="Path to file or URL to process")
    process_parser.add_argument("--output", "-o", help="Output file for text content")
    process_parser.add_argument("--metadata", "-m", help="Output file for metadata")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple content items")
    batch_parser.add_argument("--file", "-f", help="File containing list of items to process (one per line)")
    batch_parser.add_argument("--output", "-o", help="Output directory for results")
    batch_parser.add_argument("--concurrency", "-c", type=int, default=5, help="Maximum concurrent tasks")
    batch_parser.add_argument("items", nargs="*", help="Items to process (files or URLs)")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run a processing pipeline")
    pipeline_parser.add_argument("--file", "-f", help="File containing list of items to process (one per line)")
    pipeline_parser.add_argument("--output", "-o", help="Output directory for results")
    pipeline_parser.add_argument("--concurrency", "-c", type=int, default=5, help="Maximum concurrent tasks")
    pipeline_parser.add_argument("--extract-text", action="store_true", help="Extract text from content")
    pipeline_parser.add_argument("--extract-metadata", action="store_true", help="Extract metadata from content")
    pipeline_parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings for text")
    pipeline_parser.add_argument("items", nargs="*", help="Items to process (files or URLs)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Run appropriate command
    if args.command == "process":
        asyncio.run(process_content(args))
    elif args.command == "batch":
        asyncio.run(process_batch(args))
    elif args.command == "pipeline":
        asyncio.run(run_pipeline(args))

if __name__ == "__main__":
    main()