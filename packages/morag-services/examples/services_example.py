"""Example usage of MoRAG Services.

This example demonstrates how to use the MoRAGServices class to process
different types of content and build processing pipelines.
"""

import asyncio
import os
from pathlib import Path

import structlog
from morag_services.pipeline import Pipeline
from morag_services.services import ContentType, MoRAGServices, ServiceConfig

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()


async def process_single_item():
    """Process a single content item."""
    # Initialize services
    services = MoRAGServices()

    # Process a document
    document_path = "path/to/document.pdf"
    if os.path.exists(document_path):
        logger.info("Processing document")
        result = await services.process_content(document_path)

        if result.success:
            logger.info("Document processed successfully")
            print(f"Text content (first 200 chars): {result.text_content[:200]}...")
            print(f"Metadata: {result.metadata}")
        else:
            logger.error("Document processing failed", error=result.error_message)

    # Process a web URL
    url = "https://example.com"
    logger.info("Processing URL", url=url)
    result = await services.process_content(url)

    if result.success:
        logger.info("URL processed successfully")
        print(f"Text content (first 200 chars): {result.text_content[:200]}...")
        print(f"Metadata: {result.metadata}")
    else:
        logger.error("URL processing failed", error=result.error_message)

    # Process a YouTube URL
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    logger.info("Processing YouTube URL", url=youtube_url)
    result = await services.process_content(youtube_url)

    if result.success:
        logger.info("YouTube URL processed successfully")
        print(f"Metadata: {result.metadata}")
        print(f"Extracted files: {result.extracted_files}")
    else:
        logger.error("YouTube processing failed", error=result.error_message)


async def process_batch():
    """Process multiple content items in batch."""
    # Initialize services with custom configuration
    config = ServiceConfig(max_concurrent_tasks=3)  # Limit concurrent processing
    services = MoRAGServices(config)

    # Define batch of items to process
    items = [
        "path/to/document1.pdf",
        "path/to/image.jpg",
        "https://example.com",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ]

    # Filter to only existing files and valid URLs
    valid_items = [
        item for item in items if item.startswith("http") or os.path.exists(item)
    ]

    logger.info("Processing batch", items=valid_items)
    results = await services.process_batch(valid_items)

    # Print results summary
    print("\nBatch Processing Results:")
    print("-" * 50)
    for item, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"{status} {result.content_type}: {item}")
        if not result.success:
            print(f"  Error: {result.error_message}")


async def use_pipeline():
    """Use a processing pipeline."""
    # Initialize services
    services = MoRAGServices()

    # Create a pipeline
    pipeline = Pipeline(services, name="example_pipeline")

    # Configure pipeline steps
    pipeline.process_content()
    pipeline.extract_text()
    pipeline.extract_metadata()
    pipeline.generate_embeddings()

    # Add a custom step to count words in each text
    async def count_words(texts, context):
        word_counts = {}
        for item_id, text in texts.items():
            word_counts[item_id] = len(text.split())
        return word_counts

    pipeline.custom_step(
        name="count_words",
        process_fn=count_words,
        input_key="texts",
        output_key="word_counts",
    )

    # Define items to process
    items = ["https://example.com", "https://www.python.org"]

    # Execute pipeline
    logger.info("Executing pipeline", items=items)
    context = await pipeline.process_batch(items)

    # Print pipeline results
    print("\nPipeline Results:")
    print("-" * 50)
    for item_id, result in context.results.items():
        status = "✓" if result.success else "✗"
        print(f"{status} {item_id}")

        if item_id in context.word_counts:
            print(f"  Word count: {context.word_counts[item_id]}")

        if item_id in context.embeddings:
            embedding = context.embeddings[item_id]
            print(
                f"  Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...] (length: {len(embedding)})"
            )


async def detect_content_types():
    """Demonstrate content type detection."""
    services = MoRAGServices()

    test_items = [
        "document.pdf",
        "image.jpg",
        "audio.mp3",
        "video.mp4",
        "https://example.com",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ]

    print("\nContent Type Detection:")
    print("-" * 50)
    for item in test_items:
        content_type = services.detect_content_type(item)
        print(f"{item}: {content_type}")


async def main():
    """Run all examples."""
    print("MoRAG Services Examples")
    print("=" * 50)

    print("\n1. Processing Single Items")
    await process_single_item()

    print("\n2. Batch Processing")
    await process_batch()

    print("\n3. Using Pipelines")
    await use_pipeline()

    print("\n4. Content Type Detection")
    await detect_content_types()


if __name__ == "__main__":
    asyncio.run(main())
