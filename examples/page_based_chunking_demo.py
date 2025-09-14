#!/usr/bin/env python3
"""
Demo script showing page-based chunking functionality.

This script demonstrates how the new page-based chunking works
compared to traditional sentence-based chunking.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_document import DocumentProcessor
from morag_core.models.document import DocumentChunk
from morag_core.chunking import SemanticChunker


async def create_sample_document_chunks():
    """Create sample document chunks to simulate a parsed PDF."""
    return [
        DocumentChunk(
            text="Introduction to Machine Learning",
            chunk_type="title",
            page_number=1,
            element_id="title_1",
            metadata={"element_type": "Title"}
        ),
        DocumentChunk(
            text="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            chunk_type="text",
            page_number=1,
            element_id="para_1",
            metadata={"element_type": "Text"}
        ),
        DocumentChunk(
            text="It focuses on developing algorithms that can access data and use it to learn for themselves.",
            chunk_type="text",
            page_number=1,
            element_id="para_2",
            metadata={"element_type": "Text"}
        ),
        DocumentChunk(
            text="Types of Machine Learning",
            chunk_type="title",
            page_number=2,
            element_id="title_2",
            metadata={"element_type": "Title"}
        ),
        DocumentChunk(
            text="There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.",
            chunk_type="text",
            page_number=2,
            element_id="para_3",
            metadata={"element_type": "Text"}
        ),
        DocumentChunk(
            text="Supervised learning uses labeled training data to learn a mapping function from input variables to output variables.",
            chunk_type="text",
            page_number=2,
            element_id="para_4",
            metadata={"element_type": "Text"}
        ),
        DocumentChunk(
            text="Applications and Future",
            chunk_type="title",
            page_number=3,
            element_id="title_3",
            metadata={"element_type": "Title"}
        ),
        DocumentChunk(
            text="Machine learning has numerous applications in various fields including healthcare, finance, transportation, and entertainment.",
            chunk_type="text",
            page_number=3,
            element_id="para_5",
            metadata={"element_type": "Text"}
        ),
    ]


async def demonstrate_page_based_chunking():
    """Demonstrate page-based chunking vs traditional chunking."""
    print("=== Page-Based Chunking Demo ===\n")
    
    # Create sample document chunks
    sample_chunks = await create_sample_document_chunks()
    
    # Create a sample parse result
    parse_result = DocumentParseResult(
        chunks=sample_chunks,
        metadata={
            "parser": "demo",
            "file_name": "ml_guide.pdf",
            "total_chunks": len(sample_chunks)
        },
        images=[],
        total_pages=3,
        word_count=150
    )
    
    print("Original chunks (sentence/element level):")
    print(f"Total chunks: {len(parse_result.chunks)}")
    for i, chunk in enumerate(parse_result.chunks):
        print(f"  {i+1}. Page {chunk.page_number} ({chunk.chunk_type}): {chunk.text[:60]}...")
    
    print("\n" + "="*60 + "\n")
    
    # Apply page-based chunking
    processor = DocumentProcessor()
    page_based_result = await processor._apply_page_based_chunking(parse_result)
    
    print("After page-based chunking:")
    print(f"Total chunks: {len(page_based_result.chunks)}")
    for i, chunk in enumerate(page_based_result.chunks):
        print(f"\n  {i+1}. Page {chunk.page_number} chunk:")
        print(f"     Length: {len(chunk.text)} characters")
        print(f"     Type: {chunk.chunk_type}")
        print(f"     Original chunks combined: {chunk.metadata.get('original_chunks_count', 'N/A')}")
        print(f"     Content preview: {chunk.text[:100]}...")
    
    print("\n" + "="*60 + "\n")
    
    # Show the difference in vector points
    print("Vector Database Impact:")
    print(f"  Before: {len(parse_result.chunks)} vector points")
    print(f"  After:  {len(page_based_result.chunks)} vector points")
    reduction = (len(parse_result.chunks) - len(page_based_result.chunks)) / len(parse_result.chunks) * 100
    print(f"  Reduction: {reduction:.1f}%")
    
    return page_based_result


async def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies."""
    print("\n=== Chunking Strategy Comparison ===\n")
    
    sample_text = """
    Machine learning is a powerful technology that has revolutionized many industries. 
    It enables computers to learn patterns from data without explicit programming.
    
    There are several types of machine learning algorithms. Supervised learning uses 
    labeled data to train models. Unsupervised learning finds hidden patterns in 
    unlabeled data. Reinforcement learning learns through trial and error.
    
    Applications of machine learning are vast and growing. They include image 
    recognition, natural language processing, recommendation systems, and autonomous 
    vehicles. The future of AI depends heavily on advances in machine learning.
    """
    
    chunker = SemanticChunker()
    
    strategies = ["page", "semantic", "sentence", "paragraph", "simple"]
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Chunking:")
        try:
            chunks = await chunker.chunk_text(sample_text, chunk_size=200, strategy=strategy)
            print(f"  Number of chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"    {i+1}. ({chunk.word_count} words): {chunk.text[:50]}...")
        except Exception as e:
            print(f"  Error: {e}")


async def main():
    """Main demo function."""
    print("MoRAG Page-Based Chunking Demonstration")
    print("=" * 50)
    
    # Demonstrate page-based chunking
    await demonstrate_page_based_chunking()
    
    # Demonstrate different chunking strategies
    await demonstrate_chunking_strategies()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("\nKey Benefits of Page-Based Chunking:")
    print("  ✓ Preserves document structure and page context")
    print("  ✓ Reduces number of vector points in database")
    print("  ✓ Provides better context for RAG operations")
    print("  ✓ Maintains relationships between page elements")
    print("  ✓ Configurable with fallback for large pages")


if __name__ == "__main__":
    asyncio.run(main())
