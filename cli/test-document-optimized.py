#!/usr/bin/env python3
"""Optimized document processing test script with performance enhancements."""

import asyncio
import sys
import os
import time
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from morag_document.processor import DocumentProcessor
from morag_core.config import Settings
from morag_core.optimization import optimize_for_document, performance_tracker

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def process_document_optimized(
    file_path: str,
    language: str = "en",
    use_optimization: bool = True
) -> dict:
    """Process document with performance optimizations."""
    
    print("============================================================")
    print("  MoRAG Optimized Document Processing Test")
    print("============================================================")
    print(f"[INFO] Input File: {file_path}")
    
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file info
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    file_ext = Path(file_path).suffix.lower()
    
    print(f"[INFO] File Size: {file_size:.2f} MB")
    print(f"[INFO] File Extension: {file_ext}")
    
    # Initialize settings and processor
    settings = Settings()
    processor = DocumentProcessor()
    
    print(f"[INFO] Document Processor: [OK] Initialized successfully")
    
    # Get optimized configuration
    if use_optimization:
        chunk_config, embedding_config = optimize_for_document(file_path)
        print(f"[INFO] Optimization: [ENABLED] Using optimized settings")
        print(f"[INFO] Chunk Size: {chunk_config['chunk_size']} characters")
        print(f"[INFO] Chunk Overlap: {chunk_config['chunk_overlap']} characters")
        print(f"[INFO] Chunking Strategy: {chunk_config['chunking_strategy']}")
        print(f"[INFO] Fast Track: {'[ENABLED]' if chunk_config.get('enable_fast_track', False) else '[DISABLED]'}")
    else:
        chunk_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunking_strategy": "page"
        }
        embedding_config = {}
        print(f"[INFO] Optimization: [DISABLED] Using default settings")
        print(f"[INFO] Chunk Size: {chunk_config['chunk_size']} characters")
        print(f"[INFO] Chunk Overlap: {chunk_config['chunk_overlap']} characters")
    
    print("\n----------------------------------------")
    print("  Processing Document File")
    print("----------------------------------------")
    print("[PROCESSING] Starting document processing...")
    
    # Start performance tracking
    start_time = performance_tracker.start_operation("document_processing", {
        "file_path": file_path,
        "file_size_mb": file_size,
        "optimization_enabled": use_optimization
    })
    
    try:
        # Process document with optimized settings
        result = await processor.process_file(
            file_path=file_path,
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            chunking_strategy=chunk_config["chunking_strategy"]
        )
        
        # End performance tracking
        processing_time = performance_tracker.end_operation("document_processing", start_time)
        
        print("[OK] Document processing completed successfully!")
        
        print("\n----------------------------------------")
        print("  Processing Results")
        print("----------------------------------------")
        print(f"[INFO] Status: [OK] Success")
        print(f"[INFO] Processing Time: {processing_time:.2f} seconds")
        
        if use_optimization:
            # Calculate performance metrics
            chunks_per_second = len(result.document.chunks) / processing_time if processing_time > 0 else 0
            mb_per_second = file_size / processing_time if processing_time > 0 else 0
            
            print(f"[INFO] Performance: {mb_per_second:.2f} MB/s")
            print(f"[INFO] Throughput: {chunks_per_second:.1f} chunks/s")
        
        print("\n----------------------------------------")
        print("  Metadata")
        print("----------------------------------------")
        print(f"[INFO] quality_score: {result.metadata.get('quality_score', 'N/A')}")
        print(f"[INFO] quality_issues: {result.metadata.get('quality_issues', [])}")
        print(f"[INFO] warnings: {result.metadata.get('warnings', [])}")
        
        print("\n----------------------------------------")
        print("  Document Information")
        print("----------------------------------------")
        print(f"[INFO] Title: {result.document.metadata.title or 'N/A'}")
        print(f"[INFO] Author: {result.document.metadata.author or 'N/A'}")
        print(f"[INFO] Page Count: {result.document.metadata.page_count or 'N/A'}")
        print(f"[INFO] Word Count: {result.document.metadata.word_count or 'N/A'}")
        print(f"[INFO] Chunks Count: {len(result.document.chunks)}")
        
        print("\n----------------------------------------")
        print("  Content Preview")
        print("----------------------------------------")
        # Get content from chunks if document.content is not available
        if hasattr(result.document, 'content') and result.document.content:
            content = result.document.content
        else:
            # Reconstruct content from chunks
            content = "\n\n".join(chunk.content for chunk in result.document.chunks)

        content_preview = content[:500] + "..." if len(content) > 500 else content
        print(f"📄 Raw Text ({len(content)} characters):")
        print(content_preview)
        
        print("\n----------------------------------------")
        print("  Chunks Preview (first 3)")
        print("----------------------------------------")
        for i, chunk in enumerate(result.document.chunks[:3]):
            chunk_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            print(f"  Chunk {i+1}:")
            print(f"    Content: {chunk_preview}")
            print(f"    Metadata: {chunk.metadata}")
        
        # Performance summary
        if use_optimization:
            perf_summary = performance_tracker.get_performance_summary()
            print("\n----------------------------------------")
            print("  Performance Summary")
            print("----------------------------------------")
            for op_name, metrics in perf_summary.get("operations", {}).items():
                print(f"[PERF] {op_name}: {metrics['avg_time']:.2f}s avg, {metrics['count']} operations")
        
        # Save results
        output_file = f"{file_path}_optimized_test_result.json"
        
        # Convert result to JSON-serializable format
        result_data = {
            "mode": "optimized_processing",
            "success": result.success,
            "processing_time": processing_time,
            "optimization_enabled": use_optimization,
            "chunk_config": chunk_config,
            "embedding_config": embedding_config if use_optimization else {},
            "file_info": {
                "file_path": file_path,
                "file_size_mb": file_size,
                "file_extension": file_ext
            },
            "document": {
                "title": result.document.metadata.title,
                "author": result.document.metadata.author,
                "page_count": result.document.metadata.page_count,
                "word_count": result.document.metadata.word_count,
                "raw_text": content,
                "chunks_count": len(result.document.chunks),
                "chunks": [
                    {
                        "content": chunk.content,
                        "page_number": getattr(chunk, 'page_number', None),
                        "metadata": chunk.metadata
                    }
                    for chunk in result.document.chunks
                ]
            },
            "metadata": result.metadata,
            "performance_summary": performance_tracker.get_performance_summary() if use_optimization else {}
        }
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print("\n----------------------------------------")
        print("  Output")
        print("----------------------------------------")
        print(f"[INFO] Results saved to: {output_file}")
        
        print(f"\n[SUCCESS] Optimized document processing test completed successfully!")
        
        return result_data
        
    except Exception as e:
        processing_time = performance_tracker.end_operation("document_processing", start_time)
        
        print(f"[ERROR] Document processing failed: {str(e)}")
        logger.error("Document processing failed", error=str(e), processing_time=processing_time)
        
        error_data = {
            "mode": "optimized_processing",
            "success": False,
            "processing_time": processing_time,
            "error": str(e),
            "file_info": {
                "file_path": file_path,
                "file_size_mb": file_size,
                "file_extension": file_ext
            }
        }
        
        return error_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test optimized document processing")
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument("--language", default="en", help="Document language (default: en)")
    parser.add_argument("--no-optimization", action="store_true", help="Disable performance optimizations")
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(process_document_optimized(
            args.file_path,
            args.language,
            use_optimization=not args.no_optimization
        ))
        
        if result["success"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
