"""Performance optimization utilities for MoRAG processing."""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class ProcessingOptimizer:
    """Optimize processing parameters based on document characteristics."""

    @staticmethod
    def get_optimal_chunk_config(
        file_size_mb: float, content_length: int, document_type: str = "pdf"
    ) -> Dict[str, Any]:
        """Get optimal chunking configuration based on document characteristics."""

        # Base configuration
        config = {
            "chunk_size": 4000,
            "chunk_overlap": 200,
            "chunking_strategy": "page",
            "enable_fast_track": False,
        }

        # Optimize based on file size
        if file_size_mb < 0.5:  # Small files (< 500KB)
            config.update(
                {
                    "chunk_size": 6000,  # Larger chunks for small files
                    "chunk_overlap": 300,
                    "enable_fast_track": True,
                }
            )
        elif file_size_mb < 2.0:  # Medium files (< 2MB)
            config.update({"chunk_size": 4000, "chunk_overlap": 200})
        else:  # Large files (> 2MB)
            config.update(
                {
                    "chunk_size": 3000,  # Smaller chunks for large files
                    "chunk_overlap": 150,
                }
            )

        # Optimize based on content length
        if content_length < 10000:  # Very short content
            config.update(
                {
                    "chunk_size": min(content_length + 1000, 8000),
                    "chunk_overlap": 100,
                    "enable_fast_track": True,
                }
            )
        elif content_length > 100000:  # Very long content
            config.update(
                {
                    "chunking_strategy": "semantic",  # Better for long documents
                    "chunk_size": 3000,
                }
            )

        logger.info(
            "Optimized chunk configuration",
            file_size_mb=file_size_mb,
            content_length=content_length,
            document_type=document_type,
            config=config,
        )

        return config

    @staticmethod
    def get_optimal_embedding_config(
        chunk_count: int, processing_mode: str = "standard"
    ) -> Dict[str, Any]:
        """Get optimal embedding configuration based on chunk count."""

        config = {
            "batch_size": 100,
            "delay_between_batches": 0.05,
            "use_native_batch": True,
            "max_concurrent": 5,
        }

        if chunk_count <= 5:  # Very few chunks
            config.update(
                {
                    "batch_size": chunk_count,
                    "delay_between_batches": 0.01,  # Minimal delay
                    "max_concurrent": 1,
                }
            )
        elif chunk_count <= 20:  # Small number of chunks
            config.update(
                {
                    "batch_size": min(chunk_count, 50),
                    "delay_between_batches": 0.02,
                    "max_concurrent": 2,
                }
            )
        elif chunk_count > 100:  # Many chunks
            config.update(
                {
                    "batch_size": 150,  # Larger batches
                    "delay_between_batches": 0.1,  # More conservative delay
                    "max_concurrent": 3,
                }
            )

        # Fast track mode for small documents
        if processing_mode == "fast_track":
            config.update({"delay_between_batches": 0.01, "max_concurrent": 1})

        logger.info(
            "Optimized embedding configuration",
            chunk_count=chunk_count,
            processing_mode=processing_mode,
            config=config,
        )

        return config

    @staticmethod
    def should_use_fast_track(
        file_size_mb: float, content_length: int, estimated_chunks: int
    ) -> bool:
        """Determine if fast-track processing should be used."""

        # Use fast track for small documents
        if file_size_mb < 1.0 and content_length < 50000 and estimated_chunks <= 10:
            return True

        return False

    @staticmethod
    def estimate_processing_time(
        file_size_mb: float, chunk_count: int, config: Dict[str, Any]
    ) -> float:
        """Estimate processing time based on document characteristics."""

        # Base time estimates (in seconds)
        base_conversion_time = file_size_mb * 2.0  # 2 seconds per MB
        base_chunking_time = chunk_count * 0.1  # 0.1 seconds per chunk
        base_embedding_time = chunk_count * 0.3  # 0.3 seconds per chunk

        # Apply optimizations
        if config.get("enable_fast_track", False):
            base_conversion_time *= 0.7  # 30% faster conversion
            base_embedding_time *= 0.5  # 50% faster embedding

        if config.get("chunk_size", 4000) > 5000:
            base_chunking_time *= 0.8  # Larger chunks = less chunking time

        total_time = base_conversion_time + base_chunking_time + base_embedding_time

        logger.debug(
            "Estimated processing time",
            file_size_mb=file_size_mb,
            chunk_count=chunk_count,
            estimated_time=total_time,
            breakdown={
                "conversion": base_conversion_time,
                "chunking": base_chunking_time,
                "embedding": base_embedding_time,
            },
        )

        return total_time


class PerformanceTracker:
    """Track and analyze performance metrics."""

    def __init__(self):
        self.metrics = []

    def start_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Start tracking an operation."""
        start_time = time.time()

        metric = {
            "operation": operation_name,
            "start_time": start_time,
            "metadata": metadata or {},
        }

        self.metrics.append(metric)
        return start_time

    def end_operation(self, operation_name: str, start_time: float) -> float:
        """End tracking an operation."""
        end_time = time.time()
        duration = end_time - start_time

        # Find and update the metric
        for metric in reversed(self.metrics):
            if (
                metric["operation"] == operation_name
                and metric["start_time"] == start_time
            ):
                metric.update({"end_time": end_time, "duration": duration})
                break

        logger.info("Operation completed", operation=operation_name, duration=duration)

        return duration

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        completed_metrics = [m for m in self.metrics if "duration" in m]

        if not completed_metrics:
            return {"total_operations": 0}

        operations = {}
        for metric in completed_metrics:
            op_name = metric["operation"]
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(metric["duration"])

        summary = {"total_operations": len(completed_metrics), "operations": {}}

        for op_name, durations in operations.items():
            summary["operations"][op_name] = {
                "count": len(durations),
                "total_time": sum(durations),
                "avg_time": sum(durations) / len(durations),
                "min_time": min(durations),
                "max_time": max(durations),
            }

        return summary


# Global performance tracker
performance_tracker = PerformanceTracker()


def optimize_for_document(
    file_path: str, content_length: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get optimized configuration for a specific document."""

    path = Path(file_path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    # Estimate content length if not provided
    if content_length is None:
        # Rough estimate: 1KB per page for PDFs, 1:1 for text files
        if path.suffix.lower() == ".pdf":
            content_length = int(file_size_mb * 1024 * 2)  # 2 chars per byte estimate
        else:
            content_length = int(file_size_mb * 1024 * 1024)  # 1:1 for text

    # Get optimal configurations
    chunk_config = ProcessingOptimizer.get_optimal_chunk_config(
        file_size_mb, content_length, path.suffix.lower()
    )

    # Estimate chunk count
    estimated_chunks = max(1, content_length // chunk_config["chunk_size"])

    embedding_config = ProcessingOptimizer.get_optimal_embedding_config(
        estimated_chunks,
        "fast_track" if chunk_config.get("enable_fast_track", False) else "standard",
    )

    # Estimate processing time
    estimated_time = ProcessingOptimizer.estimate_processing_time(
        file_size_mb, estimated_chunks, {**chunk_config, **embedding_config}
    )

    logger.info(
        "Document optimization complete",
        file_path=file_path,
        file_size_mb=file_size_mb,
        estimated_chunks=estimated_chunks,
        estimated_time=estimated_time,
        chunk_config=chunk_config,
        embedding_config=embedding_config,
    )

    return chunk_config, embedding_config
