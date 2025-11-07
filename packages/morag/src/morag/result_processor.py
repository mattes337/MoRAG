"""Result and data processing for ingestion workflow."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone
import uuid

from morag_core.utils.logging import get_logger

logger = get_logger(__name__)


class ResultProcessor:
    """Handles creation and writing of ingest results and data."""

    def create_ingest_result(
        self,
        source_path: str,
        content: str,
        chunks: List[str],
        embeddings: List[List[float]],
        chunk_metadata_list: List[Dict[str, Any]],
        summary: str,
        content_type: str = 'document',
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create comprehensive ingest result with all processing information.

        Args:
            source_path: Path to source file
            content: Original content
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            chunk_metadata_list: Metadata for each chunk
            summary: Document summary
            content_type: Type of content
            base_metadata: Base metadata

        Returns:
            Complete ingest result dictionary
        """
        if base_metadata is None:
            base_metadata = {}

        # Calculate processing statistics
        total_chunks = len(chunks)
        total_words = len(content.split()) if content else 0
        avg_chunk_size = sum(len(chunk.split()) for chunk in chunks) / total_chunks if chunks else 0

        ingest_result = {
            "source": {
                "path": source_path,
                "filename": Path(source_path).name,
                "content_type": content_type,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "file_size": len(content) if content else 0,
            },
            "content": {
                "raw_text": content,
                "summary": summary,
                "word_count": total_words,
                "character_count": len(content) if content else 0,
            },
            "processing": {
                "chunk_count": total_chunks,
                "average_chunk_size": round(avg_chunk_size, 1),
                "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
                "processing_method": "chunk_processor_v2",
            },
            "chunks": []
        }

        # Add base metadata
        if base_metadata:
            ingest_result["metadata"] = base_metadata.copy()

        # Process each chunk
        for i, (chunk_text, embedding, chunk_metadata) in enumerate(zip(chunks, embeddings, chunk_metadata_list)):
            chunk_data = {
                "index": i,
                "text": chunk_text,
                "embedding": embedding,
                "metadata": chunk_metadata,
                "statistics": {
                    "word_count": len(chunk_text.split()),
                    "character_count": len(chunk_text),
                    "embedding_norm": round(sum(x*x for x in embedding) ** 0.5, 4) if embedding else 0,
                }
            }

            ingest_result["chunks"].append(chunk_data)

        logger.info("Created ingest result",
                   source=source_path,
                   chunk_count=total_chunks,
                   content_type=content_type)

        return ingest_result

    def _serialize_enhanced_processing(self, enhanced_processing: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize enhanced processing results for JSON storage."""
        serialized = {}

        for key, value in enhanced_processing.items():
            if hasattr(value, '__dict__'):
                # Object with attributes - convert to dict
                serialized[key] = value.__dict__ if hasattr(value, '__dict__') else str(value)
            elif isinstance(value, list):
                # List - process each item
                serialized[key] = []
                for item in value:
                    if hasattr(item, '__dict__'):
                        serialized[key].append(item.__dict__)
                    else:
                        serialized[key].append(str(item) if not isinstance(item, (str, int, float, bool, dict)) else item)
            elif isinstance(value, dict):
                # Dictionary - recursively process
                serialized[key] = self._serialize_enhanced_processing(value)
            else:
                # Primitive type or already serializable
                serialized[key] = value

        return serialized

    def write_ingest_result_file(self, source_path: str, ingest_result: Dict[str, Any]) -> str:
        """Write ingest result to JSON file."""
        source_path_obj = Path(source_path)
        result_filename = f"{source_path_obj.stem}.ingest_result.json"
        result_path = source_path_obj.parent / result_filename

        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(ingest_result, f, indent=2, ensure_ascii=False)

            logger.info("Wrote ingest result file", path=str(result_path))
            return str(result_path)

        except Exception as e:
            logger.error("Failed to write ingest result file",
                        path=str(result_path),
                        error=str(e))
            raise

    def create_ingest_data(
        self,
        source_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        chunk_metadata_list: List[Dict[str, Any]],
        document_id: str = None,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """Create ingest data for database storage.

        Args:
            source_path: Path to source file
            chunks: List of text chunks
            embeddings: List of embeddings
            chunk_metadata_list: Metadata for each chunk
            document_id: Document identifier
            collection_name: Collection name for storage

        Returns:
            Ingest data dictionary ready for database storage
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        if collection_name is None:
            collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')

        # Prepare data points for database storage
        data_points = []

        for i, (chunk_text, embedding, chunk_metadata) in enumerate(zip(chunks, embeddings, chunk_metadata_list)):
            point_id = str(uuid.uuid4())

            # Prepare payload with all metadata
            payload = {
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk_text,
                "source_path": source_path,
                "source_filename": Path(source_path).name,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                **chunk_metadata  # Include all chunk-specific metadata
            }

            data_point = {
                "id": point_id,
                "vector": embedding,
                "payload": payload
            }

            data_points.append(data_point)

        ingest_data = {
            "document_id": document_id,
            "collection_name": collection_name,
            "source_path": source_path,
            "total_chunks": len(chunks),
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_points": data_points
        }

        logger.info("Created ingest data",
                   document_id=document_id,
                   chunk_count=len(chunks),
                   collection=collection_name)

        return ingest_data

    def write_ingest_data_file(self, source_path: str, ingest_data: Dict[str, Any]) -> str:
        """Write ingest data to JSON file."""
        source_path_obj = Path(source_path)
        data_filename = f"{source_path_obj.stem}.ingest_data.json"
        data_path = source_path_obj.parent / data_filename

        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(ingest_data, f, indent=2, ensure_ascii=False)

            logger.info("Wrote ingest data file", path=str(data_path))
            return str(data_path)

        except Exception as e:
            logger.error("Failed to write ingest data file",
                        path=str(data_path),
                        error=str(e))
            raise


__all__ = ["ResultProcessor"]
