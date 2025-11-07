"""Data file writer for serializing processing results to JSON files."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .services import ProcessingResult

logger = logging.getLogger(__name__)


class DataFileWriter:
    """Writes entities, relations, summaries, chunks, and metadata to JSON files."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the data file writer.

        Args:
            output_dir: Directory to write data files to. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_entity(self, entity) -> Dict[str, Any]:
        """Serialize an entity to a dictionary.

        Args:
            entity: Entity object to serialize

        Returns:
            Dictionary representation of the entity
        """
        try:
            return {
                "id": getattr(entity, "id", None),
                "name": getattr(entity, "name", ""),
                "type": getattr(entity, "type", "UNKNOWN"),
                "confidence": getattr(entity, "confidence", 0.0),
                "source_doc_id": getattr(entity, "source_doc_id", ""),
                "metadata": getattr(entity, "metadata", {}),
                "created_at": getattr(entity, "created_at", None),
                "updated_at": getattr(entity, "updated_at", None),
            }
        except Exception as e:
            logger.warning(f"Failed to serialize entity: {str(e)}")
            return {"error": f"Failed to serialize entity: {str(e)}"}

    def _serialize_relation(self, relation) -> Dict[str, Any]:
        """Serialize a relation to a dictionary.

        Args:
            relation: Relation object to serialize

        Returns:
            Dictionary representation of the relation
        """
        try:
            return {
                "id": getattr(relation, "id", None),
                "source_entity_id": getattr(relation, "source_entity_id", ""),
                "target_entity_id": getattr(relation, "target_entity_id", ""),
                "type": getattr(relation, "type", "RELATED_TO"),
                "confidence": getattr(relation, "confidence", 0.0),
                "source_doc_id": getattr(relation, "source_doc_id", ""),
                "metadata": getattr(relation, "metadata", {}),
                "created_at": getattr(relation, "created_at", None),
                "updated_at": getattr(relation, "updated_at", None),
            }
        except Exception as e:
            logger.warning(f"Failed to serialize relation: {str(e)}")
            return {"error": f"Failed to serialize relation: {str(e)}"}

    def _serialize_chunk(self, chunk) -> Dict[str, Any]:
        """Serialize a chunk to a dictionary.

        Args:
            chunk: Chunk object to serialize

        Returns:
            Dictionary representation of the chunk
        """
        try:
            if isinstance(chunk, tuple) and len(chunk) == 2:
                # Handle (content, metadata) tuple format
                content, metadata = chunk
                return {"content": content, "metadata": metadata or {}}
            else:
                # Handle object format
                return {
                    "id": getattr(chunk, "id", None),
                    "content": getattr(chunk, "content", getattr(chunk, "text", "")),
                    "metadata": getattr(chunk, "metadata", {}),
                    "chunk_index": getattr(chunk, "chunk_index", None),
                    "start_char": getattr(chunk, "start_char", None),
                    "end_char": getattr(chunk, "end_char", None),
                }
        except Exception as e:
            logger.warning(f"Failed to serialize chunk: {str(e)}")
            return {"error": f"Failed to serialize chunk: {str(e)}"}

    def write_processing_data(
        self,
        source_path: str,
        entities: Optional[List] = None,
        relations: Optional[List] = None,
        chunks: Optional[List] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processing_result: Optional[ProcessingResult] = None,
        graph_result: Optional[Any] = None,
    ) -> Path:
        """Write processing data to a JSON file.

        Args:
            source_path: Path to the source file being processed
            entities: List of entities extracted
            relations: List of relations extracted
            chunks: List of chunks created
            summary: Summary text
            metadata: Additional metadata
            processing_result: Complete processing result object
            graph_result: Graph processing result

        Returns:
            Path to the created data file
        """
        try:
            # Generate output filename based on source path
            source_name = Path(source_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{source_name}_data_{timestamp}.json"
            output_path = self.output_dir / output_filename

            # Prepare data structure
            data = {
                "source_path": source_path,
                "created_at": datetime.now().isoformat(),
                "entities": [],
                "relations": [],
                "chunks": [],
                "summary": summary or "",
                "metadata": metadata or {},
                "processing_result": None,
                "graph_result": None,
            }

            # Serialize entities
            if entities:
                data["entities"] = [
                    self._serialize_entity(entity) for entity in entities
                ]
                logger.info(f"Serialized {len(entities)} entities")

            # Serialize relations
            if relations:
                data["relations"] = [
                    self._serialize_relation(relation) for relation in relations
                ]
                logger.info(f"Serialized {len(relations)} relations")

            # Serialize chunks
            if chunks:
                data["chunks"] = [self._serialize_chunk(chunk) for chunk in chunks]
                logger.info(f"Serialized {len(chunks)} chunks")

            # Serialize processing result
            if processing_result:
                try:
                    data["processing_result"] = {
                        "content_type": str(processing_result.content_type)
                        if processing_result.content_type
                        else None,
                        "content_path": processing_result.content_path,
                        "content_url": processing_result.content_url,
                        "text_content": processing_result.text_content,
                        "metadata": processing_result.metadata,
                        "extracted_files": processing_result.extracted_files,
                        "processing_time": processing_result.processing_time,
                        "success": processing_result.success,
                        "error_message": processing_result.error_message,
                        "raw_result": processing_result.raw_result,
                    }
                except Exception as e:
                    logger.warning(f"Failed to serialize processing result: {str(e)}")
                    data["processing_result"] = {
                        "error": f"Failed to serialize: {str(e)}"
                    }

            # Serialize graph result
            if graph_result:
                try:
                    data["graph_result"] = {
                        "success": getattr(graph_result, "success", False),
                        "entities_count": getattr(graph_result, "entities_count", 0),
                        "relations_count": getattr(graph_result, "relations_count", 0),
                        "processing_time": getattr(
                            graph_result, "processing_time", 0.0
                        ),
                        "error_message": getattr(graph_result, "error_message", None),
                        "database_results": getattr(
                            graph_result, "database_results", []
                        ),
                    }
                except Exception as e:
                    logger.warning(f"Failed to serialize graph result: {str(e)}")
                    data["graph_result"] = {"error": f"Failed to serialize: {str(e)}"}

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Data file written successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to write data file: {str(e)}")
            raise

    def read_processing_data(self, data_file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read processing data from a JSON file.

        Args:
            data_file_path: Path to the data file

        Returns:
            Dictionary containing the processing data
        """
        try:
            with open(data_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Data file read successfully: {data_file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to read data file {data_file_path}: {str(e)}")
            raise

    def generate_filename_for_source(
        self, source_path: str, suffix: str = "data"
    ) -> str:
        """Generate a data filename for a given source path.

        Args:
            source_path: Path to the source file
            suffix: Suffix to add to the filename

        Returns:
            Generated filename
        """
        source_name = Path(source_path).stem
        return f"{source_name}_{suffix}.json"
