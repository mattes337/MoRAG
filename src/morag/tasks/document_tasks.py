from typing import Dict, Any, List
import structlog
from pathlib import Path

from morag.core.celery_app import celery_app
from morag.tasks.base import ProcessingTask
from morag.processors.document import document_processor
from morag.services.embedding import gemini_service
from morag.services.chunking import chunking_service
from morag.services.storage import qdrant_service
from morag.utils.text_processing import combine_text_and_summary

logger = structlog.get_logger()

@celery_app.task(bind=True, base=ProcessingTask)
async def process_document_task(
    self,
    file_path: str,
    source_type: str,
    metadata: Dict[str, Any],
    use_docling: bool = False
) -> Dict[str, Any]:
    """Process a document file through the complete pipeline."""

    try:
        self.log_step("Starting document processing", file_path=file_path)
        self.update_progress(0.1, "Validating file")

        # Validate file
        document_processor.validate_file(file_path)

        self.update_progress(0.2, "Parsing document")

        # Parse document
        parse_result = await document_processor.parse_document(
            file_path,
            use_docling=use_docling
        )

        self.log_step(
            "Document parsed",
            chunks_count=len(parse_result.chunks),
            images_count=len(parse_result.images),
            word_count=parse_result.word_count
        )

        self.update_progress(0.4, "Processing chunks")

        # Process chunks for embedding
        processed_chunks = []

        for i, chunk in enumerate(parse_result.chunks):
            # Generate summary for chunk
            self.update_progress(
                0.4 + (0.3 * i / len(parse_result.chunks)),
                f"Generating summary for chunk {i+1}/{len(parse_result.chunks)}"
            )

            try:
                summary_result = await gemini_service.generate_summary(
                    chunk.text,
                    max_length=100,
                    style="concise"
                )
                summary = summary_result.summary
            except Exception as e:
                logger.warning("Failed to generate summary for chunk", error=str(e))
                summary = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text

            # Combine text and summary for embedding
            combined_text = combine_text_and_summary(chunk.text, summary)

            processed_chunk = {
                "text": chunk.text,
                "summary": summary,
                "combined_text": combined_text,
                "source": file_path,
                "source_type": source_type,
                "chunk_index": i,
                "chunk_type": chunk.chunk_type,
                "metadata": {
                    **metadata,
                    **parse_result.metadata,
                    "page_number": chunk.page_number,
                    "element_id": chunk.element_id,
                    "chunk_metadata": chunk.metadata or {}
                }
            }
            processed_chunks.append(processed_chunk)

        self.update_progress(0.7, "Generating embeddings")

        # Generate embeddings
        texts_for_embedding = [chunk["combined_text"] for chunk in processed_chunks]

        embedding_results = await gemini_service.generate_embeddings_batch(
            texts_for_embedding,
            batch_size=5
        )

        embeddings = [result.embedding for result in embedding_results]

        self.update_progress(0.9, "Storing in vector database")

        # Store in vector database
        point_ids = await qdrant_service.store_chunks(processed_chunks, embeddings)

        self.update_progress(1.0, "Document processing completed")

        result = {
            "status": "success",
            "file_path": file_path,
            "chunks_processed": len(processed_chunks),
            "images_found": len(parse_result.images),
            "word_count": parse_result.word_count,
            "total_pages": parse_result.total_pages,
            "point_ids": point_ids,
            "metadata": parse_result.metadata
        }

        self.log_step("Document processing completed", **result)
        return result

    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        logger.error("Document processing task failed", error=str(e), file_path=file_path)
        self.update_progress(0.0, error_msg)
        raise
