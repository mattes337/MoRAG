"""Contextual retrieval service implementing Anthropic's contextual retrieval approach."""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
import structlog

from morag_core.models.document import Document, DocumentChunk
from morag_core.exceptions import ProcessingError, ExternalServiceError
from .embedding import GeminiEmbeddingService

logger = structlog.get_logger(__name__)


class ContextualRetrievalService:
    """Service for implementing Anthropic's contextual retrieval approach.

    This service generates contextual summaries for each chunk in relation to the complete document,
    and stores both dense and sparse vectors for improved retrieval accuracy.
    """

    def __init__(self, embedding_service: GeminiEmbeddingService):
        """Initialize contextual retrieval service.

        Args:
            embedding_service: Embedding service for generating vectors
        """
        self.embedding_service = embedding_service
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the service."""
        if not self._initialized:
            await self.embedding_service.initialize()
            self._initialized = True

    async def generate_contextual_chunks(
        self,
        document: Document,
        max_context_length: int = 500
    ) -> List[Dict[str, Any]]:
        """Generate contextual chunks with summaries for improved retrieval.

        Args:
            document: Document to process
            max_context_length: Maximum length for contextual summaries

        Returns:
            List of contextual chunk data with embeddings

        Raises:
            ProcessingError: If contextual processing fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Generate document summary for context
            document_summary = await self._generate_document_summary(document, max_context_length)

            contextual_chunks = []

            for i, chunk in enumerate(document.chunks):
                # Generate contextual summary for this chunk
                contextual_summary = await self._generate_chunk_context(
                    chunk, document_summary, document, max_context_length
                )

                # Create enhanced chunk content with context
                enhanced_content = f"{contextual_summary}\n\n{chunk.content}"

                # Generate dense embedding for enhanced content
                dense_embedding = await self.embedding_service.generate_embedding(
                    enhanced_content,
                    task_type="retrieval_document"
                )

                # Generate sparse embedding for original content (keyword-based)
                sparse_embedding = await self._generate_sparse_embedding(chunk.content)

                # Create contextual chunk data
                contextual_chunk = {
                    "chunk_id": f"{document.metadata.document_id or 'doc'}_{i}",
                    "original_content": chunk.content,
                    "contextual_summary": contextual_summary,
                    "enhanced_content": enhanced_content,
                    "dense_embedding": dense_embedding,
                    "sparse_embedding": sparse_embedding,
                    "metadata": {
                        "chunk_index": i,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "document_id": document.metadata.document_id,
                        "document_title": document.metadata.title,
                        "has_contextual_summary": True,
                        "context_length": len(contextual_summary),
                        **chunk.metadata
                    }
                }

                contextual_chunks.append(contextual_chunk)

                logger.debug("Generated contextual chunk",
                           chunk_index=i,
                           context_length=len(contextual_summary),
                           enhanced_content_length=len(enhanced_content))

            logger.info("Generated contextual chunks for document",
                       document_id=document.metadata.document_id,
                       chunk_count=len(contextual_chunks),
                       document_summary_length=len(document_summary))

            return contextual_chunks

        except Exception as e:
            logger.error("Failed to generate contextual chunks",
                        error=str(e),
                        error_type=e.__class__.__name__,
                        document_id=document.metadata.document_id)
            raise ProcessingError(f"Contextual retrieval processing failed: {str(e)}")

    async def _generate_document_summary(self, document: Document, max_length: int) -> str:
        """Generate a summary of the entire document for context.

        Args:
            document: Document to summarize
            max_length: Maximum summary length

        Returns:
            Document summary
        """
        try:
            # Use the embedding service's summarization capability
            full_text = document.raw_text or "\n\n".join([chunk.content for chunk in document.chunks])

            if not full_text:
                return "Document content not available."

            # Generate summary using Gemini
            language = document.metadata.get("language") if hasattr(document, 'metadata') else None
            summary_result = await self.embedding_service.generate_summary(
                full_text,
                max_length=max_length,
                language=language
            )

            return summary_result.summary if hasattr(summary_result, 'summary') else str(summary_result)

        except Exception as e:
            logger.warning("Failed to generate document summary, using fallback",
                          error=str(e))
            # Fallback: use first few sentences
            sentences = full_text.split('. ')[:3]
            return '. '.join(sentences) + '.' if sentences else "Document summary not available."

    async def _generate_chunk_context(
        self,
        chunk: DocumentChunk,
        document_summary: str,
        document: Document,
        max_length: int
    ) -> str:
        """Generate contextual summary for a specific chunk.

        Args:
            chunk: Chunk to generate context for
            document_summary: Overall document summary
            document: Full document
            max_length: Maximum context length

        Returns:
            Contextual summary for the chunk
        """
        try:
            # Create context prompt
            context_prompt = f"""
Document Summary: {document_summary}

Document Title: {document.metadata.title or 'Untitled'}

Please provide a brief contextual summary (max {max_length} characters) for the following text chunk,
explaining how it relates to the overall document and what specific information it contains:

Chunk Content:
{chunk.content[:1000]}...
"""

            # Generate contextual summary using Gemini
            summary_result = await self.embedding_service.summarize(
                context_prompt,
                max_length=max_length
            )

            context = summary_result.summary if hasattr(summary_result, 'summary') else str(summary_result)

            # Ensure context is within length limit
            if len(context) > max_length:
                context = context[:max_length-3] + "..."

            return context

        except Exception as e:
            logger.warning("Failed to generate chunk context, using fallback",
                          error=str(e),
                          chunk_index=getattr(chunk, 'index', 'unknown'))

            # Fallback context
            section_info = f" from section '{chunk.section}'" if chunk.section else ""
            page_info = f" on page {chunk.page_number}" if chunk.page_number else ""

            return f"This text chunk{section_info}{page_info} discusses: {chunk.content[:100]}..."

    async def _generate_sparse_embedding(self, text: str) -> Dict[str, float]:
        """Generate sparse (keyword-based) embedding for text.

        Args:
            text: Text to generate sparse embedding for

        Returns:
            Sparse embedding as keyword-weight dictionary
        """
        try:
            # Simple TF-IDF-like sparse embedding
            import re
            from collections import Counter

            # Extract keywords (simple approach)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_counts = Counter(words)

            # Calculate simple TF scores
            total_words = len(words)
            sparse_embedding = {}

            for word, count in word_counts.most_common(50):  # Top 50 keywords
                tf_score = count / total_words
                sparse_embedding[word] = tf_score

            return sparse_embedding

        except Exception as e:
            logger.warning("Failed to generate sparse embedding",
                          error=str(e))
            return {}

    async def store_contextual_chunks(
        self,
        contextual_chunks: List[Dict[str, Any]],
        vector_storage,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Store contextual chunks in vector database with dual embeddings.

        Args:
            contextual_chunks: List of contextual chunk data
            vector_storage: Vector storage service
            collection_name: Optional collection name

        Returns:
            List of stored point IDs
        """
        try:
            point_ids = []

            for chunk_data in contextual_chunks:
                # Store dense embedding
                dense_point_id = await vector_storage.store_vectors(
                    vectors=[chunk_data["dense_embedding"]],
                    metadata=[{
                        **chunk_data["metadata"],
                        "content": chunk_data["enhanced_content"],
                        "embedding_type": "dense",
                        "contextual_summary": chunk_data["contextual_summary"]
                    }],
                    collection_name=collection_name
                )

                # Store sparse embedding metadata (Qdrant doesn't natively support sparse vectors)
                sparse_metadata = {
                    **chunk_data["metadata"],
                    "content": chunk_data["original_content"],
                    "embedding_type": "sparse",
                    "sparse_keywords": json.dumps(chunk_data["sparse_embedding"]),
                    "contextual_summary": chunk_data["contextual_summary"]
                }

                # Create a dummy dense vector for sparse metadata storage
                dummy_vector = [0.0] * len(chunk_data["dense_embedding"])
                sparse_point_id = await vector_storage.store_vectors(
                    vectors=[dummy_vector],
                    metadata=[sparse_metadata],
                    collection_name=collection_name
                )

                point_ids.extend(dense_point_id + sparse_point_id)

            logger.info("Stored contextual chunks with dual embeddings",
                       chunk_count=len(contextual_chunks),
                       point_count=len(point_ids))

            return point_ids

        except Exception as e:
            logger.error("Failed to store contextual chunks",
                        error=str(e),
                        error_type=e.__class__.__name__)
            raise ProcessingError(f"Failed to store contextual chunks: {str(e)}")
