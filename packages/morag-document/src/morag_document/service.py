"""Document processing service implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import structlog

from morag_core.interfaces.service import BaseService, ServiceConfig, ServiceStatus
from morag_core.interfaces.processor import ProcessingConfig, ProcessingResult
from morag_core.interfaces.converter import ChunkingStrategy, ConversionOptions
from morag_core.models.document import Document
from morag_core.exceptions import ValidationError, ProcessingError
from morag_core.config import get_settings
from morag_embedding.service import GeminiEmbeddingService

from .processor import DocumentProcessor

logger = structlog.get_logger(__name__)


class DocumentService(BaseService):
    """Document processing service implementation."""

    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize document service.

        Args:
            config: Service configuration
        """
        self.config = config or ServiceConfig()
        self.processor = DocumentProcessor()
        self.embedding_service = None
        self._status = ServiceStatus.INITIALIZING

    async def initialize(self) -> bool:
        """Initialize service.

        Returns:
            True if initialization was successful
        """
        try:
            # Initialize embedding service if configured
            if hasattr(self.config, 'custom_options') and self.config.custom_options.get("enable_embedding", False):
                embedding_config = self.config.custom_options.get("embedding", {})
                self.embedding_service = GeminiEmbeddingService(ServiceConfig(**embedding_config))
                await self.embedding_service.initialize()

            self._status = ServiceStatus.READY
            return True
        except Exception as e:
            logger.error(
                "Failed to initialize document service",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            self._status = ServiceStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """Shutdown service.

        Returns:
            True if shutdown was successful
        """
        try:
            # Shutdown embedding service if initialized
            if self.embedding_service:
                await self.embedding_service.shutdown()

            self._status = ServiceStatus.STOPPED
            return True
        except Exception as e:
            logger.error(
                "Failed to shutdown document service",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            self._status = ServiceStatus.ERROR
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health check result
        """
        health = {
            "status": self._status.value,
            "processor": "ok",
        }

        # Check embedding service if initialized
        if self.embedding_service:
            embedding_health = await self.embedding_service.health_check()
            health["embedding"] = embedding_health

        return health

    async def process_document(self, file_path: Union[str, Path], **kwargs) -> ProcessingResult:
        """Process document.

        Args:
            file_path: Path to document file
            **kwargs: Additional processing options

        Returns:
            Processing result

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input is invalid
        """
        try:
            # Process document
            result = await self.processor.process_file(file_path, **kwargs)

            # Generate embeddings if enabled
            if self.embedding_service and kwargs.get("generate_embeddings", False):
                await self._generate_embeddings(result.document)

            return result
        except (ValidationError, ProcessingError) as e:
            # Re-raise validation and processing errors
            raise e
        except Exception as e:
            # Log and wrap other exceptions
            logger.error(
                "Document processing failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            raise ProcessingError(f"Failed to process document: {str(e)}")

    async def process_document_to_json(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """Process document and return structured JSON.

        Args:
            file_path: Path to document file
            **kwargs: Additional processing options

        Returns:
            Dictionary with structured JSON data

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input is invalid
        """
        try:
            # Process document
            result = await self.process_document(file_path, **kwargs)

            # Convert to JSON format
            return await self._convert_to_json(result, file_path)

        except Exception as e:
            logger.error(
                "Document JSON processing failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            return {
                "title": "",
                "filename": str(Path(file_path).name),
                "metadata": {},
                "chapters": [],
                "error": str(e)
            }

    async def process_text(self, text: str, **kwargs) -> ProcessingResult:
        """Process text document.

        Args:
            text: Text content
            **kwargs: Additional processing options

        Returns:
            Processing result

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input is invalid
        """
        try:
            # Create document from text
            document = Document(
                raw_text=text,
                metadata={
                    "title": kwargs.get("title", "Text Document"),
                    "file_type": "text",
                    "word_count": len(text.split()),
                }
            )

            # Get settings for default chunk configuration
            settings = get_settings()

            # Apply chunking
            chunking_strategy = kwargs.get("chunking_strategy", ChunkingStrategy.PARAGRAPH)
            chunk_size = kwargs.get("chunk_size", settings.default_chunk_size)
            chunk_overlap = kwargs.get("chunk_overlap", settings.default_chunk_overlap)

            # Create processing config
            config = ProcessingConfig(
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Apply chunking using processor's converter
            text_converter = self.processor.converters.get("text")
            if text_converter:
                options = ConversionOptions(
                    format_type="text",
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                await text_converter._chunk_document(document, options)

            # Generate embeddings if enabled
            if self.embedding_service and kwargs.get("generate_embeddings", False):
                await self._generate_embeddings(document)

            # Return processing result
            return ProcessingResult(
                document=document,
                metadata={
                    "quality_score": 1.0,  # Assume perfect quality for direct text input
                    "quality_issues": [],
                    "warnings": [],
                },
            )
        except Exception as e:
            # Log and wrap exceptions
            logger.error(
                "Text processing failed",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            raise ProcessingError(f"Failed to process text: {str(e)}")

    async def _convert_to_json(self, result: ProcessingResult, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Convert document processing result to structured JSON.

        Args:
            result: Document processing result
            file_path: Original file path

        Returns:
            Dictionary with structured JSON data
        """
        try:
            # Extract basic information
            filename = Path(file_path).name
            title = result.document.metadata.title or Path(file_path).stem

            # Build chapters/chunks
            chapters = []
            if result.document.chunks:
                for chunk in result.document.chunks:
                    chapter_data = {
                        "title": chunk.section or f"Section {chunk.chunk_index + 1}",
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "chapter_index": chunk.chunk_index,
                        "metadata": chunk.metadata
                    }
                    chapters.append(chapter_data)
            else:
                # No chunks - create single chapter from raw text
                chapters.append({
                    "title": title,
                    "content": result.document.raw_text or "",
                    "page_number": 1,
                    "chapter_index": 0,
                    "metadata": {}
                })

            # Build metadata
            metadata = {
                "source_type": result.document.metadata.source_type.value if result.document.metadata.source_type else "unknown",
                "source_name": result.document.metadata.source_name,
                "source_path": result.document.metadata.source_path,
                "file_size": result.document.metadata.file_size,
                "page_count": result.document.metadata.page_count,
                "word_count": result.document.metadata.word_count,
                "author": result.document.metadata.author,
                "created_at": result.document.metadata.created_at.isoformat() if result.document.metadata.created_at else None,
                "modified_at": result.document.metadata.modified_at.isoformat() if result.document.metadata.modified_at else None,
                "language": result.document.metadata.language,
                "mime_type": result.document.metadata.mime_type,
                "quality_score": result.metadata.get("quality_score", 0.0),
                "quality_issues": result.metadata.get("quality_issues", []),
                "warnings": result.metadata.get("warnings", []),
                "chunks_count": len(result.document.chunks),
                "processing_time": getattr(result, 'processing_time', 0.0)
            }

            return {
                "title": title,
                "filename": filename,
                "metadata": metadata,
                "chapters": chapters
            }

        except Exception as e:
            logger.error("Failed to convert document result to JSON", error=str(e))
            return {
                "title": "",
                "filename": str(Path(file_path).name),
                "metadata": {},
                "chapters": [],
                "error": str(e)
            }

    async def _generate_embeddings(self, document: Document) -> None:
        """Generate embeddings for document chunks.

        Args:
            document: Document to generate embeddings for

        Raises:
            ProcessingError: If embedding generation fails
        """
        if not self.embedding_service:
            raise ProcessingError("Embedding service not initialized")

        try:
            # Get text from chunks
            texts = [chunk.content for chunk in document.chunks]
            if not texts:
                logger.warning("No chunks found in document for embedding generation")
                return

            # Generate embeddings
            batch_result = await self.embedding_service.embed_batch(texts)

            # Update chunks with embeddings
            for i, result in enumerate(batch_result.results):
                if i < len(document.chunks):
                    document.chunks[i].embedding = result.embedding
                    document.chunks[i].embedding_model = result.model

        except Exception as e:
            logger.error(
                "Failed to generate embeddings for document",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            raise ProcessingError(f"Failed to generate embeddings: {str(e)}")

    async def summarize_document(self, document: Document, **kwargs) -> str:
        """Generate summary for document.

        Args:
            document: Document to summarize
            **kwargs: Additional summarization options

        Returns:
            Document summary

        Raises:
            ProcessingError: If summarization fails
        """
        if not self.embedding_service:
            raise ProcessingError("Embedding service not initialized")

        try:
            # Get document text
            text = document.raw_text
            if not text:
                # Fallback to joining chunks
                text = "\n\n".join([chunk.content for chunk in document.chunks])

            if not text:
                raise ProcessingError("No text found in document for summarization")

            # Generate summary
            max_length = kwargs.get("max_length", 1000)
            summary_result = await self.embedding_service.summarize(
                text, max_length=max_length
            )

            return summary_result.summary

        except Exception as e:
            logger.error(
                "Failed to summarize document",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            raise ProcessingError(f"Failed to summarize document: {str(e)}")