"""Document processor wrapper for stage processing."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import structlog

from .interface import StageProcessor, ProcessorResult

logger = structlog.get_logger(__name__)

# Import core exceptions
try:
    from morag_core.exceptions import ProcessingError
except ImportError:
    class ProcessingError(Exception):  # type: ignore
        pass


class DocumentStageProcessor(StageProcessor):
    """Stage processor for document content using morag_document package."""

    def __init__(self):
        """Initialize document stage processor."""
        self._document_processor = None
        self._services = None

    def _get_document_processor(self):
        """Get or create document processor instance."""
        if self._document_processor is None:
            try:
                from morag_document import DocumentProcessor, DocumentConfig
                config = DocumentConfig(
                    extract_text=True,
                    extract_tables=True,
                    extract_images=True,
                    ocr_enabled=True
                )
                self._document_processor = DocumentProcessor(config)
            except ImportError as e:
                raise ProcessingError(f"Document processor not available: {e}")
        return self._document_processor

    def _get_services(self):
        """Get MoRAG services for document processing."""
        if self._services is None:
            try:
                from morag_services import MoRAGServices
                self._services = MoRAGServices()
            except ImportError as e:
                raise ProcessingError(f"MoRAG services not available: {e}")
        return self._services

    def supports_content_type(self, content_type: str) -> bool:
        """Check if this processor supports the given content type."""
        return content_type.upper() == "DOCUMENT"

    async def process(
        self,
        input_file: Path,
        output_file: Path,
        config: Dict[str, Any]
    ) -> ProcessorResult:
        """Process document file to markdown."""
        logger.info("Processing document file", input_file=str(input_file))

        try:
            # Try using morag_document processor first
            try:
                processor = self._get_document_processor()

                # Convert config to DocumentConfig
                from morag_document import DocumentConfig
                doc_config = DocumentConfig(
                    extract_text=config.get('extract_text', True),
                    extract_tables=config.get('extract_tables', True),
                    extract_images=config.get('extract_images', True),
                    ocr_enabled=config.get('ocr_enabled', True),
                    preserve_formatting=config.get('preserve_formatting', True)
                )

                result = await processor.process_document(input_file, doc_config)

                metadata = {
                    "title": result.metadata.title or input_file.stem,
                    "source": str(input_file),
                    "type": "document",
                    "format": input_file.suffix.lower(),
                    "pages": result.metadata.pages,
                    "language": result.metadata.language,
                    "author": result.metadata.author,
                    "created_date": result.metadata.created_date,
                    "modified_date": result.metadata.modified_date,
                    "created_at": datetime.now().isoformat()
                }

                content = f"\n# Document Analysis\n\n"

                if result.text_content:
                    content += f"## Content\n\n{result.text_content}\n\n"

                if result.tables:
                    content += f"## Tables\n\n"
                    for i, table in enumerate(result.tables):
                        content += f"### Table {i+1}\n\n"
                        content += table.to_markdown() + "\n\n"

                if result.images:
                    content += f"## Images\n\n"
                    for i, image in enumerate(result.images):
                        content += f"### Image {i+1}\n\n"
                        if image.caption:
                            content += f"**Caption**: {image.caption}\n\n"
                        if image.text:
                            content += f"**Extracted Text**: {image.text}\n\n"

                content += f"## Document Information\n\n"
                content += f"- **Format**: {input_file.suffix.lower()}\n"
                if result.metadata.pages:
                    content += f"- **Pages**: {result.metadata.pages}\n"
                if result.metadata.author:
                    content += f"- **Author**: {result.metadata.author}\n"
                if result.metadata.language:
                    content += f"- **Language**: {result.metadata.language}\n"

                markdown_content = self.create_markdown_with_metadata(content, metadata)
                output_file.write_text(markdown_content, encoding='utf-8')

                return ProcessorResult(
                    content=content,
                    metadata=metadata,
                    metrics={
                        "pages": result.metadata.pages,
                        "content_length": len(result.text_content or ""),
                        "has_tables": len(result.tables) > 0 if result.tables else False,
                        "has_images": len(result.images) > 0 if result.images else False
                    },
                    final_output_file=output_file
                )

            except Exception as doc_error:
                logger.warning("morag_document processor failed, trying MoRAG services", error=str(doc_error))

                # Fallback to MoRAG services
                services = self._get_services()

                # Prepare options for document service
                options = {
                    'extract_tables': config.get('extract_tables', True),
                    'extract_images': config.get('extract_images', True),
                    'ocr_enabled': config.get('ocr_enabled', True),
                    'preserve_formatting': config.get('preserve_formatting', True)
                }

                # Use document service
                result = await services.process_document(str(input_file), options)

                metadata = {
                    "title": result.metadata.get('title') or input_file.stem,
                    "source": str(input_file),
                    "type": "document",
                    "format": input_file.suffix.lower(),
                    "created_at": datetime.now().isoformat(),
                    **result.metadata
                }

                content = f"\n# Document Analysis\n\n"
                if result.text_content:
                    content += f"## Content\n\n{result.text_content}\n\n"

                markdown_content = self.create_markdown_with_metadata(content, metadata)
                output_file.write_text(markdown_content, encoding='utf-8')

                return ProcessorResult(
                    content=content,
                    metadata=metadata,
                    metrics={
                        "content_length": len(result.text_content or ""),
                        "has_tables": config.get('extract_tables', True)
                    },
                    final_output_file=output_file
                )

        except Exception as e:
            logger.error("Document processing failed", input_file=str(input_file), error=str(e))
            raise ProcessingError(f"Document processing failed for {input_file}: {e}")
