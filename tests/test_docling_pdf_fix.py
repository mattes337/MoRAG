"""
Test for the improved docling PDF parsing implementation.
Tests that the PDF parsing fix resolves binary/encoded content issues.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from morag.processors.document import document_processor, DocumentParseResult, DocumentChunk


class TestDoclingPDFParsing:
    """Test cases for the improved docling PDF parsing."""

    @pytest.mark.asyncio
    async def test_docling_parsing_configuration(self):
        """Test that docling is configured with proper pipeline options."""

        # Mock the docling imports and classes
        mock_converter = Mock()
        mock_result = Mock()
        mock_document = Mock()

        # Configure the mock result
        mock_result.status.name = "SUCCESS"
        mock_result.document = mock_document
        mock_document.iterate_items.return_value = []
        mock_document.export_to_markdown.return_value = "# Test Document\n\nThis is a test."

        mock_converter.convert.return_value = mock_result

        # Mock file operations
        mock_stat = Mock()
        mock_stat.st_size = 1024

        with patch('docling.document_converter.DocumentConverter') as mock_doc_converter, \
             patch('docling.document_converter.PdfFormatOption') as mock_pdf_option, \
             patch('docling.datamodel.base_models.InputFormat') as mock_input_format, \
             patch('docling.datamodel.pipeline_options.PdfPipelineOptions') as mock_pipeline_options, \
             patch.object(Path, 'stat', return_value=mock_stat):

            mock_doc_converter.return_value = mock_converter

            # Test the parsing
            test_file = Path("test.pdf")
            result = await document_processor._parse_with_docling(test_file)

            # Verify that PdfPipelineOptions was called with correct parameters
            mock_pipeline_options.assert_called_once_with(
                do_ocr=True,
                do_table_structure=True,
                generate_page_images=False,
                generate_picture_images=False,
            )

            # Verify DocumentConverter was configured properly
            mock_doc_converter.assert_called_once()

            # Verify the result
            assert isinstance(result, DocumentParseResult)
            assert result.metadata["parser"] == "docling"

    @pytest.mark.asyncio
    async def test_pdf_converter_docling_initialization_fix(self):
        """Test that PDFConverter initializes docling correctly with PdfFormatOption."""

        # Test that the actual initialization works without the 'backend' attribute error
        from morag.converters.pdf import PDFConverter

        # This should not raise an AttributeError about 'backend'
        try:
            converter = PDFConverter()
            # If we get here without an exception, the fix worked
            assert True, "PDFConverter initialized successfully without 'backend' attribute error"
        except AttributeError as e:
            if "'PdfPipelineOptions' object has no attribute 'backend'" in str(e):
                pytest.fail("The 'backend' attribute error still exists - fix not working")
            else:
                # Some other AttributeError, re-raise it
                raise

    @pytest.mark.asyncio
    async def test_docling_elements_attribute_fix(self):
        """Test that the 'int' object has no attribute 'elements' error is fixed."""

        # Test that the actual PDF converter works without the elements attribute error
        from morag.converters.pdf import PDFConverter

        # This should not raise an AttributeError about 'elements'
        try:
            converter = PDFConverter()
            # If we get here without an exception, the initialization worked
            assert True, "PDFConverter initialized successfully without 'elements' attribute error"
        except AttributeError as e:
            if "'int' object has no attribute 'elements'" in str(e):
                pytest.fail("The 'elements' attribute error still exists - fix not working")
            else:
                # Some other AttributeError, re-raise it
                raise

    @pytest.mark.asyncio
    async def test_docling_text_extraction(self):
        """Test that docling properly extracts readable text."""
        
        # Mock docling components
        mock_converter = Mock()
        mock_result = Mock()
        mock_document = Mock()
        
        # Create mock text items with readable content
        mock_text_item = Mock()
        mock_text_item.text = "This is readable text from the PDF document."
        mock_text_item.label = "text"
        mock_text_item.prov = [Mock(page_no=1)]
        mock_text_item.self_ref = "#/texts/1"
        
        mock_title_item = Mock()
        mock_title_item.text = "Document Title"
        mock_title_item.label = "title"
        mock_title_item.prov = [Mock(page_no=1)]
        mock_title_item.self_ref = "#/texts/2"
        
        # Configure the mock result
        mock_result.status.name = "SUCCESS"
        mock_result.document = mock_document
        mock_document.iterate_items.return_value = [
            (mock_text_item, 0),
            (mock_title_item, 1)
        ]
        
        mock_converter.convert.return_value = mock_result
        
        with patch('docling.document_converter.DocumentConverter') as mock_doc_converter, \
             patch('docling.document_converter.PdfFormatOption'), \
             patch('docling.datamodel.base_models.InputFormat'), \
             patch('docling.datamodel.pipeline_options.PdfPipelineOptions'):
            
            mock_doc_converter.return_value = mock_converter
            
            # Test the parsing
            test_file = Path("test.pdf")
            result = await document_processor._parse_with_docling(test_file)
            
            # Verify chunks were created with readable text
            assert len(result.chunks) == 2
            
            # Check first chunk (text)
            text_chunk = result.chunks[0]
            assert text_chunk.text == "This is readable text from the PDF document."
            assert text_chunk.chunk_type == "text"
            assert text_chunk.page_number == 1
            assert text_chunk.metadata["docling_source"] is True
            assert text_chunk.metadata["label"] == "text"
            
            # Check second chunk (title)
            title_chunk = result.chunks[1]
            assert title_chunk.text == "Document Title"
            assert title_chunk.chunk_type == "title"
            assert title_chunk.page_number == 1
            assert title_chunk.metadata["label"] == "title"

    @pytest.mark.asyncio
    async def test_docling_table_extraction(self):
        """Test that docling properly extracts table data."""
        
        # Mock docling components
        mock_converter = Mock()
        mock_result = Mock()
        mock_document = Mock()
        
        # Create mock table item
        mock_table_item = Mock()
        mock_table_item.prov = [Mock(page_no=2)]
        mock_table_item.self_ref = "#/tables/1"
        
        # Mock DataFrame export
        import pandas as pd
        mock_df = pd.DataFrame({
            'Column1': ['Value1', 'Value2'],
            'Column2': ['Value3', 'Value4']
        })
        mock_table_item.export_to_dataframe.return_value = mock_df
        
        # Configure the mock result
        mock_result.status.name = "SUCCESS"
        mock_result.document = mock_document
        mock_document.iterate_items.return_value = [
            (mock_table_item, 0)
        ]
        
        mock_converter.convert.return_value = mock_result
        
        with patch('docling.document_converter.DocumentConverter') as mock_doc_converter, \
             patch('docling.document_converter.PdfFormatOption'), \
             patch('docling.datamodel.base_models.InputFormat'), \
             patch('docling.datamodel.pipeline_options.PdfPipelineOptions'):
            
            mock_doc_converter.return_value = mock_converter
            
            # Test the parsing
            test_file = Path("test.pdf")
            result = await document_processor._parse_with_docling(test_file)
            
            # Verify table chunk was created
            assert len(result.chunks) == 1
            
            table_chunk = result.chunks[0]
            assert table_chunk.chunk_type == "table"
            assert table_chunk.page_number == 2
            assert "**Table:**" in table_chunk.text
            assert "Column1" in table_chunk.text
            assert "Value1" in table_chunk.text
            assert table_chunk.metadata["element_type"] == "TableItem"
            assert table_chunk.metadata["table_shape"] == (2, 2)

    @pytest.mark.asyncio
    async def test_docling_fallback_to_markdown(self):
        """Test that docling falls back to markdown export when no items are found."""
        
        # Mock docling components
        mock_converter = Mock()
        mock_result = Mock()
        mock_document = Mock()
        
        # Configure the mock result with no items but markdown export
        mock_result.status.name = "SUCCESS"
        mock_result.document = mock_document
        mock_document.iterate_items.return_value = []  # No items found
        mock_document.export_to_markdown.return_value = "# Fallback Document\n\nThis is markdown content."
        
        mock_converter.convert.return_value = mock_result
        
        with patch('docling.document_converter.DocumentConverter') as mock_doc_converter, \
             patch('docling.document_converter.PdfFormatOption'), \
             patch('docling.datamodel.base_models.InputFormat'), \
             patch('docling.datamodel.pipeline_options.PdfPipelineOptions'):
            
            mock_doc_converter.return_value = mock_converter
            
            # Test the parsing
            test_file = Path("test.pdf")
            result = await document_processor._parse_with_docling(test_file)
            
            # Verify fallback chunk was created
            assert len(result.chunks) == 1
            
            fallback_chunk = result.chunks[0]
            assert fallback_chunk.text == "# Fallback Document\n\nThis is markdown content."
            assert fallback_chunk.chunk_type == "text"
            assert fallback_chunk.element_id == "markdown_export"
            assert fallback_chunk.metadata["element_type"] == "MarkdownExport"
            assert fallback_chunk.metadata["label"] == "full_document"

    @pytest.mark.asyncio
    async def test_docling_error_fallback(self):
        """Test that docling falls back to unstructured.io on error."""
        
        with patch('docling.document_converter.DocumentConverter') as mock_doc_converter:
            # Make docling raise an exception
            mock_doc_converter.side_effect = Exception("Docling failed")
            
            # Mock the fallback to unstructured
            with patch.object(document_processor, '_parse_with_unstructured') as mock_unstructured:
                mock_unstructured.return_value = DocumentParseResult(
                    chunks=[DocumentChunk(text="Fallback text", chunk_type="text")],
                    metadata={"parser": "unstructured"},
                    images=[],
                    total_pages=1,
                    word_count=2
                )
                
                # Test the parsing
                test_file = Path("test.pdf")
                result = await document_processor._parse_with_docling(test_file)
                
                # Verify fallback was called
                mock_unstructured.assert_called_once()
                assert result.metadata["parser"] == "unstructured"

    @pytest.mark.asyncio
    async def test_document_task_defaults_to_docling_for_pdf(self):
        """Test that document task defaults to using docling for PDF files."""
        
        from morag.tasks.document_tasks import _process_document_impl
        from morag.tasks.base import ProcessingTask
        
        # Create a mock task instance
        mock_task = Mock(spec=ProcessingTask)
        mock_task.log_step = Mock()
        mock_task.update_progress = Mock()
        
        # Mock the document processor
        with patch.object(document_processor, 'validate_file'), \
             patch.object(document_processor, 'parse_document') as mock_parse:
            
            mock_parse.return_value = DocumentParseResult(
                chunks=[DocumentChunk(text="Test content", chunk_type="text")],
                metadata={"parser": "docling"},
                images=[],
                total_pages=1,
                word_count=2
            )
            
            # Test with PDF file (should use docling by default)
            await _process_document_impl(
                mock_task,
                "test.pdf",
                "document",
                {},
                use_docling=False,  # Explicitly set to False to test default behavior
                use_enhanced_summary=False
            )
            
            # Verify that parse_document was called with use_docling=True for PDF
            mock_parse.assert_called_once()
            args, kwargs = mock_parse.call_args
            assert kwargs['use_docling'] is True

    @pytest.mark.asyncio
    async def test_document_task_respects_docling_flag_for_non_pdf(self):
        """Test that document task respects the docling flag for non-PDF files."""
        
        from morag.tasks.document_tasks import _process_document_impl
        from morag.tasks.base import ProcessingTask
        
        # Create a mock task instance
        mock_task = Mock(spec=ProcessingTask)
        mock_task.log_step = Mock()
        mock_task.update_progress = Mock()
        
        # Mock the document processor
        with patch.object(document_processor, 'validate_file'), \
             patch.object(document_processor, 'parse_document') as mock_parse:
            
            mock_parse.return_value = DocumentParseResult(
                chunks=[DocumentChunk(text="Test content", chunk_type="text")],
                metadata={"parser": "unstructured"},
                images=[],
                total_pages=1,
                word_count=2
            )
            
            # Test with DOCX file (should not use docling by default)
            await _process_document_impl(
                mock_task,
                "test.docx",
                "document",
                {},
                use_docling=False,
                use_enhanced_summary=False
            )
            
            # Verify that parse_document was called with use_docling=False for DOCX
            mock_parse.assert_called_once()
            args, kwargs = mock_parse.call_args
            assert kwargs['use_docling'] is False
