"""Text document converter implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import structlog
import markdown
from bs4 import BeautifulSoup

from morag_core.interfaces.converter import (
    ConversionOptions,
    ConversionError,
)
from morag_core.models.document import Document

from .base import DocumentConverter

logger = structlog.get_logger(__name__)


class TextConverter(DocumentConverter):
    """Text document converter implementation."""

    def __init__(self):
        """Initialize text converter."""
        super().__init__()
        self.supported_formats = {"text", "txt", "markdown", "md", "html", "htm"}

    async def _extract_text(self, file_path: Path, document: Document, options: ConversionOptions) -> Document:
        """Extract text from text document.

        Args:
            file_path: Path to text file
            document: Document to update
            options: Conversion options

        Returns:
            Updated document

        Raises:
            ConversionError: If text extraction fails
        """
        try:
            # Determine file type from extension
            file_extension = file_path.suffix.lower().lstrip('.')
            
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()
            
            # Process based on file type
            if file_extension in {"md", "markdown"}:
                # Convert markdown to HTML, then extract text
                html_content = markdown.markdown(content)
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n\n")
                
                # Extract title from first heading if available
                title_tag = soup.find(["h1", "h2", "h3"])
                if title_tag and not document.metadata.title:
                    document.metadata.title = title_tag.get_text()
                
                # Process sections if requested
                if options.chunking_strategy == "section":
                    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                    if headings:
                        current_section = ""
                        section_title = ""
                        
                        for heading in headings:
                            # Add previous section as chunk
                            if current_section and section_title:
                                document.add_chunk(
                                    content=current_section,
                                    section=section_title,
                                )
                            
                            # Start new section
                            section_title = heading.get_text()
                            current_section = section_title + "\n\n"
                            
                            # Get content until next heading
                            element = heading.next_sibling
                            while element and element.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                                if element.string and element.string.strip():
                                    current_section += element.string.strip() + "\n\n"
                                element = element.next_sibling
                        
                        # Add last section
                        if current_section and section_title:
                            document.add_chunk(
                                content=current_section,
                                section=section_title,
                            )
                
            elif file_extension in {"html", "htm"}:
                # Parse HTML and extract text
                soup = BeautifulSoup(content, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Extract text
                text = soup.get_text(separator="\n\n")
                
                # Extract title
                title_tag = soup.find("title")
                if title_tag and not document.metadata.title:
                    document.metadata.title = title_tag.get_text()
                
                # Process sections if requested
                if options.chunking_strategy == "section":
                    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                    if headings:
                        for heading in headings:
                            # Get heading text
                            section_title = heading.get_text().strip()
                            
                            # Get content until next heading
                            section_content = section_title + "\n\n"
                            element = heading.next_sibling
                            
                            while element and element.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                                if hasattr(element, "get_text"):
                                    content_text = element.get_text().strip()
                                    if content_text:
                                        section_content += content_text + "\n\n"
                                elif element.string and element.string.strip():
                                    section_content += element.string.strip() + "\n\n"
                                
                                element = element.next_sibling
                            
                            # Add section as chunk
                            if section_content.strip():
                                document.add_chunk(
                                    content=section_content,
                                    section=section_title,
                                )
            else:
                # Plain text
                text = content
            
            # Clean up text
            text = self._clean_text(text)
            
            # Set raw text
            document.raw_text = text
            
            # Estimate word count
            document.metadata.word_count = len(text.split())
            
            return document
            
        except Exception as e:
            logger.error(
                "Text document extraction failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            raise ConversionError(f"Failed to extract text from document: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Replace multiple newlines with double newline
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text