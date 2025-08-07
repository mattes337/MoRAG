"""Document formatter for LLM documentation compliance."""

import re
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class DocumentFormatter:
    """Formats document content according to LLM documentation specifications."""

    def format_document_content(
        self, 
        raw_content: str, 
        file_path: Path, 
        metadata: Dict[str, Any]
    ) -> str:
        """Format document content according to LLM documentation format.

        Args:
            raw_content: Raw markdown content from markitdown
            file_path: Path to the original file
            metadata: Document metadata

        Returns:
            Formatted markdown content following LLM specifications
        """
        filename = file_path.name
        
        # Build formatted content
        formatted_parts = []
        
        # Header Section: "Document: filename.ext"
        formatted_parts.append(f"# Document: {filename}")
        formatted_parts.append("")
        
        # Document Information Section
        formatted_parts.append("## Document Information")
        
        # Extract metadata for information section
        page_count = metadata.get('page_count', 'Unknown')
        word_count = metadata.get('word_count', 'Unknown')
        doc_type = metadata.get('document_type', file_path.suffix.lstrip('.').upper())
        language = metadata.get('language', 'Unknown')
        file_size = metadata.get('file_size', 'Unknown')
        
        # Format file size if it's a number
        if isinstance(file_size, (int, float)):
            if file_size < 1024:
                file_size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                file_size_str = f"{file_size / 1024:.1f} KB"
            else:
                file_size_str = f"{file_size / (1024 * 1024):.1f} MB"
        else:
            file_size_str = str(file_size)
        
        formatted_parts.extend([
            f"- **Page Count**: {page_count}",
            f"- **Word Count**: {word_count}",
            f"- **Document Type**: {doc_type}",
            f"- **Language**: {language}",
            f"- **File Size**: {file_size_str}",
            ""
        ])
        
        # Content Structure: Preserve original document structure
        content_section = self._format_content_structure(raw_content)
        formatted_parts.append(content_section)
        
        return "\n".join(formatted_parts)
    
    def _format_content_structure(self, raw_content: str) -> str:
        """Format the content structure preserving document hierarchy.

        Args:
            raw_content: Raw markdown content

        Returns:
            Formatted content with proper hierarchy
        """
        if not raw_content.strip():
            return "## Content\n\nNo content available."
        
        # Clean up the raw content
        lines = raw_content.split('\n')
        formatted_lines = []
        
        # Track if we need to add a content header
        has_headers = any(line.strip().startswith('#') for line in lines)
        
        if not has_headers:
            # No headers found, add a general content header
            formatted_lines.append("## Content")
            formatted_lines.append("")
        
        # Process each line to ensure proper formatting
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines at the beginning
            if not stripped_line and not formatted_lines:
                continue
            
            # Ensure proper heading hierarchy
            if stripped_line.startswith('#'):
                # Adjust heading levels to start from ## (since # is used for document title)
                heading_level = len(stripped_line) - len(stripped_line.lstrip('#'))
                heading_text = stripped_line.lstrip('#').strip()

                # Increment all heading levels by 1 (# becomes ##, ## becomes ###, etc.)
                adjusted_level = heading_level + 1
                formatted_lines.append(f"{'#' * adjusted_level} {heading_text}")
            else:
                # Regular content line
                formatted_lines.append(line)
        
        # Clean up multiple consecutive empty lines
        cleaned_lines = []
        prev_empty = False
        
        for line in formatted_lines:
            is_empty = not line.strip()
            
            if is_empty and prev_empty:
                continue  # Skip multiple consecutive empty lines
            
            cleaned_lines.append(line)
            prev_empty = is_empty
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def extract_metadata_from_content(self, content: str) -> Dict[str, Any]:
        """Extract metadata from document content.

        Args:
            content: Document content

        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {}
        
        # Count words (approximate)
        word_count = len(re.findall(r'\b\w+\b', content))
        metadata['word_count'] = word_count
        
        # Count pages (estimate based on content length)
        # Rough estimate: 250 words per page
        estimated_pages = max(1, word_count // 250)
        metadata['page_count'] = estimated_pages
        
        # Detect language (basic detection)
        try:
            from langdetect import detect
            detected_lang = detect(content[:1000])  # Use first 1000 chars for detection
            metadata['language'] = detected_lang.upper()
        except:
            metadata['language'] = 'Unknown'
        
        # Extract chapter/section information
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        metadata['sections'] = headers
        metadata['section_count'] = len(headers)
        
        return metadata
    
    def should_preserve_structure(self, content: str) -> bool:
        """Determine if the document structure should be preserved.

        Args:
            content: Document content

        Returns:
            True if structure should be preserved, False otherwise
        """
        # Check for structured content indicators
        has_headers = bool(re.search(r'^#+\s+', content, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*+]\s+', content, re.MULTILINE))
        has_numbered_lists = bool(re.search(r'^\s*\d+\.\s+', content, re.MULTILINE))
        has_tables = bool(re.search(r'\|.*\|', content))
        
        # If content has structural elements, preserve them
        return has_headers or has_lists or has_numbered_lists or has_tables
    
    def clean_markitdown_artifacts(self, content: str) -> str:
        """Clean up common markitdown artifacts and formatting issues.

        Args:
            content: Raw markitdown content

        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix common formatting issues
        content = re.sub(r'^\s*\n', '', content)  # Remove leading empty lines
        content = re.sub(r'\n\s*$', '', content)  # Remove trailing whitespace
        
        # Fix header spacing
        content = re.sub(r'^(#+\s*.+)\n([^\n#])', r'\1\n\n\2', content, flags=re.MULTILINE)
        
        # Ensure proper list formatting
        content = re.sub(r'^(\s*[-*+]\s+)', r'\1', content, flags=re.MULTILINE)
        
        return content.strip()
