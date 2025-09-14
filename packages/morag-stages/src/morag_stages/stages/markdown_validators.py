"""Markdown conversion validation utilities."""

import re
from typing import Union
from pathlib import Path

# Import error handling decorator
from ..error_handling import standalone_validation_handler

# Import URL utilities if available
try:
    from morag.utils.url_path import URLPath
    URL_PATH_AVAILABLE = True
except ImportError:
    URL_PATH_AVAILABLE = False
    class URLPath:  # type: ignore
        pass


@standalone_validation_handler("validate_conversion_quality")
def validate_conversion_quality(content: str, file_path: Union[Path, 'URLPath']) -> bool:
    """Validate that conversion produced proper markdown for any supported file type.

    Args:
        content: Converted content
        file_path: Original file path

    Returns:
        True if conversion appears successful, False otherwise
    """
    if not content or len(content.strip()) < 5:
        return False

    # Handle URLPath objects which don't have suffix attribute
    if URL_PATH_AVAILABLE and hasattr(file_path, 'url_str'):
        # URLPath object - determine validation based on URL content
        url_str = file_path.url_str.lower()
        if 'youtube.com' in url_str or 'youtu.be' in url_str:
            return validate_media_conversion(content)
        else:
            return validate_html_conversion(content)  # Most web content
    else:
        file_ext = file_path.suffix.lower()

        # Format-specific validation
        if file_ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx']:
            return validate_document_conversion(content)
        elif file_ext in ['.html', '.htm']:
            return validate_html_conversion(content)
        elif file_ext in ['.csv', '.xlsx', '.xls']:
            return validate_data_conversion(content)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            return validate_image_conversion(content)
        elif file_ext in ['.json', '.xml']:
            return validate_structured_data_conversion(content)
        elif file_ext in ['.txt', '.md', '.rst']:
            return validate_text_conversion(content)
        elif file_ext in ['.mp3', '.wav', '.mp4', '.avi', '.mov', '.wmv', '.flv']:
            return validate_media_conversion(content)
        else:
            return validate_general_conversion(content)


@standalone_validation_handler("validate_document_conversion")
def validate_document_conversion(content: str) -> bool:
    """Validate document file conversion (PDF, DOC, PPT, etc.)."""

    # Check for markdown structure
    markdown_patterns = [
        r'^#+ ',  # Headers
        r'^\* ',  # Bullet points
        r'^\d+\. ',  # Numbered lists
        r'\*\*.*?\*\*',  # Bold text
        r'`.*?`',  # Code
        r'^\|.*\|',  # Tables
    ]

    pattern_matches = sum(1 for pattern in markdown_patterns
                        if re.search(pattern, content, re.MULTILINE))

    # Check for signs of poor conversion
    lines = content.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    if not non_empty_lines:
        return False

    # Check for very long lines (sign of poor text extraction)
    long_lines = sum(1 for line in non_empty_lines if len(line) > 200)
    long_line_ratio = long_lines / len(non_empty_lines) if non_empty_lines else 0

    # Check line break density
    line_break_ratio = content.count('\n') / max(1, len(content))

    # Document should have some structure and reasonable formatting
    has_structure = pattern_matches >= 1 or '\n\n' in content
    has_reasonable_formatting = long_line_ratio < 0.7 and line_break_ratio > 0.005

    return has_structure and has_reasonable_formatting


@standalone_validation_handler("validate_html_conversion")
def validate_html_conversion(content: str) -> bool:
    """Validate HTML conversion."""

    # HTML should convert to structured markdown
    has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))
    has_paragraphs = '\n\n' in content
    has_lists = bool(re.search(r'^\* |^\d+\. ', content, re.MULTILINE))

    # Should not contain excessive raw HTML tags
    html_tag_ratio = len(re.findall(r'<[^>]+>', content)) / max(1, len(content.split()))

    return (has_headers or has_paragraphs or has_lists) and html_tag_ratio < 0.1


def validate_data_conversion(content: str) -> bool:
    """Validate data file conversion (CSV, Excel)."""

    # Data files should convert to tables or structured content
    has_tables = bool(re.search(r'^\|.*\|', content, re.MULTILINE))
    has_structured_data = bool(re.search(r'^\w+:\s+\w+', content, re.MULTILINE))
    has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))

    # Should not be just a wall of text
    lines = content.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    avg_line_length = sum(len(line) for line in non_empty_lines) / max(1, len(non_empty_lines))

    return (has_tables or has_structured_data or has_headers) and avg_line_length < 300


def validate_image_conversion(content: str) -> bool:
    """Validate image conversion (OCR results)."""
    # Images should produce meaningful text content
    if len(content.strip()) < 3:
        return False

    # Should not be just error messages
    error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported', 'no text found']
    content_lower = content.lower()

    return not any(indicator in content_lower for indicator in error_indicators)


def validate_structured_data_conversion(content: str) -> bool:
    """Validate structured data conversion (JSON, XML)."""

    # Should convert to readable markdown format
    has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))
    has_structure = bool(re.search(r'^\w+:\s+', content, re.MULTILINE))
    has_code_blocks = '```' in content

    return has_headers or has_structure or has_code_blocks


def validate_text_conversion(content: str) -> bool:
    """Validate text file conversion."""
    # Text files should preserve content reasonably
    return len(content.strip()) > 0


@standalone_validation_handler("validate_media_conversion")
def validate_media_conversion(content: str) -> bool:
    """Validate media file conversion (audio/video transcription)."""
    # Media files should produce transcribed text
    if len(content.strip()) < 10:
        return False

    # Should not be just error messages
    error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported', 'no audio', 'no video']
    content_lower = content.lower()

    return not any(indicator in content_lower for indicator in error_indicators)


def validate_general_conversion(content: str) -> bool:
    """General validation for unknown file types."""
    # Basic check - should have some content and not be obviously broken
    if len(content.strip()) < 5:
        return False

    # Should not be just error messages
    error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported']
    content_lower = content.lower()

    return not any(indicator in content_lower for indicator in error_indicators)