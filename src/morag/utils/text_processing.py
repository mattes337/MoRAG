"""Text processing utilities for MoRAG."""

import re
import unicodedata
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger()

def prepare_text_for_embedding(text: str) -> str:
    """Prepare text for embedding generation."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\']', ' ', text)

    # Trim and ensure reasonable length
    text = text.strip()

    # Truncate if too long (Gemini has token limits)
    max_chars = 30000  # Conservative limit
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text


def clean_pdf_text_encoding(text: str) -> str:
    """Clean PDF text encoding issues and artifacts.

    Args:
        text: Raw text extracted from PDF

    Returns:
        Cleaned text with encoding issues resolved
    """
    if not text:
        return text

    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Common PDF encoding fixes
    replacements = {
        # Zero-width characters (but NOT soft hyphen - handled by regex below)
        '\u200b': '',  # Zero-width space
        '\u200c': '',  # Zero-width non-joiner
        '\u200d': '',  # Zero-width joiner
        '\ufeff': '',  # Byte order mark

        # Common ligature issues
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',

        # Quote marks
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",

        # Dashes
        '–': '-',  # En dash
        '—': '-',  # Em dash

        # Common encoding artifacts
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€¦': '...',
        'â€"': '-',
        'â€"': '-',

        # Note: Soft hyphens are handled by regex patterns below
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Fix hyphenated words with soft hyphens BEFORE cleaning whitespace
    # Handle cases like "ange­ schlagen" -> "angeschlagen"
    # This needs to handle the case where there might be multiple spaces before the soft hyphen
    text = re.sub(r'(\w+)\s*\u00ad\s+(\w+)', r'\1\2', text)  # word + optional space + soft hyphen + space(s) + word
    text = re.sub(r'(\w+)\u00ad\s+(\w+)', r'\1\2', text)     # soft hyphen + space(s) + word
    text = re.sub(r'(\w+)\u00ad(\w+)', r'\1\2', text)        # soft hyphen + word directly

    # Fix broken words that span lines (common in PDFs)
    # Pattern: word- newline word -> wordword
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Remove any remaining soft hyphens that weren't caught by the patterns above
    text = text.replace('\u00ad', '')

    # Clean up multiple spaces (but preserve single newlines)
    text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces and tabs, not newlines

    # Clean up multiple newlines (preserve double newlines for paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def normalize_text_encoding(text: str) -> str:
    """Normalize text encoding for consistent processing.

    Args:
        text: Input text that may have encoding issues

    Returns:
        Normalized text
    """
    if text is None:
        return ""
    if isinstance(text, bytes) and not text:
        return ""
    if not text:
        return text

    try:
        # Try to encode/decode to fix encoding issues
        if isinstance(text, bytes):
            # If it's bytes, try to decode with various encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text = text.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        # Apply PDF-specific cleaning
        text = clean_pdf_text_encoding(text)

        return text

    except Exception as e:
        logger.warning("Text encoding normalization failed", error=str(e))
        return str(text) if text else ""

def prepare_text_for_summary(text: str) -> str:
    """Prepare text for summarization."""
    # Similar cleaning but preserve more structure
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces

    text = text.strip()

    # Truncate if too long
    max_chars = 25000  # Leave room for prompt
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text

def combine_text_and_summary(text: str, summary: str) -> str:
    """Combine original text and summary for embedding generation."""
    # Use CRAG-inspired approach: summary first, then full text
    combined = f"Summary: {summary}\n\nFull Text: {text}"
    return combined

def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract key phrases from text (simple implementation)."""
    # Simple keyword extraction - can be enhanced with NLP libraries
    words = text.lower().split()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Filter words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top phrases
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_phrases]]

def clean_extracted_text(text: str) -> str:
    """Clean text extracted from documents."""
    # Remove excessive line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove page numbers and headers/footers (simple patterns)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page \d+.*$', '', text, flags=re.MULTILINE)
    
    # Remove excessive spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes."""
    word_count = len(text.split())
    return max(1, round(word_count / words_per_minute))

def truncate_text(text: str, max_length: int, preserve_words: bool = True) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    
    if preserve_words:
        # Find the last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Only if we don't lose too much
            return truncated[:last_space] + "..."
    
    return text[:max_length-3] + "..."

def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """Extract basic metadata from text content."""
    lines = text.split('\n')
    
    metadata = {
        'word_count': len(text.split()),
        'character_count': len(text),
        'line_count': len(lines),
        'paragraph_count': len([line for line in lines if line.strip()]),
        'estimated_reading_time': estimate_reading_time(text),
        'has_urls': bool(re.search(r'https?://\S+', text)),
        'has_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'has_phone_numbers': bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
    }
    
    return metadata

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    
    return '\n'.join(lines).strip()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences (simple implementation)."""
    # Simple sentence splitting - can be enhanced with NLP libraries
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def calculate_text_similarity_score(text1: str, text2: str) -> float:
    """Calculate simple text similarity score based on word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0
