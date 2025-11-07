"""Content chunking strategies for different content types."""

import re
from typing import Dict, List, Any

from morag_core.utils.logging import get_logger

logger = get_logger(__name__)


class ContentChunkers:
    """Handles different content chunking strategies."""

    def create_topic_based_chunks(self, content: str, chunk_size: int, chunk_overlap: int, metadata: Dict[str, Any]) -> List[str]:
        """Create chunks based on topic boundaries in audio/video content."""
        # Extract topics from metadata if available
        topics = metadata.get('topics', [])
        if not topics:
            return self._create_timestamp_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        for topic in topics:
            topic_start = topic.get('start_time', 0)
            topic_end = topic.get('end_time', len(content))
            topic_title = topic.get('title', 'Topic')

            # Find content for this topic based on timestamps
            topic_content = self._extract_topic_content(content, topic_start, topic_end)

            if len(topic_content) <= chunk_size:
                # Topic fits in one chunk
                chunks.append(f"## {topic_title}\n\n{topic_content}")
            else:
                # Split topic into multiple chunks
                topic_chunks = self._split_topic_at_timestamps(topic_content, chunk_size, chunk_overlap)
                for i, chunk in enumerate(topic_chunks):
                    if i == 0:
                        chunks.append(f"## {topic_title}\n\n{chunk}")
                    else:
                        chunks.append(f"## {topic_title} (continued)\n\n{chunk}")

        return chunks

    def _extract_topic_content(self, content: str, start_time: float, end_time: float) -> str:
        """Extract content between timestamp markers."""
        lines = content.split('\n')
        topic_lines = []

        for line in lines:
            # Look for timestamp patterns in transcription
            timestamp_match = re.search(r'\[?(\d{1,2}):(\d{2}):(\d{2})\]?', line)
            if timestamp_match:
                hours, minutes, seconds = map(int, timestamp_match.groups())
                line_time = hours * 3600 + minutes * 60 + seconds

                if start_time <= line_time <= end_time:
                    topic_lines.append(line)
            else:
                # Include lines without timestamps if we're in the topic range
                if topic_lines:  # We've started collecting for this topic
                    topic_lines.append(line)

        return '\n'.join(topic_lines)

    def create_timestamp_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on timestamp boundaries in transcribed content."""
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        chunks = []

        for line in lines:
            # Check if this line contains a timestamp
            has_timestamp = re.search(r'\[?(\d{1,2}):(\d{2}):(\d{2})\]?', line)

            # Calculate potential new size
            line_size = len(line) + 1  # +1 for newline

            # If adding this line would exceed chunk size and we have content
            if current_size + line_size > chunk_size and current_chunk:
                # If this line has a timestamp, it's a good breaking point
                if has_timestamp:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    # No timestamp, add to current chunk anyway to avoid losing content
                    current_chunk.append(line)
                    current_size += line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _split_topic_at_timestamps(self, topic_content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a long topic at natural timestamp boundaries."""
        # First try to split at timestamp boundaries
        timestamp_chunks = self.create_timestamp_chunks(topic_content, chunk_size, chunk_overlap)

        # If we only got one chunk back, the topic is still too long
        if len(timestamp_chunks) == 1 and len(timestamp_chunks[0]) > chunk_size:
            # Fall back to paragraph splitting
            return self._create_character_chunks(topic_content, chunk_size, chunk_overlap)

        return timestamp_chunks

    def create_image_section_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks for image content with OCR results."""
        # Look for image section markers
        sections = re.split(r'\n## Image \d+:', content)

        if len(sections) <= 1:
            # No clear image sections, use semantic chunking
            return self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        for i, section in enumerate(sections):
            if i == 0 and not section.strip():
                continue  # Skip empty first section

            section_content = section.strip()
            if not section_content:
                continue

            # Add image header back
            if i > 0:
                section_header = f"## Image {i}:"
                full_section = f"{section_header}\n{section_content}"
            else:
                full_section = section_content

            if len(full_section) <= chunk_size:
                chunks.append(full_section)
            else:
                # Split long sections
                section_chunks = self._process_sections_into_chunks([full_section], chunk_size, chunk_overlap)
                chunks.extend(section_chunks)

        return chunks

    def create_web_article_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks for web article content."""
        # Look for article structure markers
        structure_patterns = [
            r'\n## [A-Z].*?:',  # Section headers like "## Introduction:"
            r'\n### .*?$',      # Subsection headers
            r'\n\*\*[A-Z].*?\*\*',  # Bold section titles
            r'\n[A-Z][A-Z\s]{10,}\n',  # All caps section titles
        ]

        # Find all structural boundaries
        boundaries = []
        for pattern in structure_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                boundaries.append(match.start())

        boundaries = sorted(set(boundaries))

        if not boundaries:
            # No clear structure, use semantic chunking
            return self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)

        # Split content at boundaries
        sections = []
        start = 0
        for boundary in boundaries:
            if start < boundary:
                section = content[start:boundary].strip()
                if section:
                    sections.append(section)
            start = boundary

        # Don't forget the final section
        if start < len(content):
            final_section = content[start:].strip()
            if final_section:
                sections.append(final_section)

        return self._process_sections_into_chunks(sections, chunk_size, chunk_overlap)

    def create_text_semantic_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create semantically meaningful chunks for text content."""
        # Split by double newlines (paragraphs) first
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]

        if not paragraphs:
            return self._create_character_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)

            # If single paragraph is too large
            if paragraph_size > chunk_size:
                # Add current chunk if exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sentences = self._split_into_sentences(paragraph)
                temp_chunk = []
                temp_size = 0

                for sentence in sentences:
                    sentence_size = len(sentence) + 1  # +1 for space

                    if temp_size + sentence_size > chunk_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_size = len(sentence)
                    else:
                        temp_chunk.append(sentence)
                        temp_size += sentence_size

                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                continue

            # Check if adding this paragraph would exceed chunk size
            additional_size = paragraph_size + (2 if current_chunk else 0)  # +2 for \n\n

            if current_size + additional_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += additional_size

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def create_code_structural_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks for code content preserving structure."""
        # Look for code block patterns
        code_blocks = list(re.finditer(r'```[\s\S]*?```', content))

        if not code_blocks:
            # No code blocks found, treat as regular text
            return self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        last_end = 0

        for block in code_blocks:
            block_start = block.start()
            block_end = block.end()
            block_content = block.group()

            # Add text before this code block
            if block_start > last_end:
                text_before = content[last_end:block_start].strip()
                if text_before:
                    if len(text_before) > chunk_size:
                        text_chunks = self._create_text_semantic_chunks(text_before, chunk_size, chunk_overlap)
                        chunks.extend(text_chunks)
                    else:
                        chunks.append(text_before)

            # Handle the code block
            if len(block_content) <= chunk_size:
                chunks.append(block_content)
            else:
                # Split large code blocks by preserving function/class boundaries
                code_chunks = self._split_code_block(block_content, chunk_size, chunk_overlap)
                chunks.extend(code_chunks)

            last_end = block_end

        # Add remaining content after last code block
        if last_end < len(content):
            remaining = content[last_end:].strip()
            if remaining:
                if len(remaining) > chunk_size:
                    remaining_chunks = self._create_text_semantic_chunks(remaining, chunk_size, chunk_overlap)
                    chunks.extend(remaining_chunks)
                else:
                    chunks.append(remaining)

        return chunks

    def _split_code_block(self, code_block: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split large code blocks preserving structure."""
        lines = code_block.split('\n')

        # Find function/class boundaries
        boundaries = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*(def|class|function|var|const|let)\s+\w+', line, re.IGNORECASE):
                boundaries.append(i)

        if not boundaries:
            # No structural boundaries found, split by lines
            return self._create_character_chunks(code_block, chunk_size, chunk_overlap)

        chunks = []
        start = 0

        for boundary in boundaries[1:]:  # Skip first boundary
            section_lines = lines[start:boundary]
            section = '\n'.join(section_lines)

            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Still too large, split by character
                char_chunks = self._create_character_chunks(section, chunk_size, chunk_overlap)
                chunks.extend(char_chunks)

            start = boundary

        # Handle final section
        if start < len(lines):
            final_section = '\n'.join(lines[start:])
            if len(final_section) <= chunk_size:
                chunks.append(final_section)
            else:
                char_chunks = self._create_character_chunks(final_section, chunk_size, chunk_overlap)
                chunks.extend(char_chunks)

        return chunks

    def create_archive_file_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks for archive file listings."""
        # Archive content typically has file listings
        lines = content.split('\n')

        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_size = len(line) + 1  # +1 for newline

            # Check if this looks like a directory separator or file header
            is_separator = any([
                line.startswith('====='),
                line.startswith('-----'),
                line.endswith('/') and len(line.split()) == 1,  # Directory path
                re.match(r'^\w+/.*:', line),  # Path with colon
            ])

            if current_size + line_size > chunk_size and current_chunk:
                # If this is a separator, it's a good place to break
                if is_separator:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    # Not a separator, but we need to break anyway
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def create_document_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks for document content (PDF, Word, etc.)."""
        # Look for document structure
        structure_patterns = [
            r'\n## .*$',           # Level 2 headers
            r'\n### .*$',          # Level 3 headers
            r'\n\d+\.\s+[A-Z]',    # Numbered sections like "1. Introduction"
            r'\nChapter\s+\d+',    # Chapter markers
            r'\nSection\s+\d+',    # Section markers
        ]

        # Find structural boundaries
        boundaries = [0]
        for pattern in structure_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                boundaries.append(match.start())

        boundaries = sorted(set(boundaries))
        boundaries.append(len(content))

        sections = []
        for i in range(len(boundaries) - 1):
            section = content[boundaries[i]:boundaries[i+1]].strip()
            if section:
                sections.append(section)

        if not sections:
            # No structure found, use semantic chunking
            return self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)

        return self._process_sections_into_chunks(sections, chunk_size, chunk_overlap)

    def create_character_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on character count with word boundary preservation."""
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            # Calculate end position
            end = min(start + chunk_size, len(content))

            # Find word boundary if we're not at the end
            if end < len(content):
                # Look backwards for word boundary
                while end > start and content[end] not in ' \n\t.!?;:,-':
                    end -= 1

                # If we couldn't find a good boundary, use the original end
                if end == start:
                    end = min(start + chunk_size, len(content))

            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(end - chunk_overlap, start + 1)

            # Ensure we make progress
            if start >= end:
                start = end

        return chunks

    def _process_sections_into_chunks(self, sections: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Process sections into appropriately sized chunks."""
        chunks = []

        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Section is too large, split it
                section_chunks = self._create_text_semantic_chunks(section, chunk_size, chunk_overlap)
                chunks.extend(section_chunks)

        return chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= overlap_size:
            return text

        # Get the last overlap_size characters
        overlap = text[-overlap_size:]

        # Find the first word boundary to avoid cutting words
        for i, char in enumerate(overlap):
            if char in ' \n\t':
                return overlap[i:].strip()

        # If no word boundary found, return the full overlap
        return overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


__all__ = ["ContentChunkers"]
