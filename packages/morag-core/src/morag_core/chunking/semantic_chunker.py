"""Semantic chunker implementation using PydanticAI."""

import asyncio
from typing import List, Optional, Dict, Any
import structlog

from .config import ChunkingConfig, ChunkingStrategy
try:
    from agents import get_agent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class SemanticChunker:
    """Universal semantic chunker for all content types."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the semantic chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self.config.validate_config()
        
        # Initialize AI agent if semantic analysis is enabled
        if self.config.use_ai_analysis and AGENTS_AVAILABLE:
            self.agent = get_agent("semantic_chunking")
        else:
            self.agent = None
        
        self.logger = logger.bind(component="semantic_chunker")
    
    async def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text according to the configured strategy.
        
        Args:
            text: Text to chunk
            **kwargs: Additional parameters that override config
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Override config with kwargs
        effective_config = self._merge_config_with_kwargs(**kwargs)
        
        self.logger.info(
            "Starting text chunking",
            text_length=len(text),
            strategy=effective_config.strategy,
            max_chunk_size=effective_config.max_chunk_size,
            content_type=effective_config.content_type
        )
        
        try:
            if effective_config.strategy == ChunkingStrategy.SEMANTIC and self.agent:
                chunks = await self.agent.chunk_text(
                    text=text,
                    max_chunk_size=effective_config.max_chunk_size,
                    min_chunk_size=effective_config.min_chunk_size,
                    strategy="semantic"
                )
            elif effective_config.strategy == ChunkingStrategy.HYBRID and self.agent:
                chunks = await self.agent.chunk_text(
                    text=text,
                    max_chunk_size=effective_config.max_chunk_size,
                    min_chunk_size=effective_config.min_chunk_size,
                    strategy="hybrid"
                )
            elif effective_config.strategy == ChunkingStrategy.TOPIC_BASED:
                chunks = await self._topic_based_chunking(text, effective_config)
            elif effective_config.strategy == ChunkingStrategy.SENTENCE_BASED:
                chunks = self._sentence_based_chunking(text, effective_config)
            elif effective_config.strategy == ChunkingStrategy.PARAGRAPH_BASED:
                chunks = self._paragraph_based_chunking(text, effective_config)
            else:  # SIZE_BASED or fallback
                chunks = self._size_based_chunking(text, effective_config)
            
            # Apply overlap if configured
            if effective_config.overlap_size > 0:
                chunks = self._apply_overlap(chunks, effective_config.overlap_size)
            
            self.logger.info(
                "Text chunking completed",
                original_length=len(text),
                num_chunks=len(chunks),
                avg_chunk_size=sum(len(c) for c in chunks) // len(chunks) if chunks else 0
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error("Text chunking failed", error=str(e), error_type=type(e).__name__)
            # Fallback to size-based chunking
            return self._size_based_chunking(text, effective_config)
    
    async def _topic_based_chunking(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text based on topic boundaries."""
        if self.agent:
            # Use AI agent for topic detection
            return await self.agent.chunk_text(
                text=text,
                max_chunk_size=config.max_chunk_size,
                min_chunk_size=config.min_chunk_size,
                strategy="semantic"
            )
        else:
            # Fallback to paragraph-based chunking
            return self._paragraph_based_chunking(text, config)
    
    def _sentence_based_chunking(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text at sentence boundaries."""
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            if current_chunk and len(current_chunk) + len(sentence) + 1 > config.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return self._validate_chunk_sizes(chunks, config)
    
    def _paragraph_based_chunking(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text at paragraph boundaries."""
        # Split on paragraph boundaries (double newlines)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed max size
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > config.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return self._validate_chunk_sizes(chunks, config)
    
    def _size_based_chunking(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text based on size with word boundaries."""
        if len(text) <= config.max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + config.max_chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Find word boundary if configured
            if config.respect_sentence_boundaries:
                # Try to find sentence boundary
                while end > start and text[end] not in '.!?':
                    end -= 1
                if end > start:
                    end += 1  # Include the punctuation
                else:
                    end = start + config.max_chunk_size
            
            if config.respect_sentence_boundaries and end == start + config.max_chunk_size:
                # No sentence boundary found, try word boundary
                while end > start and text[end] not in ' \n\t':
                    end -= 1
                
                if end == start:
                    # No word boundary found, force split
                    end = start + config.max_chunk_size
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def _validate_chunk_sizes(self, chunks: List[str], config: ChunkingConfig) -> List[str]:
        """Validate and adjust chunk sizes."""
        validated_chunks = []

        i = 0
        while i < len(chunks):
            chunk = chunks[i].strip()

            # Skip empty chunks
            if not chunk:
                i += 1
                continue

            # If chunk is too small, try to merge with next chunk
            if len(chunk) < config.min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1].strip()
                if next_chunk and len(chunk) + len(next_chunk) + 1 <= config.max_chunk_size:
                    merged_chunk = chunk + " " + next_chunk
                    validated_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    continue
                else:
                    # Can't merge, but chunk is too small - only keep if it's substantial
                    if len(chunk) >= 50:  # Minimum 50 characters to be meaningful
                        validated_chunks.append(chunk)

            # If chunk is too large, split it carefully
            elif len(chunk) > config.max_chunk_size:
                # Avoid recursive splitting that creates tiny chunks
                sub_chunks = self._safe_split_chunk(chunk, config)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)

            i += 1

        return validated_chunks

    def _safe_split_chunk(self, chunk: str, config: ChunkingConfig) -> List[str]:
        """Safely split a large chunk without creating tiny fragments."""
        if len(chunk) <= config.max_chunk_size:
            return [chunk]

        chunks = []
        start = 0

        while start < len(chunk):
            end = min(start + config.max_chunk_size, len(chunk))

            # If this would be the last chunk and it's very small, merge with previous
            remaining = len(chunk) - end
            if remaining > 0 and remaining < config.min_chunk_size:
                end = len(chunk)  # Take the rest

            # Find a good break point (sentence or word boundary)
            if end < len(chunk):
                # Look for sentence boundary first
                for punct in ['. ', '! ', '? ']:
                    last_punct = chunk.rfind(punct, start, end)
                    if last_punct > start + config.min_chunk_size:
                        end = last_punct + 2
                        break
                else:
                    # Look for word boundary
                    last_space = chunk.rfind(' ', start + config.min_chunk_size, end)
                    if last_space > start:
                        end = last_space

            chunk_text = chunk[start:end].strip()
            if chunk_text and len(chunk_text) >= 50:  # Only keep substantial chunks
                chunks.append(chunk_text)

            start = end
            if start >= len(chunk):
                break

        return chunks
    
    def _apply_overlap(self, chunks: List[str], overlap_size: int) -> List[str]:
        """Apply overlap between chunks."""
        if len(chunks) <= 1 or overlap_size <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no prefix overlap
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk - but limit to prevent exponential growth
                prev_chunk = chunks[i - 1]
                if len(prev_chunk) > overlap_size:
                    # Find word boundary for clean overlap
                    overlap_text = prev_chunk[-overlap_size:]
                    # Find the start of the first complete word in overlap
                    first_space = overlap_text.find(' ')
                    if first_space > 0:
                        overlap_text = overlap_text[first_space + 1:]

                    # Only add overlap if it doesn't make chunk too large
                    potential_chunk = overlap_text + " " + chunk
                    if len(potential_chunk) <= self.config.max_chunk_size * 1.2:  # Allow 20% over
                        overlapped_chunks.append(potential_chunk)
                    else:
                        # Skip overlap if it would make chunk too large
                        overlapped_chunks.append(chunk)
                else:
                    # Previous chunk is small, just add current chunk without overlap
                    overlapped_chunks.append(chunk)

        return overlapped_chunks
    
    def _merge_config_with_kwargs(self, **kwargs) -> ChunkingConfig:
        """Merge configuration with keyword arguments."""
        config_dict = self.config.to_dict()
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            if key in config_dict:
                config_dict[key] = value
        
        return ChunkingConfig.from_dict(config_dict)
