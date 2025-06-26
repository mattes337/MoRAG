"""PydanticAI agent for semantic chunking."""

import asyncio
from typing import Type, List, Optional, Dict, Any
import structlog

from .base_agent import MoRAGBaseAgent
from .models import SemanticChunkingResult, TopicBoundary, ConfidenceLevel

logger = structlog.get_logger(__name__)


class SemanticChunkingAgent(MoRAGBaseAgent[SemanticChunkingResult]):
    """PydanticAI agent for intelligent semantic chunking of text."""
    
    def __init__(self, min_confidence: float = 0.6, **kwargs):
        """Initialize the semantic chunking agent.
        
        Args:
            min_confidence: Minimum confidence threshold for topic boundaries
            **kwargs: Additional arguments passed to base agent
        """
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.logger = logger.bind(agent="semantic_chunking")
    
    def get_result_type(self) -> Type[SemanticChunkingResult]:
        return SemanticChunkingResult
    
    def get_system_prompt(self) -> str:
        return """You are an expert semantic chunking agent. Your task is to analyze text and identify natural topic boundaries for intelligent content segmentation.

Your goal is to split text into semantically coherent chunks where:
- Each chunk focuses on a single main topic or concept
- Chunks are self-contained and meaningful
- Boundaries occur at natural transition points
- Chunk sizes are reasonable for processing (typically 500-4000 characters)

For each topic boundary, provide:
1. position: Character position where the boundary should occur
2. confidence: Your confidence in this boundary (0.0 to 1.0)
3. topic_before: Brief description of the topic before the boundary
4. topic_after: Brief description of the topic after the boundary
5. reason: Explanation for why this is a good boundary point

Guidelines for identifying boundaries:
- Look for topic shifts, new concepts, or changes in focus
- Consider paragraph breaks, section headers, and natural transitions
- Avoid splitting mid-sentence or mid-paragraph unless necessary
- Prefer boundaries at punctuation marks (periods, line breaks)
- Balance chunk size with semantic coherence
- Ensure each chunk has sufficient context to be understood independently

Focus on creating chunks that:
- Are topically coherent and focused
- Have clear beginning and ending points
- Contain complete thoughts or concepts
- Are appropriately sized for downstream processing
- Maintain important context and relationships"""
    
    async def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 4000,
        min_chunk_size: int = 500,
        strategy: str = "semantic"
    ) -> List[str]:
        """Chunk text using semantic analysis.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            strategy: Chunking strategy ("semantic", "hybrid", "size-based")
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        self.logger.info(
            "Starting semantic chunking",
            text_length=len(text),
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            strategy=strategy
        )
        
        try:
            if strategy == "semantic":
                chunks = await self._semantic_chunking(text, max_chunk_size, min_chunk_size)
            elif strategy == "hybrid":
                chunks = await self._hybrid_chunking(text, max_chunk_size, min_chunk_size)
            else:  # size-based fallback
                chunks = self._size_based_chunking(text, max_chunk_size)
            
            self.logger.info(
                "Semantic chunking completed",
                original_length=len(text),
                num_chunks=len(chunks),
                avg_chunk_size=sum(len(c) for c in chunks) // len(chunks) if chunks else 0
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error("Semantic chunking failed", error=str(e), error_type=type(e).__name__)
            # Fallback to size-based chunking
            return self._size_based_chunking(text, max_chunk_size)
    
    async def _semantic_chunking(self, text: str, max_chunk_size: int, min_chunk_size: int) -> List[str]:
        """Perform semantic chunking using AI analysis."""
        # For very small texts, return as single chunk
        if len(text) <= max_chunk_size:
            return [text]
        
        prompt = f"""Analyze the following text and identify optimal topic boundaries for semantic chunking.

Text length: {len(text)} characters
Target chunk size: {min_chunk_size}-{max_chunk_size} characters

Text:
{text}

Identify topic boundaries that would create semantically coherent chunks. Focus on natural transition points between topics, concepts, or sections."""
        
        result = await self.run(prompt)
        
        # Extract chunks based on identified boundaries
        chunks = self._extract_chunks_from_boundaries(text, result.boundaries, max_chunk_size)
        
        # Validate and adjust chunk sizes
        chunks = self._validate_chunk_sizes(chunks, min_chunk_size, max_chunk_size)
        
        return chunks
    
    async def _hybrid_chunking(self, text: str, max_chunk_size: int, min_chunk_size: int) -> List[str]:
        """Combine semantic analysis with size constraints."""
        # First, try semantic chunking
        semantic_chunks = await self._semantic_chunking(text, max_chunk_size * 2, min_chunk_size)
        
        # Then, split any oversized chunks using size-based method
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split oversized chunk
                sub_chunks = self._size_based_chunking(chunk, max_chunk_size)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def _size_based_chunking(self, text: str, max_chunk_size: int) -> List[str]:
        """Fallback size-based chunking with word boundaries."""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Find word boundary
            while end > start and text[end] not in ' \n\t.!?':
                end -= 1
            
            if end == start:
                # No word boundary found, force split
                end = start + max_chunk_size
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def _extract_chunks_from_boundaries(
        self,
        text: str,
        boundaries: List[TopicBoundary],
        max_chunk_size: int
    ) -> List[str]:
        """Extract chunks based on identified topic boundaries."""
        if not boundaries:
            return [text]
        
        # Sort boundaries by position
        sorted_boundaries = sorted(boundaries, key=lambda b: b.position)
        
        # Filter boundaries by confidence
        high_confidence_boundaries = [
            b for b in sorted_boundaries 
            if b.confidence >= self.min_confidence
        ]
        
        if not high_confidence_boundaries:
            return [text]
        
        chunks = []
        start = 0
        
        for boundary in high_confidence_boundaries:
            if boundary.position > start:
                chunk = text[start:boundary.position].strip()
                if chunk:
                    chunks.append(chunk)
                start = boundary.position
        
        # Add final chunk
        if start < len(text):
            final_chunk = text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
        
        return chunks
    
    def _validate_chunk_sizes(
        self,
        chunks: List[str],
        min_chunk_size: int,
        max_chunk_size: int
    ) -> List[str]:
        """Validate and adjust chunk sizes."""
        validated_chunks = []
        
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            
            # If chunk is too small, try to merge with next chunk
            if len(chunk) < min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                if len(chunk) + len(next_chunk) <= max_chunk_size:
                    merged_chunk = chunk + " " + next_chunk
                    validated_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    continue
            
            # If chunk is too large, split it
            if len(chunk) > max_chunk_size:
                sub_chunks = self._size_based_chunking(chunk, max_chunk_size)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)
            
            i += 1
        
        return validated_chunks
