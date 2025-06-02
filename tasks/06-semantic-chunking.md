# Task 06: Semantic Chunking Implementation

## Overview
Implement intelligent semantic chunking using spaCy for natural language processing, creating meaningful text segments optimized for retrieval.

## Prerequisites
- Task 01: Project Setup completed
- spaCy language model downloaded

## Dependencies
- Task 01: Project Setup

## Implementation Steps

### 1. Install spaCy Model
```bash
# Download English language model
python -m spacy download en_core_web_sm

# For better performance, optionally download larger model
python -m spacy download en_core_web_md
```

### 2. Enhanced Chunking Service
Update `src/morag/services/chunking.py`:
```python
from typing import List, Dict, Any, Optional, Tuple
import spacy
import structlog
from dataclasses import dataclass
import re
from collections import defaultdict

from morag.core.config import settings
from morag.core.exceptions import ProcessingError

logger = structlog.get_logger()

@dataclass
class ChunkInfo:
    """Information about a text chunk."""
    text: str
    start_char: int
    end_char: int
    sentence_count: int
    word_count: int
    entities: List[Dict[str, Any]]
    topics: List[str]
    chunk_type: str = "semantic"

class SemanticChunker:
    """Advanced semantic chunking using spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
            logger.info("spaCy model loaded", model=model_name)
        except OSError:
            logger.warning(f"spaCy model {model_name} not found, falling back to basic chunking")
            self.nlp = None
        
        self.max_chunk_size = settings.max_chunk_size
        self.min_chunk_size = 100
        self.overlap_size = 50
    
    async def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        strategy: str = "semantic"
    ) -> List[ChunkInfo]:
        """Chunk text using specified strategy."""
        
        chunk_size = chunk_size or self.max_chunk_size
        
        if not text.strip():
            return []
        
        if self.nlp is None or strategy == "simple":
            return await self._simple_chunking(text, chunk_size)
        
        if strategy == "semantic":
            return await self._semantic_chunking(text, chunk_size)
        elif strategy == "sentence":
            return await self._sentence_chunking(text, chunk_size)
        elif strategy == "paragraph":
            return await self._paragraph_chunking(text, chunk_size)
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, using semantic")
            return await self._semantic_chunking(text, chunk_size)
    
    async def _semantic_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Advanced semantic chunking using NLP analysis."""
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Group sentences by semantic similarity and topics
            sentence_groups = self._group_sentences_semantically(doc)
            
            # Create chunks from sentence groups
            chunks = []
            current_chunk_text = ""
            current_start = 0
            current_sentences = []
            
            for group in sentence_groups:
                group_text = " ".join([sent.text for sent in group])
                
                # Check if adding this group would exceed chunk size
                if (len(current_chunk_text) + len(group_text) > chunk_size and 
                    len(current_chunk_text) > self.min_chunk_size):
                    
                    # Finalize current chunk
                    if current_chunk_text:
                        chunk_info = self._create_chunk_info(
                            current_chunk_text,
                            current_start,
                            current_start + len(current_chunk_text),
                            current_sentences
                        )
                        chunks.append(chunk_info)
                    
                    # Start new chunk
                    current_chunk_text = group_text
                    current_start = group[0].start_char
                    current_sentences = group
                else:
                    # Add to current chunk
                    if current_chunk_text:
                        current_chunk_text += " " + group_text
                    else:
                        current_chunk_text = group_text
                        current_start = group[0].start_char
                    current_sentences.extend(group)
            
            # Add final chunk
            if current_chunk_text:
                chunk_info = self._create_chunk_info(
                    current_chunk_text,
                    current_start,
                    current_start + len(current_chunk_text),
                    current_sentences
                )
                chunks.append(chunk_info)
            
            # Add overlap between chunks if needed
            chunks = self._add_overlap(chunks, text)
            
            logger.info(
                "Semantic chunking completed",
                original_length=len(text),
                chunks_created=len(chunks),
                avg_chunk_size=sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
            )
            
            return chunks
            
        except Exception as e:
            logger.error("Semantic chunking failed", error=str(e))
            # Fallback to simple chunking
            return await self._simple_chunking(text, chunk_size)
    
    def _group_sentences_semantically(self, doc) -> List[List]:
        """Group sentences by semantic similarity and topic coherence."""
        
        sentences = list(doc.sents)
        if not sentences:
            return []
        
        groups = []
        current_group = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current_sent = sentences[i]
            prev_sent = sentences[i-1]
            
            # Calculate semantic similarity
            similarity = self._calculate_sentence_similarity(current_sent, prev_sent)
            
            # Check for topic continuity
            topic_continuity = self._check_topic_continuity(current_sent, prev_sent)
            
            # Decide whether to continue current group or start new one
            if similarity > 0.3 or topic_continuity:
                current_group.append(current_sent)
            else:
                groups.append(current_group)
                current_group = [current_sent]
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _calculate_sentence_similarity(self, sent1, sent2) -> float:
        """Calculate semantic similarity between sentences."""
        
        try:
            # Use spaCy's built-in similarity if available
            if sent1.vector.any() and sent2.vector.any():
                return sent1.similarity(sent2)
            
            # Fallback: lexical overlap
            words1 = set(token.lemma_.lower() for token in sent1 if not token.is_stop and token.is_alpha)
            words2 = set(token.lemma_.lower() for token in sent2 if not token.is_stop and token.is_alpha)
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _check_topic_continuity(self, sent1, sent2) -> bool:
        """Check if sentences share similar topics or entities."""
        
        # Check for shared named entities
        entities1 = set(ent.text.lower() for ent in sent1.ents)
        entities2 = set(ent.text.lower() for ent in sent2.ents)
        
        if entities1 and entities2:
            shared_entities = entities1.intersection(entities2)
            if len(shared_entities) > 0:
                return True
        
        # Check for shared noun phrases
        noun_phrases1 = set(chunk.text.lower() for chunk in sent1.noun_chunks)
        noun_phrases2 = set(chunk.text.lower() for chunk in sent2.noun_chunks)
        
        if noun_phrases1 and noun_phrases2:
            shared_phrases = noun_phrases1.intersection(noun_phrases2)
            if len(shared_phrases) > 0:
                return True
        
        return False
    
    async def _sentence_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Chunk text by sentences, respecting size limits."""
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        current_sentences = []
        
        for sent in sentences:
            sent_text = sent.text.strip()
            
            if len(current_chunk) + len(sent_text) > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_info = self._create_chunk_info(
                    current_chunk,
                    current_start,
                    current_start + len(current_chunk),
                    current_sentences
                )
                chunks.append(chunk_info)
                
                # Start new chunk
                current_chunk = sent_text
                current_start = sent.start_char
                current_sentences = [sent]
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sent_text
                else:
                    current_chunk = sent_text
                    current_start = sent.start_char
                current_sentences.append(sent)
        
        # Add final chunk
        if current_chunk:
            chunk_info = self._create_chunk_info(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                current_sentences
            )
            chunks.append(chunk_info)
        
        return chunks
    
    async def _paragraph_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Chunk text by paragraphs, splitting large paragraphs."""
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) <= chunk_size:
                # Paragraph fits in one chunk
                chunk_info = ChunkInfo(
                    text=para,
                    start_char=current_pos,
                    end_char=current_pos + len(para),
                    sentence_count=len(list(self.nlp(para).sents)) if self.nlp else para.count('.'),
                    word_count=len(para.split()),
                    entities=[],
                    topics=[],
                    chunk_type="paragraph"
                )
                chunks.append(chunk_info)
            else:
                # Split large paragraph using sentence chunking
                para_chunks = await self._sentence_chunking(para, chunk_size)
                chunks.extend(para_chunks)
            
            current_pos += len(para) + 2  # +2 for paragraph separator
        
        return chunks
    
    async def _simple_chunking(self, text: str, chunk_size: int) -> List[ChunkInfo]:
        """Simple character-based chunking with word boundaries."""
        
        words = text.split()
        chunks = []
        current_chunk = ""
        current_start = 0
        word_start = 0
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_info = ChunkInfo(
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    sentence_count=current_chunk.count('.'),
                    word_count=len(current_chunk.split()),
                    entities=[],
                    topics=[],
                    chunk_type="simple"
                )
                chunks.append(chunk_info)
                
                # Start new chunk
                current_chunk = word
                current_start = word_start
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
                    current_start = word_start
            
            word_start += len(word) + 1
        
        # Add final chunk
        if current_chunk:
            chunk_info = ChunkInfo(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                sentence_count=current_chunk.count('.'),
                word_count=len(current_chunk.split()),
                entities=[],
                topics=[],
                chunk_type="simple"
            )
            chunks.append(chunk_info)
        
        return chunks
    
    def _create_chunk_info(
        self,
        text: str,
        start_char: int,
        end_char: int,
        sentences: List
    ) -> ChunkInfo:
        """Create ChunkInfo object with extracted metadata."""
        
        # Extract entities
        entities = []
        for sent in sentences:
            for ent in sent.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Extract topics (simplified - could be enhanced with topic modeling)
        topics = []
        noun_phrases = set()
        for sent in sentences:
            for chunk in sent.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep short phrases
                    noun_phrases.add(chunk.text.lower())
        
        topics = list(noun_phrases)[:5]  # Top 5 topics
        
        return ChunkInfo(
            text=text.strip(),
            start_char=start_char,
            end_char=end_char,
            sentence_count=len(sentences),
            word_count=len(text.split()),
            entities=entities,
            topics=topics,
            chunk_type="semantic"
        )
    
    def _add_overlap(self, chunks: List[ChunkInfo], original_text: str) -> List[ChunkInfo]:
        """Add overlap between consecutive chunks."""
        
        if len(chunks) <= 1 or self.overlap_size <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no overlap at beginning
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk.text[-self.overlap_size:]
                
                # Create new chunk with overlap
                new_text = overlap_text + " " + chunk.text
                new_chunk = ChunkInfo(
                    text=new_text,
                    start_char=chunk.start_char - len(overlap_text) - 1,
                    end_char=chunk.end_char,
                    sentence_count=chunk.sentence_count,
                    word_count=len(new_text.split()),
                    entities=chunk.entities,
                    topics=chunk.topics,
                    chunk_type=chunk.chunk_type
                )
                overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks

class ChunkingService:
    """Service for semantic chunking of text."""
    
    def __init__(self):
        self.chunker = SemanticChunker()
        self.max_chunk_size = settings.max_chunk_size
    
    async def semantic_chunk(
        self,
        text: str,
        chunk_size: int = None,
        strategy: str = "semantic"
    ) -> List[str]:
        """Perform semantic chunking and return text chunks."""
        
        chunk_infos = await self.chunker.chunk_text(
            text=text,
            chunk_size=chunk_size,
            strategy=strategy
        )
        
        return [chunk.text for chunk in chunk_infos]
    
    async def chunk_with_metadata(
        self,
        text: str,
        chunk_size: int = None,
        strategy: str = "semantic"
    ) -> List[Dict[str, Any]]:
        """Perform chunking and return chunks with metadata."""
        
        chunk_infos = await self.chunker.chunk_text(
            text=text,
            chunk_size=chunk_size,
            strategy=strategy
        )
        
        return [
            {
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "sentence_count": chunk.sentence_count,
                "word_count": chunk.word_count,
                "entities": chunk.entities,
                "topics": chunk.topics,
                "chunk_type": chunk.chunk_type
            }
            for chunk in chunk_infos
        ]

# Global instance
chunking_service = ChunkingService()
```

## Testing Instructions

### 1. Unit Tests
Create `tests/unit/test_chunking.py`:
```python
import pytest
import asyncio
from morag.services.chunking import chunking_service, SemanticChunker

@pytest.mark.asyncio
async def test_simple_chunking():
    """Test simple text chunking."""
    text = "This is a test. " * 100  # Long text
    
    chunks = await chunking_service.semantic_chunk(
        text=text,
        chunk_size=200,
        strategy="simple"
    )
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 250 for chunk in chunks)  # Allow some flexibility

@pytest.mark.asyncio
async def test_semantic_chunking():
    """Test semantic chunking with spaCy."""
    text = """
    Machine learning is a subset of artificial intelligence. It focuses on algorithms that learn from data.
    
    Deep learning is a type of machine learning. It uses neural networks with multiple layers.
    
    Natural language processing is another AI field. It deals with understanding human language.
    """
    
    chunks = await chunking_service.semantic_chunk(
        text=text,
        chunk_size=100,
        strategy="semantic"
    )
    
    assert len(chunks) >= 2
    assert all(isinstance(chunk, str) for chunk in chunks)

@pytest.mark.asyncio
async def test_chunking_with_metadata():
    """Test chunking with metadata extraction."""
    text = "Apple Inc. is a technology company based in Cupertino, California. Tim Cook is the CEO."
    
    chunks = await chunking_service.chunk_with_metadata(
        text=text,
        strategy="semantic"
    )
    
    assert len(chunks) >= 1
    chunk = chunks[0]
    
    assert "text" in chunk
    assert "entities" in chunk
    assert "word_count" in chunk
    assert chunk["word_count"] > 0

def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = SemanticChunker()
    
    # Should not raise exception
    assert chunker is not None
```

### 2. Integration Test
Create `tests/integration/test_chunking_integration.py`:
```python
import pytest
from morag.services.chunking import chunking_service

@pytest.mark.asyncio
async def test_chunking_strategies():
    """Test different chunking strategies."""
    
    text = """
    The quick brown fox jumps over the lazy dog. This is a simple sentence for testing.
    
    Machine learning algorithms can process large amounts of data. They identify patterns and make predictions.
    
    Natural language processing enables computers to understand human language. It's a fascinating field of study.
    """
    
    strategies = ["simple", "semantic", "sentence", "paragraph"]
    
    for strategy in strategies:
        chunks = await chunking_service.semantic_chunk(
            text=text,
            chunk_size=150,
            strategy=strategy
        )
        
        assert len(chunks) > 0, f"Strategy {strategy} produced no chunks"
        assert all(len(chunk.strip()) > 0 for chunk in chunks), f"Strategy {strategy} produced empty chunks"
        
        print(f"Strategy {strategy}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk[:50]}...")
```

### 3. Manual Testing Script
Create `scripts/test_chunking.py`:
```python
#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.services.chunking import chunking_service

async def main():
    # Test text
    text = """
    Artificial Intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

    Machine Learning (ML) is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions.

    Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.

    Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language in a valuable way.
    """
    
    print("Testing different chunking strategies...")
    
    strategies = ["simple", "semantic", "sentence", "paragraph"]
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} CHUNKING ---")
        
        chunks = await chunking_service.chunk_with_metadata(
            text=text,
            chunk_size=200,
            strategy=strategy
        )
        
        print(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Words: {chunk['word_count']}")
            print(f"  Sentences: {chunk['sentence_count']}")
            print(f"  Entities: {[e['text'] for e in chunk['entities'][:3]]}")
            print(f"  Topics: {chunk['topics'][:3]}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Success Criteria
- [ ] spaCy model loads successfully
- [ ] Semantic chunking creates meaningful segments
- [ ] Different chunking strategies work correctly
- [ ] Chunk metadata is extracted properly
- [ ] Named entities are identified in chunks
- [ ] Topic extraction works for chunks
- [ ] Overlap between chunks is handled correctly
- [ ] Fallback to simple chunking works when spaCy fails
- [ ] Unit and integration tests pass
- [ ] Performance is acceptable for typical document sizes

## Next Steps
- Task 05: Document Parser (uses enhanced chunking)
- Task 07: Summary Generation (works with chunked content)
- Task 15: Vector Storage (stores chunked content with metadata)
