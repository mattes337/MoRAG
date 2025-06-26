# Task 3: Universal Semantic Chunking Implementation

## Background

Semantic chunking represents a significant advancement over traditional rule-based chunking strategies. Instead of splitting content based on arbitrary character counts or simple patterns, semantic chunking uses AI to identify natural boundaries in content based on meaning, context, and structure.

### Current Chunking Limitations

1. **Documents**: Fixed-size chunks that may split related concepts
2. **Audio**: Time-based chunks that ignore topic boundaries
3. **Video**: Simple time segments without considering content flow
4. **Web**: HTML structure-based chunks without semantic understanding

### Semantic Chunking Benefits

1. **Coherent Chunks**: Each chunk contains semantically related content
2. **Better Retrieval**: More relevant search results due to coherent chunks
3. **Improved Context**: Chunks maintain logical flow and context
4. **Content-Aware**: Adapts to different content types and structures

## Implementation Strategy

### Phase 1: Document Semantic Chunking (2 days)

#### 1.1 Semantic Boundary Detection Agent
**File**: `packages/morag-document/src/morag_document/ai/semantic_chunking_agent.py`

```python
from pydantic_ai import Agent
from morag_core.ai.base_agent import MoRAGBaseAgent
from pydantic import BaseModel, Field
from typing import List, Optional

class SemanticBoundary(BaseModel):
    """Represents a semantic boundary in text."""
    position: int = Field(description="Character position of boundary")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in boundary")
    reason: str = Field(description="Reason for boundary placement")
    boundary_type: str = Field(description="Type of boundary (topic, section, etc.)")

class SemanticChunkingResult(BaseModel):
    """Result of semantic chunking analysis."""
    boundaries: List[SemanticBoundary]
    suggested_chunks: List[str]
    metadata: dict = Field(default_factory=dict)

class SemanticChunkingAgent(MoRAGBaseAgent[SemanticChunkingResult]):
    """PydanticAI agent for semantic boundary detection."""
    
    def get_result_type(self) -> type[SemanticChunkingResult]:
        return SemanticChunkingResult
    
    def get_system_prompt(self) -> str:
        return """You are an expert text analysis system that identifies semantic boundaries in documents.

        Your task is to find natural breaking points where the content shifts topics, themes, or concepts.
        
        Consider these factors for boundary placement:
        - Topic changes and transitions
        - Conceptual shifts
        - Narrative flow breaks
        - Section boundaries
        - Logical argument structure
        - Paragraph groupings that form coherent units
        
        For each boundary, provide:
        - position: Character position where the boundary should be placed
        - confidence: How confident you are (0.0-1.0)
        - reason: Brief explanation for the boundary
        - boundary_type: Type of semantic shift (topic_change, section_break, etc.)
        """
    
    async def find_boundaries(
        self, 
        text: str, 
        target_chunk_size: int = 4000,
        max_chunk_size: int = 8000
    ) -> SemanticChunkingResult:
        """Find semantic boundaries in text."""
        
        user_prompt = f"""Analyze this text and identify semantic boundaries:

        Text length: {len(text)} characters
        Target chunk size: {target_chunk_size} characters
        Maximum chunk size: {max_chunk_size} characters

        Text:
        {text}

        Find boundaries that create chunks roughly {target_chunk_size} characters long,
        but never exceed {max_chunk_size} characters per chunk.
        Prioritize semantic coherence over exact size matching.
        """
        
        return await self.run(user_prompt)
```

#### 1.2 Semantic Chunking Strategy
**File**: `packages/morag-document/src/morag_document/chunking/semantic_strategy.py`

```python
from morag_core.interfaces.converter import ChunkingStrategy
from morag_document.ai.semantic_chunking_agent import SemanticChunkingAgent
from morag_core.models.document import Document, DocumentChunk
from typing import List, Optional
import structlog

logger = structlog.get_logger(__name__)

class SemanticChunkingStrategy:
    """Semantic chunking strategy using AI boundary detection."""
    
    def __init__(self, 
                 target_chunk_size: int = 4000,
                 max_chunk_size: int = 8000,
                 min_chunk_size: int = 500,
                 fallback_strategy: str = "paragraph"):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.fallback_strategy = fallback_strategy
        self.agent = SemanticChunkingAgent()
    
    async def chunk_document(self, document: Document) -> Document:
        """Apply semantic chunking to document."""
        
        if not document.raw_text:
            logger.warning("No text content for semantic chunking")
            return document
        
        try:
            # Use AI to find semantic boundaries
            result = await self.agent.find_boundaries(
                text=document.raw_text,
                target_chunk_size=self.target_chunk_size,
                max_chunk_size=self.max_chunk_size
            )
            
            # Create chunks based on boundaries
            chunks = self._create_chunks_from_boundaries(
                document.raw_text, 
                result.boundaries
            )
            
            # Add chunks to document
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    document.add_chunk(
                        content=chunk_text,
                        metadata={
                            "chunking_strategy": "semantic",
                            "chunk_method": "ai_boundary_detection",
                            "chunk_index": i,
                            "semantic_confidence": self._calculate_chunk_confidence(
                                result.boundaries, i
                            )
                        }
                    )
            
            logger.info(
                "Semantic chunking completed",
                chunks_created=len(document.chunks),
                boundaries_found=len(result.boundaries)
            )
            
        except Exception as e:
            logger.error("Semantic chunking failed, using fallback", error=str(e))
            # Fallback to simple paragraph-based chunking
            await self._apply_fallback_chunking(document)
        
        return document
    
    def _create_chunks_from_boundaries(self, text: str, boundaries: List) -> List[str]:
        """Create text chunks based on semantic boundaries."""
        if not boundaries:
            return [text]
        
        chunks = []
        start_pos = 0
        
        # Sort boundaries by position
        sorted_boundaries = sorted(boundaries, key=lambda b: b.position)
        
        for boundary in sorted_boundaries:
            chunk_text = text[start_pos:boundary.position].strip()
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            start_pos = boundary.position
        
        # Add final chunk
        final_chunk = text[start_pos:].strip()
        if final_chunk and len(final_chunk) >= self.min_chunk_size:
            chunks.append(final_chunk)
        
        return chunks
    
    async def _apply_fallback_chunking(self, document: Document):
        """Apply simple fallback chunking strategy."""
        # Simple paragraph-based fallback - NO dependency on old code
        paragraphs = document.raw_text.split('\n\n')

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) >= self.min_chunk_size:
                document.add_chunk(
                    content=paragraph.strip(),
                    metadata={
                        "chunking_strategy": "fallback_paragraph",
                        "chunk_method": "simple_paragraph_split",
                        "chunk_index": i,
                        "semantic_confidence": 0.0
                    }
                )
```

### Phase 2: Audio Semantic Chunking (2 days)

#### 2.1 Audio Topic Boundary Detection
**File**: `packages/morag-audio/src/morag_audio/ai/topic_chunking_agent.py`

```python
from pydantic_ai import Agent
from morag_core.ai.base_agent import MoRAGBaseAgent
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TopicBoundary(BaseModel):
    """Represents a topic boundary in audio transcript."""
    timestamp: float = Field(description="Timestamp in seconds")
    confidence: float = Field(ge=0.0, le=1.0)
    topic_before: str = Field(description="Topic before boundary")
    topic_after: str = Field(description="Topic after boundary")
    speaker_change: bool = Field(default=False, description="Whether speaker changes")

class AudioTopicChunkingResult(BaseModel):
    """Result of audio topic chunking."""
    boundaries: List[TopicBoundary]
    topics: List[str]
    chunks: List[Dict[str, Any]]

class AudioTopicChunkingAgent(MoRAGBaseAgent[AudioTopicChunkingResult]):
    """Agent for detecting topic boundaries in audio transcripts."""
    
    def get_result_type(self) -> type[AudioTopicChunkingResult]:
        return AudioTopicChunkingResult
    
    def get_system_prompt(self) -> str:
        return """You are an expert at analyzing audio transcripts and identifying topic boundaries.

        Analyze the transcript and identify where topics change, considering:
        - Subject matter shifts
        - Speaker changes and dialogue flow
        - Conceptual transitions
        - Natural conversation breaks
        - Question-answer patterns
        
        For each boundary, identify:
        - timestamp: When the topic change occurs
        - confidence: How confident you are in this boundary
        - topic_before: Brief description of the previous topic
        - topic_after: Brief description of the new topic
        - speaker_change: Whether this coincides with a speaker change
        """
    
    async def find_topic_boundaries(
        self, 
        transcript: str,
        speaker_info: Optional[List[Dict]] = None,
        min_chunk_duration: float = 30.0
    ) -> AudioTopicChunkingResult:
        """Find topic boundaries in audio transcript."""
        
        speaker_context = ""
        if speaker_info:
            speaker_context = f"\nSpeaker information: {speaker_info}"
        
        user_prompt = f"""Analyze this audio transcript for topic boundaries:

        Minimum chunk duration: {min_chunk_duration} seconds
        {speaker_context}

        Transcript:
        {transcript}

        Identify topic boundaries that create semantically coherent segments,
        each at least {min_chunk_duration} seconds long.
        """
        
        return await self.run(user_prompt)
```

### Phase 3: Configuration System (1 day)

#### 3.1 Universal Chunking Configuration
**File**: `packages/morag-core/src/morag_core/config/chunking_config.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum

class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    WORD = "word"
    CHARACTER = "character"
    PAGE = "page"
    TOPIC = "topic"  # For audio/video
    SCENE = "scene"  # For video
    TIME = "time"    # For audio/video

class ContentType(str, Enum):
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    WEB = "web"
    IMAGE = "image"

class ChunkingConfig(BaseModel):
    """Universal chunking configuration."""
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.SEMANTIC)
    target_chunk_size: int = Field(default=4000, ge=100, le=32000)
    max_chunk_size: int = Field(default=8000, ge=500, le=64000)
    min_chunk_size: int = Field(default=100, ge=50, le=2000)
    overlap_size: int = Field(default=200, ge=0, le=2000)
    
    # Content-specific settings
    content_type: ContentType = Field(default=ContentType.DOCUMENT)
    
    # Audio/Video specific
    min_duration: Optional[float] = Field(default=30.0, description="Minimum chunk duration in seconds")
    max_duration: Optional[float] = Field(default=300.0, description="Maximum chunk duration in seconds")
    
    # Semantic chunking specific
    use_ai_boundaries: bool = Field(default=True)
    fallback_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.PARAGRAPH)
    
    # Per-request overrides
    custom_settings: Dict[str, Any] = Field(default_factory=dict)

# Default configurations per content type
DEFAULT_CONFIGS = {
    ContentType.DOCUMENT: ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        target_chunk_size=4000,
        max_chunk_size=8000
    ),
    ContentType.AUDIO: ChunkingConfig(
        strategy=ChunkingStrategy.TOPIC,
        min_duration=30.0,
        max_duration=300.0
    ),
    ContentType.VIDEO: ChunkingConfig(
        strategy=ChunkingStrategy.SCENE,
        min_duration=30.0,
        max_duration=300.0
    ),
    ContentType.WEB: ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        target_chunk_size=3000,
        max_chunk_size=6000
    )
}
```

## Testing and Documentation Strategy

### Automated Testing (Each Step)
- Run automated tests in `/tests/test_semantic_chunking.py` after each implementation step
- Test semantic boundary detection with various document types
- Test fallback mechanisms when AI fails
- Test configuration system with different content types
- Test performance against baseline metrics

### Documentation Updates (Mandatory)
- Update `packages/morag-document/README.md` with semantic chunking details
- Update `packages/morag-audio/README.md` with topic-based chunking
- Update `packages/morag-video/README.md` with scene-based chunking
- Update `packages/morag-web/README.md` with semantic web chunking
- Update `CLI.md` with new chunking strategy options
- Update `docs/chunking-strategies.md` with comprehensive semantic approach guide
- Remove documentation for old chunking methods

### Code Cleanup (Mandatory)
- Remove ALL old chunking implementations
- Remove old time-based chunking for audio/video
- Remove old HTML-structure-only chunking for web
- Remove old fixed-size chunking strategies
- Update all imports and dependencies

## Success Criteria

1. ✅ Semantic chunking improves retrieval relevance by 20%
2. ✅ All content types support appropriate semantic strategies
3. ✅ Configuration system allows per-request strategy selection
4. ✅ ALL old chunking code completely removed
5. ✅ Comprehensive automated test coverage
6. ✅ All documentation updated
7. ✅ Performance validated with automated testing

## Dependencies

- Completed PydanticAI foundation
- Content type detection system
- NO dependency on existing chunking infrastructure (complete replacement)
