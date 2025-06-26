# Task 2: Entity Extraction PydanticAI Migration

## Background

Entity extraction is a core component of MoRAG's knowledge graph functionality. Currently implemented in `packages/morag-graph/src/morag_graph/extraction/entity_extractor.py`, it uses direct Gemini API calls with manual JSON parsing and validation.

### Current Implementation Issues

1. **Manual JSON Parsing**: Error-prone response parsing
2. **No Validation**: No automatic validation of extracted entities
3. **Inconsistent Error Handling**: Basic try-catch without structured recovery
4. **Hard-coded Prompts**: Prompts embedded in code without easy modification
5. **Limited Type Safety**: No compile-time checking of entity structures

## Implementation Strategy

### Phase 1: Analyze Current Implementation (0.5 days)

#### Current Entity Extractor Analysis
**File**: `packages/morag-graph/src/morag_graph/extraction/entity_extractor.py`

**Current Flow**:
1. Text preprocessing and chunking
2. Direct Gemini API call with JSON prompt
3. Manual JSON parsing with error handling
4. Entity object creation from parsed data
5. Confidence scoring and filtering

**Key Methods to Migrate**:
- `extract()` - Main extraction method
- `_extract_single()` - Single-pass extraction
- `_extract_chunked()` - Chunked extraction for large texts
- `parse_response()` - JSON response parsing

### Phase 2: Create Entity Extraction Agent (1 day)

#### 2.1 Entity Extraction Agent
**File**: `packages/morag-graph/src/morag_graph/ai/entity_agent.py`

```python
from pydantic_ai import Agent
from morag_core.ai.base_agent import MoRAGBaseAgent
from morag_core.ai.models.entity_models import EntityExtractionResult
from typing import List, Optional

class EntityExtractionAgent(MoRAGBaseAgent[EntityExtractionResult]):
    """PydanticAI agent for entity extraction."""
    
    def get_result_type(self) -> type[EntityExtractionResult]:
        return EntityExtractionResult
    
    def get_system_prompt(self) -> str:
        return """You are an expert entity extraction system. Extract entities from the given text.

        For each entity, identify:
        - name: The exact entity name as it appears in text
        - type: One of PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, CONCEPT, EVENT
        - description: Brief description of the entity's role/significance
        - confidence: Your confidence in this extraction (0.0-1.0)
        
        Focus on:
        - Named entities (proper nouns)
        - Technical terms and concepts
        - Organizations and companies
        - People and their roles
        - Locations and places
        - Events and processes
        
        Avoid:
        - Common words and generic terms
        - Pronouns and articles
        - Very short entities (< 2 characters)
        """
    
    async def extract_entities(
        self, 
        text: str, 
        context: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> EntityExtractionResult:
        """Extract entities from text with optional context."""
        
        user_prompt = f"""Extract entities from this text:

        {text}
        
        {f"Context: {context}" if context else ""}
        
        Return only entities with confidence >= {min_confidence}
        """
        
        result = await self.run(user_prompt)
        
        # Filter by confidence
        filtered_entities = [
            entity for entity in result.entities 
            if entity.confidence >= min_confidence
        ]
        
        return EntityExtractionResult(
            entities=filtered_entities,
            total_count=len(filtered_entities),
            processing_metadata={
                "original_count": len(result.entities),
                "filtered_count": len(filtered_entities),
                "min_confidence": min_confidence,
                "text_length": len(text)
            }
        )
```

#### 2.2 Enhanced Entity Models
**File**: `packages/morag-core/src/morag_core/ai/models/entity_models.py` (extend existing)

```python
# Add to existing entity models

class EntityExtractionConfig(BaseModel):
    """Configuration for entity extraction."""
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_entities_per_chunk: int = Field(default=50)
    chunk_size: int = Field(default=3000)
    overlap_size: int = Field(default=500)
    enable_chunking: bool = Field(default=True)
    entity_types: List[EntityType] = Field(default_factory=lambda: list(EntityType))

class ChunkedExtractionResult(BaseModel):
    """Result of chunked entity extraction."""
    chunks_processed: int
    total_entities: int
    unique_entities: int
    entities_by_chunk: List[EntityExtractionResult]
    merged_entities: List[ExtractedEntity]
    processing_time: float
```

### Phase 3: Migrate Existing Functionality (1 day)

#### 3.1 Update EntityExtractor Class
**File**: `packages/morag-graph/src/morag_graph/extraction/entity_extractor.py`

```python
# COMPLETELY REPLACE existing implementation - DELETE ALL OLD CODE

from morag_graph.ai.entity_agent import EntityExtractionAgent
from morag_core.ai.models.entity_models import EntityExtractionConfig

class EntityExtractor:
    """PydanticAI-based entity extractor - completely new implementation."""
    
    def __init__(self, config: Optional[EntityExtractionConfig] = None):
        self.config = config or EntityExtractionConfig()
        self.agent = EntityExtractionAgent()
        self.logger = structlog.get_logger(__name__)
    
    async def extract(
        self, 
        text: str, 
        doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Entity]:
        """Extract entities using PydanticAI agent."""
        
        # Use chunked extraction for large texts
        if len(text) > self.config.chunk_size and self.config.enable_chunking:
            return await self._extract_chunked(text, doc_id, **kwargs)
        else:
            return await self._extract_single(text, doc_id, **kwargs)
    
    async def _extract_single(self, text: str, doc_id: Optional[str] = None) -> List[Entity]:
        """Single-pass extraction using PydanticAI."""
        try:
            result = await self.agent.extract_entities(
                text=text,
                min_confidence=self.config.min_confidence
            )
            
            # Convert to Entity objects
            entities = []
            for extracted_entity in result.entities:
                entity = Entity(
                    name=extracted_entity.name,
                    type=extracted_entity.type.value,
                    description=extracted_entity.description,
                    confidence=extracted_entity.confidence,
                    properties=extracted_entity.properties
                )
                entities.append(entity)
            
            self.logger.info(
                "Entity extraction completed",
                entities_found=len(entities),
                text_length=len(text),
                doc_id=doc_id
            )
            
            return entities
            
        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e), doc_id=doc_id)
            return []
    
    async def _extract_chunked(self, text: str, doc_id: Optional[str] = None) -> List[Entity]:
        """Chunked extraction for large texts."""
        # Implementation for chunked processing
        # Similar to current implementation but using PydanticAI agent
        pass
```

### Phase 4: Testing and Validation (0.5 days)

#### 4.1 Unit Tests
**File**: `tests/test_entity_extraction_migration.py`

```python
import pytest
from morag_graph.ai.entity_agent import EntityExtractionAgent
from morag_core.ai.models.entity_models import EntityType

@pytest.mark.asyncio
async def test_entity_extraction_agent():
    """Test basic entity extraction functionality."""
    agent = EntityExtractionAgent()
    
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    result = await agent.extract_entities(text)
    
    assert len(result.entities) > 0
    
    # Check for expected entities
    entity_names = [e.name for e in result.entities]
    assert "Apple Inc." in entity_names
    assert "Steve Jobs" in entity_names
    assert "Cupertino" in entity_names

@pytest.mark.asyncio
async def test_confidence_filtering():
    """Test confidence-based filtering."""
    agent = EntityExtractionAgent()
    
    text = "Some text with entities"
    result = await agent.extract_entities(text, min_confidence=0.8)
    
    # All entities should have confidence >= 0.8
    for entity in result.entities:
        assert entity.confidence >= 0.8
```

## Migration Strategy

### Backward Compatibility
1. Keep existing `EntityExtractor` interface
2. Add feature flag for PydanticAI vs legacy implementation
3. Gradual rollout with A/B testing

### Performance Considerations
1. Benchmark PydanticAI vs current implementation
2. Monitor response times and accuracy
3. Optimize prompts for better performance

### Testing and Documentation Strategy

#### Automated Testing (Each Step)
- Run automated tests in `/tests/test_entity_extraction.py` after each implementation step
- Test basic entity extraction functionality with real Gemini API calls
- Test confidence filtering and validation
- Test chunked extraction for large documents
- Test error handling scenarios

#### Documentation Updates (Mandatory)
- Update `packages/morag-graph/README.md` with new PydanticAI entity extraction
- Update `docs/entity-extraction.md` with new implementation details
- Remove documentation for old JSON parsing methods
- Update API documentation with new structured models

#### Code Cleanup (Mandatory)
- Remove ALL old entity extraction code
- Remove old JSON parsing methods (`parse_response`, `parse_json_response`)
- Remove old LLM configuration classes
- Remove old prompt templates
- Update all imports throughout the codebase

## Success Criteria

1. ✅ Entity extraction accuracy maintained or improved
2. ✅ Structured validation eliminates parsing errors
3. ✅ ALL old code completely removed
4. ✅ Comprehensive automated test coverage
5. ✅ All documentation updated
6. ✅ Production deployment successful

## Dependencies

- Completed PydanticAI foundation setup
- Entity models defined
- Testing infrastructure ready

## Risks and Mitigation

1. **Risk**: Performance degradation
   **Mitigation**: Thorough benchmarking, optimization

2. **Risk**: Accuracy changes
   **Mitigation**: A/B testing, prompt engineering

3. **Risk**: Integration issues
   **Mitigation**: Gradual rollout, feature flags
