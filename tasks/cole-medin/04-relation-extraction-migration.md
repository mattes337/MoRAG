# Task 4: Relation Extraction PydanticAI Migration

## Background

Relation extraction is crucial for building meaningful knowledge graphs. Currently implemented in `packages/morag-graph/src/morag_graph/extraction/relation_extractor.py`, it uses direct Gemini API calls with manual parsing and entity resolution.

### Current Implementation Issues

1. **Complex Entity Resolution**: Manual matching of relation entities to extracted entities
2. **Inconsistent Relation Types**: No standardized relation taxonomy
3. **Error-Prone Parsing**: Manual JSON parsing with limited validation
4. **Missing Context**: Relations extracted without sufficient context preservation
5. **Performance Issues**: Sequential processing without optimization

## Implementation Strategy

### Phase 1: Relation Models and Agent (1 day)

#### 1.1 Enhanced Relation Models
**File**: `packages/morag-core/src/morag_core/ai/models/relation_models.py`

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class RelationType(str, Enum):
    """Standardized relation types."""
    # Organizational relations
    WORKS_FOR = "WORKS_FOR"
    FOUNDED_BY = "FOUNDED_BY"
    OWNS = "OWNS"
    PART_OF = "PART_OF"
    
    # Personal relations
    KNOWS = "KNOWS"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    REPORTS_TO = "REPORTS_TO"
    
    # Technical relations
    USES_TECHNOLOGY = "USES_TECHNOLOGY"
    DEVELOPS = "DEVELOPS"
    IMPLEMENTS = "IMPLEMENTS"
    BASED_ON = "BASED_ON"
    
    # Spatial relations
    LOCATED_IN = "LOCATED_IN"
    NEAR = "NEAR"
    
    # Temporal relations
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DURING = "DURING"
    
    # Causal relations
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    PREVENTS = "PREVENTS"
    
    # Generic
    RELATED_TO = "RELATED_TO"
    MENTIONS = "MENTIONS"

class ExtractedRelation(BaseModel):
    """Enhanced model for extracted relations."""
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relation_type: RelationType = Field(description="Standardized relation type")
    description: str = Field(description="Natural language description of relation")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    context: str = Field(description="Text context where relation was found")
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('source_entity', 'target_entity')
    def validate_entities(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Entity names must be at least 2 characters")
        return v.strip()

class RelationExtractionResult(BaseModel):
    """Result of relation extraction with metadata."""
    relations: List[ExtractedRelation]
    total_count: int
    entities_referenced: List[str]
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('total_count')
    def validate_count(cls, v, values):
        if 'relations' in values and v != len(values['relations']):
            raise ValueError("Total count must match relations list length")
        return v
```

#### 1.2 Relation Extraction Agent
**File**: `packages/morag-graph/src/morag_graph/ai/relation_agent.py`

```python
from pydantic_ai import Agent
from morag_core.ai.base_agent import MoRAGBaseAgent
from morag_core.ai.models.relation_models import RelationExtractionResult, RelationType
from typing import List, Optional, Dict, Any

class RelationExtractionAgent(MoRAGBaseAgent[RelationExtractionResult]):
    """PydanticAI agent for relation extraction."""
    
    def get_result_type(self) -> type[RelationExtractionResult]:
        return RelationExtractionResult
    
    def get_system_prompt(self) -> str:
        relation_types = ", ".join([rt.value for rt in RelationType])
        
        return f"""You are an expert relation extraction system. Extract meaningful relationships between entities in the given text.

        Available relation types: {relation_types}

        For each relation, identify:
        - source_entity: The entity that is the subject of the relation
        - target_entity: The entity that is the object of the relation
        - relation_type: One of the standardized relation types
        - description: Natural language description of the relationship
        - confidence: Your confidence in this extraction (0.0-1.0)
        - context: The specific text snippet that supports this relation

        Guidelines:
        - Focus on explicit, clearly stated relationships
        - Avoid inferring relationships not directly supported by text
        - Use the most specific relation type available
        - Ensure both entities are meaningful (not pronouns or generic terms)
        - Include sufficient context to understand the relationship
        - Prefer higher confidence relations over uncertain ones
        """
    
    async def extract_relations(
        self,
        text: str,
        known_entities: Optional[List[str]] = None,
        min_confidence: float = 0.6
    ) -> RelationExtractionResult:
        """Extract relations from text with optional entity context."""
        
        entity_context = ""
        if known_entities:
            entity_context = f"\nKnown entities in this text: {', '.join(known_entities)}"
        
        user_prompt = f"""Extract relations from this text:

        {text}
        {entity_context}

        Focus on relations involving the known entities if provided.
        Return only relations with confidence >= {min_confidence}
        """
        
        result = await self.run(user_prompt)
        
        # Filter by confidence
        filtered_relations = [
            relation for relation in result.relations
            if relation.confidence >= min_confidence
        ]
        
        # Extract unique entities referenced
        entities_referenced = set()
        for relation in filtered_relations:
            entities_referenced.add(relation.source_entity)
            entities_referenced.add(relation.target_entity)
        
        return RelationExtractionResult(
            relations=filtered_relations,
            total_count=len(filtered_relations),
            entities_referenced=list(entities_referenced),
            processing_metadata={
                "original_count": len(result.relations),
                "filtered_count": len(filtered_relations),
                "min_confidence": min_confidence,
                "text_length": len(text),
                "known_entities_count": len(known_entities) if known_entities else 0
            }
        )
```

### Phase 2: Migrate Existing Functionality (1 day)

#### 2.1 Update RelationExtractor Class
**File**: `packages/morag-graph/src/morag_graph/extraction/relation_extractor.py`

```python
from morag_graph.ai.relation_agent import RelationExtractionAgent
from morag_core.ai.models.relation_models import RelationExtractionResult
from morag_graph.models.entity import Entity
from morag_graph.models.relation import Relation
from typing import List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

class RelationExtractor:
    """PydanticAI-based relation extractor - completely new implementation."""
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.agent = RelationExtractionAgent()
        self.logger = structlog.get_logger(__name__)
    
    async def extract(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Relation]:
        """Extract relations using PydanticAI agent."""
        
        # Extract entity names for context
        known_entities = [entity.name for entity in entities] if entities else None
        
        try:
            # Use PydanticAI agent for extraction
            result = await self.agent.extract_relations(
                text=text,
                known_entities=known_entities,
                min_confidence=self.min_confidence
            )
            
            # Convert to Relation objects with entity ID resolution
            relations = []
            entity_name_to_id = {entity.name: entity.id for entity in entities} if entities else {}
            
            for extracted_relation in result.relations:
                # Resolve entity IDs
                source_id = entity_name_to_id.get(extracted_relation.source_entity)
                target_id = entity_name_to_id.get(extracted_relation.target_entity)
                
                # Create relation object
                relation = Relation(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=extracted_relation.relation_type.value,
                    description=extracted_relation.description,
                    confidence=extracted_relation.confidence,
                    properties={
                        **extracted_relation.properties,
                        "context": extracted_relation.context,
                        "extraction_method": "pydantic_ai"
                    }
                )
                
                relations.append(relation)
            
            self.logger.info(
                "Relation extraction completed",
                relations_found=len(relations),
                entities_provided=len(entities) if entities else 0,
                text_length=len(text),
                doc_id=doc_id
            )
            
            return relations
            
        except Exception as e:
            self.logger.error(
                "Relation extraction failed",
                error=str(e),
                doc_id=doc_id,
                text_length=len(text)
            )
            return []
    
    async def extract_with_chunking(
        self,
        text: str,
        entities: List[Entity],
        chunk_size: int = 3000,
        overlap: int = 500
    ) -> List[Relation]:
        """Extract relations using chunked processing for large texts."""
        
        if len(text) <= chunk_size:
            return await self.extract(text, entities)
        
        all_relations = []
        processed_relations = set()  # For deduplication
        
        # Create overlapping chunks
        chunks = self._create_chunks(text, chunk_size, overlap)
        
        for i, chunk in enumerate(chunks):
            chunk_relations = await self.extract(chunk, entities)
            
            # Deduplicate relations
            for relation in chunk_relations:
                relation_key = (
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation.relation_type
                )
                
                if relation_key not in processed_relations:
                    processed_relations.add(relation_key)
                    all_relations.append(relation)
        
        return all_relations
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping text chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.7:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk)
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
```

## Testing and Documentation Strategy

### Automated Testing (Each Step)
- Run automated tests in `/tests/test_relation_extraction.py` after each implementation step
- Test relation extraction accuracy with standardized relation types
- Test entity resolution and ID mapping
- Test confidence filtering and validation
- Test chunked processing for large documents
- Test deduplication logic

### Documentation Updates (Mandatory)
- Update `packages/morag-graph/README.md` with new PydanticAI relation extraction
- Update `docs/relation-extraction.md` with standardized relation types
- Update API documentation with new structured relation models
- Remove documentation for old JSON parsing methods

### Code Cleanup (Mandatory)
- Remove ALL old relation extraction code
- Remove old JSON parsing methods for relations
- Remove old relation type definitions
- Remove old entity resolution logic
- Update all imports throughout the codebase

## Success Criteria

1. ✅ Relation extraction accuracy maintained or improved
2. ✅ Structured validation eliminates parsing errors
3. ✅ Entity resolution reliability improved
4. ✅ ALL old code completely removed
5. ✅ Comprehensive automated test coverage
6. ✅ All documentation updated

## Dependencies

- Completed PydanticAI foundation
- Enhanced relation models
- New entity extraction system (no dependency on old system)
