# Enhanced Entity Normalization Implementation

## Overview

This implementation addresses the requirements for enhanced entity normalization and relationship modeling in the MoRAG fact extraction system. The key improvements include:

1. **Generic Entity Labels**: All entities use the generic "ENTITY" label instead of specific types
2. **Entity Normalization**: Comprehensive normalization to canonical forms
3. **Global Entity Uniqueness**: Entities are unique by normalized name across all documents
4. **Semantic Relationships**: Declarative relationship types between entities and facts
5. **Traversable Graph Structure**: Enables intelligent entity retrieval through graph traversal

## Key Changes

### 1. Entity Model Updates

**File**: `packages/morag-graph/src/morag_graph/models/entity.py`

- Modified `get_neo4j_label()` method to always return "ENTITY"
- All entities now use the same Neo4j label for unified querying
- Original entity type is preserved in the `type` property

```python
def get_neo4j_label(self) -> str:
    """Get the Neo4j label for this entity.
    
    All entities use the generic 'ENTITY' label for unified querying.
    The specific type is stored in the 'type' property.
    """
    return "ENTITY"
```

### 2. Enhanced Entity Normalization

**File**: `packages/morag-graph/src/morag_graph/extraction/entity_normalizer.py`

Enhanced the normalization logic to create canonical entity forms:

- **Remove qualifiers**: Strips adjectives, brand names, and descriptive modifiers
- **Singularization**: Converts plural forms to singular
- **Parentheses removal**: Removes content like "Engelwurz (Wurzel)" → "Engelwurz"
- **Gender neutralization**: Uses base forms instead of gender-specific variants
- **Proper capitalization**: Maintains appropriate capitalization for the language

#### Basic Normalization Rules:
```python
def _apply_basic_normalization(self, entity_name: str) -> str:
    # Remove articles (the, a, an, der, die, das, etc.)
    # Remove content in parentheses
    # Remove common qualifiers (pure, natural, organic, etc.)
    # Convert to singular form
    # Apply proper capitalization
```

#### LLM-Enhanced Normalization:
- Uses Gemini API for advanced normalization
- Provides context-aware canonical forms
- Handles complex linguistic patterns

### 3. Enhanced Fact Processing Service

**File**: `packages/morag-graph/src/morag_graph/services/enhanced_fact_processing_service.py`

Major updates to implement the new requirements:

#### Entity Creation with Normalization:
```python
async def _create_entities_from_facts(self, facts: List[Fact]) -> List[Entity]:
    """Create normalized entities from fact subjects and objects.
    
    All entities use generic 'ENTITY' label and are normalized to canonical forms.
    Entities are unique by normalized name globally across all documents.
    """
```

#### Semantic Relationship Types:
The service now determines semantic relationship types based on fact content:

**Subject Relationships**:
- `CAUSES`: When fact indicates causation
- `TREATS`: When fact indicates treatment/healing
- `PREVENTS`: When fact indicates prevention
- `CONTAINS`: When fact indicates containment
- `AFFECTS`: When fact indicates influence
- `PRODUCES`: When fact indicates production
- `INVOLVES`: Default semantic relationship

**Object Relationships**:
- `SOLVES`: When fact addresses/fixes something
- `TARGETS`: When fact focuses on something
- `IMPROVES`: When fact enhances something
- `REDUCES`: When fact decreases something
- `SUPPORTS`: When fact aids something
- `REQUIRES`: When fact needs something
- `RELATES_TO`: Default semantic relationship

**Keyword Relationships**:
- `CATEGORIZED_AS`: For domain/category keywords
- `ADDRESSES`: For symptom/condition keywords
- `USES_METHOD`: For method/technique keywords
- `TAGGED_WITH`: Default for general keywords

### 4. Global Entity Uniqueness

The implementation ensures global entity uniqueness through:

1. **Normalized Name Keys**: Entities are deduplicated using normalized names
2. **Neo4j MERGE Operations**: Database-level uniqueness enforcement
3. **Cross-Document Consistency**: Same entity referenced from multiple documents

### 5. Graph Traversal Capabilities

The new structure enables intelligent retrieval patterns:

```cypher
// Find all facts related to an entity
MATCH (e:ENTITY {name: "Engelwurz"})<-[r]-(f:Fact)
RETURN f, r.type

// Recursive entity traversal
MATCH (e1:ENTITY {name: "ADHD"})<-[:SOLVES]-(f:Fact)-[:INVOLVES]->(e2:ENTITY)
RETURN e2.name, f
```

## Testing

Comprehensive tests verify the implementation:

### Test Coverage:
1. **Entity Normalization**: Verifies canonical form creation
2. **Global Uniqueness**: Ensures no duplicate entities
3. **Generic Labels**: Confirms all entities use "ENTITY" label
4. **Semantic Relationships**: Validates relationship type determination
5. **Keyword Processing**: Tests keyword entity normalization

### Test Results:
```
test_entity_normalization_and_uniqueness PASSED
test_semantic_relationship_types PASSED
test_keyword_entity_normalization PASSED
test_entity_model_generic_label PASSED
```

## Usage Examples

### Creating Normalized Entities:
```python
from morag_graph.services.enhanced_fact_processing_service import EnhancedFactProcessingService
from morag_graph.extraction.entity_normalizer import LLMEntityNormalizer

# Initialize with normalizer
normalizer = LLMEntityNormalizer()
service = EnhancedFactProcessingService(neo4j_storage, normalizer)

# Process facts with entity normalization
result = await service.process_facts_with_entities(
    facts=facts,
    create_keyword_entities=True,
    create_mandatory_relations=True
)
```

### Querying Normalized Entities:
```cypher
// Find all entities related to a specific topic
MATCH (e:ENTITY)<-[r:INVOLVES|TREATS|CAUSES]-(f:Fact)
WHERE e.name = "Stress"
RETURN f, r.type, e

// Find treatment relationships
MATCH (treatment:ENTITY)-[:TREATS]->(condition:ENTITY)
RETURN treatment.name, condition.name
```

## Benefits

1. **Unified Querying**: Single ENTITY label simplifies graph queries
2. **Canonical Consistency**: Normalized entities prevent duplicates
3. **Semantic Clarity**: Relationship types provide clear traversal paths
4. **Global Scope**: Entities work across all documents and domains
5. **Intelligent Retrieval**: Enables sophisticated fact discovery patterns

## Migration Notes

- Existing entities will be migrated to use the generic ENTITY label
- Original type information is preserved in the `type` property
- Relationship types are enhanced but backward compatible
- Entity normalization is optional but recommended for best results

## Performance Considerations

- Entity normalization adds processing time but improves query performance
- Global uniqueness reduces storage overhead
- Semantic relationships enable more efficient graph traversal
- Caching mechanisms optimize repeated normalization operations
