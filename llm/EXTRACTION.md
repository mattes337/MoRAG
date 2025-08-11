# Entity and Relation Extraction Rules

## Core Principles

### Domain Agnostic Approach
- NO hardcoded entity types or relation types
- LLM determines types dynamically based on content
- Support ALL possible domains (medical, legal, technical, general, etc.)
- Avoid static type definitions

### Graph Structure
```
Document -> DocumentChunk -> Fact -> Entity
                         -> Relation
```

**Document**: Source file with metadata
**DocumentChunk**: Segmented content piece with position info
**Fact**: Extracted statement with subject-predicate-object structure
**Entity**: Unique named entity (person, organization, concept, etc.)
**Relation**: Semantic connection between entities

## Entity Extraction

### Entity Identification Rules
1. Extract ALL meaningful entities (persons, organizations, locations, concepts, objects)
2. Use broad LLM-generated labels instead of generic 'Entity'
3. Normalize entity names for deduplication
4. Generate deterministic IDs based on normalized names

### Entity Structure
```json
{
  "id": "ent_abc123",
  "name": "Apple Inc.",
  "type": "ORGANIZATION",
  "confidence": 0.95,
  "source_doc_id": "doc_001",
  "attributes": {
    "domain": "technology",
    "normalized_name": "apple inc",
    "source_chunk_id": "chunk_001"
  }
}
```

### Entity Types (LLM-Generated)
- PERSON, ORGANIZATION, LOCATION
- CONCEPT, TECHNOLOGY, PRODUCT
- EVENT, PROCESS, METHOD
- DISEASE, MEDICATION, SYMPTOM (medical)
- LAW, REGULATION, CASE (legal)
- Any domain-specific type the LLM identifies

### Entity Normalization
**Process**: Convert entity names to lowercase, strip whitespace, and generate deterministic IDs using MD5 hash of normalized name (first 12 characters with "ent_" prefix).

## Relation Extraction

### Semantic Relationship Types
Use domain-specific, meaningful relation types instead of generic ones:

**Medical Domain**:
- TREATS, CAUSES, PREVENTS, DIAGNOSES
- PRESCRIBES, ADMINISTERS, MONITORS
- INDICATES, CONTRAINDICATES, INTERACTS_WITH

**Business Domain**:
- PARTNERS_WITH, COMPETES_WITH, ACQUIRES
- SUPPLIES, DISTRIBUTES, MANUFACTURES
- INVESTS_IN, COLLABORATES_WITH

**Technical Domain**:
- IMPLEMENTS, EXTENDS, DEPENDS_ON
- PROCESSES, GENERATES, CONSUMES
- INTEGRATES_WITH, REPLACES, ENHANCES

**General Relations**:
- LOCATED_IN, PART_OF, MEMBER_OF
- CREATED_BY, OWNED_BY, MANAGED_BY
- RELATED_TO, ASSOCIATED_WITH

### Relation Structure
```json
{
  "id": "rel_xyz789",
  "source_id": "ent_abc123",
  "target_id": "ent_def456",
  "type": "PARTNERS_WITH",
  "confidence": 0.87,
  "source_document": "doc_001",
  "attributes": {
    "context": "Apple partners with Stanford for AI research",
    "source_chunk_id": "chunk_001"
  }
}
```

### Relation Type Normalization
- Convert to uppercase: "partners with" -> "PARTNERS_WITH"
- Use singular forms: "TREATS" not "TREAT"
- Normalize similar relations: "collaborates with" -> "COLLABORATES_WITH"

## Fact Extraction

### Fact Structure
```json
{
  "id": "fact_001",
  "subject": "Dr. Smith",
  "predicate": "prescribed",
  "object": "aspirin",
  "confidence": 0.95,
  "source_document_id": "doc_001",
  "source_chunk_id": "chunk_001",
  "domain": "medical",
  "context": "Dr. Smith prescribed aspirin to treat headache"
}
```

### Fact Extraction Rules
1. Extract actionable, direct facts (not meta-commentary)
2. Include ALL relevant metadata (timecodes, pages, chapters)
3. Create entities for subjects, objects, and keywords
4. Generate exhaustive facts above confidence threshold
5. No maximum facts limit - extract all relevant information

### Fact-to-Entity Conversion
Each fact generates:
- Subject entity with HAS_FACT relation
- Object entity with DESCRIBED_BY_FACT relation
- Keyword entities with MENTIONED_IN_FACT relations

## Deduplication

### Entity Deduplication
1. **Normalization**: Convert to lowercase, remove punctuation
2. **Similarity Matching**: Use embedding similarity + string matching
3. **LLM Validation**: Confirm merges with LLM analysis
4. **Merge Application**: Update all references to merged entities

### Deduplication Process
**Steps**:
1. **Name Normalization**: Convert entity names to lowercase and strip whitespace
2. **Candidate Finding**: Identify similar entities using similarity threshold (0.8)
3. **LLM Validation**: Use LLM to confirm which entities should be merged
4. **Merge Application**: Update all entity references to point to canonical entities

### Cross-Document Deduplication
- Entities are global across all documents
- Same entity mentioned in multiple documents gets single ID
- Relations updated to point to canonical entity IDs

## Extraction Context

### Required Context
```json
{
  "domain": "medical",
  "language": "en",
  "source_file_name": "medical_notes.txt",
  "chunk_id": "chunk_001",
  "document_id": "doc_001"
}
```

### Domain Detection
- Infer domain from content when not specified
- Use domain-specific extraction prompts
- Adapt entity/relation types to domain

### Quality Validation
- Confidence thresholds: entities (0.7), relations (0.6), facts (0.8)
- Fact validation never fails (always returns valid facts)
- Cross-reference extracted entities with chunk content

## Implementation Notes

### Vector Embeddings
- Store embeddings in Neo4j for entity-chunk mapping
- Use text-embedding-004 for consistency
- Enable vector search for entity retrieval

### Source Attribution
- Reference actual documents (page/chapter/timecode)
- Not knowledge graph entities
- Include in final answer references

### Performance Considerations
- Batch entity processing for efficiency
- Cache normalized entity lookups
- Use deterministic IDs for consistency
