# Hybrid Fact Extraction Implementation

## Overview

This document describes the implementation of the hybrid fact extraction approach in MoRAG, which combines self-contained fact text with structured metadata for optimal quality and graph building capabilities.

## Architecture

### Core Components

1. **Hybrid Fact Model** (`morag_graph.models.fact.Fact`)
   - `fact_text`: Complete, self-contained fact statement
   - `structured_metadata`: Extracted entities, relationships, and concepts

2. **Enhanced Extraction Prompts** (`morag_graph.extraction.fact_prompts`)
   - Focus on self-contained fact generation
   - Metadata extraction for graph building
   - Improved few-shot examples

3. **Updated Graph Building** (`morag_graph.extraction.fact_entity_converter`)
   - Entity creation from metadata
   - Relationship mapping from fact content
   - Pure hybrid format support

4. **Response Generation Updates** (`morag_reasoning.response_generator`)
   - Direct use of fact_text for responses
   - Enhanced readability

## Hybrid Fact Structure

### New Format
```json
{
  "fact_text": "PostgreSQL query performance can be optimized by creating B-tree indexes on frequently queried columns using CREATE INDEX syntax, with composite queries requiring multi-column indexes where the most selective column is placed first.",
  "structured_metadata": {
    "primary_entities": ["PostgreSQL", "B-tree index", "query performance"],
    "relationships": ["optimizes", "improves", "requires"],
    "domain_concepts": ["CREATE INDEX", "composite queries", "column selectivity"]
  },
  "fact_type": "methodological",
  "confidence": 0.95,
  "keywords": ["PostgreSQL", "B-tree index", "query optimization"]
}
```



## Benefits

### Quality Improvements
- **Self-contained facts**: No need for additional context
- **Natural language**: Better readability and comprehension
- **Complete information**: All relevant details in single statement
- **Reduced hallucination**: Less rigid schema constraints

### Graph Building Capabilities
- **Entity extraction**: From primary_entities and domain_concepts
- **Relationship mapping**: From relationships metadata
- **Keyword indexing**: Enhanced search capabilities
- **Legacy support**: Backward compatibility maintained

### Response Generation
- **Direct usage**: fact_text used directly in responses
- **Better flow**: Natural language integration
- **Metadata context**: Additional context from structured data
- **Fallback support**: Graceful handling of legacy formats

## Implementation Details

### Fact Extraction Process

1. **LLM Prompt**: Enhanced prompts request hybrid format
2. **Response Parsing**: Handles both hybrid and legacy formats
3. **Fact Construction**: Creates Fact objects with hybrid structure
4. **Validation**: Ensures completeness and quality

### Graph Building Process

1. **Entity Creation**: From primary_entities and domain_concepts
2. **Relationship Mapping**: Based on relationships metadata
3. **Deduplication**: Prevents duplicate entities

## Usage Examples

### Creating Hybrid Facts
```python
from morag_graph.models.fact import Fact, StructuredMetadata

metadata = StructuredMetadata(
    primary_entities=["Ashwagandha", "stress", "anxiety"],
    relationships=["treats", "reduces"],
    domain_concepts=["herbal medicine", "adaptogen", "dosage"]
)

fact = Fact(
    fact_text="Ashwagandha extract containing 5% withanolides should be taken at 300-600mg twice daily with meals for 8-12 weeks to effectively manage chronic stress and anxiety.",
    structured_metadata=metadata,
    source_chunk_id="chunk_123",
    source_document_id="doc_456",
    extraction_confidence=0.9,
    fact_type="methodological"
)
```



### Graph Building
```python
from morag_graph.extraction.fact_entity_converter import FactEntityConverter

converter = FactEntityConverter()
entities, relationships = converter.convert_facts_to_entities(hybrid_facts)
```

## Testing

Comprehensive tests are provided in `packages/morag-graph/tests/test_hybrid_fact_extraction.py`:

- **Model Tests**: StructuredMetadata and Fact model validation
- **Extraction Tests**: Hybrid format parsing
- **Conversion Tests**: Entity and relationship creation

## Performance Considerations

### Advantages
- **Reduced Processing**: Direct fact_text usage eliminates reconstruction
- **Better Caching**: Self-contained facts cache more effectively
- **Improved Search**: Enhanced metadata supports better indexing
- **Flexible Querying**: Multiple access patterns supported

### Considerations
- **Storage Overhead**: Slightly larger fact objects due to metadata
- **Prompt Complexity**: More sophisticated LLM prompts required

## Future Enhancements

1. **Advanced Metadata**: Additional structured fields as needed
2. **Domain-Specific Templates**: Specialized formats for different domains
3. **Automatic Enhancement**: Post-processing to improve fact quality
4. **Multi-language Support**: Enhanced language-specific processing
5. **Quality Metrics**: Automated quality assessment and improvement

## Conclusion

The hybrid fact extraction approach successfully combines the benefits of natural language facts with structured metadata, providing:

- **Better Quality**: Self-contained, readable facts
- **Graph Compatibility**: Structured metadata for entity/relationship extraction
- **Enhanced Responses**: Direct usage in response generation
- **Future Flexibility**: Extensible architecture for future improvements

This implementation provides a clean, modern approach to fact extraction that significantly improves fact quality and usability across the MoRAG system.
