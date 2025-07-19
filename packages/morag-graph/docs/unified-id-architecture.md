# Unified ID Architecture

This document describes the unified ID architecture implemented for the morag-graph package, providing consistent and deterministic identification across all entities in the knowledge graph.

## Overview

The unified ID system replaces the previous mixed approach of UUIDs and custom IDs with a standardized format that:

- Provides deterministic ID generation for reproducible results
- Enables easy identification of entity types from IDs
- Supports hierarchical relationships (e.g., chunks belonging to documents)
- Maintains compatibility with existing storage systems (Neo4j and Qdrant)
- Includes validation and collision detection mechanisms

## ID Format Specifications

### Document IDs
**Format**: `doc_{filename}_{checksum|timestamp}`

**Examples**:
- `doc_research_paper.pdf_a1b2c3d4e5f6`
- `doc_meeting_notes.txt_1234567890`

**Components**:
- `doc_`: Fixed prefix identifying document type
- `{filename}`: Sanitized source filename
- `{checksum|timestamp}`: File checksum if available, otherwise timestamp

### Document Chunk IDs
**Format**: `{document_id}:chunk:{index}`

**Examples**:
- `doc_research_paper.pdf_a1b2c3d4e5f6:chunk:0`
- `doc_research_paper.pdf_a1b2c3d4e5f6:chunk:15`

**Components**:
- `{document_id}`: Parent document's unified ID
- `:chunk:`: Fixed separator
- `{index}`: Zero-based chunk index within document

### Entity IDs
**Format**: `ent_{name}_{type}_{hash}`

**Examples**:
- `ent_john_doe_person_a1b2c3d4`
- `ent_acme_corporation_organization_e5f6g7h8`

**Components**:
- `ent_`: Fixed prefix identifying entity type
- `{name}`: Sanitized entity name
- `{type}`: Entity type (person, organization, etc.)
- `{hash}`: Hash of name, type, and source document for uniqueness

### Relation IDs
**Format**: `rel_{type}_{hash}`

**Examples**:
- `rel_works_for_a1b2c3d4e5f6g7h8`
- `rel_located_in_i9j0k1l2m3n4o5p6`

**Components**:
- `rel_`: Fixed prefix identifying relation type
- `{type}`: Relation type (works_for, located_in, etc.)
- `{hash}`: Hash of source entity, target entity, and relation type

## Implementation

### Core Classes

#### UnifiedIDGenerator
Central class for generating all types of unified IDs.

```python
from morag_graph.utils import UnifiedIDGenerator

# Generate document ID
doc_id = UnifiedIDGenerator.generate_document_id(
    source_file="research.pdf",
    checksum="abc123"
)

# Generate chunk ID
chunk_id = UnifiedIDGenerator.generate_chunk_id(
    document_id=doc_id,
    chunk_index=5
)

# Generate entity ID
entity_id = UnifiedIDGenerator.generate_entity_id(
    name="John Doe",
    entity_type=EntityType.PERSON,
    source_doc_id=doc_id
)
```

#### IDValidator
Validates ID formats and provides format checking utilities.

```python
from morag_graph.utils import IDValidator

# Validate specific ID types
IDValidator.validate_document_id(doc_id)
IDValidator.validate_chunk_id(chunk_id)
IDValidator.validate_entity_id(entity_id)

# Check if ID is in unified format
if IDValidator.is_unified_format(some_id):
    print("ID is in unified format")
```

#### IDCollisionDetector
Detects potential ID collisions before they occur.

```python
from morag_graph.utils import IDCollisionDetector

detector = IDCollisionDetector()

# Check single ID
detector.check_collision(new_id, existing_ids)

# Batch check multiple IDs
detector.batch_check_collisions(new_ids, existing_ids)

# Get collision report
report = detector.get_collision_report(new_ids, existing_ids)
```

### Model Integration

All models have been updated to use the unified ID system:

#### Document Model
```python
from morag_graph.models import Document

# ID is automatically generated using unified format
doc = Document(
    source_file="research.pdf",
    checksum="abc123",
    # id will be auto-generated as doc_research.pdf_abc123
)

# Access unified ID methods
print(doc.get_unified_id())
print(doc.is_unified_format())
```

#### DocumentChunk Model
```python
from morag_graph.models import DocumentChunk

# Chunk ID automatically includes parent document ID
chunk = DocumentChunk(
    document_id="doc_research.pdf_abc123",
    chunk_index=5,
    text="Chunk content..."
    # id will be auto-generated as doc_research.pdf_abc123:chunk:5
)

# Extract information from chunk ID
print(chunk.extract_document_id())  # Returns parent document ID
print(chunk.extract_chunk_index())  # Returns chunk index
```

## Fresh Data Ingestion

For new data ingestion, the unified ID system automatically generates appropriate IDs for all entities. No migration is needed for fresh data - simply use the standard ingestion pipeline and all IDs will be generated in the unified format.

```python
from morag_graph.utils import UnifiedIDGenerator

# ID generation is automatic during ingestion
# All new entities will use the unified format
generator = UnifiedIDGenerator()

# Generate IDs for new entities
document_id = generator.generate_document_id("example.pdf")
chunk_id = generator.generate_chunk_id(document_id, 0)
entity_id = generator.generate_entity_id("John Doe", "PERSON")
```

## Benefits

### Deterministic Generation
- Same input always produces same ID
- Enables reproducible processing
- Simplifies testing and debugging

### Type Identification
- ID prefix immediately identifies entity type
- Enables type-specific processing without database queries
- Improves code readability and maintenance

### Hierarchical Relationships
- Chunk IDs contain parent document ID
- Enables efficient parent-child queries
- Maintains referential integrity

### Storage Compatibility
- Works with existing Neo4j and Qdrant storage
- Maintains performance characteristics
- Supports both graph and vector operations

### Validation and Safety
- Built-in format validation
- Collision detection prevents duplicates
- Migration tools ensure data integrity

## Performance Considerations

### ID Generation
- Deterministic hashing is fast (O(1))
- No database queries required for generation
- Suitable for high-throughput processing

### Storage Impact
- Unified IDs are slightly longer than UUIDs
- Human-readable format aids debugging
- Indexing performance remains excellent

### Migration Performance
- Batch processing minimizes database load
- Configurable batch sizes for different environments
- Progress tracking and error handling

## Testing

Comprehensive test suites are provided:

### Unit Tests
```bash
# Run ID generation tests
pytest tests/test_id_generation.py

# Run migration tests
pytest tests/test_id_migration.py
```

### Integration Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=morag_graph tests/
```

## Rollback Strategy

In case of migration issues:

1. **Stop Processing**: Halt any ongoing operations
2. **Assess Impact**: Review migration logs and error reports
3. **Restore from Backup**: Use database backups if available
4. **Manual Rollback**: Use migration logs to reverse changes

```python
# Generate rollback plan from migration logs
successful_migrations = [
    log for log in migration_service.migration_log 
    if log['status'] == 'success'
]

# Create rollback operations
for migration in successful_migrations:
    # Reverse the ID change
    rollback_operation = {
        'type': migration['type'],
        'current_id': migration['new_id'],
        'rollback_to_id': migration['old_id']
    }
```

## Future Enhancements

### Planned Features
- **Version Support**: Add version suffixes to IDs
- **Namespace Support**: Support multiple knowledge graphs
- **Custom Prefixes**: Allow configurable ID prefixes
- **Compression**: Optimize ID length for storage efficiency

### Extension Points
- Custom ID generators for specific entity types
- Pluggable validation rules
- Custom migration strategies
- Integration with external ID systems

## Troubleshooting

### Common Issues

#### Migration Failures
- Check database connectivity
- Verify sufficient permissions
- Review batch size settings
- Check available disk space

#### ID Validation Errors
- Ensure proper ID format
- Check for special characters in names
- Verify entity type values
- Review hash generation inputs

#### Performance Issues
- Adjust batch sizes
- Monitor database load
- Check index performance
- Review memory usage

### Debug Tools

```python
# Enable debug logging
import logging
logging.getLogger('morag_graph.utils').setLevel(logging.DEBUG)

# Validate ID format
from morag_graph.utils import IDValidator
try:
    IDValidator.validate_document_id(suspect_id)
except IDValidationError as e:
    print(f"Validation error: {e}")

# Check ID type
from morag_graph.utils import UnifiedIDGenerator
id_type = UnifiedIDGenerator.parse_id_type(suspect_id)
print(f"ID type: {id_type}")
```

## Conclusion

The unified ID architecture provides a robust, scalable, and maintainable foundation for entity identification in the morag-graph system. The deterministic nature ensures reproducibility, while the hierarchical structure supports complex relationships. The comprehensive migration tools and validation mechanisms ensure safe deployment and ongoing data integrity.