# Document Structure Refactoring Summary

## Overview

This document summarizes the major refactoring of the morag-graph package to implement a new document-centric structure. The refactoring introduces proper separation between documents, document chunks, and entities, enabling better organization and more sophisticated graph queries.

## Key Changes

### 1. New Model Classes

#### Document Model (`src/morag_graph/models/document.py`)
- **Purpose**: Represents source documents with metadata
- **Key Fields**:
  - `id`: Unique identifier
  - `source_file`: Path to source file
  - `file_name`: Name of the source file
  - `file_size`: Size in bytes
  - `checksum`: SHA256 checksum for deduplication
  - `mime_type`: File type
  - `ingestion_timestamp`: When ingested
  - `metadata`: Additional document metadata

#### DocumentChunk Model (`src/morag_graph/models/document_chunk.py`)
- **Purpose**: Represents chunks/sections of documents containing actual text
- **Key Fields**:
  - `id`: Unique identifier
  - `document_id`: Reference to parent document
  - `chunk_index`: Position within document
  - `text`: Actual text content
  - `start_position`/`end_position`: Character positions
  - `chunk_type`: Type of chunk (e.g., "section", "paragraph")
  - `metadata`: Additional chunk metadata

### 2. Updated Relationship Types

#### New RelationTypes in `src/morag_graph/models/types.py`:
- `CONTAINS`: Document -> DocumentChunk relationships
- `MENTIONS`: DocumentChunk -> Entity relationships

### 3. Entity and Relation Model Updates

#### Removed Fields from Entity Model (`src/morag_graph/models/entity.py`):
- `source_text`: Now handled by DocumentChunk
- `source_doc_id`: Now handled by Document -> Chunk -> Entity relationships

#### Removed Fields from Relation Model (`src/morag_graph/models/relation.py`):
- `source_text`: Now handled by DocumentChunk
- `source_doc_id`: Now handled by Document -> Chunk -> Entity relationships

### 4. Neo4j Storage Layer Updates

#### New Methods in `src/morag_graph/storage/neo4j_storage.py`:
- `store_document(document: Document)`: Store document nodes
- `store_document_chunk(chunk: DocumentChunk)`: Store document chunk nodes
- `create_document_contains_chunk_relation(document_id, chunk_id)`: Create CONTAINS relationships
- `create_chunk_mentions_entity_relation(chunk_id, entity_id)`: Create MENTIONS relationships

### 5. File Ingestion Process Refactoring

#### Updated `src/morag_graph/ingestion/file_ingestion.py`:
- **New Process Flow**:
  1. Create Document node with file metadata
  2. Group entities by source text to create DocumentChunks
  3. Store Document and DocumentChunk nodes
  4. Create Document -> CONTAINS -> DocumentChunk relationships
  5. Create DocumentChunk -> MENTIONS -> Entity relationships
  6. Store entities and relations without document-specific fields

- **New Helper Method**:
  - `_group_entities_into_chunks()`: Groups entities by source text for chunk creation

### 6. Model Registration Updates

#### Updated `src/morag_graph/models/__init__.py`:
- Added imports for `Document` and `DocumentChunk`
- Updated `__all__` list to include new models

## New Graph Structure

### Before Refactoring:
```
Entity -> RELATION -> Entity
(with source_text and source_doc_id fields)
```

### After Refactoring:
```
Document -> CONTAINS -> DocumentChunk -> MENTIONS -> Entity
                                     \-> MENTIONS -> Entity
                                     
Entity -> RELATION -> Entity
(without document-specific fields)
```

## Benefits of New Structure

### 1. **Proper Separation of Concerns**
- Document metadata is separated from entity data
- Entities are now global and can be shared across documents
- Text context is preserved through DocumentChunk nodes

### 2. **Enhanced Query Capabilities**
- Find all entities mentioned in a specific document
- Discover entities that co-occur in the same chunks
- Trace entity mentions back to their source documents
- Enable document-to-document connections through shared entities

### 3. **Better Scalability**
- Entities can be referenced by multiple documents without duplication
- Document-level operations (e.g., deletion) are more efficient
- Supports incremental updates and version tracking

### 4. **Improved Data Integrity**
- Clear hierarchical structure: Document -> Chunk -> Entity
- Consistent relationship types throughout the graph
- Better support for document deduplication via checksums

## Testing and Validation

### Mock Test Results
The `test_document_structure_mock.py` script demonstrates the new structure:

- **Node Counts**: 1 Document, multiple DocumentChunks, multiple Entities
- **Relationship Increase**: ~106% increase in total relationships due to new CONTAINS and MENTIONS relationships
- **Query Demonstrations**:
  - Entities in document traversal
  - Co-occurring entity detection
  - Explicit relation mapping

### Test Files Created:
- `test_document_structure_mock.py`: Mock test without Neo4j dependency
- `test_new_document_structure.py`: Full Neo4j integration test (requires running Neo4j)

## Migration Considerations

### For Existing Data:
1. **Backup**: Ensure existing graph data is backed up
2. **Migration Script**: A migration script would be needed to:
   - Extract document metadata from existing entities
   - Create Document and DocumentChunk nodes
   - Establish new relationships
   - Remove old document-specific fields from entities

### For Applications Using the API:
1. **Entity Queries**: Update queries to traverse through DocumentChunk relationships
2. **Document Operations**: Use new Document-level operations for file management
3. **Context Retrieval**: Use DocumentChunk nodes to get text context for entities

## Files Modified/Created

### New Files:
- `src/morag_graph/models/document.py`
- `src/morag_graph/models/document_chunk.py`
- `test_document_structure_mock.py`
- `test_new_document_structure.py`
- `DOCUMENT_STRUCTURE_REFACTORING_SUMMARY.md`

### Modified Files:
- `src/morag_graph/models/types.py`
- `src/morag_graph/models/__init__.py`
- `src/morag_graph/models/entity.py`
- `src/morag_graph/models/relation.py`
- `src/morag_graph/storage/neo4j_storage.py`
- `src/morag_graph/ingestion/file_ingestion.py`

## Next Steps

1. **Production Testing**: Test with real Neo4j database
2. **Migration Script**: Create script to migrate existing data
3. **API Updates**: Update any client applications to use new structure
4. **Documentation**: Update API documentation and examples
5. **Performance Testing**: Validate query performance with new structure

## Conclusion

This refactoring significantly improves the morag-graph package's architecture by introducing proper document management, better entity organization, and more sophisticated relationship modeling. The new structure provides a solid foundation for advanced knowledge graph operations while maintaining backward compatibility where possible.