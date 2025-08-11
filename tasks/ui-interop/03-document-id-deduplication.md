# Document ID and Deduplication System

## Overview
Implement document ID-based deduplication system that allows UI applications to control document identity and prevent duplicate processing.

## Requirements

### Document ID Handling
- Accept optional document ID in request body or as form parameter
- Support string-valued document IDs provided by client applications
- Generate UUID automatically if no ID provided
- Validate document ID format and length constraints

### Database Integration
- Store document ID in all database records:
  - Neo4j nodes (DocumentChunk, Entity, etc.)
  - Qdrant points metadata
  - Any other persistent storage
- Use document ID as primary key for deduplication checks
- Replace existing filename/checksum-based deduplication logic

### Deduplication Logic
- Check for existing documents with same ID before processing
- Return appropriate response for duplicate document attempts
- Provide options for:
  - Reject duplicates with error
  - Update existing document
  - Version existing document
- Ensure all existing deduplication logic uses the document ID

## Implementation Details

### API Changes
Both endpoints (`/api/convert/markdown` and `/api/process/ingest`) should accept:
```json
{
  "document_id": "user-provided-string-id",
  "file": "uploaded-file",
  "webhook_url": "https://example.com/webhook",
  // other parameters
}
```

### Database Schema Updates
- Add `document_id` field to all relevant database entities
- Create indexes on `document_id` for efficient lookups
- Migrate existing records to use generated UUIDs

### Deduplication Response Format
```json
{
  "error": "duplicate_document",
  "message": "Document with ID 'user-doc-123' already exists",
  "existing_document": {
    "document_id": "user-doc-123",
    "created_at": "2024-01-15T10:30:00Z",
    "status": "completed|processing|failed",
    "facts_count": 156,
    "keywords_count": 89
  },
  "options": {
    "update_url": "/api/process/update/user-doc-123",
    "version_url": "/api/process/version/user-doc-123"
  }
}
```

### Configuration Options
- Enable/disable strict deduplication
- Configure deduplication behavior (reject/update/version)
- Set document ID format validation rules
- Configure retention policies for document IDs

## Migration Strategy

### Phase 1: Add Document ID Support
- Add document_id fields to database schemas
- Update API endpoints to accept document_id parameter
- Generate UUIDs for existing documents

### Phase 2: Update Deduplication Logic
- Replace filename/checksum checks with document_id checks
- Implement new deduplication responses
- Add configuration options

### Phase 3: Remove Legacy Code
- Remove old deduplication logic
- Clean up unused checksum fields
- Update documentation

## Testing Requirements
- Unit tests for document ID validation
- Integration tests for deduplication scenarios
- Database migration tests
- API endpoint tests with and without document IDs
- Performance tests for document ID lookups

## Security Considerations
- Validate document ID format to prevent injection attacks
- Implement rate limiting per document ID
- Consider document ID namespace isolation for multi-tenant scenarios
- Audit logging for document ID operations

## Dependencies
- Database migration tools
- Existing MoRAG database schemas
- UUID generation libraries
- API framework updates
