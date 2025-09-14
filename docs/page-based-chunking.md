# Page-Based Chunking for Document Processing

## Overview

The MoRAG system now supports page-based chunking for PDF documents, which groups content by page rather than creating individual vector points for each sentence or paragraph. This approach provides better context for RAG operations while reducing the number of vector points in the database.

## Problem Solved

Previously, the document indexing system created one vector point for each sentence or small text element, resulting in:
- Too many fine-grained chunks that lacked sufficient context
- Increased storage requirements in the vector database
- Potential loss of document structure and page-level context
- Suboptimal retrieval performance for questions requiring broader context

## Solution

Page-based chunking groups all content from a single page into one or more chunks, preserving:
- Page-level context and structure
- Relationships between elements on the same page
- Better semantic coherence for RAG operations
- Reduced number of vector points while maintaining searchability

## Configuration

### Environment Variables

Add these settings to your `.env` file:

```env
# Document Processing Configuration
DEFAULT_CHUNKING_STRATEGY=page
ENABLE_PAGE_BASED_CHUNKING=true
MAX_PAGE_CHUNK_SIZE=8000
```

### Configuration Options

- **DEFAULT_CHUNKING_STRATEGY**: Sets the default chunking strategy for document processing
  - Options: `page`, `semantic`, `sentence`, `paragraph`, `simple`
  - Default: `page`

- **ENABLE_PAGE_BASED_CHUNKING**: Enables or disables page-based chunking
  - Default: `true`

- **MAX_PAGE_CHUNK_SIZE**: Maximum size (in characters) for a page-based chunk
  - Default: `8000`
  - If a page exceeds this size, it will be split into smaller chunks while preserving page context

## How It Works

1. **Document Parsing**: Documents are first parsed using docling or unstructured.io to extract individual elements
2. **Page Grouping**: Elements are grouped by their page number
3. **Content Combination**: All text content from each page is combined into a single chunk
4. **Size Management**: If a page's content exceeds `MAX_PAGE_CHUNK_SIZE`, it's intelligently split:
   - First by paragraphs (double newlines)
   - Then by sentences if needed
   - Maintains page context metadata

## Benefits

### For RAG Performance
- **Better Context**: Larger chunks provide more context for LLM reasoning
- **Improved Retrieval**: Page-level chunks are more likely to contain complete thoughts and concepts
- **Reduced Noise**: Fewer, more meaningful chunks reduce retrieval noise

### For Storage Efficiency
- **Fewer Vector Points**: Significantly reduces the number of vectors stored in Qdrant
- **Lower Costs**: Reduced embedding generation and storage costs
- **Faster Queries**: Fewer vectors to search through

### For Document Structure
- **Preserved Layout**: Maintains the logical structure of documents
- **Page Awareness**: Retains information about which page content came from
- **Element Relationships**: Preserves relationships between elements on the same page

## Metadata

Page-based chunks include rich metadata:

```json
{
  "page_based_chunking": true,
  "original_chunks_count": 5,
  "original_chunk_types": ["text", "title", "list"],
  "page_number": 1,
  "chunk_type": "page"
}
```

For split pages:
```json
{
  "page_based_chunking": true,
  "chunk_index_on_page": 0,
  "is_partial_page": true,
  "page_number": 1
}
```

## Backward Compatibility

The system maintains backward compatibility with existing chunking strategies:
- `semantic`: Advanced semantic chunking using NLP
- `sentence`: Sentence-based chunking
- `paragraph`: Paragraph-based chunking  
- `simple`: Basic character-based chunking

You can override the default strategy per document by specifying the `chunking_strategy` parameter.

## Testing

The implementation includes comprehensive tests covering:
- Basic page-based chunking functionality
- Large page handling and splitting
- Empty document handling
- Configuration validation
- Integration with document processing pipeline

Run tests with:
```bash
python -m pytest tests/test_page_based_chunking.py -v
```

## Migration

Existing documents will continue to work with their current chunking. To apply page-based chunking to existing documents, you would need to reprocess them with the new configuration.

## Performance Considerations

- **Memory Usage**: Page-based chunks use more memory per chunk but fewer total chunks
- **Processing Time**: Slightly increased processing time for grouping and combining content
- **Embedding Costs**: Reduced total embedding generation due to fewer chunks
- **Query Performance**: Generally improved due to fewer vectors to search

## Troubleshooting

### Large Pages
If you encounter memory issues with very large pages, reduce `MAX_PAGE_CHUNK_SIZE`:
```env
MAX_PAGE_CHUNK_SIZE=4000
```

### Too Many Small Chunks
If pages are being split too aggressively, increase the chunk size:
```env
MAX_PAGE_CHUNK_SIZE=12000
```

### Disable Page-Based Chunking
To revert to the previous behavior:
```env
DEFAULT_CHUNKING_STRATEGY=semantic
ENABLE_PAGE_BASED_CHUNKING=false
```
