# Embedding Operations and Vector Search

## Embedding Configuration

### Primary Embedding Model
**Model**: `text-embedding-004` (Google Gemini)
**Dimensions**: 768
**Provider**: Gemini API
**Task Types**: `retrieval_document`, `retrieval_query`, `semantic_similarity`

**Configuration**:
```json
{
  "model_name": "text-embedding-004",
  "provider": "gemini",
  "batch_size": 50,
  "max_tokens": 8192,
  "normalize": true,
  "api_key": "GEMINI_API_KEY"
}
```

### Task-Specific Embeddings
**Document Embedding** (for storage): Use `retrieval_document` task type when generating embeddings for document storage and indexing.

**Query Embedding** (for search): Use `retrieval_query` task type when generating embeddings for user queries and search operations.

**Similarity Comparison**: Use `semantic_similarity` task type when generating embeddings for content comparison and similarity analysis.

## Batch Processing Strategies

### Optimal Batch Sizes
- **Minimum Batch Size**: 50 embeddings
- **Recommended Batch Size**: 100 embeddings
- **Maximum Batch Size**: 200 embeddings (API limits)
- **Delay Between Batches**: 0.05 seconds

### Batch Processing Implementation
**Process**: Generate embeddings in batches to optimize API usage and performance.

**Steps**:
1. **Batch Creation**: Split input texts into batches of specified size (default 100)
2. **Batch Processing**: Generate embeddings for each batch using specified task type
3. **Result Aggregation**: Combine all batch results into single embedding list
4. **Rate Limiting**: Apply delay between batches (0.05 seconds) to respect API limits

### Performance Optimization
**Concurrency Control**: Use semaphore to limit concurrent requests (recommended: 5 concurrent requests).

**Parallel Processing**: Process multiple embedding requests simultaneously while respecting rate limits.

**Implementation**: Create tasks for each text input and use async gathering to process them efficiently.

## Vector Search Patterns

### Similarity Search Configuration
```json
{
  "score_threshold": 0.5,
  "limit": 20,
  "filters": {
    "content_type": "document",
    "language": "en"
  }
}
```

### Search Implementation
**Process**: Generate query embedding and search for similar vectors in storage.

**Steps**:
1. **Query Embedding**: Generate embedding for search query using `retrieval_query` task type
2. **Vector Search**: Search vector storage for similar embeddings within specified limits
3. **Result Filtering**: Apply score threshold and limit to filter results
4. **Result Return**: Return ranked list of similar content

### Hybrid Search (Vector + Graph)
**Approach**: Combine vector similarity search with graph traversal for comprehensive results.

**Process**:
1. **Vector Search**: Perform similarity search using embeddings
2. **Graph Search**: Execute graph traversal for relationship-based results
3. **Fusion Strategy**: Combine results using one of three strategies:
   - **Weighted Combination**: Apply configurable weights to vector and graph results
   - **Rank Fusion**: Merge results based on ranking positions
   - **Adaptive**: Dynamically choose fusion method based on query analysis

## Similarity Thresholds

### Content Type Specific Thresholds
```python
similarity_thresholds = {
    "exact_match": 0.95,      # Near-identical content
    "high_similarity": 0.8,   # Very similar content
    "medium_similarity": 0.6, # Moderately similar content
    "low_similarity": 0.4,    # Loosely related content
    "minimum_relevance": 0.3  # Minimum for consideration
}
```

### Dynamic Threshold Adjustment
**Algorithm**: Adjust similarity threshold based on query complexity and result availability.

**Adjustment Rules**:
- **Complex Queries** (>0.7): Lower threshold by 0.1 to capture more results
- **Simple Queries** (<0.3): Raise threshold by 0.1 for more precise results
- **Few Results** (<5): Lower threshold by 0.1 to find more matches
- **Many Results** (>50): Raise threshold by 0.1 to filter results
- **Bounds**: Keep threshold between 0.2 and 0.8

## Entity Embedding Strategies

### Entity Representation
**Approach**: Combine entity information into comprehensive text for embedding generation.

**Text Components**:
- Entity name and type information
- Description from entity attributes
- Context from related facts (up to 3 most relevant)
- Pipe-separated format for structured representation

### Entity Similarity Search
**Approach**: Search for similar entities using cosine similarity calculation on stored embeddings.

**Process**:
1. **Query Execution**: Find entities with non-null embeddings in Neo4j
2. **Similarity Calculation**: Use cosine similarity between query and entity embeddings
3. **Threshold Filtering**: Return only entities above similarity threshold (default 0.3)
4. **Result Ranking**: Order results by similarity score in descending order
5. **Limit Application**: Return top N results (default 10)

## Chunking for Embeddings

### Content-Aware Chunking
**Strategy**: Apply different chunking approaches based on content type for optimal embedding generation.

**Chunking Methods**:
- **Audio/Video**: Line-based chunking for transcripts, split by newlines and filter empty chunks
- **Documents**: Semantic chunking with 4000 max size, 1000 min size for coherent content blocks
- **Default**: Size-based chunking with 4000 character chunks and 200 character overlap

### Chunk Metadata Enhancement
**Purpose**: Enrich chunk metadata with content-specific information for better retrieval and context.

**Base Metadata**:
- Chunk index, size, document ID, content type, source file
- Language information and processing timestamp

**Content-Specific Enhancements**:
- **Audio Content**: Speaker information and timestamp ranges
- **Document Content**: Page numbers and section titles
- **Other Types**: Additional metadata based on content characteristics

## Vector Storage Operations

### Qdrant Configuration
```json
{
  "collection_name": "morag_vectors",
  "vector_size": 768,
  "distance": "Cosine",
  "on_disk_payload": true,
  "optimizers_config": {
    "deleted_threshold": 0.2,
    "vacuum_min_vector_number": 1000,
    "default_segment_number": 0,
    "max_segment_size": null,
    "memmap_threshold": null,
    "indexing_threshold": 20000,
    "flush_interval_sec": 5,
    "max_optimization_threads": 1
  }
}
```

### Storage Operations
**Process**: Store embeddings with metadata in vector database using batch operations.

**Steps**:
1. **Point Creation**: Create vector points with IDs, embeddings, and metadata payloads
2. **ID Generation**: Use provided IDs or generate UUIDs for new points
3. **Batch Upsert**: Store multiple points efficiently using batch operations
4. **Collection Management**: Use specified collection or default collection name

### Search with Filters
**Approach**: Combine vector similarity search with metadata filtering for precise results.

**Process**:
1. **Filter Construction**: Create field conditions for each filter criteria
2. **Query Execution**: Search vectors with combined similarity and metadata filters
3. **Result Limiting**: Apply limit to control number of returned results
4. **Payload Inclusion**: Return results with full metadata payloads

## Performance Monitoring

### Embedding Generation Metrics
**Key Metrics**:
- **Generation Time**: Time to generate embeddings
- **Batch Efficiency**: Embeddings per second in batch processing
- **API Call Count**: Number of API calls made
- **Error Rate**: Percentage of failed generations
- **Cache Hit Rate**: Percentage of cached embeddings used

### Search Performance Metrics
**Key Metrics**:
- **Search Latency**: Time to complete search operations
- **Result Relevance**: Average similarity score of results
- **Recall Rate**: Percentage of relevant results found
- **Precision Rate**: Percentage of results that are relevant
- **Throughput**: Searches per second
