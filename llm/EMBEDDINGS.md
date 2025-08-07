# Embedding Operations and Vector Search

## Embedding Configuration

### Primary Embedding Model
**Model**: `text-embedding-004` (Google Gemini)
**Dimensions**: 768
**Provider**: Gemini API
**Task Types**: `retrieval_document`, `retrieval_query`, `semantic_similarity`

```python
# Configuration
embedding_config = {
    "model_name": "text-embedding-004",
    "provider": "gemini",
    "batch_size": 50,
    "max_tokens": 8192,
    "normalize": True,
    "api_key": os.getenv("GEMINI_API_KEY")
}
```

### Task-Specific Embeddings
```python
# Document embedding (for storage)
doc_embedding = await embedding_service.generate_embedding(
    text=document_text,
    task_type="retrieval_document"
)

# Query embedding (for search)
query_embedding = await embedding_service.generate_embedding(
    text=user_query,
    task_type="retrieval_query"
)

# Similarity comparison
similarity_embedding = await embedding_service.generate_embedding(
    text=comparison_text,
    task_type="semantic_similarity"
)
```

## Batch Processing Strategies

### Optimal Batch Sizes
- **Minimum Batch Size**: 50 embeddings
- **Recommended Batch Size**: 100 embeddings
- **Maximum Batch Size**: 200 embeddings (API limits)
- **Delay Between Batches**: 0.05 seconds

### Batch Processing Implementation
```python
async def generate_embeddings_batch(texts, task_type="retrieval_document"):
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Generate embeddings for batch
        batch_embeddings = await embedding_service.generate_batch_embeddings(
            texts=batch,
            task_type=task_type
        )
        
        all_embeddings.extend(batch_embeddings)
        
        # Rate limiting delay
        if i + batch_size < len(texts):
            await asyncio.sleep(0.05)
    
    return all_embeddings
```

### Performance Optimization
```python
# Async processing with semaphore
semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

async def generate_embedding_with_limit(text, task_type):
    async with semaphore:
        return await embedding_service.generate_embedding(text, task_type)

# Parallel processing
tasks = [
    generate_embedding_with_limit(text, "retrieval_document")
    for text in texts
]
embeddings = await asyncio.gather(*tasks)
```

## Vector Search Patterns

### Similarity Search Configuration
```python
search_config = {
    "score_threshold": 0.5,  # Minimum similarity threshold
    "limit": 20,             # Maximum results
    "filters": {             # Optional metadata filters
        "content_type": "document",
        "language": "en"
    }
}
```

### Search Implementation
```python
async def search_similar_content(query, limit=10, score_threshold=0.5):
    # Generate query embedding
    query_embedding = await embedding_service.generate_embedding(
        text=query,
        task_type="retrieval_query"
    )
    
    # Search for similar vectors
    results = await vector_storage.search_similar(
        query_vector=query_embedding,
        limit=limit,
        score_threshold=score_threshold
    )
    
    return results
```

### Hybrid Search (Vector + Graph)
```python
async def hybrid_search(query, config):
    # Vector search
    vector_results = await vector_search(query, config.max_vector_results)
    
    # Graph search
    graph_results = await graph_search(query, config.max_graph_results)
    
    # Fusion strategies
    if config.fusion_strategy == "weighted_combination":
        combined_results = weighted_fusion(
            vector_results, graph_results,
            vector_weight=config.vector_weight,
            graph_weight=config.graph_weight
        )
    elif config.fusion_strategy == "rank_fusion":
        combined_results = rank_fusion(vector_results, graph_results)
    else:  # adaptive
        combined_results = adaptive_fusion(
            vector_results, graph_results, query_analysis
        )
    
    return combined_results
```

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
```python
def calculate_dynamic_threshold(query_complexity, result_count):
    base_threshold = 0.5
    
    # Adjust based on query complexity
    if query_complexity > 0.7:
        base_threshold -= 0.1  # Lower threshold for complex queries
    elif query_complexity < 0.3:
        base_threshold += 0.1  # Higher threshold for simple queries
    
    # Adjust based on result availability
    if result_count < 5:
        base_threshold -= 0.1  # Lower threshold if few results
    elif result_count > 50:
        base_threshold += 0.1  # Higher threshold if many results
    
    return max(0.2, min(0.8, base_threshold))
```

## Entity Embedding Strategies

### Entity Representation
```python
def create_entity_embedding_text(entity):
    # Combine entity information for embedding
    text_parts = [
        f"Entity: {entity.name}",
        f"Type: {entity.type}",
        f"Description: {entity.attributes.get('description', '')}"
    ]
    
    # Add context from related facts
    if entity.related_facts:
        fact_texts = [fact.content for fact in entity.related_facts[:3]]
        text_parts.extend(fact_texts)
    
    return " | ".join(text_parts)
```

### Entity Similarity Search
```python
async def search_similar_entities(query_embedding, limit=10, threshold=0.3):
    # Manual similarity calculation for Neo4j stored embeddings
    query = """
    MATCH (e:Entity)
    WHERE e.embedding IS NOT NULL
    WITH e, gds.similarity.cosine(e.embedding, $query_embedding) AS similarity
    WHERE similarity >= $threshold
    RETURN e.id, e.name, e.type, similarity
    ORDER BY similarity DESC
    LIMIT $limit
    """
    
    results = await neo4j_storage.execute_query(query, {
        "query_embedding": query_embedding,
        "threshold": threshold,
        "limit": limit
    })
    
    return results
```

## Chunking for Embeddings

### Content-Aware Chunking
```python
def create_embedding_chunks(content, content_type, metadata):
    if content_type in ["audio", "video"]:
        # Line-based chunking for transcripts
        chunks = content.split('\n')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    elif content_type == "document":
        # Semantic chunking for documents
        chunks = semantic_chunker.chunk(
            text=content,
            max_chunk_size=4000,
            min_chunk_size=1000
        )
    
    else:
        # Default size-based chunking
        chunks = size_based_chunker.chunk(
            text=content,
            chunk_size=4000,
            overlap=200
        )
    
    return chunks
```

### Chunk Metadata Enhancement
```python
def enhance_chunk_metadata(chunk, chunk_index, document_metadata):
    enhanced_metadata = {
        "chunk_index": chunk_index,
        "chunk_size": len(chunk),
        "document_id": document_metadata["document_id"],
        "content_type": document_metadata["content_type"],
        "source_file": document_metadata["source_file"],
        "language": document_metadata.get("language", "en"),
        "processing_timestamp": datetime.now().isoformat()
    }
    
    # Add content-specific metadata
    if document_metadata["content_type"] == "audio":
        enhanced_metadata.update({
            "speaker_info": extract_speaker_info(chunk),
            "timestamp_range": extract_timestamp_range(chunk)
        })
    elif document_metadata["content_type"] == "document":
        enhanced_metadata.update({
            "page_number": extract_page_number(chunk),
            "section_title": extract_section_title(chunk)
        })
    
    return enhanced_metadata
```

## Vector Storage Operations

### Qdrant Configuration
```python
qdrant_config = {
    "collection_name": os.getenv("MORAG_QDRANT_COLLECTION", "morag_vectors"),
    "vector_size": 768,
    "distance": "Cosine",
    "on_disk_payload": True,
    "optimizers_config": {
        "deleted_threshold": 0.2,
        "vacuum_min_vector_number": 1000,
        "default_segment_number": 0,
        "max_segment_size": None,
        "memmap_threshold": None,
        "indexing_threshold": 20000,
        "flush_interval_sec": 5,
        "max_optimization_threads": 1
    }
}
```

### Storage Operations
```python
async def store_vectors(embeddings, metadata_list, collection_name=None):
    points = []
    for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
        point = {
            "id": metadata.get("id", str(uuid.uuid4())),
            "vector": embedding,
            "payload": metadata
        }
        points.append(point)
    
    # Batch upsert
    await qdrant_client.upsert(
        collection_name=collection_name or default_collection,
        points=points
    )
```

### Search with Filters
```python
async def search_with_filters(query_vector, filters, limit=10):
    search_filter = Filter(
        must=[
            FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
            for key, value in filters.items()
        ]
    )
    
    results = await qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True
    )
    
    return results
```

## Performance Monitoring

### Embedding Generation Metrics
```python
embedding_metrics = {
    "generation_time": "Time to generate embeddings",
    "batch_efficiency": "Embeddings per second in batch",
    "api_call_count": "Number of API calls made",
    "error_rate": "Percentage of failed generations",
    "cache_hit_rate": "Percentage of cached embeddings used"
}
```

### Search Performance Metrics
```python
search_metrics = {
    "search_latency": "Time to complete search",
    "result_relevance": "Average similarity score of results",
    "recall_rate": "Percentage of relevant results found",
    "precision_rate": "Percentage of results that are relevant",
    "throughput": "Searches per second"
}
```
