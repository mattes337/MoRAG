# Task 4: Increase Default Chunk Size for Better Context

## Problem Analysis

### Current Issues
1. **Small default chunk size**: Current default of 1000 characters is too small for meaningful context in modern LLM applications.

2. **Fragmented content**: Small chunks break up coherent content unnecessarily, reducing retrieval quality.

3. **Poor context preservation**: Important relationships between sentences and paragraphs are lost with small chunks.

4. **Suboptimal for embeddings**: Modern embedding models can handle larger contexts and perform better with more complete content.

### Current Implementation Analysis
From multiple files showing 1000 character default:
- `packages/morag/src/morag/ingest_tasks.py` line 91: `chunk_size = 1000`
- `packages/morag-document/src/morag_document/processor.py` line 110: `chunk_size=config.chunk_size or 1000`
- `packages/morag-document/src/morag_document/service.py` line 202: `chunk_size = kwargs.get("chunk_size", 1000)`
- `packages/morag-document/src/morag_document/converters/base.py` line 250: `chunk_size = options.chunk_size or 1000`

### Impact of Small Chunks
1. **Reduced semantic coherence**: Important context is split across multiple chunks
2. **Increased vector storage overhead**: More chunks mean more vectors to store and search
3. **Fragmented search results**: Users get incomplete context in search results
4. **Poor retrieval quality**: Related information is scattered across multiple chunks

## Solution Approach

### 1. Increase Default Chunk Size
- Change default from 1000 to 4000 characters
- Ensure this works well with embedding model token limits
- Maintain configurability for different use cases

### 2. Environment Variable Configuration
- Add `DEFAULT_CHUNK_SIZE` environment variable
- Allow runtime configuration without code changes
- Provide clear documentation on optimal chunk sizes

### 3. Validate Token Limits
- Ensure 4000 characters stay within embedding model limits
- Add validation for chunk size vs model capabilities
- Provide warnings for oversized chunks

### 4. Backward Compatibility
- Maintain ability to use smaller chunk sizes
- Ensure existing configurations continue to work
- Provide migration guidance for existing deployments

## Implementation Plan

### Phase 1: Configuration Infrastructure
1. **Add environment variable support**
   - Add `DEFAULT_CHUNK_SIZE` to core configuration
   - Update all default chunk size references
   - Add validation for reasonable chunk sizes

2. **Update configuration classes**
   - Add chunk size to Settings class
   - Provide validation and bounds checking
   - Add documentation for optimal values

### Phase 2: Update Default Values
1. **Change hardcoded defaults**
   - Update all files with 1000 character defaults
   - Change to use configuration-driven defaults
   - Ensure consistency across all components

2. **Add token limit validation**
   - Calculate approximate token count for chunks
   - Warn when chunks might exceed model limits
   - Provide guidance on optimal chunk sizes

### Phase 3: Documentation and Testing
1. **Update documentation**
   - Document new default chunk size
   - Provide guidance on chunk size selection
   - Add examples for different use cases

2. **Comprehensive testing**
   - Test with various chunk sizes
   - Validate embedding model compatibility
   - Test retrieval quality with larger chunks

### Phase 4: Migration Support
1. **Migration guidance**
   - Provide guidance for existing deployments
   - Document impact of changing chunk sizes
   - Offer tools for re-chunking existing content

## Technical Implementation

### Configuration Updates
```python
# packages/morag-core/src/morag_core/config.py
class Settings(BaseSettings):
    # Document processing settings
    default_chunk_size: int = Field(
        default=4000,
        ge=500,  # Minimum 500 characters
        le=16000,  # Maximum 16000 characters (safe for most models)
        description="Default chunk size for document processing"
    )
    
    default_chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Default overlap between chunks"
    )
    
    # Token limit validation
    max_tokens_per_chunk: int = Field(
        default=8000,
        description="Maximum tokens per chunk for embedding models"
    )
    
    # Environment variable mapping
    class Config:
        env_prefix = "MORAG_"
        env_file = ".env"
```

### Chunk Size Validation
```python
def validate_chunk_size(chunk_size: int, content: str) -> tuple[bool, str]:
    """Validate chunk size against content and model limits."""
    
    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(content) // 4
    
    if estimated_tokens > settings.max_tokens_per_chunk:
        return False, f"Chunk too large: ~{estimated_tokens} tokens (max: {settings.max_tokens_per_chunk})"
    
    if chunk_size < 500:
        return False, "Chunk size too small: minimum 500 characters recommended"
    
    if chunk_size > 16000:
        return False, "Chunk size too large: maximum 16000 characters recommended"
    
    return True, "Chunk size valid"
```

### Updated Ingestion Task
```python
# packages/morag/src/morag/ingest_tasks.py
async def ingest_file_task(file_path: str, task_options: Dict[str, Any] = None):
    """Process and ingest a file with configurable chunk size."""
    
    # Get chunk size from options or use configured default
    chunk_size = task_options.get('chunk_size', settings.default_chunk_size)
    chunk_overlap = task_options.get('chunk_overlap', settings.default_chunk_overlap)
    
    # Validate chunk size
    if chunk_size < 500 or chunk_size > 16000:
        logger.warning(
            "Chunk size outside recommended range",
            chunk_size=chunk_size,
            recommended_range="500-16000"
        )
    
    # Create chunks with configured size
    chunks = []
    if len(content) <= chunk_size:
        chunks = [content]
    else:
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                # Validate chunk before processing
                is_valid, message = validate_chunk_size(chunk_size, chunk)
                if not is_valid:
                    logger.warning("Chunk validation warning", message=message)
                
                chunks.append(chunk)
```

### Environment Configuration
```env
# .env.example
# Document Processing Configuration
MORAG_DEFAULT_CHUNK_SIZE=4000
MORAG_DEFAULT_CHUNK_OVERLAP=200
MORAG_MAX_TOKENS_PER_CHUNK=8000

# Chunk size recommendations:
# - 1000-2000: Fine-grained retrieval, more precise matching
# - 4000-6000: Balanced context and precision (recommended)
# - 8000-12000: Maximum context, fewer chunks
```

## Files to Modify

### Core Configuration
1. `packages/morag-core/src/morag_core/config.py`
   - Add chunk size configuration settings
   - Add validation and bounds checking
   - Add token limit configuration

### Ingestion and Processing
1. `packages/morag/src/morag/ingest_tasks.py`
   - Update default chunk size from 1000 to use configuration
   - Add chunk size validation
   - Add logging for chunk size decisions

2. `packages/morag-document/src/morag_document/processor.py`
   - Update default chunk size reference
   - Use configuration-driven defaults
   - Add validation

3. `packages/morag-document/src/morag_document/service.py`
   - Update default chunk size in text processing
   - Use configuration settings
   - Add validation

4. `packages/morag-document/src/morag_document/converters/base.py`
   - Update chunking logic to use configuration
   - Add chunk size validation
   - Improve logging

### API and Server
1. `packages/morag/src/morag/server.py`
   - Update API endpoints to use configurable defaults
   - Add chunk size validation in request processing
   - Add documentation for chunk size parameters

### Configuration Files
1. `.env.example`
   - Add chunk size configuration examples
   - Document recommended values
   - Add usage guidance

2. `.env.prod.example`
   - Add production-optimized chunk size settings
   - Include performance considerations
   - Add monitoring recommendations

## Testing Strategy

### Chunk Size Validation Tests
1. **Configuration tests**
   - Test environment variable loading
   - Validate bounds checking
   - Test default value handling

2. **Chunk size validation tests**
   - Test with various chunk sizes
   - Validate token limit checking
   - Test warning generation

### Processing Tests
1. **Document processing tests**
   - Test with different chunk sizes
   - Validate chunk quality and coherence
   - Test with various document types

2. **Embedding compatibility tests**
   - Test chunk sizes with embedding models
   - Validate token limits
   - Test embedding quality

### Performance Tests
1. **Retrieval quality tests**
   - Compare search results with different chunk sizes
   - Measure context preservation
   - Test semantic coherence

2. **Storage efficiency tests**
   - Compare storage requirements
   - Measure search performance
   - Test with large documents

## Success Criteria

### Configuration Goals
1. ✅ Default chunk size increased to 4000 characters
2. ✅ Configurable via environment variables
3. ✅ Proper validation and bounds checking
4. ✅ Backward compatibility maintained

### Quality Improvements
1. ✅ Better context preservation in chunks
2. ✅ Improved search result coherence
3. ✅ Reduced chunk fragmentation
4. ✅ Better semantic relationships preserved

### Performance Goals
1. ✅ Reduced number of chunks for same content
2. ✅ Improved retrieval quality
3. ✅ Better embedding utilization
4. ✅ Maintained processing performance

## Embedding Model Compatibility

### Token Limit Analysis
- **Gemini text-embedding-004**: ~8192 tokens max
- **4000 characters**: ~1000 tokens (safe)
- **8000 characters**: ~2000 tokens (safe)
- **16000 characters**: ~4000 tokens (safe with margin)

### Recommended Chunk Sizes
- **Small documents**: 2000-4000 characters
- **Medium documents**: 4000-8000 characters
- **Large documents**: 6000-12000 characters
- **Maximum safe**: 16000 characters

## Configuration Examples

### Development Configuration
```env
# Development - smaller chunks for testing
MORAG_DEFAULT_CHUNK_SIZE=2000
MORAG_DEFAULT_CHUNK_OVERLAP=200
```

### Production Configuration
```env
# Production - optimized for quality
MORAG_DEFAULT_CHUNK_SIZE=4000
MORAG_DEFAULT_CHUNK_OVERLAP=400
```

### High-Context Configuration
```env
# Maximum context preservation
MORAG_DEFAULT_CHUNK_SIZE=8000
MORAG_DEFAULT_CHUNK_OVERLAP=800
```

## Risk Mitigation

### Compatibility Risks
1. **Existing deployments**: Changed defaults might affect existing systems
   - Mitigation: Environment variable override capability
   - Documentation: Clear migration guidance

2. **Model token limits**: Larger chunks might exceed model limits
   - Mitigation: Token limit validation and warnings
   - Fallback: Automatic chunk splitting for oversized content

### Performance Risks
1. **Increased memory usage**: Larger chunks use more memory
   - Mitigation: Monitor memory usage and provide guidance
   - Configuration: Allow tuning based on system resources

2. **Processing time**: Larger chunks might take longer to process
   - Mitigation: Performance testing and optimization
   - Monitoring: Track processing time metrics

## Implementation Timeline

### Week 1: Configuration Infrastructure
- [ ] Add chunk size configuration to core settings
- [ ] Implement validation and bounds checking
- [ ] Update environment variable handling

### Week 2: Update Defaults
- [ ] Update all hardcoded chunk size defaults
- [ ] Implement configuration-driven defaults
- [ ] Add token limit validation

### Week 3: Testing and Validation
- [ ] Test with various chunk sizes
- [ ] Validate embedding model compatibility
- [ ] Performance testing and optimization

### Week 4: Documentation and Migration
- [ ] Update documentation and examples
- [ ] Create migration guidance
- [ ] Final testing and deployment

## Next Steps

1. **Add configuration infrastructure**: Start with core settings and environment variables
2. **Update default values**: Change hardcoded defaults to use configuration
3. **Add validation**: Implement chunk size and token limit validation
4. **Test thoroughly**: Validate with various content types and sizes

This task focuses on improving document chunking quality through larger, more contextual chunks while maintaining flexibility and compatibility.
