# MoRAG Implementation Guide

## Quick Start

### Prerequisites
1. **Development Environment**
   ```bash
   # Clone repository
   git clone https://github.com/mattes337/morag
   cd morag
   
   # Install dependencies
   pip install -e packages/morag-core
   pip install -e packages/morag-graph
   pip install -e packages/morag-reasoning
   pip install -e packages/morag-services
   pip install -e packages/morag
   ```

2. **External Dependencies**
   ```bash
   # SpaCy language models
   python -m spacy download en_core_web_lg
   python -m spacy download de_core_news_lg
   python -m spacy download es_core_news_lg
   
   # Neo4j (Docker recommended)
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.15
   
   # Qdrant (Docker recommended)
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Environment Configuration**
   ```bash
   # Create .env file
   GEMINI_API_KEY=your_gemini_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   QDRANT_URL=http://localhost:6333
   ```

### Implementation Order

#### Phase 1: Foundation (Weeks 1-2)
Start with the core entity and relation extraction improvements:

1. **Begin with Task 1: SpaCy NER Integration**
   ```bash
   # Create the SpaCy extractor
   touch packages/morag-graph/src/morag_graph/extraction/spacy_extractor.py
   
   # Run existing tests to ensure no regressions
   pytest packages/morag-graph/tests/
   ```

2. **Continue with Task 2: OpenIE Pipeline Enhancement**
   ```bash
   # Enhance existing OpenIE components
   # Focus on entity linking improvements
   ```

#### Phase 2: Intelligence (Weeks 3-4)
Build the intelligent traversal and fact gathering:

3. **Implement Task 3: Recursive Resolution**
   ```bash
   # Create traversal components
   mkdir -p packages/morag-graph/src/morag_graph/traversal
   mkdir -p packages/morag-graph/src/morag_graph/discovery
   ```

4. **Build Task 4: Fact Gathering System**
   ```bash
   # Enhance reasoning components
   # Focus on fact extraction and scoring
   ```

#### Phase 3: Orchestration (Weeks 5-6)
Complete the pipeline and response generation:

5. **Create Task 5: Pipeline Orchestration**
   ```bash
   # Build unified pipeline
   mkdir -p packages/morag/src/morag/agents
   mkdir -p packages/morag/src/morag/pipeline
   ```

6. **Finish with Task 6: Response Generation**
   ```bash
   # Complete the response system
   # Focus on citation integration
   ```

## Development Workflow

### 1. Task Setup
For each task:
```bash
# Create feature branch
git checkout -b feature/task-X-description

# Create necessary directories
mkdir -p packages/package-name/src/package/new-module

# Create test directories
mkdir -p packages/package-name/tests/new-module
```

### 2. Implementation Pattern
Follow this pattern for each component:

```python
# 1. Create base class/interface
class BaseComponent:
    async def process(self, input_data: InputType) -> OutputType:
        raise NotImplementedError

# 2. Implement concrete class
class ConcreteComponent(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def process(self, input_data: InputType) -> OutputType:
        # Implementation with proper error handling
        try:
            result = await self._process_internal(input_data)
            self.logger.info("Processing completed", result_size=len(result))
            return result
        except Exception as e:
            self.logger.error("Processing failed", error=str(e))
            raise

# 3. Create comprehensive tests
class TestConcreteComponent:
    @pytest.mark.asyncio
    async def test_process_success(self):
        component = ConcreteComponent(test_config)
        result = await component.process(test_input)
        assert result is not None
```

### 3. Testing Strategy
For each component:

```bash
# Unit tests
pytest packages/package-name/tests/test_component.py -v

# Integration tests
pytest packages/package-name/tests/integration/ -v

# Performance tests
pytest packages/package-name/tests/performance/ -v --benchmark-only
```

### 4. Documentation
Update documentation for each component:

```python
# Docstring format
async def process_entities(self, text: str, language: Optional[str] = None) -> List[Entity]:
    """Extract and normalize entities from text.
    
    Args:
        text: Input text to process
        language: Optional language code (auto-detected if None)
        
    Returns:
        List of normalized entities with confidence scores
        
    Raises:
        ProcessingError: If entity extraction fails
        ValidationError: If input text is invalid
        
    Example:
        >>> extractor = SpacyEntityExtractor()
        >>> entities = await extractor.process_entities("Einstein was a physicist")
        >>> print(entities[0].name)  # "Einstein"
    """
```

## Testing Guidelines

### Unit Test Structure
```python
import pytest
from unittest.mock import AsyncMock, patch
from your_module import YourClass

class TestYourClass:
    @pytest.fixture
    def mock_config(self):
        return YourClassConfig(param1="value1", param2="value2")
    
    @pytest.fixture
    def your_instance(self, mock_config):
        return YourClass(mock_config)
    
    @pytest.mark.asyncio
    async def test_success_case(self, your_instance):
        # Test successful operation
        result = await your_instance.method(valid_input)
        assert result.success is True
        assert len(result.data) > 0
    
    @pytest.mark.asyncio
    async def test_error_case(self, your_instance):
        # Test error handling
        with pytest.raises(ExpectedError):
            await your_instance.method(invalid_input)
    
    @pytest.mark.asyncio
    async def test_edge_case(self, your_instance):
        # Test edge cases
        result = await your_instance.method(edge_case_input)
        assert result is not None
```

### Integration Test Structure
```python
@pytest.mark.integration
class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(self):
        # Test complete ingestion pipeline
        pipeline = MoRAGPipelineAgent(test_config)
        result = await pipeline.process_ingestion(test_document)
        
        # Verify all stages completed
        assert result.conversion_success is True
        assert len(result.entities) > 0
        assert len(result.relations) > 0
        assert result.graph_storage_success is True
    
    @pytest.mark.asyncio
    async def test_end_to_end_resolution(self):
        # Test complete resolution pipeline
        pipeline = MoRAGPipelineAgent(test_config)
        result = await pipeline.process_resolution(test_query)
        
        # Verify response quality
        assert result.response is not None
        assert len(result.citations) > 0
        assert result.quality_score > 0.8
```

## Performance Optimization

### Memory Management
```python
# Use generators for large datasets
async def process_large_dataset(self, items: List[Item]) -> AsyncGenerator[Result, None]:
    for batch in self._batch_items(items, batch_size=100):
        results = await self._process_batch(batch)
        for result in results:
            yield result

# Clean up resources
async def __aenter__(self):
    await self.initialize()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.cleanup()
```

### Caching Strategy
```python
from functools import lru_cache
from typing import Dict, Any

class CachedProcessor:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    @lru_cache(maxsize=1000)
    def _expensive_computation(self, input_hash: str) -> Result:
        # Expensive computation here
        pass
    
    async def process_with_cache(self, input_data: InputType) -> Result:
        cache_key = self._generate_cache_key(input_data)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = await self._process(input_data)
        self._cache[cache_key] = result
        return result
```

## Debugging and Monitoring

### Logging Configuration
```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Use in components
logger = structlog.get_logger(__name__)
logger.info("Processing started", document_id=doc_id, stage="entity_extraction")
```

### Performance Monitoring
```python
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def monitor_performance(operation_name: str):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info("Operation completed", 
                   operation=operation_name, 
                   duration_seconds=duration)
```

## Common Patterns

### Error Handling
```python
from morag_core.exceptions import ProcessingError, ValidationError

async def robust_processing(self, input_data: InputType) -> OutputType:
    try:
        # Validate input
        self._validate_input(input_data)
        
        # Process with retries
        for attempt in range(self.max_retries):
            try:
                return await self._process_internal(input_data)
            except TemporaryError as e:
                if attempt == self.max_retries - 1:
                    raise ProcessingError(f"Failed after {self.max_retries} attempts") from e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
    except ValidationError:
        # Don't retry validation errors
        raise
    except Exception as e:
        # Log and re-raise as ProcessingError
        logger.error("Unexpected error", error=str(e), error_type=type(e).__name__)
        raise ProcessingError(f"Processing failed: {e}") from e
```

### Configuration Management
```python
from pydantic import BaseModel, Field
from typing import Optional

class ComponentConfig(BaseModel):
    """Configuration for component."""
    
    # Required parameters
    api_key: str = Field(..., description="API key for external service")
    
    # Optional parameters with defaults
    timeout: float = Field(30.0, description="Timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")
    batch_size: int = Field(100, description="Batch size for processing")
    
    # Validation
    @validator('timeout')
    def timeout_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v
```

This implementation guide provides the foundation for successfully implementing all MoRAG high-level tasks. Follow the patterns and guidelines to ensure consistent, maintainable, and well-tested code.
