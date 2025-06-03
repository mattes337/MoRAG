# Task 23: Abstract LLM and Embedding Provider APIs

## Objective
Create a unified abstraction layer for LLM and embedding providers to support multiple providers beyond Gemini, with fallback mechanisms and provider switching capabilities.

## Research Phase

### Libraries to Research
1. **LiteLLM** - Unified API for 100+ LLMs (OpenAI, Anthropic, Cohere, etc.)
   - Pros: Extensive provider support, consistent interface, built-in retry logic
   - Cons: Additional dependency, potential overhead
   
2. **LangChain** - Comprehensive framework with provider abstractions
   - Pros: Rich ecosystem, well-documented, extensive integrations
   - Cons: Heavy framework, potential overkill for simple abstraction
   
3. **Haystack** - NLP framework with provider abstractions
   - Pros: Focus on document processing, good for RAG systems
   - Cons: Learning curve, framework lock-in
   
4. **Custom Abstraction** - Build our own lightweight abstraction
   - Pros: Full control, minimal dependencies, tailored to needs
   - Cons: More development time, maintenance burden

### Provider Support Requirements
- **Primary**: Gemini (Google AI), OpenAI GPT models
- **Secondary**: Anthropic Claude, Cohere, Azure OpenAI
- **Local**: Ollama, Hugging Face Transformers
- **Embedding**: text-embedding-004, OpenAI embeddings, Cohere embeddings

## Implementation Strategy

### Phase 1: Core Abstraction Layer
```python
# Abstract base classes for providers
class LLMProvider(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str
    
    @abstractmethod
    async def generate_summary(self, text: str, **kwargs) -> str
    
    @abstractmethod
    def get_provider_info(self) -> ProviderInfo

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]
    
    @abstractmethod
    def get_embedding_dimension(self) -> int
```

### Phase 2: Provider Implementations
- **GeminiProvider**: Current implementation wrapped in abstraction
- **OpenAIProvider**: GPT-4, GPT-3.5-turbo support
- **AnthropicProvider**: Claude models support
- **OllamaProvider**: Local model support

### Phase 3: Provider Manager
```python
class ProviderManager:
    def __init__(self, config: ProviderConfig):
        self.providers = {}
        self.fallback_chain = []
        self.circuit_breakers = {}
    
    async def get_llm_response(self, prompt: str, provider: str = None) -> str:
        # Try specified provider or use fallback chain
        
    async def get_embeddings(self, text: str, provider: str = None) -> List[float]:
        # Try specified provider or use fallback chain
```

## Architecture Design

### Configuration Structure
```yaml
providers:
  llm:
    primary: "gemini"
    fallback: ["openai", "anthropic"]
    
  embedding:
    primary: "gemini"
    fallback: ["openai"]
    
gemini:
  api_key: "${GEMINI_API_KEY}"
  model: "gemini-1.5-pro"
  embedding_model: "text-embedding-004"
  
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  embedding_model: "text-embedding-3-large"
```

### Error Handling Strategy
1. **Circuit Breaker Pattern**: Prevent cascading failures
2. **Exponential Backoff**: Handle rate limits gracefully
3. **Provider Health Checks**: Monitor provider availability
4. **Graceful Degradation**: Fall back to simpler models when needed

## Integration Points

### Current MoRAG Integration
- Replace direct Gemini calls in `services/gemini_service.py`
- Update document processing tasks to use provider manager
- Modify embedding generation in vector storage
- Update summarization service to use abstracted LLM calls

### Database Schema Updates
```sql
-- Track which provider was used for each operation
ALTER TABLE documents ADD COLUMN llm_provider VARCHAR(50);
ALTER TABLE documents ADD COLUMN embedding_provider VARCHAR(50);

-- Provider usage statistics
CREATE TABLE provider_usage (
    id SERIAL PRIMARY KEY,
    provider_name VARCHAR(50),
    operation_type VARCHAR(20), -- 'llm' or 'embedding'
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_response_time FLOAT,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Testing Requirements

### Unit Tests
- [ ] Test each provider implementation individually
- [ ] Test provider manager fallback logic
- [ ] Test circuit breaker functionality
- [ ] Test configuration loading and validation

### Integration Tests
- [ ] Test provider switching during failures
- [ ] Test rate limit handling
- [ ] Test embedding consistency across providers
- [ ] Test LLM response quality across providers

### Performance Tests
- [ ] Benchmark response times for each provider
- [ ] Test concurrent request handling
- [ ] Test memory usage with multiple providers

## Implementation Steps

### Step 1: Research and Design (2-3 days)
- [ ] Evaluate LiteLLM vs custom abstraction
- [ ] Design provider interface contracts
- [ ] Create configuration schema
- [ ] Design fallback and error handling logic

### Step 2: Core Abstraction (3-4 days)
- [ ] Implement abstract base classes
- [ ] Create provider manager
- [ ] Implement configuration system
- [ ] Add basic error handling

### Step 3: Provider Implementations (4-5 days)
- [ ] Wrap existing Gemini implementation
- [ ] Implement OpenAI provider
- [ ] Implement Anthropic provider (optional)
- [ ] Add provider health checks

### Step 4: Integration (2-3 days)
- [ ] Update existing services to use abstraction
- [ ] Migrate database calls
- [ ] Update API endpoints
- [ ] Add provider selection endpoints

### Step 5: Testing and Validation (3-4 days)
- [ ] Write comprehensive test suite
- [ ] Performance testing
- [ ] Failover testing
- [ ] Documentation updates

## Success Criteria
- [ ] All existing functionality works with new abstraction
- [ ] At least 2 LLM providers supported (Gemini + OpenAI)
- [ ] At least 2 embedding providers supported
- [ ] Automatic failover works correctly
- [ ] Performance impact < 10% overhead
- [ ] 100% test coverage for abstraction layer
- [ ] Configuration-driven provider selection

## Dependencies
- Research outcome on LiteLLM vs custom implementation
- API keys for additional providers (OpenAI, Anthropic)
- Updated configuration management system
- Enhanced error handling and logging

## Risks and Mitigation
- **Risk**: Provider API changes breaking abstraction
  - **Mitigation**: Version pinning, comprehensive testing
- **Risk**: Performance overhead from abstraction
  - **Mitigation**: Benchmarking, optimization, caching
- **Risk**: Complex fallback logic causing bugs
  - **Mitigation**: Extensive testing, simple fallback rules
