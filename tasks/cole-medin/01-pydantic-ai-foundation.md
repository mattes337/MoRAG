# Task 1: Setup PydanticAI Foundation

## Background

PydanticAI is a modern framework for building AI agents with structured outputs, automatic validation, and robust error handling. It provides a clean abstraction over LLM APIs while ensuring type safety and data validation through Pydantic models.

### Why PydanticAI for MoRAG?

1. **Structured Outputs**: Automatic validation of LLM responses using Pydantic models
2. **Type Safety**: Full type checking and IDE support for AI interactions
3. **Error Handling**: Built-in retry logic and error recovery mechanisms
4. **Provider Abstraction**: Easy switching between different LLM providers
5. **Observability**: Built-in logging and monitoring capabilities

## Current State Analysis

MoRAG currently uses direct API calls to Gemini across multiple components:
- Entity extraction in `morag-graph`
- Relation extraction in `morag-graph`
- Document summarization in `morag-services`
- Query processing in `morag-services`
- Content analysis across various packages

## Implementation Strategy

### Phase 1: Foundation Setup (2 days)

#### 1.1 Install PydanticAI Dependencies
**File**: `requirements.txt`, `pyproject.toml`
```python
# Add to requirements.txt
pydantic-ai==0.0.14  # Latest version
pydantic-ai-slim==0.0.14  # If we need minimal dependencies
```

#### 1.2 Create Base Agent Classes
**File**: `packages/morag-core/src/morag_core/ai/base_agent.py`

```python
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import TypeVar, Generic, Optional, Dict, Any
from abc import ABC, abstractmethod

T = TypeVar('T', bound=BaseModel)

class MoRAGBaseAgent(Generic[T], ABC):
    """Base class for all MoRAG AI agents."""
    
    def __init__(self, model: str = "gemini-2.0-flash-001"):
        self.model = model
        self.agent = Agent(
            model=model,
            result_type=self.get_result_type(),
            system_prompt=self.get_system_prompt()
        )
    
    @abstractmethod
    def get_result_type(self) -> type[T]:
        """Return the Pydantic model for structured output."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    async def run(self, user_prompt: str, **kwargs) -> T:
        """Execute the agent with error handling."""
        try:
            result = await self.agent.run(user_prompt, **kwargs)
            return result.data
        except Exception as e:
            # Log error and potentially retry
            raise AIAgentError(f"Agent execution failed: {e}")
```

#### 1.3 Implement Gemini Provider Integration
**File**: `packages/morag-core/src/morag_core/ai/providers.py`

```python
import google.generativeai as genai
from pydantic_ai.models import Model
from typing import Optional

class GeminiProvider:
    """Gemini provider for PydanticAI."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-001"):
        genai.configure(api_key=api_key)
        self.model = model
    
    def get_model(self) -> Model:
        """Get configured Gemini model for PydanticAI."""
        return genai.GenerativeModel(self.model)
```

### Phase 2: Structured Response Models (1 day)

#### 2.1 Entity Models
**File**: `packages/morag-core/src/morag_core/ai/models/entity_models.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"

class ExtractedEntity(BaseModel):
    """Structured model for extracted entities."""
    name: str = Field(description="Entity name")
    type: EntityType = Field(description="Entity type")
    description: Optional[str] = Field(description="Entity description")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")

class EntityExtractionResult(BaseModel):
    """Result of entity extraction."""
    entities: List[ExtractedEntity]
    total_count: int
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### 2.2 Relation Models
**File**: `packages/morag-core/src/morag_core/ai/models/relation_models.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ExtractedRelation(BaseModel):
    """Structured model for extracted relations."""
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relation_type: str = Field(description="Type of relation")
    description: Optional[str] = Field(description="Relation description")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    properties: Dict[str, Any] = Field(default_factory=dict)

class RelationExtractionResult(BaseModel):
    """Result of relation extraction."""
    relations: List[ExtractedRelation]
    total_count: int
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Phase 3: Error Handling and Configuration (1 day)

#### 3.1 Error Handling
**File**: `packages/morag-core/src/morag_core/ai/exceptions.py`

```python
class AIAgentError(Exception):
    """Base exception for AI agent errors."""
    pass

class ModelTimeoutError(AIAgentError):
    """Raised when model request times out."""
    pass

class ValidationError(AIAgentError):
    """Raised when model output validation fails."""
    pass
```

#### 3.2 Configuration System
**File**: `packages/morag-core/src/morag_core/ai/config.py`

```python
from pydantic import BaseModel, Field
from typing import Optional

class AIAgentConfig(BaseModel):
    """Configuration for AI agents."""
    model: str = Field(default="gemini-2.0-flash-001")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None)
```

## Testing and Documentation Strategy

### Automated Testing (Each Step)
- Run automated tests in `/tests/test_pydantic_ai_foundation.py` after each implementation step
- Test base agent functionality with real Gemini API calls
- Test structured model validation with edge cases
- Test error handling and retry scenarios
- Test configuration loading and validation

### Documentation Updates (Mandatory)
- Update `README.md` with PydanticAI integration overview
- Update `CLI.md` with new AI agent configuration options
- Update `docs/ARCHITECTURE.md` with new AI agent patterns
- Create `docs/pydantic-ai-integration.md` with detailed implementation guide
- Remove any documentation referencing old direct API call patterns

## Success Criteria

1. ✅ PydanticAI successfully installed and configured
2. ✅ Base agent classes created and tested with automated tests
3. ✅ Gemini provider integration working and validated
4. ✅ Structured response models defined and tested
5. ✅ Error handling and retry logic implemented and tested
6. ✅ Configuration system operational and documented
7. ✅ All documentation files updated to reflect new implementation
8. ✅ Old direct API call code completely removed

## Dependencies

- PydanticAI framework
- Google Generative AI SDK
- Existing MoRAG configuration system

## Code Cleanup Strategy

### Remove Old LLM Integration Code
- Remove all direct Gemini API calls throughout the codebase
- Remove old JSON parsing utilities
- Remove old prompt template systems
- Remove old LLM configuration classes
- Remove old error handling for direct API calls

### Update All Dependencies
- Update all imports to use new PydanticAI agents
- Update all LLM service calls to use structured agents
- Update all configuration references
- Remove old LLM provider abstractions

## Risks and Mitigation

1. **Risk**: PydanticAI API changes
   **Mitigation**: Pin specific version, monitor releases

2. **Risk**: Performance overhead
   **Mitigation**: Benchmark against baseline, optimize prompts

3. **Risk**: Complete system replacement complexity
   **Mitigation**: Thorough testing at each step, comprehensive validation
