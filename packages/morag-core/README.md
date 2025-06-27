# MoRAG Core

Core components for the MoRAG (Multimodal RAG Ingestion Pipeline) project.

## Overview

This package contains the essential core components of the MoRAG system, including:

- **PydanticAI Agent Framework**: Modern AI agent foundation with structured outputs
- Base exceptions and error handling
- Configuration management
- Common utilities
- Interface definitions and base classes

## Installation

```bash
pip install morag-core
```

## Usage

### PydanticAI Agent Framework

Create AI agents with structured outputs and automatic validation:

```python
from morag_core.ai import MoRAGBaseAgent, create_agent_with_config
from pydantic import BaseModel, Field
from typing import Type, List

# Define your structured output
class AnalysisResult(BaseModel):
    topic: str = Field(description="Main topic")
    sentiment: str = Field(description="Sentiment analysis")
    key_points: List[str] = Field(description="Key points")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

# Create your agent
class TextAnalysisAgent(MoRAGBaseAgent[AnalysisResult]):
    def get_result_type(self) -> Type[AnalysisResult]:
        return AnalysisResult

    def get_system_prompt(self) -> str:
        return "Analyze text and provide structured insights."

# Use the agent
agent = create_agent_with_config(
    TextAnalysisAgent,
    model="google-gla:gemini-1.5-flash",
    temperature=0.1
)

# Run analysis (requires GEMINI_API_KEY)
result = await agent.run("Analyze this text...")
print(f"Topic: {result.topic}")
print(f"Sentiment: {result.sentiment}")
```

### Core Components

```python
from morag_core.exceptions import MoRAGException, ValidationError
from morag_core.config import Settings

# Use core components in your application
settings = Settings()
print(f"API Host: {settings.api_host}")

try:
    # Your code here
    pass
except ValidationError as e:
    print(f"Validation error: {e}")
except MoRAGException as e:
    print(f"General error: {e}")
```

## Features

### PydanticAI Integration

- **Structured Outputs**: Automatic validation of LLM responses using Pydantic models
- **Type Safety**: Full type checking and IDE support for AI interactions
- **Error Handling**: Built-in retry logic and error recovery mechanisms
- **Provider Abstraction**: Easy switching between different LLM providers
- **Observability**: Built-in logging and monitoring capabilities

### Agent Types

- **EntityExtractionAgent**: Extract entities from text with confidence scores
- **RelationExtractionAgent**: Identify relationships between entities
- **SummaryAgent**: Generate structured summaries with key points
- **SemanticChunkingAgent**: Intelligent content segmentation
- **QueryAnalysisAgent**: Analyze user queries for intent and entities

## Dependencies

This package has minimal dependencies to avoid dependency conflicts:

- pydantic>=2.10.0
- pydantic-settings>=2.7.0
- pydantic-ai>=0.0.14
- structlog>=24.4.0
- python-dotenv>=1.0.0
- google-generativeai>=0.3.0 (for Gemini provider)

## License

MIT