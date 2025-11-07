# MoRAG Agents Framework

A comprehensive agentic pattern implementation for all LLM interactions in MoRAG. This framework provides specialized agents for different tasks, configurable prompt templates, and standardized interfaces.

## Overview

The MoRAG Agents Framework abstracts all LLM interactions behind specialized agents, each designed for specific tasks. This approach provides:

- **Specialized Agents**: Each agent is optimized for specific LLM use cases
- **Configurable Prompts**: Prompts are abstracted from code and made configurable
- **Standardized Interfaces**: Consistent APIs across all agents
- **Dynamic Configuration**: Runtime configuration of agent behavior
- **Template System**: Reusable prompt templates with examples
- **Factory Pattern**: Easy creation and management of agent instances

## Quick Start

```python
from agents import create_agent, get_agent

# Create a fact extraction agent
fact_agent = create_agent("fact_extraction")

# Extract facts from text
result = await fact_agent.extract_facts(
    "Ginkgo biloba extract (120-240mg daily) improves cognitive function.",
    domain="medical"
)

print(f"Extracted {result.total_facts} facts")
for fact in result.facts:
    print(f"- {fact.subject} -> {fact.object}")
```

## Agent Categories

### Extraction Agents
- **FactExtractionAgent**: Extracts structured facts from text
- **EntityExtractionAgent**: Identifies named entities
- **RelationExtractionAgent**: Finds relationships between entities
- **KeywordExtractionAgent**: Extracts relevant keywords

### Analysis Agents
- **QueryAnalysisAgent**: Analyzes user queries for intent and entities
- **ContentAnalysisAgent**: Analyzes content structure and topics
- **SentimentAnalysisAgent**: Performs sentiment analysis
- **TopicAnalysisAgent**: Identifies and models topics

### Reasoning Agents
- **PathSelectionAgent**: Selects optimal reasoning paths
- **ReasoningAgent**: Performs multi-step reasoning
- **DecisionMakingAgent**: Makes decisions and evaluates options
- **ContextAnalysisAgent**: Analyzes context and relevance

### Generation Agents
- **SummarizationAgent**: Generates text summaries
- **ResponseGenerationAgent**: Generates responses to queries
- **ExplanationAgent**: Creates explanations for complex topics
- **SynthesisAgent**: Synthesizes information from multiple sources

### Processing Agents
- **ChunkingAgent**: Performs intelligent text chunking
- **ClassificationAgent**: Classifies text into categories
- **ValidationAgent**: Validates content quality
- **FilteringAgent**: Filters content based on criteria

## Configuration

### Agent Configuration

Each agent can be configured through YAML files or programmatically:

```yaml
# agents/config/defaults/fact_extraction.yaml
name: "fact_extraction"
description: "Extracts structured facts from text"

model:
  provider: "gemini"
  model: "gemini-1.5-flash"
  temperature: 0.1
  max_tokens: 4000

prompt:
  domain: "general"
  include_examples: true
  output_format: "json"
  strict_json: true

agent_config:
  max_facts: 20
  filter_generic_advice: true
```

### Environment Variables

Configuration can be overridden with environment variables:

```bash
export MORAG_FACT_EXTRACTION_MAX_FACTS=50
export MORAG_FACT_EXTRACTION_DOMAIN=medical
export GEMINI_API_KEY=your_api_key
```

## Usage Examples

### Basic Usage

```python
from agents import get_agent

# Get a configured agent
agent = get_agent("fact_extraction")

# Use the agent
result = await agent.execute("Your text here")
```

### Custom Configuration

```python
from agents import create_agent, AgentConfig, PromptConfig

# Create custom configuration
config = AgentConfig(
    name="custom_fact_extraction",
    prompt=PromptConfig(
        domain="medical",
        include_examples=True,
        min_confidence=0.8
    ),
    agent_config={
        "max_facts": 50,
        "focus_on_actionable": True
    }
)

# Create agent with custom config
agent = create_agent("fact_extraction", config)
```

### Batch Processing

```python
# Process multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
results = await agent.batch_execute(texts)
```

### Using Multiple Agents

```python
from agents import get_agent

# Analyze query intent
query_agent = get_agent("query_analysis")
query_result = await query_agent.analyze_query("What are the benefits of meditation?")

# Extract facts based on query
fact_agent = get_agent("fact_extraction")
fact_result = await fact_agent.extract_facts(
    document_text,
    domain=query_result.metadata.get("domain", "general")
)

# Generate response
response_agent = get_agent("response_generation")
response = await response_agent.execute(
    query_result.intent,
    facts=fact_result.facts
)
```

## Creating Custom Agents

### 1. Define Result Model

```python
from pydantic import BaseModel, Field
from typing import List

class CustomResult(BaseModel):
    output: str = Field(..., description="Custom output")
    confidence: float = Field(..., description="Confidence score")
```

### 2. Implement Agent

```python
from agents.base import BaseAgent, AgentConfig, PromptConfig
from agents.base.template import ConfigurablePromptTemplate

class CustomAgent(BaseAgent[CustomResult]):
    def _get_default_config(self) -> AgentConfig:
        return AgentConfig(
            name="custom_agent",
            description="Custom agent description",
            prompt=PromptConfig(output_format="json")
        )

    def _create_template(self) -> ConfigurablePromptTemplate:
        system_prompt = "You are a custom agent..."
        user_prompt = "Process: {{ input }}"
        return ConfigurablePromptTemplate(
            self.config.prompt,
            system_prompt,
            user_prompt
        )

    def get_result_type(self) -> Type[CustomResult]:
        return CustomResult
```

### 3. Register Agent

```python
from agents import register_agent

register_agent("custom_agent", CustomAgent)
```

## Template System

### Template Structure

Templates are defined in YAML files with the following structure:

```yaml
system_prompt: |
  You are an expert agent...

user_prompt: |
  Process the following input: {{ input }}

examples:
  - input: "Example input"
    output: "Example output"
    explanation: "Why this is a good example"
```

### Template Variables

Templates support Jinja2 templating with these variables:
- `input`: User input
- `config`: Agent configuration
- `domain`: Domain context
- `examples`: Formatted examples
- `output_requirements`: Formatted output requirements

## Best Practices

1. **Use Appropriate Agents**: Choose the right agent for each task
2. **Configure Domains**: Set appropriate domain contexts
3. **Handle Errors**: Implement proper error handling
4. **Cache Results**: Enable caching for repeated operations
5. **Monitor Performance**: Use metrics and logging
6. **Validate Outputs**: Always validate agent outputs
7. **Batch When Possible**: Use batch processing for multiple items

## Migration Guide

To migrate existing LLM calls to use agents:

1. **Identify the Task**: Determine which agent category fits your use case
2. **Choose Agent**: Select the appropriate specialized agent
3. **Configure**: Set up configuration for your specific needs
4. **Replace Calls**: Replace direct LLM calls with agent calls
5. **Test**: Verify that outputs match expectations

### Before (Direct LLM Call)

```python
# Old direct LLM call
response = await llm_client.generate(
    prompt="Extract facts from: " + text,
    temperature=0.1
)
```

### After (Using Agent)

```python
# New agent-based approach
fact_agent = get_agent("fact_extraction")
result = await fact_agent.extract_facts(text)
```

## Performance Considerations

- **Caching**: Enable caching for repeated operations
- **Batch Processing**: Use batch methods for multiple items
- **Configuration**: Tune temperature and token limits
- **Concurrency**: Agents handle concurrent requests safely
- **Monitoring**: Use built-in metrics and logging

## Troubleshooting

### Common Issues

1. **API Key Missing**: Set appropriate environment variables
2. **Configuration Errors**: Validate configuration files
3. **Template Errors**: Check template syntax and variables
4. **Model Errors**: Verify model availability and limits
5. **Validation Errors**: Check output format and structure

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("agents").setLevel(logging.DEBUG)
```

## Contributing

To add new agents:

1. Create agent class inheriting from `BaseAgent`
2. Define result models
3. Create prompt templates
4. Add configuration files
5. Register in the factory
6. Add tests and documentation
