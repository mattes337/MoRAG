# LLM Batch Processing in MoRAG

MoRAG now supports batch processing for LLM calls to reduce API usage and avoid quota exhaustion. This feature is particularly useful when processing multiple documents, chunks, or prompts that would otherwise require individual API calls.

## Overview

The batch processing system combines multiple prompts into a single API call using Gemini's 1M token context window. This significantly reduces the number of API calls and helps avoid 429 "resource exhausted" errors on free plans.

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# LLM Batch Configuration
MORAG_LLM_BATCH_SIZE=10                    # Number of prompts per batch (1-50)
MORAG_ENABLE_LLM_BATCHING=true             # Enable/disable batch processing
MORAG_LLM_BATCH_DELAY=1.0                  # Delay between batches (seconds)
MORAG_LLM_MAX_BATCH_TOKENS=800000          # Max tokens per batch (up to 1M)
MORAG_LLM_BATCH_TIMEOUT=120                # Timeout for batch requests (seconds)
```

### Programmatic Configuration

```python
from morag_reasoning.llm import LLMConfig, LLMClient

config = LLMConfig(
    provider="gemini",
    model="gemini-1.5-flash",
    api_key="your-api-key",
    batch_size=10,
    enable_batching=True,
    batch_delay=1.0,
    max_batch_tokens=800000
)

llm_client = LLMClient(config)
```

## Usage

### Basic Batch Processing

```python
from morag_reasoning.llm import LLMClient, LLMConfig

# Initialize client with batch processing enabled
config = LLMConfig(
    provider="gemini",
    api_key="your-api-key",
    enable_batching=True,
    batch_size=5
)
client = LLMClient(config)

# Process multiple prompts in batches
prompts = [
    "What is artificial intelligence?",
    "Explain machine learning briefly.",
    "What are neural networks?",
    "Define deep learning.",
    "What is natural language processing?"
]

responses = await client.generate_batch(prompts)
print(f"Processed {len(prompts)} prompts, got {len(responses)} responses")
```

### Text Analysis Batch Processing

```python
from morag_reasoning import batch_text_analysis

texts = [
    "Apple Inc. is a technology company founded by Steve Jobs.",
    "Microsoft was founded by Bill Gates and Paul Allen.",
    "Google was created by Larry Page and Sergey Brin."
]

# Batch entity extraction
results = await batch_text_analysis(
    llm_client, 
    texts, 
    analysis_type="entity_extraction"
)

for result in results:
    if result.success:
        entities = result.result.get("analysis", [])
        print(f"Found {len(entities)} entities")
    else:
        print(f"Failed: {result.error_message}")
```

### Document Chunk Batch Processing

```python
from morag_reasoning import batch_document_chunks

chunks = [
    {
        "id": "chunk_1",
        "text": "Apple Inc. is a multinational technology company...",
        "document_id": "doc_1"
    },
    {
        "id": "chunk_2", 
        "text": "The iPhone is Apple's flagship smartphone...",
        "document_id": "doc_1"
    }
]

# Batch extraction of entities and relations
results = await batch_document_chunks(
    llm_client,
    chunks,
    processing_type="extraction"
)

for result in results:
    if result.success:
        entities = result.result.get("entities", [])
        relations = result.result.get("relations", [])
        print(f"Chunk {result.item['id']}: {len(entities)} entities, {len(relations)} relations")
```

### Custom Batch Processing

```python
from morag_reasoning import batch_llm_calls

def create_summary_prompt(text: str) -> str:
    return f"Summarize this text in one sentence: {text}"

def parse_summary_response(response: str, text: str) -> dict:
    return {"summary": response.strip(), "original_length": len(text)}

texts = ["Long text 1...", "Long text 2...", "Long text 3..."]

results = await batch_llm_calls(
    llm_client,
    texts,
    create_summary_prompt,
    parse_summary_response
)
```

## How It Works

1. **Prompt Combination**: Multiple prompts are combined into a single request with clear delimiters and instructions
2. **Token Management**: The system estimates token usage and splits batches if they exceed the limit
3. **Response Parsing**: The combined response is parsed to extract individual answers
4. **Error Handling**: Failed batches fall back to individual processing
5. **Rate Limiting**: Configurable delays between batches prevent overwhelming the API

## Benefits

- **Reduced API Calls**: Process 10 prompts with 1-2 API calls instead of 10
- **Quota Management**: Avoid 429 errors on free/limited plans
- **Cost Efficiency**: Fewer API calls mean lower costs
- **Better Throughput**: Leverage Gemini's large context window efficiently

## Automatic Integration

Batch processing is automatically used in:

- **Graph Processing**: When processing multiple document chunks
- **Entity Extraction**: When extracting entities from multiple texts
- **Relation Extraction**: When finding relations across multiple chunks
- **Document Analysis**: When analyzing multiple documents

## Performance Tips

1. **Optimal Batch Size**: Start with 5-10 prompts per batch
2. **Token Awareness**: Monitor token usage to avoid hitting limits
3. **Fallback Strategy**: Always have individual processing as fallback
4. **Delay Configuration**: Adjust delays based on your quota limits

## Monitoring

Enable logging to monitor batch processing:

```python
import logging
logging.getLogger("morag_reasoning.llm").setLevel(logging.INFO)
logging.getLogger("morag_reasoning.batch_processor").setLevel(logging.INFO)
```

Log messages will show:
- Batch sizes and processing times
- Token estimates and usage
- Fallback scenarios
- Error conditions

## Troubleshooting

### Common Issues

1. **Batch Processing Disabled**: Check `MORAG_ENABLE_LLM_BATCHING=true`
2. **Token Limit Exceeded**: Reduce `MORAG_LLM_BATCH_SIZE` or `MORAG_LLM_MAX_BATCH_TOKENS`
3. **Parse Errors**: The system automatically falls back to individual processing
4. **API Errors**: Increase `MORAG_LLM_BATCH_DELAY` to reduce rate limiting

### Debugging

```python
# Test batch processing with debug info
config = LLMConfig(enable_batching=True, batch_size=2)
client = LLMClient(config)

prompts = ["Test 1", "Test 2", "Test 3"]
responses = await client.generate_batch(prompts)

# Check if batching was used
if len(responses) == len(prompts):
    print("✅ Batch processing successful")
else:
    print("⚠️ Some responses missing")
```

## Migration Guide

### From Individual Calls

**Before:**
```python
results = []
for chunk in chunks:
    response = await llm_client.generate_text(create_prompt(chunk))
    results.append(parse_response(response, chunk))
```

**After:**
```python
results = await batch_llm_calls(
    llm_client, 
    chunks, 
    create_prompt, 
    parse_response
)
```

### Gradual Adoption

1. Enable batch processing: `MORAG_ENABLE_LLM_BATCHING=true`
2. Start with small batch sizes: `MORAG_LLM_BATCH_SIZE=3`
3. Monitor performance and adjust
4. Gradually increase batch size as needed

The system automatically falls back to individual processing if batch processing fails, ensuring reliability during migration.
