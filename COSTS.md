# MoRAG Document Ingestion Cost Analysis

This document provides a comprehensive cost analysis for ingesting documents of various sizes using different AI models in MoRAG. The analysis covers the main AI operations performed during document processing and ingestion.

## AI Operations in MoRAG Document Ingestion

MoRAG performs several AI-powered operations during document ingestion:

1. **Text Embedding Generation** - Converting text chunks to vector embeddings
2. **Document Summarization** - Generating document summaries using LLMs
3. **Entity Extraction** - Extracting entities and relationships from text
4. **Semantic Chunking** - AI-powered intelligent text chunking
5. **Vision Processing** - OCR and image analysis for PDFs/images
6. **Audio/Video Transcription** - Converting media to text

## Model Pricing (as of January 2025)

### Google Gemini Models (Primary MoRAG Provider)

| Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) | Use Case in MoRAG |
|-------|----------------------------|------------------------------|-------------------|
| **text-embedding-004** | $0.15 | N/A | Text embeddings |
| **gemini-2.0-flash** | $0.10 (text/image/video) | $0.40 | General processing, entity extraction |
| **gemini-2.5-flash** | $0.30 (text/image/video) | $2.50 | Advanced processing |
| **gemini-2.5-flash-lite** | $0.10 (text/image/video) | $0.40 | Lightweight processing |
| **gemini-1.5-flash** | $0.075 (≤128k), $0.15 (>128k) | $0.30 (≤128k), $0.60 (>128k) | Legacy processing |
| **gemini-1.5-pro** | $1.25 (≤128k), $2.50 (>128k) | $5.00 (≤128k), $10.00 (>128k) | Complex reasoning |

### OpenAI Models (Alternative Provider)

| Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) | Use Case |
|-------|----------------------------|------------------------------|----------|
| **text-embedding-3-small** | $0.02 | N/A | Text embeddings |
| **text-embedding-3-large** | $0.13 | N/A | High-quality embeddings |
| **gpt-4o** | $2.50 | $10.00 | General processing |
| **gpt-4o-mini** | $0.15 | $0.60 | Lightweight processing |
| **gpt-3.5-turbo** | $0.50 | $1.50 | Basic processing |

### Anthropic Claude Models (Alternative Provider)

| Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) | Use Case |
|-------|----------------------------|------------------------------|----------|
| **claude-3.5-sonnet** | $3.00 | $15.00 | Advanced reasoning |
| **claude-3.5-haiku** | $0.25 | $1.25 | Fast processing |

## Document Size Categories and Token Estimates

### Token Estimation Rules
- **English text**: ~750 tokens per 1,000 characters (~4 characters per token)
- **Code**: ~500-600 tokens per 1,000 characters
- **Technical documents**: ~700-800 tokens per 1,000 characters

### Document Categories

| Document Size | Characters | Pages (approx) | Estimated Tokens | Example Documents |
|---------------|------------|----------------|------------------|-------------------|
| **Small** | 5,000 | 1-2 | 3,750 | Email, short article |
| **Medium** | 25,000 | 5-10 | 18,750 | Research paper, manual chapter |
| **Large** | 100,000 | 20-40 | 75,000 | Technical manual, book chapter |
| **Very Large** | 500,000 | 100-200 | 375,000 | Full book, comprehensive report |
| **Massive** | 2,000,000 | 400-800 | 1,500,000 | Large technical documentation |

## MoRAG Processing Configuration

### Default Chunking Settings
- **Chunk size**: 4,000 characters (default)
- **Chunk overlap**: 200 characters
- **Embedding batch size**: 50 chunks
- **Effective tokens per chunk**: ~3,000 tokens

### Processing Steps Token Usage

| Operation | Tokens per Document | Notes |
|-----------|-------------------|-------|
| **Embedding Generation** | Document tokens ÷ chunk_size × chunks | Each chunk embedded separately |
| **Document Summary** | Input tokens + ~200 output tokens | Full document → summary |
| **Entity Extraction** | Input tokens + ~500 output tokens | Per chunk or full document |
| **Semantic Chunking** | Input tokens + ~100 output tokens | AI-powered chunking |

## Cost Analysis by Document Size

### Small Document (5,000 characters, ~3,750 tokens)

**Using Gemini (Default MoRAG Configuration):**
- Chunks created: ~2 chunks
- Embedding cost: 3,750 tokens × $0.15/1M = **$0.0006**
- Summary cost: 3,750 input + 200 output = 3,950 tokens × $0.10/1M input + 200 × $0.40/1M output = **$0.0004**
- Entity extraction: 3,750 input + 500 output = $0.0004 + $0.0002 = **$0.0006**
- **Total: ~$0.0016**

**Using OpenAI:**
- Embedding (text-embedding-3-small): 3,750 × $0.02/1M = **$0.00008**
- Processing (gpt-4o-mini): 3,750 × $0.15/1M + 700 × $0.60/1M = **$0.0010**
- **Total: ~$0.0011**

### Medium Document (25,000 characters, ~18,750 tokens)

**Using Gemini:**
- Chunks: ~7 chunks
- Embedding: 18,750 × $0.15/1M = **$0.0028**
- Summary: 18,750 × $0.10/1M + 200 × $0.40/1M = **$0.0019**
- Entity extraction: 18,750 × $0.10/1M + 500 × $0.40/1M = **$0.0021**
- **Total: ~$0.0068**

**Using OpenAI:**
- Embedding: 18,750 × $0.02/1M = **$0.0004**
- Processing: 18,750 × $0.15/1M + 700 × $0.60/1M = **$0.0032**
- **Total: ~$0.0036**

### Large Document (100,000 characters, ~75,000 tokens)

**Using Gemini:**
- Chunks: ~25 chunks
- Embedding: 75,000 × $0.15/1M = **$0.0113**
- Summary: 75,000 × $0.10/1M + 200 × $0.40/1M = **$0.0076**
- Entity extraction: 75,000 × $0.10/1M + 500 × $0.40/1M = **$0.0077**
- **Total: ~$0.0266**

**Using OpenAI:**
- Embedding: 75,000 × $0.02/1M = **$0.0015**
- Processing: 75,000 × $0.15/1M + 700 × $0.60/1M = **$0.0117**
- **Total: ~$0.0132**

### Very Large Document (500,000 characters, ~375,000 tokens)

**Using Gemini:**
- Chunks: ~125 chunks
- Embedding: 375,000 × $0.15/1M = **$0.0563**
- Summary: 375,000 × $0.10/1M + 200 × $0.40/1M = **$0.0376**
- Entity extraction: 375,000 × $0.10/1M + 500 × $0.40/1M = **$0.0377**
- **Total: ~$0.1316**

**Using OpenAI:**
- Embedding: 375,000 × $0.02/1M = **$0.0075**
- Processing: 375,000 × $0.15/1M + 700 × $0.60/1M = **$0.0567**
- **Total: ~$0.0642**

### Massive Document (2,000,000 characters, ~1,500,000 tokens)

**Using Gemini:**
- Chunks: ~500 chunks
- Embedding: 1,500,000 × $0.15/1M = **$0.2250**
- Summary: 1,500,000 × $0.10/1M + 200 × $0.40/1M = **$0.1501**
- Entity extraction: 1,500,000 × $0.10/1M + 500 × $0.40/1M = **$0.1502**
- **Total: ~$0.5253**

**Using OpenAI:**
- Embedding: 1,500,000 × $0.02/1M = **$0.0300**
- Processing: 1,500,000 × $0.15/1M + 700 × $0.60/1M = **$0.2254**
- **Total: ~$0.2554**

## Cost Summary Table

| Document Size | Gemini Cost | OpenAI Cost | Anthropic Cost* | Cost Difference |
|---------------|-------------|-------------|-----------------|-----------------|
| Small (3.7k tokens) | $0.0016 | $0.0011 | $0.0047 | OpenAI 31% cheaper |
| Medium (18.7k tokens) | $0.0068 | $0.0036 | $0.0234 | OpenAI 47% cheaper |
| Large (75k tokens) | $0.0266 | $0.0132 | $0.0938 | OpenAI 50% cheaper |
| Very Large (375k tokens) | $0.1316 | $0.0642 | $0.4688 | OpenAI 51% cheaper |
| Massive (1.5M tokens) | $0.5253 | $0.2554 | $1.8750 | OpenAI 51% cheaper |

*Anthropic costs calculated using Claude 3.5 Haiku for processing

## Cost Optimization Strategies

### 1. Model Selection
- **Use OpenAI for cost-sensitive applications** (50% cheaper for large documents)
- **Use Gemini for balanced performance/cost** (better integration with MoRAG)
- **Avoid Anthropic for high-volume processing** (3-4x more expensive)

### 2. Chunking Optimization
- **Increase chunk size** to reduce embedding calls (fewer chunks = lower cost)
- **Reduce chunk overlap** to minimize redundant processing
- **Use semantic chunking** only when necessary (adds LLM cost)

### 3. Processing Optimization
- **Disable unnecessary features** (entity extraction, summarization) for cost-sensitive use cases
- **Use batch processing** to reduce API overhead
- **Implement caching** to avoid reprocessing identical content

### 4. Provider-Specific Optimizations
- **Gemini**: Use free tier for development (15 RPM limit)
- **OpenAI**: Use batch API for 50% discount on non-urgent processing
- **Consider hybrid approach**: OpenAI for embeddings, Gemini for processing

## Recommendations

### For Development/Testing
- Use **Gemini free tier** for initial development
- Switch to **OpenAI** for cost-effective testing with larger datasets

### For Production
- **Small to Medium documents**: Either provider is cost-effective
- **Large documents**: **OpenAI is recommended** (50% cost savings)
- **High-volume processing**: **OpenAI with batch API** (additional 50% discount)

### For Enterprise
- Consider **volume discounts** from providers
- Implement **intelligent caching** to reduce redundant processing
- Use **hybrid model selection** based on document characteristics

## Notes

- Costs are based on January 2025 pricing and may change
- Actual costs may vary based on document complexity and processing options
- Network and storage costs are not included in this analysis
- Free tier limits apply to development usage
