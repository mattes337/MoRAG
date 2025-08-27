# MoRAG API Usage Guide

This guide provides comprehensive API usage examples for all MoRAG content types using the stage-based processing system.

## 🚀 Stage-Based Processing API

**MoRAG uses a stage-based processing system with the following endpoints:**

- `POST /api/v1/stages/{stage_name}/execute` - Execute individual stages
- `POST /api/v1/stages/chain` - Execute multiple stages in sequence
- `POST /api/v1/stages/execute-all` - Execute all stages with form data
- `GET /api/v1/stages/list` - List available stages and their configurations

**Available Stages:**
- **markdown-conversion** - Convert content to markdown format
- **markdown-optimizer** - Optimize and clean markdown content
- **chunker** - Split content into semantic chunks
- **fact-generator** - Extract facts, entities, and relations
- **ingestor** - Store processed content in vector and graph databases

**Processing Modes:**
- **Single Stage** - Execute one stage at a time
- **Stage Chain** - Execute multiple stages in sequence
- **Full Pipeline** - Execute all stages from input to storage

## 📊 Stage-Based Processing Benefits

**MoRAG's stage-based architecture provides:**

- **Modularity**: Execute only the stages you need
- **Flexibility**: Configure each stage independently
- **Efficiency**: Resume processing from any stage
- **Debugging**: Inspect intermediate outputs between stages
- **Scalability**: Distribute stages across different workers

## 📚 API Usage Examples

### 1. Single Stage Execution - Markdown Conversion

```bash
# Convert PDF to markdown
curl -X POST "http://localhost:8000/api/v1/stages/markdown-conversion/execute" \
  -F "file=@document.pdf" \
  -F 'config={"include_timestamps": true, "preserve_formatting": true}'

# Convert image with text extraction
curl -X POST "http://localhost:8000/api/v1/stages/markdown-conversion/execute" \
  -F "file=@image.png" \
  -F 'config={"extract_text": true, "generate_descriptions": false}'
```

### 2. Stage Chain Execution - Multiple Stages

```bash
# Execute conversion and optimization stages
curl -X POST "http://localhost:8000/api/v1/stages/chain" \
  -F "file=@document.pdf" \
  -F 'request={
    "stages": ["markdown-conversion", "markdown-optimizer"],
    "global_config": {"language": "en"},
    "stage_configs": {
      "markdown-conversion": {"preserve_formatting": true},
      "markdown-optimizer": {"fix_transcription_errors": true}
    },
    "output_dir": "./output",
    "stop_on_failure": true
  }'

# Process audio through conversion and chunking
curl -X POST "http://localhost:8000/api/v1/stages/chain" \
  -F "file=@audio.mp3" \
  -F 'request={
    "stages": ["markdown-conversion", "chunker"],
    "stage_configs": {
      "markdown-conversion": {
        "include_timestamps": true,
        "speaker_diarization": true,
        "topic_segmentation": true
      },
      "chunker": {
        "chunk_strategy": "topic",
        "chunk_size": 4000,
        "generate_summary": true
      }
    }
  }'
```

### 3. Full Pipeline - Complete Processing with Storage

```bash
# Execute all stages with form data (easier API consumption)
curl -X POST "http://localhost:8000/api/v1/stages/execute-all" \
  -F "file=@document.pdf" \
  -F 'stages=["markdown-conversion", "markdown-optimizer", "chunker", "fact-generator", "ingestor"]' \
  -F 'global_config={"language": "en"}' \
  -F 'stage_configs={
    "ingestor": {
      "databases": ["qdrant", "neo4j"],
      "collection_name": "documents",
      "qdrant_config": {
        "host": "localhost",
        "port": 6333,
        "api_key": null
      },
      "neo4j_config": {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
      }
    }
  }' \
  -F 'webhook_url=https://your-app.com/webhook' \
  -F 'output_dir=./output' \
  -F 'stop_on_failure=true'
```

### 4. Ingestor Stage with Explicit Database Configuration

```bash
# Execute ingestor stage with explicit Neo4j and Qdrant servers
curl -X POST "http://localhost:8000/api/v1/stages/ingestor/execute" \
  -F 'input_files=["./output/document.chunks.json", "./output/document.facts.json"]' \
  -F 'config={
    "databases": ["qdrant", "neo4j"],
    "collection_name": "research_documents",
    "batch_size": 100,
    "enable_deduplication": true,
    "dedup_threshold": 0.95,
    "overwrite_existing": false,
    "validate_data": true,
    "qdrant_config": {
      "host": "qdrant.example.com",
      "port": 6333,
      "grpc_port": 6334,
      "prefer_grpc": false,
      "https": true,
      "api_key": "qdr_1234567890abcdef",
      "collection_name": "research_documents",
      "vector_size": 384,
      "timeout": 60.0,
      "verify_ssl": true
    },
    "neo4j_config": {
      "uri": "neo4j+s://neo4j.example.com:7687",
      "username": "research_user",
      "password": "secure_password_123",
      "database": "research_graph",
      "max_connection_lifetime": 3600,
      "max_connection_pool_size": 50,
      "connection_acquisition_timeout": 60,
      "verify_ssl": true,
      "trust_all_certificates": false
    }
  }' \
  -F 'webhook_url=https://your-app.com/webhook/ingest-complete'

# Example with local development servers
curl -X POST "http://localhost:8000/api/v1/stages/ingestor/execute" \
  -F 'input_files=["./output/document.chunks.json", "./output/document.facts.json"]' \
  -F 'config={
    "databases": ["qdrant", "neo4j"],
    "collection_name": "dev_documents",
    "qdrant_config": {
      "host": "localhost",
      "port": 6333,
      "api_key": null,
      "collection_name": "dev_documents"
    },
    "neo4j_config": {
      "uri": "bolt://localhost:7687",
      "username": "neo4j",
      "password": "development",
      "database": "neo4j"
    }
  }'

# Example with Docker Compose setup
curl -X POST "http://localhost:8000/api/v1/stages/ingestor/execute" \
  -F 'input_files=["./output/document.chunks.json", "./output/document.facts.json"]' \
  -F 'config={
    "databases": ["qdrant", "neo4j"],
    "collection_name": "docker_documents",
    "qdrant_config": {
      "host": "morag-qdrant",
      "port": 6333,
      "collection_name": "docker_documents"
    },
    "neo4j_config": {
      "uri": "bolt://morag-neo4j:7687",
      "username": "neo4j",
      "password": "morag_password",
      "database": "neo4j"
    }
  }'
```

### 5. Stage Execution Response Format

**Expected JSON Response Format for Single Stage:**
```json
{
  "success": true,
  "stage_type": "markdown-conversion",
  "status": "completed",
  "output_files": [
    {
      "filename": "document.md",
      "file_path": "./output/document.md",
      "file_size": 15420,
      "created_at": "2024-01-01T12:00:00",
      "stage_type": "markdown-conversion",
      "content_type": "text/markdown",
      "checksum": null,
      "content": "# Document Title\n\nContent here..."
    }
  ],
  "metadata": {
    "execution_time": 12.5,
    "start_time": "2024-01-01T12:00:00",
    "end_time": "2024-01-01T12:00:12",
    "input_files": ["./temp/document.pdf"],
    "config_used": {"preserve_formatting": true},
    "warnings": []
  },
  "error_message": null,
  "webhook_sent": false
}
```

**Expected JSON Response Format for Stage Chain:**
```json
{
  "success": true,
  "stages_executed": [
    {
      "success": true,
      "stage_type": "markdown-conversion",
      "status": "completed",
      "output_files": [...],
      "metadata": {...}
    },
    {
      "success": true,
      "stage_type": "chunker",
      "status": "completed",
      "output_files": [...],
      "metadata": {...}
    }
  ],
  "total_execution_time": 25.8,
  "failed_stage": null,
  "final_output_files": [...]
}
```

## 🎯 Complete Configuration Options for All Stages

### Markdown Conversion Stage Options

```json
{
  "markdown-conversion": {
    // Audio/Video processing
    "include_timestamps": true,
    "transcription_model": "whisper-large",
    "speaker_diarization": true,
    "topic_segmentation": true,
    "language": "en",

    // Document processing
    "chunk_on_sentences": true,
    "preserve_formatting": true,
    "extract_images": false,
    "quality_threshold": 0.8,

    // Image processing
    "extract_text": true,
    "generate_descriptions": false,
    "ocr_engine": "tesseract",
    "resize_max_dimension": 1024,

    // Web processing
    "follow_links": false,
    "max_depth": 1,
    "respect_robots": true,
    "extract_metadata": true
  }
}
```

### Markdown Optimizer Stage Options

```json
{
  "markdown-optimizer": {
    "model": "gemini-pro",
    "max_tokens": 8192,
    "temperature": 0.1,
    "fix_transcription_errors": true,
    "improve_readability": true,
    "preserve_timestamps": true,
    "normalize_formatting": true,
    "remove_redundancy": true,
    "enhance_structure": true
  }
}
```

### Chunker Stage Options

```json
{
  "chunker": {
    "chunk_strategy": "semantic",  // Options: semantic, page-level, topic-based, sentence, paragraph
    "chunk_size": 4000,
    "overlap": 200,
    "generate_summary": true,
    "preserve_structure": true,
    "min_chunk_size": 100,
    "max_chunk_size": 8000,
    "split_on_headers": true,
    "include_metadata": true
  }
}
```

### Fact Generator Stage Options

```json
{
  "fact-generator": {
    "max_facts_per_chunk": 10,
    "confidence_threshold": 0.7,
    "extract_entities": true,
    "entity_types": ["PERSON", "ORG", "GPE", "EVENT"],
    "extract_relations": true,
    "relation_confidence": 0.6,
    "extract_keywords": true,
    "domain": "general",  // Options: general, medical, legal, technical
    "model": "gemini-pro",
    "temperature": 0.1,
    "max_tokens": 4096
  }
}
```

### Ingestor Stage Options

```json
{
  "ingestor": {
    "databases": ["qdrant", "neo4j"],
    "collection_name": "my_documents",
    "batch_size": 50,
    "enable_deduplication": true,
    "dedup_threshold": 0.95,
    "conflict_resolution": "merge",  // Options: merge, replace, skip
    "overwrite_existing": false,
    "validate_data": true,
    "generate_embeddings": true,

    // Qdrant-specific configuration
    "qdrant_config": {
      "host": "localhost",
      "port": 6333,
      "grpc_port": 6334,
      "prefer_grpc": false,
      "https": false,
      "api_key": null,
      "timeout": 30.0,
      "collection_name": "morag_documents",
      "vector_size": 384,
      "verify_ssl": null
    },

    // Neo4j-specific configuration
    "neo4j_config": {
      "uri": "bolt://localhost:7687",
      "username": "neo4j",
      "password": "password",
      "database": "neo4j",
      "max_connection_lifetime": 3600,
      "max_connection_pool_size": 50,
      "connection_acquisition_timeout": 60,
      "verify_ssl": true,
      "trust_all_certificates": false
    }
  }
}
```

## 🔄 Stage-Based API Summary

| Stage | Purpose | Input | Output |
|-------|---------|-------|--------|
| `markdown-conversion` | Convert content to markdown | Raw files (PDF, audio, video, images, web) | `.md` files |
| `markdown-optimizer` | Clean and optimize markdown | `.md` files | Optimized `.md` files |
| `chunker` | Split content into chunks | `.md` files | `.chunks.json` files |
| `fact-generator` | Extract facts and entities | `.chunks.json` files | `.facts.json` files |
| `ingestor` | Store in databases | `.chunks.json` + `.facts.json` | Database storage |

| Execution Mode | Endpoint | Use Case |
|----------------|----------|----------|
| Single Stage | `/api/v1/stages/{stage}/execute` | Execute one stage, debug, testing |
| Stage Chain | `/api/v1/stages/chain` | Execute multiple stages with JSON config |
| Execute All | `/api/v1/stages/execute-all` | Execute all stages with form data |

## 🔧 Webhook Configuration

### Webhook Configuration for Stage Notifications
```json
{
  "webhook_config": {
    "url": "https://your-app.com/webhook",
    "auth_token": "optional-bearer-token",
    "headers": {
      "Custom-Header": "value"
    },
    "retry_count": 3,
    "timeout": 30
  }
}
```

### Webhook Payload Examples

**Single Stage Completion:**
```json
{
  "type": "stage_completion",
  "stage_type": "markdown-conversion",
  "success": true,
  "execution_time": 12.5,
  "timestamp": "2024-01-01T12:00:00Z",
  "output_files": [
    {
      "filename": "document.md",
      "file_id": "abc123",
      "download_url": "/api/v1/files/download/abc123"
    }
  ]
}
```

**Stage Chain Completion:**
```json
{
  "type": "stage_chain_completion",
  "success": true,
  "total_execution_time": 45.8,
  "stages_executed": 3,
  "stages_successful": 3,
  "stages_failed": 0,
  "timestamp": "2024-01-01T12:00:45Z"
}
```

## 🗂️ File Management

### List Available Files
```bash
# List all output files
curl "http://localhost:8000/api/v1/files/list?output_dir=./output"

# Filter by stage type
curl "http://localhost:8000/api/v1/files/list?stage_type=chunker&file_extension=json"
```

### Download Files
```bash
# Download a specific file
curl "http://localhost:8000/api/v1/files/download/output/document.chunks.json"

# Get file information
curl "http://localhost:8000/api/v1/files/info/output/document.md"
```

## 🚀 Python API Usage

```python
import asyncio
import httpx
import json

async def main():
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # Execute single stage - markdown conversion
        with open("document.pdf", "rb") as f:
            response = await client.post(
                f"{base_url}/api/v1/stages/markdown-conversion/execute",
                files={"file": f},
                data={"config": json.dumps({
                    "preserve_formatting": True,
                    "extract_metadata": True
                })}
            )
        conversion_result = response.json()

        # Execute stage chain - conversion + chunking
        with open("audio.mp3", "rb") as f:
            response = await client.post(
                f"{base_url}/api/v1/stages/chain",
                files={"file": f},
                data={"request": json.dumps({
                    "stages": ["markdown-conversion", "chunker"],
                    "stage_configs": {
                        "markdown-conversion": {
                            "include_timestamps": True,
                            "speaker_diarization": True
                        },
                        "chunker": {
                            "chunk_strategy": "topic",
                            "chunk_size": 4000
                        }
                    }
                })}
            )
        chain_result = response.json()

        # Execute full pipeline with database storage
        with open("document.pdf", "rb") as f:
            response = await client.post(
                f"{base_url}/api/v1/stages/execute-all",
                files={"file": f},
                data={
                    "stages": json.dumps([
                        "markdown-conversion", "markdown-optimizer",
                        "chunker", "fact-generator", "ingestor"
                    ]),
                    "stage_configs": json.dumps({
                        "ingestor": {
                            "databases": ["qdrant", "neo4j"],
                            "collection_name": "documents",
                            "qdrant_config": {
                                "host": "localhost",
                                "port": 6333
                            },
                            "neo4j_config": {
                                "uri": "bolt://localhost:7687",
                                "username": "neo4j",
                                "password": "password"
                            }
                        }
                    }),
                    "webhook_url": "https://your-app.com/webhook"
                }
            )
        full_pipeline_result = response.json()

        print(f"Conversion: {conversion_result['success']}")
        print(f"Chain: {chain_result['success']}")
        print(f"Full Pipeline: {full_pipeline_result['success']}")

asyncio.run(main())
```

## 🗄️ Stage Status Monitoring

The stage-based API provides comprehensive status monitoring and job tracking capabilities.

### Check Stage Execution Status

```bash
# Check overall stage status
curl "http://localhost:8000/api/v1/stages/status?output_dir=./output"

# Response
{
  "job_id": "abc123-def456",
  "stages_completed": ["markdown-conversion", "chunker"],
  "current_stage": "fact-generator",
  "available_files": [
    {
      "filename": "document.md",
      "file_path": "./output/document.md",
      "stage_type": "markdown-conversion",
      "file_size": 15420
    },
    {
      "filename": "document.chunks.json",
      "file_path": "./output/document.chunks.json",
      "stage_type": "chunker",
      "file_size": 8750
    }
  ],
  "overall_status": "in_progress",
  "progress_percentage": 60.0
}
```

### List Available Stages

```bash
# Get information about all available stages
curl "http://localhost:8000/api/v1/stages/list"

# Response includes stage descriptions, dependencies, and configuration options
{
  "stages": [
    {
      "stage_type": "markdown-conversion",
      "description": "Convert various content types to markdown format",
      "input_formats": ["pdf", "docx", "mp3", "mp4", "jpg", "png"],
      "output_formats": ["md"],
      "required_config": [],
      "optional_config": ["include_timestamps", "preserve_formatting"],
      "dependencies": []
    }
  ],
  "total_count": 5
}
```

### Search Stored Content (Enhanced Query API)

```bash
# Search using the enhanced query API with graph-augmented retrieval
curl -X POST "http://localhost:8000/api/v2/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 5,
    "use_graph_retrieval": true,
    "use_fact_retrieval": true,
    "database_servers": [
      {
        "type": "qdrant",
        "hostname": "localhost",
        "port": 6333,
        "database_name": "my_documents"
      },
      {
        "type": "neo4j",
        "hostname": "localhost",
        "port": 7687,
        "username": "neo4j",
        "password": "password",
        "database_name": "neo4j"
      }
    ]
  }'
```

## 🔄 Stage Execution Comparison

| Execution Mode | Response Time | Use Case | Configuration | Output |
|----------------|---------------|----------|---------------|--------|
| Single Stage | Fast | Testing, debugging, specific processing | Simple JSON config | Single stage output |
| Stage Chain | Medium | Custom workflows, partial processing | Complex JSON config | Multiple stage outputs |
| Execute All | Slow | Complete processing pipeline | Form data + JSON | Full pipeline output |

## 🌐 Environment Variables

### Database Configuration via Environment Variables

```bash
# Qdrant Configuration
export QDRANT_HOST=192.168.1.100
export QDRANT_PORT=6333
export QDRANT_API_KEY=your-qdrant-api-key
export QDRANT_COLLECTION_NAME=my_documents

# Neo4j Configuration
export NEO4J_URI=bolt://192.168.1.101:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your-neo4j-password
export NEO4J_DATABASE=morag_graph

# LLM Configuration
export GOOGLE_API_KEY=your-google-api-key
export OPENAI_API_KEY=your-openai-api-key

# Processing Configuration
export MORAG_MAX_FILE_SIZE=100MB
export MORAG_TEMP_DIR=/tmp/morag
export MORAG_OUTPUT_DIR=./output
```

## 🔍 Content Type Auto-Detection

MoRAG automatically detects content types based on file extensions and patterns:

- **Documents**: `.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.rtf`
- **Audio**: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.aac`
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`
- **Web**: URLs starting with `http://` or `https://`
- **YouTube**: URLs containing `youtube.com` or `youtu.be`

## 🔄 Migration from Legacy API

**⚠️ Important Notice**: The legacy unified endpoint (`/api/v1/process`) has been deprecated and replaced with the stage-based API.

### Legacy vs New API Mapping

| Legacy Endpoint | New Endpoint | Notes |
|----------------|--------------|-------|
| `POST /api/v1/process` (mode=convert) | `POST /api/v1/stages/markdown-conversion/execute` | Direct stage execution |
| `POST /api/v1/process` (mode=process) | `POST /api/v1/stages/chain` | Multi-stage execution |
| `POST /api/v1/process` (mode=ingest) | `POST /api/v1/stages/execute-all` | Full pipeline with storage |

### Migration Examples

**Legacy Convert Mode:**
```bash
# OLD (deprecated)
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'request_data={"mode":"convert","source_type":"file"}'

# NEW (recommended)
curl -X POST "http://localhost:8000/api/v1/stages/markdown-conversion/execute" \
  -F "file=@document.pdf" \
  -F 'config={"preserve_formatting": true}'
```

**Legacy Ingest Mode:**
```bash
# OLD (deprecated)
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'request_data={"mode":"ingest","source_type":"file","webhook_config":{"url":"https://webhook.com"}}'

# NEW (recommended)
curl -X POST "http://localhost:8000/api/v1/stages/execute-all" \
  -F "file=@document.pdf" \
  -F 'stages=["markdown-conversion","markdown-optimizer","chunker","fact-generator","ingestor"]' \
  -F 'webhook_url=https://webhook.com'
```

## 🎯 Best Practices

### 1. Stage Configuration
- Use environment variables for database credentials
- Configure webhooks for long-running operations
- Set appropriate timeouts for your use case
- Use deduplication to avoid processing the same content twice

### 2. Error Handling
- Always check the `success` field in responses
- Monitor webhook notifications for async operations
- Use the status endpoints to track progress
- Implement retry logic for failed stages

### 3. Performance Optimization
- Use appropriate batch sizes for ingestion
- Configure chunk sizes based on your content type
- Use semantic chunking for better search results
- Enable deduplication to reduce storage costs

### 4. Security
- Use HTTPS for webhook URLs
- Secure database credentials with environment variables
- Enable SSL verification for production databases
- Use API keys for Qdrant in production environments
