# MoRAG vs. WeKnora (Tencent)

This note contrasts MoRAG with Tencent's WeKnora to help choose the right tool for a given scenario.

Links:
- MoRAG (this repo)
- WeKnora: https://github.com/Tencent/WeKnora

## TL;DR
- If you want a modular, stage-based Python system with structured fact extraction, type-safe AI (PydanticAI), Neo4j knowledge graph, and Qdrant vector search integrated with a Celery queue and CLI/REST APIs, choose MoRAG.
- If you want a turnkey Go-based knowledge base with a polished Web UI, hybrid retrieval (BM25/Dense/GraphRAG), knowledge-graph-enhanced search, Docker-first deploy, and WeChat/MCP integrations, choose WeKnora.

## Focus and Scope
- MoRAG: Multimodal RAG pipeline framework; emphasizes canonical processing stages (markdown-conversion → chunker → fact-generator → ingestor), structured citations and facts, graph operations, and resumable pipelines; strong CLI/REST with Celery workers for scale.
- WeKnora: Product-like knowledge base platform; emphasizes end-to-end UX (web upload, processing progress, QA chat), hybrid retrieval with KG augmentation, easy Docker start scripts, and integrations (WeChat Dialog Platform, MCP server).

## Architecture & Tech Stack
- MoRAG
  - Language/Runtime: Python
  - Interfaces: CLI, FastAPI REST
  - Processing: Stage-based pipeline with resume capability and standardized outputs
  - AI layer: PydanticAI for validated, type-safe agent I/O; Gemini-first embeddings with batch APIs
  - Storage: Qdrant (vectors), Neo4j (knowledge graph), Redis/Celery for tasks
  - Modularity: Separate packages per modality (document, audio, video, web, etc.)
- WeKnora
  - Language/Runtime: Go backend, Vue frontend, Python helpers
  - Interfaces: Full Web UI + REST API; MCP server integration
  - Processing: Modular services (docreader, retrieval, LLM); knowledge graph visualization
  - Storage: pgvector (PostgreSQL) and/or Elasticsearch for vectors; hybrid retrieval pipelines
  - Deployment: Docker Compose and start scripts (./scripts/start_all.sh)

## Retrieval, Indexing, and Graph
- MoRAG
  - Embeddings: Gemini-based with optimized batching; multiple chunking strategies
  - Vector DB: Qdrant
  - Graph: First-class Neo4j with entity/fact extraction, relationships, traversal tools and auditing
  - Citations: Unified, structured citations across all content types
- WeKnora
  - Retrieval: BM25, Dense Retrieval, GraphRAG; customizable retrieve→rerank→generate
  - Vector DB: pgvector (Postgres) and Elasticsearch
  - Graph: Document knowledge graph and KG-enhanced retrieval; built-in visualization in Web UI

## Models and Inference
- MoRAG: Gemini-centric configuration with agent-specific model overrides (fact/entity/relation/summarization/chunking), fallback chains via env/CLI.
- WeKnora: Supports Qwen/DeepSeek and local engines (e.g., via Ollama); embedding model flexibility (BGE/GTE/local or APIs), “thinking/non-thinking” mode switch.

## Deployment & Operations
- MoRAG: Python packages (editable installs), Docker Compose options, Celery worker is required for processing; remote GPU worker support for audio/video.
- WeKnora: Docker-first startup, login auth (v0.1.3+), simple Web UI initialization, service health via compose; optional Ollama, Jaeger links in README.

## User Experience & DX
- MoRAG: Developer-first pipeline control via CLI/REST; powerful for building custom multimodal pipelines, research-grade fact/graph capabilities.
- WeKnora: Operator/end-user-first; Web UI for uploading documents, progress tracking, Q&A chat, graph views, and configuration via browser.

## Licensing
- Both projects are MIT-licensed (as of referenced READMEs).

## When to Choose Which
- Choose MoRAG when you need:
  - Fine-grained, reproducible stage orchestration across multimodal inputs
  - Structured fact extraction with citations and a robust Neo4j graph layer
  - Python ecosystem integration and Celery-based horizontal scaling
- Choose WeKnora when you need:
  - A turnkey knowledge base with a production-ready Web UI out of the box
  - Hybrid retrieval (BM25/Dense/GraphRAG) with KG visualization
  - Docker-first deployment, WeChat/MCP integrations, Go+Vue stack

## Notes and References
- WeKnora README confirms: BM25/Dense/GraphRAG; pgvector/Elasticsearch; Web UI; Docker Compose; MCP server; login/auth; Go+Vue stack.
- MoRAG README confirms: stage-based architecture; PydanticAI; Qdrant and Neo4j; Celery; citations; multimodal packages and CLI/REST.

