# CLI Hybrid Episode Strategy Guide

This guide explains how to use the new hybrid episode strategy options in MoRAG CLI scripts for optimal knowledge graph ingestion.

## 🎯 Overview

The hybrid episode strategy provides both granular chunk-level episodes and contextual processing, giving you the best of both worlds:

- **Granular Processing**: Each chunk becomes its own episode for precise retrieval
- **Contextual Processing**: Rich contextual summaries preserve document-level understanding
- **Flexible Querying**: Query at both document and chunk levels
- **AI-Enhanced**: Automatic contextual summaries using LLM capabilities

## 📋 Episode Strategies

### 1. `hybrid` (Recommended)
Creates both document-level and contextual chunk-level episodes.
- **Best for**: Most documents, comprehensive knowledge capture
- **Episodes created**: 1 document + N contextual chunks
- **Use case**: When you need both overview and detailed access

### 2. `contextual_chunks`
Creates only chunk episodes with rich contextual summaries.
- **Best for**: Large documents, detailed analysis
- **Episodes created**: N contextual chunks only
- **Use case**: When you want granular access with context

### 3. `document_only`
Creates a single episode for the entire document.
- **Best for**: Short documents, simple storage
- **Episodes created**: 1 document episode
- **Use case**: When document-level access is sufficient

### 4. `chunk_only`
Creates basic chunk episodes without contextual summaries.
- **Best for**: Simple chunking, minimal processing
- **Episodes created**: N basic chunks
- **Use case**: When you need basic chunking without AI enhancement

## 🎚️ Context Levels

### `minimal`
- Basic metadata only
- No AI summarization
- Fastest processing

### `standard`
- Document summary + chunk relationships
- Basic contextual information
- Balanced processing

### `rich` (Recommended)
- Full context with surrounding chunks
- AI-powered summaries
- Semantic analysis

### `comprehensive`
- All available context
- Cross-document relationships
- Maximum contextual information

## 🚀 CLI Usage Examples

### test-graphiti.py (Dedicated Graphiti CLI)

```bash
# Basic hybrid ingestion (recommended)
python cli/test-graphiti.py ingest document.pdf

# Contextual chunks with comprehensive context
python cli/test-graphiti.py ingest document.pdf \
  --episode-strategy contextual_chunks \
  --context-level comprehensive

# Document-only with minimal context (fast)
python cli/test-graphiti.py ingest document.pdf \
  --episode-strategy document_only \
  --context-level minimal

# Custom episode naming
python cli/test-graphiti.py ingest quarterly_report.pdf \
  --episode-prefix "Q1_2024_report" \
  --context-level rich

# Disable AI summarization (use basic summaries)
python cli/test-graphiti.py ingest document.pdf \
  --disable-ai-summarization \
  --context-level standard
```

### test-document.py (Document Processing)

```bash
# Hybrid strategy with rich context (recommended)
python cli/test-document.py document.pdf --graphiti \
  --episode-strategy hybrid \
  --context-level rich

# Contextual chunks for detailed analysis
python cli/test-document.py research_paper.pdf --graphiti \
  --episode-strategy contextual_chunks \
  --context-level comprehensive \
  --episode-prefix "research_2024"

# Simple document-only ingestion
python cli/test-document.py summary.pdf --graphiti \
  --episode-strategy document_only \
  --context-level minimal

# With custom metadata and chunking
python cli/test-document.py manual.pdf --graphiti \
  --episode-strategy hybrid \
  --context-level rich \
  --chunking-strategy chapter \
  --metadata '{"category": "technical", "version": "2.0"}'
```

### test-audio.py (Audio Processing)

```bash
# Audio with contextual chunks (default for audio)
python cli/test-audio.py meeting.mp3 --graphiti \
  --episode-strategy contextual_chunks \
  --context-level rich

# Hybrid strategy for comprehensive audio analysis
python cli/test-audio.py interview.wav --graphiti \
  --episode-strategy hybrid \
  --context-level comprehensive \
  --enable-diarization \
  --enable-topics

# Simple document-only for short audio
python cli/test-audio.py announcement.mp3 --graphiti \
  --episode-strategy document_only \
  --context-level standard
```

### test-image.py (Image Processing)

```bash
# Document-only for images (default)
python cli/test-image.py diagram.png --graphiti \
  --episode-strategy document_only \
  --context-level standard

# Rich context for complex images
python cli/test-image.py technical_diagram.jpg --graphiti \
  --episode-strategy document_only \
  --context-level rich \
  --episode-prefix "tech_diagram"

# Minimal processing for simple images
python cli/test-image.py screenshot.png --graphiti \
  --episode-strategy document_only \
  --context-level minimal
```

## 🎯 Recommended Configurations

### For Different Content Types

**📄 Text Documents (PDF, DOCX, TXT)**
```bash
--episode-strategy hybrid --context-level rich
```

**🎵 Audio Files (MP3, WAV, M4A)**
```bash
--episode-strategy contextual_chunks --context-level rich
```

**🖼️ Images (JPG, PNG, GIF)**
```bash
--episode-strategy document_only --context-level standard
```

**📊 Large Documents (>50 pages)**
```bash
--episode-strategy contextual_chunks --context-level comprehensive
```

**📝 Short Documents (<10 pages)**
```bash
--episode-strategy hybrid --context-level rich
```

### For Different Use Cases

**🔍 Detailed Research & Analysis**
```bash
--episode-strategy contextual_chunks --context-level comprehensive
```

**⚡ Quick Ingestion & Overview**
```bash
--episode-strategy document_only --context-level minimal
```

**🎯 Balanced Approach (Recommended)**
```bash
--episode-strategy hybrid --context-level rich
```

**💾 Resource-Constrained Environments**
```bash
--episode-strategy chunk_only --context-level minimal --disable-ai-summarization
```

## 🔧 Advanced Options

### Custom Episode Naming
```bash
--episode-prefix "project_alpha_docs"
# Results in episodes named: project_alpha_docs_chunk_1, project_alpha_docs_chunk_2, etc.
```

### Disable AI Summarization
```bash
--disable-ai-summarization
# Uses basic rule-based summaries instead of AI-powered contextual summaries
```

### Combine with Traditional Options
```bash
python cli/test-document.py document.pdf --graphiti \
  --episode-strategy hybrid \
  --context-level rich \
  --chunking-strategy paragraph \
  --chunk-size 2000 \
  --chunk-overlap 300 \
  --metadata '{"project": "alpha", "version": "1.0"}'
```

## 📊 Performance Considerations

### Processing Speed (fastest to slowest)
1. `document_only` + `minimal`
2. `chunk_only` + `minimal` + `--disable-ai-summarization`
3. `hybrid` + `standard`
4. `contextual_chunks` + `rich`
5. `hybrid` + `comprehensive`

### Storage Efficiency
- **Most efficient**: `document_only`
- **Balanced**: `hybrid`
- **Most detailed**: `contextual_chunks` + `comprehensive`

### Query Performance
- **Document-level queries**: `document_only` or `hybrid`
- **Granular queries**: `contextual_chunks` or `hybrid`
- **Mixed queries**: `hybrid` (recommended)

## 🎉 Quick Start

For most use cases, start with this command:

```bash
python cli/test-graphiti.py ingest your_document.pdf
```

This uses the default hybrid strategy with rich context, providing optimal balance of granularity and contextual understanding.

## 🔗 Related Documentation

- [Hybrid Episode Strategy Implementation](packages/morag-graph/examples/hybrid_episode_mapping.py)
- [Graphiti Integration Guide](packages/morag-graph/src/morag_graph/graphiti/README.md)
- [CLI Reference](CLI.md)
