# MoRAG Agents

## Core LLM Agents

### 1. PathSelectionAgent
**Purpose**: Selects optimal reasoning paths through knowledge graph for multi-hop queries
**Input**: Query string, available graph paths, strategy (bidirectional/forward/backward)
**Output**: Scored and ranked reasoning paths

```json
{
  "query": "How are Apple's AI research efforts related to their partnership with universities?",
  "selected_paths": [
    {
      "path_id": "path_001",
      "relevance_score": 0.92,
      "entities": ["Apple Inc.", "AI research", "Stanford University"],
      "relations": ["PARTNERS_WITH", "CONDUCTS_RESEARCH"]
    }
  ]
}
```

### 2. IterativeRetriever
**Purpose**: Refines context through iterative analysis until sufficiency threshold reached
**Input**: Query, initial context, max_iterations, sufficiency_threshold
**Output**: Refined context with analysis metadata

```json
{
  "final_context": "...",
  "analysis": {
    "is_sufficient": true,
    "confidence": 0.85,
    "gaps": [],
    "iterations_performed": 3
  }
}
```

### 3. SemanticChunkingAgent
**Purpose**: Intelligently chunks text based on semantic boundaries
**Input**: Text content, chunking strategy (semantic/hybrid/size-based), max/min chunk sizes
**Output**: List of semantically coherent text chunks

```json
{
  "chunks": [
    "First semantic chunk...",
    "Second semantic chunk..."
  ],
  "boundaries": [
    {"position": 150, "confidence": 0.8, "topic_change": true}
  ]
}
```

### 4. SummarizationAgent
**Purpose**: Creates structured summaries with key points and confidence scores
**Input**: Text, max_length, style (brief/detailed), context
**Output**: Summary with metadata

```json
{
  "summary": "Main content summary...",
  "key_points": ["Point 1", "Point 2"],
  "confidence": 0.9,
  "compression_ratio": 0.15
}
```

### 5. QueryAnalysisAgent
**Purpose**: Analyzes user queries to extract intent and entities
**Input**: Query string, context
**Output**: Structured query analysis

```json
{
  "intent": "factual_retrieval",
  "entities": ["Apple", "AI research"],
  "query_type": "multi_hop",
  "complexity": "medium"
}
```

### 6. FactExtractionAgent
**Purpose**: Extracts structured facts from text chunks
**Input**: Text chunk, domain context, max_facts
**Output**: List of extracted facts with confidence

```json
{
  "facts": [
    {
      "subject": "Dr. Smith",
      "predicate": "prescribed",
      "object": "aspirin",
      "confidence": 0.95,
      "source_chunk_id": "chunk_001"
    }
  ]
}
```

### 7. EntityExtractionAgent
**Purpose**: Identifies and extracts entities from text
**Input**: Text, domain, auto_infer_domain flag
**Output**: List of entities with types and confidence

```json
{
  "entities": [
    {
      "name": "Apple Inc.",
      "type": "ORGANIZATION",
      "confidence": 0.98,
      "attributes": {"domain": "technology"}
    }
  ]
}
```

### 8. RelationExtractionAgent
**Purpose**: Extracts semantic relationships between entities
**Input**: Text, extracted entities, domain
**Output**: List of relations with types

```json
{
  "relations": [
    {
      "source": "Apple Inc.",
      "target": "Stanford University",
      "type": "PARTNERS_WITH",
      "confidence": 0.87
    }
  ]
}
```

## Configuration

All agents use LLMConfig:
```json
{
  "provider": "gemini",
  "model": "gemini-1.5-flash",
  "temperature": 0.1,
  "max_tokens": 2000,
  "max_retries": 5
}
```

Environment variables:
- MORAG_LLM_PROVIDER
- MORAG_GEMINI_MODEL
- GEMINI_API_KEY
- MORAG_LLM_TEMPERATURE
- MORAG_LLM_MAX_TOKENS
