# Fact-Based Knowledge Extraction Pipeline

## Overview

This task folder contains the implementation plan for transitioning from the current generic entity-relation extraction approach to a more focused fact-based knowledge extraction system. The current approach has several limitations that make the knowledge graph too broad and generic, leading to poor retrieval performance and actionable insights.

## Current Issues Analysis

### 1. Language Word Removal Problems
- **Location**: `packages/morag-graph/src/morag_graph/extraction/relation_extractor.py:502-505`
- **Issue**: Hardcoded language word removal is too aggressive and language-specific
- **Code**: 
```python
words_to_remove = [
    'sich', 'zu', 'der', 'die', 'das', 'den', 'dem', 'des',
    'the', 'a', 'an', 'to', 'of', 'for', 'with', 'by'
]
```
- **Impact**: Removes meaningful parts of entity names, causing incorrect entity matching

### 2. Entity Type Labeling Issues
- **Location**: Entity extraction and storage operations
- **Issue**: All entities are getting labeled as "ORGANIZATION" regardless of their actual type
- **Root Cause**: 
  - Default entity type assignment in auto-creation logic
  - Insufficient type determination in LLM extraction prompts
  - Type normalization logic that defaults to generic labels
- **Impact**: Loss of semantic meaning and poor query performance

### 3. Graph Becoming Too Large and Generic
- **Issue**: Current entity-relation approach creates too many generic entities and relationships
- **Problems**:
  - Generic entity types like "CUSTOM", "ENTITY", "THING"
  - Overly broad relationship types
  - No focus on actionable knowledge
  - Poor signal-to-noise ratio in retrieval

## Proposed Fact-Based Solution

### Core Concept
Instead of extracting generic entities and relations, extract structured **facts** that represent actionable knowledge:

- **Subject**: What the fact is about
- **Object**: What is being described or acted upon
- **Approach**: How something is done or achieved
- **Solution**: What solves a problem or achieves a goal
- **Remarks**: Additional context or qualifications

### Benefits
1. **Focused Knowledge**: Facts are inherently more actionable than generic entities
2. **Better Provenance**: Each fact links directly to source chunks with full context
3. **Semantic Clarity**: Facts have clear semantic structure
4. **Reduced Noise**: Only meaningful, structured information is extracted
5. **Better Retrieval**: Query for specific types of facts rather than generic entities

## Implementation Tasks

1. **Current Issues Analysis** - Document and understand current limitations
2. **Fact-Based Design** - Design the new fact extraction architecture
3. **Fact Extractor** - Implement core fact extraction component
4. **Fact Graph Builder** - Build graphs based on facts, not generic entities
5. **Fact Storage** - Store and index facts in Neo4j with proper relationships
6. **System Integration** - Integrate with existing MoRAG pipeline
7. **Testing & Validation** - Ensure the new approach produces better results

## Expected Outcomes

- **Smaller, More Focused Graphs**: Only meaningful facts are stored
- **Better Retrieval Performance**: Query specific fact types instead of generic entities
- **Actionable Knowledge**: Facts provide direct answers to user questions
- **Improved Provenance**: Clear links from facts to source documents and chunks
- **Domain Adaptability**: Fact types can be customized per domain

## Migration Strategy

1. Implement fact extraction alongside current system
2. Compare results and performance
3. Gradually migrate to fact-based approach
4. Deprecate generic entity extraction once validated
