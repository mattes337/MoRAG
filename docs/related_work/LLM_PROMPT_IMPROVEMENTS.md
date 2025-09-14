# LLM Prompt Improvements from RAG-Anything + Ottomator

## Overview

This analysis focuses specifically on LLM prompt engineering techniques from RAG-Anything and Ottomator that can improve MoRAG's current prompting system. We exclude multimodal aspects and temporal awareness, concentrating on text-based prompt optimization and agent-based decision making.

## Current MoRAG Prompt Analysis

### Current Fact Extraction Prompt Structure
```python
# MoRAG's current fact extraction approach
"""
You are a knowledge extraction expert. Extract structured facts from the following text that represent specific, domain-relevant information that can be used to answer questions or understand concepts.

Extract facts with this structure:
{
  "subject": "specific substance/entity",
  "object": "specific condition/problem/target", 
  "approach": "exact method/dosage/procedure with specific details",
  "solution": "specific outcome/benefit/result",
  "condition": "question/precondition/situation when this applies",
  "remarks": "safety warnings/contraindications/context",
  "fact_type": "process|definition|causal|methodological|safety",
  "confidence": 0.0-1.0,
  "keywords": ["domain-specific", "technical", "terms"]
}
"""
```

### Current Entity Extraction Prompt
```python
# MoRAG's current entity extraction
"""
Extract entities from the text in order of appearance.
Focus on important entities like people, organizations, locations, concepts, and objects.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.
"""
```

### Current Relationship Extraction Prompt
```python
# MoRAG's current relationship extraction
"""
Identify relationships like:
- SUPPORTS: One fact provides evidence for another
- ELABORATES: One fact provides more detail about another
- CONTRADICTS: Facts that present conflicting information
- SEQUENCE: Facts that represent steps in a process
- COMPARISON: Facts that compare different approaches/solutions
- CAUSATION: One fact describes the cause of another
- TEMPORAL_ORDER: Facts that have a time-based sequence
"""
```

## Key Prompt Improvement Opportunities

### 1. Agent-Based Query Analysis (from Ottomator)

**Current Problem**: MoRAG uses static prompts without query intent analysis
**Improvement**: Add intelligent query analysis before fact extraction

**Enhanced Query Analysis Prompt**:
```python
"""
Analyze the user query to optimize fact extraction strategy:

Query: {user_query}
Context: {domain_context}

Determine:
1. Query Intent:
   - factual_lookup: Simple fact retrieval
   - analytical: Requires reasoning and connections
   - comparative: Needs comparison between entities/facts
   - causal: Focuses on cause-effect relationships
   - sequential: Process or step-based information

2. Extraction Focus:
   - entity_types: [list of relevant entity types]
   - fact_types: [prioritized fact types for this query]
   - relationship_depth: shallow|medium|deep
   - complexity_level: simple|moderate|complex

3. Search Strategy:
   - primary_approach: vector|graph|hybrid
   - confidence_threshold: 0.0-1.0
   - max_facts_needed: estimated number

Respond with JSON:
{
  "query_intent": "factual_lookup|analytical|comparative|causal|sequential",
  "extraction_focus": {
    "entity_types": ["person", "organization", "concept"],
    "fact_types": ["causal", "process", "definition"],
    "relationship_depth": "medium",
    "complexity_level": "moderate"
  },
  "search_strategy": {
    "primary_approach": "hybrid",
    "confidence_threshold": 0.7,
    "max_facts_needed": 15
  },
  "reasoning": "explanation for the analysis"
}
"""
```

### 2. Enhanced Fact Extraction with Context Awareness

**Current Problem**: Static fact extraction without query context
**Improvement**: Query-aware fact extraction with relevance scoring

**Enhanced Fact Extraction Prompt**:
```python
"""
Extract facts optimized for this specific query context:

Query Context: {query_analysis}
User Query: {user_query}
Text Chunk: {chunk_text}
Domain: {domain}

Based on the query analysis, focus on extracting facts that are:
- Directly relevant to the query intent: {query_intent}
- Match prioritized fact types: {prioritized_fact_types}
- Support the identified search strategy: {search_strategy}

For each fact, provide:
{
  "subject": "specific entity/concept",
  "object": "target/condition/problem",
  "approach": "method/procedure/mechanism",
  "solution": "outcome/result/effect",
  "condition": "context/precondition/scope",
  "remarks": "additional context/warnings/notes",
  "fact_type": "process|definition|causal|methodological|comparative|temporal",
  "confidence": 0.0-1.0,
  "query_relevance": 0.0-1.0,  # NEW: relevance to specific query
  "evidence_strength": "strong|moderate|weak",  # NEW: evidence quality
  "keywords": ["domain-specific", "technical", "terms"],
  "source_span": "exact text span supporting this fact"  # NEW: traceability
}

Prioritize facts with high query_relevance and strong evidence_strength.
Extract {max_facts} most relevant facts.
"""
```

### 3. Enhanced Relationship Extraction (from Ottomator)

**Current Problem**: Basic relationship extraction without context awareness
**Improvement**: Add relationship strength assessment and evidence quality

**Enhanced Relationship Extraction Prompt**:
```python
"""
Extract relationships between facts with enhanced context awareness:

Facts: {facts_list}
Query Context: {query_analysis}

For each relationship, consider:
1. Relationship Strength:
   - direct: explicitly stated in text
   - inferred: logically derivable
   - contextual: based on domain knowledge

2. Evidence Quality:
   - explicit: clearly stated
   - implicit: requires inference
   - speculative: uncertain/hypothetical

3. Logical Structure:
   - sequential: logical or process sequence
   - hierarchical: parent-child or containment
   - associative: related concepts or entities

Relationship Types (enhanced):
- SUPPORTS: One fact provides evidence for another
- ELABORATES: Provides additional detail/context
- CONTRADICTS: Conflicting information
- SEQUENCE: Logical or process sequence
- CAUSATION: Cause-effect relationship
- COMPARISON: Comparative analysis
- PREREQUISITE: Required condition/dependency
- ALTERNATIVE: Alternative approach/solution
- HIERARCHY: Parent-child or containment relationship

Respond with JSON:
[
  {
    "source_fact_id": "fact_id_1",
    "target_fact_id": "fact_id_2",
    "relation_type": "SUPPORTS|ELABORATES|CONTRADICTS|SEQUENCE|CAUSATION|COMPARISON|PREREQUISITE|ALTERNATIVE|HIERARCHY",
    "confidence": 0.0-1.0,
    "relationship_strength": "direct|inferred|contextual",
    "evidence_quality": "explicit|implicit|speculative",
    "logical_structure": "sequential|hierarchical|associative",
    "context": "explanation of the relationship",
    "source_evidence": "text span supporting this relationship"
  }
]
"""
```

### 4. Enhanced Entity Extraction with Normalization

**Current Problem**: Basic entity extraction without normalization
**Improvement**: Add entity normalization and disambiguation

**Enhanced Entity Extraction Prompt**:
```python
"""
Extract and normalize entities with disambiguation:

Text: {chunk_text}
Query Context: {query_analysis}
Domain: {domain}

For each entity:
1. Extract exact mention from text
2. Determine canonical form (normalized name)
3. Identify entity type and attributes
4. Assess relevance to query context

Entity Types (prioritized for this query):
{prioritized_entity_types}

For each entity, provide:
{
  "mention": "exact text as it appears",
  "canonical_name": "normalized/standardized name",
  "entity_type": "person|organization|location|concept|technology|event|process",
  "attributes": {
    "role": "function/position/purpose",
    "domain": "field/area of relevance",
    "specificity": "general|specific|highly_specific"
  },
  "confidence": 0.0-1.0,
  "query_relevance": 0.0-1.0,
  "disambiguation": "context for distinguishing from similar entities",
  "aliases": ["alternative names/forms"],
  "context_scope": "scope or domain where this entity is relevant",
  "source_span": "exact text span"
}

Focus on entities with high query_relevance.
Normalize similar entities to the same canonical_name.
"""
```

### 5. Intelligent Retrieval Strategy Selection (from Ottomator)

**Current Problem**: Fixed recursive fact traversal strategy
**Improvement**: Dynamic strategy selection based on query analysis

**Strategy Selection Prompt**:
```python
"""
Select optimal retrieval strategy based on query analysis:

Query Analysis: {query_analysis}
Available Strategies:
1. vector_search: Best for semantic similarity and content retrieval
2. graph_traversal: Best for relationship exploration and connected knowledge
3. hybrid_approach: Best for complex queries requiring both approaches
4. fact_focused: Best for specific factual information
5. entity_focused: Best for entity-centric exploration

Consider:
- Query complexity: {query_complexity}
- Information type needed: {info_type}
- Relationship depth required: {relationship_depth}
- Domain specificity: {domain_specificity}

Respond with JSON:
{
  "selected_strategy": "vector_search|graph_traversal|hybrid_approach|fact_focused|entity_focused",
  "strategy_parameters": {
    "max_depth": 1-5,
    "confidence_threshold": 0.0-1.0,
    "max_results": 10-100,
    "domain_filter": "general|specific|highly_specific",
    "relationship_types": ["list of relevant relationship types"]
  },
  "reasoning": "explanation for strategy selection",
  "fallback_strategy": "backup strategy if primary fails"
}
"""
```

### 6. Enhanced Fact Validation and Scoring

**Current Problem**: Basic confidence scoring without validation
**Improvement**: Multi-dimensional fact validation

**Enhanced Fact Validation Prompt**:
```python
"""
Validate and score extracted facts with multiple dimensions:

Fact: {fact_json}
Source Text: {source_text}
Query Context: {query_context}

Evaluate on multiple dimensions:

1. Factual Accuracy (0.0-1.0):
   - Is the fact correctly extracted from source?
   - Are all components (subject, object, approach, solution) accurate?
   - Is there any misinterpretation or hallucination?

2. Query Relevance (0.0-1.0):
   - How relevant is this fact to the user query?
   - Does it help answer the question or provide useful context?
   - Is it central or peripheral to the query intent?

3. Completeness (0.0-1.0):
   - Does the fact contain sufficient context to be useful?
   - Are all necessary components present?
   - Is additional context needed for understanding?

4. Actionability (0.0-1.0):
   - Can this fact be used to make decisions or take action?
   - Is it specific enough to be practically useful?
   - Does it provide clear guidance or information?

5. Evidence Strength (0.0-1.0):
   - How well is this fact supported by the source text?
   - Is the evidence explicit or inferred?
   - Are there any contradictions or uncertainties?

6. Domain Relevance (0.0-1.0):
   - How relevant is this fact to the specified domain?
   - Does it fit within the domain context and scope?
   - Is it appropriately specific for the domain level?

Respond with JSON:
{
  "overall_score": 0.0-1.0,
  "dimension_scores": {
    "factual_accuracy": 0.0-1.0,
    "query_relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "actionability": 0.0-1.0,
    "evidence_strength": 0.0-1.0,
    "domain_relevance": 0.0-1.0
  },
  "validation_issues": ["list of specific issues found"],
  "improvement_suggestions": ["specific suggestions for enhancement"],
  "confidence_factors": ["factors that increase/decrease confidence"],
  "domain_context": "domain scope and applicability of this fact"
}
"""
```

## Implementation Roadmap

### Phase 1: Query Analysis Enhancement (1-2 weeks)
1. **Implement Query Intent Analysis**
   - Add query analysis prompt before fact extraction
   - Create query classification system
   - Implement strategy selection logic

2. **Enhance Fact Extraction Context**
   - Modify fact extraction to use query context
   - Add query relevance scoring
   - Implement temporal awareness

### Phase 2: Advanced Relationship Processing (2-3 weeks)
1. **Enhanced Relationships**
   - Add relationship strength assessment
   - Implement evidence quality evaluation
   - Add logical structure classification

2. **Enhanced Entity Normalization**
   - Implement entity disambiguation
   - Add canonical name normalization
   - Create entity attribute extraction

### Phase 3: Intelligent Strategy Selection (1-2 weeks)
1. **Dynamic Strategy Selection**
   - Implement strategy selection based on query analysis
   - Add fallback mechanisms
   - Create strategy effectiveness tracking

2. **Enhanced Validation**
   - Implement multi-dimensional fact validation
   - Add comprehensive scoring system
   - Create validation feedback loops

### Phase 4: Integration and Optimization (1 week)
1. **System Integration**
   - Integrate all enhanced prompts into existing services
   - Add configuration options for prompt selection
   - Implement A/B testing for prompt effectiveness

2. **Performance Optimization**
   - Optimize prompt length and complexity
   - Add caching for query analysis results
   - Implement prompt template management

## Expected Improvements

### Quantitative Benefits
- **Query Relevance**: 20-30% improvement in fact relevance to user queries
- **Fact Quality**: 15-25% improvement in fact accuracy and completeness
- **Relationship Accuracy**: 25-35% improvement in relationship detection
- **Response Time**: 10-15% improvement through better strategy selection

### Qualitative Benefits
- **Context Sensitivity**: More relevant fact extraction based on query intent
- **Evidence Quality**: Better assessment of fact reliability and strength
- **Transparency**: Clear reasoning for strategy and fact selection
- **Adaptability**: Dynamic adjustment to different query types and domains

## Technical Considerations

### Prompt Management
- Create prompt template system for easy updates
- Implement version control for prompt changes
- Add A/B testing framework for prompt optimization

### Performance Impact
- Query analysis adds one additional LLM call per query
- Enhanced prompts are longer but provide better results
- Caching can mitigate repeated analysis costs

### Backward Compatibility
- Maintain existing API interfaces
- Add new features as optional enhancements
- Provide configuration flags for prompt selection
