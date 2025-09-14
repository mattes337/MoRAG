# Task 2: Fact-Based Knowledge Extraction Design

## Objective
Design a new fact-based approach that extracts structured, actionable knowledge instead of generic entities and relations.

## Core Concept: Facts vs. Entities

### Current Approach (Entity-Relation)
```
Entity: "John Doe" (PERSON)
Entity: "Stanford University" (ORGANIZATION)
Relation: John Doe -> WORKS_AT -> Stanford University
```

### New Approach (Fact-Based)
```
Fact: {
  subject: "John Doe",
  object: "machine learning research",
  approach: "neural network architectures",
  solution: "improved accuracy in image recognition",
  remarks: "published in Nature 2024",
  source: "document_chunk_123",
  confidence: 0.95
}
```

## Fact Schema Design

### Core Fact Structure
```python
class Fact(BaseModel):
    id: str = Field(description="Unique fact identifier")
    subject: str = Field(description="What the fact is about")
    object: str = Field(description="What is being described or acted upon")
    approach: Optional[str] = Field(description="How something is done/achieved")
    solution: Optional[str] = Field(description="What solves a problem/achieves goal")
    remarks: Optional[str] = Field(description="Additional context/qualifications")
    
    # Provenance
    source_chunk_id: str = Field(description="Source document chunk")
    source_document: str = Field(description="Source document identifier")
    extraction_confidence: float = Field(description="Confidence in extraction")
    
    # Metadata
    domain: Optional[str] = Field(description="Domain/topic area")
    fact_type: str = Field(description="Type of fact (research, process, definition, etc.)")
    keywords: List[str] = Field(description="Key terms for indexing")
    created_at: datetime = Field(description="Extraction timestamp")
```

### Fact Types
1. **Research Facts**: Findings, methodologies, results
2. **Process Facts**: How-to information, procedures, workflows
3. **Definition Facts**: What something is, characteristics
4. **Causal Facts**: Cause-effect relationships
5. **Comparative Facts**: Comparisons, evaluations, rankings
6. **Temporal Facts**: Time-based information, sequences

## Graph Structure Design

### Neo4j Schema
```cypher
// Core nodes
(:Fact {id, subject, object, approach, solution, remarks, confidence, fact_type})
(:DocumentChunk {id, content, index, document_id})
(:Document {id, title, source, checksum})
(:Keyword {name, frequency, importance})
(:Domain {name, description})

// Relationships
(:Fact)-[:EXTRACTED_FROM]->(:DocumentChunk)
(:DocumentChunk)-[:PART_OF]->(:Document)
(:Fact)-[:TAGGED_WITH]->(:Keyword)
(:Fact)-[:BELONGS_TO]->(:Domain)
(:Fact)-[:RELATES_TO]->(:Fact)  // Semantic relationships between facts
```

### Relationship Types
- `EXTRACTED_FROM`: Fact to source chunk (provenance)
- `TAGGED_WITH`: Fact to keywords (indexing)
- `BELONGS_TO`: Fact to domain (categorization)
- `RELATES_TO`: Fact to fact (semantic connections)
- `SUPPORTS`: One fact supports another
- `CONTRADICTS`: Facts that contradict each other
- `ELABORATES`: One fact provides more detail on another

## Extraction Pipeline Design

### Stage 1: Document Processing
1. **Input Formatter**: Convert documents to Markdown (existing)
2. **Analyzer Agent**: Extract metadata, domain, keywords (existing)
3. **Chunker**: Split into semantic chunks (existing)

### Stage 2: Fact Extraction (NEW)
```python
class FactExtractor:
    def extract_facts(self, chunk: str, context: Dict) -> List[Fact]:
        """Extract structured facts from document chunk."""
        
    def _identify_fact_candidates(self, text: str) -> List[str]:
        """Identify sentences/passages that contain factual information."""
        
    def _structure_fact(self, candidate: str, context: Dict) -> Optional[Fact]:
        """Structure candidate text into fact schema."""
        
    def _validate_fact(self, fact: Fact) -> bool:
        """Validate that extracted fact meets quality criteria."""
```

### Stage 3: Fact Graph Building (NEW)
```python
class FactGraphBuilder:
    def build_fact_graph(self, facts: List[Fact]) -> GraphResult:
        """Build knowledge graph from extracted facts."""
        
    def _create_fact_relationships(self, facts: List[Fact]) -> List[Relation]:
        """Create semantic relationships between facts."""
        
    def _index_facts(self, facts: List[Fact]) -> None:
        """Create keyword and domain indexes for facts."""
```

## LLM Prompting Strategy

### Fact Extraction Prompt
```
Extract structured facts from the following text. A fact should contain actionable, specific information that answers questions like:
- What is being done? (subject + approach)
- What is the outcome? (solution)
- What is being studied/described? (object)
- What additional context matters? (remarks)

Text: {chunk_text}

For each fact found, provide:
1. Subject: The main entity or concept the fact is about
2. Object: What is being described, studied, or acted upon
3. Approach: The method, technique, or way something is done (if applicable)
4. Solution: The result, outcome, or answer provided (if applicable)
5. Remarks: Important context, limitations, or qualifications

Only extract facts that are:
- Specific and actionable
- Verifiable from the text
- Useful for answering questions
- Not generic statements

Domain: {domain}
```

### Fact Relationship Prompt
```
Given these facts from the same document, identify semantic relationships:

Facts: {facts_list}

Identify relationships like:
- SUPPORTS: One fact provides evidence for another
- ELABORATES: One fact provides more detail about another
- CONTRADICTS: Facts that present conflicting information
- SEQUENCE: Facts that represent steps in a process
- COMPARISON: Facts that compare different approaches/solutions

Only create relationships that are clearly supported by the text.
```

## Quality Criteria

### Fact Quality Metrics
1. **Specificity**: Avoids generic statements
2. **Actionability**: Provides useful, applicable information
3. **Verifiability**: Can be traced back to source text
4. **Completeness**: Contains sufficient context to be standalone
5. **Relevance**: Pertains to the document's domain/purpose

### Filtering Rules
- Minimum confidence threshold (e.g., 0.7)
- Reject facts with generic subjects ("it", "this", "that")
- Require at least subject + object or subject + solution
- Filter out meta-statements about the document itself

## Integration Points

### With Existing System
1. **Document Processing**: Reuse existing pipeline up to chunking
2. **Vector Storage**: Store fact embeddings in Qdrant alongside chunk embeddings
3. **API Endpoints**: Extend existing endpoints to support fact-based queries
4. **Retrieval**: Hybrid retrieval using both facts and chunks

### Migration Strategy
1. **Parallel Implementation**: Run fact extraction alongside entity extraction
2. **A/B Testing**: Compare retrieval quality between approaches
3. **Gradual Migration**: Phase out entity extraction once fact approach is validated
4. **Backward Compatibility**: Maintain existing API contracts during transition

## Success Metrics

1. **Graph Size Reduction**: 50-70% fewer nodes while maintaining information quality
2. **Retrieval Precision**: Improved relevance of retrieved information
3. **Query Performance**: Faster queries due to more focused graph structure
4. **User Satisfaction**: Better answers to domain-specific questions
