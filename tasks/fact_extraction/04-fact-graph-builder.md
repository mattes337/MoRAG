# Task 4: Fact-Based Graph Builder Implementation

## Objective
Create a new graph builder that constructs knowledge graphs based on extracted facts rather than generic entities, with relationships like Keyword→Fact←DocumentChunk←Document.

## Implementation Plan

### 4.1 Core Graph Builder Class

**File**: `packages/morag-graph/src/morag_graph/builders/fact_graph_builder.py`

```python
class FactGraphBuilder:
    """Build knowledge graphs from extracted facts."""
    
    def __init__(
        self,
        storage: BaseStorage,
        relationship_threshold: float = 0.7,
        max_relationships_per_fact: int = 5,
        enable_fact_clustering: bool = True
    ):
        """Initialize fact-based graph builder."""
        
    async def build_fact_graph(
        self, 
        facts: List[Fact], 
        document_id: str,
        chunk_mappings: Dict[str, str]
    ) -> FactGraphResult:
        """Build knowledge graph from extracted facts."""
        
    async def _store_facts(self, facts: List[Fact]) -> int:
        """Store facts in Neo4j with proper indexing."""
        
    async def _create_fact_relationships(self, facts: List[Fact]) -> List[FactRelation]:
        """Create semantic relationships between facts."""
        
    async def _link_facts_to_sources(self, facts: List[Fact], chunk_mappings: Dict[str, str]) -> None:
        """Create relationships from facts to source chunks and documents."""
        
    async def _create_keyword_index(self, facts: List[Fact]) -> None:
        """Create keyword nodes and link them to facts."""
        
    async def _cluster_related_facts(self, facts: List[Fact]) -> Dict[str, List[Fact]]:
        """Group related facts into clusters for better organization."""
```

### 4.2 Fact Relationship Detection

**File**: `packages/morag-graph/src/morag_graph/builders/fact_relationships.py`

```python
class FactRelationshipDetector:
    """Detect semantic relationships between facts."""
    
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        self.model = self._initialize_model(model_id)
        
    async def detect_relationships(self, facts: List[Fact]) -> List[FactRelation]:
        """Detect relationships between a set of facts."""
        
    async def _analyze_fact_pair(self, fact1: Fact, fact2: Fact) -> Optional[FactRelation]:
        """Analyze a pair of facts for potential relationships."""
        
    def _calculate_semantic_similarity(self, fact1: Fact, fact2: Fact) -> float:
        """Calculate semantic similarity between two facts."""
        
    async def _classify_relationship_type(self, fact1: Fact, fact2: Fact) -> Optional[str]:
        """Use LLM to classify the type of relationship between facts."""

class FactRelation(BaseModel):
    """Represents a relationship between two facts."""
    
    id: str = Field(description="Unique relationship identifier")
    source_fact_id: str = Field(description="Source fact ID")
    target_fact_id: str = Field(description="Target fact ID")
    relationship_type: str = Field(description="Type of relationship")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in relationship")
    reasoning: Optional[str] = Field(description="Explanation of the relationship")
    
    # Relationship types
    SUPPORTS = "SUPPORTS"           # One fact supports another
    ELABORATES = "ELABORATES"       # One fact provides more detail
    CONTRADICTS = "CONTRADICTS"     # Facts present conflicting information
    SEQUENCE = "SEQUENCE"           # Facts represent steps in a process
    COMPARISON = "COMPARISON"       # Facts compare different approaches
    CAUSAL = "CAUSAL"              # One fact causes or leads to another
    TEMPORAL = "TEMPORAL"          # Facts have temporal relationship
```

### 4.3 Graph Storage Operations

**File**: `packages/morag-graph/src/morag_graph/storage/neo4j_operations/fact_operations.py`

```python
class FactOperations:
    """Neo4j operations for fact storage and retrieval."""
    
    def __init__(self, driver):
        self.driver = driver
        
    async def store_fact(self, fact: Fact) -> str:
        """Store a single fact in Neo4j."""
        
    async def store_facts_batch(self, facts: List[Fact]) -> int:
        """Store multiple facts efficiently."""
        
    async def create_fact_relationship(self, relation: FactRelation) -> None:
        """Create relationship between two facts."""
        
    async def link_fact_to_chunk(self, fact_id: str, chunk_id: str) -> None:
        """Link fact to its source document chunk."""
        
    async def create_keyword_links(self, fact_id: str, keywords: List[str]) -> None:
        """Create links from fact to keyword nodes."""
        
    async def search_facts(
        self, 
        query: str, 
        fact_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Fact]:
        """Search for facts based on content and metadata."""
        
    async def get_related_facts(self, fact_id: str, max_depth: int = 2) -> List[Fact]:
        """Get facts related to a given fact through graph relationships."""
```

### 4.4 Neo4j Schema for Facts

**Cypher Schema**:
```cypher
// Create constraints and indexes
CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT keyword_name_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;

// Create indexes for performance
CREATE INDEX fact_subject_index IF NOT EXISTS FOR (f:Fact) ON (f.subject);
CREATE INDEX fact_type_index IF NOT EXISTS FOR (f:Fact) ON (f.fact_type);
CREATE INDEX fact_domain_index IF NOT EXISTS FOR (f:Fact) ON (f.domain);
CREATE INDEX fact_confidence_index IF NOT EXISTS FOR (f:Fact) ON (f.confidence);

// Node labels and properties
(:Fact {
    id: string,
    subject: string,
    object: string,
    approach: string?,
    solution: string?,
    remarks: string?,
    fact_type: string,
    domain: string?,
    confidence: float,
    language: string,
    created_at: datetime,
    keywords: string  // comma-separated for simple queries
})

(:Keyword {
    name: string,
    frequency: int,
    importance: float
})

(:Domain {
    name: string,
    description: string
})

// Relationships
(:Fact)-[:EXTRACTED_FROM]->(:DocumentChunk)
(:Fact)-[:TAGGED_WITH]->(:Keyword)
(:Fact)-[:BELONGS_TO]->(:Domain)
(:Fact)-[:SUPPORTS]->(:Fact)
(:Fact)-[:ELABORATES]->(:Fact)
(:Fact)-[:CONTRADICTS]->(:Fact)
(:Fact)-[:SEQUENCE]->(:Fact)
(:Fact)-[:COMPARISON]->(:Fact)
(:Fact)-[:CAUSAL]->(:Fact)
(:Fact)-[:TEMPORAL]->(:Fact)
```

## Implementation Tasks

### Task 4.1: Core Graph Builder
- [ ] Implement FactGraphBuilder class with basic functionality
- [ ] Create fact storage operations in Neo4j
- [ ] Implement batch processing for efficient fact storage
- [ ] Add error handling and transaction management

### Task 4.2: Relationship Detection
- [ ] Implement FactRelationshipDetector with LLM integration
- [ ] Create relationship classification prompts
- [ ] Add semantic similarity calculation
- [ ] Implement relationship validation and filtering

### Task 4.3: Graph Schema and Operations
- [ ] Create Neo4j schema for facts and relationships
- [ ] Implement fact storage operations
- [ ] Add keyword indexing and linking
- [ ] Create domain categorization system

### Task 4.4: Source Linking and Provenance
- [ ] Implement fact-to-chunk linking
- [ ] Create chunk-to-document relationships
- [ ] Add provenance tracking throughout the pipeline
- [ ] Implement source attribution queries

### Task 4.5: Query and Retrieval
- [ ] Implement fact search functionality
- [ ] Create related fact discovery
- [ ] Add fact clustering and organization
- [ ] Implement performance optimization

## LLM Prompts for Relationship Detection

### Relationship Classification Prompt
```
Analyze the relationship between these two facts:

Fact 1: {fact1_json}
Fact 2: {fact2_json}

Determine if there is a meaningful relationship and classify it:

Relationship Types:
- SUPPORTS: Fact 2 provides evidence or support for Fact 1
- ELABORATES: Fact 2 provides more detail or explanation about Fact 1
- CONTRADICTS: Facts present conflicting or opposing information
- SEQUENCE: Facts represent steps in a process or temporal sequence
- COMPARISON: Facts compare different approaches, methods, or outcomes
- CAUSAL: Fact 1 causes or leads to Fact 2
- TEMPORAL: Facts have a time-based relationship

Respond with JSON:
{
  "has_relationship": true/false,
  "relationship_type": "TYPE" or null,
  "confidence": 0.0-1.0,
  "reasoning": "explanation of the relationship",
  "direction": "fact1_to_fact2" or "fact2_to_fact1" or "bidirectional"
}

Only identify relationships that are clearly supported by the fact content.
```

### Fact Clustering Prompt
```
Group these facts into related clusters based on their semantic similarity and topical relevance:

Facts: {facts_list}

Create clusters where facts within each cluster:
- Share similar subjects or objects
- Relate to the same process or methodology
- Describe different aspects of the same phenomenon
- Form a logical sequence or workflow

Respond with JSON:
{
  "clusters": [
    {
      "cluster_id": "unique_id",
      "theme": "brief description of cluster theme",
      "fact_ids": ["fact1", "fact2", ...],
      "confidence": 0.0-1.0
    }
  ]
}
```

## Performance Considerations

### Batch Processing
- Process facts in batches to reduce database round trips
- Use Neo4j transactions for consistency
- Implement parallel processing where possible

### Indexing Strategy
- Create indexes on frequently queried properties
- Use composite indexes for complex queries
- Monitor query performance and optimize as needed

### Memory Management
- Stream large fact sets instead of loading all into memory
- Implement pagination for fact retrieval
- Use connection pooling for database operations

## Success Criteria

1. **Graph Quality**: Facts are properly stored with meaningful relationships
2. **Performance**: Efficient storage and retrieval of facts
3. **Scalability**: Can handle large numbers of facts and relationships
4. **Accuracy**: Relationship detection is precise and meaningful
5. **Integration**: Seamlessly works with existing document processing pipeline
