# Multi-Hop Reasoning Strategies

## Core Reasoning Approaches

### 1. Forward Chaining
**Strategy**: Start from query entities and explore forward through relationships
**Use Case**: When you know starting entities and want to discover what they lead to
**Max Depth**: 4 hops
**Bidirectional**: False

**Configuration**: Uses ReasoningStrategy with forward chaining enabled, maximum depth of 4 hops, and relationship weights for scoring.

**Example**: For query "What products does Apple make?", starts with "Apple Inc." entity and explores forward through MANUFACTURES relationships to discover iPhone, iPad, MacBook products.

### 2. Backward Chaining
**Strategy**: Start from potential answers and work backward to query entities
**Use Case**: When you know target entities and want to find connections
**Max Depth**: 3 hops
**Bidirectional**: False

**Configuration**: Uses ReasoningStrategy with backward chaining enabled, maximum depth of 3 hops, and relationship weights for scoring.

**Example**: For query "Who invented the iPhone?", starts with "iPhone" entity and explores backward through INVENTED_BY and WORKED_AT relationships to discover Steve Jobs and Apple connections.

### 3. Bidirectional Search
**Strategy**: Search from both query entities and target entities, meet in middle
**Use Case**: Complex multi-hop queries with known start and end points
**Max Depth**: 5 hops
**Bidirectional**: True

**Configuration**: Uses ReasoningStrategy with bidirectional search enabled, maximum depth of 5 hops, and relationship weights for scoring.

**Example**: For query "How is Apple's AI research connected to Stanford University?", starts from both "Apple Inc./AI research" and "Stanford University" entities, exploring until paths meet in the middle through PARTNERS_WITH relationships.

## Path Selection Algorithm

### LLM-Based Path Ranking
**Process**: Uses LLM analysis to rank and select the most relevant reasoning paths for a given query.

**Steps**:
1. **Path Selection Prompt**: Create structured prompt with query and available paths
2. **LLM Analysis**: Generate response using low temperature (0.1) for consistency, max 1000 tokens
3. **Response Parsing**: Extract selected paths from LLM response
4. **Strategy Scoring**: Apply additional scoring based on reasoning strategy
5. **Ranking**: Sort paths by relevance score in descending order

### Path Scoring Criteria
1. **Semantic Relevance**: How well path entities relate to query
2. **Path Coherence**: Logical flow of relationships
3. **Entity Importance**: Centrality and confidence of entities
4. **Relationship Strength**: Confidence scores of connections
5. **Path Length**: Shorter paths often more reliable

## Iterative Context Refinement

### Refinement Process
**Purpose**: Iteratively improve context quality until sufficiency threshold is reached.

**Process**:
1. **Context Analysis**: Analyze current context sufficiency for the query
2. **Sufficiency Check**: Determine if context meets confidence threshold requirements
3. **Gap Identification**: Identify missing information types and knowledge gaps
4. **Additional Retrieval**: Retrieve supplementary information based on identified gaps
5. **Context Merging**: Combine new context with existing information
6. **Iteration Control**: Continue until max iterations reached or sufficiency achieved

### Context Analysis Criteria
```json
{
  "is_sufficient": "boolean - can query be answered with current context",
  "confidence": "float 0.0-1.0 - confidence in sufficiency assessment",
  "gaps": ["list of missing information types"],
  "reasoning": "explanation of analysis",
  "entities_coverage": "percentage of query entities covered",
  "relationship_density": "how well connected the entities are",
  "information_depth": "level of detail available"
}
```

## Graph Traversal Patterns

### Breadth-First Search (BFS)
**Use Case**: Finding all entities within a specific distance
**Pattern**: Explore all neighbors at distance 1, then distance 2, etc.
**Implementation**: Uses graph traversal to find distinct neighbors within specified hop distance, ordered by distance and name.

### Depth-First Search (DFS)
**Use Case**: Following specific relationship chains
**Pattern**: Follow one path deeply before exploring alternatives
**Implementation**: Traverses paths from start entity to end entities, returning complete path information including nodes and relationships, ordered by path length.

### Shortest Path Finding
**Use Case**: Finding most direct connections between entities
**Pattern**: Use graph database shortest path algorithms
**Implementation**: Finds shortest paths between source and target entities within specified hop limits, returning path details including entity and relationship information, ordered by path length.

## Intelligent Retrieval Workflow

### Step-by-Step Process
1. **Entity Identification**: Extract entities from user query
2. **Initial Path Discovery**: Find potential reasoning paths
3. **Path Selection**: Use LLM to rank and select best paths
4. **Recursive Exploration**: Follow selected paths iteratively
5. **Context Expansion**: Gather related entities and facts
6. **Fact Extraction**: Extract relevant facts from discovered content
7. **Response Synthesis**: Generate final answer with citations

### Entity Identification Service
**Purpose**: Extract and link entities from user queries to knowledge graph entities.

**Process**:
1. **Entity Extraction**: Use NER (Named Entity Recognition) to identify entities in query text
2. **Entity Linking**: Match extracted entities to existing knowledge graph entities
3. **Graph Lookup**: Find corresponding graph entities by name matching
4. **Result Compilation**: Return list of linked entities for further processing

### Recursive Path Following
**Purpose**: Iteratively explore graph paths based on LLM decisions until termination criteria met.

**Process**:
1. **Path Decision**: Use LLM to decide which entities to explore in current iteration
2. **Neighbor Discovery**: Find neighboring entities within specified distance
3. **Path Recording**: Track shortest paths between explored entities and neighbors
4. **Iteration Tracking**: Record exploration details including entities, paths, and reasoning
5. **Entity Expansion**: Add newly discovered entities to current entity set
6. **Termination Check**: Continue until max iterations or LLM decides to stop

## Decision Making Patterns

### Path Decision Criteria
1. **Query Relevance**: How likely the path leads to query-relevant information
2. **Information Density**: How much useful information the path might contain
3. **Exploration Efficiency**: Balance between breadth and depth
4. **Computational Cost**: Consider API calls and processing time
5. **Diminishing Returns**: Stop when new information becomes sparse

### LLM Decision Prompts
**Template Structure**: Provides current exploration state and asks LLM to decide next exploration steps.

**Input Context**:
- Current query being processed
- Set of currently known entities
- Current iteration number
- Exploration history and context

**Decision Criteria**:
1. **Relevance Assessment**: Which entities most likely lead to query-relevant information
2. **Exploration Strategy**: Whether to explore broadly (many entities) or deeply (fewer entities)
3. **Sufficiency Evaluation**: Whether sufficient information has been gathered
4. **Value Analysis**: Expected value of continuing exploration vs. stopping

**Output Requirements**: Decision with detailed reasoning for transparency and debugging.

## Performance Optimization

### Caching Strategies
- **Path Cache**: Store frequently used paths
- **Entity Cache**: Cache entity neighborhood information
- **Decision Cache**: Cache LLM decisions for similar contexts
- **Result Cache**: Cache final results for repeated queries

### Batch Processing
- **Entity Batch**: Process multiple entities in single graph query
- **LLM Batch**: Combine multiple LLM calls when possible
- **Path Batch**: Discover multiple paths simultaneously

### Early Termination
- **Confidence Threshold**: Stop when confidence exceeds threshold
- **Information Saturation**: Stop when new information becomes redundant
- **Time Limits**: Enforce maximum processing time
- **Cost Limits**: Limit number of LLM calls
