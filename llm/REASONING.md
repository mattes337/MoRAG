# Multi-Hop Reasoning Strategies

## Core Reasoning Approaches

### 1. Forward Chaining
**Strategy**: Start from query entities and explore forward through relationships
**Use Case**: When you know starting entities and want to discover what they lead to
**Max Depth**: 4 hops
**Bidirectional**: False

```python
# Configuration
strategy = ReasoningStrategy(
    name="forward_chaining",
    description="Start from query entities and explore forward",
    max_depth=4,
    bidirectional=False,
    use_weights=True
)

# Example Query: "What products does Apple make?"
# Starting entities: ["Apple Inc."]
# Forward exploration: Apple -> MANUFACTURES -> iPhone, iPad, MacBook
```

### 2. Backward Chaining
**Strategy**: Start from potential answers and work backward to query entities
**Use Case**: When you know target entities and want to find connections
**Max Depth**: 3 hops
**Bidirectional**: False

```python
# Configuration
strategy = ReasoningStrategy(
    name="backward_chaining",
    description="Start from potential answers and work backward",
    max_depth=3,
    bidirectional=False,
    use_weights=True
)

# Example Query: "Who invented the iPhone?"
# Target entities: ["iPhone"]
# Backward exploration: iPhone <- INVENTED_BY <- Steve Jobs <- WORKED_AT <- Apple
```

### 3. Bidirectional Search
**Strategy**: Search from both query entities and target entities, meet in middle
**Use Case**: Complex multi-hop queries with known start and end points
**Max Depth**: 5 hops
**Bidirectional**: True

```python
# Configuration
strategy = ReasoningStrategy(
    name="bidirectional",
    description="Search from both ends and meet in the middle",
    max_depth=5,
    bidirectional=True,
    use_weights=True
)

# Example Query: "How is Apple's AI research connected to Stanford University?"
# Start: ["Apple Inc.", "AI research"]
# Target: ["Stanford University"]
# Meet in middle: Apple -> PARTNERS_WITH -> Stanford
```

## Path Selection Algorithm

### LLM-Based Path Ranking
```python
async def select_paths(query, available_paths, strategy):
    # 1. Create path selection prompt
    prompt = create_path_selection_prompt(query, available_paths)
    
    # 2. Get LLM analysis (temperature=0.1 for consistency)
    response = await llm_client.generate(
        prompt=prompt,
        max_tokens=1000,
        temperature=0.1
    )
    
    # 3. Parse LLM response to extract selected paths
    selected_paths = parse_path_selection(response, available_paths)
    
    # 4. Apply additional scoring based on strategy
    scored_paths = await score_paths(query, selected_paths, strategy)
    
    # 5. Sort by relevance and return top paths
    return sorted(scored_paths, key=lambda x: x.relevance_score, reverse=True)
```

### Path Scoring Criteria
1. **Semantic Relevance**: How well path entities relate to query
2. **Path Coherence**: Logical flow of relationships
3. **Entity Importance**: Centrality and confidence of entities
4. **Relationship Strength**: Confidence scores of connections
5. **Path Length**: Shorter paths often more reliable

## Iterative Context Refinement

### Refinement Process
```python
async def refine_context(query, initial_context):
    iteration_count = 0
    current_context = initial_context
    
    while iteration_count < max_iterations:
        # 1. Analyze current context sufficiency
        analysis = await analyze_context(query, current_context)
        
        # 2. Check if context is sufficient
        if analysis.is_sufficient and analysis.confidence >= threshold:
            break
        
        # 3. Retrieve additional information based on gaps
        additional_context = await retrieve_additional(
            query, analysis.gaps, current_context
        )
        
        # 4. Merge new context with existing
        current_context = merge_contexts(current_context, additional_context)
        
        iteration_count += 1
    
    return current_context
```

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

```cypher
MATCH (start:Entity {id: $entity_id})
MATCH path = (start)-[r*1..3]-(neighbor:Entity)
WHERE start <> neighbor
RETURN DISTINCT neighbor, length(path) as distance
ORDER BY distance, neighbor.name
```

### Depth-First Search (DFS)
**Use Case**: Following specific relationship chains
**Pattern**: Follow one path deeply before exploring alternatives

```cypher
MATCH path = (start:Entity {id: $entity_id})-[r*1..4]-(end:Entity)
WHERE start <> end
RETURN path, nodes(path) as entities, relationships(path) as relations
ORDER BY length(path)
```

### Shortest Path Finding
**Use Case**: Finding most direct connections between entities
**Pattern**: Use Neo4j's shortestPath algorithm

```cypher
MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
MATCH path = shortestPath((source)-[r*1..5]-(target))
RETURN path,
       length(path) as path_length,
       [node in nodes(path) | {id: node.id, name: node.name, type: node.type}] as entities,
       [rel in relationships(path) | {type: rel.type, confidence: rel.confidence}] as relationships
ORDER BY path_length
```

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
```python
async def identify_entities(query):
    # Extract entities using NER and entity linking
    entities = await extract_entities(query)
    
    # Link to knowledge graph entities
    linked_entities = []
    for entity in entities:
        graph_entity = await find_graph_entity(entity.name)
        if graph_entity:
            linked_entities.append(graph_entity)
    
    return linked_entities
```

### Recursive Path Following
```python
async def follow_paths_recursively(query, initial_entities, max_iterations):
    iterations = []
    current_entities = set(initial_entities)
    
    for i in range(max_iterations):
        # Make LLM decision about which paths to follow
        path_decision = await make_path_decision(query, current_entities, i)
        
        # Follow selected paths
        new_entities = set()
        paths_followed = []
        
        for entity in path_decision.entities_to_explore:
            neighbors = await find_neighbors(entity, max_distance=1)
            new_entities.update([n.id for n in neighbors])
            
            # Record path information
            for neighbor in neighbors:
                path = await find_shortest_path(entity, neighbor.id)
                if path:
                    paths_followed.append(path)
        
        # Record iteration
        iteration = RetrievalIteration(
            iteration_number=i,
            entities_explored=list(path_decision.entities_to_explore),
            new_entities_found=list(new_entities - current_entities),
            paths_followed=paths_followed,
            reasoning=path_decision.reasoning
        )
        iterations.append(iteration)
        
        # Update current entities
        current_entities.update(new_entities)
        
        # Check if we should continue
        if not path_decision.should_continue:
            break
    
    return iterations
```

## Decision Making Patterns

### Path Decision Criteria
1. **Query Relevance**: How likely the path leads to query-relevant information
2. **Information Density**: How much useful information the path might contain
3. **Exploration Efficiency**: Balance between breadth and depth
4. **Computational Cost**: Consider API calls and processing time
5. **Diminishing Returns**: Stop when new information becomes sparse

### LLM Decision Prompts
```
Given the current exploration state, decide which entities to explore next.

Query: {query}
Current Entities: {entities}
Iteration: {iteration_number}

Consider:
1. Which entities are most likely to lead to query-relevant information?
2. Should we explore broadly (many entities) or deeply (fewer entities)?
3. Have we gathered sufficient information to answer the query?
4. What is the expected value of continuing exploration?

Return decision with reasoning.
```

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
