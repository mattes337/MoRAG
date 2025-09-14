# MoRAG Query Testing CLI

The `test-query.py` script is a comprehensive command-line tool for testing various query mechanisms against MoRAG's Neo4j and Qdrant databases. It supports multi-hop reasoning, graph traversal, analytics, and hybrid retrieval testing.

## Features

- **Database Testing**: Test Neo4j and/or Qdrant connections
- **Simple Queries**: Basic entity and vector search
- **Entity Queries**: Detailed entity exploration with relationships
- **Multi-hop Reasoning**: Test reasoning paths between entities
- **Graph Traversal**: Find paths between specific entities
- **Graph Analytics**: Database statistics and connectivity analysis
- **Hybrid Retrieval**: Combined vector and graph search

## Usage

### Basic Usage

```bash
# Show usage examples
python cli/test-query.py

# Show help
python cli/test-query.py --help
```

### Database Selection

```bash
# Test Neo4j only
python cli/test-query.py --neo4j "your query"

# Test Qdrant only  
python cli/test-query.py --qdrant "your query"

# Test both databases
python cli/test-query.py --all-dbs "your query"
```

### Query Types

#### Simple Query
```bash
python cli/test-query.py --all-dbs "artificial intelligence"
```

#### Entity-Focused Query
```bash
python cli/test-query.py --neo4j --entity-query "Apple Inc" "company partnerships"
```

#### Multi-hop Reasoning
```bash
python cli/test-query.py --neo4j --enable-multi-hop \
  --start-entities "Apple Inc" "AI research" \
  "How are Apple's AI efforts connected?"
```

#### Graph Traversal
```bash
python cli/test-query.py --neo4j --test-traversal \
  --start-entity "Apple Inc" --end-entity "Stanford University" \
  "connection path"
```

#### Graph Analytics
```bash
python cli/test-query.py --neo4j --test-analytics "database overview"
```

#### Hybrid Retrieval
```bash
python cli/test-query.py --all-dbs --test-hybrid "machine learning applications"
```

### Advanced Options

#### Reasoning Configuration
```bash
python cli/test-query.py --neo4j --enable-multi-hop \
  --reasoning-strategy bidirectional \
  --max-depth 4 \
  --max-paths 20 \
  --start-entities "Entity1" "Entity2" \
  "complex reasoning query"
```

#### Expansion and Fusion Strategies
```bash
python cli/test-query.py --all-dbs --test-hybrid \
  --expansion-strategy breadth_first \
  --expansion-depth 3 \
  --fusion-strategy weighted \
  "hybrid query"
```

#### Run All Tests
```bash
python cli/test-query.py --all-dbs --test-all \
  --start-entities "Apple Inc" \
  --start-entity "Apple Inc" --end-entity "AI research" \
  "comprehensive test"
```

## Command-Line Options

### Database Selection
- `--neo4j`: Test Neo4j queries
- `--qdrant`: Test Qdrant queries  
- `--all-dbs`: Test all available databases

### Query Parameters
- `--max-results N`: Maximum results (default: 10)
- `--query-type TYPE`: Query type (simple, entity_focused, relation_focused, multi_hop, analytical)

### Multi-hop Reasoning
- `--enable-multi-hop`: Enable multi-hop reasoning
- `--start-entities ENTITY [ENTITY ...]`: Starting entities for reasoning
- `--reasoning-strategy STRATEGY`: forward_chaining, backward_chaining, bidirectional
- `--max-depth N`: Maximum reasoning depth (default: 3)
- `--max-paths N`: Maximum reasoning paths (default: 10)

### Graph Traversal
- `--test-traversal`: Test graph traversal
- `--start-entity ENTITY`: Start entity for traversal
- `--end-entity ENTITY`: End entity for traversal

### Testing Options
- `--test-analytics`: Test graph analytics
- `--test-hybrid`: Test hybrid retrieval
- `--test-all`: Run all available tests

### Expansion and Fusion
- `--expansion-strategy STRATEGY`: direct_neighbors, breadth_first, shortest_path, adaptive, none
- `--expansion-depth N`: Expansion depth (default: 2)
- `--fusion-strategy STRATEGY`: weighted, reciprocal_rank_fusion, adaptive, vector_only, graph_only

### Entity Queries
- `--entity-query ENTITY`: Test entity-specific queries

### Output Options
- `--json`: Output results as JSON
- `--verbose`: Verbose output
- `--quiet`: Minimal output

## Examples

### Basic Database Testing
```bash
# Test connection and basic search
python cli/test-query.py --all-dbs "machine learning"
```

### Entity Exploration
```bash
# Find and explore a specific entity
python cli/test-query.py --neo4j --entity-query "ADHS" "medical condition"
```

### Multi-hop Reasoning Example
```bash
# Explore connections between concepts
python cli/test-query.py --neo4j --enable-multi-hop \
  --start-entities "ADHS" "Ernährung" \
  --reasoning-strategy bidirectional \
  "How does nutrition affect ADHS?"
```

### Graph Analysis
```bash
# Get database statistics
python cli/test-query.py --neo4j --test-analytics "database stats"
```

### Path Finding
```bash
# Find path between entities
python cli/test-query.py --neo4j --test-traversal \
  --start-entity "ADHS" --end-entity "Omega-3" \
  "treatment connection"
```

### Comprehensive Testing
```bash
# Run all tests with detailed configuration
python cli/test-query.py --all-dbs --test-all \
  --start-entities "ADHS" "Ernährung" \
  --start-entity "ADHS" --end-entity "Omega-3" \
  --verbose \
  "comprehensive analysis"
```

## Output

The script provides structured output including:

- **Connection Status**: Database connectivity verification
- **Search Results**: Entity and vector search results
- **Graph Statistics**: Node counts, relationship types, connectivity metrics
- **Reasoning Paths**: Multi-hop reasoning results
- **Traversal Paths**: Graph traversal results
- **Performance Metrics**: Query timing and result counts

## Requirements

- MoRAG packages installed and configured
- Environment variables set in `.env` file:
  - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
  - `QDRANT_URL` or `QDRANT_HOST`/`QDRANT_PORT`
  - `GEMINI_API_KEY` (for multi-hop reasoning)

## Troubleshooting

1. **Import Errors**: Ensure all MoRAG packages are installed
2. **Connection Errors**: Check database configuration in `.env`
3. **No Results**: Verify database contains data
4. **Timeout Errors**: Increase timeout or reduce query complexity

## Notes

- The script automatically handles database connections and cleanup
- Multi-hop reasoning requires a valid Gemini API key
- Graph analytics work best with populated Neo4j databases
- Hybrid retrieval requires both Neo4j and Qdrant connections
