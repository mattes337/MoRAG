# Recursive Fact Retrieval Usage Guide

This guide shows how to use the recursive fact retrieval system that was implemented based on `RECURSIVE_FACT_RETRIEVAL.md`.

## Quick Start

### 1. Environment Configuration

The system automatically loads configuration from the `.env` file in the project root. Key variables:

```bash
# Required: Gemini API Key
GEMINI_API_KEY=your-gemini-api-key

# LLM Models
MORAG_GEMINI_MODEL=gemini-2.5-flash              # For GraphTraversalAgent & FactCriticAgent
MORAG_STRONGER_GEMINI_MODEL=gemini-1.5-pro      # For final answer synthesis

# Neo4j Configuration
NEO4J_URI=bolt+ssc://your-neo4j-server:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# Qdrant Configuration
QDRANT_URL=https://your-qdrant-server
QDRANT_API_KEY=your-qdrant-api-key
MORAG_QDRANT_COLLECTION=morag_documents
```

### 2. Check Configuration

View your current environment configuration:

```bash
python cli/test-recursive-fact-retrieval.py --show-config
```

### 3. Test the System

#### Basic Usage
```bash
python cli/test-recursive-fact-retrieval.py "What are the main symptoms of ADHD?"
```

#### With Verbose Output
```bash
python cli/test-recursive-fact-retrieval.py "What are the main symptoms of ADHD?" --verbose
```

#### Health Check
```bash
python cli/test-recursive-fact-retrieval.py "test" --check-health
```

#### Custom Parameters
```bash
python cli/test-recursive-fact-retrieval.py "What are the main symptoms of ADHD?" \
  --max-depth 5 \
  --decay-rate 0.15 \
  --max-facts-per-node 10 \
  --min-fact-score 0.2 \
  --max-total-facts 100 \
  --verbose
```

#### Save Results to JSON
```bash
python cli/test-recursive-fact-retrieval.py "What are the main symptoms of ADHD?" \
  --output-json results.json \
  --verbose
```

## REST API Usage

### Start the Server
```bash
python -m morag.server
```

### Basic Request
```bash
curl -X POST "http://localhost:8000/api/v2/recursive-fact-retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "What are the main symptoms of ADHD?",
    "max_depth": 3,
    "decay_rate": 0.2,
    "max_facts_per_node": 5,
    "min_fact_score": 0.1,
    "max_total_facts": 50
  }'
```

### API Information
```bash
curl "http://localhost:8000/api/v2/recursive-fact-retrieval/info"
```

### Health Check
```bash
curl "http://localhost:8000/api/v2/recursive-fact-retrieval/health"
```

## Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_depth` | 3 | 1-10 | Maximum graph traversal depth |
| `decay_rate` | 0.2 | 0.0-1.0 | Score decay rate per depth level |
| `max_facts_per_node` | 5 | 1-20 | Maximum facts extracted per node |
| `min_fact_score` | 0.1 | 0.0-1.0 | Minimum score threshold for facts |
| `max_total_facts` | 50 | 1-200 | Maximum total facts to collect |

## Understanding the Output

### CLI Output
```
✅ Test successful!
   Query ID: 45775a40-2018-4f3d-bd58-b1a150f6a787
   Query: What are the main symptoms of ADHD?
   Processing time: 2257.55ms
   Initial entities: ADHD, symptoms, attention deficit
   Nodes explored: 12
   Max depth reached: 3
   Raw facts extracted: 45
   Scored facts: 38
   Final facts: 25
   LLM calls - GTA: 12, FCA: 45, Final: 1
   Confidence score: 0.85

📋 Final Facts:
   1. ADHD is characterized by persistent patterns of inattention and hyperactivity
      Score: 0.920 (original: 0.950)
      Depth: 0
      Source: From medical diagnostic criteria document

💡 Final Answer:
   ADHD (Attention Deficit Hyperactivity Disorder) is characterized by...
```

### Key Metrics
- **Processing time**: Total time for the complete process
- **Initial entities**: Entities identified from the user query
- **Nodes explored**: Number of graph nodes visited
- **Max depth reached**: Deepest level of graph traversal
- **LLM calls**: Number of AI model calls (GTA + FCA + Final)
- **Confidence score**: Overall confidence in the final answer (0.0-1.0)

## Algorithm Flow

1. **Entity Extraction**: Identify key entities from user query
2. **Node Mapping**: Map entities to graph nodes in Neo4j
3. **Graph Traversal**: Breadth-first exploration with LLM decisions
4. **Fact Extraction**: Extract facts from each explored node
5. **Relevance Scoring**: Evaluate and score fact relevance
6. **Decay Application**: Apply depth-based score decay
7. **Filtering**: Remove low-scoring facts and apply limits
8. **Final Synthesis**: Generate comprehensive answer using stronger LLM

## Troubleshooting

### Common Issues

1. **"Could not identify key entities"**
   - Try more specific queries with clear entities
   - Check if your knowledge graph contains relevant data

2. **"No relevant starting nodes found"**
   - Verify your Neo4j database contains the expected entities
   - Check entity naming conventions in your graph

3. **"Model overloaded" errors**
   - The system includes automatic retry with exponential backoff
   - Consider using a different Gemini model if issues persist

4. **Database connection errors**
   - Verify your Neo4j and Qdrant credentials in `.env`
   - Check network connectivity to your database servers
   - Use `--check-health` to diagnose connection issues

### Debug Mode
Use `--verbose` flag to see detailed information including:
- Environment configuration
- Traversal steps with LLM decisions
- Individual fact scores and sources
- Detailed error messages

## Performance Tips

1. **Optimize Parameters**:
   - Lower `max_depth` for faster responses
   - Adjust `min_fact_score` to filter low-quality facts
   - Use `max_total_facts` to limit processing time

2. **Model Selection**:
   - Use faster models (e.g., `gemini-1.5-flash`) for GTA/FCA
   - Reserve stronger models (e.g., `gemini-1.5-pro`) for final synthesis

3. **Graph Structure**:
   - Ensure your Neo4j graph has good connectivity
   - Use meaningful entity and relationship names
   - Consider graph size vs. query specificity
