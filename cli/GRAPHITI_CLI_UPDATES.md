# CLI Scripts Updated for Graphiti Integration

## 📋 Summary

All CLI scripts have been updated to support Graphiti-based knowledge graph ingestion as the recommended approach. The traditional ingestion methods are still available for backward compatibility.

## 🔄 Updated Scripts

### 1. `cli/graph_extraction.py` - Core Graph Extraction Module ✅

**New Features:**
- `GraphitiExtractionService` class for Graphiti operations
- `extract_and_ingest_with_graphiti()` function for episode-based ingestion
- `search_with_graphiti()` function for knowledge graph search
- Automatic API key detection (Gemini or OpenAI)
- Built-in error handling and connection management

**Usage:**
```python
from graph_extraction import extract_and_ingest_with_graphiti, search_with_graphiti

# Ingest content
result = await extract_and_ingest_with_graphiti(
    text_content="Your content here",
    doc_id="doc_001",
    title="Document Title",
    metadata={"category": "research"}
)

# Search knowledge graph
results = await search_with_graphiti("artificial intelligence", limit=10)
```

### 2. `cli/test-document.py` - Document Processing Script ✅

**New Features:**
- `--graphiti` flag for Graphiti-based ingestion (recommended)
- `--traditional-extraction` flag for comparison with traditional methods
- `test_document_with_graphiti()` function for complete document processing
- Updated help text with Graphiti examples

**Usage:**
```bash
# Recommended: Use Graphiti for knowledge graph ingestion
python test-document.py my-document.pdf --graphiti
python test-document.py research.docx --graphiti --metadata '{"category": "research"}'
python test-document.py paper.pdf --graphiti --traditional-extraction

# Traditional ingestion (still available)
python test-document.py my-document.pdf --ingest --qdrant --neo4j
```

### 3. `cli/test-graphiti.py` - New Dedicated Graphiti CLI ✅

**Purpose:** Direct access to Graphiti functionality

**Commands:**
- `ingest <file>` - Ingest documents into knowledge graph
- `search <query>` - Search the knowledge graph
- `status` - Check system status

**Usage:**
```bash
# Ingest documents
python test-graphiti.py ingest my-document.pdf
python test-graphiti.py ingest research.docx --metadata '{"category": "research"}'

# Search knowledge graph
python test-graphiti.py search "artificial intelligence"
python test-graphiti.py search "machine learning" --limit 5

# Check status
python test-graphiti.py status
```

### 4. `cli/test-ingestion-coordinator.py` - Ingestion Testing ✅

**New Features:**
- `test_graphiti_ingestion()` function for Graphiti testing
- Dual testing approach (Graphiti + traditional)
- Comprehensive test summary and recommendations

**Usage:**
```bash
python test-ingestion-coordinator.py
# Tests both Graphiti and traditional ingestion methods
```

### 5. `packages/morag-graph/run_graphiti_ingestion.py` - New Production Script ✅

**Purpose:** Production-ready Graphiti ingestion script

**Commands:**
- `ingest <file>` - Ingest files with full configuration
- `search <query>` - Advanced search capabilities
- `status` - Comprehensive system status check

**Usage:**
```bash
# Production ingestion
python run_graphiti_ingestion.py ingest document.txt --title "Research Paper"
python run_graphiti_ingestion.py ingest data.txt --metadata '{"category": "research"}'

# Search
python run_graphiti_ingestion.py search "artificial intelligence" --limit 5

# Status check
python run_graphiti_ingestion.py status
```

## 🚀 Key Improvements

### 1. Unified API Key Management
- Automatic detection of `GEMINI_API_KEY` or `OPENAI_API_KEY`
- Fallback mechanism for maximum compatibility
- Clear error messages for missing configuration

### 2. Episode-Based Knowledge Representation
- Documents are converted to Graphiti episodes
- Automatic entity extraction and relationship building
- Built-in deduplication and temporal queries
- Hybrid search capabilities

### 3. Backward Compatibility
- All traditional ingestion methods still available
- Gradual migration path for existing workflows
- Side-by-side comparison capabilities

### 4. Enhanced Error Handling
- Graceful handling of API key issues
- Clear error messages and troubleshooting guidance
- Comprehensive status checking

## 📊 Migration Guide

### For New Projects (Recommended)
Use Graphiti-based ingestion for all new implementations:

```bash
# Document processing
python test-document.py document.pdf --graphiti

# Direct file ingestion
python test-graphiti.py ingest document.txt

# Production ingestion
python run_graphiti_ingestion.py ingest document.txt --title "My Document"
```

### For Existing Projects
Gradual migration approach:

1. **Test Graphiti alongside existing methods:**
   ```bash
   python test-document.py document.pdf --graphiti --traditional-extraction
   ```

2. **Compare results and performance:**
   ```bash
   python test-ingestion-coordinator.py
   ```

3. **Switch to Graphiti when ready:**
   ```bash
   python test-document.py document.pdf --graphiti
   ```

## 🔧 Configuration

### Environment Variables
```bash
# API Keys (use either)
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"

# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
export NEO4J_DATABASE="neo4j"

# Graphiti Models (optional)
export GRAPHITI_MODEL="gpt-4"
export GRAPHITI_EMBEDDING_MODEL="text-embedding-3-small"
```

### Prerequisites
```bash
# Install required packages
pip install -e packages/morag-core
pip install -e packages/morag-document
pip install -e packages/morag-graph
pip install -e packages/morag-services

# Start Neo4j (Docker)
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

## 🎯 Benefits of Graphiti Integration

### 1. **Automatic Entity Extraction**
- No need for manual entity/relation extraction
- Built-in NLP processing with LLM integration
- Consistent and accurate knowledge representation

### 2. **Temporal Knowledge Graphs**
- Episodes represent knowledge with temporal context
- Query knowledge based on time periods
- Track knowledge evolution over time

### 3. **Built-in Deduplication**
- Automatic detection and merging of similar content
- Prevents knowledge graph bloat
- Maintains data quality and consistency

### 4. **Hybrid Search**
- Combines semantic and keyword search
- Better search relevance and coverage
- Advanced query capabilities

### 5. **Simplified Workflow**
- Single command for complete ingestion
- No need for separate extraction and storage steps
- Reduced complexity and maintenance

## 🔍 Testing and Validation

### Quick Test
```bash
# Test basic functionality
python test-graphiti.py status

# Test document ingestion
python test-graphiti.py ingest test-document.txt

# Test search
python test-graphiti.py search "test content"
```

### Comprehensive Test
```bash
# Run all ingestion tests
python test-ingestion-coordinator.py

# Test document processing with comparison
python test-document.py document.pdf --graphiti --traditional-extraction
```

## 📈 Next Steps

1. **Test the updated CLI scripts** with your documents
2. **Compare Graphiti vs traditional results** using the comparison features
3. **Migrate gradually** from traditional to Graphiti-based workflows
4. **Monitor performance** and adjust configuration as needed
5. **Provide feedback** on the new functionality

## 🎉 Conclusion

The CLI scripts now provide a comprehensive, modern approach to knowledge graph ingestion using Graphiti while maintaining full backward compatibility. The new episode-based architecture offers significant advantages in terms of automation, accuracy, and search capabilities.

**Recommended workflow:** Start with `--graphiti` flag for new documents and gradually migrate existing workflows as you validate the results.
