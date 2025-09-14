# Multi-Hop Reasoning CLI Testing Scripts

This directory contains CLI scripts to test the multi-hop reasoning capabilities of MoRAG. These scripts provide different levels of testing from simple component validation to full API integration testing.

## 📋 Available Scripts

### 1. `test_reasoning_simple.py` - Component Testing
**Purpose**: Quick validation of individual reasoning components without requiring external services.

**Features**:
- Tests path selection agent functionality
- Tests iterative context refinement
- Tests component integration
- Works with or without API keys (uses mocks when needed)
- Lightweight and fast execution

**Usage**:
```bash
# Run all component tests
python test_reasoning_simple.py

# Run with detailed output
python test_reasoning_simple.py --verbose

# Test specific component only
python test_reasoning_simple.py --component path_selection
python test_reasoning_simple.py --component iterative_retrieval
python test_reasoning_simple.py --component integration
```

### 2. `test_multi_hop_cli.py` - Standalone Reasoning Test
**Purpose**: Test complete multi-hop reasoning pipeline with mock graph data.

**Features**:
- Full multi-hop reasoning workflow
- LLM-guided path selection
- Iterative context refinement
- Mock graph engine for testing
- Configurable reasoning strategies
- Performance measurement
- JSON output support

**Usage**:
```bash
# Basic multi-hop reasoning test
python test_multi_hop_cli.py "How are Apple's AI research efforts related to their partnership with universities?"

# With specific strategy
python test_multi_hop_cli.py "What connects Steve Jobs to iPhone development?" --strategy bidirectional

# With detailed output and result saving
python test_multi_hop_cli.py "How does AI research influence product development?" --verbose --output results.json

# Custom parameters
python test_multi_hop_cli.py "query" --strategy backward_chaining --max-paths 20 --max-iterations 3 --model gemini-1.5-pro
```

**Options**:
- `--strategy`: Reasoning strategy (forward_chaining, backward_chaining, bidirectional)
- `--max-paths`: Maximum number of paths to discover (default: 50)
- `--max-iterations`: Maximum context refinement iterations (default: 5)
- `--api-key`: Gemini API key (or set GEMINI_API_KEY environment variable)
- `--model`: LLM model to use (default: gemini-1.5-flash)
- `--verbose`: Show detailed reasoning output
- `--output`: Save results to JSON file

### 3. `test_reasoning_api.py` - API Integration Testing
**Purpose**: Test multi-hop reasoning through the MoRAG API endpoints.

**Features**:
- Tests actual API integration
- Validates reasoning endpoints
- Tests both component and unified endpoints
- Real-world performance testing
- API health and status checking

**Usage**:
```bash
# Test via API (requires running MoRAG server)
python test_reasoning_api.py "How are Apple's AI research efforts related to their partnership with universities?"

# Test with custom API URL
python test_reasoning_api.py "query" --api-url http://localhost:8000

# Test unified reasoning endpoint
python test_reasoning_api.py "query" --test-unified --verbose

# With custom parameters
python test_reasoning_api.py "query" --strategy bidirectional --max-paths 20 --verbose --output api_results.json
```

**Options**:
- `--api-url`: MoRAG API base URL (default: http://localhost:8000)
- `--strategy`: Reasoning strategy (forward_chaining, backward_chaining, bidirectional)
- `--max-paths`: Maximum number of paths to discover (default: 50)
- `--max-iterations`: Maximum context refinement iterations (default: 5)
- `--test-unified`: Test the unified reasoning query endpoint
- `--verbose`: Show detailed reasoning output
- `--output`: Save results to JSON file

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies**:
   ```bash
   cd packages/morag-reasoning
   pip install -e .
   pip install httpx  # For API testing
   ```

2. **Set up environment** (for LLM-based tests):
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

3. **Start MoRAG server** (for API testing):
   ```bash
   # From project root
   ./scripts/debug-session.ps1
   # Or manually start the API server
   uvicorn morag.api.main:app --host 0.0.0.0 --port 8000
   ```

### Testing Workflow

1. **Start with component testing**:
   ```bash
   python test_reasoning_simple.py --verbose
   ```

2. **Test standalone reasoning**:
   ```bash
   python test_multi_hop_cli.py "How are Apple's founding and product development connected?" --verbose
   ```

3. **Test API integration** (requires running server):
   ```bash
   python test_reasoning_api.py "How are Apple's AI research efforts related to their partnership with universities?" --verbose
   ```

## 📊 Example Queries

Here are some example multi-hop reasoning queries to test:

### Technology & Innovation
- "How are Apple's AI research efforts related to their partnership with universities?"
- "What connects Steve Jobs to iPhone development through key innovations?"
- "How does artificial intelligence research influence modern product development?"

### Business & Relationships
- "What is the relationship between Apple's founding and its current product strategy?"
- "How are technology partnerships connected to innovation outcomes?"
- "What links university research to commercial product development?"

### Complex Multi-Hop
- "How do academic partnerships influence technology company product development through research collaboration?"
- "What connects early technology pioneers to modern AI development through institutional relationships?"

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure you're in the right directory
   cd packages/morag-reasoning

   # Install package in development mode
   pip install -e .
   ```

2. **API Connection Issues**:
   ```bash
   # Check if MoRAG server is running
   curl http://localhost:8000/health

   # Check reasoning status
   curl http://localhost:8000/reasoning/status
   ```

3. **Missing API Key**:
   ```bash
   # Set environment variable
   export GEMINI_API_KEY="your-api-key"

   # Or pass directly to script
   python test_multi_hop_cli.py "query" --api-key "your-api-key"
   ```

4. **JSON Parsing Errors** (Fixed in latest version):
   - **Issue**: "Error parsing LLM response: Expecting value: line 1 column 1 (char 0)"
   - **Cause**: LLM responses wrapped in markdown code blocks or containing extra text
   - **Fix**: Improved JSON parsing with automatic markdown extraction and boundary detection
   - **Status**: ✅ Resolved - The test now properly handles various LLM response formats

5. **Timeout Issues**:
   - Reduce `--max-paths` and `--max-iterations` for faster execution
   - Use simpler queries for initial testing
   - Check network connectivity for API tests

### Performance Tips

1. **For faster testing**:
   - Use `test_reasoning_simple.py` for quick validation
   - Reduce max-paths to 10-20 for initial tests
   - Use forward_chaining strategy (fastest)

2. **For comprehensive testing**:
   - Use bidirectional strategy for complex queries
   - Increase max-iterations for thorough context refinement
   - Save results to JSON for analysis

## 📈 Understanding Results

### Path Selection Results
- **Relevance Score**: 0-10 scale indicating path relevance to query
- **Confidence**: 0-10 scale indicating LLM confidence in scoring
- **Reasoning**: LLM explanation for path selection

### Context Refinement Results
- **Iterations Used**: Number of refinement cycles performed
- **Final Confidence**: Overall confidence in context sufficiency
- **Context Sufficient**: Boolean indicating if context is adequate
- **Entity/Document Counts**: Size of refined context

### Performance Metrics
- **Path Finding Time**: Time to discover and select reasoning paths
- **Refinement Time**: Time for iterative context refinement
- **Total Time**: Complete reasoning pipeline execution time

## 🎯 Success Criteria

A successful multi-hop reasoning test should show:

1. **Path Selection**: 
   - Multiple relevant paths found (>0)
   - Reasonable relevance scores (>5.0)
   - Logical reasoning explanations

2. **Context Refinement**:
   - Context improvement over iterations
   - Final confidence >0.7
   - Reasonable iteration count (<5)

3. **Performance**:
   - Path finding <5 seconds
   - Context refinement <10 seconds
   - Total time <15 seconds

4. **Integration**:
   - All components working together
   - No critical errors
   - Meaningful results for test queries

## � Recent Improvements

### JSON Parsing Fixes (2025-06-24)
- **Fixed**: "Expecting value: line 1 column 1 (char 0)" errors in LLM response parsing
- **Improved**: Robust JSON extraction from markdown-wrapped responses
- **Enhanced**: Better error handling and fallback mechanisms
- **Added**: Debug logging for troubleshooting LLM responses

### Test Enhancements
- **Improved**: Mock data relevance based on query content
- **Added**: Multi-language support (German and English examples)
- **Enhanced**: Success criteria evaluation
- **Fixed**: Document retrieval in iterative context refinement

### Example Working Queries
```bash
# German query about pineal gland (now works correctly)
python test_multi_hop_cli.py "Was beeinträchtigt die Zirbeldrüse?" --verbose

# English query about Apple and universities
python test_multi_hop_cli.py "How are Apple's AI research efforts related to their partnership with universities?" --verbose
```

## �📝 Next Steps

After successful testing:

1. **Validate with real data**: Test with actual graph database and document corpus
2. **Performance tuning**: Optimize parameters based on test results
3. **Quality assessment**: Evaluate reasoning quality with domain experts
4. **Integration testing**: Test with full MoRAG pipeline including ingestion

For more information about the multi-hop reasoning implementation, see the main [README.md](README.md) and the task documentation in `tasks/graph-extension/task-4.1-multi-hop-reasoning.md`.
