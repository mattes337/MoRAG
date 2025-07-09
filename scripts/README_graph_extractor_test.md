# Graph Extractor Test Script

This script (`test_graph_extractor.py`) is designed to test and validate the graph extraction functionality of MoRAG. It takes a markdown file as input and produces a detailed JSON output containing extracted entities, relations, and analysis metadata.

## Purpose

The script is primarily used for:
1. **System Prompt Verification**: Inspect the actual system prompts used by entity and relation extractors
2. **Fine-tuning**: Analyze extraction results to improve prompt engineering
3. **Quality Assurance**: Validate extraction accuracy and consistency
4. **Performance Analysis**: Monitor confidence scores and extraction statistics

## Usage

### Basic Usage
```bash
python scripts/test_graph_extractor.py input.md
```

### Advanced Usage
```bash
python scripts/test_graph_extractor.py input.md output.json --language en --verbose
```

### Dry-Run Mode (No API Calls)
```bash
python scripts/test_graph_extractor.py input.md --dry-run --verbose
```

### Parameters

- `input_file`: Path to the input markdown file (required)
- `output_file`: Path to the output JSON file (optional, defaults to `input_file.graph.json`)
- `--language, -l`: Language code for processing (e.g., 'en', 'de', 'fr')
- `--verbose, -v`: Enable verbose output with detailed logging
- `--dry-run`: Run in dry-run mode with mock data (no API calls required)

## Prerequisites

### Environment Variables
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### Optional Configuration
```bash
export MORAG_GEMINI_MODEL="gemini-1.5-flash"  # Default model
export MORAG_GRAPH_MAX_RETRIES="5"            # Retry attempts
export MORAG_GRAPH_BASE_DELAY="1.0"           # Base delay for retries
```

### Package Installation
```bash
pip install -e packages/morag-core
pip install -e packages/morag-graph
pip install -e packages/morag
```

## Output Format

The script generates a comprehensive JSON file with the following structure:

```json
{
  "input_file": "path/to/input.md",
  "content_length": 1234,
  "language": "en",
  "extraction_timestamp": 1234567890.123,
  "system_prompts": {
    "entity_prompt": "Full entity extraction system prompt...",
    "relation_prompt": "Full relation extraction system prompt..."
  },
  "extraction_results": {
    "entities": [
      {
        "id": "ent_12345",
        "name": "Dr. Sarah Johnson",
        "type": "PERSON",
        "description": "renowned cardiologist",
        "confidence": 0.95,
        "attributes": {...},
        "source_doc_id": "path/to/input.md"
      }
    ],
    "relations": [
      {
        "id": "rel_67890",
        "source_entity_id": "ent_12345",
        "target_entity_id": "ent_54321",
        "relation_type": "WORKS_AT",
        "description": "employment relationship",
        "confidence": 0.88,
        "attributes": {...},
        "source_doc_id": "path/to/input.md"
      }
    ],
    "metadata": {
      "entity_count": 15,
      "relation_count": 12,
      "source_path": "path/to/input.md",
      "content_length": 1234
    }
  },
  "analysis": {
    "entity_count": 15,
    "relation_count": 12,
    "entity_types": {
      "PERSON": 5,
      "ORGANIZATION": 4,
      "LOCATION": 3,
      "CONCEPT": 3
    },
    "relation_types": {
      "WORKS_AT": 3,
      "LOCATED_IN": 2,
      "TREATS": 2,
      "DEVELOPS": 1
    },
    "confidence_stats": {
      "entity_confidence": {
        "min": 0.65,
        "max": 0.98,
        "avg": 0.82
      },
      "relation_confidence": {
        "min": 0.70,
        "max": 0.95,
        "avg": 0.84
      }
    }
  }
}
```

## Examples

### Test with Sample Document
```bash
# Use the provided test document
python scripts/test_graph_extractor.py test_document.md

# With verbose output
python scripts/test_graph_extractor.py test_document.md --verbose

# Specify language and output file
python scripts/test_graph_extractor.py test_document.md medical_graph.json --language en
```

### Test with Custom Document
```bash
# Create your own markdown file
echo "# My Document\nJohn works at Microsoft in Seattle." > my_test.md

# Run extraction
python scripts/test_graph_extractor.py my_test.md --verbose
```

### Dry-Run Testing
```bash
# Test script structure without API calls
python scripts/test_graph_extractor.py test_document.md --dry-run

# Inspect output format and structure
python scripts/test_graph_extractor.py test_document.md --dry-run --verbose

# Test with different languages (mock mode)
python scripts/test_graph_extractor.py test_document.md --dry-run --language de
```

## Analyzing Results

### System Prompt Inspection
The output includes the complete system prompts used by both entity and relation extractors. Use this to:
- Verify prompt content matches expectations
- Identify areas for improvement
- Compare different prompt versions

### Entity Analysis
Review the `entity_types` distribution to ensure:
- Appropriate entity types are being extracted
- No over-generalization or under-specification
- Consistent type naming conventions

### Relation Analysis
Examine the `relation_types` to validate:
- Meaningful relationships are captured
- Relation types follow naming conventions
- No duplicate or redundant relations

### Confidence Analysis
Monitor confidence statistics to:
- Identify low-confidence extractions that need review
- Adjust confidence thresholds if needed
- Track extraction quality over time

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed correctly
2. **API Key Missing**: Set the `GEMINI_API_KEY` environment variable
3. **File Not Found**: Check input file path and permissions
4. **Low Extraction Quality**: Review and adjust system prompts

### Debug Mode
Use `--verbose` flag to get detailed output including:
- System prompts (truncated)
- Extraction progress
- Detailed statistics
- Error traces

## Integration with Development Workflow

### Prompt Engineering Cycle
1. Modify system prompts in the extractor classes
2. Run test script on representative documents
3. Analyze results and confidence scores
4. Iterate on prompt improvements
5. Validate with diverse test cases

### Quality Assurance
- Run tests on various document types
- Compare results across different languages
- Monitor extraction consistency
- Track performance metrics over time

This script is an essential tool for maintaining and improving the quality of MoRAG's graph extraction capabilities.
