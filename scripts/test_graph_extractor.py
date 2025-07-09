#!/usr/bin/env python3
"""
Graph Extractor Test Script

This script takes a markdown file as input and uses the graph extractor to write a JSON file
containing the graph hierarchy for the document (entities and relations between entities).
This script is used to verify the system prompt(s) used by the extractor and fine-tune it.

Usage:
    python scripts/test_graph_extractor.py input.md [output.json] [--language en] [--verbose]

Example:
    python scripts/test_graph_extractor.py test_document.md graph_output.json --language en --verbose
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the packages to the Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-graph" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))

try:
    from morag.graph_extractor_wrapper import GraphExtractor
    from morag_graph.extraction.entity_extractor import EntityExtractor
    from morag_graph.extraction.relation_extractor import RelationExtractor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the required packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-graph")
    print("  pip install -e packages/morag")
    sys.exit(1)


class GraphExtractorTester:
    """Test class for graph extraction with detailed output and system prompt inspection."""
    
    def __init__(self, language: Optional[str] = None, verbose: bool = False, dry_run: bool = False):
        """Initialize the tester.

        Args:
            language: Language code for processing (e.g., 'en', 'de', 'fr')
            verbose: Whether to print verbose output
            dry_run: Whether to run in dry-run mode (no API calls)
        """
        self.language = language
        self.verbose = verbose
        self.dry_run = dry_run
        self.graph_extractor = GraphExtractor() if not dry_run else None
        
    async def test_extraction(self, markdown_file: Path, output_file: Path) -> Dict[str, Any]:
        """Test graph extraction on a markdown file.
        
        Args:
            markdown_file: Path to input markdown file
            output_file: Path to output JSON file
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        if not markdown_file.exists():
            raise FileNotFoundError(f"Input file not found: {markdown_file}")
        
        # Read the markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if self.verbose:
            print(f"[FILE] Reading file: {markdown_file}")
            print(f"[INFO] Content length: {len(content)} characters")
            print(f"[LANG] Language: {self.language or 'auto-detect'}")
            print("=" * 60)
        
        # Handle dry-run mode
        if self.dry_run:
            if self.verbose:
                print("[DRY-RUN] Running in dry-run mode (no API calls)")

            # Create mock system prompts
            entity_prompt = "Mock entity extraction system prompt for testing purposes..."
            relation_prompt = "Mock relation extraction system prompt for testing purposes..."

            # Create mock extraction results
            extraction_result = self._create_mock_extraction_result(content, str(markdown_file))
        else:
            # Initialize the graph extractor
            await self.graph_extractor.initialize()

            # Get system prompts for inspection
            entity_prompt = self.graph_extractor.entity_extractor.get_system_prompt()
            relation_prompt = self.graph_extractor.relation_extractor.get_system_prompt()
        
        if self.verbose:
            print("[PROMPT] Entity Extraction System Prompt:")
            print("-" * 40)
            print(entity_prompt[:500] + "..." if len(entity_prompt) > 500 else entity_prompt)
            print("\n[PROMPT] Relation Extraction System Prompt:")
            print("-" * 40)
            print(relation_prompt[:500] + "..." if len(relation_prompt) > 500 else relation_prompt)
            print("=" * 60)
        
        # Extract entities and relations (if not in dry-run mode)
        if not self.dry_run:
            if self.verbose:
                print("[EXTRACT] Starting extraction...")

            extraction_result = await self.graph_extractor.extract_entities_and_relations(
                content=content,
                source_path=str(markdown_file),
                language=self.language
            )
        
        # Prepare detailed output
        detailed_result = {
            "input_file": str(markdown_file),
            "content_length": len(content),
            "language": self.language,
            "extraction_timestamp": asyncio.get_event_loop().time(),
            "system_prompts": {
                "entity_prompt": entity_prompt,
                "relation_prompt": relation_prompt
            },
            "extraction_results": extraction_result,
            "analysis": {
                "entity_count": len(extraction_result.get("entities", [])),
                "relation_count": len(extraction_result.get("relations", [])),
                "entity_types": self._analyze_entity_types(extraction_result.get("entities", [])),
                "relation_types": self._analyze_relation_types(extraction_result.get("relations", [])),
                "confidence_stats": self._analyze_confidence_stats(extraction_result)
            }
        }
        
        # Write results to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_result, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"[OK] Extraction completed!")
            print(f"[RESULTS] Summary:")
            print(f"   - Entities: {detailed_result['analysis']['entity_count']}")
            print(f"   - Relations: {detailed_result['analysis']['relation_count']}")
            print(f"   - Entity types: {list(detailed_result['analysis']['entity_types'].keys())}")
            print(f"   - Relation types: {list(detailed_result['analysis']['relation_types'].keys())}")
            print(f"[OUTPUT] Output saved to: {output_file}")
        
        return detailed_result
    
    def _analyze_entity_types(self, entities: list) -> Dict[str, int]:
        """Analyze entity types distribution."""
        type_counts = {}
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def _analyze_relation_types(self, relations: list) -> Dict[str, int]:
        """Analyze relation types distribution."""
        type_counts = {}
        for relation in relations:
            relation_type = relation.get("relation_type", "UNKNOWN")
            type_counts[relation_type] = type_counts.get(relation_type, 0) + 1
        return type_counts
    
    def _analyze_confidence_stats(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence statistics."""
        entities = extraction_result.get("entities", [])
        relations = extraction_result.get("relations", [])
        
        entity_confidences = [e.get("confidence", 0.0) for e in entities]
        relation_confidences = [r.get("confidence", 0.0) for r in relations]
        
        stats = {}
        
        if entity_confidences:
            stats["entity_confidence"] = {
                "min": min(entity_confidences),
                "max": max(entity_confidences),
                "avg": sum(entity_confidences) / len(entity_confidences)
            }
        
        if relation_confidences:
            stats["relation_confidence"] = {
                "min": min(relation_confidences),
                "max": max(relation_confidences),
                "avg": sum(relation_confidences) / len(relation_confidences)
            }
        
        return stats

    def _create_mock_extraction_result(self, content: str, source_path: str) -> Dict[str, Any]:
        """Create mock extraction results for dry-run mode."""
        return {
            "entities": [
                {
                    "id": "ent_mock_001",
                    "name": "Mock Entity 1",
                    "type": "PERSON",
                    "description": "A mock person entity for testing",
                    "confidence": 0.85,
                    "attributes": {"mock": True},
                    "source_doc_id": source_path
                },
                {
                    "id": "ent_mock_002",
                    "name": "Mock Organization",
                    "type": "ORGANIZATION",
                    "description": "A mock organization entity for testing",
                    "confidence": 0.90,
                    "attributes": {"mock": True},
                    "source_doc_id": source_path
                }
            ],
            "relations": [
                {
                    "id": "rel_mock_001",
                    "source_entity_id": "ent_mock_001",
                    "target_entity_id": "ent_mock_002",
                    "relation_type": "WORKS_AT",
                    "description": "Mock employment relationship",
                    "confidence": 0.80,
                    "attributes": {"mock": True},
                    "source_doc_id": source_path
                }
            ],
            "metadata": {
                "entity_count": 2,
                "relation_count": 1,
                "source_path": source_path,
                "content_length": len(content)
            }
        }


async def main():
    """Main function to run the graph extractor test."""
    parser = argparse.ArgumentParser(
        description="Test graph extractor on markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_graph_extractor.py test.md
  python scripts/test_graph_extractor.py test.md output.json --language en
  python scripts/test_graph_extractor.py test.md output.json --language de --verbose
        """
    )
    
    parser.add_argument("input_file", help="Input markdown file")
    parser.add_argument("output_file", nargs="?", help="Output JSON file (default: input_file.graph.json)")
    parser.add_argument("--language", "-l", help="Language code (e.g., en, de, fr)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no API calls, mock data)")
    
    args = parser.parse_args()
    
    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)
    
    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.with_suffix('.graph.json')
    
    # Check for required environment variables (unless in dry-run mode)
    if not args.dry_run and not os.getenv('GEMINI_API_KEY'):
        print("[ERROR] GEMINI_API_KEY environment variable is required")
        print("[INFO] Set it with: export GEMINI_API_KEY='your-api-key'")
        print("[INFO] For testing without API, use --dry-run flag")
        sys.exit(1)

    try:
        # Create tester and run extraction
        tester = GraphExtractorTester(language=args.language, verbose=args.verbose, dry_run=args.dry_run)
        result = await tester.test_extraction(input_file, output_file)
        
        if not args.verbose:
            print(f"[OK] Graph extraction completed")
            print(f"[STATS] Entities: {result['analysis']['entity_count']}, Relations: {result['analysis']['relation_count']}")
            print(f"[OUTPUT] Results saved to: {output_file}")

    except Exception as e:
        print(f"[ERROR] Error during extraction: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
