#!/usr/bin/env python3
"""Extraction runner script for morag-graph package.

This script provides an easy way to run entity and relation extraction on markdown files.
It extracts entities and relations from the input file and saves the results to a JSON file.

Usage:
    python run_extraction.py input.md --api-key YOUR_GEMINI_API_KEY
    
Or set GEMINI_API_KEY environment variable and run:
    python run_extraction.py input.md
    
Options:
    --entity-only    Extract only entities
    --relation-only  Extract only relations (requires entities first)
    --model          Specify LLM model (default: gemini-1.5-flash)
    --verbose        Show detailed extraction output
    --output         Specify output file (default: input_file.json)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from morag_graph.extraction import EntityExtractor, RelationExtractor
    from morag_graph.models import Entity, Relation, Graph
    from morag_graph.storage import JsonStorage
    from morag_graph.storage.json_storage import JsonConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the morag-graph package directory")
    print("   and all dependencies are installed.")
    sys.exit(1)


def check_dependencies() -> bool:
    """Check if required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    required_packages = [
        "google-generativeai",
        "httpx",
        "pydantic",
        "python-dotenv",
        "aiofiles"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def setup_environment(api_key: Optional[str] = None) -> bool:
    """Setup environment for extraction.
    
    Args:
        api_key: Optional Gemini API key
        
    Returns:
        True if environment is properly set up, False otherwise
    """
    # Check API key
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Gemini API key is required for extraction.")
        print("   Set it via --api-key argument or GEMINI_API_KEY environment variable.")
        return False
    
    # Set environment variable
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "src" / "morag_graph").exists():
        print("❌ Please run this script from the morag-graph package root directory.")
        return False
    
    return True


async def extract_from_file(
    input_file: Path,
    api_key: str,
    model: str = "gemini-1.5-flash",
    entity_only: bool = False,
    relation_only: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """Extract entities and relations from a markdown file.
    
    Args:
        input_file: Path to input markdown file
        api_key: Gemini API key
        model: LLM model to use
        entity_only: Extract only entities
        relation_only: Extract only relations
        verbose: Show detailed output
        
    Returns:
        Dictionary containing extraction results
    """
    # Read the input file
    try:
        content = input_file.read_text(encoding='utf-8')
    except Exception as e:
        print(f"❌ Error reading file {input_file}: {e}")
        return {}
    
    if verbose:
        print(f"📄 Processing file: {input_file.name}")
        print(f"📝 Content length: {len(content)} characters")
    
    # Configure LLM
    llm_config = {
        "provider": "gemini",
        "api_key": api_key,
        "model": model,
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    results = {
        "source_file": str(input_file),
        "model": model,
        "entities": [],
        "relations": [],
        "statistics": {
            "entity_count": 0,
            "relation_count": 0,
            "entity_types": {},
            "relation_types": {}
        }
    }
    
    entities = []
    
    # Extract entities (unless relation-only)
    if not relation_only:
        if verbose:
            print("🔍 Extracting entities...")
        
        try:
            entity_extractor = EntityExtractor(llm_config)
            entities = await entity_extractor.extract(
                text=content,
                source_doc_id=str(input_file)
            )
            
            # Convert entities to serializable format
            entities_data = []
            entity_types = {}
            
            for entity in entities:
                entity_dict = {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "source_doc_id": entity.source_doc_id,
                    "attributes": entity.attributes
                }
                entities_data.append(entity_dict)
                
                # Count entity types
                entity_type = entity.type.value
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            results["entities"] = entities_data
            results["statistics"]["entity_count"] = len(entities)
            results["statistics"]["entity_types"] = entity_types
            
            if verbose:
                print(f"✅ Found {len(entities)} entities:")
                for entity in entities:
                    print(f"  • {entity.name} ({entity.type.value}) - confidence: {entity.confidence:.2f}")
        
        except Exception as e:
            print(f"❌ Error during entity extraction: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Extract relations (unless entity-only)
    if not entity_only:
        if verbose:
            print("🔗 Extracting relations...")
        
        try:
            relation_extractor = RelationExtractor(llm_config)
            relations = await relation_extractor.extract(
                text=content,
                entities=entities,
                doc_id=str(input_file)
            )
            
            # Convert relations to serializable format
            relations_data = []
            relation_types = {}
            
            for relation in relations:
                relation_dict = {
                    "id": relation.id,
                    "type": relation.type.value,
                    "source_entity_id": relation.source_entity_id,
                    "target_entity_id": relation.target_entity_id,
                    "confidence": relation.confidence,
                    "context": relation.context,
                    "source_doc_id": relation.source_doc_id,
                    "attributes": relation.attributes
                }
                relations_data.append(relation_dict)
                
                # Count relation types
                relation_type = relation.type.value
                relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
            
            results["relations"] = relations_data
            results["statistics"]["relation_count"] = len(relations)
            results["statistics"]["relation_types"] = relation_types
            
            if verbose:
                print(f"✅ Found {len(relations)} relations:")
                for relation in relations:
                    source_entity = next((e for e in entities if e.id == relation.source_entity_id), None)
                    target_entity = next((e for e in entities if e.id == relation.target_entity_id), None)
                    
                    if source_entity and target_entity:
                        print(f"  • {source_entity.name} --[{relation.type.value}]--> {target_entity.name} (confidence: {relation.confidence:.2f})")
        
        except Exception as e:
            print(f"❌ Error during relation extraction: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    return results


def save_results(results: Dict[str, Any], output_file: Path, verbose: bool = False) -> bool:
    """Save extraction results to JSON file.
    
    Args:
        results: Extraction results dictionary
        output_file: Path to output JSON file
        verbose: Show detailed output
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"💾 Results saved to: {output_file}")
            print(f"📊 Statistics:")
            stats = results.get("statistics", {})
            print(f"   • Entities: {stats.get('entity_count', 0)}")
            print(f"   • Relations: {stats.get('relation_count', 0)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving results to {output_file}: {e}")
        return False


async def main():
    """Main function to run extraction."""
    parser = argparse.ArgumentParser(
        description="Extract entities and relations from markdown files using morag-graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_extraction.py document.md                    # Extract entities and relations
  python run_extraction.py document.md --entity-only     # Extract only entities
  python run_extraction.py document.md --relation-only   # Extract only relations
  python run_extraction.py document.md --verbose         # Show detailed output
  python run_extraction.py document.md --output result.json  # Custom output file
"""
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input markdown file to process"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--entity-only",
        action="store_true",
        help="Extract only entities"
    )
    
    parser.add_argument(
        "--relation-only",
        action="store_true",
        help="Extract only relations (requires entities first)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model to use (default: gemini-1.5-flash)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed extraction output"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file (default: input_file.json)"
    )
    
    args = parser.parse_args()
    
    print("🧠 MoRAG Graph - Entity and Relation Extraction")
    print("" + "="*60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Setup environment
    if not setup_environment(args.api_key):
        return 1
    
    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    if not input_file.is_file():
        print(f"❌ Input path is not a file: {input_file}")
        return 1
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.with_suffix('.json')
    
    # Validate arguments
    if args.entity_only and args.relation_only:
        print("❌ Cannot specify both --entity-only and --relation-only")
        return 1
    
    try:
        print(f"📄 Processing: {input_file.name}")
        print(f"🤖 Using model: {args.model}")
        print(f"💾 Output file: {output_file.name}")
        print("" + "="*60)
        
        # Run extraction
        results = await extract_from_file(
            input_file=input_file,
            api_key=args.api_key or os.getenv("GEMINI_API_KEY"),
            model=args.model,
            entity_only=args.entity_only,
            relation_only=args.relation_only,
            verbose=args.verbose
        )
        
        if not results:
            print("❌ Extraction failed")
            return 1
        
        # Save results
        if save_results(results, output_file, args.verbose):
            print("" + "="*60)
            print("✅ Extraction completed successfully!")
            print(f"📁 Results saved to: {output_file}")
            return 0
        else:
            return 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Extraction interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))