#!/usr/bin/env python3
"""Fact extraction runner script for morag-graph package.

This script provides an easy way to run fact extraction on markdown files.
It extracts structured facts from the input file and saves the results to a JSON file.

Usage:
    python run_extraction.py input.md --api-key YOUR_GEMINI_API_KEY

Or set GEMINI_API_KEY environment variable and run:
    python run_extraction.py input.md

Options:
    --domain         Specify domain context (default: general)
    --model          Specify LLM model (default: gemini-2.0-flash)
    --verbose        Show detailed extraction output
    --output         Specify output file (default: input_file.json)
    --min-confidence Minimum confidence threshold (default: 0.3)
    --max-facts      Maximum facts per chunk (default: 10)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from morag_graph.extraction.fact_extractor import FactExtractor
    from morag_graph.extraction.fact_graph_builder import FactGraphBuilder
    from morag_graph.models.fact import Fact, FactRelation
    from morag_graph.models.document_chunk import DocumentChunk
    from morag_graph.services.fact_extraction_service import FactExtractionService
    from morag_graph.storage.neo4j_storage import Neo4jStorage
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
    # Map package names to their import names
    required_packages = {
        "google-generativeai": "google.generativeai",
        "httpx": "httpx",
        "pydantic": "pydantic",
        "python-dotenv": "dotenv",
        "aiofiles": "aiofiles"
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
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


async def extract_facts_from_file(
    input_file: Path,
    api_key: str,
    model: str = "gemini-2.0-flash",
    domain: str = "general",
    min_confidence: float = 0.3,
    max_facts: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """Extract structured facts from a markdown file.

    Args:
        input_file: Path to input markdown file
        api_key: Gemini API key
        model: LLM model to use
        domain: Domain context for extraction
        min_confidence: Minimum confidence threshold
        max_facts: Maximum facts per chunk
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

    # Detect language from content
    def detect_language(text):
        """Simple language detection based on common words."""
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'mit', 'von', 'zu', 'auf', 'für', 'über', 'kann', 'sind', 'haben', 'werden']
        english_indicators = ['the', 'and', 'is', 'a', 'an', 'with', 'of', 'to', 'on', 'for', 'over', 'can', 'are', 'have', 'will']

        text_lower = text.lower()
        german_count = sum(1 for word in german_indicators if f' {word} ' in text_lower)
        english_count = sum(1 for word in english_indicators if f' {word} ' in text_lower)

        return "de" if german_count > english_count else "en"

    detected_language = detect_language(content)

    if verbose:
        print(f"📄 Processing file: {input_file.name}")
        print(f"📝 Content length: {len(content)} characters")
        print(f"🌍 Detected language: {detected_language}")
        print(f"🎯 Domain: {domain}")
        print(f"🎚️ Min confidence: {min_confidence}")
        print(f"📊 Max facts per chunk: {max_facts}")

    results = {
        "source_file": str(input_file),
        "model": model,
        "domain": domain,
        "language": detected_language,
        "facts": [],
        "relationships": [],
        "statistics": {
            "facts_extracted": 0,
            "relationships_created": 0,
            "chunks_processed": 0,
            "fact_types": {}
        }
    }

    try:
        if verbose:
            print("🔍 Extracting facts...")

        # Initialize variables
        facts = []

        # Initialize fact extractor
        fact_extractor = FactExtractor(
            model_id=model,
            api_key=api_key,
            min_confidence=min_confidence,
            max_facts_per_chunk=max_facts,
            domain=domain,
            language=detected_language,
            require_entities=False,  # Allow facts without strict entity requirements
            strict_validation=False  # Use more lenient validation
        )

        # Create a document chunk from the content
        document_id = f"doc_{input_file.stem}"
        chunk = DocumentChunk(
            id=f"{document_id}:chunk:0",
            document_id=document_id,
            chunk_index=0,
            text=content,
            metadata={
                "source_file": str(input_file),
                "file_name": input_file.name
            }
        )

        # Extract facts from the chunk
        facts = await fact_extractor.extract_facts(
            chunk_text=content,
            chunk_id=chunk.id,
            document_id=document_id,
            context={
                "domain": domain,
                "language": detected_language
            }
        )

        # Convert facts to serializable format
        facts_data = []
        fact_types = {}

        for fact in facts:
            fact_dict = {
                "id": fact.id,
                "fact_text": fact.fact_text,
                "structured_metadata": {
                    "primary_entities": fact.structured_metadata.primary_entities if fact.structured_metadata else [],
                    "relationships": fact.structured_metadata.relationships if fact.structured_metadata else [],
                    "domain_concepts": fact.structured_metadata.domain_concepts if fact.structured_metadata else []
                } if fact.structured_metadata else None,
                "fact_type": fact.fact_type,
                "domain": fact.domain,
                "confidence": fact.extraction_confidence,
                "keywords": fact.keywords,
                "source_chunk_id": fact.source_chunk_id,
                "source_document_id": fact.source_document_id,
                "created_at": fact.created_at.isoformat(),
                "language": fact.language
            }
            facts_data.append(fact_dict)

            # Count fact types
            fact_types[fact.fact_type] = fact_types.get(fact.fact_type, 0) + 1

        results["facts"] = facts_data
        results["statistics"]["facts_extracted"] = len(facts)
        results["statistics"]["chunks_processed"] = 1
        results["statistics"]["fact_types"] = fact_types

        if verbose:
            print(f"✅ Extracted {len(facts)} facts:")
            for fact in facts:
                entities = fact.structured_metadata.primary_entities if fact.structured_metadata and fact.structured_metadata.primary_entities else ["N/A"]
                # Ensure entities are strings (filter out any EntityRelationship objects that might have been mixed in)
                string_entities = [str(entity) for entity in entities if isinstance(entity, (str, int, float))]
                if not string_entities:
                    string_entities = ["N/A"]
                entities_str = ", ".join(string_entities[:2])  # Show first 2 entities
                print(f"  • {fact.fact_text[:50]}... ({fact.fact_type}) - entities: {entities_str} - confidence: {fact.extraction_confidence:.2f}")

    except Exception as e:
        print(f"❌ Error during fact extraction: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

    # Extract fact relationships if we have multiple facts
    if len(facts) > 1:
        if verbose:
            print("🔗 Extracting fact relationships...")

        try:
            fact_graph_builder = FactGraphBuilder(
                model_id=model,
                api_key=api_key
            )

            # Build fact graph to identify relationships
            fact_graph = await fact_graph_builder.build_fact_graph(facts)

            # Extract relationships from the graph edges
            relationships = []
            if hasattr(fact_graph, 'edges'):
                if verbose:
                    print(f"🔍 Found {len(fact_graph.edges)} edges in graph")
                # Convert graph edges to FactRelation objects
                for edge in fact_graph.edges:
                    try:
                        # Create a FactRelation from the edge
                        relationship = FactRelation(
                            source_fact_id=edge.source,
                            target_fact_id=edge.target,
                            relation_type=edge.type,
                            confidence=edge.properties.get('confidence', 0.7),
                            context=edge.properties.get('context', '')
                        )
                        relationships.append(relationship)
                        if verbose:
                            print(f"✅ Created relationship: {edge.source} --[{edge.type}]--> {edge.target}")
                    except Exception as edge_error:
                        if verbose:
                            print(f"❌ Error creating relationship from edge: {edge_error}")
                            print(f"   Edge: {edge}")
            else:
                if verbose:
                    print("❌ No edges found in fact graph")

            # Convert relationships to serializable format
            relationships_data = []
            relationship_types = {}

            for relationship in relationships:
                try:
                    relationship_dict = {
                        "id": getattr(relationship, 'id', f"rel_{len(relationships_data)}"),
                        "source_fact_id": relationship.source_fact_id,
                        "target_fact_id": relationship.target_fact_id,
                        "relationship_type": relationship.relation_type,
                        "confidence": relationship.confidence,
                        "description": getattr(relationship, 'description', relationship.context),
                        "created_at": getattr(relationship, 'created_at', datetime.now()).isoformat() if hasattr(getattr(relationship, 'created_at', datetime.now()), 'isoformat') else str(getattr(relationship, 'created_at', datetime.now()))
                    }
                    relationships_data.append(relationship_dict)

                    # Count relationship types
                    relationship_types[relationship.relation_type] = relationship_types.get(relationship.relation_type, 0) + 1
                except Exception as e:
                    if verbose:
                        print(f"❌ Error serializing relationship: {e}")
                        print(f"   Relationship attributes: {dir(relationship)}")
                    continue

            results["relationships"] = relationships_data
            results["statistics"]["relationships_created"] = len(relationships)

            if verbose:
                print(f"✅ Found {len(relationships)} fact relationships:")
                for rel in relationships:
                    source_fact = next((f for f in facts if f.id == rel.source_fact_id), None)
                    target_fact = next((f for f in facts if f.id == rel.target_fact_id), None)

                    if source_fact and target_fact:
                        # Show fact text preview instead of entities (since entity extraction may not be enabled)
                        source_preview = source_fact.fact_text[:50] + "..." if len(source_fact.fact_text) > 50 else source_fact.fact_text
                        target_preview = target_fact.fact_text[:50] + "..." if len(target_fact.fact_text) > 50 else target_fact.fact_text
                        print(f"  • \"{source_preview}\" --[{rel.relation_type}]--> \"{target_preview}\"")

        except Exception as e:
            print(f"❌ Error during relationship extraction: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            if verbose:
                import traceback
                traceback.print_exc()
            # Still return results even if relationship extraction fails
            pass

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
    def json_serializer(obj):
        """Custom JSON serializer for Pydantic models and other objects."""
        if hasattr(obj, 'model_dump'):
            # Pydantic v2 models
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            # Pydantic v1 models
            return obj.dict()
        elif hasattr(obj, 'to_dict'):
            # Custom to_dict method
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            # Generic objects with __dict__
            return obj.__dict__
        else:
            # Fallback to string representation
            return str(obj)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        if verbose:
            print(f"💾 Results saved to: {output_file}")
            print(f"📊 Statistics:")
            stats = results.get("statistics", {})
            print(f"   • Facts: {stats.get('facts_extracted', 0)}")
            print(f"   • Relationships: {stats.get('relationships_created', 0)}")
            print(f"   • Chunks processed: {stats.get('chunks_processed', 0)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving results to {output_file}: {e}")
        return False


async def main():
    """Main function to run extraction."""
    parser = argparse.ArgumentParser(
        description="Extract structured facts from markdown files using morag-graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_extraction.py document.md                           # Extract facts with default settings
  python run_extraction.py document.md --domain research        # Extract facts for research domain
  python run_extraction.py document.md --min-confidence 0.5     # Higher confidence threshold
  python run_extraction.py document.md --max-facts 20           # More facts per chunk
  python run_extraction.py document.md --verbose                # Show detailed output
  python run_extraction.py document.md --output result.json     # Custom output file
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
        "--domain",
        type=str,
        default="general",
        help="Domain context for extraction (default: general)"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence threshold (default: 0.3)"
    )

    parser.add_argument(
        "--max-facts",
        type=int,
        default=10,
        help="Maximum facts per chunk (default: 10)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="LLM model to use (default: gemini-2.0-flash)"
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
    
    print("🧠 MoRAG Graph - Fact Extraction")
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
    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        print("❌ Minimum confidence must be between 0.0 and 1.0")
        return 1

    if args.max_facts < 1:
        print("❌ Maximum facts must be at least 1")
        return 1
    
    try:
        print(f"📄 Processing: {input_file.name}")
        print(f"🤖 Using model: {args.model}")
        print(f"🎯 Domain: {args.domain}")
        print(f"🎚️ Min confidence: {args.min_confidence}")
        print(f"� Max facts: {args.max_facts}")
        print(f"�💾 Output file: {output_file.name}")
        print("" + "="*60)

        # Run fact extraction
        results = await extract_facts_from_file(
            input_file=input_file,
            api_key=args.api_key or os.getenv("GEMINI_API_KEY"),
            model=args.model,
            domain=args.domain,
            min_confidence=args.min_confidence,
            max_facts=args.max_facts,
            verbose=args.verbose
        )
        
        if not results:
            print("❌ Extraction failed")
            return 1
        
        # Save results
        if save_results(results, output_file, args.verbose):
            print("" + "="*60)
            print("✅ Fact extraction completed successfully!")
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