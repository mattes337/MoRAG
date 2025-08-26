#!/usr/bin/env python3
"""CLI command for entity deduplication in MoRAG knowledge graphs."""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_graph.extraction.systematic_deduplicator import SystematicDeduplicator
    from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
    from morag_core.ai import create_agent_with_config
    from morag_core.ai.providers import GeminiProvider
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


class EntityDeduplicationCLI:
    """CLI interface for entity deduplication."""
    
    def __init__(self):
        self.neo4j_storage = None
        self.deduplicator = None
    
    async def setup_services(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, neo4j_database: str = "neo4j"):
        """Setup Neo4j storage and deduplicator services."""
        try:
            # Setup Neo4j configuration
            neo4j_config = Neo4jConfig(
                uri=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
                database=neo4j_database
            )
            
            # Initialize Neo4j storage
            self.neo4j_storage = Neo4jStorage(neo4j_config)
            await self.neo4j_storage.connect()
            
            # Setup LLM service if API key is available
            llm_service = None
            if os.getenv("GEMINI_API_KEY"):
                try:
                    llm_service = GeminiProvider()
                    print("✅ LLM service initialized for enhanced normalization")
                except Exception as e:
                    print(f"⚠️  LLM service initialization failed: {e}")
                    print("   Falling back to rule-based normalization")
            else:
                print("⚠️  GEMINI_API_KEY not found. Using rule-based normalization only.")
            
            # Initialize deduplicator
            self.deduplicator = SystematicDeduplicator(
                similarity_threshold=0.7,
                merge_confidence_threshold=0.8,
                enable_llm_validation=llm_service is not None
            )
            
            print("✅ Services initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to setup services: {e}")
            raise
    
    async def run_deduplication(self, collection_name: Optional[str] = None, language: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Run entity deduplication."""
        if not self.deduplicator:
            raise RuntimeError("Services not initialized. Call setup_services first.")

        print(f"🔍 Starting entity deduplication...")
        print(f"   Collection: {collection_name or 'All collections'}")
        print(f"   Language: {language or 'Auto-detect'}")
        print(f"   Mode: {'Dry run' if dry_run else 'Apply changes'}")
        print()

        try:
            # Get all entities from Neo4j (simplified approach)
            # Note: This is a simplified implementation - the original CLI interface
            # doesn't match the current SystematicDeduplicator interface
            print("⚠️  Note: This CLI uses a simplified deduplication approach.")
            print("   For full deduplication, use the maintenance jobs or graph extraction pipeline.")

            # Return a basic result structure
            result = {
                "status": "completed",
                "message": "Deduplication interface updated - use maintenance jobs for full functionality",
                "dry_run": dry_run
            }
            return result
        except Exception as e:
            print(f"❌ Deduplication failed: {e}")
            raise
    
    async def show_candidates(self, collection_name: Optional[str] = None, language: Optional[str] = None) -> None:
        """Show merge candidates without applying changes."""
        if not self.deduplicator:
            raise RuntimeError("Services not initialized. Call setup_services first.")
        
        print(f"🔍 Finding duplicate entity candidates...")
        print(f"   Collection: {collection_name or 'All collections'}")
        print(f"   Language: {language or 'Auto-detect'}")
        print()
        
        try:
            print("⚠️  Note: Candidate preview not available with current interface.")
            print("   Use the maintenance jobs for detailed deduplication analysis.")
            print("   See: python scripts/maintenance_runner.py --help")

        except Exception as e:
            print(f"❌ Failed to find candidates: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.neo4j_storage:
            await self.neo4j_storage.close()


def print_results(result: Dict[str, Any], dry_run: bool = False):
    """Print deduplication results in a formatted way."""
    print("📊 Deduplication Results:")
    print(f"   Entities before: {result['total_entities_before']}")
    print(f"   Merge candidates found: {result['merge_candidates_found']}")
    
    if dry_run:
        print(f"   Merges that would be applied: {len([c for c in result.get('candidates', []) if c['confidence'] >= 0.8])}")
        print()
        print("🔍 Merge candidates (dry run):")
        for candidate in result.get('candidates', []):
            if candidate['confidence'] >= 0.8:
                print(f"   • {', '.join(candidate['entities'])} → {candidate['canonical_form']} (confidence: {candidate['confidence']:.2f})")
    else:
        print(f"   Merges applied: {result['merges_applied']}")
        print()
        if result['merge_results']:
            print("✅ Applied merges:")
            for merge in result['merge_results']:
                print(f"   • {', '.join(merge['merged_entities'])} → {merge['canonical_form']} (confidence: {merge['confidence']:.2f})")


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Deduplicate entities in MoRAG knowledge graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show duplicate candidates without making changes
  python cli/deduplicate-entities.py --dry-run

  # Deduplicate entities in a specific collection
  python cli/deduplicate-entities.py --collection my_collection

  # Deduplicate with language context
  python cli/deduplicate-entities.py --language en --collection docs

  # Apply deduplication to entire graph
  python cli/deduplicate-entities.py

Environment variables:
  NEO4J_URI          Neo4j connection URI (default: neo4j://localhost:7687)
  NEO4J_USERNAME     Neo4j username (default: neo4j)
  NEO4J_PASSWORD     Neo4j password (required)
  NEO4J_DATABASE     Neo4j database name (default: neo4j)
  GEMINI_API_KEY     Google Gemini API key for enhanced normalization (optional)
        """
    )
    
    parser.add_argument('--collection', '-c', help='Collection name to deduplicate (optional)')
    parser.add_argument('--language', '-l', help='Language context for normalization (e.g., en, es, de)')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Show candidates without applying merges')
    parser.add_argument('--candidates-only', action='store_true', help='Only show merge candidates (same as --dry-run)')
    parser.add_argument('--neo4j-uri', default=os.getenv('NEO4J_URI', 'neo4j://localhost:7687'), help='Neo4j URI')
    parser.add_argument('--neo4j-username', default=os.getenv('NEO4J_USERNAME', 'neo4j'), help='Neo4j username')
    parser.add_argument('--neo4j-password', default=os.getenv('NEO4J_PASSWORD'), help='Neo4j password')
    parser.add_argument('--neo4j-database', default=os.getenv('NEO4J_DATABASE', 'neo4j'), help='Neo4j database')
    parser.add_argument('--output', '-o', help='Output results to JSON file')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.neo4j_password:
        print("❌ Neo4j password is required. Set NEO4J_PASSWORD environment variable or use --neo4j-password")
        sys.exit(1)
    
    # Handle aliases
    if args.candidates_only:
        args.dry_run = True
    
    cli = EntityDeduplicationCLI()
    
    try:
        # Setup services
        await cli.setup_services(
            args.neo4j_uri,
            args.neo4j_username, 
            args.neo4j_password,
            args.neo4j_database
        )
        
        # Run deduplication or show candidates
        if args.dry_run:
            await cli.show_candidates(args.collection, args.language)
        else:
            result = await cli.run_deduplication(args.collection, args.language, dry_run=False)
            print_results(result, dry_run=False)

            # Show metrics if available
            if hasattr(cli.deduplicator, 'normalizer') and hasattr(cli.deduplicator.normalizer, 'get_metrics'):
                print("\n📊 Normalization Metrics:")
                metrics = cli.deduplicator.normalizer.get_metrics()
                print(f"   Entities processed: {metrics.get('entities_processed', 0)}")
                print(f"   Entities normalized: {metrics.get('entities_normalized', 0)}")
                print(f"   Average confidence: {metrics.get('average_confidence', 0):.3f}")
                print(f"   Processing time: {metrics.get('total_processing_time', 0):.2f}s")
                if metrics.get('error_rate', 0) > 0:
                    print(f"   Error rate: {metrics.get('error_rate', 0):.1f}%")

            # Save results to file if requested
            if args.output:
                # Include metrics in output
                output_data = result.copy()
                if hasattr(cli.deduplicator, 'normalizer') and hasattr(cli.deduplicator.normalizer, 'get_metrics'):
                    output_data['metrics'] = cli.deduplicator.normalizer.get_metrics()

                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"📄 Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
