#!/usr/bin/env python3
"""
MoRAG Graphiti Knowledge Graph CLI Script

This script provides direct access to Graphiti functionality for knowledge graph operations.

Usage:
    python test-graphiti.py ingest <file> [options]
    python test-graphiti.py search <query> [options]
    python test-graphiti.py status

Commands:
    ingest <file>    Ingest a document into Graphiti knowledge graph
    search <query>   Search the knowledge graph
    status          Check Graphiti and Neo4j status

Examples:
    # Ingest documents
    python test-graphiti.py ingest my-document.pdf
    python test-graphiti.py ingest research.docx --metadata '{"category": "research"}'
    
    # Search knowledge graph
    python test-graphiti.py search "artificial intelligence"
    python test-graphiti.py search "machine learning" --limit 5
    
    # Check status
    python test-graphiti.py status
"""

import sys
import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

try:
    from morag_document import DocumentProcessor
    from graph_extraction import extract_and_ingest_with_graphiti, search_with_graphiti, GraphitiExtractionService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-document")
    print("  pip install -e packages/morag-graph")
    sys.exit(1)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str):
    """Print a section header."""
    print(f"\n📋 {title}")
    print("-" * 40)


def print_result(key: str, value: str):
    """Print a key-value result."""
    print(f"   {key}: {value}")


async def ingest_document(
    document_file: Path,
    metadata: Optional[Dict[str, Any]] = None,
    chunking_strategy: str = "paragraph",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> bool:
    """Ingest a document into Graphiti knowledge graph."""
    print_header("Graphiti Document Ingestion")

    if not document_file.exists():
        print(f"❌ Error: Document file not found: {document_file}")
        return False

    print_result("Input File", str(document_file))
    print_result("File Size", f"{document_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("Chunking Strategy", chunking_strategy)
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")

    try:
        print_section("Processing Document")
        
        # Process document
        processor = DocumentProcessor()
        result = await processor.process_file(
            document_file,
            extract_metadata=True,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not result.success:
            print(f"❌ Document processing failed: {result.error}")
            return False

        print(f"✅ Document processed successfully")
        print_result("Title", result.document.metadata.get('title', 'Unknown'))
        print_result("Chunks", str(len(result.chunks)))

        # Prepare content for Graphiti
        full_content = "\n\n".join([chunk.content for chunk in result.chunks])
        doc_id = f"doc_{document_file.stem}_{hash(str(document_file))}"
        title = result.document.metadata.get('title') or document_file.stem

        graphiti_metadata = {
            'source_file': str(document_file),
            'file_type': document_file.suffix.lower(),
            'chunk_count': len(result.chunks),
            'processing_strategy': chunking_strategy
        }
        if metadata:
            graphiti_metadata.update(metadata)

        print_section("Graphiti Ingestion")
        
        # Ingest using Graphiti
        graphiti_result = await extract_and_ingest_with_graphiti(
            text_content=full_content,
            doc_id=doc_id,
            title=title,
            context=f"Document: {document_file.name}",
            metadata=graphiti_metadata
        )

        if graphiti_result['graphiti']['success']:
            print_section("✅ Ingestion Complete")
            print_result("Episode Name", graphiti_result['graphiti']['episode_name'])
            print_result("Content Length", f"{graphiti_result['graphiti']['content_length']} characters")
            print_result("Document ID", doc_id)
            print("\n🎉 Document successfully ingested into Graphiti knowledge graph!")
            return True
        else:
            print(f"❌ Graphiti ingestion failed: {graphiti_result['graphiti'].get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


async def search_knowledge_graph(
    query: str,
    limit: int = 10,
    search_type: str = "hybrid"
) -> bool:
    """Search the Graphiti knowledge graph."""
    print_header("Graphiti Knowledge Graph Search")
    
    print_result("Query", query)
    print_result("Limit", str(limit))
    print_result("Search Type", search_type)

    try:
        search_result = await search_with_graphiti(
            query=query,
            limit=limit,
            search_type=search_type
        )

        if search_result['success']:
            print_section(f"✅ Search Results ({search_result['count']} found)")
            
            if search_result['count'] == 0:
                print("   No results found.")
            else:
                for i, result in enumerate(search_result['results'], 1):
                    print(f"\n   Result {i}:")
                    print(f"   Content: {result.content[:200]}...")
                    if hasattr(result, 'score'):
                        print(f"   Score: {result.score}")
            
            return True
        else:
            print(f"❌ Search failed: {search_result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_status():
    """Check Graphiti and Neo4j status."""
    print_header("Graphiti System Status")

    try:
        # Test Graphiti connection
        graphiti_service = GraphitiExtractionService(use_graphiti=True)
        connection_service = await graphiti_service.get_connection_service()
        
        if connection_service.is_connected:
            print("✅ Graphiti connection: OK")
            print_result("Neo4j URI", connection_service._graphiti_config.neo4j_uri if connection_service._graphiti_config else "Unknown")
            print_result("Database", connection_service._graphiti_config.neo4j_database if connection_service._graphiti_config else "Unknown")
        else:
            print("❌ Graphiti connection: Failed")
            
        await graphiti_service.close()
        
        # Test Neo4j directly
        try:
            import neo4j
            driver = neo4j.GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "password")
            )
            
            with driver.session() as session:
                result = session.run("MATCH (e:Episodic) RETURN count(e) as episode_count")
                episode_count = result.single()["episode_count"]
                print_result("Episodes in Neo4j", str(episode_count))
                
                result = session.run("MATCH (e:Entity) RETURN count(e) as entity_count")
                entity_count = result.single()["entity_count"]
                print_result("Entities in Neo4j", str(entity_count))
            
            driver.close()
            print("✅ Neo4j direct connection: OK")
            
        except Exception as e:
            print(f"❌ Neo4j direct connection failed: {e}")
        
        return True

    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MoRAG Graphiti Knowledge Graph CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a document')
    ingest_parser.add_argument('file', help='Document file to ingest')
    ingest_parser.add_argument('--metadata', help='Additional metadata as JSON string')
    ingest_parser.add_argument('--chunking-strategy', choices=['paragraph', 'sentence', 'page', 'chapter'],
                              default='paragraph', help='Chunking strategy (default: paragraph)')
    ingest_parser.add_argument('--chunk-size', type=int, default=1000,
                              help='Maximum chunk size in characters (default: 1000)')
    ingest_parser.add_argument('--chunk-overlap', type=int, default=200,
                              help='Overlap between chunks in characters (default: 200)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search the knowledge graph')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Maximum number of results (default: 10)')
    search_parser.add_argument('--search-type', choices=['hybrid', 'semantic', 'keyword'],
                              default='hybrid', help='Search type (default: hybrid)')
    
    # Status command
    subparsers.add_parser('status', help='Check system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'ingest':
            document_file = Path(args.file)
            metadata = None
            if args.metadata:
                try:
                    metadata = json.loads(args.metadata)
                except json.JSONDecodeError as e:
                    print(f"❌ Error: Invalid JSON in metadata: {e}")
                    return 1
            
            success = asyncio.run(ingest_document(
                document_file,
                metadata=metadata,
                chunking_strategy=args.chunking_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            ))
            return 0 if success else 1
            
        elif args.command == 'search':
            success = asyncio.run(search_knowledge_graph(
                args.query,
                limit=args.limit,
                search_type=args.search_type
            ))
            return 0 if success else 1
            
        elif args.command == 'status':
            success = asyncio.run(check_status())
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
