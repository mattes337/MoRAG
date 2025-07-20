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
    # Basic ingestion with hybrid strategy (recommended)
    python test-graphiti.py ingest my-document.pdf
    python test-graphiti.py ingest research.docx --metadata '{"category": "research"}'

    # Different episode strategies
    python test-graphiti.py ingest document.pdf --episode-strategy hybrid --context-level rich
    python test-graphiti.py ingest document.pdf --episode-strategy contextual_chunks --context-level comprehensive
    python test-graphiti.py ingest document.pdf --episode-strategy document_only --context-level minimal
    python test-graphiti.py ingest document.pdf --episode-strategy chunk_only --context-level standard

    # Custom episode naming and AI options
    python test-graphiti.py ingest report.pdf --episode-prefix "quarterly_report" --context-level rich
    python test-graphiti.py ingest document.pdf --disable-ai-summarization --context-level minimal

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
    chunk_overlap: int = 200,
    episode_strategy: str = "hybrid",
    context_level: str = "rich",
    enable_ai_summarization: bool = True,
    episode_prefix: Optional[str] = None
) -> bool:
    """Ingest a document into Graphiti knowledge graph using hybrid episode strategy."""
    print_header("Graphiti Document Ingestion with Hybrid Episodes")

    if not document_file.exists():
        print(f"❌ Error: Document file not found: {document_file}")
        return False

    print_result("Input File", str(document_file))
    print_result("File Size", f"{document_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("Chunking Strategy", chunking_strategy)
    print_result("Episode Strategy", episode_strategy)
    print_result("Context Level", context_level)
    print_result("AI Summarization", "✅ Enabled" if enable_ai_summarization else "❌ Disabled")
    print_result("Episode Prefix", episode_prefix or "Auto-generated")
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

        print_section("Hybrid Episode Mapping")

        # Import hybrid episode mapper
        from morag_graph.graphiti import (
            DocumentEpisodeMapper, EpisodeStrategy, ContextLevel,
            GraphitiConfig, create_hybrid_episode_mapper
        )

        # Convert string values to enums
        strategy_map = {
            'document_only': EpisodeStrategy.DOCUMENT_ONLY,
            'chunk_only': EpisodeStrategy.CHUNK_ONLY,
            'contextual_chunks': EpisodeStrategy.CONTEXTUAL_CHUNKS,
            'hybrid': EpisodeStrategy.HYBRID
        }

        context_map = {
            'minimal': ContextLevel.MINIMAL,
            'standard': ContextLevel.STANDARD,
            'rich': ContextLevel.RICH,
            'comprehensive': ContextLevel.COMPREHENSIVE
        }

        strategy_enum = strategy_map[episode_strategy]
        context_enum = context_map[context_level]

        # Create Graphiti config with automatic API key detection
        # Determine which API key is available
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Choose appropriate models based on available API keys
        if gemini_key:
            # Use Gemini models if Gemini API key is available
            model = os.getenv("GRAPHITI_MODEL", "gemini-1.5-flash")
            embedding_model = os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-004")
            api_key = gemini_key
        elif openai_key:
            # Use OpenAI models if only OpenAI API key is available
            model = os.getenv("GRAPHITI_MODEL", "gpt-4")
            embedding_model = os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-small")
            api_key = openai_key
        else:
            # Default to Gemini models but will fail without API key
            model = "gemini-1.5-flash"
            embedding_model = "text-embedding-004"
            api_key = None

        config = GraphitiConfig(
            neo4j_uri=os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("GRAPHITI_NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("GRAPHITI_NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("GRAPHITI_NEO4J_DATABASE", "morag_graphiti"),
            openai_api_key=api_key,
            openai_model=model,
            openai_embedding_model=embedding_model,
            enable_telemetry=os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true",
            parallel_runtime=os.getenv("USE_PARALLEL_RUNTIME", "false").lower() == "true"
        )

        # Create episode mapper with specified strategy
        if episode_strategy == 'hybrid':
            mapper = create_hybrid_episode_mapper(
                config=config,
                context_level=context_enum
            )
        else:
            mapper = DocumentEpisodeMapper(
                config=config,
                strategy=strategy_enum,
                context_level=context_enum,
                enable_ai_summarization=enable_ai_summarization
            )

        # Convert processed result to Document model
        from morag_core.models import Document, DocumentMetadata

        doc_metadata = DocumentMetadata(
            title=title,
            source_name=document_file.name,
            source_type="document",
            mime_type=f"application/{document_file.suffix[1:]}" if document_file.suffix else "text/plain"
        )

        document = Document(metadata=doc_metadata)
        document.chunks = result.chunks
        document.raw_text = full_content

        # Determine episode prefix
        prefix = episode_prefix or f"{document_file.stem}"

        # Map document using selected strategy
        if episode_strategy == 'hybrid':
            mapping_result = await mapper.map_document_hybrid(
                document=document,
                episode_name_prefix=prefix,
                source_description=f"CLI ingestion: {document_file.name}"
            )
        elif episode_strategy == 'document_only':
            mapping_result = await mapper.map_document_to_episode(
                document=document,
                episode_name=f"{prefix}_document",
                source_description=f"CLI ingestion: {document_file.name}"
            )
        elif episode_strategy in ['chunk_only', 'contextual_chunks']:
            if episode_strategy == 'contextual_chunks':
                chunk_results = await mapper.map_document_chunks_to_contextual_episodes(
                    document=document,
                    chunk_episode_prefix=prefix
                )
            else:
                chunk_results = await mapper.map_document_chunks_to_episodes(
                    document=document,
                    chunk_episode_prefix=prefix
                )

            # Create summary result
            successful_chunks = [r for r in chunk_results if r.get('success')]
            mapping_result = {
                'success': len(successful_chunks) > 0,
                'strategy': episode_strategy,
                'chunk_episodes': chunk_results,
                'total_episodes': len(successful_chunks)
            }

        # Report results
        if mapping_result.get('success'):
            print_section("✅ Episode Mapping Complete")

            if episode_strategy == 'hybrid':
                print_result("Strategy", "Hybrid (Document + Contextual Chunks)")
                print_result("Total Episodes", str(mapping_result['total_episodes']))

                doc_ep = mapping_result.get('document_episode', {})
                if doc_ep.get('success'):
                    print_result("Document Episode", doc_ep['episode_name'])

                chunk_eps = mapping_result.get('chunk_episodes', [])
                successful_chunks = [ep for ep in chunk_eps if ep.get('success')]
                print_result("Chunk Episodes", f"{len(successful_chunks)}/{len(chunk_eps)} successful")

            elif episode_strategy == 'document_only':
                print_result("Strategy", "Document Only")
                print_result("Episode Name", mapping_result['episode_name'])
                print_result("Content Length", f"{mapping_result['content_length']} characters")

            else:  # chunk_only or contextual_chunks
                print_result("Strategy", f"{'Contextual ' if episode_strategy == 'contextual_chunks' else ''}Chunks Only")
                print_result("Total Episodes", str(mapping_result['total_episodes']))

                if episode_strategy == 'contextual_chunks':
                    chunk_eps = mapping_result.get('chunk_episodes', [])
                    for i, ep in enumerate(chunk_eps[:3]):  # Show first 3
                        if ep.get('success'):
                            print_result(f"Chunk {i+1}", f"{ep['episode_name']} (context: {len(ep.get('contextual_summary', ''))} chars)")

                    if len(chunk_eps) > 3:
                        print_result("...", f"and {len(chunk_eps) - 3} more chunks")

            print_result("Context Level", context_level.title())
            print_result("AI Summarization", "✅ Used" if enable_ai_summarization else "❌ Basic only")
            print("\n🎉 Document successfully ingested with hybrid episode strategy!")
            return True
        else:
            error_msg = mapping_result.get('error', 'Unknown error')
            print(f"❌ Episode mapping failed: {error_msg}")
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

    # Hybrid Episode Strategy Options
    ingest_parser.add_argument('--episode-strategy',
                              choices=['document_only', 'chunk_only', 'contextual_chunks', 'hybrid'],
                              default='hybrid',
                              help='Episode creation strategy (default: hybrid)')
    ingest_parser.add_argument('--context-level',
                              choices=['minimal', 'standard', 'rich', 'comprehensive'],
                              default='rich',
                              help='Context enrichment level (default: rich)')
    ingest_parser.add_argument('--disable-ai-summarization', action='store_true',
                              help='Disable AI-powered contextual summaries (use basic summaries)')
    ingest_parser.add_argument('--episode-prefix',
                              help='Custom prefix for episode names (default: auto-generated from filename)')
    
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
                chunk_overlap=args.chunk_overlap,
                episode_strategy=args.episode_strategy,
                context_level=args.context_level,
                enable_ai_summarization=not args.disable_ai_summarization,
                episode_prefix=args.episode_prefix
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
