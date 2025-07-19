#!/usr/bin/env python3
"""
Graphiti-based ingestion script for MoRAG knowledge graphs.

This script provides a modern approach to knowledge graph ingestion using Graphiti's
episode-based architecture with built-in entity extraction and deduplication.

Usage:
    python run_graphiti_ingestion.py ingest <file> [options]
    python run_graphiti_ingestion.py search <query> [options]
    python run_graphiti_ingestion.py status

Examples:
    # Ingest a text file
    python run_graphiti_ingestion.py ingest document.txt --title "Research Paper"
    
    # Ingest with metadata
    python run_graphiti_ingestion.py ingest data.txt --metadata '{"category": "research"}'
    
    # Search the knowledge graph
    python run_graphiti_ingestion.py search "artificial intelligence" --limit 5
    
    # Check system status
    python run_graphiti_ingestion.py status
"""

import json
import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from morag_graph.graphiti import (
        GraphitiConfig, GraphitiConnectionService, 
        DocumentEpisodeMapper, GraphitiSearchService
    )
    from morag_graph.models import Document
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the morag-graph package:")
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


def get_graphiti_config() -> GraphitiConfig:
    """Get Graphiti configuration from environment variables."""
    # Get API key (try Gemini first, then OpenAI)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY environment variable.")
    
    return GraphitiConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        openai_api_key=api_key,
        openai_model=os.getenv("GRAPHITI_MODEL", "gpt-4"),
        openai_embedding_model=os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


async def ingest_file(
    file_path: Path,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Ingest a file into Graphiti knowledge graph."""
    print_header("Graphiti File Ingestion")
    
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    print_result("Input File", str(file_path))
    print_result("File Size", f"{file_path.stat().st_size / 1024:.2f} KB")
    print_result("Title", title or file_path.stem)
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")
    
    try:
        # Read file content
        print_section("Reading File")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print_result("Content Length", f"{len(content)} characters")
        print_result("Word Count", f"~{len(content.split())} words")
        
        # Get Graphiti configuration
        config = get_graphiti_config()
        print_section("Graphiti Configuration")
        print_result("Neo4j URI", config.neo4j_uri)
        print_result("Database", config.neo4j_database)
        print_result("Model", config.openai_model)
        
        # Create connection service
        print_section("Connecting to Graphiti")
        connection_service = GraphitiConnectionService(config)
        connected = await connection_service.connect()
        
        if not connected:
            print("❌ Failed to connect to Graphiti")
            return False
        
        print("✅ Connected to Graphiti successfully")
        
        # Create episode
        print_section("Creating Episode")
        episode_name = title or f"Document: {file_path.stem}"
        source_description = f"File ingestion: {file_path.name}"
        
        success = await connection_service.create_episode(
            name=episode_name,
            content=content,
            source_description=source_description,
            metadata=metadata or {}
        )
        
        if success:
            print_section("✅ Ingestion Complete")
            print_result("Episode Name", episode_name)
            print_result("Source", source_description)
            print_result("Content Length", f"{len(content)} characters")
            print("\n🎉 File successfully ingested into Graphiti knowledge graph!")
            print("💡 The content is now available for semantic search and knowledge graph queries.")
        else:
            print("❌ Failed to create episode")
            return False
        
        # Disconnect
        await connection_service.disconnect()
        return True
        
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
        # Get configuration and create search service
        config = get_graphiti_config()
        search_service = GraphitiSearchService(config)
        
        print_section("Searching Knowledge Graph")
        results, metrics = await search_service.search(
            query=query,
            limit=limit,
            search_type=search_type
        )
        
        print_section(f"✅ Search Results ({len(results)} found)")
        
        if len(results) == 0:
            print("   No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n   Result {i}:")
                print(f"   Content: {result.content[:200]}...")
                if hasattr(result, 'score'):
                    print(f"   Score: {result.score}")
        
        if metrics:
            print_section("📊 Search Metrics")
            for key, value in metrics.items():
                print_result(key.replace('_', ' ').title(), str(value))
        
        return True
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_status():
    """Check Graphiti and Neo4j status."""
    print_header("Graphiti System Status")
    
    try:
        # Test configuration
        config = get_graphiti_config()
        print_section("Configuration")
        print_result("Neo4j URI", config.neo4j_uri)
        print_result("Database", config.neo4j_database)
        print_result("Model", config.openai_model)
        print_result("Embedding Model", config.openai_embedding_model)
        
        # Test Graphiti connection
        print_section("Graphiti Connection")
        connection_service = GraphitiConnectionService(config)
        connected = await connection_service.connect()
        
        if connected:
            print("✅ Graphiti connection: OK")
            await connection_service.disconnect()
        else:
            print("❌ Graphiti connection: Failed")
            return False
        
        # Test Neo4j directly
        print_section("Neo4j Database")
        try:
            import neo4j
            driver = neo4j.GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_username, config.neo4j_password)
            )
            
            with driver.session(database=config.neo4j_database) as session:
                # Check for episodes
                result = session.run("MATCH (e:Episodic) RETURN count(e) as episode_count")
                episode_count = result.single()["episode_count"]
                print_result("Episodes", str(episode_count))
                
                # Check for entities
                result = session.run("MATCH (e:Entity) RETURN count(e) as entity_count")
                entity_count = result.single()["entity_count"]
                print_result("Entities", str(entity_count))
                
                # Check for relationships
                result = session.run("MATCH ()-[r]->() RETURN count(r) as relation_count")
                relation_count = result.single()["relation_count"]
                print_result("Relationships", str(relation_count))
            
            driver.close()
            print("✅ Neo4j database: OK")
            
        except Exception as e:
            print(f"❌ Neo4j database error: {e}")
            return False
        
        print_section("✅ System Status: All OK")
        return True
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Graphiti-based knowledge graph ingestion for MoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a file into knowledge graph')
    ingest_parser.add_argument('file', help='File to ingest')
    ingest_parser.add_argument('--title', help='Episode title (default: filename)')
    ingest_parser.add_argument('--metadata', help='Additional metadata as JSON string')
    
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
            file_path = Path(args.file)
            metadata = None
            if args.metadata:
                try:
                    metadata = json.loads(args.metadata)
                except json.JSONDecodeError as e:
                    print(f"❌ Error: Invalid JSON in metadata: {e}")
                    return 1
            
            success = asyncio.run(ingest_file(
                file_path,
                title=args.title,
                metadata=metadata
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
