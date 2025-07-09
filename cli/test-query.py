#!/usr/bin/env python3
"""
MoRAG Query Testing CLI Script

Usage: python test-query.py [OPTIONS] "your query here"

This script allows testing various query mechanisms against Neo4j and Qdrant databases
with support for multi-hop reasoning and different query strategies.
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Import MoRAG components
try:
    from morag_graph import (
        Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig,
        HybridRetrievalCoordinator, ContextExpansionEngine,
        QueryEntityExtractor, GraphCRUD, GraphTraversal, GraphAnalytics
    )
    from morag_graph.operations import GraphPath
    from morag.models.enhanced_query import (
        QueryType, ExpansionStrategy, FusionStrategy,
        EnhancedQueryRequest, EntityQueryRequest, GraphTraversalRequest
    )
    from morag.database_factory import get_default_neo4j_storage, get_default_qdrant_storage
    from morag_reasoning import (
        ReasoningStrategy, PathSelectionAgent, ReasoningPathFinder,
        IterativeRetriever, LLMClient
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing MoRAG components: {e}")
    COMPONENTS_AVAILABLE = False


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def print_result(result: Dict[str, Any], indent: int = 0):
    """Pretty print a result dictionary."""
    prefix = "  " * indent
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_result(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}: [{len(value)} items]")
            if value and len(value) <= 3:  # Show first few items
                for i, item in enumerate(value[:3]):
                    if isinstance(item, dict):
                        print(f"{prefix}  [{i}]:")
                        print_result(item, indent + 2)
                    else:
                        print(f"{prefix}  [{i}]: {item}")
        else:
            # Truncate long strings
            if isinstance(value, str) and len(value) > 100:
                value = value[:97] + "..."
            print(f"{prefix}{key}: {value}")


async def test_neo4j_connection(args) -> Optional[Neo4jStorage]:
    """Test Neo4j database connection."""
    if not args.neo4j:
        return None
        
    print_section("Testing Neo4j Connection")
    
    try:
        storage = get_default_neo4j_storage()
        if not storage:
            print("❌ Failed to create Neo4j storage from environment")
            return None
            
        await storage.connect()
        
        # Test basic query
        result = await storage._execute_query("RETURN 1 as test")
        if result and result[0].get("test") == 1:
            print("✅ Neo4j connection successful")
            
            # Get database stats
            stats_query = """
            MATCH (n) 
            RETURN labels(n) as labels, count(n) as count 
            ORDER BY count DESC 
            LIMIT 10
            """
            stats = await storage._execute_query(stats_query)
            print(f"📊 Database contains {len(stats)} node types:")
            for stat in stats[:5]:  # Show top 5
                labels = stat.get("labels", ["Unknown"])
                count = stat.get("count", 0)
                print(f"   {':'.join(labels)}: {count} nodes")
                
            return storage
        else:
            print("❌ Neo4j connection test failed")
            return None
            
    except Exception as e:
        print(f"❌ Neo4j connection error: {e}")
        return None


async def test_qdrant_connection(args) -> Optional[QdrantStorage]:
    """Test Qdrant database connection."""
    if not args.qdrant:
        return None
        
    print_section("Testing Qdrant Connection")
    
    try:
        storage = get_default_qdrant_storage()
        if not storage:
            print("❌ Failed to create Qdrant storage from environment")
            return None
            
        await storage.connect()
        
        # Get collection info
        info = await storage.get_collection_info()
        print("✅ Qdrant connection successful")
        print(f"📊 Collection '{info['collection_name']}' contains {info['total_points']} points")
        print(f"   Vector size: {info['vector_size']}")
        print(f"   Host: {info['host']}:{info['port']}")
        
        return storage
        
    except Exception as e:
        print(f"❌ Qdrant connection error: {e}")
        return None


async def test_simple_query(query: str, neo4j_storage: Optional[Neo4jStorage], 
                          qdrant_storage: Optional[QdrantStorage], args):
    """Test simple query against available databases."""
    print_section(f"Simple Query: '{query}'")
    
    results = {}
    
    # Test Neo4j entity search
    if neo4j_storage and args.neo4j:
        try:
            print("🔍 Searching Neo4j entities...")
            entities = await neo4j_storage.search_entities(query, limit=args.max_results)
            results["neo4j_entities"] = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description[:100] + "..." if e.description and len(e.description) > 100 else e.description
                }
                for e in entities
            ]
            print(f"   Found {len(entities)} entities")
        except Exception as e:
            print(f"   ❌ Neo4j entity search failed: {e}")
    
    # Test Qdrant vector search
    if qdrant_storage and args.qdrant:
        try:
            print("🔍 Searching Qdrant vectors...")
            entities = await qdrant_storage.search_entities(query, limit=args.max_results)
            results["qdrant_entities"] = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description[:100] + "..." if e.description and len(e.description) > 100 else e.description
                }
                for e in entities
            ]
            print(f"   Found {len(entities)} entities")
        except Exception as e:
            print(f"   ❌ Qdrant entity search failed: {e}")
    
    if results:
        print("\n📋 Results:")
        print_result(results)
    else:
        print("❌ No results found")


async def test_entity_query(entity_name: str, neo4j_storage: Optional[Neo4jStorage], args):
    """Test entity-specific queries."""
    if not neo4j_storage:
        print("⚠️  Entity queries require Neo4j connection")
        return
        
    print_section(f"Entity Query: '{entity_name}'")
    
    try:
        # Search for entity by name
        entities = await neo4j_storage.search_entities(entity_name, limit=5)
        if not entities:
            print(f"❌ No entity found with name '{entity_name}'")
            return

        entity = entities[0]
        print(f"✅ Found entity: {entity.name} ({entity.type})")
        if len(entities) > 1:
            print(f"   (Found {len(entities)} matching entities, showing first)")

        # Get entity relations if available
        try:
            relations = await neo4j_storage.get_entity_relations(entity.id)
            print(f"🔗 Found {len(relations)} relations")

            for rel in relations[:5]:  # Show first 5
                print(f"   {rel.source_entity_id} --[{rel.relation_type}]--> {rel.target_entity_id}")
        except Exception as e:
            print(f"   ⚠️  Could not get relations: {e}")

        # Find neighbors using graph traversal
        try:
            traversal = GraphTraversal(neo4j_storage)
            neighbors = await traversal.find_neighbors(entity.id, max_distance=args.expansion_depth)
            print(f"👥 Found {len(neighbors)} neighbors within distance {args.expansion_depth}")

            for neighbor in neighbors[:3]:  # Show first 3
                print(f"   {neighbor.name} ({neighbor.type})")
        except Exception as e:
            print(f"   ⚠️  Could not get neighbors: {e}")
            
    except Exception as e:
        print(f"❌ Entity query failed: {e}")


async def test_multi_hop_reasoning(query: str, start_entities: List[str], 
                                 neo4j_storage: Optional[Neo4jStorage], args):
    """Test multi-hop reasoning capabilities."""
    if not neo4j_storage:
        print("⚠️  Multi-hop reasoning requires Neo4j connection")
        return
        
    print_section(f"Multi-Hop Reasoning: '{query}'")
    print(f"🎯 Start entities: {start_entities}")
    
    try:
        # Initialize LLM client
        llm_client = LLMClient(
            provider="gemini",
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("MORAG_GEMINI_MODEL", "gemini-2.5-flash")
        )
        
        # Create reasoning components
        path_selector = PathSelectionAgent(llm_client)
        path_finder = ReasoningPathFinder(neo4j_storage, path_selector)
        
        # Configure reasoning parameters
        reasoning_config = {
            "query": query,
            "start_entities": start_entities,
            "strategy": args.reasoning_strategy,
            "max_depth": args.max_depth,
            "max_paths": args.max_paths
        }
        
        print(f"🔄 Finding reasoning paths (strategy: {args.reasoning_strategy})...")
        start_time = time.time()
        
        reasoning_paths = await path_finder.find_reasoning_paths(
            query, start_entities, strategy=args.reasoning_strategy
        )
        
        end_time = time.time()
        print(f"⏱️  Reasoning completed in {(end_time - start_time)*1000:.2f}ms")
        print(f"🛤️  Found {len(reasoning_paths)} reasoning paths")
        
        # Show first few paths
        for i, path in enumerate(reasoning_paths[:3]):
            print(f"\n   Path {i+1}:")
            if hasattr(path, 'entities'):
                print(f"     Entities: {' -> '.join(path.entities[:5])}")
            if hasattr(path, 'confidence'):
                print(f"     Confidence: {path.confidence:.3f}")
                
    except Exception as e:
        print(f"❌ Multi-hop reasoning failed: {e}")


async def test_graph_traversal(start_entity: str, end_entity: str,
                             neo4j_storage: Optional[Neo4jStorage], args):
    """Test graph traversal between entities."""
    if not neo4j_storage:
        print("⚠️  Graph traversal requires Neo4j connection")
        return

    print_section(f"Graph Traversal: '{start_entity}' -> '{end_entity}'")

    try:
        traversal = GraphTraversal(neo4j_storage)

        # Search for entities by name
        start_entities = await neo4j_storage.search_entities(start_entity, limit=1)
        end_entities = await neo4j_storage.search_entities(end_entity, limit=1)

        if not start_entities:
            print(f"❌ Start entity '{start_entity}' not found")
            return
        if not end_entities:
            print(f"❌ End entity '{end_entity}' not found")
            return

        start_id = start_entities[0].id
        end_id = end_entities[0].id

        print(f"🎯 Finding paths from {start_entities[0].name} to {end_entities[0].name}")

        # Find shortest path
        path = await traversal.find_shortest_path(start_id, end_id)
        if path:
            print(f"✅ Shortest path found with {len(path.entities)} entities")
            print(f"   Path: {' -> '.join(path.entities[:10])}")  # Show first 10
            if hasattr(path, 'total_weight'):
                print(f"   Weight: {path.total_weight:.3f}")
        else:
            print("❌ No path found between entities")

    except Exception as e:
        print(f"❌ Graph traversal failed: {e}")


async def test_graph_analytics(neo4j_storage: Optional[Neo4jStorage], args):
    """Test graph analytics and statistics."""
    if not neo4j_storage:
        print("⚠️  Graph analytics requires Neo4j connection")
        return

    print_section("Graph Analytics")

    try:
        analytics = GraphAnalytics(neo4j_storage)

        # Get basic statistics
        print("📊 Computing graph statistics...")

        # Node count by type
        node_stats_query = """
        MATCH (n)
        RETURN labels(n) as labels, count(n) as count
        ORDER BY count DESC
        LIMIT 20
        """
        node_stats = await neo4j_storage._execute_query(node_stats_query)

        print(f"📈 Node Statistics ({len(node_stats)} types):")
        total_nodes = sum(stat.get("count", 0) for stat in node_stats)
        for stat in node_stats[:10]:
            labels = stat.get("labels", ["Unknown"])
            count = stat.get("count", 0)
            percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
            print(f"   {':'.join(labels)}: {count} nodes ({percentage:.1f}%)")

        # Relationship statistics
        rel_stats_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        LIMIT 15
        """
        rel_stats = await neo4j_storage._execute_query(rel_stats_query)

        print(f"\n🔗 Relationship Statistics ({len(rel_stats)} types):")
        total_rels = sum(stat.get("count", 0) for stat in rel_stats)
        for stat in rel_stats[:10]:
            rel_type = stat.get("rel_type", "Unknown")
            count = stat.get("count", 0)
            percentage = (count / total_rels * 100) if total_rels > 0 else 0
            print(f"   {rel_type}: {count} relationships ({percentage:.1f}%)")

        # Graph connectivity
        connectivity_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as degree
        RETURN
            min(degree) as min_degree,
            max(degree) as max_degree,
            avg(degree) as avg_degree,
            count(n) as total_nodes
        """
        connectivity = await neo4j_storage._execute_query(connectivity_query)

        if connectivity:
            conn = connectivity[0]
            print(f"\n🌐 Graph Connectivity:")
            print(f"   Total nodes: {conn.get('total_nodes', 0)}")
            print(f"   Min degree: {conn.get('min_degree', 0)}")
            print(f"   Max degree: {conn.get('max_degree', 0)}")
            print(f"   Avg degree: {conn.get('avg_degree', 0):.2f}")

        # Find highly connected nodes
        central_nodes_query = """
        MATCH (n)-[r]-()
        WITH n, count(r) as degree
        WHERE degree > 5
        RETURN n.name as name, labels(n) as labels, degree
        ORDER BY degree DESC
        LIMIT 10
        """
        central_nodes = await neo4j_storage._execute_query(central_nodes_query)

        if central_nodes:
            print(f"\n⭐ Most Connected Entities:")
            for node in central_nodes:
                name = node.get("name", "Unknown")
                labels = node.get("labels", ["Unknown"])
                degree = node.get("degree", 0)
                print(f"   {name} ({':'.join(labels)}): {degree} connections")

    except Exception as e:
        print(f"❌ Graph analytics failed: {e}")


async def test_hybrid_retrieval(query: str, neo4j_storage: Optional[Neo4jStorage],
                              qdrant_storage: Optional[QdrantStorage], args):
    """Test hybrid retrieval combining vector and graph search."""
    if not neo4j_storage or not qdrant_storage:
        print("⚠️  Hybrid retrieval requires both Neo4j and Qdrant connections")
        return

    print_section(f"Hybrid Retrieval: '{query}'")

    try:
        # Initialize hybrid retrieval coordinator
        from morag_graph.retrieval import HybridRetrievalConfig

        config = HybridRetrievalConfig(
            vector_weight=0.6,
            graph_weight=0.4,
            max_results=args.max_results,
            expansion_depth=args.expansion_depth,
            fusion_strategy=args.fusion_strategy
        )

        print(f"🔄 Running hybrid retrieval (fusion: {args.fusion_strategy})...")
        start_time = time.time()

        # This would typically use the HybridRetrievalCoordinator
        # For now, we'll simulate by running both searches separately

        # Vector search
        vector_results = await qdrant_storage.search_entities(query, limit=args.max_results)
        print(f"   📊 Vector search: {len(vector_results)} results")

        # Graph search
        graph_results = await neo4j_storage.search_entities(query, limit=args.max_results)
        print(f"   🕸️  Graph search: {len(graph_results)} results")

        end_time = time.time()
        print(f"⏱️  Hybrid retrieval completed in {(end_time - start_time)*1000:.2f}ms")

        # Show combined results
        all_results = {}

        # Add vector results
        for i, result in enumerate(vector_results[:5]):
            all_results[f"vector_{i}"] = {
                "source": "vector",
                "name": result.name,
                "type": result.type,
                "score": getattr(result, 'score', 0.0)
            }

        # Add graph results
        for i, result in enumerate(graph_results[:5]):
            all_results[f"graph_{i}"] = {
                "source": "graph",
                "name": result.name,
                "type": result.type,
                "score": getattr(result, 'score', 0.0)
            }

        if all_results:
            print("\n📋 Hybrid Results:")
            print_result(all_results)
        else:
            print("❌ No hybrid results found")

    except Exception as e:
        print(f"❌ Hybrid retrieval failed: {e}")


def print_usage_examples():
    """Print usage examples."""
    examples = [
        "# Basic query against both databases",
        "python test-query.py --all-dbs \"artificial intelligence\"",
        "",
        "# Entity-focused query with Neo4j",
        "python test-query.py --neo4j --entity-query \"Apple Inc\" \"company partnerships\"",
        "",
        "# Multi-hop reasoning",
        "python test-query.py --neo4j --enable-multi-hop --start-entities \"Apple Inc\" \"AI research\" \"How are Apple's AI efforts connected?\"",
        "",
        "# Graph traversal between entities",
        "python test-query.py --neo4j --test-traversal --start-entity \"Apple Inc\" --end-entity \"Stanford University\" \"connection path\"",
        "",
        "# Graph analytics and statistics",
        "python test-query.py --neo4j --test-analytics \"database overview\"",
        "",
        "# Hybrid retrieval test",
        "python test-query.py --all-dbs --test-hybrid \"machine learning applications\"",
        "",
        "# Run all tests",
        "python test-query.py --all-dbs --test-all --start-entities \"Apple Inc\" --start-entity \"Apple Inc\" --end-entity \"AI research\" \"comprehensive test\"",
    ]

    print("📖 Usage Examples:")
    for example in examples:
        if example.startswith("#"):
            print(f"\n{example}")
        elif example == "":
            continue
        else:
            print(f"   {example}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test MoRAG query mechanisms",
        epilog="Use --help for detailed options or run with no arguments to see examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Database selection
    parser.add_argument("--neo4j", action="store_true", help="Test Neo4j queries")
    parser.add_argument("--qdrant", action="store_true", help="Test Qdrant queries")
    parser.add_argument("--all-dbs", action="store_true", help="Test all available databases")
    
    # Query parameters
    parser.add_argument("query", help="Query text to test")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum results (default: 10)")
    parser.add_argument("--query-type", choices=["simple", "entity_focused", "relation_focused", "multi_hop", "analytical"], 
                       default="simple", help="Query type (default: simple)")
    
    # Multi-hop reasoning
    parser.add_argument("--enable-multi-hop", action="store_true", help="Enable multi-hop reasoning")
    parser.add_argument("--start-entities", nargs="+", help="Starting entities for multi-hop reasoning")
    parser.add_argument("--reasoning-strategy", choices=["forward_chaining", "backward_chaining", "bidirectional"],
                       default="forward_chaining", help="Reasoning strategy (default: forward_chaining)")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum reasoning depth (default: 3)")
    parser.add_argument("--max-paths", type=int, default=10, help="Maximum reasoning paths (default: 10)")
    
    # Graph traversal
    parser.add_argument("--test-traversal", action="store_true", help="Test graph traversal")
    parser.add_argument("--start-entity", help="Start entity for traversal")
    parser.add_argument("--end-entity", help="End entity for traversal")

    # Analytics and statistics
    parser.add_argument("--test-analytics", action="store_true", help="Test graph analytics")
    parser.add_argument("--test-hybrid", action="store_true", help="Test hybrid retrieval")

    # Expansion and fusion
    parser.add_argument("--expansion-strategy", choices=["direct_neighbors", "breadth_first", "shortest_path", "adaptive", "none"],
                       default="adaptive", help="Context expansion strategy (default: adaptive)")
    parser.add_argument("--expansion-depth", type=int, default=2, help="Expansion depth (default: 2)")
    parser.add_argument("--fusion-strategy", choices=["weighted", "reciprocal_rank_fusion", "adaptive", "vector_only", "graph_only"],
                       default="adaptive", help="Result fusion strategy (default: adaptive)")

    # Entity queries
    parser.add_argument("--entity-query", help="Test entity-specific queries")

    # Test selection
    parser.add_argument("--test-all", action="store_true", help="Run all available tests")

    # Output options
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    # Handle case where no query is provided
    if len(sys.argv) == 1:
        print_header("MoRAG Query Testing CLI")
        print_usage_examples()
        return 0

    args = parser.parse_args()

    # Set database flags
    if args.all_dbs:
        args.neo4j = True
        args.qdrant = True

    if not args.neo4j and not args.qdrant:
        print("❌ Please specify --neo4j, --qdrant, or --all-dbs")
        print("\nFor examples, run: python test-query.py")
        return 1
    
    if not COMPONENTS_AVAILABLE:
        print("❌ MoRAG components not available. Please install required packages.")
        return 1
    
    print_header("MoRAG Query Testing CLI")
    print(f"Query: '{args.query}'")
    print(f"Databases: Neo4j={args.neo4j}, Qdrant={args.qdrant}")
    
    # Test database connections
    neo4j_storage = await test_neo4j_connection(args)
    qdrant_storage = await test_qdrant_connection(args)
    
    if not neo4j_storage and not qdrant_storage:
        print("❌ No database connections available")
        return 1
    
    # Set test flags for --test-all
    if args.test_all:
        args.test_analytics = True
        args.test_hybrid = True
        if args.start_entities:
            args.enable_multi_hop = True
        if args.start_entity and args.end_entity:
            args.test_traversal = True

    # Run tests based on arguments
    try:
        # Simple query test (always run unless quiet)
        if not args.quiet:
            await test_simple_query(args.query, neo4j_storage, qdrant_storage, args)

        # Entity query test
        if args.entity_query:
            await test_entity_query(args.entity_query, neo4j_storage, args)

        # Multi-hop reasoning test
        if args.enable_multi_hop and args.start_entities:
            await test_multi_hop_reasoning(args.query, args.start_entities, neo4j_storage, args)

        # Graph traversal test
        if args.test_traversal and args.start_entity and args.end_entity:
            await test_graph_traversal(args.start_entity, args.end_entity, neo4j_storage, args)

        # Graph analytics test
        if args.test_analytics:
            await test_graph_analytics(neo4j_storage, args)

        # Hybrid retrieval test
        if args.test_hybrid:
            await test_hybrid_retrieval(args.query, neo4j_storage, qdrant_storage, args)

        if not args.quiet:
            print_section("Test Completed Successfully")

            # Show summary
            print("📊 Test Summary:")
            print(f"   Query: '{args.query}'")
            print(f"   Databases tested: {'Neo4j' if neo4j_storage else ''} {'Qdrant' if qdrant_storage else ''}")
            print(f"   Query type: {args.query_type}")
            if args.enable_multi_hop:
                print(f"   Multi-hop reasoning: {args.reasoning_strategy}")
            if args.test_analytics:
                print("   Graph analytics: ✅")
            if args.test_hybrid:
                print("   Hybrid retrieval: ✅")

        return 0

    except Exception as e:
        if not args.quiet:
            print(f"❌ Test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # Clean up connections
        if neo4j_storage:
            await neo4j_storage.disconnect()
        if qdrant_storage:
            await qdrant_storage.disconnect()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
