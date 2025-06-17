#!/usr/bin/env python3
"""Test script for graph operations.

This script demonstrates the CRUD, traversal, and analytics capabilities
of the morag-graph package.
"""

import asyncio
import json
import logging
from pathlib import Path

# Add the package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

from morag_graph import (
    Neo4jStorage, 
    GraphCRUD, 
    GraphTraversal, 
    GraphAnalytics,
    Entity,
    Relation
)
from morag_graph.storage.neo4j_storage import Neo4jConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_graph_operations():
    """Test comprehensive graph operations."""
    logger.info("Starting graph operations test")
    
    # Initialize storage
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="morag_password"
    )
    storage = Neo4jStorage(config)
    
    try:
        # Connect to Neo4j
        await storage.connect()
        logger.info("Connected to Neo4j")
        
        # Initialize operation classes
        crud = GraphCRUD(storage)
        traversal = GraphTraversal(storage)
        analytics = GraphAnalytics(storage)
        
        # Test 1: CRUD Operations
        logger.info("\n=== Testing CRUD Operations ===")
        
        # Get graph summary
        summary = await crud.get_graph_summary()
        logger.info(f"Graph summary: {summary}")
        
        # Find entities by type
        persons = await crud.find_entities_by_type("PERSON", limit=5)
        logger.info(f"Found {len(persons)} PERSON entities")
        for person in persons:
            logger.info(f"  - {person.name} (ID: {person.id})")
        
        # Find relations by type
        relations = await crud.find_relations_by_type("RELATED_TO", limit=5)
        logger.info(f"Found {len(relations)} RELATED_TO relations")
        
        # Test 2: Traversal Operations
        logger.info("\n=== Testing Traversal Operations ===")
        
        if persons:
            test_person = persons[0]
            logger.info(f"Testing traversal from: {test_person.name}")
            
            # Find neighbors
            neighbors = await traversal.find_neighbors(test_person.id, max_depth=1)
            logger.info(f"Found {len(neighbors)} direct neighbors")
            
            # Find entities within 2 hops
            nearby_entities = await traversal.find_neighbors(test_person.id, max_depth=2)
            logger.info(f"Found {len(nearby_entities)} entities within 2 hops")
            
            # Test shortest path if we have multiple persons
            if len(persons) > 1:
                target_person = persons[1]
                paths = await traversal.find_shortest_path(
                    test_person.id, 
                    target_person.id
                )
                if paths:
                    logger.info(f"Shortest path from {test_person.name} to {target_person.name}:")
                    for i, path in enumerate(paths[:3]):  # Show first 3 paths
                        path_names = []
                        for entity_id in path.entity_ids:
                            entity = await crud.get_entity_by_id(entity_id)
                            if entity:
                                path_names.append(entity.name)
                        logger.info(f"  Path {i+1}: {' -> '.join(path_names)}")
                else:
                    logger.info(f"No path found between {test_person.name} and {target_person.name}")
        
        # Find connected components
        components = await traversal.find_connected_components()
        logger.info(f"Found {len(components)} connected components")
        if components:
            logger.info(f"Largest component has {len(components[0])} entities")
        
        # Test 3: Analytics Operations
        logger.info("\n=== Testing Analytics Operations ===")
        
        # Calculate degree centrality
        degree_centrality = await analytics.calculate_degree_centrality(top_k=10)
        logger.info("Top 10 entities by degree centrality:")
        for i, result in enumerate(degree_centrality[:10]):
            entity = result["entity"]
            degree = result["degree"]
            logger.info(f"  {i+1}. {entity.name} (degree: {degree})")
        
        # Calculate betweenness centrality (for smaller graphs)
        if summary.get("total_entities", 0) < 50:  # Only for smaller graphs
            logger.info("\nCalculating betweenness centrality...")
            betweenness_centrality = await analytics.calculate_betweenness_centrality(top_k=5)
            logger.info("Top 5 entities by betweenness centrality:")
            for i, result in enumerate(betweenness_centrality[:5]):
                entity = result["entity"]
                betweenness = result["betweenness"]
                logger.info(f"  {i+1}. {entity.name} (betweenness: {betweenness:.4f})")
        
        # Calculate closeness centrality (for smaller graphs)
        if summary.get("total_entities", 0) < 30:  # Only for very small graphs
            logger.info("\nCalculating closeness centrality...")
            closeness_centrality = await analytics.calculate_closeness_centrality(top_k=5)
            logger.info("Top 5 entities by closeness centrality:")
            for i, result in enumerate(closeness_centrality[:5]):
                entity = result["entity"]
                closeness = result["closeness"]
                reachable = result["reachable_entities"]
                logger.info(f"  {i+1}. {entity.name} (closeness: {closeness:.4f}, reachable: {reachable})")
        
        # Analyze entity types
        entity_analysis = await analytics.analyze_entity_types()
        logger.info("\nEntity type analysis:")
        logger.info(f"  Total entities: {entity_analysis['total_entities']}")
        logger.info(f"  Unique types: {entity_analysis['unique_types']}")
        logger.info(f"  Most common type: {entity_analysis['most_common_type']}")
        logger.info("  Type distribution:")
        for entity_type, stats in list(entity_analysis['type_distribution'].items())[:5]:
            logger.info(f"    {entity_type}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        # Analyze relation types
        relation_analysis = await analytics.analyze_relation_types()
        logger.info("\nRelation type analysis:")
        logger.info(f"  Total relations: {relation_analysis['total_relations']}")
        logger.info(f"  Unique types: {relation_analysis['unique_types']}")
        logger.info(f"  Most common type: {relation_analysis['most_common_type']}")
        logger.info("  Type distribution:")
        for relation_type, stats in list(relation_analysis['type_distribution'].items())[:5]:
            logger.info(f"    {relation_type}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        # Calculate graph density
        density = await analytics.calculate_graph_density()
        logger.info(f"\nGraph density: {density:.4f}")
        
        # Find entity clusters
        clusters = await analytics.find_entity_clusters(min_cluster_size=3)
        logger.info(f"\nFound {len(clusters)} entity clusters (min size 3):")
        for i, cluster in enumerate(clusters[:5]):  # Show first 5 clusters
            logger.info(f"  Cluster {i+1}: {cluster['size']} entities, "
                       f"density: {cluster['density']:.3f}, "
                       f"dominant type: {cluster['dominant_type']}")
        
        # Get comprehensive statistics
        stats = await analytics.get_graph_statistics()
        logger.info("\nComprehensive graph statistics:")
        logger.info(f"  Basic metrics: {stats['basic_metrics']}")
        logger.info(f"  Degree stats: avg={stats['degree_statistics']['average_degree']:.2f}, "
                   f"max={stats['degree_statistics']['max_degree']}, "
                   f"min={stats['degree_statistics']['min_degree']}")
        logger.info(f"  Connectivity: {stats['connectivity']['num_components']} components, "
                   f"largest={stats['connectivity']['largest_component_size']}, "
                   f"connected={stats['connectivity']['is_connected']}")
        
        # Test 4: Advanced Queries
        logger.info("\n=== Testing Advanced Queries ===")
        
        # Find entities with high degree
        high_degree_entities = await crud.find_entities_by_property(
            "degree", 
            min_value=5,  # Entities with at least 5 connections
            limit=5
        )
        logger.info(f"Found {len(high_degree_entities)} entities with high degree")
        
        # Search for entities by name pattern
        if persons:
            # Try to find entities with similar names
            search_term = persons[0].name.split()[0] if ' ' in persons[0].name else persons[0].name[:3]
            similar_entities = await crud.search_entities_by_name(search_term, limit=5)
            logger.info(f"Found {len(similar_entities)} entities matching '{search_term}'")
        
        logger.info("\n=== Graph Operations Test Completed ===")
        
    except Exception as e:
        logger.error(f"Error during graph operations test: {e}")
        raise
    finally:
        # Disconnect from Neo4j
        await storage.disconnect()
        logger.info("Disconnected from Neo4j")


if __name__ == "__main__":
    asyncio.run(test_graph_operations())