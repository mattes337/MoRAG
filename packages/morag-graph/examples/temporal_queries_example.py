"""Example demonstrating Graphiti temporal query capabilities."""

import asyncio
from datetime import datetime, timedelta
from morag_graph.graphiti import (
    GraphitiConfig,
    GraphitiTemporalService,
    create_temporal_service
)


async def temporal_queries_example():
    """Demonstrate temporal query functionality."""
    
    # Configure Graphiti (you'll need to set your actual API key)
    config = GraphitiConfig(
        openai_api_key="your-openai-api-key-here",
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="morag_graphiti"
    )
    
    # Create temporal service
    temporal_service = create_temporal_service(config)
    
    print("=== Graphiti Temporal Query Examples ===\n")
    
    # Example 1: Point-in-time query
    print("1. Point-in-time Query")
    print("-" * 30)
    
    target_time = datetime.now() - timedelta(days=7)  # 7 days ago
    
    try:
        point_in_time_results = await temporal_service.query_point_in_time(
            target_time=target_time,
            query="artificial intelligence",
            limit=10
        )
        
        print(f"Knowledge graph state at {target_time.isoformat()}:")
        print(f"- Documents: {point_in_time_results['counts']['documents']}")
        print(f"- Entities: {point_in_time_results['counts']['entities']}")
        print(f"- Relations: {point_in_time_results['counts']['relations']}")
        print(f"- Query time: {point_in_time_results['metrics']['query_time']:.3f}s")
        
        # Show some example results
        if point_in_time_results['results']['entities']:
            print("\nSample entities from that time:")
            for entity in point_in_time_results['results']['entities'][:3]:
                print(f"  - {entity['name']} ({entity['type']}) - confidence: {entity['confidence']}")
    
    except Exception as e:
        print(f"Point-in-time query failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Time range query
    print("2. Time Range Query")
    print("-" * 30)
    
    start_time = datetime.now() - timedelta(days=30)  # 30 days ago
    end_time = datetime.now() - timedelta(days=1)     # 1 day ago
    
    try:
        time_range_results = await temporal_service.query_time_range(
            start_time=start_time,
            end_time=end_time,
            query="machine learning",
            group_by_interval="week"
        )
        
        print(f"Changes from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}:")
        print(f"- Total results: {time_range_results['total_results']}")
        print(f"- Grouped by: {time_range_results['group_by_interval']}")
        
        # Show grouped results
        if time_range_results['grouped_results']:
            print("\nActivity by week:")
            for week, items in list(time_range_results['grouped_results'].items())[:5]:
                print(f"  - {week}: {len(items)} items")
    
    except Exception as e:
        print(f"Time range query failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Entity evolution tracking
    print("3. Entity Evolution Tracking")
    print("-" * 30)
    
    try:
        evolution = await temporal_service.track_entity_evolution(
            entity_name="OpenAI",
            time_window_days=90
        )
        
        print(f"Evolution of entity '{evolution['entity_name']}':")
        print(f"- Total mentions: {evolution['total_mentions']}")
        print(f"- Time span: {evolution['evolution_metrics'].get('time_span_days', 0)} days")
        print(f"- Confidence trend: {evolution['evolution_metrics'].get('confidence_trend', 'unknown')}")
        print(f"- Average confidence: {evolution['evolution_metrics'].get('avg_confidence', 0):.2f}")
        
        # Show timeline highlights
        if evolution['timeline']:
            print("\nTimeline highlights:")
            for i, entry in enumerate(evolution['timeline'][:3]):
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                print(f"  {i+1}. {timestamp.strftime('%Y-%m-%d %H:%M')} - confidence: {entry['confidence']:.2f}")
    
    except Exception as e:
        print(f"Entity evolution tracking failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Change detection
    print("4. Change Detection")
    print("-" * 30)
    
    try:
        # This would typically use a real entity ID from your data
        entity_id = "example_entity_123"
        
        changes = await temporal_service.detect_changes(
            entity_id=entity_id,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        print(f"Changes detected for entity {entity_id}:")
        print(f"- Total changes: {len(changes)}")
        
        if changes:
            print("\nRecent changes:")
            for change in changes[:5]:
                print(f"  - {change.change_type} {change.entity_type} at {change.timestamp.strftime('%Y-%m-%d %H:%M')}")
                print(f"    Confidence: {change.confidence:.2f}")
        else:
            print("  No changes detected (entity may not exist in test data)")
    
    except Exception as e:
        print(f"Change detection failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Temporal snapshot
    print("5. Temporal Snapshot")
    print("-" * 30)
    
    try:
        snapshot_time = datetime.now() - timedelta(days=14)  # 2 weeks ago
        
        snapshot = await temporal_service.create_temporal_snapshot(
            timestamp=snapshot_time,
            include_metadata=True
        )
        
        print(f"Knowledge graph snapshot at {snapshot.timestamp.strftime('%Y-%m-%d %H:%M')}:")
        print(f"- Documents: {snapshot.document_count}")
        print(f"- Entities: {snapshot.entity_count}")
        print(f"- Relations: {snapshot.relation_count}")
        print(f"- Snapshot created: {snapshot.metadata.get('snapshot_created', 'unknown')}")
    
    except Exception as e:
        print(f"Temporal snapshot failed: {e}")
    
    print("\n" + "="*50)
    print("Temporal query examples completed!")


if __name__ == "__main__":
    print("Graphiti Temporal Queries Example")
    print("=" * 50)
    print("Note: Make sure to set your OpenAI API key and Neo4j connection details")
    print("=" * 50)
    
    # Run the example
    asyncio.run(temporal_queries_example())
