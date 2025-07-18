"""Temporal query service leveraging Graphiti's bi-temporal capabilities."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .config import create_graphiti_instance, GraphitiConfig
from .search_service import GraphitiSearchService
from .entity_storage import GraphitiEntityStorage

logger = logging.getLogger(__name__)


class TemporalQueryType(Enum):
    """Types of temporal queries."""
    POINT_IN_TIME = "point_in_time"
    TIME_RANGE = "time_range"
    CHANGE_DETECTION = "change_detection"
    EVOLUTION_TRACKING = "evolution_tracking"


@dataclass
class TemporalSnapshot:
    """Snapshot of knowledge graph at a specific time."""
    timestamp: datetime
    document_count: int
    entity_count: int
    relation_count: int
    episode_ids: List[str]
    metadata: Dict[str, Any]


@dataclass
class TemporalChange:
    """Represents a change in the knowledge graph over time."""
    change_type: str  # "added", "modified", "removed"
    entity_type: str  # "document", "entity", "relation"
    entity_id: str
    timestamp: datetime
    old_value: Optional[Dict[str, Any]]
    new_value: Optional[Dict[str, Any]]
    confidence: float


class GraphitiTemporalService:
    """Service for temporal queries and analysis using Graphiti."""
    
    def __init__(self, config: Optional[GraphitiConfig] = None):
        self.config = config
        self.graphiti = create_graphiti_instance(config)
        self.search_service = GraphitiSearchService(config)
        self.entity_storage = GraphitiEntityStorage(config)
    
    async def query_point_in_time(
        self,
        target_time: datetime,
        query: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Query knowledge graph state at a specific point in time.
        
        Args:
            target_time: Target timestamp for the query
            query: Optional search query
            entity_types: Optional entity type filters
            limit: Maximum results
            
        Returns:
            Point-in-time query results
        """
        # Build temporal search query
        search_query_parts = []
        
        # Add timestamp filter (episodes created before target time)
        target_iso = target_time.isoformat()
        search_query_parts.append(f"conversion_timestamp:<{target_iso}")
        
        # Add user query if provided
        if query:
            search_query_parts.append(query)
        
        # Add entity type filters
        if entity_types:
            type_filter = " OR ".join([f"type:{et}" for et in entity_types])
            search_query_parts.append(f"({type_filter})")
        
        # Only include MoRAG episodes
        search_query_parts.append("morag_integration:true")
        
        combined_query = " AND ".join(search_query_parts)
        
        # Execute search
        results, metrics = await self.search_service.search(
            query=combined_query,
            limit=limit
        )
        
        # Organize results by type
        documents = []
        entities = []
        relations = []
        
        for result in results:
            metadata = result.metadata or {}
            adapter_type = metadata.get('adapter_type', 'unknown')
            
            if adapter_type == 'document':
                documents.append({
                    'id': metadata.get('morag_document_id'),
                    'name': metadata.get('file_name'),
                    'timestamp': metadata.get('conversion_timestamp'),
                    'content_preview': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    'score': result.score
                })
            elif adapter_type == 'entity':
                entities.append({
                    'id': metadata.get('id'),
                    'name': metadata.get('name'),
                    'type': metadata.get('type'),
                    'confidence': metadata.get('confidence'),
                    'timestamp': metadata.get('conversion_timestamp'),
                    'score': result.score
                })
            elif adapter_type == 'relation':
                relations.append({
                    'id': metadata.get('id'),
                    'source_entity_id': metadata.get('source_entity_id'),
                    'target_entity_id': metadata.get('target_entity_id'),
                    'relation_type': metadata.get('relation_type'),
                    'timestamp': metadata.get('conversion_timestamp'),
                    'score': result.score
                })
        
        return {
            'target_time': target_time.isoformat(),
            'query': query,
            'results': {
                'documents': documents,
                'entities': entities,
                'relations': relations
            },
            'counts': {
                'total_results': len(results),
                'documents': len(documents),
                'entities': len(entities),
                'relations': len(relations)
            },
            'metrics': {
                'query_time': metrics.query_time,
                'search_method': metrics.search_method
            }
        }
    
    async def query_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        query: Optional[str] = None,
        group_by_interval: Optional[str] = "day"  # "hour", "day", "week", "month"
    ) -> Dict[str, Any]:
        """Query knowledge graph changes over a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            query: Optional search query
            group_by_interval: Grouping interval for results
            
        Returns:
            Time range query results
        """
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()
        
        # Build temporal range query
        search_query_parts = [
            f"conversion_timestamp:>={start_iso}",
            f"conversion_timestamp:<={end_iso}",
            "morag_integration:true"
        ]
        
        if query:
            search_query_parts.append(query)
        
        combined_query = " AND ".join(search_query_parts)
        
        # Execute search with larger limit for time range
        results, metrics = await self.search_service.search(
            query=combined_query,
            limit=500
        )
        
        # Group results by time interval
        grouped_results = self._group_results_by_time(results, group_by_interval)
        
        return {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'query': query,
            'group_by_interval': group_by_interval,
            'grouped_results': grouped_results,
            'total_results': len(results),
            'metrics': {
                'query_time': metrics.query_time,
                'search_method': metrics.search_method
            }
        }
    
    def _group_results_by_time(
        self, 
        results: List[Any], 
        interval: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group search results by time interval."""
        grouped = {}
        
        for result in results:
            metadata = result.metadata or {}
            timestamp_str = metadata.get('conversion_timestamp')
            
            if not timestamp_str:
                continue
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                # Create interval key based on grouping
                if interval == "hour":
                    key = timestamp.strftime("%Y-%m-%d %H:00")
                elif interval == "day":
                    key = timestamp.strftime("%Y-%m-%d")
                elif interval == "week":
                    # Get Monday of the week
                    monday = timestamp - timedelta(days=timestamp.weekday())
                    key = monday.strftime("%Y-%m-%d (Week)")
                elif interval == "month":
                    key = timestamp.strftime("%Y-%m")
                else:
                    key = timestamp.strftime("%Y-%m-%d")
                
                if key not in grouped:
                    grouped[key] = []
                
                grouped[key].append({
                    'content': result.content[:100] + "..." if len(result.content) > 100 else result.content,
                    'score': result.score,
                    'timestamp': timestamp_str,
                    'adapter_type': metadata.get('adapter_type'),
                    'entity_id': metadata.get('id') or metadata.get('morag_document_id')
                })
                
            except ValueError:
                continue
        
        return grouped

    async def detect_changes(
        self,
        entity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TemporalChange]:
        """Detect changes to a specific entity over time.

        Args:
            entity_id: Entity ID to track
            start_time: Optional start time (defaults to 30 days ago)
            end_time: Optional end time (defaults to now)

        Returns:
            List of detected changes
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        if not end_time:
            end_time = datetime.now()

        # Search for all episodes related to this entity
        search_query = f"id:{entity_id} OR source_entity_id:{entity_id} OR target_entity_id:{entity_id}"
        results, _ = await self.search_service.search(query=search_query, limit=100)

        # Sort results by timestamp
        timestamped_results = []
        for result in results:
            metadata = result.metadata or {}
            timestamp_str = metadata.get('conversion_timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if start_time <= timestamp <= end_time:
                        timestamped_results.append((timestamp, result))
                except ValueError:
                    continue

        timestamped_results.sort(key=lambda x: x[0])

        # Detect changes between consecutive versions
        changes = []
        previous_state = None

        for timestamp, result in timestamped_results:
            current_state = result.metadata

            if previous_state:
                detected_changes = self._compare_states(
                    previous_state, current_state, timestamp, entity_id
                )
                changes.extend(detected_changes)

            previous_state = current_state

        return changes

    async def track_entity_evolution(
        self,
        entity_name: str,
        time_window_days: int = 90
    ) -> Dict[str, Any]:
        """Track how an entity has evolved over time.

        Args:
            entity_name: Name of entity to track
            time_window_days: Number of days to look back

        Returns:
            Entity evolution timeline
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_window_days)

        # Search for all mentions of this entity
        search_query = f"name:\"{entity_name}\" adapter_type:entity"
        results, _ = await self.search_service.search(query=search_query, limit=200)

        # Create timeline of entity states
        timeline = []
        confidence_history = []
        attribute_changes = []

        for result in results:
            metadata = result.metadata or {}
            timestamp_str = metadata.get('conversion_timestamp')

            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if start_time <= timestamp <= end_time:
                        timeline.append({
                            'timestamp': timestamp.isoformat(),
                            'confidence': metadata.get('confidence', 0.0),
                            'type': metadata.get('type'),
                            'attributes': metadata.get('attributes', {}),
                            'source_doc_id': metadata.get('source_doc_id'),
                            'episode_id': getattr(result, 'episode_id', None)
                        })

                        confidence_history.append({
                            'timestamp': timestamp.isoformat(),
                            'confidence': metadata.get('confidence', 0.0)
                        })
                except ValueError:
                    continue

        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        confidence_history.sort(key=lambda x: x['timestamp'])

        # Calculate evolution metrics
        evolution_metrics = self._calculate_evolution_metrics(timeline, confidence_history)

        return {
            'entity_name': entity_name,
            'time_window_days': time_window_days,
            'timeline': timeline,
            'confidence_history': confidence_history,
            'evolution_metrics': evolution_metrics,
            'total_mentions': len(timeline)
        }

    async def create_temporal_snapshot(
        self,
        timestamp: Optional[datetime] = None,
        include_metadata: bool = True
    ) -> TemporalSnapshot:
        """Create a snapshot of the knowledge graph at a specific time.

        Args:
            timestamp: Target timestamp (defaults to now)
            include_metadata: Whether to include detailed metadata

        Returns:
            Temporal snapshot
        """
        if not timestamp:
            timestamp = datetime.now()

        # Query all episodes up to the target timestamp
        point_in_time_results = await self.query_point_in_time(
            target_time=timestamp,
            limit=1000
        )

        # Count different types
        counts = point_in_time_results['counts']

        # Collect episode IDs if requested
        episode_ids = []
        metadata = {}

        if include_metadata:
            # This would require additional queries to get episode IDs
            # For now, we'll include summary metadata
            metadata = {
                'query_time': point_in_time_results['metrics']['query_time'],
                'search_method': point_in_time_results['metrics']['search_method'],
                'snapshot_created': datetime.now().isoformat()
            }

        return TemporalSnapshot(
            timestamp=timestamp,
            document_count=counts['documents'],
            entity_count=counts['entities'],
            relation_count=counts['relations'],
            episode_ids=episode_ids,
            metadata=metadata
        )

    def _compare_states(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        timestamp: datetime,
        entity_id: str
    ) -> List[TemporalChange]:
        """Compare two entity states to detect changes."""
        changes = []

        # Compare confidence
        old_confidence = old_state.get('confidence', 0.0)
        new_confidence = new_state.get('confidence', 0.0)

        if abs(old_confidence - new_confidence) > 0.1:  # Significant confidence change
            changes.append(TemporalChange(
                change_type="modified",
                entity_type="entity",
                entity_id=entity_id,
                timestamp=timestamp,
                old_value={"confidence": old_confidence},
                new_value={"confidence": new_confidence},
                confidence=0.8
            ))

        # Compare attributes
        old_attrs = old_state.get('attributes', {})
        new_attrs = new_state.get('attributes', {})

        for key in set(old_attrs.keys()) | set(new_attrs.keys()):
            old_val = old_attrs.get(key)
            new_val = new_attrs.get(key)

            if old_val != new_val:
                changes.append(TemporalChange(
                    change_type="modified" if old_val and new_val else ("added" if new_val else "removed"),
                    entity_type="attribute",
                    entity_id=f"{entity_id}.{key}",
                    timestamp=timestamp,
                    old_value={key: old_val} if old_val else None,
                    new_value={key: new_val} if new_val else None,
                    confidence=0.9
                ))

        return changes

    def _calculate_evolution_metrics(
        self,
        timeline: List[Dict[str, Any]],
        confidence_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics for entity evolution."""
        if not timeline:
            return {}

        # Calculate confidence trend
        confidences = [item['confidence'] for item in confidence_history]
        confidence_trend = "stable"

        if len(confidences) > 1:
            first_half_avg = sum(confidences[:len(confidences)//2]) / (len(confidences)//2)
            second_half_avg = sum(confidences[len(confidences)//2:]) / (len(confidences) - len(confidences)//2)

            if second_half_avg > first_half_avg + 0.1:
                confidence_trend = "increasing"
            elif second_half_avg < first_half_avg - 0.1:
                confidence_trend = "decreasing"

        # Count attribute changes
        all_attributes = set()
        for item in timeline:
            all_attributes.update(item.get('attributes', {}).keys())

        return {
            'confidence_trend': confidence_trend,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'unique_attributes': len(all_attributes),
            'mention_frequency': len(timeline),
            'time_span_days': (
                datetime.fromisoformat(timeline[-1]['timestamp'].replace('Z', '+00:00')) -
                datetime.fromisoformat(timeline[0]['timestamp'].replace('Z', '+00:00'))
            ).days if len(timeline) > 1 else 0
        }


# Convenience function for creating temporal service
def create_temporal_service(config: Optional[GraphitiConfig] = None) -> GraphitiTemporalService:
    """Create a Graphiti temporal service.

    Args:
        config: Optional Graphiti configuration

    Returns:
        GraphitiTemporalService instance
    """
    return GraphitiTemporalService(config)
