# Step 8: Temporal Query Implementation

**Duration**: 3-4 days  
**Phase**: Advanced Features  
**Prerequisites**: Steps 1-7 completed, chunk-entity relationships working

## Objective

Implement Graphiti's temporal capabilities for point-in-time queries, document versioning, and historical analysis of knowledge graph evolution.

## Deliverables

1. Temporal query service using Graphiti's bi-temporal model
2. Document versioning and change tracking
3. Point-in-time knowledge graph snapshots
4. Historical entity and relationship analysis
5. Temporal search and filtering capabilities

## Implementation

### 1. Create Temporal Query Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/temporal_service.py`

```python
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
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_temporal_queries.py`

```python
"""Unit tests for temporal query functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from morag_graph.graphiti.temporal_service import GraphitiTemporalService, TemporalChange


class TestGraphitiTemporalService:
    """Test temporal query service."""
    
    @pytest.fixture
    def mock_temporal_service(self):
        """Create mock temporal service."""
        service = GraphitiTemporalService()
        service.graphiti = Mock()
        service.search_service = Mock()
        service.search_service.search = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_query_point_in_time(self, mock_temporal_service):
        """Test point-in-time queries."""
        from morag_graph.graphiti.search_service import SearchResult, SearchMetrics
        
        # Mock search results
        mock_result = SearchResult(
            content="Test content",
            score=0.9,
            metadata={
                'adapter_type': 'entity',
                'id': 'entity_1',
                'name': 'Test Entity',
                'type': 'PERSON',
                'confidence': 0.8,
                'conversion_timestamp': '2024-01-01T12:00:00'
            }
        )
        mock_metrics = SearchMetrics(0.1, 1, 1, "temporal")
        
        mock_temporal_service.search_service.search.return_value = ([mock_result], mock_metrics)
        
        target_time = datetime(2024, 1, 2, 12, 0, 0)
        result = await mock_temporal_service.query_point_in_time(target_time)
        
        assert result['target_time'] == target_time.isoformat()
        assert result['counts']['entities'] == 1
        assert result['results']['entities'][0]['name'] == 'Test Entity'
    
    @pytest.mark.asyncio
    async def test_query_time_range(self, mock_temporal_service):
        """Test time range queries."""
        from morag_graph.graphiti.search_service import SearchResult, SearchMetrics
        
        # Mock multiple results across time
        mock_results = [
            SearchResult(
                content="Content 1",
                score=0.9,
                metadata={
                    'adapter_type': 'document',
                    'morag_document_id': 'doc_1',
                    'conversion_timestamp': '2024-01-01T12:00:00'
                }
            ),
            SearchResult(
                content="Content 2",
                score=0.8,
                metadata={
                    'adapter_type': 'document',
                    'morag_document_id': 'doc_2',
                    'conversion_timestamp': '2024-01-02T12:00:00'
                }
            )
        ]
        mock_metrics = SearchMetrics(0.2, 2, 2, "temporal_range")
        
        mock_temporal_service.search_service.search.return_value = (mock_results, mock_metrics)
        
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 3)
        
        result = await mock_temporal_service.query_time_range(start_time, end_time)
        
        assert result['start_time'] == start_time.isoformat()
        assert result['end_time'] == end_time.isoformat()
        assert result['total_results'] == 2
        assert len(result['grouped_results']) > 0
    
    @pytest.mark.asyncio
    async def test_detect_changes(self, mock_temporal_service):
        """Test change detection."""
        from morag_graph.graphiti.search_service import SearchResult, SearchMetrics
        
        # Mock results showing entity evolution
        mock_results = [
            SearchResult(
                content="Entity v1",
                score=0.9,
                metadata={
                    'adapter_type': 'entity',
                    'id': 'entity_1',
                    'confidence': 0.7,
                    'attributes': {'role': 'engineer'},
                    'conversion_timestamp': '2024-01-01T12:00:00'
                }
            ),
            SearchResult(
                content="Entity v2",
                score=0.9,
                metadata={
                    'adapter_type': 'entity',
                    'id': 'entity_1',
                    'confidence': 0.9,
                    'attributes': {'role': 'senior engineer'},
                    'conversion_timestamp': '2024-01-02T12:00:00'
                }
            )
        ]
        mock_metrics = SearchMetrics(0.1, 2, 2, "change_detection")
        
        mock_temporal_service.search_service.search.return_value = (mock_results, mock_metrics)
        
        changes = await mock_temporal_service.detect_changes('entity_1')
        
        assert len(changes) > 0
        # Should detect confidence change and attribute change
        change_types = [change.change_type for change in changes]
        assert "modified" in change_types
    
    @pytest.mark.asyncio
    async def test_track_entity_evolution(self, mock_temporal_service):
        """Test entity evolution tracking."""
        from morag_graph.graphiti.search_service import SearchResult, SearchMetrics
        
        # Mock entity mentions over time
        mock_results = [
            SearchResult(
                content="First mention",
                score=0.9,
                metadata={
                    'adapter_type': 'entity',
                    'name': 'John Doe',
                    'confidence': 0.6,
                    'conversion_timestamp': '2024-01-01T12:00:00'
                }
            ),
            SearchResult(
                content="Second mention",
                score=0.8,
                metadata={
                    'adapter_type': 'entity',
                    'name': 'John Doe',
                    'confidence': 0.8,
                    'conversion_timestamp': '2024-01-15T12:00:00'
                }
            )
        ]
        mock_metrics = SearchMetrics(0.15, 2, 2, "evolution_tracking")
        
        mock_temporal_service.search_service.search.return_value = (mock_results, mock_metrics)
        
        evolution = await mock_temporal_service.track_entity_evolution('John Doe')
        
        assert evolution['entity_name'] == 'John Doe'
        assert evolution['total_mentions'] == 2
        assert len(evolution['timeline']) == 2
        assert len(evolution['confidence_history']) == 2
        assert 'evolution_metrics' in evolution
    
    def test_group_results_by_time(self, mock_temporal_service):
        """Test time-based result grouping."""
        from morag_graph.graphiti.search_service import SearchResult
        
        # Create mock results with different timestamps
        results = [
            SearchResult(
                content="Content 1",
                score=0.9,
                metadata={'conversion_timestamp': '2024-01-01T10:00:00'}
            ),
            SearchResult(
                content="Content 2",
                score=0.8,
                metadata={'conversion_timestamp': '2024-01-01T14:00:00'}
            ),
            SearchResult(
                content="Content 3",
                score=0.7,
                metadata={'conversion_timestamp': '2024-01-02T10:00:00'}
            )
        ]
        
        # Test daily grouping
        grouped = mock_temporal_service._group_results_by_time(results, "day")
        
        assert "2024-01-01" in grouped
        assert "2024-01-02" in grouped
        assert len(grouped["2024-01-01"]) == 2
        assert len(grouped["2024-01-02"]) == 1
    
    def test_compare_states(self, mock_temporal_service):
        """Test state comparison for change detection."""
        old_state = {
            'confidence': 0.7,
            'attributes': {'role': 'engineer', 'department': 'IT'}
        }
        
        new_state = {
            'confidence': 0.9,
            'attributes': {'role': 'senior engineer', 'department': 'IT', 'team': 'AI'}
        }
        
        timestamp = datetime.now()
        changes = mock_temporal_service._compare_states(old_state, new_state, timestamp, 'entity_1')
        
        assert len(changes) > 0
        
        # Should detect confidence change and attribute changes
        change_types = [change.change_type for change in changes]
        assert "modified" in change_types
        
        # Check specific changes
        confidence_changes = [c for c in changes if c.entity_type == "entity"]
        attribute_changes = [c for c in changes if c.entity_type == "attribute"]
        
        assert len(confidence_changes) == 1  # Confidence change
        assert len(attribute_changes) >= 1   # Attribute changes
```

## Validation Checklist

- [ ] Point-in-time queries return accurate historical state
- [ ] Time range queries properly filter by timestamp
- [ ] Change detection identifies entity modifications
- [ ] Entity evolution tracking shows progression over time
- [ ] Temporal snapshots capture knowledge graph state
- [ ] Time-based grouping works for different intervals
- [ ] State comparison accurately detects changes
- [ ] Performance is acceptable for large time ranges
- [ ] Unit tests cover all temporal functionality
- [ ] Integration with existing search capabilities

## Success Criteria

1. **Temporal Accuracy**: Queries return correct historical state
2. **Change Detection**: System identifies meaningful changes over time
3. **Performance**: Temporal queries execute within reasonable time
4. **Usability**: API provides intuitive temporal query interface
5. **Scalability**: Handles large temporal datasets efficiently

## Next Steps

After completing this step:
1. Test temporal queries with real historical data
2. Validate change detection accuracy
3. Optimize performance for large time ranges
4. Proceed to [Step 9: Custom Schema and Entity Types](./step-09-custom-schema.md)

## Performance Considerations

- Index temporal metadata for efficient time-based queries
- Limit result sets for large time ranges
- Cache frequently accessed temporal snapshots
- Optimize change detection algorithms for large datasets
