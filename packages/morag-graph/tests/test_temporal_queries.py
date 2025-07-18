"""Unit tests for temporal query functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from morag_graph.graphiti.temporal_service import GraphitiTemporalService, TemporalChange


class TestGraphitiTemporalService:
    """Test temporal query service."""
    
    @pytest.fixture
    def mock_temporal_service(self):
        """Create mock temporal service."""
        with patch('morag_graph.graphiti.temporal_service.create_graphiti_instance') as mock_create:
            mock_create.return_value = Mock()
            service = GraphitiTemporalService()
            service.search_service = Mock()
            service.search_service.search = AsyncMock()
            service.entity_storage = Mock()
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

        # Use time range that includes the test timestamps
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 3)
        changes = await mock_temporal_service.detect_changes('entity_1', start_time, end_time)
        
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

        # Mock datetime.now() to return a date that includes our test timestamps
        with patch('morag_graph.graphiti.temporal_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 1, 12, 0, 0)
            mock_datetime.fromisoformat = datetime.fromisoformat

            evolution = await mock_temporal_service.track_entity_evolution('John Doe', time_window_days=90)

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
