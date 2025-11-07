"""Integration tests for enhanced API."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from morag.models.enhanced_query import (
    EnhancedQueryRequest, QueryType, ExpansionStrategy, FusionStrategy
)
from morag.utils.response_builder import EnhancedResponseBuilder
from morag.utils.query_validator import QueryValidator


class TestAPIIntegration:
    """Test API integration components."""

    @pytest.mark.asyncio
    async def test_query_validator_integration(self):
        """Test query validator with various inputs."""
        validator = QueryValidator()

        # Valid query
        request = EnhancedQueryRequest(
            query="How does machine learning work?",
            query_type=QueryType.SIMPLE,
            max_results=10
        )

        result = await validator.validate_query_request(request)
        assert result.is_valid

        # Invalid query (too short)
        request_invalid = EnhancedQueryRequest(
            query="AI",
            query_type=QueryType.SIMPLE,
            max_results=10
        )

        result_invalid = await validator.validate_query_request(request_invalid)
        assert not result_invalid.is_valid
        assert "too short" in result_invalid.error_message

    @pytest.mark.asyncio
    async def test_response_builder_integration(self):
        """Test response builder with mock data."""
        builder = EnhancedResponseBuilder()

        # Mock retrieval result
        mock_retrieval_result = [
            {
                'id': 'result_1',
                'content': 'Machine learning is a subset of AI...',
                'score': 0.95,
                'document_id': 'doc_1',
                'source_type': 'vector',
                'metadata': {'topic': 'AI'}
            },
            {
                'id': 'result_2',
                'content': 'Neural networks are computational models...',
                'score': 0.87,
                'document_id': 'doc_2',
                'source_type': 'graph',
                'metadata': {'topic': 'ML'}
            }
        ]

        request = EnhancedQueryRequest(
            query="What is machine learning?",
            query_type=QueryType.ENTITY_FOCUSED,
            max_results=10,
            include_graph_context=True,
            include_reasoning_path=True
        )

        response = await builder.build_response(
            query_id="test-123",
            request=request,
            retrieval_result=mock_retrieval_result,
            processing_time=150.5
        )

        assert response.query_id == "test-123"
        assert response.query == request.query
        assert len(response.results) == 2
        assert response.processing_time_ms == 150.5
        assert response.confidence_score > 0
        assert response.completeness_score > 0

        # Check results
        assert response.results[0].id == 'result_1'
        assert response.results[0].relevance_score == 0.95
        assert response.results[1].source_type == 'graph'

        # Check graph context
        assert response.graph_context is not None

        # Check reasoning steps
        assert response.graph_context.reasoning_steps is not None
        assert len(response.graph_context.reasoning_steps) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self):
        """Test end-to-end query processing simulation."""
        # This would test the complete pipeline in a real scenario
        # For now, we'll test the components working together

        validator = QueryValidator()
        builder = EnhancedResponseBuilder()

        # Step 1: Validate query
        request = EnhancedQueryRequest(
            query="Explain the relationship between deep learning and neural networks",
            query_type=QueryType.MULTI_HOP,
            expansion_strategy=ExpansionStrategy.BREADTH_FIRST,
            expansion_depth=3,
            include_graph_context=True,
            include_reasoning_path=True,
            enable_multi_hop=True
        )

        validation_result = await validator.validate_query_request(request)
        assert validation_result.is_valid

        # Step 2: Simulate retrieval (would be actual hybrid retrieval)
        mock_results = [
            {
                'id': 'result_1',
                'content': 'Deep learning is a subset of machine learning...',
                'score': 0.92,
                'document_id': 'doc_1',
                'source_type': 'hybrid',
                'metadata': {'entities': ['deep_learning', 'machine_learning']},
                'connected_entities': ['deep_learning', 'neural_networks'],
                'relation_context': []
            }
        ]

        # Step 3: Build response
        response = await builder.build_response(
            query_id="integration-test",
            request=request,
            retrieval_result=mock_results,
            processing_time=250.0
        )

        # Verify complete response
        assert response.query_id == "integration-test"
        assert len(response.results) == 1
        assert response.results[0].connected_entities == ['deep_learning', 'neural_networks']
        assert response.fusion_strategy_used == FusionStrategy.ADAPTIVE
        assert response.expansion_strategy_used == ExpansionStrategy.BREADTH_FIRST
        assert response.graph_context.reasoning_steps is not None

    def test_model_serialization(self):
        """Test that all models can be properly serialized."""
        request = EnhancedQueryRequest(
            query="Test query",
            query_type=QueryType.SIMPLE,
            max_results=5
        )

        # Test serialization
        request_dict = request.dict()
        assert request_dict['query'] == "Test query"
        assert request_dict['query_type'] == "simple"
        assert request_dict['max_results'] == 5

        # Test deserialization
        request_restored = EnhancedQueryRequest(**request_dict)
        assert request_restored.query == request.query
        assert request_restored.query_type == request.query_type

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across components."""
        builder = EnhancedResponseBuilder()

        # Test with invalid/empty retrieval result
        request = EnhancedQueryRequest(
            query="Test query",
            query_type=QueryType.SIMPLE,
            max_results=5
        )

        # Should handle empty results gracefully
        response = await builder.build_response(
            query_id="error-test",
            request=request,
            retrieval_result=[],
            processing_time=100.0
        )

        assert response.query_id == "error-test"
        assert len(response.results) == 0
        assert response.confidence_score == 0.0

        # Test with malformed retrieval result
        malformed_result = [{"invalid": "data"}]

        response_malformed = await builder.build_response(
            query_id="malformed-test",
            request=request,
            retrieval_result=malformed_result,
            processing_time=100.0
        )

        assert response_malformed.query_id == "malformed-test"
        # Should handle gracefully and return minimal response
        assert len(response_malformed.results) <= 1


class TestPerformanceIntegration:
    """Test performance aspects of the integration."""

    @pytest.mark.asyncio
    async def test_large_result_set_handling(self):
        """Test handling of large result sets."""
        builder = EnhancedResponseBuilder()

        # Create large mock result set
        large_results = []
        for i in range(100):
            large_results.append({
                'id': f'result_{i}',
                'content': f'Content {i}',
                'score': 0.9 - (i * 0.001),  # Decreasing scores
                'document_id': f'doc_{i}',
                'source_type': 'vector',
                'metadata': {'index': i}
            })

        request = EnhancedQueryRequest(
            query="Large test query",
            query_type=QueryType.SIMPLE,
            max_results=50  # Limit to 50
        )

        response = await builder.build_response(
            query_id="large-test",
            request=request,
            retrieval_result=large_results,
            processing_time=500.0
        )

        # Should respect max_results limit
        assert len(response.results) == 50
        assert response.results[0].relevance_score >= response.results[-1].relevance_score

    @pytest.mark.asyncio
    async def test_complex_query_processing(self):
        """Test processing of complex queries with all features enabled."""
        validator = QueryValidator()
        builder = EnhancedResponseBuilder()

        complex_request = EnhancedQueryRequest(
            query="Analyze the multi-faceted relationship between artificial intelligence, machine learning, deep learning, and neural networks in the context of modern computational paradigms",
            query_type=QueryType.ANALYTICAL,
            max_results=20,
            expansion_strategy=ExpansionStrategy.ADAPTIVE,
            expansion_depth=4,
            fusion_strategy=FusionStrategy.ADAPTIVE,
            entity_types=["CONCEPT", "TECHNOLOGY", "METHOD"],
            relation_types=["SUBSET_OF", "USES", "IMPLEMENTS"],
            include_graph_context=True,
            include_reasoning_path=True,
            enable_multi_hop=True,
            min_relevance_score=0.3
        )

        # Validate complex request
        validation = await validator.validate_query_request(complex_request)
        assert validation.is_valid

        # Check for performance warnings
        assert len(validation.warnings) > 0  # Should have complexity warnings

        # Simulate complex results
        complex_results = [
            {
                'id': f'complex_result_{i}',
                'content': f'Complex content about AI topic {i}',
                'score': 0.8 - (i * 0.05),
                'document_id': f'complex_doc_{i}',
                'source_type': 'hybrid',
                'metadata': {'complexity': 'high'},
                'connected_entities': [f'entity_{i}', f'entity_{i+1}'],
                'relation_context': []
            }
            for i in range(10)
        ]

        response = await builder.build_response(
            query_id="complex-test",
            request=complex_request,
            retrieval_result=complex_results,
            processing_time=1200.0
        )

        assert response.query_id == "complex-test"
        assert len(response.results) <= 20
        assert response.processing_time_ms == 1200.0
        assert response.graph_context.reasoning_steps is not None
        assert len(response.graph_context.reasoning_steps) > 5  # Complex reasoning
