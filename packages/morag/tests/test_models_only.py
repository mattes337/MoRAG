"""Test models directly without importing the full server."""

import sys
import os
from pathlib import Path

# Add src to path to import models directly
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_legacy_models():
    """Test legacy models can be imported and work correctly."""
    try:
        # Import models directly without going through __init__.py
        from morag.models.legacy import LegacyQueryRequest, LegacyQueryResponse, LegacyResult
        
        # Test LegacyQueryRequest
        request = LegacyQueryRequest(
            query="What is AI?",
            max_results=10,
            min_score=0.1
        )
        assert request.query == "What is AI?"
        assert request.max_results == 10
        assert request.min_score == 0.1
        
        # Test with limit field (legacy compatibility)
        request_with_limit = LegacyQueryRequest(
            query="What is AI?",
            limit=15
        )
        assert request_with_limit.max_results == 15
        
        # Test LegacyResult
        result = LegacyResult(
            id="result_1",
            content="Test content",
            score=0.9,
            metadata={"test": "data"}
        )
        assert result.id == "result_1"
        assert result.content == "Test content"
        assert result.score == 0.9
        
        # Test LegacyQueryResponse
        response = LegacyQueryResponse(
            query="What is AI?",
            results=[result],
            total_results=1,
            processing_time_ms=100.0
        )
        assert response.query == "What is AI?"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.total == 1  # Legacy field should be set
        
        print("✅ Legacy models test passed")
        return True
        
    except Exception as e:
        print(f"❌ Legacy models test failed: {e}")
        return False


def test_enhanced_models():
    """Test enhanced query models can be imported and work correctly."""
    try:
        from morag.models.enhanced_query import (
            EnhancedQueryRequest, QueryType, ExpansionStrategy, FusionStrategy,
            EntityQueryRequest, GraphTraversalRequest
        )
        
        # Test EnhancedQueryRequest
        request = EnhancedQueryRequest(
            query="What is machine learning?",
            query_type=QueryType.ENTITY_FOCUSED,
            max_results=5,
            expansion_strategy=ExpansionStrategy.ADAPTIVE,
            fusion_strategy=FusionStrategy.ADAPTIVE
        )
        assert request.query == "What is machine learning?"
        assert request.query_type == QueryType.ENTITY_FOCUSED
        assert request.max_results == 5
        assert request.expansion_strategy == ExpansionStrategy.ADAPTIVE
        
        # Test EntityQueryRequest
        entity_request = EntityQueryRequest(
            entity_name="machine learning",
            include_relations=True,
            relation_depth=2
        )
        assert entity_request.entity_name == "machine learning"
        assert entity_request.include_relations is True
        assert entity_request.relation_depth == 2
        
        # Test GraphTraversalRequest
        traversal_request = GraphTraversalRequest(
            start_entity="entity_1",
            end_entity="entity_2",
            traversal_type="shortest_path",
            max_depth=3
        )
        assert traversal_request.start_entity == "entity_1"
        assert traversal_request.end_entity == "entity_2"
        assert traversal_request.traversal_type == "shortest_path"
        
        print("✅ Enhanced models test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced models test failed: {e}")
        return False


def test_model_validation():
    """Test model validation works correctly."""
    try:
        from morag.models.legacy import LegacyQueryRequest
        from morag.models.enhanced_query import EnhancedQueryRequest
        from pydantic import ValidationError
        
        # Test validation errors
        try:
            # Should fail - missing required query field
            LegacyQueryRequest()
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        try:
            # Should fail - invalid max_results
            LegacyQueryRequest(query="test", max_results=0)
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        try:
            # Should fail - invalid min_score
            LegacyQueryRequest(query="test", min_score=1.5)
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        print("✅ Model validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model validation test failed: {e}")
        return False


def main():
    """Run all model tests."""
    print("Running model tests...")
    
    tests = [
        test_legacy_models,
        test_enhanced_models,
        test_model_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All model tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
