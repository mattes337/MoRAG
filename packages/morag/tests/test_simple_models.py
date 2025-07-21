"""Simple test for the models we created."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_legacy_models_direct():
    """Test legacy models by importing them directly."""
    try:
        # Import the specific model files directly
        import importlib.util
        
        # Load legacy models module directly
        legacy_path = src_path / "morag" / "models" / "legacy.py"
        spec = importlib.util.spec_from_file_location("legacy", legacy_path)
        legacy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_module)
        
        # Test LegacyQueryRequest
        LegacyQueryRequest = legacy_module.LegacyQueryRequest
        request = LegacyQueryRequest(
            query="What is AI?",
            max_results=10,
            min_score=0.1
        )
        assert request.query == "What is AI?"
        assert request.max_results == 10
        assert request.min_score == 0.1
        
        # Test with limit field
        request_with_limit = LegacyQueryRequest(
            query="What is AI?",
            limit=15
        )
        assert request_with_limit.max_results == 15
        
        # Test LegacyResult
        LegacyResult = legacy_module.LegacyResult
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
        LegacyQueryResponse = legacy_module.LegacyQueryResponse
        response = LegacyQueryResponse(
            query="What is AI?",
            results=[result],
            total_results=1,
            processing_time_ms=100.0
        )
        assert response.query == "What is AI?"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.total == 1
        
        print("✅ Legacy models direct test passed")
        return True
        
    except Exception as e:
        print(f"❌ Legacy models direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_models_direct():
    """Test enhanced models by importing them directly."""
    try:
        import importlib.util
        
        # Load enhanced query models module directly
        enhanced_path = src_path / "morag" / "models" / "enhanced_query.py"
        spec = importlib.util.spec_from_file_location("enhanced_query", enhanced_path)
        enhanced_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_module)
        
        # Test EnhancedQueryRequest
        EnhancedQueryRequest = enhanced_module.EnhancedQueryRequest
        QueryType = enhanced_module.QueryType
        ExpansionStrategy = enhanced_module.ExpansionStrategy
        FusionStrategy = enhanced_module.FusionStrategy
        
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
        EntityQueryRequest = enhanced_module.EntityQueryRequest
        entity_request = EntityQueryRequest(
            entity_name="machine learning",
            include_relations=True,
            relation_depth=2
        )
        assert entity_request.entity_name == "machine learning"
        assert entity_request.include_relations is True
        assert entity_request.relation_depth == 2
        
        print("✅ Enhanced models direct test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced models direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run simple model tests."""
    print("Running simple model tests...")
    
    tests = [
        test_legacy_models_direct,
        test_enhanced_models_direct
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple model tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
