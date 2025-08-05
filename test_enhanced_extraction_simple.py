"""Simple test for enhanced extraction components."""

import asyncio
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'morag-graph', 'src'))

async def test_enhanced_extraction_imports():
    """Test that enhanced extraction components can be imported."""
    try:
        from morag_graph.extraction.enhanced_entity_extractor import (
            EnhancedEntityExtractor,
            ConfidenceEntity,
            EntityConfidenceModel
        )
        print("✅ Enhanced entity extractor imports successful")
        
        from morag_graph.extraction.enhanced_relation_extractor import (
            EnhancedRelationExtractor,
            RelationValidator
        )
        print("✅ Enhanced relation extractor imports successful")
        
        from morag_graph.extraction.systematic_deduplicator import (
            SystematicDeduplicator,
            EntitySimilarityCalculator
        )
        print("✅ Systematic deduplicator imports successful")
        
        from morag_graph.extraction.unified_extraction_pipeline import (
            UnifiedExtractionPipeline,
            PipelineConfig
        )
        print("✅ Unified extraction pipeline imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_entity_similarity_calculator():
    """Test the EntitySimilarityCalculator."""
    try:
        from morag_graph.extraction.systematic_deduplicator import EntitySimilarityCalculator
        from morag_graph.models import Entity
        
        calculator = EntitySimilarityCalculator()
        
        # Test exact match
        entity1 = Entity(name="Python", type="TECHNOLOGY", confidence=0.9)
        entity2 = Entity(name="Python", type="TECHNOLOGY", confidence=0.8)
        
        similarity = calculator.calculate_similarity(entity1, entity2)
        print(f"✅ Exact match similarity: {similarity:.2f}")
        assert similarity > 0.9, f"Expected high similarity, got {similarity}"
        
        # Test partial match
        entity3 = Entity(name="Dr. John Smith", type="PERSON", confidence=0.9)
        entity4 = Entity(name="John Smith", type="PERSON", confidence=0.8)
        
        similarity2 = calculator.calculate_similarity(entity3, entity4)
        print(f"✅ Partial match similarity: {similarity2:.2f}")
        assert 0.5 < similarity2 < 1.0, f"Expected medium similarity, got {similarity2}"
        
        # Test different entities
        entity5 = Entity(name="Python", type="TECHNOLOGY", confidence=0.9)
        entity6 = Entity(name="Java", type="TECHNOLOGY", confidence=0.8)
        
        similarity3 = calculator.calculate_similarity(entity5, entity6)
        print(f"✅ Different entities similarity: {similarity3:.2f}")
        assert similarity3 < 0.5, f"Expected low similarity, got {similarity3}"
        
        return True
        
    except Exception as e:
        print(f"❌ EntitySimilarityCalculator test failed: {e}")
        return False


async def test_pipeline_config():
    """Test the PipelineConfig."""
    try:
        from morag_graph.extraction.unified_extraction_pipeline import PipelineConfig
        
        # Test default config
        config = PipelineConfig()
        config_dict = config.to_dict()
        
        print("✅ Default pipeline config created")
        print(f"   Entity max rounds: {config.entity_max_rounds}")
        print(f"   Relation max rounds: {config.relation_max_rounds}")
        print(f"   Deduplication enabled: {config.enable_deduplication}")
        
        # Test custom config
        custom_config = PipelineConfig(
            entity_max_rounds=5,
            relation_max_rounds=3,
            enable_deduplication=False,
            domain="medical"
        )
        
        assert custom_config.entity_max_rounds == 5
        assert custom_config.relation_max_rounds == 3
        assert custom_config.enable_deduplication == False
        assert custom_config.domain == "medical"
        
        print("✅ Custom pipeline config created and validated")
        
        return True
        
    except Exception as e:
        print(f"❌ PipelineConfig test failed: {e}")
        return False


async def test_confidence_model():
    """Test the EntityConfidenceModel."""
    try:
        from morag_graph.extraction.enhanced_entity_extractor import EntityConfidenceModel
        from morag_graph.models import Entity
        
        model = EntityConfidenceModel()
        
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            confidence=0.8,
            source_doc_id="test_doc"
        )
        
        text = "Python is a programming language used for data science."
        existing_entities = []
        
        score = await model.score_entity(entity, text, existing_entities)
        
        print(f"✅ Entity confidence score: {score:.2f}")
        assert 0.0 <= score <= 1.0, f"Score should be between 0 and 1, got {score}"
        
        return True
        
    except Exception as e:
        print(f"❌ EntityConfidenceModel test failed: {e}")
        return False


async def test_relation_validator():
    """Test the RelationValidator."""
    try:
        from morag_graph.extraction.enhanced_relation_extractor import RelationValidator
        from morag_graph.models import Entity, Relation
        
        validator = RelationValidator()
        
        # Create test entities and relation
        source_entity = Entity(name="John", type="PERSON", confidence=0.9)
        target_entity = Entity(name="Microsoft", type="ORGANIZATION", confidence=0.9)
        
        relation = Relation(
            source_entity_id=source_entity.id,
            target_entity_id=target_entity.id,
            type="WORKS_FOR",
            description="John works for Microsoft",
            confidence=0.8
        )
        
        context_text = "John Smith is a software engineer at Microsoft Corporation."
        
        # Test semantic validation
        result = await validator._validate_semantic(
            relation, context_text, source_entity, target_entity
        )
        
        print(f"✅ Relation validation result:")
        print(f"   Valid: {result.is_valid}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Issues: {len(result.issues)}")
        
        return True
        
    except Exception as e:
        print(f"❌ RelationValidator test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Extraction Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_enhanced_extraction_imports),
        ("Entity Similarity Calculator", test_entity_similarity_calculator),
        ("Pipeline Config", test_pipeline_config),
        ("Entity Confidence Model", test_confidence_model),
        ("Relation Validator", test_relation_validator)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
