"""Example demonstrating the enhanced entity and relation extraction pipeline."""

import asyncio
import sys
import os
import json
from typing import Dict, Any

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'morag-graph', 'src'))

from morag_graph.extraction.unified_extraction_pipeline import (
    UnifiedExtractionPipeline,
    PipelineConfig
)


class SimpleDocument:
    """Simple document class for testing."""
    
    def __init__(self, content: str, doc_id: str):
        self.content = content
        self.id = doc_id


async def demonstrate_basic_extraction():
    """Demonstrate basic enhanced extraction."""
    print("🔍 Basic Enhanced Extraction Demo")
    print("-" * 50)
    
    # Sample text for extraction
    text = """
    Dr. Alice Johnson is a senior researcher at Microsoft Research in Seattle, Washington. 
    She specializes in machine learning and artificial intelligence. Alice founded the 
    AI Ethics Lab in 2020 and has published over 50 papers on neural networks. 
    She collaborates with Stanford University and MIT on various research projects.
    The AI Ethics Lab focuses on responsible AI development and has received funding 
    from the National Science Foundation.
    """
    
    # Create pipeline with enhanced settings
    config = PipelineConfig(
        entity_max_rounds=3,
        entity_target_confidence=0.8,
        enable_entity_gleaning=True,
        relation_max_rounds=2,
        enable_relation_validation=True,
        enable_deduplication=True,
        domain="research",
        language="en"
    )
    
    pipeline = UnifiedExtractionPipeline(config=config)
    
    print("📄 Processing text...")
    print(f"Text length: {len(text)} characters")
    
    # Process the text
    result = await pipeline.process_text(text, "demo_doc_1")
    
    print(f"\n✅ Extraction completed in {result.processing_time:.2f} seconds")
    print(f"📊 Results:")
    print(f"   Entities extracted: {len(result.entities)}")
    print(f"   Relations extracted: {len(result.relations)}")
    print(f"   Chunks processed: {result.chunks_processed}")
    
    # Display entities
    print(f"\n👥 Entities Found:")
    for i, entity in enumerate(result.entities, 1):
        print(f"   {i}. {entity.name} ({entity.type}) - Confidence: {entity.confidence:.2f}")
    
    # Display relations
    print(f"\n🔗 Relations Found:")
    entity_dict = {e.id: e.name for e in result.entities}
    for i, relation in enumerate(result.relations, 1):
        source_name = entity_dict.get(relation.source_entity_id, "Unknown")
        target_name = entity_dict.get(relation.target_entity_id, "Unknown")
        print(f"   {i}. {source_name} --[{relation.type}]--> {target_name}")
        print(f"      Confidence: {relation.confidence:.2f}")
        if relation.description:
            print(f"      Description: {relation.description}")
    
    return result


async def demonstrate_multi_document_processing():
    """Demonstrate multi-document processing with cross-document deduplication."""
    print("\n\n📚 Multi-Document Processing Demo")
    print("-" * 50)
    
    # Sample documents with overlapping entities
    documents = [
        SimpleDocument(
            content="""
            Dr. Alice Johnson works at Microsoft Research and leads the AI Ethics Lab. 
            She has a PhD from Stanford University and specializes in machine learning ethics.
            """,
            doc_id="doc_1"
        ),
        SimpleDocument(
            content="""
            Alice Johnson, a researcher at Microsoft, published a paper on neural network bias. 
            The AI Ethics Lab at Microsoft Research focuses on responsible AI development.
            Stanford University collaborates with Microsoft on this research.
            """,
            doc_id="doc_2"
        ),
        SimpleDocument(
            content="""
            The National Science Foundation awarded a grant to Microsoft Research for AI ethics research. 
            Dr. Johnson leads this initiative and works closely with academic institutions.
            """,
            doc_id="doc_3"
        )
    ]
    
    # Create pipeline with cross-document deduplication
    config = PipelineConfig(
        entity_max_rounds=2,
        enable_entity_gleaning=True,
        enable_deduplication=True,
        enable_parallel_processing=True,
        domain="research"
    )
    
    pipeline = UnifiedExtractionPipeline(config=config)
    
    print(f"📄 Processing {len(documents)} documents...")
    
    # Process multiple documents
    results = await pipeline.process_multiple_documents(
        documents, 
        enable_cross_document_deduplication=True
    )
    
    print(f"\n✅ Multi-document processing completed")
    
    # Display results for each document
    total_entities = 0
    total_relations = 0
    
    for i, result in enumerate(results):
        doc_id = documents[i].id
        print(f"\n📋 Document {doc_id}:")
        print(f"   Entities: {len(result.entities)}")
        print(f"   Relations: {len(result.relations)}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
        total_entities += len(result.entities)
        total_relations += len(result.relations)
        
        # Show deduplication stats if available
        if result.deduplication_result:
            dedup = result.deduplication_result
            print(f"   Deduplication: {dedup.original_count} → {dedup.deduplicated_count} entities")
            print(f"   Merges performed: {dedup.merges_performed}")
    
    print(f"\n📊 Total Results:")
    print(f"   Total entities: {total_entities}")
    print(f"   Total relations: {total_relations}")
    
    return results


async def demonstrate_pipeline_configuration():
    """Demonstrate different pipeline configurations."""
    print("\n\n⚙️  Pipeline Configuration Demo")
    print("-" * 50)
    
    # Test different configurations
    configs = [
        ("Basic", PipelineConfig(
            entity_max_rounds=1,
            relation_max_rounds=1,
            enable_entity_gleaning=False,
            enable_relation_validation=False,
            enable_deduplication=False
        )),
        ("Enhanced", PipelineConfig(
            entity_max_rounds=3,
            relation_max_rounds=2,
            enable_entity_gleaning=True,
            enable_relation_validation=True,
            enable_deduplication=True
        )),
        ("High Performance", PipelineConfig(
            entity_max_rounds=2,
            relation_max_rounds=1,
            enable_entity_gleaning=True,
            enable_relation_validation=False,
            enable_deduplication=True,
            enable_parallel_processing=True,
            max_workers=8
        ))
    ]
    
    test_text = """
    Python is a programming language developed by Guido van Rossum. 
    It is used by companies like Google, Netflix, and Instagram for web development and data science.
    """
    
    results = {}
    
    for config_name, config in configs:
        print(f"\n🔧 Testing {config_name} configuration...")
        
        pipeline = UnifiedExtractionPipeline(config=config)
        result = await pipeline.process_text(test_text, f"test_{config_name.lower()}")
        
        results[config_name] = {
            'entities': len(result.entities),
            'relations': len(result.relations),
            'processing_time': result.processing_time,
            'config': config.to_dict()
        }
        
        print(f"   Entities: {len(result.entities)}")
        print(f"   Relations: {len(result.relations)}")
        print(f"   Time: {result.processing_time:.3f}s")
    
    # Compare results
    print(f"\n📊 Configuration Comparison:")
    print(f"{'Config':<15} {'Entities':<10} {'Relations':<10} {'Time (s)':<10}")
    print("-" * 50)
    
    for config_name, stats in results.items():
        print(f"{config_name:<15} {stats['entities']:<10} {stats['relations']:<10} {stats['processing_time']:<10.3f}")
    
    return results


async def demonstrate_error_handling():
    """Demonstrate error handling and fallback mechanisms."""
    print("\n\n🛡️  Error Handling Demo")
    print("-" * 50)
    
    # Test with problematic input
    problematic_texts = [
        ("Empty text", ""),
        ("Very short text", "Hi."),
        ("Special characters", "!@#$%^&*()_+{}|:<>?"),
        ("Very long text", "A" * 10000),
        ("Mixed languages", "Hello world. Bonjour le monde. Hola mundo.")
    ]
    
    config = PipelineConfig(
        entity_max_rounds=2,
        enable_entity_gleaning=True,
        enable_deduplication=True
    )
    
    pipeline = UnifiedExtractionPipeline(config=config)
    
    for test_name, text in problematic_texts:
        print(f"\n🧪 Testing: {test_name}")
        try:
            result = await pipeline.process_text(text, f"test_{test_name.replace(' ', '_')}")
            print(f"   ✅ Success: {len(result.entities)} entities, {len(result.relations)} relations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Error handling tests completed")


async def main():
    """Run all demonstrations."""
    print("🚀 Enhanced Entity and Relation Extraction Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        await demonstrate_basic_extraction()
        await demonstrate_multi_document_processing()
        await demonstrate_pipeline_configuration()
        await demonstrate_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\n📋 Key Features Demonstrated:")
        print("   ✅ Multi-round entity gleaning")
        print("   ✅ Relation validation and refinement")
        print("   ✅ Cross-chunk entity deduplication")
        print("   ✅ Multi-document processing")
        print("   ✅ Configurable pipeline settings")
        print("   ✅ Error handling and fallbacks")
        print("   ✅ Performance optimization")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
