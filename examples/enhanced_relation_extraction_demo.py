"""Demo script showing enhanced relation extraction capabilities."""

import asyncio
import json
from typing import List, Dict
import sys
import os

# Add the packages to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-graph', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-core', 'src'))

from morag_graph.ai.multi_pass_extractor import MultiPassRelationExtractor
from morag_graph.ai.relation_agent import RelationExtractionAgent
from morag_graph.models import Entity


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_relations(relations: List, title: str = "Relations"):
    """Print relations in a formatted way."""
    print(f"\n{title} ({len(relations)} found):")
    print("-" * 40)
    
    if not relations:
        print("  No relations found.")
        return
    
    for i, relation in enumerate(relations, 1):
        source = getattr(relation, 'source_entity_name', 
                        getattr(relation, 'source_entity', 'Unknown'))
        target = getattr(relation, 'target_entity_name',
                        getattr(relation, 'target_entity', 'Unknown'))
        rel_type = getattr(relation, 'type', getattr(relation, 'relation_type', 'Unknown'))
        confidence = getattr(relation, 'confidence', 0.0)
        
        print(f"  {i:2d}. {source} --[{rel_type}]--> {target} (conf: {confidence:.3f})")


async def demo_medical_extraction():
    """Demonstrate enhanced extraction on medical text."""
    print_section("Medical Domain Extraction Demo")
    
    # Sample medical text with rich relationships
    medical_text = """
    Dr. Sarah Chen, a cardiologist at Johns Hopkins Hospital, has been researching 
    the effects of statins on cardiovascular disease prevention. Her recent study 
    demonstrates that atorvastatin significantly reduces LDL cholesterol levels, 
    which directly prevents atherosclerosis development. However, statins can cause 
    muscle pain in approximately 10% of patients, making them contraindicated for 
    people with existing muscle disorders.
    
    The research shows that patients taking atorvastatin experienced a 40% reduction 
    in heart attacks compared to the control group. Dr. Chen's methodology involves 
    analyzing patient data from multiple clinical trials to establish causal 
    relationships between medication dosage and cardiovascular outcomes.
    
    The study also reveals that combining atorvastatin with lifestyle changes 
    produces even better results, as diet and exercise enhance the drug's effectiveness.
    """
    
    # Create medical entities
    medical_entities = [
        Entity(name="Dr. Sarah Chen", type="PERSON", confidence=0.95, source_doc_id="demo"),
        Entity(name="Johns Hopkins Hospital", type="ORGANIZATION", confidence=0.9, source_doc_id="demo"),
        Entity(name="atorvastatin", type="SUBSTANCE", confidence=0.9, source_doc_id="demo"),
        Entity(name="statins", type="SUBSTANCE", confidence=0.85, source_doc_id="demo"),
        Entity(name="cardiovascular disease", type="CONCEPT", confidence=0.9, source_doc_id="demo"),
        Entity(name="LDL cholesterol", type="SUBSTANCE", confidence=0.85, source_doc_id="demo"),
        Entity(name="atherosclerosis", type="CONCEPT", confidence=0.8, source_doc_id="demo"),
        Entity(name="muscle pain", type="CONCEPT", confidence=0.8, source_doc_id="demo"),
        Entity(name="heart attacks", type="CONCEPT", confidence=0.85, source_doc_id="demo"),
    ]
    
    print(f"Text length: {len(medical_text)} characters")
    print(f"Entities: {len(medical_entities)}")
    
    # Test with dynamic relation extraction
    print_section("Dynamic Relation Extraction")
    dynamic_agent = RelationExtractionAgent(
        min_confidence=0.6,
        use_enhanced_extraction=True,
        enable_multi_pass=False,  # Single pass for comparison
        dynamic_types=True  # Enable fully dynamic types
    )

    dynamic_relations = await dynamic_agent.extract_relations(
        text=medical_text,
        entities=medical_entities,
        source_doc_id="demo_medical",
        domain_hint="medical"
    )

    print_relations(dynamic_relations, "Dynamic Relations")
    
    # Test with enhanced multi-pass extraction
    print_section("Enhanced Multi-Pass Extraction")
    enhanced_extractor = MultiPassRelationExtractor(
        min_confidence=0.6,
        enable_semantic_analysis=True,
        enable_domain_extraction=True,
        enable_contextual_enhancement=True
    )
    
    result = await enhanced_extractor.extract_relations_multi_pass(
        text=medical_text,
        entities=medical_entities,
        source_doc_id="demo_medical",
        domain_hint="medical"
    )
    
    print_relations(result.final_relations, "Enhanced Relations")
    
    # Print extraction summary
    print_section("Extraction Analysis")
    print(enhanced_extractor.get_extraction_summary(result))
    
    # Compare relation types
    dynamic_types = [getattr(r, 'type', 'Unknown') for r in dynamic_relations]
    enhanced_types = [r.type for r in result.final_relations]

    print(f"\nRelation Type Comparison:")
    print(f"Dynamic types: {set(dynamic_types)}")
    print(f"Multi-pass enhanced types: {set(enhanced_types)}")

    # Show the AI's creativity in relation type creation
    print(f"\nAI-Generated Relation Types:")
    all_types = set(dynamic_types + enhanced_types)
    for rel_type in sorted(all_types):
        print(f"  - {rel_type}")

    print(f"\nTotal unique relation types created: {len(all_types)}")
    print("Notice how the AI creates specific, meaningful relation types based on context!")


async def demo_technical_extraction():
    """Demonstrate enhanced extraction on technical text."""
    print_section("Technical Domain Extraction Demo")
    
    # Sample technical text
    technical_text = """
    Python is a high-level programming language that enables rapid application development.
    Django, a popular Python web framework, implements the Model-View-Controller architecture
    and integrates seamlessly with PostgreSQL databases. The framework provides built-in
    support for REST API development, which allows applications to communicate with external
    microservices.
    
    Django's Object-Relational Mapping (ORM) layer abstracts database operations and
    transforms Python objects into SQL queries. This architecture enables developers to
    build scalable web applications that can handle millions of concurrent requests.
    
    The framework also includes a powerful admin interface that automatically generates
    forms based on model definitions, significantly reducing development time. Additionally,
    Django's security features prevent common vulnerabilities like SQL injection and
    cross-site scripting attacks.
    """
    
    # Create technical entities
    technical_entities = [
        Entity(name="Python", type="TECHNOLOGY", confidence=0.95, source_doc_id="demo"),
        Entity(name="Django", type="SOFTWARE", confidence=0.9, source_doc_id="demo"),
        Entity(name="PostgreSQL", type="SOFTWARE", confidence=0.85, source_doc_id="demo"),
        Entity(name="REST API", type="CONCEPT", confidence=0.8, source_doc_id="demo"),
        Entity(name="ORM", type="TECHNOLOGY", confidence=0.85, source_doc_id="demo"),
        Entity(name="SQL", type="TECHNOLOGY", confidence=0.8, source_doc_id="demo"),
        Entity(name="microservices", type="CONCEPT", confidence=0.8, source_doc_id="demo"),
    ]
    
    print(f"Text length: {len(technical_text)} characters")
    print(f"Entities: {len(technical_entities)}")
    
    # Enhanced extraction
    enhanced_extractor = MultiPassRelationExtractor(
        min_confidence=0.6,
        enable_semantic_analysis=True,
        enable_domain_extraction=True
    )
    
    result = await enhanced_extractor.extract_relations_multi_pass(
        text=technical_text,
        entities=technical_entities,
        source_doc_id="demo_technical",
        domain_hint="technical"
    )
    
    print_relations(result.final_relations, "Technical Relations")
    
    # Print summary
    print_section("Technical Extraction Analysis")
    print(enhanced_extractor.get_extraction_summary(result))


async def demo_comparison():
    """Compare basic vs enhanced extraction side by side."""
    print_section("Basic vs Enhanced Comparison")
    
    # Simple test text
    test_text = """
    Apple Inc. was founded by Steve Jobs in 1976. The company develops innovative
    technology products including the iPhone and MacBook. Steve Jobs revolutionized
    the smartphone industry with the iPhone launch in 2007. Apple's success depends
    on continuous innovation and premium design philosophy.
    """
    
    entities = [
        Entity(name="Apple Inc.", type="ORGANIZATION", confidence=0.95, source_doc_id="demo"),
        Entity(name="Steve Jobs", type="PERSON", confidence=0.95, source_doc_id="demo"),
        Entity(name="iPhone", type="PRODUCT", confidence=0.9, source_doc_id="demo"),
        Entity(name="MacBook", type="PRODUCT", confidence=0.85, source_doc_id="demo"),
    ]
    
    print(f"Test text: {test_text[:100]}...")
    print(f"Entities: {[e.name for e in entities]}")
    
    # Dynamic extraction
    dynamic_agent = RelationExtractionAgent(
        min_confidence=0.5,
        use_enhanced_extraction=True,
        enable_multi_pass=False,
        dynamic_types=True
    )
    dynamic_relations = await dynamic_agent.extract_relations(
        text=test_text,
        entities=entities,
        domain_hint="business"
    )

    # Multi-pass enhanced extraction
    enhanced_agent = RelationExtractionAgent(
        min_confidence=0.5,
        use_enhanced_extraction=True,
        enable_multi_pass=True,
        dynamic_types=True
    )
    enhanced_relations = await enhanced_agent.extract_relations(
        text=test_text,
        entities=entities,
        domain_hint="business"
    )

    print_relations(dynamic_relations, "Dynamic Relations")
    print_relations(enhanced_relations, "Multi-Pass Enhanced Relations")

    # Analysis
    print(f"\nDynamic Relation Analysis:")
    print(f"Dynamic relations: {len(dynamic_relations)}")
    print(f"Multi-pass enhanced relations: {len(enhanced_relations)}")

    dynamic_types = set(getattr(r, 'type', 'Unknown') for r in dynamic_relations)
    enhanced_types = set(r.type for r in enhanced_relations)

    print(f"Dynamic types: {dynamic_types}")
    print(f"Enhanced types: {enhanced_types}")
    print(f"Additional types from multi-pass: {enhanced_types - dynamic_types}")

    print(f"\nAll AI-generated relation types:")
    all_types = dynamic_types.union(enhanced_types)
    for rel_type in sorted(all_types):
        print(f"  - {rel_type}")


async def main():
    """Run all demos."""
    print("Enhanced Relation Extraction Demo")
    print("=" * 60)
    print("This demo shows the improvements in relation extraction")
    print("using semantic analysis, domain-specific patterns, and")
    print("multi-pass enhancement techniques.")
    
    try:
        await demo_medical_extraction()
        await demo_technical_extraction()
        await demo_comparison()
        
        print_section("Demo Complete")
        print("The dynamic relation extraction system demonstrates:")
        print("• Fully AI-generated relation types based on context")
        print("• No predefined relation vocabularies or constraints")
        print("• Domain-aware but flexible extraction")
        print("• Semantic analysis for deeper understanding")
        print("• Multi-pass refinement for higher quality")
        print("• Creative and precise relation type naming")
        print("• Context-specific relationship descriptions")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
