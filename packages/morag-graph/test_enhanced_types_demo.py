#!/usr/bin/env python3
"""
Demo script to show enhanced entity and relationship type generation.

This script demonstrates how the improved graph extraction now creates
diverse, specialized entity labels and relationship types instead of
generic ones.
"""

import asyncio
from morag_graph.models import Entity, Relation

def demo_entity_labels():
    """Demonstrate enhanced entity label generation."""
    print("🏷️  Enhanced Entity Label Generation")
    print("=" * 50)
    
    # Test various entity types that should generate specific Neo4j labels
    test_entities = [
        ("Dr. Sarah Johnson", "MEDICAL_RESEARCHER"),
        ("Stanford University", "RESEARCH_INSTITUTION"),
        ("Alzheimer's Disease", "NEUROLOGICAL_CONDITION"),
        ("Pfizer Inc.", "PHARMACEUTICAL_COMPANY"),
        ("Machine Learning", "AI_TECHNOLOGY"),
        ("Clinical Trial Protocol", "RESEARCH_METHODOLOGY"),
        ("Patient Care Unit", "MEDICAL_FACILITY"),
        ("Data Analytics Platform", "SOFTWARE_SYSTEM"),
    ]
    
    for name, entity_type in test_entities:
        entity = Entity(
            name=name,
            type=entity_type,
            confidence=0.9,
            attributes={}
        )
        
        neo4j_label = entity.get_neo4j_label()
        print(f"  Entity: {name}")
        print(f"    Type: {entity_type}")
        print(f"    Neo4j Label: :{neo4j_label}")
        print()

def demo_relationship_types():
    """Demonstrate enhanced relationship type generation."""
    print("🔗 Enhanced Relationship Type Generation")
    print("=" * 50)
    
    # Test various relationship types that should generate specific Neo4j types
    test_relations = [
        ("CONDUCTS_RESEARCH_ON", "Researcher studies a specific topic"),
        ("IS_EMPLOYED_BY", "Person works for an organization"),
        ("TREATS_CONDITION", "Medical intervention addresses disease"),
        ("COLLABORATES_WITH", "Entities work together"),
        ("MANUFACTURES_DRUG", "Company produces pharmaceutical"),
        ("IS_LOCATED_IN", "Entity has physical location"),
        ("DEVELOPS_TECHNOLOGY", "Organization creates new tech"),
        ("DIAGNOSES_PATIENT", "Doctor identifies medical condition"),
    ]
    
    for rel_type, description in test_relations:
        relation = Relation(
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            type=rel_type,
            confidence=0.9,
            attributes={"description": description}
        )
        
        neo4j_type = relation.get_neo4j_type()
        print(f"  Relationship: {rel_type}")
        print(f"    Description: {description}")
        print(f"    Neo4j Type: -{neo4j_type}->")
        print()

def demo_prompt_improvements():
    """Demonstrate the improved LLM prompts."""
    print("💬 Enhanced LLM Prompts")
    print("=" * 50)
    
    try:
        from morag_graph.extraction.entity_extractor import EntityExtractor
        from morag_graph.extraction.relation_extractor import RelationExtractor
        
        # Import LLMConfig
        try:
            from morag_reasoning.llm import LLMConfig
        except ImportError:
            from pydantic import BaseModel
            class LLMConfig(BaseModel):
                provider: str = "gemini"
                model: str = "gemini-1.5-flash"
                api_key: str = None
                temperature: float = 0.1
                max_tokens: int = 2000
        
        config = LLMConfig(provider="mock", model="test")
        
        # Test medical domain
        entity_extractor = EntityExtractor(config, domain="medical")
        relation_extractor = RelationExtractor(config, domain="medical")
        
        entity_prompt = entity_extractor._create_entity_prompt()
        relation_prompt = relation_extractor._create_relation_prompt()
        
        print("Entity Extraction Prompt Key Features:")
        print("  ✓ Encourages SPECIFIC, DESCRIPTIVE entity types")
        print("  ✓ Provides examples: RESEARCHER, HOSPITAL, MEDICAL_CONDITION")
        print("  ✓ Discourages generic types: ENTITY, THING, ITEM, OBJECT")
        print("  ✓ Specifies uppercase underscore format")
        print("  ✓ Domain-specific guidance for medical context")
        print()
        
        print("Relation Extraction Prompt Key Features:")
        print("  ✓ Encourages SPECIFIC, DESCRIPTIVE relationship types")
        print("  ✓ Provides examples: EMPLOYS, TREATS, COLLABORATES_WITH")
        print("  ✓ Discourages generic types: RELATES, CONNECTS, LINKS")
        print("  ✓ Specifies action-oriented relationship types")
        print("  ✓ Domain-specific guidance for medical context")
        print()
        
    except Exception as e:
        print(f"Error demonstrating prompts: {e}")

def demo_neo4j_storage_changes():
    """Demonstrate how Neo4j storage now uses dynamic labels."""
    print("🗄️  Neo4j Storage Improvements")
    print("=" * 50)
    
    print("Before (Generic Labels):")
    print("  Entities: All stored as :Entity nodes")
    print("  Relations: All stored as :RELATION relationships")
    print("  Types: Stored only in 'type' property")
    print()
    
    print("After (Dynamic Labels):")
    print("  Entities: Stored with specific labels like :RESEARCHER, :HOSPITAL")
    print("  Relations: Stored with specific types like :EMPLOYS, :TREATS")
    print("  Types: Used as actual Neo4j labels and relationship types")
    print()
    
    print("Benefits:")
    print("  ✓ Better query performance (can filter by label)")
    print("  ✓ More intuitive graph structure")
    print("  ✓ Richer semantic information")
    print("  ✓ Domain-specific graph schemas")
    print()

def main():
    """Run the demo."""
    print("🚀 Enhanced Graph Entity and Relationship Type Generation Demo")
    print("=" * 70)
    print()
    
    demo_entity_labels()
    print()
    
    demo_relationship_types()
    print()
    
    demo_prompt_improvements()
    print()
    
    demo_neo4j_storage_changes()
    print()
    
    print("✅ Demo completed! The graph extraction now generates diverse,")
    print("   specialized entity labels and relationship types instead of")
    print("   generic ones, leading to much richer knowledge graphs.")

if __name__ == "__main__":
    main()
