"""Example demonstrating custom schema functionality in Graphiti."""

import asyncio
from datetime import datetime
from morag_graph.graphiti import (
    GraphitiConfig,
    MoragEntityType, MoragRelationType,
    PersonEntity, OrganizationEntity, TechnologyEntity,
    SemanticRelation, TemporalRelation,
    SchemaRegistry, schema_registry,
    create_schema_aware_storage, create_schema_aware_search
)


async def custom_schema_example():
    """Demonstrate custom schema functionality."""
    
    print("=== MoRAG Custom Schema Examples ===\n")
    
    # Example 1: Creating and validating custom entities
    print("1. Custom Entity Creation and Validation")
    print("-" * 40)
    
    # Create a person entity with validation
    person_data = {
        'id': 'person_001',
        'name': 'Dr. Sarah Chen',
        'type': MoragEntityType.PERSON,
        'confidence': 0.95,
        'title': 'Dr.',
        'organization': 'Stanford University',
        'role': 'AI Research Scientist',
        'email': 'sarah.chen@stanford.edu',
        'expertise_areas': ['Machine Learning', 'Natural Language Processing', 'Computer Vision'],
        'publications': ['Attention Is All You Need', 'BERT: Pre-training of Deep Bidirectional Transformers']
    }
    
    try:
        person = PersonEntity(**person_data)
        print(f"✅ Created person entity: {person.name}")
        print(f"   Organization: {person.organization}")
        print(f"   Expertise: {', '.join(person.expertise_areas)}")
        print(f"   Confidence: {person.confidence}")
    except Exception as e:
        print(f"❌ Person entity creation failed: {e}")
    
    # Create an organization entity
    org_data = {
        'id': 'org_001',
        'name': 'OpenAI',
        'type': MoragEntityType.ORGANIZATION,
        'confidence': 0.98,
        'organization_type': 'company',
        'industry': 'Artificial Intelligence',
        'location': 'San Francisco, CA',
        'website': 'https://openai.com',
        'founded_year': 2015,
        'size': 'large'
    }
    
    try:
        organization = OrganizationEntity(**org_data)
        print(f"✅ Created organization entity: {organization.name}")
        print(f"   Type: {organization.organization_type}")
        print(f"   Industry: {organization.industry}")
        print(f"   Founded: {organization.founded_year}")
    except Exception as e:
        print(f"❌ Organization entity creation failed: {e}")
    
    # Create a technology entity
    tech_data = {
        'id': 'tech_001',
        'name': 'GPT-4',
        'type': MoragEntityType.TECHNOLOGY,
        'confidence': 0.92,
        'category': 'language_model',
        'version': '4.0',
        'vendor': 'OpenAI',
        'license': 'Proprietary',
        'maturity_level': 'stable',
        'documentation_url': 'https://platform.openai.com/docs'
    }
    
    try:
        technology = TechnologyEntity(**tech_data)
        print(f"✅ Created technology entity: {technology.name}")
        print(f"   Category: {technology.category}")
        print(f"   Version: {technology.version}")
        print(f"   Maturity: {technology.maturity_level}")
    except Exception as e:
        print(f"❌ Technology entity creation failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Creating and validating custom relations
    print("2. Custom Relation Creation and Validation")
    print("-" * 40)
    
    # Create a semantic relation
    semantic_relation_data = {
        'id': 'rel_001',
        'source_entity_id': 'person_001',
        'target_entity_id': 'org_001',
        'relation_type': MoragRelationType.BELONGS_TO,
        'confidence': 0.88,
        'strength': 0.9,
        'directionality': 'unidirectional',
        'temporal_scope': 'present',
        'evidence_text': 'Dr. Sarah Chen works at OpenAI as a research scientist',
        'description': 'Employment relationship'
    }
    
    try:
        semantic_relation = SemanticRelation(**semantic_relation_data)
        print(f"✅ Created semantic relation: {semantic_relation.relation_type}")
        print(f"   Strength: {semantic_relation.strength}")
        print(f"   Directionality: {semantic_relation.directionality}")
        print(f"   Evidence: {semantic_relation.evidence_text}")
    except Exception as e:
        print(f"❌ Semantic relation creation failed: {e}")
    
    # Create a temporal relation
    temporal_relation_data = {
        'id': 'rel_002',
        'source_entity_id': 'org_001',
        'target_entity_id': 'tech_001',
        'relation_type': MoragRelationType.IMPLEMENTS,
        'confidence': 0.95,
        'start_time': datetime(2022, 1, 1),
        'end_time': datetime(2023, 3, 14),
        'duration': '1 year 2 months',
        'temporal_precision': 'approximate',
        'description': 'OpenAI implements GPT-4 technology'
    }
    
    try:
        temporal_relation = TemporalRelation(**temporal_relation_data)
        print(f"✅ Created temporal relation: {temporal_relation.relation_type}")
        print(f"   Duration: {temporal_relation.duration}")
        print(f"   Precision: {temporal_relation.temporal_precision}")
        print(f"   Start: {temporal_relation.start_time.strftime('%Y-%m-%d')}")
        print(f"   End: {temporal_relation.end_time.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"❌ Temporal relation creation failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Schema Registry Usage
    print("3. Schema Registry Validation")
    print("-" * 40)
    
    registry = SchemaRegistry()
    
    # Validate entity through registry
    entity_data = {
        'id': 'concept_001',
        'name': 'Transformer Architecture',
        'type': 'CONCEPT',
        'confidence': 0.9,
        'domain': 'Machine Learning',
        'definition': 'A neural network architecture based on attention mechanisms',
        'complexity_level': 'advanced',
        'related_concepts': ['Attention Mechanism', 'Self-Attention', 'Multi-Head Attention']
    }
    
    try:
        validated_entity = registry.validate_entity(entity_data)
        print(f"✅ Validated entity through registry: {validated_entity['name']}")
        print(f"   Type: {validated_entity['type']}")
        print(f"   Domain: {validated_entity.get('domain', 'N/A')}")
        print(f"   Complexity: {validated_entity.get('complexity_level', 'N/A')}")
    except Exception as e:
        print(f"❌ Entity validation failed: {e}")
    
    # Validate relation through registry
    relation_data = {
        'id': 'rel_003',
        'source_entity_id': 'tech_001',
        'target_entity_id': 'concept_001',
        'relation_type': 'IMPLEMENTS',
        'confidence': 0.85,
        'strength': 0.8,
        'directionality': 'unidirectional',
        'evidence_text': 'GPT-4 implements the Transformer architecture'
    }
    
    try:
        validated_relation = registry.validate_relation(relation_data, "semantic")
        print(f"✅ Validated relation through registry: {validated_relation['relation_type']}")
        print(f"   Strength: {validated_relation['strength']}")
        print(f"   Evidence: {validated_relation['evidence_text']}")
    except Exception as e:
        print(f"❌ Relation validation failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Custom Schema Registration
    print("4. Custom Schema Registration")
    print("-" * 40)
    
    # Define a custom entity schema
    from morag_graph.graphiti.custom_schema import BaseEntitySchema
    from pydantic import Field
    
    class ResearchPaperEntity(BaseEntitySchema):
        """Custom schema for research papers."""
        type: MoragEntityType = MoragEntityType.DOCUMENT
        
        # Research paper specific attributes
        authors: list[str] = Field(default_factory=list)
        venue: str = None
        year: int = None
        citations: int = 0
        impact_factor: float = 0.0
        research_area: str = None
        methodology: str = None
    
    # Register the custom schema
    registry.register_entity_schema(MoragEntityType.DOCUMENT, ResearchPaperEntity)
    print("✅ Registered custom ResearchPaperEntity schema")
    
    # Use the custom schema
    paper_data = {
        'id': 'paper_001',
        'name': 'Attention Is All You Need',
        'type': 'DOCUMENT',
        'confidence': 0.99,
        'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
        'venue': 'NIPS 2017',
        'year': 2017,
        'citations': 50000,
        'impact_factor': 9.2,
        'research_area': 'Natural Language Processing',
        'methodology': 'Transformer Architecture'
    }
    
    try:
        validated_paper = registry.validate_entity(paper_data)
        print(f"✅ Validated custom paper entity: {validated_paper['name']}")
        print(f"   Authors: {', '.join(validated_paper['authors'][:2])}...")
        print(f"   Venue: {validated_paper['venue']}")
        print(f"   Citations: {validated_paper['citations']:,}")
        print(f"   Impact Factor: {validated_paper['impact_factor']}")
    except Exception as e:
        print(f"❌ Custom paper validation failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Error Handling and Fallbacks
    print("5. Error Handling and Fallback Validation")
    print("-" * 40)
    
    # Test with invalid data
    invalid_entity_data = {
        'type': 'PERSON',
        'confidence': 0.8,
        'email': 'invalid-email-format',  # Invalid email
        'some_extra_field': 'extra_value'
        # Missing required fields
    }
    
    try:
        fallback_entity = registry.validate_entity(invalid_entity_data)
        print(f"✅ Fallback validation succeeded")
        print(f"   ID: {fallback_entity.get('id', 'N/A')}")
        print(f"   Name: {fallback_entity.get('name', 'N/A')}")
        print(f"   Type: {fallback_entity.get('type', 'N/A')}")
        print(f"   Extra field preserved: {fallback_entity.get('some_extra_field', 'N/A')}")
    except Exception as e:
        print(f"❌ Even fallback validation failed: {e}")
    
    print("\n" + "="*50)
    print("Custom schema examples completed!")
    print("\nKey Features Demonstrated:")
    print("• Type-safe entity and relation creation")
    print("• Schema validation with custom attributes")
    print("• Registry-based validation system")
    print("• Custom schema registration")
    print("• Robust error handling and fallbacks")
    print("• Pydantic v2 compatibility")


if __name__ == "__main__":
    print("MoRAG Custom Schema Example")
    print("=" * 50)
    print("Note: This example demonstrates schema validation without requiring")
    print("a live Graphiti connection. For storage examples, see the storage demo.")
    print("=" * 50)
    
    # Run the example
    asyncio.run(custom_schema_example())
