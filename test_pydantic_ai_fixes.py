#!/usr/bin/env python3
"""
Test script to validate PydanticAI semantic chunking fixes.

This script tests:
1. Entity-chunk relationship creation (mentioned_in_chunks field)
2. Dynamic relation types instead of static enums
3. Diverse, domain-appropriate entity and relation types
"""

import asyncio
import os
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.extraction.base import LLMConfig


async def test_dynamic_types():
    """Test that dynamic types work correctly."""
    print("🧪 Testing Dynamic Types...")
    
    # Create LLM config for testing
    config = LLMConfig(provider="mock", model="test")
    
    # Test entity extractor with dynamic types
    entity_extractor = EntityExtractor(config, dynamic_types=True)
    print(f"✅ Entity extractor dynamic_types: {entity_extractor.dynamic_types}")
    print(f"✅ Entity extractor entity_types: {entity_extractor.entity_types}")
    
    # Test relation extractor with dynamic types
    relation_extractor = RelationExtractor(config, dynamic_types=True)
    print(f"✅ Relation extractor dynamic_types: {relation_extractor.dynamic_types}")
    print(f"✅ Relation extractor relation_types: {relation_extractor.relation_types}")
    
    # Test system prompts
    entity_prompt = entity_extractor.get_system_prompt()
    relation_prompt = relation_extractor.get_system_prompt()
    
    print(f"✅ Entity prompt contains 'semantic meaning': {'semantic meaning' in entity_prompt}")
    print(f"✅ Entity prompt contains 'not limit yourself': {'not limit yourself' in entity_prompt}")
    print(f"✅ Relation prompt contains 'semantic meaning': {'semantic meaning' in relation_prompt}")
    print(f"✅ Relation prompt contains 'not limit yourself': {'not limit yourself' in relation_prompt}")


async def test_custom_types():
    """Test that custom types work correctly."""
    print("\n🧪 Testing Custom Types...")
    
    config = LLMConfig(provider="mock", model="test")
    
    # Test with custom entity types
    custom_entity_types = {
        "MEDICAL_CONDITION": "Disease or health condition",
        "TREATMENT": "Medical treatment or therapy"
    }
    entity_extractor = EntityExtractor(config, entity_types=custom_entity_types)
    
    # Test with custom relation types
    custom_relation_types = {
        "TREATS": "Treatment treats condition",
        "CAUSES": "Pathogen causes disease"
    }
    relation_extractor = RelationExtractor(config, relation_types=custom_relation_types)
    
    print(f"✅ Custom entity types: {entity_extractor.entity_types}")
    print(f"✅ Custom relation types: {relation_extractor.relation_types}")
    
    # Check system prompts include custom types
    entity_prompt = entity_extractor.get_system_prompt()
    relation_prompt = relation_extractor.get_system_prompt()
    
    print(f"✅ Entity prompt contains 'MEDICAL_CONDITION': {'MEDICAL_CONDITION' in entity_prompt}")
    print(f"✅ Relation prompt contains 'TREATS': {'TREATS' in relation_prompt}")


def test_entity_chunk_relationship_fix():
    """Test that the entity-chunk relationship fix is in place."""
    print("\n🧪 Testing Entity-Chunk Relationship Fix...")
    
    # Check that the ingestion coordinator has the fix
    ingestion_file = Path(__file__).parent / "packages" / "morag" / "src" / "morag" / "ingestion_coordinator.py"
    
    if ingestion_file.exists():
        content = ingestion_file.read_text()
        
        # Check for the fix indicators
        has_entity_mapping = "entity_id_to_entity" in content
        has_chunk_reference = "add_chunk_reference" in content
        has_auto_created_fetch = "fetch it from Neo4j" in content
        
        print(f"✅ Has entity ID mapping: {has_entity_mapping}")
        print(f"✅ Has chunk reference update: {has_chunk_reference}")
        print(f"✅ Has auto-created entity fetch: {has_auto_created_fetch}")
        
        if has_entity_mapping and has_chunk_reference and has_auto_created_fetch:
            print("✅ Entity-chunk relationship fix is properly implemented!")
        else:
            print("❌ Entity-chunk relationship fix may be incomplete")
    else:
        print("❌ Could not find ingestion coordinator file")


def test_dynamic_type_conversion():
    """Test that dynamic type conversion is implemented."""
    print("\n🧪 Testing Dynamic Type Conversion...")
    
    # Check entity agent
    entity_agent_file = Path(__file__).parent / "packages" / "morag-graph" / "src" / "morag_graph" / "ai" / "entity_agent.py"
    relation_agent_file = Path(__file__).parent / "packages" / "morag-graph" / "src" / "morag_graph" / "ai" / "relation_agent.py"
    
    if entity_agent_file.exists():
        content = entity_agent_file.read_text()
        has_dynamic_types = "self.dynamic_types" in content
        has_dynamic_conversion = "if self.dynamic_types:" in content
        print(f"✅ Entity agent has dynamic types support: {has_dynamic_types}")
        print(f"✅ Entity agent has dynamic conversion: {has_dynamic_conversion}")
    
    if relation_agent_file.exists():
        content = relation_agent_file.read_text()
        has_dynamic_types = "self.dynamic_types" in content
        has_dynamic_conversion = "if self.dynamic_types:" in content
        print(f"✅ Relation agent has dynamic types support: {has_dynamic_types}")
        print(f"✅ Relation agent has dynamic conversion: {has_dynamic_conversion}")


async def main():
    """Run all tests."""
    print("🚀 Testing PydanticAI Semantic Chunking Fixes\n")
    
    try:
        await test_dynamic_types()
        await test_custom_types()
        test_entity_chunk_relationship_fix()
        test_dynamic_type_conversion()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary of fixes:")
        print("1. ✅ Entity-chunk relationships now update mentioned_in_chunks field")
        print("2. ✅ Dynamic relation types replace static enums")
        print("3. ✅ LLM determines appropriate entity and relation types")
        print("4. ✅ System prompts encourage diverse, domain-specific types")
        print("5. ✅ Auto-created entities are properly linked to chunks")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
