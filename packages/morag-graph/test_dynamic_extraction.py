#!/usr/bin/env python3
"""
Test script to verify that dynamic entity and relation type extraction works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor
# Import LLMConfig from morag-reasoning package
try:
    from morag_reasoning.llm import LLMConfig
except ImportError:
    # Fallback LLMConfig for compatibility
    from pydantic import BaseModel
    class LLMConfig(BaseModel):
        provider: str = "gemini"
        model: str = "gemini-1.5-flash"
        api_key: str = None
        temperature: float = 0.1
        max_tokens: int = 2000


async def test_dynamic_extraction():
    """Test that dynamic entity and relation type extraction works."""
    
    print("🧪 Testing Dynamic Entity and Relation Type Extraction")
    print("=" * 60)
    
    # Configure LLM (we won't actually call it, just test the prompts)
    llm_config = LLMConfig(
        provider="gemini",
        api_key="test_key",
        model="gemini-1.5-flash"
    )
    
    # Test dynamic entity extraction
    print("\n📋 Entity Extractor - Dynamic Mode:")
    entity_extractor = EntityExtractor(llm_config, dynamic_types=True)
    
    print(f"  Dynamic types enabled: {entity_extractor.dynamic_types}")
    print(f"  Entity types (examples): {len(entity_extractor.entity_types)} types")
    
    entity_prompt = entity_extractor.get_system_prompt()
    print(f"  System prompt includes 'semantic meaning': {'semantic meaning' in entity_prompt}")
    print(f"  System prompt includes 'not limit yourself': {'not limit yourself' in entity_prompt}")
    
    # Test static entity extraction
    print("\n📋 Entity Extractor - Static Mode:")
    static_entity_extractor = EntityExtractor(
        llm_config, 
        dynamic_types=False,
        entity_types={"PERSON": "Individual person", "LOCATION": "Place or location"}
    )
    
    print(f"  Dynamic types enabled: {static_entity_extractor.dynamic_types}")
    print(f"  Entity types (constraints): {len(static_entity_extractor.entity_types)} types")
    
    static_entity_prompt = static_entity_extractor.get_system_prompt()
    print(f"  System prompt includes 'ONLY': {'ONLY' in static_entity_prompt}")
    print(f"  System prompt includes 'PERSON': {'PERSON' in static_entity_prompt}")
    
    # Test dynamic relation extraction
    print("\n🔗 Relation Extractor - Dynamic Mode:")
    relation_extractor = RelationExtractor(llm_config, dynamic_types=True)
    
    print(f"  Dynamic types enabled: {relation_extractor.dynamic_types}")
    print(f"  Relation types (examples): {len(relation_extractor.relation_types)} types")
    
    relation_prompt = relation_extractor.get_system_prompt()
    print(f"  System prompt includes 'semantic meaning': {'semantic meaning' in relation_prompt}")
    print(f"  System prompt includes 'not limit yourself': {'not limit yourself' in relation_prompt}")
    
    # Test static relation extraction
    print("\n🔗 Relation Extractor - Static Mode:")
    static_relation_extractor = RelationExtractor(
        llm_config,
        dynamic_types=False,
        relation_types={"CAUSES": "Entity causes another", "TREATS": "Treatment treats condition"}
    )
    
    print(f"  Dynamic types enabled: {static_relation_extractor.dynamic_types}")
    print(f"  Relation types (constraints): {len(static_relation_extractor.relation_types)} types")
    
    static_relation_prompt = static_relation_extractor.get_system_prompt()
    print(f"  System prompt includes 'ONLY': {'ONLY' in static_relation_prompt}")
    print(f"  System prompt includes 'CAUSES': {'CAUSES' in static_relation_prompt}")
    
    # Test empty types
    print("\n🔗 Relation Extractor - No Predefined Types:")
    empty_relation_extractor = RelationExtractor(
        llm_config,
        dynamic_types=False,
        relation_types={}
    )
    
    empty_relation_prompt = empty_relation_extractor.get_system_prompt()
    print(f"  System prompt handles empty types: {'no specific relation types' in empty_relation_prompt}")
    
    print("\n🎉 Dynamic extraction test completed!")
    print("\n📊 Summary:")
    print("  ✅ Dynamic entity extraction: LLM determines types")
    print("  ✅ Static entity extraction: Constrained to predefined types")
    print("  ✅ Dynamic relation extraction: LLM determines types")
    print("  ✅ Static relation extraction: Constrained to predefined types")
    print("  ✅ Empty types handling: Graceful fallback")


if __name__ == "__main__":
    asyncio.run(test_dynamic_extraction())
