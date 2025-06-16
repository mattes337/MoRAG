#!/usr/bin/env python3
"""Debug script for entity pair matching in relation extraction."""

import asyncio
import os
from typing import List
from dotenv import load_dotenv

from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models import Entity, EntityType, RelationType

# Load environment variables
load_dotenv('.env')

# Sample text from the test
SAMPLE_TEXT = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
Tim Cook is the current CEO of Apple, having taken over from Steve Jobs in 2011.
Apple is known for its innovative products including the iPhone, iPad, Mac computers, and Apple Watch.
"""

async def debug_entity_pairs():
    """Debug entity pair matching."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found")
        return
    
    # Initialize extractors
    entity_extractor = EntityExtractor(
        llm_config={
            "provider": "gemini",
            "api_key": api_key,
            "model": "gemini-1.5-flash",
            "temperature": 0.1
        }
    )
    
    relation_extractor = RelationExtractor(
        llm_config={
            "provider": "gemini",
            "api_key": api_key,
            "model": "gemini-1.5-flash",
            "temperature": 0.1
        }
    )
    
    print("Extracting entities...")
    entities = await entity_extractor.extract(SAMPLE_TEXT)
    
    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.name} ({entity.type.value}) [ID: {entity.id}]")
    
    # Find specific entities
    apple = next((e for e in entities if "apple" in e.name.lower() and e.type == EntityType.ORGANIZATION), None)
    steve_jobs = next((e for e in entities if "steve jobs" in e.name.lower()), None)
    tim_cook = next((e for e in entities if "tim cook" in e.name.lower()), None)
    
    print(f"\nKey entities:")
    print(f"  Apple: {apple.name if apple else 'NOT FOUND'} [ID: {apple.id if apple else 'N/A'}]")
    print(f"  Steve Jobs: {steve_jobs.name if steve_jobs else 'NOT FOUND'} [ID: {steve_jobs.id if steve_jobs else 'N/A'}]")
    print(f"  Tim Cook: {tim_cook.name if tim_cook else 'NOT FOUND'} [ID: {tim_cook.id if tim_cook else 'N/A'}]")
    
    if not all([apple, steve_jobs, tim_cook]):
        print("\nMissing required entities, cannot continue")
        return
    
    # Create entity pairs
    entity_pairs = [
        (steve_jobs.id, apple.id),
        (tim_cook.id, apple.id)
    ]
    
    print(f"\nEntity pairs to search for:")
    for pair in entity_pairs:
        print(f"  - {pair[0]} -> {pair[1]}")
    
    # Extract all relations first
    print("\nExtracting all relations...")
    all_relations = await relation_extractor.extract(SAMPLE_TEXT, entities)
    
    print(f"\nFound {len(all_relations)} total relations:")
    for relation in all_relations:
        print(f"  - {relation.source_entity_id} --[{relation.type.value if hasattr(relation.type, 'value') else relation.type}]--> {relation.target_entity_id}")
    
    # Now extract for specific pairs
    print("\nExtracting for specific entity pairs...")
    specific_relations = await relation_extractor.extract_for_entity_pairs(SAMPLE_TEXT, entities, entity_pairs)
    
    print(f"\nFound {len(specific_relations)} specific relations:")
    for relation in specific_relations:
        print(f"  - {relation.source_entity_id} --[{relation.type.value if hasattr(relation.type, 'value') else relation.type}]--> {relation.target_entity_id}")
    
    # Debug the filtering logic
    print("\nDebugging filtering logic:")
    entity_pair_set = set(entity_pairs)
    print(f"Entity pair set: {entity_pair_set}")
    
    for relation in all_relations:
        relation_pair = (relation.source_entity_id, relation.target_entity_id)
        reverse_pair = (relation.target_entity_id, relation.source_entity_id)
        
        forward_match = relation_pair in entity_pair_set
        reverse_match = reverse_pair in entity_pair_set
        
        print(f"  Relation {relation_pair}:")
        print(f"    Forward match: {forward_match}")
        print(f"    Reverse match: {reverse_match}")
        print(f"    Should include: {forward_match or reverse_match}")

if __name__ == "__main__":
    asyncio.run(debug_entity_pairs())