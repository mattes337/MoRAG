import asyncio
import os
from dotenv import load_dotenv
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models.types import EntityType, RelationType

# Load environment variables
load_dotenv('.env')

async def debug_relation_extraction():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("No API key found")
        return
    
    # Sample text
    text = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    Tim Cook is the CEO of Apple. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."""
    
    # Create extractors
    entity_extractor = EntityExtractor(
        llm_config={
            "provider": "gemini",
            "api_key": api_key,
            "model": "gemini-1.5-flash",
            "temperature": 0.0,
            "max_tokens": 1000
        }
    )
    
    relation_extractor = RelationExtractor(
        llm_config={
            "provider": "gemini",
            "api_key": api_key,
            "model": "gemini-1.5-flash",
            "temperature": 0.0,
            "max_tokens": 1000
        }
    )
    
    try:
        # Extract entities first
        print("Extracting entities...")
        entities = await entity_extractor.extract(text)
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.name} ({entity.type}) [ID: {entity.id}]")
        
        if len(entities) == 0:
            print("No entities found, cannot test relations")
            return
        
        # Extract all relations
        print("\nExtracting all relations...")
        all_relations = await relation_extractor.extract(text, entities)
        print(f"Found {len(all_relations)} relations:")
        
        # Create entity ID to name mapping for debugging
        entity_id_to_name = {entity.id: entity.name for entity in entities}
        
        for relation in all_relations:
            source_name = entity_id_to_name.get(relation.source_entity_id, relation.source_entity_id)
            target_name = entity_id_to_name.get(relation.target_entity_id, relation.target_entity_id)
            print(f"  - {source_name} -> {target_name} ({relation.type})")
            print(f"    IDs: {relation.source_entity_id} -> {relation.target_entity_id}")
        
        # Find specific entities
        apple = next((e for e in entities if "apple" in e.name.lower() and e.type == EntityType.ORGANIZATION), None)
        steve_jobs = next((e for e in entities if "steve jobs" in e.name.lower()), None)
        tim_cook = next((e for e in entities if "tim cook" in e.name.lower()), None)
        
        print(f"\nFound entities:")
        print(f"  Apple: {apple.name if apple else 'Not found'}")
        print(f"  Steve Jobs: {steve_jobs.name if steve_jobs else 'Not found'}")
        print(f"  Tim Cook: {tim_cook.name if tim_cook else 'Not found'}")
        
        if all([apple, steve_jobs, tim_cook]):
            # Test specific entity pairs
            entity_pairs = [
                (steve_jobs.id, apple.id),
                (tim_cook.id, apple.id)
            ]
            
            print(f"\nTesting specific entity pairs: {entity_pairs}")
            
            # Debug: Check if any relations match our entity IDs
            print("\nChecking relation matches:")
            for relation in all_relations:
                for source_id, target_id in entity_pairs:
                    if (relation.source_entity_id == source_id and relation.target_entity_id == target_id):
                        print(f"  MATCH (forward): {relation.source_entity_id} -> {relation.target_entity_id} ({relation.type})")
                    elif (relation.source_entity_id == target_id and relation.target_entity_id == source_id):
                        print(f"  MATCH (reverse): {relation.source_entity_id} -> {relation.target_entity_id} ({relation.type})")
            
            specific_relations = await relation_extractor.extract_for_entity_pairs(text, entities, entity_pairs)
            print(f"\nFound {len(specific_relations)} specific relations:")
            for relation in specific_relations:
                print(f"  - {relation.source_entity_id} -> {relation.target_entity_id} ({relation.type})")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_relation_extraction())