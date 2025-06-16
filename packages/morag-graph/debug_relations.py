import asyncio
import os
from src.morag_graph.extraction.entity_extractor import EntityExtractor
from src.morag_graph.extraction.relation_extractor import RelationExtractor
from src.morag_graph.extraction.base import LLMConfig
from src.morag_graph.models.entity import Entity
from src.morag_graph.models.relation import RelationType

async def debug():
    # Set up environment
    os.environ['GEMINI_API_KEY'] = 'test'
    
    config = LLMConfig(provider='gemini', model='gemini-1.5-flash', api_key='test')
    re = RelationExtractor(config)
    
    text = 'Apple Inc. is a technology company founded by Steve Jobs. Tim Cook is the current CEO of Apple.'
    
    # Test parse_response with mock data
    print('Testing parse_response with mock data:')
    mock_response = '[{"source_entity": "Steve Jobs", "target_entity": "Apple Inc.", "relation_type": "FOUNDED", "confidence": 0.9}, {"source_entity": "Tim Cook", "target_entity": "Apple Inc.", "relation_type": "WORKS_FOR", "confidence": 0.8}]'
    relations = re.parse_response(mock_response, text)
    print(f'Parsed relations: {len(relations)}')
    for r in relations:
        print(f'  {r.source_entity_id} -> {r.target_entity_id} ({r.type})')
        print(f'    Attributes: {r.attributes}')
    
    # Create mock entities
    print('\nCreating mock entities:')
    entities = [
        Entity(id='entity_1', name='Apple Inc.', type='ORGANIZATION'),
        Entity(id='entity_2', name='Steve Jobs', type='PERSON'),
        Entity(id='entity_3', name='Tim Cook', type='PERSON')
    ]
    
    for e in entities:
        print(f'  {e.name} ({e.id}) - {e.type}')
    
    # Test extract with mock entities
    print('\nTesting extract with mock entities:')
    try:
        # This will fail due to API, but we can see the entity resolution logic
        relations = await re.extract(text, entities)
        print(f'Extracted relations: {len(relations)}')
        for r in relations:
            print(f'  {r.source_entity_id} -> {r.target_entity_id} ({r.type})')
    except Exception as e:
        print(f'Extract failed (expected): {e}')
    
    # Test entity name resolution directly
    print('\nTesting entity name resolution:')
    entity_name_to_id = {entity.name: entity.id for entity in entities}
    print('Entity name to ID mapping:')
    for name, entity_id in entity_name_to_id.items():
        print(f'  {name} -> {entity_id}')
    
    # Test _resolve_entity_id method
    print('\nTesting _resolve_entity_id:')
    test_names = ['Steve Jobs', 'Apple Inc.', 'Apple', 'steve jobs']
    for name in test_names:
        resolved_id = re._resolve_entity_id(name, entity_name_to_id)
        print(f'  "{name}" -> {resolved_id}')
    
    # Test extract_for_entity_pairs with mock data
    print('\nTesting extract_for_entity_pairs with mock data:')
    
    # Create mock relations with resolved IDs
    from src.morag_graph.models.relation import Relation
    mock_relations = [
        Relation(
            source_entity_id='entity_2',  # Steve Jobs
            target_entity_id='entity_1',  # Apple Inc.
            type=RelationType.FOUNDED,
            confidence=0.9
        ),
        Relation(
            source_entity_id='entity_3',  # Tim Cook
            target_entity_id='entity_1',  # Apple Inc.
            type=RelationType.WORKS_FOR,
            confidence=0.8
        ),
        Relation(
            source_entity_id='entity_1',  # Apple Inc.
            target_entity_id='entity_4',  # Some other entity
            type=RelationType.RELATED_TO,
            confidence=0.5
        )
    ]
    
    # Define entity pairs to filter for
    entity_pairs = [
        ('entity_2', 'entity_1'),  # Steve Jobs -> Apple Inc.
        ('entity_3', 'entity_1')   # Tim Cook -> Apple Inc.
    ]
    
    print(f'All mock relations: {len(mock_relations)}')
    for r in mock_relations:
        print(f'  {r.source_entity_id} -> {r.target_entity_id} ({r.type})')
    
    # Filter relations manually (simulating extract_for_entity_pairs logic)
    filtered_relations = []
    for relation in mock_relations:
        for source_id, target_id in entity_pairs:
            if (relation.source_entity_id == source_id and relation.target_entity_id == target_id) or \
               (relation.source_entity_id == target_id and relation.target_entity_id == source_id):
                filtered_relations.append(relation)
                break
    
    print(f'\nFiltered relations: {len(filtered_relations)}')
    for r in filtered_relations:
        print(f'  {r.source_entity_id} -> {r.target_entity_id} ({r.type})')
    
    # Check for specific relationships
    steve_jobs_founded_apple = any(
        r.source_entity_id == 'entity_2' and r.target_entity_id == 'entity_1' and r.type == RelationType.FOUNDED
        for r in filtered_relations
    )
    
    tim_cook_works_for_apple = any(
        r.source_entity_id == 'entity_3' and r.target_entity_id == 'entity_1' and r.type == RelationType.WORKS_FOR
        for r in filtered_relations
    )
    
    print(f'\nSteve Jobs founded Apple: {steve_jobs_founded_apple}')
    print(f'Tim Cook works for Apple: {tim_cook_works_for_apple}')

if __name__ == '__main__':
    asyncio.run(debug())