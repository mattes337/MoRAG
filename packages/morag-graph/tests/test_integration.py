#!/usr/bin/env python3
"""Integration tests for morag-graph package.

These tests demonstrate the complete workflow from text input to knowledge graph
construction, including entity extraction, relation extraction, and storage.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models import Entity, Relation, Graph, EntityType, RelationType
from morag_graph.storage import JsonStorage


# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Apple Inc. Overview",
        "content": """
        Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
        The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
        Tim Cook is the current CEO of Apple, having taken over from Steve Jobs in 2011.
        Apple is known for its innovative products including the iPhone, iPad, Mac computers, and Apple Watch.
        The company's headquarters, Apple Park, opened in 2017 and can accommodate 12,000 employees.
        """
    },
    {
        "id": "doc_2",
        "title": "Microsoft Corporation",
        "content": """
        Microsoft Corporation is an American multinational technology corporation founded by Bill Gates and Paul Allen in 1975.
        The company is headquartered in Redmond, Washington. Satya Nadella serves as the current CEO since 2014.
        Microsoft is best known for its Windows operating system, Office productivity suite, and Azure cloud platform.
        The company has been a major competitor to Apple in various technology sectors.
        """
    },
    {
        "id": "doc_3",
        "title": "Tech Industry Partnerships",
        "content": """
        In recent years, technology companies have formed strategic partnerships to enhance their offerings.
        Apple and Microsoft have collaborated on various projects, including Office applications for Mac and iPad.
        Both companies compete in the cloud computing space, with Apple's iCloud and Microsoft's Azure.
        Steve Jobs and Bill Gates had a complex relationship, sometimes collaborating and sometimes competing.
        """
    }
]


class IntegrationTestSuite:
    """Integration test suite for complete workflow testing."""
    
    def __init__(self, api_key: str):
        """Initialize the test suite.
        
        Args:
            api_key: Gemini API key for LLM calls
        """
        self.api_key = api_key
        self.entity_extractor = None
        self.relation_extractor = None
        self.storage = None
        self.temp_dir = None
        
    async def setup(self):
        """Set up extractors and storage for testing."""
        # Create temporary directory for storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize extractors
        self.entity_extractor = EntityExtractor(
            llm_config={
                "provider": "gemini",
                "api_key": self.api_key,
                "model": "gemini-1.5-flash",
                "temperature": 0.1
            }
        )
        
        self.relation_extractor = RelationExtractor(
            llm_config={
                "provider": "gemini",
                "api_key": self.api_key,
                "model": "gemini-1.5-flash",
                "temperature": 0.1
            }
        )
        
        # Initialize storage
        storage_path = Path(self.temp_dir) / "test_graph.json"
        self.storage = JsonStorage(str(storage_path))
        await self.storage.connect()
        
    async def teardown(self):
        """Clean up resources after testing."""
        if self.storage:
            await self.storage.disconnect()
        
        # Clean up temporary directory
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def extract_entities_from_documents(self) -> List[Entity]:
        """Extract entities from all sample documents.
        
        Returns:
            List of extracted entities
        """
        all_entities = []
        
        for doc in SAMPLE_DOCUMENTS:
            print(f"\n📄 Extracting entities from: {doc['title']}")
            
            entities = await self.entity_extractor.extract(
                text=doc["content"],
                source_doc_id=doc["id"],
                additional_context=f"Document title: {doc['title']}"
            )
            
            print(f"   Found {len(entities)} entities")
            for entity in entities:
                print(f"   • {entity.name} ({entity.type.value})")
            
            all_entities.extend(entities)
        
        return all_entities
    
    async def extract_relations_from_documents(self, entities: List[Entity]) -> List[Relation]:
        """Extract relations from all sample documents.
        
        Args:
            entities: Previously extracted entities for context
            
        Returns:
            List of extracted relations
        """
        all_relations = []
        
        for doc in SAMPLE_DOCUMENTS:
            print(f"\n🔗 Extracting relations from: {doc['title']}")
            
            # Get entities from this document
            doc_entities = [e for e in entities if e.source_doc_id == doc["id"]]
            
            relations = await self.relation_extractor.extract(
                text=doc["content"],
                entities=doc_entities,
                source_doc_id=doc["id"],
                additional_context=f"Document title: {doc['title']}"
            )
            
            print(f"   Found {len(relations)} relations")
            for relation in relations:
                source_entity = next((e for e in entities if e.id == relation.source_entity_id), None)
                target_entity = next((e for e in entities if e.id == relation.target_entity_id), None)
                if source_entity and target_entity:
                    print(f"   • {source_entity.name} --[{relation.type.value}]--> {target_entity.name}")
            
            all_relations.extend(relations)
        
        return all_relations
    
    async def build_knowledge_graph(self, entities: List[Entity], relations: List[Relation]) -> Graph:
        """Build a knowledge graph from extracted entities and relations.
        
        Args:
            entities: List of entities
            relations: List of relations
            
        Returns:
            Constructed knowledge graph
        """
        print("\n🕸️  Building knowledge graph...")
        
        graph = Graph()
        
        # Add entities
        for entity in entities:
            graph.add_entity(entity)
        
        # Add relations
        for relation in relations:
            graph.add_relation(relation)
        
        print(f"   Graph contains {len(graph.entities)} entities and {len(graph.relations)} relations")
        
        return graph
    
    async def store_graph_data(self, graph: Graph):
        """Store graph data in storage backend.
        
        Args:
            graph: Knowledge graph to store
        """
        print("\n💾 Storing graph data...")
        
        # Store entities
        for entity in graph.entities.values():
            await self.storage.store_entity(entity)
        
        # Store relations
        for relation in graph.relations.values():
            await self.storage.store_relation(relation)
        
        # Store complete graph
        await self.storage.store_graph(graph)
        
        print("   Graph data stored successfully")
    
    async def analyze_graph(self, graph: Graph) -> Dict:
        """Analyze the constructed knowledge graph.
        
        Args:
            graph: Knowledge graph to analyze
            
        Returns:
            Analysis results
        """
        print("\n📊 Analyzing knowledge graph...")
        
        analysis = {
            "total_entities": len(graph.entities),
            "total_relations": len(graph.relations),
            "entity_types": {},
            "relation_types": {},
            "most_connected_entities": [],
            "entity_pairs_with_multiple_relations": []
        }
        
        # Count entity types
        for entity in graph.entities.values():
            entity_type = entity.type.value
            analysis["entity_types"][entity_type] = analysis["entity_types"].get(entity_type, 0) + 1
        
        # Count relation types
        for relation in graph.relations.values():
            relation_type = relation.type.value
            analysis["relation_types"][relation_type] = analysis["relation_types"].get(relation_type, 0) + 1
        
        # Find most connected entities
        entity_connections = {}
        for relation in graph.relations.values():
            entity_connections[relation.source_entity_id] = entity_connections.get(relation.source_entity_id, 0) + 1
            entity_connections[relation.target_entity_id] = entity_connections.get(relation.target_entity_id, 0) + 1
        
        # Sort by connection count
        sorted_connections = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)
        
        for entity_id, connection_count in sorted_connections[:5]:  # Top 5
            entity = graph.entities.get(entity_id)
            if entity:
                analysis["most_connected_entities"].append({
                    "name": entity.name,
                    "type": entity.type.value,
                    "connections": connection_count
                })
        
        # Find entity pairs with multiple relations
        entity_pair_relations = {}
        for relation in graph.relations.values():
            pair_key = tuple(sorted([relation.source_entity_id, relation.target_entity_id]))
            if pair_key not in entity_pair_relations:
                entity_pair_relations[pair_key] = []
            entity_pair_relations[pair_key].append(relation)
        
        for pair_key, relations in entity_pair_relations.items():
            if len(relations) > 1:
                entity1 = graph.entities.get(pair_key[0])
                entity2 = graph.entities.get(pair_key[1])
                if entity1 and entity2:
                    analysis["entity_pairs_with_multiple_relations"].append({
                        "entity1": entity1.name,
                        "entity2": entity2.name,
                        "relation_count": len(relations),
                        "relation_types": [r.type.value for r in relations]
                    })
        
        # Print analysis results
        print(f"   Total entities: {analysis['total_entities']}")
        print(f"   Total relations: {analysis['total_relations']}")
        print("   Entity types:")
        for entity_type, count in analysis["entity_types"].items():
            print(f"     • {entity_type}: {count}")
        print("   Relation types:")
        for relation_type, count in analysis["relation_types"].items():
            print(f"     • {relation_type}: {count}")
        print("   Most connected entities:")
        for entity_info in analysis["most_connected_entities"]:
            print(f"     • {entity_info['name']} ({entity_info['type']}): {entity_info['connections']} connections")
        
        return analysis
    
    async def test_graph_queries(self, graph: Graph):
        """Test various graph query operations.
        
        Args:
            graph: Knowledge graph to query
        """
        print("\n🔍 Testing graph queries...")
        
        # Test 1: Find all companies
        companies = [e for e in graph.entities.values() if e.type == EntityType.ORGANIZATION]
        print(f"   Found {len(companies)} companies:")
        for company in companies:
            print(f"     • {company.name}")
        
        # Test 2: Find all people and their roles
        people = [e for e in graph.entities.values() if e.type == EntityType.PERSON]
        print(f"   Found {len(people)} people:")
        for person in people:
            # Find relations where this person is involved
            person_relations = graph.get_entity_relations(person.id)
            roles = []
            for rel in person_relations:
                if rel.type == RelationType.WORKS_FOR or rel.type == RelationType.LEADS:
                    if rel.source_entity_id == person.id:
                        target_entity = graph.entities.get(rel.target_entity_id)
                        if target_entity:
                            roles.append(f"{rel.type.value} {target_entity.name}")
            print(f"     • {person.name}: {', '.join(roles) if roles else 'No specific roles found'}")
        
        # Test 3: Find competitors
        competitor_relations = [r for r in graph.relations.values() if r.type == RelationType.COMPETES_WITH]
        print(f"   Found {len(competitor_relations)} competitor relationships:")
        for rel in competitor_relations:
            source_entity = graph.entities.get(rel.source_entity_id)
            target_entity = graph.entities.get(rel.target_entity_id)
            if source_entity and target_entity:
                print(f"     • {source_entity.name} competes with {target_entity.name}")
        
        # Test 4: Find entity neighbors
        if companies:
            test_company = companies[0]
            neighbors = graph.get_neighbors(test_company.id)
            print(f"   Neighbors of {test_company.name}:")
            for neighbor_id in neighbors:
                neighbor = graph.entities.get(neighbor_id)
                if neighbor:
                    print(f"     • {neighbor.name} ({neighbor.type.value})")
    
    async def run_complete_workflow(self) -> Dict:
        """Run the complete integration test workflow.
        
        Returns:
            Test results and analysis
        """
        print("🚀 Starting complete integration test workflow...")
        print("" + "="*60)
        
        try:
            # Step 1: Extract entities
            entities = await self.extract_entities_from_documents()
            assert len(entities) > 0, "No entities were extracted"
            
            # Step 2: Extract relations
            relations = await self.extract_relations_from_documents(entities)
            # Note: Relations might be empty if entities don't have clear relationships
            
            # Step 3: Build knowledge graph
            graph = await self.build_knowledge_graph(entities, relations)
            assert len(graph.entities) > 0, "Knowledge graph has no entities"
            
            # Step 4: Store graph data
            await self.store_graph_data(graph)
            
            # Step 5: Analyze graph
            analysis = await self.analyze_graph(graph)
            
            # Step 6: Test graph queries
            await self.test_graph_queries(graph)
            
            # Step 7: Test storage retrieval
            print("\n🔄 Testing storage retrieval...")
            stored_entities = await self.storage.get_all_entities()
            stored_relations = await self.storage.get_all_relations()
            stored_graph = await self.storage.get_graph()
            
            print(f"   Retrieved {len(stored_entities)} entities from storage")
            print(f"   Retrieved {len(stored_relations)} relations from storage")
            print(f"   Retrieved graph with {len(stored_graph.entities)} entities and {len(stored_graph.relations)} relations")
            
            # Verify data consistency
            assert len(stored_entities) == len(entities), "Entity count mismatch in storage"
            assert len(stored_relations) == len(relations), "Relation count mismatch in storage"
            assert len(stored_graph.entities) == len(graph.entities), "Graph entity count mismatch"
            assert len(stored_graph.relations) == len(graph.relations), "Graph relation count mismatch"
            
            print("\n✅ Integration test completed successfully!")
            print("" + "="*60)
            
            return {
                "success": True,
                "entities_extracted": len(entities),
                "relations_extracted": len(relations),
                "graph_analysis": analysis,
                "storage_verified": True
            }
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            print("" + "="*60)
            return {
                "success": False,
                "error": str(e)
            }


# Pytest fixtures and tests
@pytest.fixture
def api_key():
    """Get Gemini API key from environment."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return key


@pytest.fixture
async def integration_suite(api_key):
    """Create and setup integration test suite."""
    suite = IntegrationTestSuite(api_key)
    await suite.setup()
    yield suite
    await suite.teardown()


@pytest.mark.asyncio
async def test_complete_integration_workflow(integration_suite):
    """Test the complete integration workflow.
    
    This test demonstrates the full pipeline from text to knowledge graph.
    """
    results = await integration_suite.run_complete_workflow()
    
    # Verify results
    assert results["success"], f"Integration test failed: {results.get('error', 'Unknown error')}"
    assert results["entities_extracted"] > 0, "No entities were extracted"
    assert results["storage_verified"], "Storage verification failed"
    
    # Print summary
    print("\n📋 Integration Test Summary:")
    print(f"   • Entities extracted: {results['entities_extracted']}")
    print(f"   • Relations extracted: {results['relations_extracted']}")
    print(f"   • Storage verified: {results['storage_verified']}")
    print(f"   • Test result: {'✅ PASSED' if results['success'] else '❌ FAILED'}")


@pytest.mark.asyncio
async def test_entity_extraction_integration(integration_suite):
    """Test entity extraction in isolation."""
    entities = await integration_suite.extract_entities_from_documents()
    
    assert len(entities) > 0, "No entities were extracted"
    
    # Verify entity properties
    for entity in entities:
        assert entity.name, "Entity name is empty"
        assert entity.type, "Entity type is not set"
        assert entity.source_doc_id, "Entity source document ID is not set"
        assert 0 <= entity.confidence <= 1, "Entity confidence is out of range"
    
    print(f"\n✅ Entity extraction test passed: {len(entities)} entities extracted")


@pytest.mark.asyncio
async def test_relation_extraction_integration(integration_suite):
    """Test relation extraction in isolation."""
    # First extract entities
    entities = await integration_suite.extract_entities_from_documents()
    assert len(entities) > 0, "No entities available for relation extraction"
    
    # Then extract relations
    relations = await integration_suite.extract_relations_from_documents(entities)
    
    # Note: Relations might be empty if no clear relationships are found
    # This is acceptable behavior
    
    # Verify relation properties if any relations were found
    for relation in relations:
        assert relation.source_entity_id, "Relation source entity ID is empty"
        assert relation.target_entity_id, "Relation target entity ID is empty"
        assert relation.type, "Relation type is not set"
        assert 0 <= relation.confidence <= 1, "Relation confidence is out of range"
        assert relation.source_entity_id != relation.target_entity_id, "Self-referencing relation"
    
    print(f"\n✅ Relation extraction test passed: {len(relations)} relations extracted")


@pytest.mark.asyncio
async def test_storage_integration(integration_suite):
    """Test storage operations in isolation."""
    # Create test entities and relations
    test_entity = Entity(
        name="Test Company",
        type=EntityType.ORGANIZATION,
        source_doc_id="test_doc",
        confidence=0.9
    )
    
    test_relation = Relation(
        source_entity_id=test_entity.id,
        target_entity_id=test_entity.id,  # Self-reference for testing
        type=RelationType.CUSTOM,
        source_doc_id="test_doc",
        confidence=0.8
    )
    
    # Test storage operations
    await integration_suite.storage.store_entity(test_entity)
    await integration_suite.storage.store_relation(test_relation)
    
    # Test retrieval
    retrieved_entity = await integration_suite.storage.get_entity(test_entity.id)
    retrieved_relation = await integration_suite.storage.get_relation(test_relation.id)
    
    assert retrieved_entity is not None, "Failed to retrieve stored entity"
    assert retrieved_relation is not None, "Failed to retrieve stored relation"
    assert retrieved_entity.name == test_entity.name, "Entity name mismatch"
    assert retrieved_relation.type == test_relation.type, "Relation type mismatch"
    
    print("\n✅ Storage integration test passed")


if __name__ == "__main__":
    """Run integration tests directly."""
    import sys
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    async def run_tests():
        """Run all integration tests."""
        suite = IntegrationTestSuite(api_key)
        await suite.setup()
        
        try:
            results = await suite.run_complete_workflow()
            if results["success"]:
                print("\n🎉 All integration tests passed!")
                return 0
            else:
                print(f"\n❌ Integration tests failed: {results.get('error', 'Unknown error')}")
                return 1
        finally:
            await suite.teardown()
    
    # Run the tests
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)