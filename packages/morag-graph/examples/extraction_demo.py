#!/usr/bin/env python3
"""Demonstration script for entity and relation extraction using morag-graph.

This script shows how to:
1. Extract entities from text using LLM
2. Extract relations between entities
3. Store the results in different storage backends
4. Query and visualize the extracted knowledge graph

Usage:
    python extraction_demo.py --api-key YOUR_GEMINI_API_KEY
    
Or set GEMINI_API_KEY environment variable and run:
    python extraction_demo.py
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models import Entity, Relation, Graph
from morag_graph.models.types import EntityType, RelationType
from morag_graph.storage import JsonStorage, JsonConfig

# Load environment variables
load_dotenv()

# Sample documents for demonstration
SAMPLE_DOCUMENTS = {
    "apple_company": {
        "title": "Apple Inc. Company Overview",
        "content": """
        Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
        The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, to develop 
        and sell Wozniak's Apple I personal computer. It was incorporated by Jobs and Wozniak as Apple Computer 
        Company on January 3, 1977, and sales of its computers, including the Apple II, grew quickly.
        
        Tim Cook is the current CEO of Apple, having taken over from Steve Jobs in 2011. Under Cook's leadership, 
        Apple has continued to innovate with products like the iPhone, iPad, and Apple Watch. The company is 
        known for its design philosophy and has become one of the world's most valuable companies.
        
        Apple's headquarters, known as Apple Park, is located in Cupertino, California. The company employs 
        over 150,000 people worldwide and has retail stores in many countries.
        """
    },
    "eiffel_tower": {
        "title": "The Eiffel Tower",
        "content": """
        The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named 
        after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. 
        
        Originally criticized by some of France's leading artists and intellectuals for its design, it has 
        become a global cultural icon of France and one of the most recognizable structures in the world. 
        The Eiffel Tower is the most visited monument in the world, with 6.91 million visitors in 2015.
        
        The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the 
        tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
        During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest 
        man-made structure in the world, a title it held for 41 years until the Chrysler Building in 
        New York City was finished in 1930.
        """
    },
    "python_programming": {
        "title": "Python Programming Language",
        "content": """
        Python is a high-level, general-purpose programming language. Its design philosophy emphasizes 
        code readability with the use of significant indentation. Python is dynamically typed and 
        garbage-collected. It supports multiple programming paradigms, including structured, 
        object-oriented and functional programming.
        
        Python was created by Guido van Rossum and first released in 1991. Van Rossum served as Python's 
        lead developer until 2018, when he stepped down as the "Benevolent Dictator for Life" (BDFL). 
        The Python Software Foundation now oversees the development of Python.
        
        Python's syntax allows programmers to express concepts in fewer lines of code than possible in 
        languages such as C++ or Java. The language's core philosophy is summarized in the document 
        "The Zen of Python" (PEP 20), which includes aphorisms such as "Beautiful is better than ugly" 
        and "Simple is better than complex".
        
        Python is widely used in web development, data analysis, artificial intelligence, scientific 
        computing, and automation. Popular frameworks include Django for web development, NumPy and 
        Pandas for data analysis, and TensorFlow and PyTorch for machine learning.
        """
    }
}


class ExtractionDemo:
    """Demonstration class for entity and relation extraction."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """Initialize the demo with API configuration.
        
        Args:
            api_key: Google Gemini API key
            model: LLM model to use
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize extractors
        llm_config = {
            "provider": "gemini",
            "api_key": api_key,
            "model": model,
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 2000
        }
        
        self.entity_extractor = EntityExtractor(llm_config=llm_config)
        self.relation_extractor = RelationExtractor(llm_config=llm_config)
        
        # Initialize storage
        storage_config = JsonConfig(
            storage_path="./demo_graph_data",
            auto_save=True,
            backup_count=3
        )
        self.storage = JsonStorage(storage_config)
        
        # Initialize graph
        self.graph = Graph()
    
    async def extract_from_document(self, doc_id: str, title: str, content: str) -> Dict[str, Any]:
        """Extract entities and relations from a single document.
        
        Args:
            doc_id: Document identifier
            title: Document title
            content: Document content
            
        Returns:
            Dictionary containing extraction results
        """
        print(f"\n📄 Processing document: {title}")
        print(f"📝 Content length: {len(content)} characters")
        
        # Extract entities
        print("🔍 Extracting entities...")
        entities = await self.entity_extractor.extract(
            text=content,
            doc_id=doc_id,
            context=f"Document title: {title}"
        )
        
        print(f"✅ Found {len(entities)} entities:")
        for entity in entities:
            print(f"  • {entity.name} ({str(entity.type)}) - confidence: {entity.confidence:.2f}")
        
        # Extract relations
        print("🔗 Extracting relations...")
        relations = await self.relation_extractor.extract(
            text=content,
            entities=entities,
            doc_id=doc_id,
            context=f"Document title: {title}"
        )
        
        print(f"✅ Found {len(relations)} relations:")
        for relation in relations:
            source_entity = next((e for e in entities if e.id == relation.source_entity_id), None)
            target_entity = next((e for e in entities if e.id == relation.target_entity_id), None)
            
            if source_entity and target_entity:
                print(f"  • {source_entity.name} --[{str(relation.type)}]--> {target_entity.name} (confidence: {relation.confidence:.2f})")
        
        # Add to graph
        for entity in entities:
            self.graph.add_entity(entity)
        
        for relation in relations:
            self.graph.add_relation(relation)
        
        return {
            "doc_id": doc_id,
            "title": title,
            "entities": entities,
            "relations": relations,
            "entity_count": len(entities),
            "relation_count": len(relations)
        }
    
    async def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all sample documents.
        
        Returns:
            List of extraction results for each document
        """
        print("🚀 Starting entity and relation extraction demo...")
        print(f"🤖 Using model: {self.model}")
        
        results = []
        
        for doc_id, doc_data in SAMPLE_DOCUMENTS.items():
            result = await self.extract_from_document(
                doc_id=doc_id,
                title=doc_data["title"],
                content=doc_data["content"]
            )
            results.append(result)
        
        return results
    
    async def save_to_storage(self) -> None:
        """Save the extracted graph to storage."""
        print("\n💾 Saving graph to storage...")
        
        # Connect to storage
        await self.storage.connect()
        
        try:
            # Store the graph
            await self.storage.store_graph(self.graph)
            
            # Get statistics
            stats = await self.storage.get_statistics()
            print(f"✅ Saved graph with {stats['entity_count']} entities and {stats['relation_count']} relations")
            print(f"📁 Storage location: {stats['storage_path']}")
            
        finally:
            await self.storage.disconnect()
    
    def analyze_graph(self) -> Dict[str, Any]:
        """Analyze the extracted graph.
        
        Returns:
            Dictionary containing graph analysis
        """
        print("\n📊 Analyzing extracted graph...")
        
        # Count entities by type
        entity_types = {}
        for entity in self.graph.entities.values():
            entity_type = str(entity.type)
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Count relations by type
        relation_types = {}
        for relation in self.graph.relations.values():
            relation_type = str(relation.type)
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
        
        # Find most connected entities
        entity_connections = {}
        for relation in self.graph.relations.values():
            entity_connections[relation.source_entity_id] = entity_connections.get(relation.source_entity_id, 0) + 1
            entity_connections[relation.target_entity_id] = entity_connections.get(relation.target_entity_id, 0) + 1
        
        most_connected = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:5]
        
        analysis = {
            "total_entities": len(self.graph.entities),
            "total_relations": len(self.graph.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "most_connected_entities": [
                {
                    "entity_id": entity_id,
                    "entity_name": self.graph.entities[entity_id].name,
                    "connections": count
                }
                for entity_id, count in most_connected
                if entity_id in self.graph.entities
            ]
        }
        
        # Print analysis
        print(f"📈 Total entities: {analysis['total_entities']}")
        print(f"📈 Total relations: {analysis['total_relations']}")
        
        print("\n🏷️  Entity types:")
        for entity_type, count in sorted(analysis['entity_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {entity_type}: {count}")
        
        print("\n🔗 Relation types:")
        for relation_type, count in sorted(analysis['relation_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {relation_type}: {count}")
        
        print("\n⭐ Most connected entities:")
        for entity_info in analysis['most_connected_entities']:
            print(f"  • {entity_info['entity_name']}: {entity_info['connections']} connections")
        
        return analysis
    
    def demonstrate_queries(self) -> None:
        """Demonstrate graph queries."""
        print("\n🔍 Demonstrating graph queries...")
        
        # Find specific entities
        apple_entities = [e for e in self.graph.entities.values() if "apple" in e.name.lower()]
        if apple_entities:
            apple = apple_entities[0]
            print(f"\n🍎 Found Apple entity: {apple.name}")
            
            # Get Apple's relations
            apple_relations = self.graph.get_entity_relations(apple.id)
            print(f"📊 Apple has {len(apple_relations)} relations:")
            
            for relation in apple_relations[:5]:  # Show first 5
                if relation.source_entity_id == apple.id:
                    target = self.graph.entities.get(relation.target_entity_id)
                    if target:
                        print(f"  • Apple --[{str(relation.type)}]--> {target.name}")
                else:
                    source = self.graph.entities.get(relation.source_entity_id)
                    if source:
                        print(f"  • {source.name} --[{str(relation.type)}]--> Apple")
            
            # Get Apple's neighbors
            neighbors = self.graph.get_neighbors(apple.id)
            print(f"\n👥 Apple's neighbors ({len(neighbors)}):")
            for neighbor in neighbors[:5]:  # Show first 5
                print(f"  • {neighbor.name} ({neighbor.type.value})")
        
        # Find people entities
        people = [e for e in self.graph.entities.values() if e.type == EntityType.PERSON]
        if len(people) >= 2:
            person1, person2 = people[0], people[1]
            print(f"\n🔍 Looking for paths between {person1.name} and {person2.name}...")
            
            # This is a simple implementation - in a real graph database, 
            # path finding would be more efficient
            common_entities = set()
            person1_neighbors = set(n.id for n in self.graph.get_neighbors(person1.id))
            person2_neighbors = set(n.id for n in self.graph.get_neighbors(person2.id))
            common_entities = person1_neighbors.intersection(person2_neighbors)
            
            if common_entities:
                print(f"🔗 Found {len(common_entities)} common connections:")
                for entity_id in list(common_entities)[:3]:  # Show first 3
                    entity = self.graph.entities.get(entity_id)
                    if entity:
                        print(f"  • {entity.name} ({str(entity.type)})")
            else:
                print("❌ No direct common connections found")
    
    def export_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Export results to JSON files.
        
        Args:
            results: Extraction results
            analysis: Graph analysis
        """
        print("\n📤 Exporting results...")
        
        # Create output directory
        output_dir = Path("./demo_output")
        output_dir.mkdir(exist_ok=True)
        
        # Export extraction results
        results_file = output_dir / "extraction_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert entities and relations to dictionaries for JSON serialization
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                serializable_result['entities'] = [entity.model_dump() for entity in result['entities']]
                serializable_result['relations'] = [relation.model_dump() for relation in result['relations']]
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Export graph analysis
        analysis_file = output_dir / "graph_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Export graph data
        graph_file = output_dir / "graph_data.json"
        with open(graph_file, 'w', encoding='utf-8') as f:
            graph_data = {
                'entities': [entity.model_dump() for entity in self.graph.entities.values()],
                'relations': [relation.model_dump() for relation in self.graph.relations.values()]
            }
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results exported to {output_dir}:")
        print(f"  • {results_file.name} - Detailed extraction results")
        print(f"  • {analysis_file.name} - Graph analysis")
        print(f"  • {graph_file.name} - Complete graph data")
    
    async def run_demo(self) -> None:
        """Run the complete demonstration."""
        try:
            # Process all documents
            results = await self.process_all_documents()
            
            # Analyze the graph
            analysis = self.analyze_graph()
            
            # Demonstrate queries
            self.demonstrate_queries()
            
            # Save to storage
            await self.save_to_storage()
            
            # Export results
            self.export_results(results, analysis)
            
            print("\n🎉 Demo completed successfully!")
            print("\n📋 Summary:")
            print(f"  • Processed {len(results)} documents")
            print(f"  • Extracted {analysis['total_entities']} entities")
            print(f"  • Extracted {analysis['total_relations']} relations")
            print(f"  • Found {len(analysis['entity_types'])} entity types")
            print(f"  • Found {len(analysis['relation_types'])} relation types")
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            raise


def main():
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description="Entity and Relation Extraction Demo")
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model to use (default: gemini-1.5-flash)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: Gemini API key is required.")
        print("   Set it via --api-key argument or GEMINI_API_KEY environment variable.")
        return 1
    
    # Run the demo
    demo = ExtractionDemo(api_key=api_key, model=args.model)
    
    try:
        asyncio.run(demo.run_demo())
        return 0
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())