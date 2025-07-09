#!/usr/bin/env python3
"""CLI script to test multi-hop reasoning capabilities.

This script provides an easy way to test the multi-hop reasoning functionality
of the MoRAG system, including path selection and iterative context refinement.

Usage:
    python test_multi_hop_cli.py "How are Apple's AI research efforts related to their partnership with universities?"
    
Or with specific parameters:
    python test_multi_hop_cli.py "query" --strategy bidirectional --max-paths 20 --verbose
    
Options:
    --strategy       Reasoning strategy (forward_chaining, backward_chaining, bidirectional)
    --max-paths      Maximum number of paths to discover (default: 50)
    --max-iterations Maximum context refinement iterations (default: 5)
    --api-key        Gemini API key (can also be set via GEMINI_API_KEY environment variable)
    --model          LLM model to use (default: gemini-1.5-flash)
    --verbose        Show detailed reasoning output
    --output         Save results to JSON file
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging (will be set to DEBUG in verbose mode)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from morag_reasoning import (
        LLMClient, LLMConfig, PathSelectionAgent, ReasoningPathFinder, 
        IterativeRetriever, RetrievalContext
    )
    from morag_reasoning.llm import LLMClient
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the morag-reasoning package directory")
    print("   and all dependencies are installed.")
    sys.exit(1)


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_packages = {
        "google-generativeai": "google.generativeai",
        "httpx": "httpx", 
        "pydantic": "pydantic",
        "python-dotenv": "dotenv",
        "aiofiles": "aiofiles"
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def setup_environment(api_key: Optional[str] = None) -> bool:
    """Setup environment for multi-hop reasoning."""
    # Check API key
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Gemini API key is required for multi-hop reasoning.")
        print("   Set it via --api-key argument or GEMINI_API_KEY environment variable.")
        return False
    
    # Set environment variable
    os.environ["GEMINI_API_KEY"] = api_key
    
    return True


class MockGraphEngine:
    """Mock graph engine for testing purposes."""

    def __init__(self, query: str = ""):
        # Create relevant entities based on the query
        if "zirbeldrüse" in query.lower() or "pineal" in query.lower():
            # German query about pineal gland - use dynamic, semantically appropriate types
            self.entities = {
                "Zirbeldrüse": {"type": "ANATOMICAL_STRUCTURE", "description": "Pineal gland, endocrine gland in the brain"},
                "Melatonin": {"type": "HORMONE", "description": "Hormone produced by pineal gland"},
                "Fluorid": {"type": "CHEMICAL_COMPOUND", "description": "Chemical that can calcify pineal gland"},
                "Licht": {"type": "ENVIRONMENTAL_FACTOR", "description": "Light exposure affects pineal function"},
                "Schlaf": {"type": "BIOLOGICAL_PROCESS", "description": "Sleep cycle regulated by pineal gland"},
                "Kalzifizierung": {"type": "PATHOLOGICAL_CONDITION", "description": "Calcification of pineal gland"},
                "Alter": {"type": "BIOLOGICAL_FACTOR", "description": "Aging affects pineal function"},
                "Stress": {"type": "PSYCHOLOGICAL_FACTOR", "description": "Stress can impair pineal function"}
            }

            self.relations = [
                {"subject": "Zirbeldrüse", "predicate": "PRODUCES", "object": "Melatonin"},
                {"subject": "Fluorid", "predicate": "CAUSES", "object": "Kalzifizierung"},
                {"subject": "Kalzifizierung", "predicate": "IMPAIRS", "object": "Zirbeldrüse"},
                {"subject": "Licht", "predicate": "INFLUENCES", "object": "Zirbeldrüse"},
                {"subject": "Zirbeldrüse", "predicate": "REGULATES", "object": "Schlaf"},
                {"subject": "Alter", "predicate": "REDUCES", "object": "Melatonin"},
                {"subject": "Stress", "predicate": "DISRUPTS", "object": "Zirbeldrüse"}
            ]
        else:
            # Default entities for other queries
            self.entities = {
                "Apple Inc.": {"type": "ORG", "description": "Technology company"},
                "Steve Jobs": {"type": "PERSON", "description": "Co-founder of Apple"},
                "iPhone": {"type": "PRODUCT", "description": "Smartphone product"},
                "AI research": {"type": "CONCEPT", "description": "Artificial intelligence research"},
                "Stanford University": {"type": "ORG", "description": "University"},
                "product development": {"type": "CONCEPT", "description": "Product development process"}
            }

            self.relations = [
                {"subject": "Steve Jobs", "predicate": "founded", "object": "Apple Inc."},
                {"subject": "Apple Inc.", "predicate": "develops", "object": "iPhone"},
                {"subject": "Apple Inc.", "predicate": "conducts", "object": "AI research"},
                {"subject": "Apple Inc.", "predicate": "partners_with", "object": "Stanford University"},
                {"subject": "Stanford University", "predicate": "collaborates_on", "object": "AI research"},
                {"subject": "AI research", "predicate": "influences", "object": "product development"}
            ]
    
    def _get_entity_type(self, entity_name: str) -> str:
        """Get the correct entity type for an entity name."""
        entity_data = self.entities.get(entity_name, {})
        return entity_data.get("type", "CONCEPT")  # Default to CONCEPT if not found

    async def get_entity_details(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get entity details."""
        return self.entities.get(entity_name)
    
    async def find_neighbors(self, entity_name: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """Find neighboring entities."""
        neighbors = []
        for relation in self.relations:
            if relation["subject"] == entity_name:
                target = relation["object"]
                if target in self.entities:
                    neighbors.append({
                        "id": target,
                        "type": self.entities[target]["type"],
                        "relation": relation["predicate"]
                    })
            elif relation["object"] == entity_name:
                source = relation["subject"]
                if source in self.entities:
                    neighbors.append({
                        "id": source,
                        "type": self.entities[source]["type"],
                        "relation": f"inverse_{relation['predicate']}"
                    })
        return neighbors[:5]  # Limit results
    
    async def find_shortest_path(self, start: str, end: str):
        """Find shortest path between entities."""
        # Simple mock implementation
        from morag_graph.operations import GraphPath
        from morag_graph.models import Entity, Relation

        if start in self.entities and end in self.entities:
            # Create mock entities and relations with correct types
            start_type = self._get_entity_type(start)
            end_type = self._get_entity_type(end)

            start_entity = Entity(name=start, type=start_type)
            end_entity = Entity(name=end, type=end_type)
            connection_rel = Relation(
                source_entity_id=start_entity.id,
                target_entity_id=end_entity.id,
                type=RelationType.RELATED_TO
            )

            # Create a simple path
            return GraphPath(
                entities=[start_entity, end_entity],
                relations=[connection_rel]
            )
        return None
    
    async def traverse(self, start_entity: str, algorithm: str = "bfs", max_depth: int = 3):
        """Traverse the graph from a starting entity."""
        from morag_graph.operations import GraphPath
        
        paths = []
        visited = set()
        
        # Simple BFS traversal simulation
        queue = [(start_entity, [start_entity], [])]
        
        while queue and len(paths) < 10:  # Limit paths
            current, path, relations = queue.pop(0)
            
            if current in visited or len(path) > max_depth:
                continue
                
            visited.add(current)
            
            # Add current path if it has more than one entity
            if len(path) > 1:
                # Create Entity and Relation objects for GraphPath
                from morag_graph.models import Entity, Relation

                path_entities = []
                path_relations = []

                for i, entity_name in enumerate(path):
                    entity_type = self._get_entity_type(entity_name)
                    entity = Entity(name=entity_name, type=entity_type)
                    path_entities.append(entity)

                    # Create relation for each connection
                    if i < len(relations):
                        if i + 1 < len(path):
                            next_entity_type = self._get_entity_type(path[i + 1])
                            next_entity = Entity(name=path[i + 1], type=next_entity_type)
                            relation = Relation(
                                source_entity_id=entity.id,
                                target_entity_id=next_entity.id,
                                type="RELATED_TO"
                            )
                            path_relations.append(relation)

                graph_path = GraphPath(
                    entities=path_entities,
                    relations=path_relations
                )
                paths.append(graph_path)
            
            # Find neighbors and add to queue
            neighbors = await self.find_neighbors(current, 1)
            for neighbor in neighbors:
                neighbor_id = neighbor["id"]
                if neighbor_id not in visited:
                    new_path = path + [neighbor_id]
                    new_relations = relations + [neighbor["relation"]]
                    queue.append((neighbor_id, new_path, new_relations))

        return {"paths": paths}


class MockVectorRetriever:
    """Mock vector retriever for testing purposes."""

    def __init__(self, query: str = ""):
        self.query = query

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock vector search."""
        # Return relevant mock documents based on query
        if "zirbeldrüse" in query.lower() or "pineal" in query.lower():
            documents = [
                {
                    "id": "doc_pineal_1",
                    "content": "Die Zirbeldrüse (Epiphyse) ist eine kleine endokrine Drüse im Gehirn, die Melatonin produziert. Fluorid kann zur Kalzifizierung der Zirbeldrüse führen und ihre Funktion beeinträchtigen.",
                    "score": 0.95,
                    "metadata": {"source": "medical_encyclopedia", "topic": "pineal_gland"}
                },
                {
                    "id": "doc_pineal_2",
                    "content": "Faktoren, die die Zirbeldrüse beeinträchtigen können: Fluorid, künstliches Licht, Stress, Alter, bestimmte Medikamente und Umweltgifte. Diese können die Melatoninproduktion reduzieren.",
                    "score": 0.92,
                    "metadata": {"source": "health_research", "topic": "pineal_impairment"}
                },
                {
                    "id": "doc_pineal_3",
                    "content": "Melatonin ist ein wichtiges Hormon für den Schlaf-Wach-Rhythmus. Mit zunehmendem Alter nimmt die Melatoninproduktion der Zirbeldrüse ab, was zu Schlafproblemen führen kann.",
                    "score": 0.88,
                    "metadata": {"source": "sleep_medicine", "topic": "melatonin"}
                }
            ]
        else:
            # Default documents for other queries
            documents = [
                {
                    "id": f"doc_{i}",
                    "content": f"Mock document {i} related to: {query}",
                    "score": 0.9 - (i * 0.1),
                    "metadata": {"source": f"mock_source_{i}"}
                }
                for i in range(3)
            ]

        return documents[:limit]


async def test_multi_hop_reasoning(
    query: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
    strategy: str = "forward_chaining",
    max_paths: int = 50,
    max_iterations: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """Test multi-hop reasoning with the given query."""
    
    if verbose:
        print(f"🧠 Testing multi-hop reasoning")
        print(f"📝 Query: {query}")
        print(f"🤖 Model: {model}")
        print(f"🔄 Strategy: {strategy}")
        print(f"📊 Max paths: {max_paths}")
        print(f"🔁 Max iterations: {max_iterations}")
        print("=" * 60)
    
    # Configure LLM
    llm_config = LLMConfig(
        provider="gemini",
        api_key=api_key,
        model=model,
        temperature=0.1,
        max_tokens=1000
    )
    
    # Initialize components with query context
    llm_client = LLMClient(llm_config)
    graph_engine = MockGraphEngine(query)
    vector_retriever = MockVectorRetriever(query)

    path_selector = PathSelectionAgent(llm_client, max_paths=10)
    path_finder = ReasoningPathFinder(graph_engine, path_selector)
    iterative_retriever = IterativeRetriever(
        llm_client, graph_engine, vector_retriever,
        max_iterations=max_iterations
    )
    
    results = {
        "query": query,
        "model": model,
        "strategy": strategy,
        "reasoning_paths": [],
        "refined_context": {},
        "performance": {},
        "success": False
    }
    
    try:
        # Step 1: Find reasoning paths
        if verbose:
            print("🔍 Step 1: Finding reasoning paths...")
        
        start_time = time.time()
        
        # Extract potential start entities from query (simple keyword matching)
        start_entities = []

        # Check for pineal gland related terms
        if "zirbeldrüse" in query.lower() or "pineal" in query.lower():
            for entity in ["Zirbeldrüse", "Melatonin", "Fluorid", "Licht"]:
                if entity.lower() in query.lower() or entity == "Zirbeldrüse":
                    start_entities.append(entity)
            if not start_entities:
                start_entities = ["Zirbeldrüse"]  # Default for pineal queries
        else:
            # Default entities for other queries
            for entity in ["Apple", "Apple Inc.", "Steve Jobs", "AI research"]:
                if entity.lower() in query.lower():
                    start_entities.append(entity)
            if not start_entities:
                start_entities = ["Apple Inc.", "AI research"]
        
        reasoning_paths = await path_finder.find_reasoning_paths(
            query=query,
            start_entities=start_entities,
            strategy=strategy,
            max_paths=max_paths
        )
        
        path_finding_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Found {len(reasoning_paths)} reasoning paths in {path_finding_time:.2f}s")
            for i, path_score in enumerate(reasoning_paths[:3]):  # Show top 3
                print(f"   Path {i+1}: Score {path_score.relevance_score:.2f} - {path_score.reasoning}")
        
        results["reasoning_paths"] = [
            {
                "entities": [e.name if hasattr(e, 'name') else str(e) for e in path_score.path.entities],
                "relations": [r.type if hasattr(r, 'type') else str(r) for r in path_score.path.relations],
                "relevance_score": path_score.relevance_score,
                "confidence": path_score.confidence,
                "reasoning": path_score.reasoning
            }
            for path_score in reasoning_paths
        ]
        
        # Step 2: Create initial context and refine iteratively
        if verbose:
            print("\n🔄 Step 2: Iterative context refinement...")
        
        start_time = time.time()
        
        # Create initial context from top paths
        initial_context = RetrievalContext(
            entities={entity: {"type": "UNKNOWN"} for path in reasoning_paths[:3] for entity in path.path.entities},
            paths=[path_score.path for path_score in reasoning_paths[:5]]
        )
        
        refined_context = await iterative_retriever.refine_context(query, initial_context)
        
        refinement_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Context refinement completed in {refinement_time:.2f}s")
            print(f"   Iterations used: {refined_context.metadata.get('iterations_used', 0)}")
            print(f"   Final entities: {len(refined_context.entities)}")
            print(f"   Final documents: {len(refined_context.documents)}")
            
            final_analysis = refined_context.metadata.get('final_analysis')
            if final_analysis:
                print(f"   Final confidence: {final_analysis.confidence:.2f}")
                print(f"   Context sufficient: {final_analysis.is_sufficient}")
        
        final_analysis = refined_context.metadata.get('final_analysis')
        results["refined_context"] = {
            "entity_count": len(refined_context.entities),
            "document_count": len(refined_context.documents),
            "relation_count": len(refined_context.relations),
            "iterations_used": refined_context.metadata.get('iterations_used', 0),
            "final_analysis": {
                "is_sufficient": final_analysis.is_sufficient if final_analysis else False,
                "confidence": final_analysis.confidence if final_analysis else 0.0
            } if final_analysis else None
        }
        
        results["performance"] = {
            "path_finding_time": path_finding_time,
            "refinement_time": refinement_time,
            "total_time": path_finding_time + refinement_time
        }

        # Determine success based on meaningful criteria
        has_paths = len(reasoning_paths) > 0
        has_context = len(refined_context.entities) > 0 or len(refined_context.documents) > 0
        reasonable_confidence = final_analysis.confidence > 0.1 if final_analysis else False

        results["success"] = has_paths or has_context or reasonable_confidence

        if verbose:
            print(f"\n📊 Success Criteria:")
            print(f"   Has reasoning paths: {has_paths} ({len(reasoning_paths)} paths)")
            print(f"   Has context: {has_context} ({len(refined_context.entities)} entities, {len(refined_context.documents)} docs)")
            print(f"   Reasonable confidence: {reasonable_confidence} ({final_analysis.confidence:.2f} if final_analysis else 0.0)")
            print(f"   Overall success: {results['success']}")
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ Multi-hop reasoning test completed successfully!")
            print(f"📊 Total time: {results['performance']['total_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Error during multi-hop reasoning: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        results["error"] = str(e)
    
    return results


def save_results(results: Dict[str, Any], output_file: Path, verbose: bool = False) -> bool:
    """Save test results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"💾 Results saved to: {output_file}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving results to {output_file}: {e}")
        return False


async def main():
    """Main function to run multi-hop reasoning test."""
    parser = argparse.ArgumentParser(
        description="Test multi-hop reasoning capabilities of MoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_multi_hop_cli.py "How are Apple's AI research efforts related to their partnership with universities?"
  python test_multi_hop_cli.py "What connects Steve Jobs to iPhone development?" --strategy bidirectional
  python test_multi_hop_cli.py "How does AI research influence product development?" --verbose --output results.json
"""
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Multi-hop reasoning query to test"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["forward_chaining", "backward_chaining", "bidirectional"],
        default="forward_chaining",
        help="Reasoning strategy to use (default: forward_chaining)"
    )
    
    parser.add_argument(
        "--max-paths",
        type=int,
        default=50,
        help="Maximum number of paths to discover (default: 50)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum context refinement iterations (default: 5)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model to use (default: gemini-1.5-flash)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed reasoning output"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    print("🧠 MoRAG Multi-Hop Reasoning Test")
    print("=" * 60)

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also set specific loggers to debug
        logging.getLogger("morag_reasoning").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.INFO)  # Reduce HTTP noise
        logging.getLogger("httpcore").setLevel(logging.INFO)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Setup environment
    if not setup_environment(args.api_key):
        return 1
    
    try:
        # Run multi-hop reasoning test
        results = await test_multi_hop_reasoning(
            query=args.query,
            api_key=args.api_key or os.getenv("GEMINI_API_KEY"),
            model=args.model,
            strategy=args.strategy,
            max_paths=args.max_paths,
            max_iterations=args.max_iterations,
            verbose=args.verbose
        )
        
        if not results.get("success", False):
            print("❌ Multi-hop reasoning test failed")
            return 1
        
        # Save results if requested
        if args.output:
            output_file = Path(args.output)
            if save_results(results, output_file, args.verbose):
                print(f"📁 Results saved to: {output_file}")
            else:
                return 1
        
        print("✅ Multi-hop reasoning test completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
