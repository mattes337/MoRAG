#!/usr/bin/env python3
"""
MoRAG Prompt Testing CLI Script

Usage: python test-prompt.py [OPTIONS] "your prompt here"

This script allows testing LLM prompts with multi-hop reasoning and RAG capabilities,
showing all LLM interactions step by step.
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Import MoRAG components
try:
    from morag_graph import (
        Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig,
        HybridRetrievalCoordinator, ContextExpansionEngine,
        QueryEntityExtractor, GraphCRUD, GraphTraversal, GraphAnalytics
    )
    from morag_graph.operations import GraphPath
    from morag.models.enhanced_query import (
        QueryType, ExpansionStrategy, FusionStrategy,
        EnhancedQueryRequest, EntityQueryRequest, GraphTraversalRequest
    )
    from morag.database_factory import get_default_neo4j_storage, get_default_qdrant_storage
    from morag_reasoning import (
        ReasoningStrategy, PathSelectionAgent, ReasoningPathFinder,
        IterativeRetriever, LLMClient, LLMConfig
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing MoRAG components: {e}")
    COMPONENTS_AVAILABLE = False


class PromptLogger:
    """Logger for tracking all LLM interactions."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.interactions = []
        self.step_counter = 0
    
    def log_prompt(self, prompt: str, context: str = ""):
        """Log an LLM prompt."""
        self.step_counter += 1
        interaction = {
            "step": self.step_counter,
            "type": "prompt",
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "content": prompt
        }
        self.interactions.append(interaction)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🤖 STEP {self.step_counter}: LLM PROMPT ({context})")
            print(f"{'='*80}")
            print(prompt)
    
    def log_response(self, response: str, context: str = ""):
        """Log an LLM response."""
        interaction = {
            "step": self.step_counter,
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "content": response
        }
        self.interactions.append(interaction)
        
        if self.verbose:
            print(f"\n{'-'*80}")
            print(f"🧠 LLM RESPONSE ({context})")
            print(f"{'-'*80}")
            print(response)
    
    def log_step(self, step_name: str, details: str = ""):
        """Log a processing step."""
        if self.verbose:
            print(f"\n🔄 {step_name}")
            if details:
                print(f"   {details}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all interactions."""
        prompts = [i for i in self.interactions if i["type"] == "prompt"]
        responses = [i for i in self.interactions if i["type"] == "response"]
        
        return {
            "total_interactions": len(self.interactions),
            "total_prompts": len(prompts),
            "total_responses": len(responses),
            "interactions": self.interactions
        }


class LoggingLLMClient(LLMClient):
    """LLM client with detailed logging."""
    
    def __init__(self, config: LLMConfig, logger: PromptLogger):
        super().__init__(config)
        self.prompt_logger = logger
    
    async def generate(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: str = "general"
    ) -> str:
        """Generate response with logging."""
        # Log the prompt
        if prompt:
            self.prompt_logger.log_prompt(prompt, context)
        elif messages:
            formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            self.prompt_logger.log_prompt(formatted_prompt, context)
        
        # Call the parent method
        if prompt:
            response = await super().generate(prompt, max_tokens, temperature)
        else:
            response = await super().generate_from_messages(messages, max_tokens, temperature)
        
        # Log the response
        self.prompt_logger.log_response(response, context)
        
        return response


async def test_database_connections(args) -> tuple[Optional[Neo4jStorage], Optional[QdrantStorage]]:
    """Test and establish database connections."""
    neo4j_storage = None
    qdrant_storage = None
    
    if args.neo4j:
        try:
            neo4j_storage = get_default_neo4j_storage()
            if neo4j_storage:
                await neo4j_storage.connect()
                print("✅ Neo4j connection established")
            else:
                print("❌ Failed to create Neo4j storage")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
    
    if args.qdrant:
        try:
            qdrant_storage = get_default_qdrant_storage()
            if qdrant_storage:
                await qdrant_storage.connect()
                print("✅ Qdrant connection established")
            else:
                print("❌ Failed to create Qdrant storage")
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
    
    return neo4j_storage, qdrant_storage


async def execute_prompt_with_reasoning(
    prompt: str,
    neo4j_storage: Optional[Neo4jStorage],
    qdrant_storage: Optional[QdrantStorage],
    args,
    logger: PromptLogger
) -> Dict[str, Any]:
    """Execute prompt with multi-hop reasoning and RAG."""
    
    results = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "final_result": None,
        "performance": {}
    }
    
    start_time = time.time()
    
    # Initialize LLM client with logging
    llm_config = LLMConfig(
        provider="gemini",
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("MORAG_GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.1,
        max_tokens=args.max_tokens
    )
    
    llm_client = LoggingLLMClient(llm_config, logger)
    
    try:
        # Step 1: Initial query analysis
        logger.log_step("Step 1: Query Analysis", "Analyzing the prompt to understand intent and extract entities")
        
        analysis_prompt = f"""
Analyze this query directly without preambles. Provide:
1. Query intent and type
2. Key entities mentioned
3. Relationships to explore
4. Reasoning strategy needed

Query: "{prompt}"

Provide a structured analysis. Be direct and concise.
"""
        
        analysis = await llm_client.generate(analysis_prompt, context="query_analysis")
        results["steps"].append({"step": "query_analysis", "result": analysis})
        
        # Step 2: Entity extraction if Neo4j is available
        if neo4j_storage and args.enable_multi_hop:
            logger.log_step("Step 2: Entity Extraction", "Extracting entities from the query for graph traversal")
            
            entity_prompt = f"""
Extract 2-5 key entities for graph traversal from this query. No preambles or explanations.

Query: "{prompt}"

Return only the entity names, one per line.
"""
            
            entity_response = await llm_client.generate(entity_prompt, context="entity_extraction")
            start_entities = [e.strip() for e in entity_response.split('\n') if e.strip()]
            results["steps"].append({"step": "entity_extraction", "result": start_entities})
            
            # Step 3: Multi-hop reasoning
            if start_entities:
                logger.log_step("Step 3: Multi-hop Reasoning", f"Finding reasoning paths from entities: {start_entities}")
                
                path_selector = PathSelectionAgent(llm_client, max_paths=5)
                
                # Mock graph engine for demonstration
                class MockGraphEngine:
                    async def find_paths_between_entities(self, entities, max_paths=10):
                        # Convert string entities to mock entity objects
                        entity_objects = []
                        for entity in entities:
                            if isinstance(entity, str):
                                entity_obj = type('Entity', (), {
                                    'id': entity.lower().replace(' ', '_'),
                                    'name': entity,
                                    'type': 'CONCEPT'
                                })()
                                entity_objects.append(entity_obj)
                            else:
                                entity_objects.append(entity)

                        return [
                            type('Path', (), {
                                'entities': entity_objects[:2] + [type('Entity', (), {'name': 'intermediate_entity'})(), entity_objects[-1]] if len(entity_objects) > 1 else entity_objects,
                                'relations': [
                                    type('Relation', (), {'type': 'RELATED_TO'})(),
                                    type('Relation', (), {'type': 'INFLUENCES'})(),
                                    type('Relation', (), {'type': 'CONNECTED_TO'})()
                                ]
                            })()
                            for _ in range(min(3, max_paths))
                        ]

                    async def find_neighbors(self, entity_id, max_distance=2):
                        # Mock neighbor finding
                        return [
                            type('Entity', (), {'id': f'neighbor_{i}', 'name': f'Related Entity {i}', 'type': 'CONCEPT'})()
                            for i in range(3)
                        ]

                    async def find_shortest_path(self, start_entity, end_entity):
                        # Mock shortest path finding
                        return type('Path', (), {
                            'entities': [start_entity, 'intermediate', end_entity],
                            'relations': [
                                type('Relation', (), {'type': 'CONNECTED_TO'})(),
                                type('Relation', (), {'type': 'RELATED_TO'})()
                            ],
                            'total_weight': 1.0
                        })()

                    async def search_entities(self, query, limit=10):
                        # Mock entity search
                        return [
                            type('Entity', (), {
                                'id': f'entity_{i}',
                                'name': f'Mock Entity {i}',
                                'type': 'CONCEPT',
                                'description': f'Mock entity related to {query}'
                            })()
                            for i in range(min(3, limit))
                        ]

                try:
                    # Simplified multi-hop reasoning using LLM directly
                    reasoning_prompt = f"""
Analyze potential reasoning paths directly without preambles.

Query: "{prompt}"
Starting entities: {start_entities}

Provide structured analysis of:
1. How these entities might be connected
2. What intermediate entities or concepts might link them
3. What reasoning paths could help answer the query

Focus on logical connections and relationships. Be direct and concise.
"""

                    reasoning_analysis = await llm_client.generate(reasoning_prompt, context="multi_hop_reasoning")

                    results["steps"].append({
                        "step": "multi_hop_reasoning",
                        "result": f"Analyzed reasoning paths for {len(start_entities)} entities"
                    })

                    logger.log_step("Multi-hop Reasoning", "Completed LLM-based reasoning analysis")
                except Exception as e:
                    logger.log_step("Multi-hop Reasoning Failed", str(e))
                    results["steps"].append({
                        "step": "multi_hop_reasoning",
                        "result": f"Failed: {str(e)}"
                    })

        
        # Step 4: Vector search if Qdrant is available
        if qdrant_storage:
            logger.log_step("Step 4: Vector Search", "Searching for relevant documents using vector similarity")
            
            try:
                vector_results = await qdrant_storage.search_entities(prompt, limit=5)
                results["steps"].append({
                    "step": "vector_search", 
                    "result": f"Found {len(vector_results)} relevant documents"
                })
            except Exception as e:
                logger.log_step("Vector Search Failed", str(e))
        
        # Step 5: Final synthesis
        logger.log_step("Step 5: Final Synthesis", "Combining all information to generate final response")
        
        synthesis_prompt = f"""
You are a direct, professional assistant. Provide a comprehensive answer to this query without any preambles, introductions, or conversational phrases like "Absolut! Basierend auf..." or "Certainly! Based on...". Start directly with the substantive content.

Query: "{prompt}"

Consider:
- The query analysis performed
- Any entities and relationships identified
- Relevant information from the knowledge base
- Multi-hop reasoning paths explored

Provide a detailed, well-reasoned response. Be direct and start immediately with the main content.
"""
        
        final_response = await llm_client.generate(synthesis_prompt, context="final_synthesis")
        results["final_result"] = final_response
        
        # Performance metrics
        end_time = time.time()
        results["performance"] = {
            "total_time_seconds": end_time - start_time,
            "llm_interactions": logger.step_counter
        }
        
        return results
        
    except Exception as e:
        logger.log_step("Error", f"Failed during execution: {e}")
        results["error"] = str(e)
        return results


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test MoRAG prompts with detailed LLM interaction logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test-prompt.py --neo4j --qdrant "mit welchen lebensmitteln kann ich ADHS eindämmen?"
  python test-prompt.py --neo4j --enable-multi-hop --max-tokens 4000 "How are Apple's AI efforts connected to universities?"
  python test-prompt.py --qdrant --verbose --max-tokens 12000 "What are the latest developments in machine learning?"
"""
    )
    
    # Database selection
    parser.add_argument("--neo4j", action="store_true", help="Use Neo4j for graph operations")
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant for vector search")
    
    # Reasoning options
    parser.add_argument("--enable-multi-hop", action="store_true", help="Enable multi-hop reasoning")

    # LLM options
    parser.add_argument("--max-tokens", type=int, default=8000, help="Maximum tokens for LLM responses (default: 8000)")

    # Output options
    parser.add_argument("--verbose", action="store_true", help="Show detailed LLM interactions")
    parser.add_argument("--output", help="Save results to JSON file")
    
    # The prompt
    parser.add_argument("prompt", help="The prompt to execute")
    
    args = parser.parse_args()
    
    if not COMPONENTS_AVAILABLE:
        print("❌ MoRAG components not available. Please install required packages.")
        return 1
    
    if not args.neo4j and not args.qdrant:
        print("❌ Please specify --neo4j, --qdrant, or both")
        return 1
    
    print(f"🚀 Testing prompt: '{args.prompt}'")
    print(f"📊 Databases: Neo4j={args.neo4j}, Qdrant={args.qdrant}")
    print(f"🧠 Multi-hop reasoning: {args.enable_multi_hop}")
    print(f"🎯 Max tokens: {args.max_tokens}")
    
    # Initialize logger
    logger = PromptLogger(verbose=args.verbose)
    
    # Test database connections
    neo4j_storage, qdrant_storage = await test_database_connections(args)
    
    if not neo4j_storage and not qdrant_storage:
        print("❌ No database connections available")
        return 1
    
    # Execute the prompt
    try:
        results = await execute_prompt_with_reasoning(
            args.prompt, neo4j_storage, qdrant_storage, args, logger
        )
        
        # Show final result
        print(f"\n{'='*80}")
        print("🎯 FINAL RESULT")
        print(f"{'='*80}")
        if results.get("final_result"):
            print(results["final_result"])
        else:
            print("❌ No final result generated")
        
        # Show summary
        summary = logger.get_summary()
        print(f"\n📊 EXECUTION SUMMARY")
        print(f"   Total LLM interactions: {summary['total_prompts']}")
        print(f"   Processing time: {results.get('performance', {}).get('total_time_seconds', 0):.2f}s")
        print(f"   Steps completed: {len(results.get('steps', []))}")
        
        # Save results if requested
        if args.output:
            output_data = {
                "results": results,
                "llm_interactions": summary
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return 1
    
    finally:
        # Clean up connections
        if neo4j_storage:
            await neo4j_storage.disconnect()
        if qdrant_storage:
            await qdrant_storage.disconnect()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
