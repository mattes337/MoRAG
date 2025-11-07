#!/usr/bin/env python3
"""
MoRAG Prompt Testing CLI Script

Usage: python test-prompt.py [OPTIONS] "your prompt here"

This script allows testing LLM prompts with multi-hop reasoning and RAG capabilities,
showing all LLM interactions step by step.

Key options:
  --model MODEL         Specify LLM model (fallback: MORAG_GEMINI_MODEL env var or gemini-2.5-flash)
  --neo4j              Enable Neo4j graph operations
  --qdrant             Enable Qdrant vector search
  --enable-multi-hop   Enable multi-hop reasoning
  --verbose            Show detailed LLM interactions
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv

env_path = project_root / ".env"
load_dotenv(env_path)

# Import MoRAG components
try:
    from morag.database_factory import (
        get_default_neo4j_storage,
        get_default_qdrant_storage,
    )
    from morag.models.enhanced_query import (
        EnhancedQueryRequest,
        EntityQueryRequest,
        ExpansionStrategy,
        FusionStrategy,
        GraphTraversalRequest,
        QueryType,
    )
    from morag_graph import (
        ContextExpansionEngine,
        GraphAnalytics,
        GraphCRUD,
        GraphTraversal,
        HybridRetrievalCoordinator,
        Neo4jConfig,
        Neo4jStorage,
        QdrantConfig,
        QdrantStorage,
        QueryEntityExtractor,
    )
    from morag_graph.operations import GraphPath
    from morag_reasoning import (
        IterativeRetriever,
        LLMClient,
        LLMConfig,
        PathSelectionAgent,
        ReasoningPathFinder,
        ReasoningStrategy,
        RecursiveFactRetrievalRequest,
        RecursiveFactRetrievalService,
    )

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"[FAIL] Error importing MoRAG components: {e}")
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
            "content": prompt,
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
            "content": response,
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
            print(f"\n[PROCESSING] {step_name}")
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
            "interactions": self.interactions,
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
        context: str = "general",
    ) -> str:
        """Generate response with logging."""
        # Log the prompt
        if prompt:
            self.prompt_logger.log_prompt(prompt, context)
        elif messages:
            formatted_prompt = "\n".join(
                [f"{m['role']}: {m['content']}" for m in messages]
            )
            self.prompt_logger.log_prompt(formatted_prompt, context)

        # Call the parent method
        if prompt:
            response = await super().generate(prompt, max_tokens, temperature)
        else:
            response = await super().generate_from_messages(
                messages, max_tokens, temperature
            )

        # Log the response
        self.prompt_logger.log_response(response, context)

        return response


async def test_database_connections(
    args,
) -> tuple[Optional[Neo4jStorage], Optional[QdrantStorage]]:
    """Test and establish database connections."""
    neo4j_storage = None
    qdrant_storage = None

    if args.neo4j:
        try:
            neo4j_storage = get_default_neo4j_storage()
            if neo4j_storage:
                await neo4j_storage.connect()
                print("[OK] Neo4j connection established")
            else:
                print("[FAIL] Failed to create Neo4j storage")
        except Exception as e:
            print(f"[FAIL] Neo4j connection failed: {e}")

    if args.qdrant:
        try:
            qdrant_storage = get_default_qdrant_storage()
            if qdrant_storage:
                await qdrant_storage.connect()
                print("[OK] Qdrant connection established")
            else:
                print("[FAIL] Failed to create Qdrant storage")
        except Exception as e:
            print(f"[FAIL] Qdrant connection failed: {e}")

    return neo4j_storage, qdrant_storage


async def execute_prompt_with_fact_retrieval(
    prompt: str,
    neo4j_storage: Optional[Neo4jStorage],
    qdrant_storage: Optional[QdrantStorage],
    args,
    logger: PromptLogger,
) -> Dict[str, Any]:
    """Execute prompt using the new fact-based retrieval system."""

    results = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "method": "fact_retrieval",
        "final_result": None,
        "performance": {},
        "facts": [],
        "traversal_steps": [],
        "metadata": {},
    }

    start_time = time.time()

    # Initialize LLM client with logging
    model = args.model or os.getenv("MORAG_GEMINI_MODEL", "gemini-2.5-flash")
    llm_config = LLMConfig(
        provider="gemini",
        api_key=os.getenv("GEMINI_API_KEY"),
        model=model,
        temperature=0.1,
        max_tokens=args.max_tokens,
    )

    llm_client = LoggingLLMClient(llm_config, logger)

    try:
        logger.log_step(
            "Fact-Based Retrieval",
            "Using RecursiveFactRetrievalService for comprehensive fact extraction",
        )

        # Create fact retrieval service
        fact_service = RecursiveFactRetrievalService(
            llm_client=llm_client,
            neo4j_storage=neo4j_storage,
            qdrant_storage=qdrant_storage,
            stronger_llm_client=llm_client,  # Use same client for now
        )

        # Create fact retrieval request
        fact_request = RecursiveFactRetrievalRequest(
            user_query=prompt,
            max_depth=3,
            decay_rate=0.2,
            max_facts_per_node=5,
            min_fact_score=0.1,
            max_total_facts=50,
            facts_only=False,  # Generate final answer
            skip_fact_evaluation=False,
            language="en",
        )

        # Execute fact retrieval
        fact_response = await fact_service.retrieve_facts_recursively(fact_request)

        # Process results
        results["facts"] = [
            {
                "text": fact.fact_text,
                "source_node": fact.source_node_id,
                "source_property": fact.source_property,
                "depth": fact.extracted_from_depth,
                "score": fact.score,
                "final_score": fact.final_decayed_score,
                "source_description": fact.source_description,
                "metadata": fact.source_metadata,
            }
            for fact in fact_response.final_facts
        ]

        results["traversal_steps"] = [
            {
                "node_id": step.node_id,
                "node_name": step.node_name,
                "depth": step.depth,
                "facts_extracted": step.facts_extracted,
                "decision": step.next_nodes_decision,
                "reasoning": step.reasoning,
            }
            for step in fact_response.traversal_steps
        ]

        results["final_result"] = fact_response.final_answer
        results["metadata"] = {
            "query_id": fact_response.query_id,
            "initial_entities": fact_response.initial_entities,
            "total_nodes_explored": fact_response.total_nodes_explored,
            "max_depth_reached": fact_response.max_depth_reached,
            "total_raw_facts": fact_response.total_raw_facts,
            "total_scored_facts": fact_response.total_scored_facts,
            "confidence_score": fact_response.confidence_score,
            "gta_llm_calls": fact_response.gta_llm_calls,
            "fca_llm_calls": fact_response.fca_llm_calls,
            "final_llm_calls": fact_response.final_llm_calls,
        }

        # Calculate performance metrics
        end_time = time.time()
        results["performance"] = {
            "total_time_seconds": end_time - start_time,
            "processing_time_ms": fact_response.processing_time_ms,
            "total_llm_calls": fact_response.gta_llm_calls
            + fact_response.fca_llm_calls
            + fact_response.final_llm_calls,
        }

        return results

    except Exception as e:
        logger.log_step("Error", f"Fact retrieval failed: {str(e)}")
        results["final_result"] = f"Error: {str(e)}"
        results["performance"]["total_time_seconds"] = time.time() - start_time
        return results


async def execute_prompt_with_reasoning(
    prompt: str,
    neo4j_storage: Optional[Neo4jStorage],
    qdrant_storage: Optional[QdrantStorage],
    args,
    logger: PromptLogger,
) -> Dict[str, Any]:
    """Execute prompt with multi-hop reasoning and RAG."""

    results = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "final_result": None,
        "performance": {},
        "context_data": {
            "vector_chunks": [],
            "graph_entities": [],
            "graph_relations": [],
            "reasoning_paths": [],
        },
    }

    start_time = time.time()

    # Initialize LLM client with logging
    model = args.model or os.getenv("MORAG_GEMINI_MODEL", "gemini-2.5-flash")
    llm_config = LLMConfig(
        provider="gemini",
        api_key=os.getenv("GEMINI_API_KEY"),
        model=model,
        temperature=0.1,
        max_tokens=args.max_tokens,
    )

    llm_client = LoggingLLMClient(llm_config, logger)

    try:
        # Step 1: Initial query analysis
        logger.log_step(
            "Step 1: Query Analysis",
            "Analyzing the prompt to understand intent and extract entities",
        )

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
            logger.log_step(
                "Step 2: Entity Extraction",
                "Extracting entities from the query for graph traversal",
            )

            entity_prompt = f"""
Extract 2-5 key entities for graph traversal from this query. No preambles or explanations.

Query: "{prompt}"

Return only the entity names, one per line.
"""

            entity_response = await llm_client.generate(
                entity_prompt, context="entity_extraction"
            )
            start_entities = [
                e.strip() for e in entity_response.split("\n") if e.strip()
            ]
            results["steps"].append(
                {"step": "entity_extraction", "result": start_entities}
            )

            # Step 3: Multi-hop reasoning and graph traversal
            graph_entities = []
            graph_relations = []
            reasoning_paths = []

            if start_entities:
                logger.log_step(
                    "Step 3: Graph Traversal",
                    f"Finding entities and relations from: {start_entities}",
                )

                try:
                    # Real Neo4j entity search and traversal
                    for entity_name in start_entities[:3]:  # Limit to first 3 entities
                        # Search for entities matching the name
                        entity_query = """
                        MATCH (e)
                        WHERE toLower(e.name) CONTAINS toLower($entity_name)
                        RETURN e.name as name, labels(e) as types, e.description as description
                        LIMIT 5
                        """
                        entity_results = await neo4j_storage._execute_query(
                            entity_query, {"entity_name": entity_name}
                        )

                        for record in entity_results:
                            graph_entities.append(
                                {
                                    "name": record["name"],
                                    "types": record["types"],
                                    "description": record.get("description", ""),
                                }
                            )

                        # Find relations from these entities
                        relation_query = """
                        MATCH (e1)-[r]->(e2)
                        WHERE toLower(e1.name) CONTAINS toLower($entity_name)
                        RETURN e1.name as source, type(r) as relation_type, e2.name as target, r.description as description
                        LIMIT 10
                        """
                        relation_results = await neo4j_storage._execute_query(
                            relation_query, {"entity_name": entity_name}
                        )

                        for record in relation_results:
                            graph_relations.append(
                                {
                                    "source": record["source"],
                                    "relation": record["relation_type"],
                                    "target": record["target"],
                                    "description": record.get("description", ""),
                                }
                            )

                    results["context_data"]["graph_entities"] = graph_entities
                    results["context_data"]["graph_relations"] = graph_relations

                    logger.log_step(
                        "Graph Traversal",
                        f"Found {len(graph_entities)} entities and {len(graph_relations)} relations",
                    )

                    # Multi-hop reasoning with actual graph data
                    if graph_entities or graph_relations:
                        reasoning_prompt = f"""
                        Analyze reasoning paths using the actual graph data found:

                        Query: "{prompt}"

                        Graph Entities Found:
                        {chr(10).join([f"- {e['name']} ({', '.join(e['types'])}): {e['description']}" for e in graph_entities[:10]])}

                        Graph Relations Found:
                        {chr(10).join([f"- {r['source']} --{r['relation']}--> {r['target']}: {r['description']}" for r in graph_relations[:10]])}

                        Identify key reasoning paths and connections that help answer the query. Be direct and concise.
                        """

                        reasoning_analysis = await llm_client.generate(
                            reasoning_prompt, context="multi_hop_reasoning"
                        )
                        reasoning_paths.append(reasoning_analysis)
                        results["context_data"]["reasoning_paths"] = reasoning_paths

                        results["steps"].append(
                            {
                                "step": "multi_hop_reasoning",
                                "result": f"Analyzed {len(graph_entities)} entities and {len(graph_relations)} relations",
                            }
                        )

                except Exception as e:
                    logger.log_step("Graph Traversal Failed", str(e))
                    # Fallback to simple reasoning without graph data

        # Step 4: Vector search if Qdrant is available
        vector_chunks = []
        if qdrant_storage:
            logger.log_step(
                "Step 4: Vector Search",
                "Searching for relevant documents using vector similarity",
            )

            try:
                vector_results = await qdrant_storage.search_entities(prompt, limit=5)
                vector_chunks = [
                    {
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0.0),
                    }
                    for result in vector_results
                ]
                results["context_data"]["vector_chunks"] = vector_chunks
                results["steps"].append(
                    {
                        "step": "vector_search",
                        "result": f"Found {len(vector_results)} relevant documents",
                    }
                )
                logger.log_step(
                    "Vector Search", f"Retrieved {len(vector_chunks)} chunks"
                )
            except Exception as e:
                logger.log_step("Vector Search Failed", str(e))

        # Step 5: Final synthesis with actual context
        logger.log_step(
            "Step 5: Final Synthesis",
            "Combining all retrieved information to generate final response",
        )

        # Prepare context sections
        vector_context = ""
        if vector_chunks:
            vector_context = "\n\nRELEVANT DOCUMENT CHUNKS:\n" + "\n".join(
                [
                    f"- {chunk['content'][:200]}..."
                    if len(chunk["content"]) > 200
                    else f"- {chunk['content']}"
                    for chunk in vector_chunks[:5]
                ]
            )

        graph_context = ""
        if graph_entities or graph_relations:
            graph_context = "\n\nGRAPH KNOWLEDGE:\n"
            if graph_entities:
                graph_context += "Entities:\n" + "\n".join(
                    [
                        f"- {e['name']} ({', '.join(e['types'])}): {e['description']}"
                        for e in graph_entities[:10]
                    ]
                )
            if graph_relations:
                graph_context += "\nRelations:\n" + "\n".join(
                    [
                        f"- {r['source']} --{r['relation']}--> {r['target']}: {r['description']}"
                        for r in graph_relations[:10]
                    ]
                )

        reasoning_context = ""
        if reasoning_paths:
            reasoning_context = "\n\nREASONING ANALYSIS:\n" + "\n".join(reasoning_paths)

        synthesis_prompt = f"""
You are a direct, professional assistant. Provide a comprehensive answer to this query without any preambles, introductions, or conversational phrases like "Absolut! Basierend auf..." or "Certainly! Based on...". Start directly with the substantive content.

Query: "{prompt}"

Use the following information to provide a well-reasoned, comprehensive response:
{vector_context}
{graph_context}
{reasoning_context}

Synthesize all available information to provide a detailed, accurate response. Be direct and start immediately with the main content.
"""

        final_response = await llm_client.generate(
            synthesis_prompt, context="final_synthesis"
        )
        results["final_result"] = final_response

        # Performance metrics
        end_time = time.time()
        results["performance"] = {
            "total_time_seconds": end_time - start_time,
            "llm_interactions": logger.step_counter,
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
  python test-prompt.py --neo4j --qdrant --use-fact-retrieval "What are the latest developments in machine learning?"
  python test-prompt.py --qdrant --verbose --max-tokens 12000 "What are the latest developments in machine learning?"
  python test-prompt.py --neo4j --qdrant --model gemini-1.5-pro "Analyze complex relationships in the data"
""",
    )

    # Database selection
    parser.add_argument(
        "--neo4j", action="store_true", help="Use Neo4j for graph operations"
    )
    parser.add_argument(
        "--qdrant", action="store_true", help="Use Qdrant for vector search"
    )

    # Reasoning options
    parser.add_argument(
        "--enable-multi-hop", action="store_true", help="Enable multi-hop reasoning"
    )
    parser.add_argument(
        "--use-fact-retrieval",
        action="store_true",
        help="Use new fact-based retrieval system",
    )

    # LLM options
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model to use (default: from MORAG_GEMINI_MODEL env var or gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens for LLM responses (default: 8000)",
    )

    # Output options
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed LLM interactions"
    )
    parser.add_argument("--output", help="Save results to JSON file")

    # The prompt
    parser.add_argument("prompt", help="The prompt to execute")

    args = parser.parse_args()

    if not COMPONENTS_AVAILABLE:
        print(
            "[FAIL] MoRAG components not available. Please install required packages."
        )
        return 1

    if not args.neo4j and not args.qdrant:
        print("[FAIL] Please specify --neo4j, --qdrant, or both")
        return 1

    print(f"🚀 Testing prompt: '{args.prompt}'")
    print(f"📊 Databases: Neo4j={args.neo4j}, Qdrant={args.qdrant}")
    print(f"🧠 Multi-hop reasoning: {args.enable_multi_hop}")
    print(f"🔬 Fact-based retrieval: {args.use_fact_retrieval}")
    print(
        f"🤖 Model: {args.model or os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.5-flash')}"
    )
    print(f"🎯 Max tokens: {args.max_tokens}")

    # Initialize logger
    logger = PromptLogger(verbose=args.verbose)

    # Test database connections
    neo4j_storage, qdrant_storage = await test_database_connections(args)

    if not neo4j_storage and not qdrant_storage:
        print("[FAIL] No database connections available")
        return 1

    # Execute the prompt
    try:
        if args.use_fact_retrieval:
            if not neo4j_storage or not qdrant_storage:
                print(
                    "[FAIL] Fact-based retrieval requires both Neo4j and Qdrant connections"
                )
                return 1
            results = await execute_prompt_with_fact_retrieval(
                args.prompt, neo4j_storage, qdrant_storage, args, logger
            )
        else:
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
            print("[FAIL] No final result generated")

        # Show summary
        summary = logger.get_summary()
        print(f"\n📊 EXECUTION SUMMARY")
        print(f"   Total LLM interactions: {summary['total_prompts']}")
        print(
            f"   Processing time: {results.get('performance', {}).get('total_time_seconds', 0):.2f}s"
        )

        if args.use_fact_retrieval:
            metadata = results.get("metadata", {})
            print(f"   Method: Fact-based retrieval")
            print(f"   Facts extracted: {len(results.get('facts', []))}")
            print(f"   Nodes explored: {metadata.get('total_nodes_explored', 0)}")
            print(f"   Max depth reached: {metadata.get('max_depth_reached', 0)}")
            print(f"   Confidence score: {metadata.get('confidence_score', 0):.2f}")
            print(f"   GTA LLM calls: {metadata.get('gta_llm_calls', 0)}")
            print(f"   FCA LLM calls: {metadata.get('fca_llm_calls', 0)}")
            print(f"   Final LLM calls: {metadata.get('final_llm_calls', 0)}")
        else:
            print(f"   Method: Traditional multi-hop reasoning")
            print(f"   Steps completed: {len(results.get('steps', []))}")

        # Save results if requested
        if args.output:
            output_data = {"results": results, "llm_interactions": summary}
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"[FAIL] Execution failed: {e}")
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
        print("\n[STOP]  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Fatal error: {e}")
        sys.exit(1)
