#!/usr/bin/env python3
"""
MoRAG Context Generation Testing Script

This script tests the context generation for LLM calls using agentic AI to:
1. Extract entities from the input prompt
2. Navigate through the knowledge graph as needed
3. Execute the prompt with the generated context
4. Return both the context and the final response

Usage: python test-context-generation.py [OPTIONS] "your prompt here"

Example:
  python test-context-generation.py --neo4j --qdrant --verbose "How does nutrition affect ADHD symptoms?"
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import argparse
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Import MoRAG components
try:
    # Try to import the core components we need
    from morag_graph import Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig
    from morag_reasoning import LLMClient, LLMConfig

    # Optional components - import if available
    try:
        from morag_graph import GraphTraversal, ContextExpansionEngine
    except ImportError:
        GraphTraversal = None
        ContextExpansionEngine = None

    try:
        from morag_graph.ai.entity_agent import EntityExtractionAgent
        from morag_graph.utils.entity_normalizer import EntityNormalizer
        from morag_core.ai import create_agent_with_config
    except ImportError:
        EntityExtractionAgent = None
        EntityNormalizer = None
        create_agent_with_config = None

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing MoRAG components: {e}")
    COMPONENTS_AVAILABLE = False

    # Define dummy classes for type hints when imports fail
    class Neo4jStorage: pass
    class QdrantStorage: pass
    class Neo4jConfig: pass
    class QdrantConfig: pass
    class LLMClient: pass
    class LLMConfig: pass
    EntityExtractionAgent = None
    EntityNormalizer = None
    GraphTraversal = None
    ContextExpansionEngine = None
    create_agent_with_config = None


def create_neo4j_storage() -> Optional[Neo4jStorage]:
    """Create Neo4j storage from environment variables."""
    if not COMPONENTS_AVAILABLE:
        return None

    try:
        config = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
        return Neo4jStorage(config)
    except Exception as e:
        print(f"⚠️ Could not create Neo4j storage: {e}")
        return None


def create_qdrant_storage() -> Optional[QdrantStorage]:
    """Create Qdrant storage from environment variables."""
    if not COMPONENTS_AVAILABLE:
        return None

    try:
        config = QdrantConfig(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "morag_documents")
        )
        return QdrantStorage(config)
    except Exception as e:
        print(f"⚠️ Could not create Qdrant storage: {e}")
        return None


class ContextGenerationResult:
    """Result of context generation process."""
    
    def __init__(self):
        self.prompt = ""
        self.timestamp = datetime.now().isoformat()
        self.extracted_entities = []
        self.graph_entities = []
        self.graph_relations = []
        self.vector_chunks = []
        self.reasoning_paths = []
        self.context_score = 0.0
        self.final_response = ""
        self.processing_steps = []
        self.performance_metrics = {}
        self.error = None


class AgenticContextGenerator:
    """Agentic AI system for context generation."""
    
    def __init__(
        self,
        neo4j_storage: Optional[Neo4jStorage] = None,
        qdrant_storage: Optional[QdrantStorage] = None,
        llm_client: Optional[LLMClient] = None,
        verbose: bool = False
    ):
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        self.llm_client = llm_client
        self.verbose = verbose
        
        # Initialize components
        self.entity_agent = None
        self.entity_normalizer = None
        self.graph_traversal = None
        self.context_expansion = None

        if self.neo4j_storage and GraphTraversal:
            try:
                self.graph_traversal = GraphTraversal(self.neo4j_storage)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not initialize GraphTraversal: {e}")

        if self.neo4j_storage and ContextExpansionEngine:
            try:
                self.context_expansion = ContextExpansionEngine(self.neo4j_storage)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not initialize ContextExpansionEngine: {e}")

        if self.llm_client and EntityNormalizer:
            try:
                # Create a wrapper for the LLM client to fix the method name issue
                class LLMClientWrapper:
                    def __init__(self, llm_client):
                        self.llm_client = llm_client

                    async def complete(self, prompt: str):
                        """Wrapper to map complete() to generate()"""
                        return await self.llm_client.generate(prompt)

                    async def generate(self, prompt: str):
                        """Pass through generate method"""
                        return await self.llm_client.generate(prompt)

                wrapped_client = LLMClientWrapper(self.llm_client)
                self.entity_normalizer = EntityNormalizer(wrapped_client)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not initialize EntityNormalizer: {e}")

        if self.llm_client and EntityExtractionAgent and create_agent_with_config:
            # Initialize entity extraction agent
            try:
                self.entity_agent = create_agent_with_config(
                    EntityExtractionAgent,
                    model="google-gla:gemini-1.5-flash",
                    temperature=0.1
                )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not initialize entity agent: {e}")
    
    def _log(self, message: str, details: str = ""):
        """Log processing steps."""
        if self.verbose:
            print(f"🔄 {message}")
            if details:
                print(f"   {details}")
    
    async def generate_context(self, prompt: str) -> ContextGenerationResult:
        """Generate context for the given prompt using agentic AI."""
        result = ContextGenerationResult()
        result.prompt = prompt
        start_time = time.time()
        
        try:
            # Step 1: Entity Extraction using Agentic AI
            self._log("Step 1: Entity Extraction", "Using agentic AI to extract entities from prompt")
            result.extracted_entities = await self._extract_entities_agentic(prompt)
            result.processing_steps.append({
                "step": "entity_extraction",
                "entities_found": len(result.extracted_entities),
                "entities": [e.get("name", "") for e in result.extracted_entities]
            })
            
            # Step 2: Graph Navigation
            if self.neo4j_storage and result.extracted_entities:
                self._log("Step 2: Graph Navigation", f"Navigating graph for {len(result.extracted_entities)} entities")
                graph_data = await self._navigate_graph_agentic(result.extracted_entities, prompt)
                result.graph_entities = graph_data["entities"]
                result.graph_relations = graph_data["relations"]
                result.reasoning_paths = graph_data["reasoning_paths"]
                
                result.processing_steps.append({
                    "step": "graph_navigation",
                    "graph_entities": len(result.graph_entities),
                    "graph_relations": len(result.graph_relations),
                    "reasoning_paths": len(result.reasoning_paths)
                })
            
            # Step 3: Vector Search
            if self.qdrant_storage:
                self._log("Step 3: Vector Search", "Searching for relevant documents")
                result.vector_chunks = await self._search_vector_documents(prompt)
                result.processing_steps.append({
                    "step": "vector_search",
                    "chunks_found": len(result.vector_chunks)
                })
            
            # Step 4: Context Scoring
            self._log("Step 4: Context Scoring", "Evaluating context quality")
            result.context_score = await self._score_context(prompt, result)
            
            # Step 5: Final Response Generation
            self._log("Step 5: Response Generation", "Generating final response with context")
            result.final_response = await self._generate_final_response(prompt, result)
            
            # Performance metrics
            end_time = time.time()
            result.performance_metrics = {
                "total_time_seconds": end_time - start_time,
                "entities_extracted": len(result.extracted_entities),
                "graph_entities_found": len(result.graph_entities),
                "vector_chunks_found": len(result.vector_chunks),
                "context_score": result.context_score
            }
            
        except Exception as e:
            result.error = str(e)
            self._log("Error", f"Context generation failed: {e}")
        
        return result
    
    async def _extract_entities_agentic(self, prompt: str) -> List[Dict[str, Any]]:
        """Extract entities using agentic AI approach."""
        entities = []
        
        if not self.llm_client:
            return entities
        
        try:
            # Use LLM to extract entities with reasoning
            extraction_prompt = f"""
            Analyze this query and extract key entities that would be useful for knowledge graph traversal.
            Focus on concrete nouns, concepts, and named entities that could have relationships in a knowledge base.
            
            Query: "{prompt}"
            
            For each entity, provide:
            1. Entity name (normalized, singular form)
            2. Entity type (PERSON, ORGANIZATION, CONCEPT, MEDICAL_CONDITION, SUBSTANCE, etc.)
            3. Confidence score (0.0-1.0)
            4. Why this entity is relevant for answering the query
            
            Return as JSON array with format:
            [{{"name": "entity_name", "type": "ENTITY_TYPE", "confidence": 0.9, "relevance": "explanation"}}]
            
            Extract 3-7 most relevant entities. Be selective and focus on quality over quantity.
            """
            
            response = await self.llm_client.generate(extraction_prompt)
            
            # Parse JSON response
            try:
                import re
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    entities_data = json.loads(json_match.group())
                    entities = entities_data
                else:
                    # Fallback: parse line by line
                    lines = response.strip().split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            # Simple entity extraction from text
                            entity_name = line.strip().strip('-').strip()
                            if entity_name:
                                entities.append({
                                    "name": entity_name,
                                    "type": "CONCEPT",
                                    "confidence": 0.8,
                                    "relevance": "Extracted from response"
                                })
            except json.JSONDecodeError:
                self._log("Warning", "Could not parse JSON from entity extraction response")
            
            # Normalize entity names if normalizer is available
            if self.entity_normalizer:
                for entity in entities:
                    try:
                        normalized_name = await self.entity_normalizer.normalize_entity_name(
                            entity["name"], language="en"
                        )
                        entity["normalized_name"] = normalized_name
                    except Exception as e:
                        entity["normalized_name"] = entity["name"]
                        self._log("Warning", f"Could not normalize entity {entity['name']}: {e}")
            
        except Exception as e:
            self._log("Warning", f"Entity extraction failed: {e}")
        
        return entities
    
    async def _get_entity_translations(self, entity_name: str) -> List[str]:
        """Get translations and variants of an entity name for multilingual matching."""
        variants = [entity_name]

        if not self.llm_client:
            # Basic hardcoded translations for common substances
            translations = {
                'mercury': ['quecksilber', 'hg'],
                'quecksilber': ['mercury', 'hg'],
                'aluminum': ['aluminium'],
                'aluminium': ['aluminum', 'al'],
                'iron': ['eisen', 'fe'],
                'eisen': ['iron', 'fe'],
                'zinc': ['zink', 'zn'],
                'zink': ['zinc', 'zn'],
                'lead': ['blei', 'pb'],
                'blei': ['lead', 'pb'],
                'copper': ['kupfer', 'cu'],
                'kupfer': ['copper', 'cu'],
                'silver': ['silber', 'ag'],
                'silber': ['silver', 'ag'],
                'gold': ['gold', 'au'],
                'calcium': ['kalzium', 'calcium', 'ca'],
                'kalzium': ['calcium', 'ca'],
                'magnesium': ['magnesium', 'mg'],
                'vitamin': ['vitamin'],
                'protein': ['protein', 'eiweiß'],
                'eiweiß': ['protein'],
            }

            entity_lower = entity_name.lower()
            if entity_lower in translations:
                variants.extend(translations[entity_lower])

            return list(set(variants))

        try:
            # Use LLM to get translations
            translation_prompt = f"""
            Provide translations and common variants for this entity name in German and English:

            Entity: "{entity_name}"

            Return a simple list of variants separated by commas, including:
            - German translation
            - English translation
            - Common abbreviations
            - Scientific names if applicable

            Example for "Mercury": mercury, quecksilber, hg
            Example for "Vitamin C": vitamin c, ascorbic acid, ascorbinsäure

            Return only the comma-separated list, nothing else.
            """

            response = await self.llm_client.generate(translation_prompt)

            # Parse the response
            if response:
                new_variants = [v.strip().lower() for v in response.split(',') if v.strip()]
                variants.extend(new_variants)

        except Exception as e:
            self._log("Warning", f"Could not get translations for {entity_name}: {e}")

        # Remove duplicates and return
        return list(set([v.lower() for v in variants if v.strip()]))

    async def _navigate_graph_agentic(self, entities: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """Navigate the knowledge graph using agentic approach."""
        graph_data = {
            "entities": [],
            "relations": [],
            "reasoning_paths": []
        }
        
        if not self.neo4j_storage or not entities:
            return graph_data
        
        try:
            # For each extracted entity, find related entities and relations
            for entity_info in entities[:5]:  # Limit to first 5 entities
                entity_name = entity_info.get("normalized_name", entity_info.get("name", ""))

                # Get translations and variants for multilingual matching
                entity_variants = await self._get_entity_translations(entity_name)
                self._log("Debug", f"Searching for entity '{entity_name}' with variants: {entity_variants}")

                # Search for matching entities in the graph using all variants
                entity_query = """
                MATCH (e)
                WHERE any(variant IN $entity_variants WHERE
                    toLower(e.name) CONTAINS toLower(variant)
                    OR toLower(variant) CONTAINS toLower(e.name)
                    OR toLower(e.name) = toLower(variant)
                )
                RETURN e.name as name, labels(e) as types,
                       coalesce(e.description, '') as description,
                       e.name as id
                LIMIT 10
                """

                entity_results = await self.neo4j_storage._execute_query(
                    entity_query, {"entity_variants": entity_variants}
                )
                
                for record in entity_results:
                    graph_entity = {
                        "id": record["id"],
                        "name": record["name"],
                        "types": record["types"],
                        "description": record.get("description", ""),
                        "source_entity": entity_name
                    }
                    graph_data["entities"].append(graph_entity)
                    
                    # Find relations from this entity
                    relation_query = """
                    MATCH (e1)-[r]->(e2)
                    WHERE e1.name = $entity_name
                    RETURN e1.name as source, type(r) as relation_type, e2.name as target,
                           coalesce(r.description, '') as description,
                           e2.name as target_id
                    LIMIT 15
                    """
                    
                    relation_results = await self.neo4j_storage._execute_query(
                        relation_query, {"entity_name": record["name"]}
                    )
                    
                    for rel_record in relation_results:
                        graph_relation = {
                            "source": rel_record["source"],
                            "relation": rel_record["relation_type"],
                            "target": rel_record["target"],
                            "description": rel_record.get("description", ""),
                            "target_id": rel_record["target_id"]
                        }
                        graph_data["relations"].append(graph_relation)
            
            # Use LLM to analyze reasoning paths
            if graph_data["entities"] or graph_data["relations"]:
                reasoning_prompt = f"""
                Analyze the following knowledge graph data to identify reasoning paths that help answer this query:
                
                Query: "{prompt}"
                
                Graph Entities:
                {chr(10).join([f"- {e['name']} ({', '.join(e['types'])}): {e['description']}" for e in graph_data['entities'][:10]])}
                
                Graph Relations:
                {chr(10).join([f"- {r['source']} --{r['relation']}--> {r['target']}: {r['description']}" for r in graph_data['relations'][:15]])}
                
                Identify 2-3 key reasoning paths that connect the entities and help answer the query.
                For each path, explain how it contributes to understanding the query.
                """
                
                reasoning_analysis = await self.llm_client.generate(reasoning_prompt)
                graph_data["reasoning_paths"].append(reasoning_analysis)
        
        except Exception as e:
            self._log("Warning", f"Graph navigation failed: {e}")
        
        return graph_data

    async def _search_vector_documents(self, prompt: str) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity."""
        chunks = []

        if not self.qdrant_storage:
            return chunks

        try:
            # Search for relevant documents
            vector_results = await self.qdrant_storage.search_entities(prompt, limit=10)

            for result in vector_results:
                chunk = {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0),
                    "source": result.get("metadata", {}).get("source", "unknown")
                }
                chunks.append(chunk)

        except Exception as e:
            self._log("Warning", f"Vector search failed: {e}")

        return chunks

    async def _score_context(self, prompt: str, result: ContextGenerationResult) -> float:
        """Score the quality and relevance of the generated context."""
        if not self.llm_client:
            # Simple heuristic scoring
            entity_score = min(len(result.extracted_entities) / 5.0, 1.0)
            graph_score = min(len(result.graph_entities) / 10.0, 1.0)
            vector_score = min(len(result.vector_chunks) / 5.0, 1.0)
            return (entity_score + graph_score + vector_score) / 3.0

        try:
            # Use LLM to evaluate context quality
            scoring_prompt = f"""
            Evaluate the quality and relevance of this context for answering the given query.

            Query: "{prompt}"

            Context Summary:
            - Extracted entities: {len(result.extracted_entities)}
            - Graph entities: {len(result.graph_entities)}
            - Graph relations: {len(result.graph_relations)}
            - Vector chunks: {len(result.vector_chunks)}
            - Reasoning paths: {len(result.reasoning_paths)}

            Sample entities: {[e.get('name', '') for e in result.extracted_entities[:3]]}
            Sample relations: {[f"{r.get('source', '')} -> {r.get('target', '')}" for r in result.graph_relations[:3]]}

            Rate the context quality on a scale of 0.0 to 1.0 considering:
            1. Relevance to the query
            2. Completeness of information
            3. Quality of entity extraction
            4. Usefulness of graph connections
            5. Document relevance

            Return only a single number between 0.0 and 1.0.
            """

            score_response = await self.llm_client.generate(scoring_prompt)

            # Extract numeric score
            import re
            score_match = re.search(r'(\d+\.?\d*)', score_response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

        except Exception as e:
            self._log("Warning", f"Context scoring failed: {e}")

        # Fallback to heuristic scoring
        entity_score = min(len(result.extracted_entities) / 5.0, 1.0)
        graph_score = min(len(result.graph_entities) / 10.0, 1.0)
        vector_score = min(len(result.vector_chunks) / 5.0, 1.0)
        return (entity_score + graph_score + vector_score) / 3.0

    async def _generate_final_response(self, prompt: str, result: ContextGenerationResult) -> str:
        """Generate the final response using all gathered context."""
        if not self.llm_client:
            return "LLM client not available for response generation."

        try:
            # Build context sections
            entity_context = ""
            if result.extracted_entities:
                entity_context = "\n\nEXTRACTED ENTITIES:\n" + "\n".join([
                    f"- {e.get('name', '')} ({e.get('type', 'UNKNOWN')}): {e.get('relevance', '')}"
                    for e in result.extracted_entities
                ])

            graph_context = ""
            if result.graph_entities or result.graph_relations:
                graph_context = "\n\nKNOWLEDGE GRAPH INFORMATION:\n"
                if result.graph_entities:
                    graph_context += "Entities:\n" + "\n".join([
                        f"- {e.get('name', '')} ({', '.join(e.get('types', []))}): {e.get('description', '')}"
                        for e in result.graph_entities[:10]
                    ])
                if result.graph_relations:
                    graph_context += "\nRelations:\n" + "\n".join([
                        f"- {r.get('source', '')} --{r.get('relation', '')}--> {r.get('target', '')}: {r.get('description', '')}"
                        for r in result.graph_relations[:15]
                    ])

            vector_context = ""
            if result.vector_chunks:
                vector_context = "\n\nRELEVANT DOCUMENTS:\n" + "\n".join([
                    f"- {chunk.get('content', '')[:200]}..." if len(chunk.get('content', '')) > 200
                    else f"- {chunk.get('content', '')}"
                    for chunk in result.vector_chunks[:5]
                ])

            reasoning_context = ""
            if result.reasoning_paths:
                reasoning_context = "\n\nREASONING ANALYSIS:\n" + "\n".join(result.reasoning_paths)

            # Generate final response
            synthesis_prompt = f"""
            You are a knowledgeable assistant. Provide a comprehensive, accurate answer to this query using the provided context.
            Start directly with the main content without preambles or conversational phrases.

            Query: "{prompt}"

            Context Quality Score: {result.context_score:.2f}/1.0
            {entity_context}
            {graph_context}
            {vector_context}
            {reasoning_context}

            Synthesize all available information to provide a detailed, well-reasoned response.
            If the context is insufficient, clearly state what information is missing.
            Be direct and factual in your response.
            """

            final_response = await self.llm_client.generate(synthesis_prompt)
            return final_response

        except Exception as e:
            return f"Error generating response: {e}"


async def test_database_connections(args) -> Tuple[Optional[Neo4jStorage], Optional[QdrantStorage]]:
    """Test and establish database connections."""
    neo4j_storage = None
    qdrant_storage = None

    if args.neo4j:
        try:
            neo4j_storage = create_neo4j_storage()
            if neo4j_storage:
                await neo4j_storage.connect()
                print("✅ Neo4j connection established")
            else:
                print("❌ Failed to create Neo4j storage")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")

    if args.qdrant:
        try:
            qdrant_storage = create_qdrant_storage()
            if qdrant_storage:
                await qdrant_storage.connect()
                print("✅ Qdrant connection established")
            else:
                print("❌ Failed to create Qdrant storage")
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")

    return neo4j_storage, qdrant_storage


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test MoRAG context generation with agentic AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test-context-generation.py --neo4j --qdrant "How does nutrition affect ADHD symptoms?"
  python test-context-generation.py --neo4j --verbose "What are the connections between AI and healthcare?"
  python test-context-generation.py --qdrant --output results.json "Explain machine learning applications"
"""
    )

    # Database options
    parser.add_argument("--neo4j", action="store_true", help="Use Neo4j for graph operations")
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant for vector search")

    # LLM options
    parser.add_argument("--model", type=str, help="LLM model to use (default: from env or gemini-1.5-flash)")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum tokens for LLM responses")

    # Output options
    parser.add_argument("--verbose", action="store_true", help="Show detailed processing steps")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--show-context", action="store_true", help="Display the generated context")

    # The prompt
    parser.add_argument("prompt", help="The prompt to process")

    args = parser.parse_args()

    if not COMPONENTS_AVAILABLE:
        print("❌ MoRAG components not available. Please install required packages.")
        return 1

    if not args.neo4j and not args.qdrant:
        print("⚠️ No databases specified. Running in LLM-only mode for testing.")
        print("   Use --neo4j and/or --qdrant for full functionality.")

    print(f"🚀 Testing context generation for: '{args.prompt}'")
    print(f"📊 Databases: Neo4j={args.neo4j}, Qdrant={args.qdrant}")
    print(f"🤖 Model: {args.model or os.getenv('MORAG_GEMINI_MODEL', 'gemini-1.5-flash')}")

    # Test database connections
    neo4j_storage, qdrant_storage = await test_database_connections(args)

    if not neo4j_storage and not qdrant_storage and (args.neo4j or args.qdrant):
        print("❌ No database connections available")
        return 1

    # Initialize LLM client
    model = args.model or os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash")
    llm_config = LLMConfig(
        provider="gemini",
        api_key=os.getenv("GEMINI_API_KEY"),
        model=model,
        temperature=0.1,
        max_tokens=args.max_tokens
    )
    llm_client = LLMClient(llm_config)

    # Initialize context generator
    generator = AgenticContextGenerator(
        neo4j_storage=neo4j_storage,
        qdrant_storage=qdrant_storage,
        llm_client=llm_client,
        verbose=args.verbose
    )

    try:
        # Generate context
        result = await generator.generate_context(args.prompt)

        if result.error:
            print(f"❌ Context generation failed: {result.error}")
            return 1

        # Display results
        print(f"\n{'='*80}")
        print("🎯 CONTEXT GENERATION RESULTS")
        print(f"{'='*80}")

        print(f"📊 Context Quality Score: {result.context_score:.2f}/1.0")
        print(f"⚡ Processing Time: {result.performance_metrics.get('total_time_seconds', 0):.2f}s")
        print(f"🔍 Entities Extracted: {len(result.extracted_entities)}")
        print(f"🕸️  Graph Entities: {len(result.graph_entities)}")
        print(f"🔗 Graph Relations: {len(result.graph_relations)}")
        print(f"📄 Vector Chunks: {len(result.vector_chunks)}")

        if args.show_context:
            print(f"\n{'='*80}")
            print("📋 GENERATED CONTEXT")
            print(f"{'='*80}")

            if result.extracted_entities:
                print("\n🔍 Extracted Entities:")
                for entity in result.extracted_entities:
                    print(f"  - {entity.get('name', '')} ({entity.get('type', 'UNKNOWN')})")

            if result.graph_entities:
                print(f"\n🕸️  Graph Entities ({len(result.graph_entities)}):")
                for entity in result.graph_entities[:5]:
                    print(f"  - {entity.get('name', '')} ({', '.join(entity.get('types', []))})")

            if result.graph_relations:
                print(f"\n🔗 Graph Relations ({len(result.graph_relations)}):")
                # Show more relations when we have good results
                max_relations = 10 if len(result.graph_relations) > 20 else 5
                for relation in result.graph_relations[:max_relations]:
                    print(f"  - {relation.get('source', '')} --{relation.get('relation', '')}--> {relation.get('target', '')}")
                if len(result.graph_relations) > max_relations:
                    print(f"  ... and {len(result.graph_relations) - max_relations} more relations")

        print(f"\n{'='*80}")
        print("🎯 FINAL RESPONSE")
        print(f"{'='*80}")
        print(result.final_response)

        # Save results if requested
        if args.output:
            output_data = {
                "prompt": result.prompt,
                "timestamp": result.timestamp,
                "context_score": result.context_score,
                "extracted_entities": result.extracted_entities,
                "graph_entities": result.graph_entities,
                "graph_relations": result.graph_relations,
                "vector_chunks": result.vector_chunks,
                "reasoning_paths": result.reasoning_paths,
                "final_response": result.final_response,
                "processing_steps": result.processing_steps,
                "performance_metrics": result.performance_metrics
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")

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
