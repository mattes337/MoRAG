"""Recursive fact retrieval service implementing the graph-based RAG system."""

import uuid
import json
import structlog
from datetime import datetime
from collections import deque
from typing import List, Set, Dict, Any, Optional, Tuple

from morag_reasoning.llm import LLMClient
from morag_reasoning.recursive_fact_models import (
    RecursiveFactRetrievalRequest, RecursiveFactRetrievalResponse,
    RawFact, ScoredFact, FinalFact, TraversalStep
)
from morag_reasoning.graph_traversal_agent import GraphTraversalAgent
from morag_reasoning.fact_critic_agent import FactCriticAgent
from morag_reasoning.entity_identification import EntityIdentificationService
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_graph.storage.qdrant_storage import QdrantStorage


class RecursiveFactRetrievalService:
    """Service implementing the complete recursive fact retrieval system."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        neo4j_storage: Neo4jStorage,
        qdrant_storage: QdrantStorage,
        stronger_llm_client: Optional[LLMClient] = None
    ):
        """Initialize the recursive fact retrieval service.
        
        Args:
            llm_client: LLM client for GTA and FCA operations
            neo4j_storage: Neo4j storage for graph operations
            qdrant_storage: Qdrant storage for vector operations
            stronger_llm_client: Optional stronger LLM for final synthesis
        """
        self.llm_client = llm_client
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        self.stronger_llm_client = stronger_llm_client or llm_client
        self.logger = structlog.get_logger(__name__)
        
        # Initialize sub-components
        self.entity_service = EntityIdentificationService(
            llm_client=llm_client,
            graph_storage=neo4j_storage,
            min_confidence=0.2,
            max_entities=20
        )
        
        self.graph_traversal_agent = GraphTraversalAgent(
            llm_client=llm_client,
            neo4j_storage=neo4j_storage,
            qdrant_storage=qdrant_storage
        )
        
        self.fact_critic_agent = FactCriticAgent(llm_client=llm_client)
    
    async def retrieve_facts_recursively(
        self,
        request: RecursiveFactRetrievalRequest
    ) -> RecursiveFactRetrievalResponse:
        """Perform recursive fact retrieval using the graph-based RAG system.
        
        Args:
            request: Recursive fact retrieval request
            
        Returns:
            Response with extracted facts and final answer
        """
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        # Initialize counters
        gta_llm_calls = 0
        fca_llm_calls = 0
        final_llm_calls = 0
        
        self.logger.info(
            "Starting recursive fact retrieval",
            query_id=query_id,
            query=request.user_query,
            max_depth=request.max_depth
        )
        
        try:
            # Step 1: Extract initial entities from user query
            self.logger.info("Step 1: Extracting initial entities")
            initial_entities = await self._extract_initial_entities(request.user_query)
            
            if not initial_entities:
                return self._create_error_response(
                    query_id, request, start_time,
                    "Could not identify key entities in your query to start graph traversal."
                )
            
            # Step 2: Map entities to graph nodes
            self.logger.info("Step 2: Mapping entities to graph nodes")
            initial_node_ids = await self._map_entities_to_nodes(initial_entities)
            
            if not initial_node_ids:
                return self._create_error_response(
                    query_id, request, start_time,
                    "No relevant starting nodes found in the knowledge graph for your query."
                )
            
            # Step 3: Perform graph traversal and fact extraction
            self.logger.info("Step 3: Performing graph traversal")
            all_raw_facts, traversal_steps, max_depth_reached = await self._perform_graph_traversal(
                request, initial_node_ids
            )
            gta_llm_calls = len(traversal_steps)  # One LLM call per traversal step
            
            if not all_raw_facts:
                return self._create_error_response(
                    query_id, request, start_time,
                    "No relevant facts could be extracted from the knowledge graph."
                )
            
            # Step 4: Evaluate and score facts
            self.logger.info("Step 4: Evaluating and scoring facts")
            scored_facts = await self.fact_critic_agent.batch_evaluate_facts(
                request.user_query, all_raw_facts, language=request.language
            )
            fca_llm_calls = len(all_raw_facts)  # One LLM call per fact
            
            # Step 5: Apply relevance decay
            self.logger.info("Step 5: Applying relevance decay")
            final_facts = self.fact_critic_agent.apply_relevance_decay(
                scored_facts, request.decay_rate
            )
            
            # Filter facts by minimum score
            final_facts = [
                fact for fact in final_facts 
                if fact.final_decayed_score >= request.min_fact_score
            ]
            
            # Limit total facts
            final_facts = final_facts[:request.max_total_facts]
            
            # Step 6: Generate final answer (unless facts_only is requested)
            final_answer = None
            confidence_score = 0.0
            if not request.facts_only:
                self.logger.info("Step 6: Generating final answer")
                final_answer, confidence_score = await self._generate_final_answer(
                    request.user_query, final_facts, request.language
                )
                final_llm_calls = 1
            else:
                self.logger.info("Step 6: Skipping final answer generation (facts_only=True)")
                # Calculate confidence based on fact scores
                if final_facts:
                    avg_score = sum(fact.final_decayed_score for fact in final_facts) / len(final_facts)
                    confidence_score = min(1.0, avg_score * (len(final_facts) / 10))
                final_llm_calls = 0
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Create response
            response = RecursiveFactRetrievalResponse(
                query_id=query_id,
                user_query=request.user_query,
                processing_time_ms=processing_time_ms,
                initial_entities=initial_entities,
                total_nodes_explored=len(traversal_steps),
                max_depth_reached=max_depth_reached,
                traversal_steps=traversal_steps,
                total_raw_facts=len(all_raw_facts),
                total_scored_facts=len(scored_facts),
                final_facts=final_facts,
                gta_llm_calls=gta_llm_calls,
                fca_llm_calls=fca_llm_calls,
                final_llm_calls=final_llm_calls,
                final_answer=final_answer,
                confidence_score=confidence_score
            )
            
            self.logger.info(
                "Recursive fact retrieval completed",
                query_id=query_id,
                total_facts=len(final_facts),
                processing_time_ms=processing_time_ms
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Error in recursive fact retrieval",
                query_id=query_id,
                error=str(e)
            )
            return self._create_error_response(
                query_id, request, start_time,
                f"An error occurred during fact retrieval: {str(e)}"
            )
    
    async def _extract_initial_entities(self, user_query: str) -> List[str]:
        """Extract initial entities from the user query."""
        try:
            identified_entities = await self.entity_service.identify_entities(user_query)
            return [entity.name for entity in identified_entities if entity.confidence >= 0.3]
        except Exception as e:
            self.logger.error("Failed to extract initial entities", error=str(e))
            return []
    
    async def _map_entities_to_nodes(self, entities: List[str]) -> List[str]:
        """Map entity names to graph node IDs."""
        node_ids = []
        for entity in entities:
            try:
                # Search for nodes by name
                nodes = await self.neo4j_storage.search_entities(entity, limit=3)
                for node in nodes:
                    if node.id not in node_ids:
                        node_ids.append(node.id)
            except Exception as e:
                self.logger.warning("Failed to map entity to node", entity=entity, error=str(e))

        return node_ids
    
    async def _perform_graph_traversal(
        self,
        request: RecursiveFactRetrievalRequest,
        initial_node_ids: List[str]
    ) -> Tuple[List[RawFact], List[TraversalStep], int]:
        """Perform the main graph traversal and fact extraction."""
        all_raw_facts = []
        traversal_steps = []
        max_depth_reached = 0
        
        # Queue for breadth-first traversal: (node_id, current_depth)
        nodes_to_explore_queue = deque([(node_id, 0) for node_id in initial_node_ids])
        visited_nodes = set(initial_node_ids)
        
        while nodes_to_explore_queue and len(all_raw_facts) < request.max_total_facts:
            current_node_id, current_depth = nodes_to_explore_queue.popleft()
            
            # Stop if max depth reached
            if current_depth >= request.max_depth:
                continue
            
            max_depth_reached = max(max_depth_reached, current_depth)
            
            try:
                # Perform traversal and extraction for current node
                gta_response = await self.graph_traversal_agent.traverse_and_extract(
                    user_query=request.user_query,
                    current_node_id=current_node_id,
                    traversal_depth=current_depth,
                    max_depth=request.max_depth,
                    visited_nodes=visited_nodes,
                    language=request.language
                )
                
                # Record traversal step
                node_name = await self._get_node_name(current_node_id)
                step = TraversalStep(
                    node_id=current_node_id,
                    node_name=node_name,
                    depth=current_depth,
                    facts_extracted=len(gta_response.extracted_facts),
                    next_nodes_decision=gta_response.next_nodes_to_explore,
                    reasoning=gta_response.reasoning
                )
                traversal_steps.append(step)
                
                # Collect facts
                all_raw_facts.extend(gta_response.extracted_facts)
                
                # Plan next traversal
                if gta_response.next_nodes_to_explore not in ["STOP_TRAVERSAL", "NONE"]:
                    next_nodes = self._parse_next_nodes(gta_response.next_nodes_to_explore)
                    for next_node_id in next_nodes:
                        if next_node_id not in visited_nodes:
                            nodes_to_explore_queue.append((next_node_id, current_depth + 1))
                            visited_nodes.add(next_node_id)
                
            except Exception as e:
                self.logger.warning(
                    "Error in traversal step",
                    node_id=current_node_id,
                    depth=current_depth,
                    error=str(e)
                )
        
        return all_raw_facts, traversal_steps, max_depth_reached
    
    async def _get_node_name(self, node_id: str) -> str:
        """Get human-readable name for a node."""
        try:
            entity = await self.neo4j_storage.get_entity(node_id)
            if entity:
                return entity.name
        except Exception:
            pass
        return node_id
    
    def _parse_next_nodes(self, next_nodes_str: str) -> List[str]:
        """Parse next nodes string into list of node IDs."""
        try:
            # Handle format: "(node_id1, rel_type1), (node_id2, rel_type2)"
            if next_nodes_str.strip().startswith('('):
                # Extract node IDs from tuples
                import re
                matches = re.findall(r'\(([^,]+),', next_nodes_str)
                return [match.strip() for match in matches]
            else:
                # Handle comma-separated node IDs
                return [node.strip() for node in next_nodes_str.split(',') if node.strip()]
        except Exception as e:
            self.logger.warning("Failed to parse next nodes", next_nodes_str=next_nodes_str, error=str(e))
            return []
    
    async def _generate_final_answer(
        self,
        user_query: str,
        final_facts: List[FinalFact],
        language: Optional[str] = None
    ) -> Tuple[str, float]:
        """Generate the final answer using the stronger LLM."""
        if not final_facts:
            return "I could not find enough relevant information to answer your query.", 0.0
        
        # Prepare context for final LLM
        formatted_facts = []
        for fact in final_facts:
            # Create detailed source information
            source_details = fact.source_description
            if fact.source_metadata.document_name:
                source_parts = [fact.source_metadata.document_name]
                if fact.source_metadata.chunk_index is not None:
                    source_parts.append(f"chunk {fact.source_metadata.chunk_index}")
                if fact.source_metadata.page_number:
                    source_parts.append(f"page {fact.source_metadata.page_number}")
                if fact.source_metadata.section:
                    source_parts.append(f"section '{fact.source_metadata.section}'")
                if fact.source_metadata.timestamp:
                    source_parts.append(f"at {fact.source_metadata.timestamp}")
                source_details = ", ".join(source_parts)

            formatted_facts.append(
                f"Fact (Score: {fact.final_decayed_score:.2f}, Source: {source_details}): {fact.fact_text}"
            )

        context = "\n\n".join(formatted_facts)

        # Add language specification if provided
        language_instruction = ""
        if language:
            language_names = {
                'en': 'English',
                'de': 'German',
                'fr': 'French',
                'es': 'Spanish',
                'it': 'Italian',
                'pt': 'Portuguese',
                'nl': 'Dutch',
                'ru': 'Russian',
                'zh': 'Chinese',
                'ja': 'Japanese',
                'ko': 'Korean'
            }
            language_name = language_names.get(language, language)
            language_instruction = f"\n\nIMPORTANT: Please respond in {language_name} ({language}). The entire response must be in {language_name}."

        prompt = f"""Based on the following facts extracted from a knowledge graph, please provide a comprehensive answer to the user's question.

User Question: "{user_query}"

Relevant Facts:
{context}

Please synthesize these facts into a coherent, well-structured answer. Focus on directly addressing the user's question while incorporating the most relevant information from the facts. If the facts don't fully answer the question, acknowledge what information is available and what might be missing.{language_instruction}"""
        
        try:
            # Use the stronger LLM for final synthesis
            response = await self.stronger_llm_client.generate(prompt)
            
            # Calculate confidence based on fact scores and coverage
            avg_score = sum(fact.final_decayed_score for fact in final_facts) / len(final_facts)
            confidence_score = min(1.0, avg_score * (len(final_facts) / 10))  # Boost confidence with more facts
            
            return response, confidence_score
            
        except Exception as e:
            self.logger.error("Failed to generate final answer", error=str(e))
            return f"An error occurred while generating the final answer: {str(e)}", 0.0
    
    def _create_error_response(
        self,
        query_id: str,
        request: RecursiveFactRetrievalRequest,
        start_time: datetime,
        error_message: str
    ) -> RecursiveFactRetrievalResponse:
        """Create an error response."""
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return RecursiveFactRetrievalResponse(
            query_id=query_id,
            user_query=request.user_query,
            processing_time_ms=processing_time_ms,
            initial_entities=[],
            total_nodes_explored=0,
            max_depth_reached=0,
            traversal_steps=[],
            total_raw_facts=0,
            total_scored_facts=0,
            final_facts=[],
            gta_llm_calls=0,
            fca_llm_calls=0,
            final_llm_calls=0,
            final_answer=error_message,
            confidence_score=0.0
        )
