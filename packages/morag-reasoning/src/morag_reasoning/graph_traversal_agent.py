"""GraphTraversalAgent for intelligent graph traversal and fact extraction."""

import json
import structlog
from typing import List, Set, Dict, Any, Optional
from pydantic_ai import Agent
from pydantic import BaseModel, Field

from morag_reasoning.llm import LLMClient
from morag_reasoning.recursive_fact_models import RawFact, GTAResponse
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_graph.storage.qdrant_storage import QdrantStorage


class NodeContext(BaseModel):
    """Context information about a node for the GTA."""
    node_id: str = Field(..., description="Node ID")
    node_properties: Dict[str, Any] = Field(..., description="Node properties from Neo4j")
    qdrant_content: List[Dict[str, Any]] = Field(..., description="Associated content from Qdrant")
    neighbors_and_relations: List[Dict[str, Any]] = Field(..., description="Immediate neighbors and their relationships")


class GraphTraversalAgent:
    """Agent responsible for navigating the Neo4j graph and extracting raw facts."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        neo4j_storage: Neo4jStorage,
        qdrant_storage: QdrantStorage,
        max_facts_per_node: int = 5
    ):
        """Initialize the GraphTraversalAgent.
        
        Args:
            llm_client: LLM client for AI operations
            neo4j_storage: Neo4j storage for graph operations
            qdrant_storage: Qdrant storage for vector operations
            max_facts_per_node: Maximum facts to extract per node
        """
        self.llm_client = llm_client
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        self.max_facts_per_node = max_facts_per_node
        self.logger = structlog.get_logger(__name__)
        
        # Create PydanticAI agent for graph traversal decisions
        self.agent = Agent(
            model=llm_client.get_model(),
            result_type=GTAResponse,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the GraphTraversalAgent."""
        return f"""You are a GraphTraversalAgent (GTA) responsible for intelligent graph traversal and fact extraction.

Your role is to:
1. Analyze the current node's context (properties, content, neighbors)
2. Extract potentially relevant facts from the provided context
3. Decide on the next nodes to explore based on the user query

FACT EXTRACTION GUIDELINES:
- Extract up to {self.max_facts_per_node} comprehensive, detailed factual statements
- Focus on information that could be relevant to answering the user query
- Each fact should be a complete, standalone statement with full context and relevant details
- Include facts from node properties, Qdrant content, and relationship information
- Be specific and include all relevant context, numbers, dates, and supporting details
- Make facts extensive and self-contained so they can be used independently for synthesis

NEXT NODE DECISION GUIDELINES:
- Return "STOP_TRAVERSAL" if you believe sufficient information has been gathered
- Return "NONE" if no promising neighbors exist for exploration
- Return comma-separated (node_id, relationship_type) tuples for nodes to explore next
- Consider semantic relevance to the query, potential for new information discovery
- Prioritize nodes that are likely to contain relevant information
- Avoid cycles by checking visited nodes

RESPONSE FORMAT:
- extracted_facts: Array of RawFact objects with fact_text, source_node_id, source_property (if from property), source_qdrant_chunk_id (if from content), source_metadata (with document details), and extracted_from_depth
- next_nodes_to_explore: String decision as described above
- reasoning: Brief explanation of your decision-making process

Be strategic and focused - aim for comprehensive but efficient exploration."""

    def _chunk_id_to_point_id(self, chunk_id: str) -> int:
        """Convert chunk ID to Qdrant point ID."""
        return abs(hash(chunk_id)) % (2**31)
    
    async def _get_node_context(self, node_id: str) -> NodeContext:
        """Get comprehensive context for a node.
        
        Args:
            node_id: Node ID to get context for
            
        Returns:
            NodeContext with all relevant information
        """
        try:
            # Get node properties from Neo4j
            node_properties = {}
            try:
                entity = await self.neo4j_storage.get_entity(node_id)
                if entity:
                    node_properties = {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type,
                        "confidence": getattr(entity, 'confidence', 1.0),
                        "properties": getattr(entity, 'properties', {})
                    }
            except Exception as e:
                self.logger.warning("Failed to get node properties", node_id=node_id, error=str(e))
            
            # Get associated Qdrant content
            qdrant_content = []
            try:
                # Try to find chunks associated with this entity
                chunks = await self.qdrant_storage.get_chunks_by_entity_id(node_id)
                for chunk in chunks:
                    metadata = chunk.get("metadata", {})
                    qdrant_content.append({
                        "chunk_id": chunk["chunk_id"],
                        "content": metadata.get("text", ""),
                        "document_name": metadata.get("document_name", ""),
                        "chunk_index": metadata.get("chunk_index"),
                        "page_number": metadata.get("page_number"),
                        "section": metadata.get("section"),
                        "timestamp": metadata.get("timestamp"),
                        "additional_metadata": {k: v for k, v in metadata.items()
                                              if k not in ["text", "document_name", "chunk_index", "page_number", "section", "timestamp"]},
                        "score": 1.0  # No similarity score for entity-based retrieval
                    })
            except Exception as e:
                self.logger.warning("Failed to get Qdrant content", node_id=node_id, error=str(e))
            
            # Get neighbors and relationships
            neighbors_and_relations = []
            try:
                neighbors = await self.neo4j_storage.get_neighbors(node_id, max_depth=1)
                for neighbor in neighbors[:20]:  # Limit to 20 neighbors
                    neighbors_and_relations.append({
                        "neighbor_id": neighbor.id,
                        "neighbor_name": neighbor.name,
                        "neighbor_type": neighbor.type,
                        "relationship_type": "RELATED_TO",  # Simplified - could get actual relationship types
                        "relationship_properties": {}
                    })
            except Exception as e:
                self.logger.warning("Failed to get neighbors", node_id=node_id, error=str(e))
            
            return NodeContext(
                node_id=node_id,
                node_properties=node_properties,
                qdrant_content=qdrant_content,
                neighbors_and_relations=neighbors_and_relations
            )
            
        except Exception as e:
            self.logger.error("Failed to get node context", node_id=node_id, error=str(e))
            # Return minimal context to avoid breaking the flow
            return NodeContext(
                node_id=node_id,
                node_properties={"id": node_id},
                qdrant_content=[],
                neighbors_and_relations=[]
            )
    
    async def traverse_and_extract(
        self,
        user_query: str,
        current_node_id: str,
        traversal_depth: int,
        max_depth: int,
        visited_nodes: Set[str],
        graph_schema: Optional[str] = None,
        language: Optional[str] = None
    ) -> GTAResponse:
        """Perform graph traversal and fact extraction for a single node.

        Args:
            user_query: Original user query
            current_node_id: Current node being explored
            traversal_depth: Current traversal depth
            max_depth: Maximum allowed depth
            visited_nodes: Set of already visited node IDs
            graph_schema: Optional graph schema information
            language: Optional language for fact extraction

        Returns:
            GTAResponse with extracted facts and next node decisions
        """
        self.logger.info(
            "Starting traversal and extraction",
            node_id=current_node_id,
            depth=traversal_depth,
            max_depth=max_depth
        )
        
        try:
            # Get comprehensive context for the current node
            context = await self._get_node_context(current_node_id)
            
            # Prepare prompt for the LLM
            prompt = self._create_traversal_prompt(
                user_query=user_query,
                context=context,
                traversal_depth=traversal_depth,
                max_depth=max_depth,
                visited_nodes=visited_nodes,
                graph_schema=graph_schema,
                language=language
            )
            
            # Call LLM for traversal decision and fact extraction
            result = await self.agent.run(prompt)
            response = result.data
            
            # Validate and enhance the extracted facts
            enhanced_facts = []
            for fact in response.extracted_facts:
                # Ensure depth is set correctly
                fact.extracted_from_depth = traversal_depth

                # Enhance source metadata if fact comes from Qdrant content
                if fact.source_qdrant_chunk_id:
                    # Find the corresponding chunk in context
                    for chunk_info in context.qdrant_content:
                        if chunk_info["chunk_id"] == fact.source_qdrant_chunk_id:
                            from morag_reasoning.recursive_fact_models import SourceMetadata
                            fact.source_metadata = SourceMetadata(
                                document_name=chunk_info.get("document_name"),
                                chunk_index=chunk_info.get("chunk_index"),
                                page_number=chunk_info.get("page_number"),
                                section=chunk_info.get("section"),
                                timestamp=chunk_info.get("timestamp"),
                                additional_metadata=chunk_info.get("additional_metadata", {})
                            )
                            break

                enhanced_facts.append(fact)
            
            # Create final response
            final_response = GTAResponse(
                extracted_facts=enhanced_facts,
                next_nodes_to_explore=response.next_nodes_to_explore,
                reasoning=response.reasoning
            )
            
            self.logger.info(
                "Traversal and extraction completed",
                node_id=current_node_id,
                facts_extracted=len(enhanced_facts),
                next_decision=response.next_nodes_to_explore
            )
            
            return final_response
            
        except Exception as e:
            self.logger.error(
                "Error in traversal and extraction",
                node_id=current_node_id,
                error=str(e)
            )
            # Return empty response to avoid breaking the flow
            return GTAResponse(
                extracted_facts=[],
                next_nodes_to_explore="NONE",
                reasoning=f"Error occurred during traversal: {str(e)}"
            )
    
    def _create_traversal_prompt(
        self,
        user_query: str,
        context: NodeContext,
        traversal_depth: int,
        max_depth: int,
        visited_nodes: Set[str],
        graph_schema: Optional[str] = None,
        language: Optional[str] = None
    ) -> str:
        """Create the prompt for the traversal LLM."""
        
        # Format context information
        node_info = f"Node ID: {context.node_id}\n"
        node_info += f"Properties: {json.dumps(context.node_properties, indent=2)}\n"
        
        if context.qdrant_content:
            node_info += f"\nAssociated Content ({len(context.qdrant_content)} chunks):\n"
            for i, content in enumerate(context.qdrant_content[:5]):  # Limit to first 5 chunks
                node_info += f"  Chunk {i+1}: {content['content'][:200]}...\n"
        
        if context.neighbors_and_relations:
            node_info += f"\nNeighbors ({len(context.neighbors_and_relations)} total):\n"
            for neighbor in context.neighbors_and_relations[:10]:  # Limit to first 10 neighbors
                node_info += f"  - {neighbor['neighbor_name']} (ID: {neighbor['neighbor_id']}, Type: {neighbor['neighbor_type']})\n"
        
        visited_list = list(visited_nodes)
        
        prompt = f"""GRAPH TRAVERSAL AND FACT EXTRACTION TASK

User Query: "{user_query}"

Current Context:
{node_info}

Traversal Information:
- Current Depth: {traversal_depth}
- Maximum Depth: {max_depth}
- Visited Nodes: {visited_list}

{f"Graph Schema: {graph_schema}" if graph_schema else ""}

Your task:
1. Extract relevant facts from the current node's context
2. Decide which neighbors (if any) to explore next

Focus on information that helps answer the user query. Be strategic about which paths to follow."""

        # Add language instruction if specified
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
            prompt += f"\n\nIMPORTANT: Extract facts in {language_name} ({language}). All fact text must be in {language_name}."

        return prompt
