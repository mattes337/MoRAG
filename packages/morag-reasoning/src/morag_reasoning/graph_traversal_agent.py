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
- Extract MANY comprehensive, detailed factual statements (aim for {self.max_facts_per_node} facts)
- Be AGGRESSIVE in fact extraction - extract multiple facts from each relevant chunk
- Focus on information that could be relevant to answering the user query
- Each fact should be a complete, standalone statement with full context and relevant details
- Include facts from entity properties, associated content, and relationship information
- Be specific and include all relevant context, numbers, dates, and supporting details
- Make facts extensive and self-contained so they can be used independently for synthesis
- IMPORTANT: Reference entities by their names rather than technical IDs
- Focus on meaningful entity relationships and content, not database metadata
- Extract specific items, recommendations, studies, and quantitative information when available
- Extract both positive recommendations (what to do) and negative ones (what to avoid)
- Include concrete actionable advice, scientific findings, and practical guidance

NEXT NODE DECISION GUIDELINES:
- Return "STOP_TRAVERSAL" if you believe sufficient information has been gathered
- Return "NONE" if no promising neighbors exist for exploration
- Return comma-separated (node_id, relationship_type) tuples for nodes to explore next
- Consider semantic relevance to the query, potential for new information discovery
- Prioritize nodes that are likely to contain relevant information
- Avoid cycles by checking visited nodes

RESPONSE FORMAT:
- extracted_facts: Array of RawFact objects with fact_text (using entity names, not IDs), source_node_id, source_property (if from property), source_qdrant_chunk_id (if from content), source_metadata (with document details), and extracted_from_depth
- next_nodes_to_explore: String decision as described above
- reasoning: Brief explanation of your decision-making process

Be strategic and focused - aim for comprehensive but efficient exploration.
Remember: Use entity names in fact descriptions to make them user-friendly and meaningful."""

    def _chunk_id_to_point_id(self, chunk_id: str) -> int:
        """Convert chunk ID to Qdrant point ID."""
        return abs(hash(chunk_id)) % (2**31)

    async def _get_entity_name(self, node_id: str) -> str:
        """Get the entity name for a node ID.

        Args:
            node_id: Node ID to get name for

        Returns:
            Entity name or fallback description
        """
        try:
            entity = await self.neo4j_storage.get_entity(node_id)
            if entity and entity.name:
                return entity.name
            return "Unknown Entity"
        except Exception:
            return "Unknown Entity"
    
    async def _get_document_chunks_for_entities(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """Get DocumentChunk nodes for specific entity names.

        Args:
            entity_names: List of entity names to get chunks for

        Returns:
            List of chunk data with full metadata
        """
        try:
            chunks = await self.neo4j_storage.get_document_chunks_by_entity_names(entity_names)
            self.logger.debug(f"Found {len(chunks)} chunks for entities: {entity_names}")
            return chunks
        except Exception as e:
            self.logger.error("Failed to get document chunks for entities",
                            entity_names=entity_names, error=str(e))
            return []

    async def _get_related_entity_names(self, entity_names: List[str]) -> List[str]:
        """Get names of entities related to the given entities through graph relationships.

        Args:
            entity_names: List of entity names to find related entities for

        Returns:
            List of related entity names
        """
        try:
            query = """
            MATCH (e1)-[r]-(e2)
            WHERE e1.type IS NOT NULL AND e2.type IS NOT NULL
            AND e1.name IN $entity_names
            AND NOT e2.name IN $entity_names
            RETURN DISTINCT e2.name as related_entity_name
            LIMIT 50
            """

            result = await self.neo4j_storage._execute_query(query, {"entity_names": entity_names})
            related_names = [record["related_entity_name"] for record in result]

            self.logger.debug(f"Found {len(related_names)} related entities for: {entity_names}")
            return related_names

        except Exception as e:
            self.logger.error("Failed to get related entity names",
                            entity_names=entity_names, error=str(e))
            return []
    
    async def extract_facts_from_entity_chunks(
        self,
        user_query: str,
        entity_names: List[str],
        traversal_depth: int,
        language: Optional[str] = None
    ) -> GTAResponse:
        """Extract facts from DocumentChunk nodes related to specific entities.

        Args:
            user_query: Original user query
            entity_names: List of entity names to extract facts for
            traversal_depth: Current traversal depth
            language: Optional language for fact extraction

        Returns:
            GTAResponse with extracted facts and next entity names to explore
        """
        self.logger.info(
            "Extracting facts from entity chunks",
            entity_names=entity_names,
            depth=traversal_depth
        )

        try:
            # Get DocumentChunk nodes for these entities
            chunks = await self._get_document_chunks_for_entities(entity_names)

            if not chunks:
                self.logger.warning("No chunks found for entities", entity_names=entity_names)
                return GTAResponse(
                    extracted_facts=[],
                    next_nodes_to_explore="NONE",
                    reasoning=f"No document chunks found for entities: {', '.join(entity_names)}"
                )

            # Get related entities from graph relationships for next traversal
            related_entities_from_graph = []
            try:
                related_entities_from_graph = await self._get_related_entity_names(entity_names)
            except Exception as e:
                self.logger.warning("Failed to get graph-related entities", error=str(e))

            # Prepare prompt for fact extraction from chunks
            prompt = self._create_chunk_fact_extraction_prompt(
                user_query=user_query,
                chunks=chunks,
                entity_names=entity_names,
                traversal_depth=traversal_depth,
                language=language,
                related_entities_from_graph=related_entities_from_graph
            )

            # Call LLM for fact extraction
            result = await self.agent.run(prompt)
            response = result.data

            self.logger.debug(
                "LLM fact extraction response",
                facts_count=len(response.extracted_facts),
                next_nodes=response.next_nodes_to_explore,
                reasoning=response.reasoning[:200] + "..." if len(response.reasoning) > 200 else response.reasoning
            )

            # Enhance facts with proper metadata from chunks
            enhanced_facts = []
            for fact in response.extracted_facts:
                fact.extracted_from_depth = traversal_depth

                # Find the chunk this fact came from and add complete metadata
                if fact.source_qdrant_chunk_id:
                    for chunk in chunks:
                        if chunk["chunk_id"] == fact.source_qdrant_chunk_id:
                            from morag_reasoning.recursive_fact_models import SourceMetadata
                            fact.source_metadata = SourceMetadata(
                                document_name=chunk["document_name"],
                                chunk_index=chunk["chunk_index"],
                                page_number=chunk["chunk_metadata"].get("page_number"),
                                section=chunk["chunk_metadata"].get("section"),
                                timestamp=chunk["chunk_metadata"].get("timestamp"),
                                additional_metadata={
                                    "source_file": chunk["source_file"],
                                    "document_id": chunk["document_id"],
                                    "related_entities": chunk["related_entity_names"]
                                }
                            )
                            break

                enhanced_facts.append(fact)

            self.logger.info(
                "Fact extraction completed",
                entity_names=entity_names,
                chunks_processed=len(chunks),
                facts_extracted=len(enhanced_facts)
            )

            return GTAResponse(
                extracted_facts=enhanced_facts,
                next_nodes_to_explore=response.next_nodes_to_explore,
                reasoning=response.reasoning
            )

        except Exception as e:
            self.logger.error(
                "Error in fact extraction from chunks",
                entity_names=entity_names,
                depth=traversal_depth,
                error=str(e)
            )

            # Return empty response on error
            return GTAResponse(
                extracted_facts=[],
                next_nodes_to_explore="NONE",
                reasoning=f"Error occurred during fact extraction: {str(e)}"
            )
    
    def _create_chunk_fact_extraction_prompt(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]],
        entity_names: List[str],
        traversal_depth: int,
        language: Optional[str] = None,
        related_entities_from_graph: Optional[List[str]] = None
    ) -> str:
        """Create the prompt for chunk-based fact extraction."""

        # Format chunk information - show more chunks but with shorter content
        chunks_info = f"Document Chunks Related to Entities: {', '.join(entity_names)} (Total: {len(chunks)} chunks)\n\n"

        # Show up to 20 chunks with shorter content to fit more information
        for i, chunk in enumerate(chunks[:20]):
            chunks_info += f"Chunk {i+1} (ID: {chunk['chunk_id']}):\n"
            chunks_info += f"  Document: {chunk['document_name']}\n"
            chunks_info += f"  Chunk Index: {chunk['chunk_index']}\n"
            chunks_info += f"  Related Entities: {', '.join(chunk['related_entity_names'])}\n"
            chunks_info += f"  Content: {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}\n\n"

        # Get related entity names for next traversal from chunks and graph
        all_related_entities = set()

        # Get entities mentioned in chunks
        for chunk in chunks:
            all_related_entities.update(chunk['related_entity_names'])

        # Add entities from graph relationships
        if related_entities_from_graph:
            all_related_entities.update(related_entities_from_graph)

        # Remove current entities
        for entity_name in entity_names:
            all_related_entities.discard(entity_name)

        related_entities_list = list(all_related_entities)[:20]
        related_entities_info = f"Related Entities Found: {', '.join(related_entities_list)}\n"

        prompt = f"""DOCUMENT CHUNK FACT EXTRACTION TASK

User Query: "{user_query}"

{chunks_info}

{related_entities_info}

Traversal Information:
- Current Depth: {traversal_depth}
- Current Entities: {', '.join(entity_names)}

Your task:
1. Extract MULTIPLE relevant facts from the document chunks that help answer the user query
2. For each fact, specify which chunk it came from (use the chunk_id as source_qdrant_chunk_id)
3. Decide which related entities to explore next for deeper traversal

CRITICAL REQUIREMENTS:
- Extract AT LEAST 5-10 facts from the provided chunks
- Extract facts from the actual chunk content, not just entity relationships
- Each fact should be substantial and directly relevant to the user query
- Include specific details, numbers, dates, recommendations, and context from the chunks
- Look for concrete information like specific items, substances, recommendations, studies, etc.
- Extract specific recommendations, advice, and scientific findings relevant to the query
- For next_nodes_to_explore, return entity names (not IDs) separated by commas, or "STOP_TRAVERSAL" if sufficient information is gathered
- Focus on factual content that provides concrete answers or supporting evidence

EXTRACTION STRATEGY:
- Scan through ALL provided chunks for relevant information
- Extract multiple facts per chunk if they contain rich information
- Look for actionable advice, specific recommendations, research findings
- Include both positive recommendations (what to do) and negative ones (what to avoid)
- Extract specific items, substances, patterns, timing, and quantities when mentioned
- Look for scientific studies, expert opinions, practical implementation advice
- Extract information about mechanisms and causal relationships

FACT EXTRACTION EXAMPLES:
- Extract specific substances, compounds, or items mentioned with their effects and dosages
- Extract recommendations about what to avoid and what to prefer with supporting evidence
- Extract timing, frequency, and quantity information when provided
- Extract scientific findings, study results, and expert recommendations

NEXT ENTITY EXPLORATION:
Consider exploring related entities that appear in the content or are mentioned in relationships"""

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
