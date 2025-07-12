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
        return f"""You are a helpful GraphTraversalAgent that extracts useful information from document chunks.

Your job is to:
1. Look through document chunks for information related to the user's query
2. Extract any useful facts, recommendations, or advice you find
3. Suggest related topics to explore next

EXTRACTION APPROACH:
- Be VERY AGGRESSIVE in extracting information - extract MANY facts
- Look for recommendations, treatments, foods, supplements, advice
- Include both what helps and what to avoid
- Extract specific details when available (dosages, amounts, etc.)
- Extract from EVERY chunk that contains relevant information
- Extract MULTIPLE facts per chunk when possible
- Extract up to {self.max_facts_per_node} useful pieces of information (aim for the maximum!)

WHAT TO LOOK FOR:
- Treatments, therapies, or interventions
- Foods, supplements, or substances mentioned
- Things to do or avoid
- Effects, benefits, or problems
- Research findings or expert advice
- Practical tips or strategies

RESPONSE FORMAT:
- extracted_facts: List of useful information found in the chunks
- next_nodes_to_explore: Related entity names to explore next, or "STOP_TRAVERSAL"
- reasoning: Brief explanation of what you found

Be generous in extracting information - anything related to the user's query is valuable."""

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
            # Don't filter by static types - let LLM decide which paths to follow
            # Find all entities connected to our target entities through any relationship
            query = """
            MATCH (e1)-[r]-(e2)
            WHERE e1.name IN $entity_names
            AND NOT e2.name IN $entity_names
            AND e2.name IS NOT NULL
            RETURN DISTINCT e2.name as related_entity_name, type(r) as relation_type
            LIMIT 100
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

            # Debug: Log prompt length and sample content
            self.logger.debug(
                "Sending prompt to LLM for fact extraction",
                prompt_length=len(prompt),
                chunks_count=len(chunks),
                entity_names=entity_names,
                sample_prompt=prompt[:500] + "..." if len(prompt) > 500 else prompt
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
                        # Handle both string and dict chunks defensively
                        if isinstance(chunk, str):
                            continue

                        chunk_id = chunk.get("chunk_id", "")
                        if chunk_id == fact.source_qdrant_chunk_id:
                            from morag_reasoning.recursive_fact_models import SourceMetadata

                            # Safely extract metadata with defaults
                            chunk_metadata = chunk.get("chunk_metadata") or {}
                            related_entities = chunk.get("related_entity_names", [])
                            if not isinstance(related_entities, list):
                                related_entities = []

                            fact.source_metadata = SourceMetadata(
                                document_name=chunk.get("document_name", "Unknown Document"),
                                chunk_index=chunk.get("chunk_index", 0),
                                page_number=chunk_metadata.get("page_number") if isinstance(chunk_metadata, dict) else None,
                                section=chunk_metadata.get("section") if isinstance(chunk_metadata, dict) else None,
                                timestamp=chunk_metadata.get("timestamp") if isinstance(chunk_metadata, dict) else None,
                                additional_metadata={
                                    "source_file": chunk.get("source_file", ""),
                                    "document_id": chunk.get("document_id", ""),
                                    "related_entities": related_entities
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

        # Format chunk information - show ALL chunks with substantial content
        chunks_info = f"Document Chunks Related to Entities: {', '.join(entity_names)} (Total: {len(chunks)} chunks)\n\n"

        # Show ALL chunks with substantial content to maximize fact extraction
        for i, chunk in enumerate(chunks):
            # Handle both dict and string chunks defensively
            if isinstance(chunk, str):
                chunks_info += f"Chunk {i+1} (Text): {chunk[:1200]}{'...' if len(chunk) > 1200 else ''}\n\n"
                continue

            # Handle dict chunks with defensive access
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            document_name = chunk.get('document_name', 'Unknown Document')
            chunk_index = chunk.get('chunk_index', i)
            related_entities = chunk.get('related_entity_names', [])
            text = chunk.get('text', chunk.get('content', ''))

            # Ensure related_entities is a list
            if not isinstance(related_entities, list):
                related_entities = []

            chunks_info += f"Chunk {i+1} (ID: {chunk_id}):\n"
            chunks_info += f"  Document: {document_name}\n"
            chunks_info += f"  Chunk Index: {chunk_index}\n"
            chunks_info += f"  Related Entities: {', '.join(related_entities)}\n"
            chunks_info += f"  Content: {text[:1200]}{'...' if len(text) > 1200 else ''}\n\n"

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

        prompt = f"""FIND USEFUL INFORMATION FROM DOCUMENT CHUNKS

User Query: "{user_query}"

{chunks_info}

{related_entities_info}

YOUR TASK:
AGGRESSIVELY extract MANY useful facts from the document chunks that relate to the user's query.

WHAT TO EXTRACT (be very thorough):
- Any specific recommendations, treatments, or advice
- Foods, supplements, or substances mentioned
- Effects, benefits, or problems described
- Dosages, amounts, or quantities mentioned
- Things to do or avoid
- Research findings or expert opinions
- Practical tips or strategies
- Mechanisms of action
- Study results and statistics
- Personal experiences and case studies
- Contraindications and warnings

EXTRACTION STRATEGY:
- Go through EVERY chunk systematically
- Extract MULTIPLE facts from each relevant chunk
- Be VERY GENEROUS - extract anything remotely useful
- Include both detailed and general information
- Extract both positive and negative information
- Look for any connection to the user's query topic
- Don't skip chunks - scan them all for useful information

EXAMPLES OF WHAT TO EXTRACT:
- "Omega-3 helps with ADHD symptoms"
- "Avoid sugar and processed foods"
- "Exercise improves focus"
- "Sleep is important for attention"
- "Certain food additives cause hyperactivity"
- "Magnesium supplements may help"
- "Studies show 25% improvement with treatment X"
- "Doctor recommends avoiding substance Y"
- "Patient reported better focus after change Z"

REQUIREMENTS:
- Extract AS MANY facts as possible, up to {self.max_facts_per_node} pieces of information
- Aim for the MAXIMUM number of facts - don't stop early
- Make each fact clear and understandable
- Include any specific details when available
- For next exploration, suggest related entity names or "STOP_TRAVERSAL"

Remember: Extract EVERYTHING useful - the more facts the better!"""

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
