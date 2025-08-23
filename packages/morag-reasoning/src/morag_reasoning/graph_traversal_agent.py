"""GraphTraversalAgent for intelligent graph traversal and fact extraction."""

import json
import os
import structlog
from typing import List, Set, Dict, Any, Optional
from pydantic_ai import Agent
from pydantic import BaseModel, Field

from morag_reasoning.llm import LLMClient
from morag_reasoning.recursive_fact_models import RawFact, GTAResponse
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_graph.storage.qdrant_storage import QdrantStorage
from morag_services.embedding import GeminiEmbeddingService


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
        embedding_service: Optional[GeminiEmbeddingService] = None,
        max_facts_per_node: int = 1000
    ):
        """Initialize the GraphTraversalAgent.

        Args:
            llm_client: LLM client for AI operations
            neo4j_storage: Neo4j storage for graph operations
            qdrant_storage: Qdrant storage for vector operations
            embedding_service: Embedding service for vector similarity search
            max_facts_per_node: Maximum facts to extract per node
        """
        self.llm_client = llm_client
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        # Retrieval tuning for co-occurrence-based neighbor expansion
        self.enable_cooccurrence_neighbors: bool = os.getenv("MORAG_RETRIEVAL_ENABLE_COOCC", "true").lower() == "true"
        self.cooccurrence_min_share: float = float(os.getenv("MORAG_RETRIEVAL_COOCC_SHARE", "0.12"))
        self.cooccurrence_max_neighbors: int = int(os.getenv("MORAG_RETRIEVAL_COOCC_LIMIT", "100"))

        self.embedding_service = embedding_service
        self.max_facts_per_node = max_facts_per_node
        self.logger = structlog.get_logger(__name__)

        # Create PydanticAI agent for graph traversal decisions
        self.agent = Agent(
            model=llm_client.get_model(),
            output_type=GTAResponse,
            system_prompt=self._get_system_prompt()
        )

    def _extract_timestamp_from_text(self, text: str) -> Optional[str]:
        """Extract timestamp from video/audio text content."""
        import re

        # Look for timestamp patterns like [28:14], [28:15 - 28:16], [31:09 - 31:13]
        timestamp_patterns = [
            r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]',  # [28:15 - 28:16]
            r'\[(\d{1,2}:\d{2})\]',  # [28:14]
            r'\[(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]',  # [01:28:15 - 01:28:16]
            r'\[(\d{1,2}:\d{2}:\d{2})\]',  # [01:28:14]
        ]

        for pattern in timestamp_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    # Range pattern - return the start time
                    return matches[0][0]
                else:
                    # Single timestamp
                    return matches[0]

        return None

    def _extract_section_from_text(self, text: str, chunk_metadata: Dict[str, Any]) -> Optional[str]:
        """Extract section information from text or metadata."""
        # Check metadata first
        if chunk_metadata.get("section_title"):
            return chunk_metadata["section_title"]

        # For video content, try to extract topic from the beginning of the text
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove timestamp if present
            import re
            first_line = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\s*(?:-\s*\d{1,2}:\d{2}(?::\d{2})?)?\]', '', first_line).strip()
            if first_line and len(first_line) < 100:  # Reasonable section title length
                return first_line

        return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the GraphTraversalAgent."""
        return f"""You are a helpful GraphTraversalAgent that extracts useful information from document chunks.

Your job is to:
1. Look through document chunks for information related to the user's query
2. Extract any useful facts, recommendations, or advice you find
3. Suggest related topics to explore next

EXTRACTION APPROACH:
- Be EXHAUSTIVE in extracting information - extract ALL relevant facts
- Look for recommendations, treatments, foods, supplements, advice
- Include both what helps and what to avoid
- Extract specific details when available (dosages, amounts, etc.)
- Extract from EVERY chunk that contains relevant information
- Extract MULTIPLE facts per chunk when possible
- Extract ALL useful pieces of information above the relevance threshold - do not limit yourself

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
        """Get DocumentChunk nodes for specific entity names using multiple methods.

        Args:
            entity_names: List of entity names to get chunks for

        Returns:
            List of chunk data with full metadata
        """
        try:
            # Method 1: Direct Neo4j lookup
            chunks = await self.neo4j_storage.get_document_chunks_by_entity_names(entity_names)

            # Method 2: Vector search if we have few or no chunks
            if len(chunks) < 5 and hasattr(self, 'qdrant_storage') and self.qdrant_storage:
                vector_chunks = await self._get_chunks_via_vector_search(entity_names)
                chunks.extend(vector_chunks)

                # Remove duplicates
                seen_chunk_ids = set()
                unique_chunks = []
                for chunk in chunks:
                    if chunk["chunk_id"] not in seen_chunk_ids:
                        unique_chunks.append(chunk)
                        seen_chunk_ids.add(chunk["chunk_id"])
                chunks = unique_chunks

            self.logger.debug(f"Found {len(chunks)} chunks for entities: {entity_names}")
            return chunks
        except Exception as e:
            self.logger.error("Failed to get document chunks for entities",
                            entity_names=entity_names, error=str(e))
            return []

    async def _get_chunks_via_vector_search(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """Get chunks using vector similarity search.

        Args:
            entity_names: List of entity names to search for

        Returns:
            List of chunk data from vector search
        """
        try:
            # Create search query from entity names
            search_query = " ".join(entity_names)

            # Generate embedding for the search query
            if hasattr(self, 'embedding_service') and self.embedding_service:
                embedding_result = await self.embedding_service.generate_embedding(
                    search_query, task_type="retrieval_query"
                )

                # Handle both direct list return and EmbeddingResult object
                if isinstance(embedding_result, list):
                    query_embedding = embedding_result
                else:
                    query_embedding = embedding_result.embedding
            else:
                self.logger.warning("No embedding service available for vector search")
                return []

            # Search for similar chunks
            search_results = await self.qdrant_storage.search_similar(
                query_vector=query_embedding,
                limit=10,
                score_threshold=0.6
            )

            # Convert Qdrant results to chunk format
            chunks = []
            for result in search_results:
                metadata = result.get("metadata", {})
                chunk_data = {
                    "chunk_id": result["id"],
                    "document_id": metadata.get("document_id", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "content": metadata.get("text", ""),
                    "metadata": metadata,
                    "vector_score": result["score"]
                }
                chunks.append(chunk_data)

            self.logger.debug(f"Vector search found {len(chunks)} chunks for entities: {entity_names}")
            return chunks

        except Exception as e:
            self.logger.warning(f"Vector search failed: {e}")
            return []

    async def _get_related_entity_names(self, entity_names: List[str]) -> List[str]:
        """Get names of entities related to the given entities using both graph relationships and vector similarity.

        Args:
            entity_names: List of entity names to find related entities for

        Returns:
            List of related entity names
        """
        try:
            # First, find entities connected through graph relationships
            query = """
            MATCH (e1)-[r]-(e2)
            WHERE e1.name IN $entity_names
            AND NOT e2.name IN $entity_names
            AND e2.name IS NOT NULL
            RETURN DISTINCT e2.name as related_entity_name, type(r) as relation_type
            LIMIT 100
            """

            result = await self.neo4j_storage._connection_ops._execute_query(query, {"entity_names": entity_names})

            # Optionally include co-occurrence neighbors via Facts
            coocc_related_names: List[str] = []
            if self.enable_cooccurrence_neighbors:
                try:
                    coocc_related_names = await self._find_cooccurrence_related_entities(entity_names)
                except Exception as e:
                    self.logger.warning("Failed to get co-occurrence related entities", error=str(e))

            graph_related_names = [record["related_entity_name"] for record in result]

            # If we have embedding service, also find semantically similar entities
            vector_related_names = []
            if self.embedding_service:
                vector_related_names = await self._find_vector_related_entities(entity_names)

            # Combine and deduplicate results
            all_related_names = list(set(graph_related_names + vector_related_names + coocc_related_names))

            self.logger.debug(
                "Found related entities",
                entity_names=entity_names,
                graph_related=len(graph_related_names),
                vector_related=len(vector_related_names),
                total_unique=len(all_related_names)
            )

            return all_related_names

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
            response = await self.agent.run(prompt)

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
                            chunk_text = chunk.get("text", chunk.get("content", ""))
                            related_entities = chunk.get("related_entity_names", [])
                            if not isinstance(related_entities, list):
                                related_entities = []

                            # Extract timestamp from text content for video/audio files
                            timestamp = chunk_metadata.get("timestamp") if isinstance(chunk_metadata, dict) else None
                            if not timestamp and chunk_text:
                                timestamp = self._extract_timestamp_from_text(chunk_text)

                            # Extract section information
                            section = chunk_metadata.get("section_title") or chunk_metadata.get("section") if isinstance(chunk_metadata, dict) else None
                            if not section and chunk_text:
                                section = self._extract_section_from_text(chunk_text, chunk_metadata)

                            fact.source_metadata = SourceMetadata(
                                document_name=chunk.get("document_name", "Unknown Document"),
                                chunk_index=chunk.get("chunk_index", 0),
                                page_number=chunk_metadata.get("page_number") if isinstance(chunk_metadata, dict) else None,
                                section=section,
                                timestamp=timestamp,
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
            # Safely get related entity names with fallback
            related_entities = chunk.get('related_entity_names', [])
            if isinstance(related_entities, list):
                all_related_entities.update(related_entities)
            else:
                self.logger.debug(
                    "Invalid related_entity_names format in chunk",
                    chunk_id=chunk.get('chunk_id', 'unknown'),
                    type=type(related_entities).__name__
                )

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
- Extract ALL relevant facts - be exhaustive and comprehensive
- Do not limit the number of facts - extract everything above the relevance threshold
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

    async def _find_vector_related_entities(self, entity_names: List[str]) -> List[str]:
        """Find entities related through vector similarity.

        Args:
            entity_names: List of entity names to find similar entities for

        Returns:
            List of related entity names found through vector similarity
        """
        if not self.embedding_service:
            return []

        try:
            from morag_graph.services.entity_embedding_service import EntityEmbeddingService

            entity_embedding_service = EntityEmbeddingService(
                self.neo4j_storage, self.embedding_service
            )

            all_similar_names = []

            # For each entity, find similar entities
            for entity_name in entity_names:
                # Generate embedding for the entity
                embedding_result = await self.embedding_service.generate_embedding(
                    entity_name, task_type="retrieval_query"
                )

                # Handle both direct list return and EmbeddingResult object
                if isinstance(embedding_result, list):
                    query_embedding = embedding_result
                else:
                    query_embedding = embedding_result.embedding

                # Find similar entities
                similar_entities = await entity_embedding_service.search_similar_entities(
                    query_embedding, limit=10, similarity_threshold=0.3
                )

                # Extract names, excluding the original entity
                for similar in similar_entities:
                    if similar['name'] not in entity_names:
                        all_similar_names.append(similar['name'])


            # Remove duplicates and return
            unique_similar_names = list(set(all_similar_names))

            self.logger.debug(
                "Vector similarity found related entities",
                original_entities=entity_names,
                similar_entities_count=len(unique_similar_names)
            )

            return unique_similar_names

        except Exception as e:
            self.logger.warning(
                "Vector similarity search for related entities failed",
                entity_names=entity_names,
                error=str(e)
            )
            return []


    async def _find_cooccurrence_related_entities(self, entity_names: List[str]) -> List[str]:
        """Find entities that co-occur with the given entities via shared facts.

        We look for (e1:Entity)<-[]-(f:Fact)-[]->(e2:Entity), aggregate by e2, and rank by co-occurrence.
        Applies a min share threshold to reduce noise and limits the result size.
        """
        if not entity_names:
            return []
        try:
            # Compute total facts per seed entity for share calculation
            totals_query = """
            MATCH (e:Entity)
            WHERE e.name IN $entity_names
            MATCH (f:Fact)-[]->(e)
            RETURN e.name AS name, count(DISTINCT f) AS total
            """
            totals = await self.neo4j_storage._connection_ops._execute_query(totals_query, {"entity_names": entity_names})
            totals_map = {row["name"]: max(1, row["total"]) for row in totals}

            # Co-occurrence across the frontier; compute share per source entity then aggregate by target entity
            coocc_query = """
            MATCH (e1:Entity)
            WHERE e1.name IN $entity_names
            MATCH (f:Fact)-[]->(e1)
            MATCH (f)-[]->(e2:Entity)
            WHERE e2.name IS NOT NULL AND NOT e2.name IN $entity_names
            WITH e1.name AS src_name, e2.name AS tgt_name, count(DISTINCT f) AS cofacts
            RETURN src_name, tgt_name, cofacts
            """
            rows = await self.neo4j_storage._connection_ops._execute_query(coocc_query, {"entity_names": entity_names})

            # Aggregate and filter by share
            scores: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for row in rows:
                src = row.get("src_name")
                tgt = row.get("tgt_name")
                co = int(row.get("cofacts", 0))
                total = totals_map.get(src, 1)
                share = float(co) / float(total)
                if share >= self.cooccurrence_min_share:
                    scores[tgt] = max(scores.get(tgt, 0.0), share)
                    counts[tgt] = counts.get(tgt, 0) + co

            # Rank targets by counts then share
            ranked = sorted(scores.keys(), key=lambda t: (counts.get(t, 0), scores.get(t, 0.0)), reverse=True)
            if self.cooccurrence_max_neighbors > 0:
                ranked = ranked[: self.cooccurrence_max_neighbors]

            self.logger.debug(
                "Co-occurrence neighbors computed",
                seeds=entity_names,
                min_share=self.cooccurrence_min_share,
                returned=len(ranked)
            )
            return ranked
        except Exception as e:
            self.logger.warning("Co-occurrence neighbor computation failed", error=str(e))
            return []

