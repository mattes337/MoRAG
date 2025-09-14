"""Entity identification service for intelligent retrieval."""

import structlog
from typing import List, Optional, Dict, Any
from pydantic_ai import Agent
from pydantic import BaseModel, Field

from morag_reasoning.llm import LLMClient
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_services.embedding import GeminiEmbeddingService


class IdentifiedEntity(BaseModel):
    """An entity identified from a user query."""
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    context: str = Field(..., description="Context in which entity appears")
    graph_entity_id: Optional[str] = Field(None, description="Linked graph entity ID if found")


class EntityIdentificationResult(BaseModel):
    """Result of entity identification from query."""
    entities: List[IdentifiedEntity] = Field(..., description="Identified entities")


class EntityIdentificationService:
    """Service for identifying entities from user queries."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        graph_storage: Optional[Neo4jStorage] = None,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        min_confidence: float = 0.3,
        max_entities: int = 50,
        language: Optional[str] = None
    ):
        """Initialize the entity identification service.

        Args:
            llm_client: LLM client for entity extraction
            graph_storage: Neo4j storage for entity linking
            embedding_service: Embedding service for vector similarity search
            min_confidence: Minimum confidence threshold
            max_entities: Maximum entities to extract
            language: Language code for processing (e.g., 'en', 'de', 'fr')
        """
        self.llm_client = llm_client
        self.graph_storage = graph_storage
        self.embedding_service = embedding_service
        self.min_confidence = min_confidence
        self.max_entities = max_entities
        self.language = language
        self.logger = structlog.get_logger(__name__)

        # Create PydanticAI agent for entity identification
        self.agent = Agent(
            model=llm_client.get_model(),
            result_type=EntityIdentificationResult,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for entity identification."""
        return """You are an expert entity identification system. Your task is to identify key entities from user queries that would be useful for graph-based information retrieval.

Guidelines:
1. Focus on entities that are likely to exist in a knowledge graph (people, organizations, concepts, technologies, locations, etc.)
2. Prioritize entities that are central to answering the user's question
3. Use the same language as the user query for entity names to ensure accurate matching
4. Assign appropriate entity types using English (PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, LOCATION, EVENT, etc.)
5. Provide confidence scores based on how important each entity is for the query
6. Include context about how the entity relates to the query
7. Limit to the most relevant entities (typically 3-8 entities)

Entity Types to Consider:
- PERSON: Individual people, authors, researchers, leaders
- ORGANIZATION: Companies, institutions, governments, groups
- CONCEPT: Abstract ideas, theories, methodologies, principles
- TECHNOLOGY: Software, hardware, tools, platforms, programming languages
- LOCATION: Countries, cities, regions, specific places
- EVENT: Historical events, conferences, incidents, periods
- PRODUCT: Specific products, services, applications
- FIELD: Academic or professional domains, industries
- DOCUMENT: Specific papers, books, standards, specifications

Return only the most relevant entities that would help retrieve information to answer the user's query."""
    
    async def identify_entities(self, query: str) -> List[IdentifiedEntity]:
        """Identify entities from a user query.

        Args:
            query: User query text

        Returns:
            List of identified entities
        """
        if not query or not query.strip():
            return []

        self.logger.info("Starting entity identification", query=query, language=self.language)

        try:
            # Build language-specific instruction
            language_instruction = ""
            if self.language:
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
                language_name = language_names.get(self.language, self.language)
                language_instruction = f"\n\nIMPORTANT: Extract entity names in {language_name} ({self.language}) to match how they appear in the knowledge graph. Entity names must be in {language_name} for accurate graph matching."

            # Use PydanticAI agent to identify entities
            prompt = f"""Identify the key entities from this user query that would be useful for graph-based information retrieval:

Query: "{query}"

Focus on entities that are likely to exist in a knowledge graph and are essential for answering the user's question.{language_instruction}"""

            result = await self.agent.run(prompt)
            entities = result.data.entities
            
            # Filter by confidence and limit count
            filtered_entities = [
                entity for entity in entities 
                if entity.confidence >= self.min_confidence
            ]
            
            # Sort by confidence and limit
            filtered_entities.sort(key=lambda x: x.confidence, reverse=True)
            filtered_entities = filtered_entities[:self.max_entities]
            
            # Link to graph entities if graph storage is available
            if self.graph_storage:
                await self._link_to_graph_entities(filtered_entities)
            
            self.logger.info(
                "Entity identification completed",
                total_entities=len(filtered_entities),
                query=query
            )
            
            return filtered_entities
            
        except Exception as e:
            self.logger.error(
                "Entity identification failed",
                error=str(e),
                error_type=type(e).__name__,
                query=query
            )
            raise
    
    async def _link_to_graph_entities(self, entities: List[IdentifiedEntity]) -> None:
        """Link identified entities to existing graph entities using both exact search and vector similarity.

        Args:
            entities: List of identified entities to link
        """
        if not self.graph_storage:
            return

        for entity in entities:
            try:
                # First try exact/fuzzy search for entities
                candidates = await self.graph_storage.search_entities(
                    entity.name,
                    entity_type=None,  # Don't filter by type to allow cross-type matching
                    limit=10  # Get more candidates for better matching
                )

                # If embedding service is available and we have few candidates, try vector search
                if self.embedding_service and len(candidates) < 3:
                    vector_candidates = await self._find_similar_entities_by_vector(entity)
                    # Merge vector candidates with exact search candidates
                    candidates.extend(vector_candidates)
                    # Remove duplicates by entity ID
                    seen_ids = set()
                    unique_candidates = []
                    for candidate in candidates:
                        if candidate.id not in seen_ids:
                            unique_candidates.append(candidate)
                            seen_ids.add(candidate.id)
                    candidates = unique_candidates[:10]  # Limit to top 10

                if candidates:
                    # Debug: Log candidates found
                    self.logger.debug(
                        "Entity linking candidates found",
                        entity_name=entity.name,
                        candidates=[f"{c.name} ({c.type})" for c in candidates]
                    )

                    # Find best match based on name similarity
                    best_match = None
                    best_score = 0.0
                    
                    for candidate in candidates:
                        # Simple similarity scoring
                        similarity = self._calculate_name_similarity(
                            entity.name.lower(),
                            candidate.name.lower()
                        )

                        # Debug logging for entity linking
                        self.logger.debug(
                            "Entity similarity check",
                            entity_name=entity.name,
                            candidate_name=candidate.name,
                            similarity=similarity,
                            threshold=0.7
                        )

                        if similarity > best_score and similarity >= 0.7:  # Lower threshold for cross-language matching
                            best_score = similarity
                            best_match = candidate
                    
                    if best_match:
                        entity.graph_entity_id = best_match.id
                        self.logger.debug(
                            "Linked entity to graph",
                            entity_name=entity.name,
                            graph_entity_id=best_match.id,
                            similarity=best_score
                        )
                
            except Exception as e:
                self.logger.warning(
                    "Failed to link entity to graph",
                    entity_name=entity.name,
                    error=str(e)
                )
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names.

        Args:
            name1: First entity name
            name2: Second entity name

        Returns:
            Similarity score between 0 and 1
        """
        # Simple similarity calculation
        if name1 == name2:
            return 1.0

        # Check if one is contained in the other
        if name1 in name2 or name2 in name1:
            return 0.9

        # Check word overlap
        words1 = set(name1.split())
        words2 = set(name2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    async def _find_similar_entities_by_vector(self, entity: IdentifiedEntity) -> List[Any]:
        """Find similar entities using vector similarity search.

        Args:
            entity: Entity to find similar entities for

        Returns:
            List of similar entities from vector search
        """
        if not self.embedding_service:
            return []

        try:
            # Generate embedding for the entity name and context
            entity_text = f"{entity.name} ({entity.entity_type})"
            if entity.context:
                entity_text += f" - {entity.context}"

            embedding_result = await self.embedding_service.generate_embedding(
                entity_text, task_type="retrieval_query"
            )

            # Handle both direct list return and EmbeddingResult object
            if isinstance(embedding_result, list):
                query_embedding = embedding_result
            else:
                query_embedding = embedding_result.embedding

            # Search for similar entities in Neo4j using vector similarity
            # This requires the EntityEmbeddingService functionality
            from morag_graph.services.entity_embedding_service import EntityEmbeddingService

            entity_embedding_service = EntityEmbeddingService(
                self.graph_storage, self.embedding_service
            )

            similar_entities = await entity_embedding_service.search_similar_entities(
                query_embedding, limit=5, similarity_threshold=0.4
            )

            # Convert to the expected format (simplified for now)
            candidates = []
            for similar in similar_entities:
                # Create a simple candidate object with both 'type' and 'entity_type' attributes
                candidate = type('Candidate', (), {
                    'id': similar['id'],
                    'name': similar['name'],
                    'type': similar.get('type', 'Unknown'),  # Add 'type' attribute
                    'entity_type': similar.get('type', 'Unknown'),  # Keep 'entity_type' for compatibility
                    'similarity_score': similar['similarity']
                })()
                candidates.append(candidate)

            self.logger.debug(
                "Vector similarity search found candidates",
                entity_name=entity.name,
                candidates_count=len(candidates)
            )

            return candidates

        except Exception as e:
            self.logger.warning(
                "Vector similarity search failed",
                entity_name=entity.name,
                error=str(e)
            )
            return []
