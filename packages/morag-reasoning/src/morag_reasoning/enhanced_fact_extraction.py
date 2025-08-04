"""Enhanced fact extraction service with vector search capabilities."""

import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime

from morag_reasoning.llm import LLMClient
from morag_reasoning.recursive_fact_models import RawFact
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_services.embedding import GeminiEmbeddingService


class EnhancedFactExtractionService:
    """Enhanced fact extraction service that combines graph traversal with vector search."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        neo4j_storage: Neo4jStorage,
        embedding_service: Optional[GeminiEmbeddingService] = None
    ):
        """Initialize the enhanced fact extraction service.
        
        Args:
            llm_client: LLM client for fact extraction
            neo4j_storage: Neo4j storage for graph operations
            embedding_service: Embedding service for vector similarity search
        """
        self.llm_client = llm_client
        self.neo4j_storage = neo4j_storage
        self.embedding_service = embedding_service
        self.logger = structlog.get_logger(__name__)
    
    async def extract_facts_for_query(
        self,
        user_query: str,
        entity_names: List[str],
        max_facts: int = 20,
        language: Optional[str] = None
    ) -> List[RawFact]:
        """Extract facts relevant to a query using both graph relationships and vector similarity.
        
        Args:
            user_query: User's query
            entity_names: List of entity names to extract facts for
            max_facts: Maximum number of facts to return
            language: Language for fact extraction
            
        Returns:
            List of extracted facts
        """
        try:
            # Extract facts from graph relationships (existing approach)
            graph_facts = await self._extract_facts_from_graph(
                user_query, entity_names, language
            )
            
            # Extract facts using vector similarity if embedding service is available
            vector_facts = []
            if self.embedding_service:
                vector_facts = await self._extract_facts_from_vector_search(
                    user_query, max_facts // 2, language
                )
            
            # Combine and deduplicate facts
            all_facts = self._combine_and_deduplicate_facts(graph_facts, vector_facts)
            
            # Limit to max_facts
            limited_facts = all_facts[:max_facts]
            
            self.logger.info(
                "Enhanced fact extraction completed",
                user_query=user_query,
                entity_names=entity_names,
                graph_facts=len(graph_facts),
                vector_facts=len(vector_facts),
                total_unique_facts=len(all_facts),
                returned_facts=len(limited_facts)
            )
            
            return limited_facts
            
        except Exception as e:
            self.logger.error(
                "Enhanced fact extraction failed",
                user_query=user_query,
                entity_names=entity_names,
                error=str(e)
            )
            return []
    
    async def _extract_facts_from_graph(
        self,
        user_query: str,
        entity_names: List[str],
        language: Optional[str] = None
    ) -> List[RawFact]:
        """Extract facts from graph relationships (existing approach).
        
        Args:
            user_query: User's query
            entity_names: List of entity names
            language: Language for extraction
            
        Returns:
            List of facts from graph traversal
        """
        try:
            # Get facts related to entities through graph relationships
            query = """
            MATCH (e)-[r]-(f:Fact)
            WHERE e.name IN $entity_names
            RETURN DISTINCT f.id as fact_id, f.subject as subject, f.approach as approach,
                   f.object as object, f.solution as solution, f.keywords as keywords,
                   f.domain as domain
            LIMIT 50
            """
            
            result = await self.neo4j_storage._execute_query(query, {"entity_names": entity_names})
            
            facts = []
            for record in result:
                fact = RawFact(
                    id=record["fact_id"],
                    text=self._create_fact_text(record),
                    source_type="graph_relationship",
                    source_id=record["fact_id"],
                    metadata={
                        "subject": record["subject"],
                        "approach": record["approach"],
                        "object": record["object"],
                        "solution": record["solution"],
                        "keywords": record["keywords"],
                        "domain": record["domain"],
                        "extraction_method": "graph_traversal"
                    },
                    extracted_at=datetime.utcnow()
                )
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            self.logger.error(
                "Graph fact extraction failed",
                entity_names=entity_names,
                error=str(e)
            )
            return []
    
    async def _extract_facts_from_vector_search(
        self,
        user_query: str,
        max_facts: int,
        language: Optional[str] = None
    ) -> List[RawFact]:
        """Extract facts using vector similarity search.
        
        Args:
            user_query: User's query
            max_facts: Maximum number of facts to return
            language: Language for extraction
            
        Returns:
            List of facts from vector search
        """
        if not self.embedding_service:
            return []
        
        try:
            from morag_graph.services.fact_embedding_service import FactEmbeddingService
            
            fact_embedding_service = FactEmbeddingService(
                self.neo4j_storage, self.embedding_service
            )
            
            # Generate embedding for the user query
            query_embedding = await self.embedding_service.generate_embedding(
                user_query, task_type="retrieval_query"
            )
            
            # Search for similar facts
            similar_facts = await fact_embedding_service.search_similar_facts(
                query_embedding, limit=max_facts, similarity_threshold=0.3
            )
            
            # Convert to RawFact objects
            facts = []
            for fact_data in similar_facts:
                fact_text = self._create_fact_text(fact_data)
                
                fact = RawFact(
                    id=fact_data["id"],
                    text=fact_text,
                    source_type="vector_similarity",
                    source_id=fact_data["id"],
                    metadata={
                        "subject": fact_data.get("subject"),
                        "approach": fact_data.get("approach"),
                        "object": fact_data.get("object"),
                        "solution": fact_data.get("solution"),
                        "similarity_score": fact_data["similarity"],
                        "extraction_method": "vector_search"
                    },
                    extracted_at=datetime.utcnow()
                )
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            self.logger.error(
                "Vector fact extraction failed",
                user_query=user_query,
                error=str(e)
            )
            return []
    
    def _create_fact_text(self, fact_data: Dict[str, Any]) -> str:
        """Create readable fact text from fact data.
        
        Args:
            fact_data: Dictionary containing fact information
            
        Returns:
            Formatted fact text
        """
        subject = fact_data.get("subject", "")
        approach = fact_data.get("approach", "")
        object_text = fact_data.get("object", "")
        solution = fact_data.get("solution", "")
        
        # Create structured fact text
        parts = []
        if subject:
            parts.append(f"Subject: {subject}")
        if approach:
            parts.append(f"Approach: {approach}")
        if object_text:
            parts.append(f"Object: {object_text}")
        if solution:
            parts.append(f"Solution: {solution}")
        
        return ". ".join(parts) if parts else "No fact text available"
    
    def _combine_and_deduplicate_facts(
        self,
        graph_facts: List[RawFact],
        vector_facts: List[RawFact]
    ) -> List[RawFact]:
        """Combine facts from different sources and remove duplicates.
        
        Args:
            graph_facts: Facts from graph traversal
            vector_facts: Facts from vector search
            
        Returns:
            Combined and deduplicated list of facts
        """
        # Use fact ID for deduplication
        seen_ids = set()
        combined_facts = []
        
        # Add graph facts first (they have higher priority)
        for fact in graph_facts:
            if fact.id not in seen_ids:
                combined_facts.append(fact)
                seen_ids.add(fact.id)
        
        # Add vector facts if not already present
        for fact in vector_facts:
            if fact.id not in seen_ids:
                combined_facts.append(fact)
                seen_ids.add(fact.id)
        
        return combined_facts
