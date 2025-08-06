"""Enhanced fact extraction service with vector search capabilities."""

import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime

from morag_reasoning.llm import LLMClient
from morag_reasoning.recursive_fact_models import RawFact, SourceMetadata
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
            # Extract facts from graph relationships with query context
            graph_facts = await self._extract_facts_from_graph(
                user_query, entity_names, language, user_query
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
        language: Optional[str] = None,
        query_context: Optional[str] = None
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
            
            result = await self.neo4j_storage._connection_ops._execute_query(query, {"entity_names": entity_names})
            
            facts = []
            for record in result:
                fact_text = self._create_fact_text(record)

                # Get source document information
                source_info = await self._get_source_document_info(record["fact_id"])

                # Only include facts with proper document sources
                if not source_info.get("source_file") and not source_info.get("title"):
                    self.logger.warning(f"Skipping fact {record['fact_id']} - no document source found")
                    continue

                fact = RawFact(
                    fact_text=fact_text,
                    source_node_id=record["fact_id"],
                    extracted_from_depth=0,  # Graph facts are at depth 0
                    source_metadata=SourceMetadata(
                        document_name=source_info.get("title", source_info.get("source_file", "")),
                        chunk_index=source_info.get("chunk_index", 0),
                        page_number=source_info.get("page_number"),
                        section=source_info.get("section"),
                        timestamp=source_info.get("timestamp"),
                        additional_metadata={
                            "subject": record["subject"],
                            "approach": record["approach"],
                            "object": record["object"],
                            "solution": record["solution"],
                            "keywords": record["keywords"],
                            "domain": record["domain"],
                            "extraction_method": "graph_traversal",
                            "source_file": source_info.get("source_file", ""),
                            "title": source_info.get("title", ""),
                            "chapter": source_info.get("chapter"),
                            "fact_id": record["fact_id"]
                        }
                    )
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

                # Get source document information
                source_info = await self._get_source_document_info(fact_data["id"])

                # Only include facts with proper document sources
                if not source_info.get("source_file") and not source_info.get("title"):
                    self.logger.warning(f"Skipping fact {fact_data['id']} - no document source found")
                    continue

                fact = RawFact(
                    fact_text=fact_text,
                    source_node_id=fact_data["id"],
                    extracted_from_depth=0,  # Vector facts are at depth 0
                    source_metadata=SourceMetadata(
                        document_name=source_info.get("title", source_info.get("source_file", "")),
                        chunk_index=source_info.get("chunk_index", 0),
                        page_number=source_info.get("page_number"),
                        section=source_info.get("section"),
                        timestamp=source_info.get("timestamp"),
                        additional_metadata={
                            "subject": fact_data.get("subject"),
                            "approach": fact_data.get("approach"),
                            "object": fact_data.get("object"),
                            "solution": fact_data.get("solution"),
                            "similarity_score": fact_data["similarity"],
                            "extraction_method": "vector_search",
                            "source_file": source_info.get("source_file", ""),
                            "title": source_info.get("title", ""),
                            "chapter": source_info.get("chapter"),
                            "fact_id": fact_data["id"]
                        }
                    )
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
        # Use source_node_id for deduplication since RawFact doesn't have id field
        seen_ids = set()
        combined_facts = []

        # Add graph facts first (they have higher priority)
        for fact in graph_facts:
            if fact.source_node_id not in seen_ids:
                combined_facts.append(fact)
                seen_ids.add(fact.source_node_id)

        # Add vector facts if not already present
        for fact in vector_facts:
            if fact.source_node_id not in seen_ids:
                combined_facts.append(fact)
                seen_ids.add(fact.source_node_id)

        return combined_facts

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

    async def _get_source_document_info(self, fact_id: str) -> Dict[str, Any]:
        """Get source document information for a fact.

        Args:
            fact_id: Fact ID to get source info for

        Returns:
            Dictionary with source document metadata
        """
        try:
            # Try multiple approaches to find the document chunk for this fact

            # Approach 1: Using source_chunk_id
            query1 = """
            MATCH (f:Fact {id: $fact_id})
            OPTIONAL MATCH (c:DocumentChunk {id: f.source_chunk_id})
            OPTIONAL MATCH (d:Document {id: c.document_id})
            RETURN d.source_file as source_file,
                   d.name as title,
                   c.chunk_index as chunk_index,
                   c.metadata as chunk_metadata,
                   c.text as chunk_text,
                   d.metadata as document_metadata
            LIMIT 1
            """

            result = await self.neo4j_storage._connection_ops._execute_query(query1, {"fact_id": fact_id})

            # If no result, try approach 2: Find any related chunk
            if not result or not result[0].get("source_file"):
                query2 = """
                MATCH (f:Fact {id: $fact_id})-[r]-(c:DocumentChunk)
                OPTIONAL MATCH (d:Document {id: c.document_id})
                RETURN d.source_file as source_file,
                       d.name as title,
                       c.chunk_index as chunk_index,
                       c.metadata as chunk_metadata,
                       c.text as chunk_text,
                       d.metadata as document_metadata
                LIMIT 1
                """
                result = await self.neo4j_storage._connection_ops._execute_query(query2, {"fact_id": fact_id})

            if result and result[0].get("source_file"):
                record = result[0]
                chunk_metadata_str = record.get("chunk_metadata", "{}")
                doc_metadata_str = record.get("document_metadata", "{}")
                chunk_text = record.get("chunk_text", "")

                # Parse JSON metadata
                import json
                try:
                    chunk_metadata = json.loads(chunk_metadata_str) if chunk_metadata_str else {}
                except:
                    chunk_metadata = {}

                try:
                    doc_metadata = json.loads(doc_metadata_str) if doc_metadata_str else {}
                except:
                    doc_metadata = {}

                # Extract timestamp from text content for video/audio files
                timestamp = chunk_metadata.get("timestamp")
                if not timestamp and chunk_text:
                    timestamp = self._extract_timestamp_from_text(chunk_text)

                # Extract section information
                section = chunk_metadata.get("section_title")
                if not section and chunk_text:
                    section = self._extract_section_from_text(chunk_text, chunk_metadata)

                return {
                    "source_file": record.get("source_file", ""),
                    "title": record.get("title", ""),
                    "chunk_index": record.get("chunk_index", 0),
                    "page_number": chunk_metadata.get("page_number"),
                    "section": section,
                    "timestamp": timestamp,
                    "chapter": chunk_metadata.get("chapter"),
                    "metadata": {**chunk_metadata, **doc_metadata}
                }
            else:
                self.logger.debug(f"No document source found for fact {fact_id}")
                return {}

        except Exception as e:
            self.logger.warning(f"Failed to get source document info for fact {fact_id}: {e}")
            return {}
