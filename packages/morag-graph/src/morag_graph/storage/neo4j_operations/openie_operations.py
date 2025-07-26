"""Neo4j operations for OpenIE triplet storage and management."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import structlog
from uuid import uuid4

from neo4j import AsyncDriver
from .base_operations import BaseOperations
from ...processors.triplet_processor import ValidatedTriplet
from ...normalizers.entity_linker import EntityMatch
from ...normalizers.predicate_normalizer import NormalizedPredicate

logger = structlog.get_logger(__name__)


class OpenIEOperations(BaseOperations):
    """Neo4j operations for OpenIE triplet storage."""
    
    def __init__(self, driver: AsyncDriver, database: str = "neo4j"):
        """Initialize OpenIE operations.
        
        Args:
            driver: Neo4j async driver
            database: Database name
        """
        super().__init__(driver, database)
        self.batch_size = 100
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def initialize_schema(self) -> None:
        """Initialize OpenIE schema in Neo4j."""
        try:
            logger.info("Initializing OpenIE schema")
            
            # Create constraints and indexes
            schema_queries = [
                # OpenIE triplet constraints
                "CREATE CONSTRAINT openie_triplet_id IF NOT EXISTS FOR (t:OpenIETriplet) REQUIRE t.triplet_id IS UNIQUE",
                
                # OpenIE entity constraints (extends existing Entity)
                "CREATE INDEX openie_entity_canonical IF NOT EXISTS FOR (e:Entity) ON (e.canonical_form)",
                "CREATE INDEX openie_entity_extraction IF NOT EXISTS FOR (e:Entity) ON (e.extraction_method)",
                
                # Relationship indexes for OpenIE
                "CREATE INDEX openie_relation_confidence IF NOT EXISTS FOR ()-[r:OPENIE_RELATION]-() ON (r.confidence)",
                "CREATE INDEX openie_relation_document IF NOT EXISTS FOR ()-[r:OPENIE_RELATION]-() ON (r.document_id)",
                "CREATE INDEX openie_relation_method IF NOT EXISTS FOR ()-[r:OPENIE_RELATION]-() ON (r.extraction_method)",
                
                # Provenance indexes
                "CREATE INDEX openie_provenance_sentence IF NOT EXISTS FOR ()-[r]-() ON (r.source_sentence)",
                "CREATE INDEX openie_provenance_created IF NOT EXISTS FOR ()-[r]-() ON (r.created_at)"
            ]
            
            for query in schema_queries:
                try:
                    await self._execute_query(query)
                    logger.debug("Schema query executed", query=query[:50] + "...")
                except Exception as e:
                    # Some constraints might already exist, which is fine
                    if "already exists" not in str(e).lower():
                        logger.warning("Schema query failed", query=query[:50], error=str(e))
            
            logger.info("OpenIE schema initialization completed")
            
        except Exception as e:
            logger.error("Failed to initialize OpenIE schema", error=str(e))
            raise
    
    async def store_triplets(
        self,
        triplets: List[ValidatedTriplet],
        entity_matches: Optional[List[EntityMatch]] = None,
        normalized_predicates: Optional[List[NormalizedPredicate]] = None,
        source_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store OpenIE triplets in Neo4j with batch processing.
        
        Args:
            triplets: List of validated triplets
            entity_matches: Optional entity matches for linking
            normalized_predicates: Optional normalized predicates
            source_doc_id: Optional source document ID
            
        Returns:
            Storage results with counts and metadata
        """
        if not triplets:
            return {
                "triplets_stored": 0,
                "relationships_created": 0,
                "nodes_created": 0,
                "source_doc_id": source_doc_id
            }
        
        try:
            logger.info(
                "Starting OpenIE triplet storage",
                triplet_count=len(triplets),
                entity_matches=len(entity_matches) if entity_matches else 0,
                normalized_predicates=len(normalized_predicates) if normalized_predicates else 0,
                source_doc_id=source_doc_id
            )
            
            # Create lookup dictionaries
            entity_match_lookup = {}
            if entity_matches:
                entity_match_lookup = {match.openie_entity: match for match in entity_matches}
            
            predicate_lookup = {}
            if normalized_predicates:
                predicate_lookup = {pred.original: pred for pred in normalized_predicates}
            
            # Process triplets in batches
            total_stored = 0
            total_relationships = 0
            total_nodes = 0
            
            for i in range(0, len(triplets), self.batch_size):
                batch = triplets[i:i + self.batch_size]
                batch_results = await self._store_batch(
                    batch, entity_match_lookup, predicate_lookup, source_doc_id
                )
                
                total_stored += batch_results["triplets_processed"]
                total_relationships += batch_results["relationships_created"]
                total_nodes += batch_results["nodes_created"]
                
                logger.debug(
                    "Batch stored",
                    batch_size=len(batch),
                    batch_index=i // self.batch_size + 1,
                    total_batches=(len(triplets) + self.batch_size - 1) // self.batch_size
                )
            
            results = {
                "triplets_stored": total_stored,
                "relationships_created": total_relationships,
                "nodes_created": total_nodes,
                "source_doc_id": source_doc_id
            }
            
            logger.info(
                "OpenIE triplet storage completed",
                **results
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "OpenIE triplet storage failed",
                error=str(e),
                error_type=type(e).__name__,
                triplet_count=len(triplets),
                source_doc_id=source_doc_id
            )
            raise
    
    async def _store_batch(
        self,
        batch: List[ValidatedTriplet],
        entity_match_lookup: Dict[str, EntityMatch],
        predicate_lookup: Dict[str, NormalizedPredicate],
        source_doc_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Store a batch of triplets with retry logic."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with self.driver.session(database=self.database) as session:
                    result = await session.execute_write(
                        self._execute_batch_storage,
                        batch,
                        entity_match_lookup,
                        predicate_lookup,
                        source_doc_id
                    )
                    return result
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(
                        "Batch storage failed after retries",
                        error=str(e),
                        retry_count=retry_count,
                        batch_size=len(batch)
                    )
                    raise
                
                logger.warning(
                    "Batch storage failed, retrying",
                    error=str(e),
                    retry_count=retry_count,
                    max_retries=self.max_retries
                )
                await asyncio.sleep(self.retry_delay * retry_count)
    
    async def _execute_batch_storage(
        self,
        tx,
        batch: List[ValidatedTriplet],
        entity_match_lookup: Dict[str, EntityMatch],
        predicate_lookup: Dict[str, NormalizedPredicate],
        source_doc_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Execute batch storage in a transaction."""
        nodes_created = 0
        relationships_created = 0
        triplets_processed = 0
        
        current_time = datetime.utcnow().isoformat()
        
        # Prepare batch data
        entities_data = []
        triplets_data = []
        relationships_data = []
        
        for triplet in batch:
            triplet_id = str(uuid4())
            
            # Get entity matches
            subject_match = entity_match_lookup.get(triplet.subject)
            object_match = entity_match_lookup.get(triplet.object)
            
            # Get normalized predicate
            predicate_norm = predicate_lookup.get(triplet.predicate)
            
            # Determine canonical forms
            subject_canonical = subject_match.spacy_entity.canonical_name if subject_match else triplet.subject
            object_canonical = object_match.spacy_entity.canonical_name if object_match else triplet.object
            predicate_canonical = predicate_norm.canonical_form if predicate_norm else triplet.predicate
            
            # Prepare entity data
            for entity_text, canonical_form, is_linked in [
                (triplet.subject, subject_canonical, subject_match is not None),
                (triplet.object, object_canonical, object_match is not None)
            ]:
                entities_data.append({
                    "text": entity_text,
                    "canonical_form": canonical_form,
                    "extraction_method": "openie",
                    "confidence": triplet.confidence,
                    "document_id": source_doc_id,
                    "created_at": current_time,
                    "is_linked": is_linked
                })
            
            # Prepare triplet metadata
            triplets_data.append({
                "triplet_id": triplet_id,
                "subject_text": triplet.subject,
                "predicate_text": triplet.predicate,
                "object_text": triplet.object,
                "confidence": triplet.confidence,
                "validation_score": triplet.validation_score,
                "source_sentence": triplet.sentence,
                "sentence_id": triplet.sentence_id,
                "document_id": source_doc_id,
                "created_at": current_time,
                "extraction_method": "openie",
                "validation_flags": list(triplet.validation_flags)
            })
            
            # Prepare relationship data
            relationships_data.append({
                "triplet_id": triplet_id,
                "subject_canonical": subject_canonical,
                "object_canonical": object_canonical,
                "predicate_canonical": predicate_canonical,
                "original_predicate": triplet.predicate,
                "confidence": triplet.confidence,
                "validation_score": triplet.validation_score,
                "source_sentence": triplet.sentence,
                "sentence_id": triplet.sentence_id,
                "document_id": source_doc_id,
                "created_at": current_time,
                "extraction_method": "openie",
                "subject_linked": subject_match is not None,
                "object_linked": object_match is not None,
                "predicate_normalized": predicate_norm is not None
            })
        
        # 1. Create/merge entities
        if entities_data:
            entity_query = """
            UNWIND $entities AS entity
            MERGE (e:Entity {canonical_form: entity.canonical_form})
            ON CREATE SET 
                e.text = entity.text,
                e.extraction_method = entity.extraction_method,
                e.confidence = entity.confidence,
                e.document_id = entity.document_id,
                e.created_at = entity.created_at,
                e.is_openie_linked = entity.is_linked
            ON MATCH SET
                e.confidence = CASE WHEN entity.confidence > e.confidence THEN entity.confidence ELSE e.confidence END,
                e.is_openie_linked = COALESCE(e.is_openie_linked, entity.is_linked)
            RETURN count(e) as entities_processed
            """
            
            result = await tx.run(entity_query, entities=entities_data)
            record = await result.single()
            if record:
                nodes_created += record["entities_processed"]
        
        # 2. Create triplet metadata nodes
        if triplets_data:
            triplet_query = """
            UNWIND $triplets AS triplet
            CREATE (t:OpenIETriplet)
            SET t = triplet
            RETURN count(t) as triplets_created
            """
            
            result = await tx.run(triplet_query, triplets=triplets_data)
            record = await result.single()
            if record:
                nodes_created += record["triplets_created"]
        
        # 3. Create relationships
        for rel_data in relationships_data:
            # Use normalized predicate as relationship type, fallback to generic
            relationship_type = self._normalize_relationship_type(rel_data["predicate_canonical"])
            
            rel_query = f"""
            MATCH (s:Entity {{canonical_form: $subject_canonical}})
            MATCH (o:Entity {{canonical_form: $object_canonical}})
            MATCH (t:OpenIETriplet {{triplet_id: $triplet_id}})
            CREATE (s)-[r:{relationship_type}]->(o)
            SET r.confidence = $confidence,
                r.validation_score = $validation_score,
                r.extraction_method = $extraction_method,
                r.source_sentence = $source_sentence,
                r.sentence_id = $sentence_id,
                r.document_id = $document_id,
                r.created_at = $created_at,
                r.original_predicate = $original_predicate,
                r.subject_linked = $subject_linked,
                r.object_linked = $object_linked,
                r.predicate_normalized = $predicate_normalized
            CREATE (t)-[:REPRESENTS]->(r)
            RETURN count(r) as relationships_created
            """
            
            result = await tx.run(rel_query, **rel_data)
            record = await result.single()
            if record:
                relationships_created += record["relationships_created"]
        
        triplets_processed = len(batch)
        
        return {
            "triplets_processed": triplets_processed,
            "relationships_created": relationships_created,
            "nodes_created": nodes_created
        }
    
    def _normalize_relationship_type(self, predicate: str) -> str:
        """Normalize predicate to valid Neo4j relationship type."""
        # Convert to uppercase and replace invalid characters
        normalized = predicate.upper().replace(" ", "_").replace("-", "_")
        
        # Remove invalid characters for Neo4j relationship types
        import re
        normalized = re.sub(r'[^A-Z0-9_]', '', normalized)
        
        # Ensure it starts with a letter
        if not normalized or not normalized[0].isalpha():
            normalized = "OPENIE_" + normalized
        
        # Fallback for empty or invalid types
        if not normalized or len(normalized) < 2:
            normalized = "OPENIE_RELATION"
        
        return normalized
