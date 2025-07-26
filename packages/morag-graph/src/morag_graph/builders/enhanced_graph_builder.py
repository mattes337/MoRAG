"""Enhanced graph builder with OpenIE integration."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..storage.base import BaseStorage
from ..models import Entity, Relation, DocumentChunk
from ..extraction import EntityExtractor, RelationExtractor
from ..updates.checksum_manager import DocumentChecksumManager
from ..updates.cleanup_manager import DocumentCleanupManager, CleanupResult
from .graph_builder import GraphBuildResult, GraphBuildError

# OpenIE imports (optional)
try:
    from ..extractors import OpenIEExtractor
    from ..storage.neo4j_storage import Neo4jStorage
    _OPENIE_AVAILABLE = True
except ImportError:
    _OPENIE_AVAILABLE = False
    OpenIEExtractor = None


@dataclass
class EnhancedGraphBuildResult(GraphBuildResult):
    """Enhanced result with OpenIE extraction data."""
    openie_relations_created: int = 0
    openie_triplets_processed: int = 0
    openie_entity_matches: int = 0
    openie_normalized_predicates: int = 0
    openie_enabled: bool = False
    openie_metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedGraphBuilder:
    """Enhanced graph builder with OpenIE integration.
    
    This builder extends the standard graph building process to include
    OpenIE-based relation extraction alongside traditional LLM-based extraction.
    """
    
    def __init__(
        self,
        storage: BaseStorage,
        llm_config=None,
        entity_types: Optional[Dict[str, str]] = None,
        relation_types: Optional[Dict[str, str]] = None,
        enable_openie: bool = True,
        openie_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the enhanced graph builder.
        
        Args:
            storage: Graph storage backend
            llm_config: Configuration for LLM-based extraction
            entity_types: Custom entity types for extraction
            relation_types: Custom relation types for extraction
            enable_openie: Whether to enable OpenIE extraction
            openie_config: Configuration for OpenIE extraction
        """
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # Initialize standard extractors
        self.entity_extractor = EntityExtractor(
            config=llm_config,
            entity_types=entity_types
        )
        
        self.relation_extractor = RelationExtractor(
            config=llm_config,
            relation_types=relation_types
        )
        
        # Initialize OpenIE extractor if available and enabled
        self.openie_enabled = enable_openie and _OPENIE_AVAILABLE
        self.openie_extractor = None
        
        if self.openie_enabled:
            try:
                self.openie_extractor = OpenIEExtractor(openie_config)
                self.logger.info("OpenIE extractor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenIE extractor: {e}")
                self.openie_enabled = False
        elif enable_openie and not _OPENIE_AVAILABLE:
            self.logger.warning("OpenIE requested but not available")
        
        # Initialize checksum and cleanup managers
        self.checksum_manager = DocumentChecksumManager(storage)
        self.cleanup_manager = DocumentCleanupManager(storage)
        
        self.logger.info(
            "Enhanced graph builder initialized",
            openie_enabled=self.openie_enabled,
            openie_available=_OPENIE_AVAILABLE
        )
    
    async def process_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnhancedGraphBuildResult:
        """Process a document with enhanced extraction including OpenIE.
        
        Args:
            content: Document content to process
            document_id: Unique identifier for the document
            metadata: Optional metadata about the document
            
        Returns:
            EnhancedGraphBuildResult with processing results
            
        Raises:
            GraphBuildError: If processing fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting enhanced graph processing for document {document_id}")
            
            # Check if document needs processing based on checksum
            needs_update = await self.checksum_manager.needs_update(
                document_id, content, metadata
            )
            
            if not needs_update:
                # Document unchanged, skip processing
                processing_time = time.time() - start_time
                self.logger.info(f"Document {document_id} unchanged, skipped processing")
                
                return EnhancedGraphBuildResult(
                    document_id=document_id,
                    entities_created=0,
                    relations_created=0,
                    entity_ids=[],
                    relation_ids=[],
                    processing_time=processing_time,
                    skipped=True,
                    openie_enabled=self.openie_enabled
                )
            
            # Document changed, cleanup existing data first
            cleanup_result = await self.cleanup_manager.cleanup_document_data(document_id)
            
            # Initialize OpenIE schema if needed
            if self.openie_enabled and isinstance(self.storage, Neo4jStorage):
                try:
                    await self.storage.initialize_openie_schema()
                except Exception as e:
                    self.logger.warning(f"Failed to initialize OpenIE schema: {e}")
            
            # Step 1: Extract entities from content
            self.logger.debug("Extracting entities...")
            entities = await self.entity_extractor.extract(
                content,
                source_doc_id=document_id
            )
            
            # Step 2: Extract relations using standard LLM-based extractor
            self.logger.debug("Extracting relations with LLM...")
            llm_relations = await self.relation_extractor.extract(
                content,
                entities=entities,
                source_doc_id=document_id
            )
            
            # Step 3: Extract relations using OpenIE (if enabled)
            openie_relations = []
            openie_result = None
            
            if self.openie_enabled and self.openie_extractor:
                try:
                    self.logger.debug("Extracting relations with OpenIE...")
                    openie_result = await self.openie_extractor.extract_full(
                        content,
                        entities=entities,
                        source_doc_id=document_id
                    )
                    openie_relations = openie_result.relations
                    
                    # Store OpenIE triplets in Neo4j if available
                    if isinstance(self.storage, Neo4jStorage) and openie_result.triplets:
                        await self.storage.store_openie_triplets(
                            openie_result.triplets,
                            openie_result.entity_matches,
                            openie_result.normalized_predicates,
                            document_id
                        )
                    
                except Exception as e:
                    self.logger.error(f"OpenIE extraction failed: {e}")
                    # Continue with LLM-only results
            
            # Step 4: Combine relations from both extractors
            all_relations = llm_relations + openie_relations
            
            # Step 5: Store entities and relations
            result = await self._store_entities_and_relations(
                entities, all_relations, document_id
            )
            
            # Step 6: Update checksum
            await self.checksum_manager.update_checksum(
                document_id, content, metadata
            )
            
            processing_time = time.time() - start_time
            
            # Create enhanced result
            enhanced_result = EnhancedGraphBuildResult(
                document_id=document_id,
                entities_created=result.entities_created,
                relations_created=result.relations_created,
                entity_ids=result.entity_ids,
                relation_ids=result.relation_ids,
                processing_time=processing_time,
                cleanup_result=cleanup_result,
                openie_enabled=self.openie_enabled,
                openie_relations_created=len(openie_relations),
                openie_triplets_processed=len(openie_result.triplets) if openie_result else 0,
                openie_entity_matches=len(openie_result.entity_matches) if openie_result else 0,
                openie_normalized_predicates=len(openie_result.normalized_predicates) if openie_result else 0,
                openie_metadata=openie_result.metadata if openie_result else {}
            )
            
            self.logger.info(
                "Enhanced graph processing completed",
                document_id=document_id,
                entities_created=enhanced_result.entities_created,
                llm_relations_created=len(llm_relations),
                openie_relations_created=enhanced_result.openie_relations_created,
                total_relations_created=enhanced_result.relations_created,
                processing_time=processing_time
            )
            
            return enhanced_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Enhanced graph processing failed for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return EnhancedGraphBuildResult(
                document_id=document_id,
                entities_created=0,
                relations_created=0,
                entity_ids=[],
                relation_ids=[],
                processing_time=processing_time,
                errors=[error_msg],
                openie_enabled=self.openie_enabled
            )
    
    async def process_document_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnhancedGraphBuildResult:
        """Process document chunks with enhanced extraction.
        
        Args:
            chunks: List of document chunks to process
            document_id: Unique identifier for the document
            metadata: Optional metadata about the document
            
        Returns:
            EnhancedGraphBuildResult with processing results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing {len(chunks)} chunks for document {document_id}")
            
            # Initialize OpenIE schema if needed
            if self.openie_enabled and isinstance(self.storage, Neo4jStorage):
                try:
                    await self.storage.initialize_openie_schema()
                except Exception as e:
                    self.logger.warning(f"Failed to initialize OpenIE schema: {e}")
            
            all_entities = []
            all_llm_relations = []
            all_openie_relations = []
            total_openie_triplets = 0
            total_entity_matches = 0
            total_normalized_predicates = 0
            openie_metadata = {}
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                try:
                    self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Extract entities from chunk
                    entities = await self.entity_extractor.extract(
                        chunk.text,
                        source_doc_id=document_id
                    )
                    
                    # Extract relations with LLM
                    llm_relations = await self.relation_extractor.extract(
                        chunk.text,
                        entities=entities,
                        source_doc_id=document_id
                    )
                    
                    # Extract relations with OpenIE
                    openie_relations = []
                    if self.openie_enabled and self.openie_extractor:
                        try:
                            openie_result = await self.openie_extractor.extract_full(
                                chunk.text,
                                entities=entities,
                                source_doc_id=document_id
                            )
                            openie_relations = openie_result.relations
                            
                            # Store OpenIE triplets
                            if isinstance(self.storage, Neo4jStorage) and openie_result.triplets:
                                await self.storage.store_openie_triplets(
                                    openie_result.triplets,
                                    openie_result.entity_matches,
                                    openie_result.normalized_predicates,
                                    document_id
                                )
                            
                            # Accumulate OpenIE stats
                            total_openie_triplets += len(openie_result.triplets)
                            total_entity_matches += len(openie_result.entity_matches)
                            total_normalized_predicates += len(openie_result.normalized_predicates)
                            
                            # Merge metadata
                            for key, value in openie_result.metadata.items():
                                if key in openie_metadata:
                                    if isinstance(value, (int, float)):
                                        openie_metadata[key] += value
                                    else:
                                        openie_metadata[key] = value
                                else:
                                    openie_metadata[key] = value
                            
                        except Exception as e:
                            self.logger.warning(f"OpenIE extraction failed for chunk {i}: {e}")
                    
                    # Add chunk metadata
                    chunk_metadata = {
                        'chunk_id': chunk.id,
                        'chunk_index': i,
                        'chunk_type': chunk.chunk_type
                    }
                    
                    if chunk.metadata:
                        chunk_metadata.update(chunk.metadata)
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    # Update entity and relation metadata
                    for entity in entities:
                        entity.metadata.update(chunk_metadata)
                    
                    for relation in llm_relations + openie_relations:
                        relation.metadata.update(chunk_metadata)
                    
                    all_entities.extend(entities)
                    all_llm_relations.extend(llm_relations)
                    all_openie_relations.extend(openie_relations)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {i}: {e}")
                    continue
            
            # Store all entities and relations
            all_relations = all_llm_relations + all_openie_relations
            result = await self._store_entities_and_relations(
                all_entities, all_relations, document_id
            )
            
            processing_time = time.time() - start_time
            
            enhanced_result = EnhancedGraphBuildResult(
                document_id=document_id,
                entities_created=result.entities_created,
                relations_created=result.relations_created,
                entity_ids=result.entity_ids,
                relation_ids=result.relation_ids,
                processing_time=processing_time,
                chunks_processed=len(chunks),
                openie_enabled=self.openie_enabled,
                openie_relations_created=len(all_openie_relations),
                openie_triplets_processed=total_openie_triplets,
                openie_entity_matches=total_entity_matches,
                openie_normalized_predicates=total_normalized_predicates,
                openie_metadata=openie_metadata
            )
            
            self.logger.info(
                "Enhanced chunk processing completed",
                document_id=document_id,
                chunks_processed=len(chunks),
                entities_created=enhanced_result.entities_created,
                llm_relations_created=len(all_llm_relations),
                openie_relations_created=enhanced_result.openie_relations_created,
                total_relations_created=enhanced_result.relations_created,
                processing_time=processing_time
            )
            
            return enhanced_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Enhanced chunk processing failed for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return EnhancedGraphBuildResult(
                document_id=document_id,
                entities_created=0,
                relations_created=0,
                entity_ids=[],
                relation_ids=[],
                processing_time=processing_time,
                chunks_processed=len(chunks),
                errors=[error_msg],
                openie_enabled=self.openie_enabled
            )
    
    async def _store_entities_and_relations(
        self,
        entities: List[Entity],
        relations: List[Relation],
        document_id: str
    ) -> GraphBuildResult:
        """Store entities and relations in the storage backend."""
        try:
            # Store entities
            entity_ids = []
            if entities:
                await self.storage.store_entities(entities)
                entity_ids = [entity.id for entity in entities]
            
            # Store relations
            relation_ids = []
            if relations:
                await self.storage.store_relations(relations)
                relation_ids = [relation.id for relation in relations]
            
            return GraphBuildResult(
                document_id=document_id,
                entities_created=len(entities),
                relations_created=len(relations),
                entity_ids=entity_ids,
                relation_ids=relation_ids
            )
            
        except Exception as e:
            error_msg = f"Failed to store entities and relations: {str(e)}"
            self.logger.error(error_msg)
            raise GraphBuildError(error_msg) from e
