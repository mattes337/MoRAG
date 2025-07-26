"""Enhanced graph builder with OpenIE and SpaCy integration."""

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

# SpaCy imports (optional)
try:
    from ..extraction import SpacyEntityExtractor
    from ..normalizers import SpacyNormalizer
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    SpacyEntityExtractor = None
    SpacyNormalizer = None

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
    """Enhanced result with OpenIE and SpaCy extraction data."""
    # OpenIE results
    openie_relations_created: int = 0
    openie_triplets_processed: int = 0
    openie_entity_matches: int = 0
    openie_normalized_predicates: int = 0
    openie_enabled: bool = False
    openie_metadata: Dict[str, Any] = field(default_factory=dict)

    # SpaCy results
    spacy_entities_extracted: int = 0
    spacy_entities_normalized: int = 0
    spacy_entities_merged: int = 0
    spacy_enabled: bool = False
    spacy_languages_detected: List[str] = field(default_factory=list)
    spacy_metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedGraphBuilder:
    """Enhanced graph builder with OpenIE and SpaCy integration.

    This builder extends the standard graph building process to include:
    - SpaCy-based entity extraction with linguistic normalization
    - OpenIE-based relation extraction
    - Traditional LLM-based extraction
    - Entity deduplication and merging across extractors
    """

    def __init__(
        self,
        storage: BaseStorage,
        llm_config=None,
        entity_types: Optional[Dict[str, str]] = None,
        relation_types: Optional[Dict[str, str]] = None,
        enable_spacy: bool = True,
        spacy_config: Optional[Dict[str, Any]] = None,
        enable_openie: bool = True,
        openie_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the enhanced graph builder.

        Args:
            storage: Graph storage backend
            llm_config: Configuration for LLM-based extraction
            entity_types: Custom entity types for extraction
            relation_types: Custom relation types for extraction
            enable_spacy: Whether to enable SpaCy extraction
            spacy_config: Configuration for SpaCy extraction
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

        # Initialize SpaCy extractor if available and enabled
        self.spacy_enabled = enable_spacy and _SPACY_AVAILABLE
        self.spacy_extractor = None
        self.spacy_normalizer = None

        if self.spacy_enabled:
            try:
                spacy_config = spacy_config or {}
                self.spacy_extractor = SpacyEntityExtractor(**spacy_config)
                self.spacy_normalizer = SpacyNormalizer(**spacy_config)
                self.logger.info("SpaCy extractor and normalizer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SpaCy extractor: {e}")
                self.spacy_enabled = False
        elif enable_spacy and not _SPACY_AVAILABLE:
            self.logger.warning("SpaCy requested but not available")
        
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
            spacy_enabled=self.spacy_enabled,
            spacy_available=_SPACY_AVAILABLE,
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
            
            # Step 1: Extract entities from content using multiple extractors
            self.logger.debug("Extracting entities with LLM...")
            llm_entities = await self.entity_extractor.extract(
                content,
                source_doc_id=document_id
            )

            # Step 1b: Extract entities using SpaCy (if enabled)
            spacy_entities = []
            spacy_result = None

            if self.spacy_enabled and self.spacy_extractor:
                try:
                    self.logger.debug("Extracting entities with SpaCy...")
                    spacy_entities = await self.spacy_extractor.extract(
                        content,
                        source_doc_id=document_id
                    )

                    # Normalize SpaCy entities
                    if self.spacy_normalizer and spacy_entities:
                        entity_texts = [e.name for e in spacy_entities]
                        entity_types = [e.type for e in spacy_entities]

                        normalized_entities = await self.spacy_normalizer.normalize_entities(
                            entity_texts,
                            entity_types=entity_types,
                            source_doc_id=document_id
                        )

                        # Update SpaCy entities with normalized forms
                        for i, normalized in enumerate(normalized_entities):
                            if i < len(spacy_entities):
                                spacy_entities[i].name = normalized.normalized_text
                                spacy_entities[i].attributes.update({
                                    'original_name': normalized.original_text,
                                    'canonical_form': normalized.canonical_form,
                                    'normalization_confidence': normalized.confidence,
                                    'normalization_method': normalized.normalization_method,
                                    'variations': normalized.variations
                                })

                except Exception as e:
                    self.logger.error(f"SpaCy extraction failed: {e}")
                    # Continue with LLM-only results

            # Step 1c: Merge and deduplicate entities from different extractors
            entities = await self._merge_entities(llm_entities, spacy_entities, document_id)
            
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
                # SpaCy results
                spacy_enabled=self.spacy_enabled,
                spacy_entities_extracted=len(spacy_entities),
                spacy_entities_normalized=len(spacy_entities) if self.spacy_normalizer else 0,
                spacy_entities_merged=len([e for e in entities if e.attributes.get('extraction_method') == 'spacy']),
                spacy_languages_detected=list(set(e.attributes.get('language', 'unknown') for e in spacy_entities)),
                spacy_metadata={
                    'spacy_extractor_available': self.spacy_extractor is not None,
                    'spacy_normalizer_available': self.spacy_normalizer is not None,
                    'spacy_model_info': self.spacy_extractor.get_model_info() if self.spacy_extractor else {}
                },
                # OpenIE results
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
                llm_entities_extracted=len(llm_entities),
                spacy_entities_extracted=enhanced_result.spacy_entities_extracted,
                total_entities_created=enhanced_result.entities_created,
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

    async def _merge_entities(self,
                             llm_entities: List[Entity],
                             spacy_entities: List[Entity],
                             document_id: str) -> List[Entity]:
        """Merge and deduplicate entities from different extractors.

        Args:
            llm_entities: Entities from LLM extraction
            spacy_entities: Entities from SpaCy extraction
            document_id: Document ID for logging

        Returns:
            Merged list of entities with duplicates removed
        """
        if not spacy_entities:
            return llm_entities

        if not llm_entities:
            return spacy_entities

        try:
            merged_entities = []
            used_spacy_indices = set()

            # Start with LLM entities as base
            for llm_entity in llm_entities:
                best_match = None
                best_match_index = -1
                best_similarity = 0.0

                # Find best matching SpaCy entity
                for i, spacy_entity in enumerate(spacy_entities):
                    if i in used_spacy_indices:
                        continue

                    similarity = self._calculate_entity_similarity(llm_entity, spacy_entity)

                    if similarity > best_similarity and similarity > 0.7:  # Threshold for merging
                        best_match = spacy_entity
                        best_match_index = i
                        best_similarity = similarity

                if best_match:
                    # Merge entities - prefer LLM entity but add SpaCy attributes
                    merged_entity = self._merge_entity_pair(llm_entity, best_match)
                    merged_entities.append(merged_entity)
                    used_spacy_indices.add(best_match_index)
                else:
                    # No match found, keep LLM entity as-is
                    merged_entities.append(llm_entity)

            # Add remaining SpaCy entities that weren't merged
            for i, spacy_entity in enumerate(spacy_entities):
                if i not in used_spacy_indices:
                    merged_entities.append(spacy_entity)

            self.logger.debug(
                "Entity merging completed",
                document_id=document_id,
                llm_entities=len(llm_entities),
                spacy_entities=len(spacy_entities),
                merged_entities=len(merged_entities),
                entities_merged=len(used_spacy_indices)
            )

            return merged_entities

        except Exception as e:
            self.logger.error(f"Entity merging failed: {e}")
            # Fallback: return all entities without merging
            return llm_entities + spacy_entities

    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between two entities.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Exact name match
        if entity1.name.lower() == entity2.name.lower():
            return 1.0

        # Check normalized forms if available
        canonical1 = entity1.attributes.get('canonical_form', entity1.name.lower())
        canonical2 = entity2.attributes.get('canonical_form', entity2.name.lower())

        if canonical1 == canonical2:
            return 0.95

        # Check variations
        variations1 = entity1.attributes.get('variations', [])
        variations2 = entity2.attributes.get('variations', [])

        for var1 in variations1:
            if var1.lower() == entity2.name.lower():
                return 0.9
            for var2 in variations2:
                if var1.lower() == var2.lower():
                    return 0.85

        # Simple string similarity (Jaccard similarity on words)
        words1 = set(entity1.name.lower().split())
        words2 = set(entity2.name.lower().split())

        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_similarity = intersection / union

            # Boost similarity if entity types match
            if entity1.type == entity2.type:
                jaccard_similarity *= 1.2

            return min(1.0, jaccard_similarity)

        return 0.0

    def _merge_entity_pair(self, llm_entity: Entity, spacy_entity: Entity) -> Entity:
        """Merge two entities, preferring LLM entity but adding SpaCy attributes.

        Args:
            llm_entity: Entity from LLM extraction
            spacy_entity: Entity from SpaCy extraction

        Returns:
            Merged entity
        """
        # Start with LLM entity as base
        merged_entity = Entity(
            id=llm_entity.id,
            name=llm_entity.name,
            type=llm_entity.type,
            source_doc_id=llm_entity.source_doc_id,
            confidence=max(llm_entity.confidence, spacy_entity.confidence),
            attributes=llm_entity.attributes.copy()
        )

        # Add SpaCy-specific attributes with prefix
        spacy_attrs = {
            f'spacy_{key}': value
            for key, value in spacy_entity.attributes.items()
            if not key.startswith('spacy_')
        }

        merged_entity.attributes.update(spacy_attrs)

        # Add extraction methods
        extraction_methods = merged_entity.attributes.get('extraction_methods', [])
        if 'llm' not in extraction_methods:
            extraction_methods.append('llm')
        if 'spacy' not in extraction_methods:
            extraction_methods.append('spacy')
        merged_entity.attributes['extraction_methods'] = extraction_methods

        # Merge variations
        llm_variations = merged_entity.attributes.get('variations', [])
        spacy_variations = spacy_entity.attributes.get('variations', [])
        all_variations = list(set(llm_variations + spacy_variations + [spacy_entity.name]))
        merged_entity.attributes['variations'] = all_variations

        return merged_entity
