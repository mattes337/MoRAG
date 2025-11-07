"""Enhanced graph builder with LangExtract integration."""

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


@dataclass
class EnhancedGraphBuildResult(GraphBuildResult):
    """Enhanced result with LangExtract and SpaCy extraction data."""

    # LangExtract results
    langextract_entities_extracted: int = 0
    langextract_relations_created: int = 0
    langextract_enabled: bool = True
    langextract_domain: str = "general"
    langextract_metadata: Dict[str, Any] = field(default_factory=dict)

    # SpaCy results
    spacy_entities_extracted: int = 0
    spacy_entities_normalized: int = 0
    spacy_entities_merged: int = 0
    spacy_enabled: bool = False


class EnhancedGraphBuilder:
    """Enhanced graph builder with LangExtract and SpaCy integration.

    This builder provides:
    - LangExtract-based entity and relation extraction
    - Optional SpaCy integration for additional entity extraction
    - Document update tracking and cleanup
    - Comprehensive result reporting
    """

    def __init__(
        self,
        storage: BaseStorage,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        enable_spacy: bool = False,
        spacy_config: Optional[Dict[str, Any]] = None,
        domain: str = "general",
        min_confidence: float = 0.6,
        chunk_size: int = 1000,
        max_workers: int = 10,
        **kwargs
    ):
        """Initialize the enhanced graph builder.

        Args:
            storage: Storage backend for graph data
            entity_extractor: Custom entity extractor (uses LangExtract by default)
            relation_extractor: Custom relation extractor (uses LangExtract by default)
            enable_spacy: Whether to enable SpaCy extraction
            spacy_config: Configuration for SpaCy extraction
            domain: Domain for LangExtract specialization
            min_confidence: Minimum confidence threshold
            chunk_size: Text chunk size for processing
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments
        """
        self.storage = storage
        self.domain = domain
        self.min_confidence = min_confidence
        self.chunk_size = chunk_size
        self.max_workers = max_workers

        # Initialize LangExtract extractors
        self.entity_extractor = entity_extractor or EntityExtractor(
            domain=domain,
            min_confidence=min_confidence,
            chunk_size=chunk_size,
            max_workers=max_workers
        )

        self.relation_extractor = relation_extractor or RelationExtractor(
            domain=domain,
            min_confidence=min_confidence,
            chunk_size=chunk_size,
            max_workers=max_workers
        )

        # Initialize SpaCy extractor if available and enabled
        self.spacy_enabled = enable_spacy and _SPACY_AVAILABLE
        self.spacy_extractor = None
        self.spacy_normalizer = None

        if self.spacy_enabled:
            try:
                self.spacy_extractor = SpacyEntityExtractor(spacy_config)
                self.spacy_normalizer = SpacyNormalizer()
                logging.info("SpaCy extractor initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize SpaCy extractor: {e}")
                self.spacy_enabled = False
        elif enable_spacy and not _SPACY_AVAILABLE:
            logging.warning("SpaCy requested but not available")

        # Initialize document management
        self.checksum_manager = DocumentChecksumManager(storage)
        self.cleanup_manager = DocumentCleanupManager(storage)

        self.logger = logging.getLogger(__name__)

        self.logger.info(
            "Enhanced graph builder initialized",
            langextract_enabled=True,
            spacy_enabled=self.spacy_enabled,
            domain=self.domain,
            min_confidence=self.min_confidence
        )

    async def process_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_update: bool = False
    ) -> EnhancedGraphBuildResult:
        """Process a document with enhanced extraction including LangExtract.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            metadata: Optional document metadata
            force_update: Whether to force update even if unchanged

        Returns:
            EnhancedGraphBuildResult with detailed extraction statistics
        """
        start_time = time.time()

        try:
            # Check if document needs updating
            if not force_update:
                needs_update = await self.checksum_manager.needs_update(doc_id, text)
                if not needs_update:
                    self.logger.info(f"Document {doc_id} unchanged, skipping processing")
                    return EnhancedGraphBuildResult(
                        success=True,
                        doc_id=doc_id,
                        processing_time=time.time() - start_time,
                        langextract_enabled=True,
                        langextract_domain=self.domain,
                        spacy_enabled=self.spacy_enabled
                    )

            # Clean up existing data for this document
            cleanup_result = await self.cleanup_manager.cleanup_document(doc_id)

            # Step 1: Extract entities using LangExtract
            self.logger.debug("Extracting entities with LangExtract...")
            entities = await self.entity_extractor.extract(text, source_doc_id=doc_id)

            # Step 2: Extract additional entities with SpaCy (if enabled)
            spacy_entities = []
            if self.spacy_enabled and self.spacy_extractor:
                try:
                    self.logger.debug("Extracting entities with SpaCy...")
                    spacy_entities = await self.spacy_extractor.extract(text, source_doc_id=doc_id)

                    # Normalize and merge SpaCy entities
                    if self.spacy_normalizer:
                        spacy_entities = await self.spacy_normalizer.normalize_entities(spacy_entities)

                except Exception as e:
                    self.logger.error(f"SpaCy extraction failed: {e}")

            # Combine entities
            all_entities = entities + spacy_entities

            # Step 3: Extract relations using LangExtract
            self.logger.debug("Extracting relations with LangExtract...")
            relations = await self.relation_extractor.extract(text, entities=all_entities, source_doc_id=doc_id)

            # Step 4: Store entities and relations
            stored_entities = 0
            stored_relations = 0

            if all_entities:
                stored_entities = await self.storage.store_entities(all_entities)

            if relations:
                stored_relations = await self.storage.store_relations(relations)

            # Update document checksum
            await self.checksum_manager.update_checksum(doc_id, text)

            processing_time = time.time() - start_time

            # Create enhanced result
            result = EnhancedGraphBuildResult(
                success=True,
                doc_id=doc_id,
                entities_created=stored_entities,
                relations_created=stored_relations,
                processing_time=processing_time,

                # LangExtract results
                langextract_enabled=True,
                langextract_entities_extracted=len(entities),
                langextract_relations_created=len(relations),
                langextract_domain=self.domain,
                langextract_metadata={
                    "entity_types": list(set(e.type for e in entities)),
                    "relation_types": list(set(r.type for r in relations)),
                    "avg_entity_confidence": sum(e.confidence for e in entities) / len(entities) if entities else 0,
                    "avg_relation_confidence": sum(r.confidence for r in relations) / len(relations) if relations else 0
                },

                # SpaCy results
                spacy_enabled=self.spacy_enabled,
                spacy_entities_extracted=len(spacy_entities),
                spacy_entities_normalized=len(spacy_entities) if self.spacy_normalizer else 0,
                spacy_entities_merged=len(spacy_entities)
            )

            self.logger.info(
                "Document processing completed",
                doc_id=doc_id,
                entities_created=stored_entities,
                relations_created=stored_relations,
                langextract_entities=len(entities),
                langextract_relations=len(relations),
                spacy_entities=len(spacy_entities),
                processing_time=processing_time
            )

            return result

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}", exc_info=True)
            return EnhancedGraphBuildResult(
                success=False,
                doc_id=doc_id,
                error=str(e),
                processing_time=time.time() - start_time,
                langextract_enabled=True,
                langextract_domain=self.domain,
                spacy_enabled=self.spacy_enabled
            )

    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        force_update: bool = False
    ) -> EnhancedGraphBuildResult:
        """Process multiple documents with enhanced extraction.

        Args:
            documents: List of documents with 'id', 'text', and optional 'metadata'
            force_update: Whether to force update even if unchanged

        Returns:
            Aggregated EnhancedGraphBuildResult
        """
        start_time = time.time()

        total_entities = 0
        total_relations = 0
        total_langextract_entities = 0
        total_langextract_relations = 0
        total_spacy_entities = 0
        successful_docs = 0
        failed_docs = 0

        langextract_metadata = {}

        for doc in documents:
            try:
                result = await self.process_document(
                    doc_id=doc['id'],
                    text=doc['text'],
                    metadata=doc.get('metadata'),
                    force_update=force_update
                )

                if result.success:
                    successful_docs += 1
                    total_entities += result.entities_created
                    total_relations += result.relations_created
                    total_langextract_entities += result.langextract_entities_extracted
                    total_langextract_relations += result.langextract_relations_created
                    total_spacy_entities += result.spacy_entities_extracted

                    # Aggregate metadata
                    for key, value in result.langextract_metadata.items():
                        if key in langextract_metadata:
                            if isinstance(value, (int, float)):
                                langextract_metadata[key] += value
                            elif isinstance(value, list):
                                langextract_metadata[key].extend(value)
                        else:
                            langextract_metadata[key] = value
                else:
                    failed_docs += 1

            except Exception as e:
                self.logger.error(f"Failed to process document {doc.get('id', 'unknown')}: {e}")
                failed_docs += 1

        processing_time = time.time() - start_time

        return EnhancedGraphBuildResult(
            success=failed_docs == 0,
            entities_created=total_entities,
            relations_created=total_relations,
            processing_time=processing_time,

            # LangExtract results
            langextract_enabled=True,
            langextract_entities_extracted=total_langextract_entities,
            langextract_relations_created=total_langextract_relations,
            langextract_domain=self.domain,
            langextract_metadata=langextract_metadata,

            # SpaCy results
            spacy_enabled=self.spacy_enabled,
            spacy_entities_extracted=total_spacy_entities,

            # Additional stats
            metadata={
                "documents_processed": len(documents),
                "successful_documents": successful_docs,
                "failed_documents": failed_docs
            }
        )
