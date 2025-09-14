"""Unified extraction pipeline integrating all enhanced components."""

import structlog
import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from ..models import Entity, Relation, Document
from .enhanced_entity_extractor import EnhancedEntityExtractor
from .enhanced_relation_extractor import EnhancedRelationExtractor
from .systematic_deduplicator import SystematicDeduplicator, DeduplicationResult
from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor

logger = structlog.get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of unified extraction pipeline processing."""
    entities: List[Entity]
    relations: List[Relation]
    processing_metadata: Dict[str, Any]
    deduplication_result: Optional[DeduplicationResult] = None
    processing_time: float = 0.0
    chunks_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations],
            'processing_metadata': self.processing_metadata,
            'deduplication_stats': {
                'original_entities': self.deduplication_result.original_count if self.deduplication_result else len(self.entities),
                'deduplicated_entities': self.deduplication_result.deduplicated_count if self.deduplication_result else len(self.entities),
                'merges_performed': self.deduplication_result.merges_performed if self.deduplication_result else 0
            },
            'performance_stats': {
                'processing_time': self.processing_time,
                'chunks_processed': self.chunks_processed,
                'entities_per_second': len(self.entities) / self.processing_time if self.processing_time > 0 else 0,
                'relations_per_second': len(self.relations) / self.processing_time if self.processing_time > 0 else 0
            }
        }


@dataclass
class PipelineConfig:
    """Configuration for unified extraction pipeline."""
    # Entity extraction settings
    entity_max_rounds: int = 3
    entity_target_confidence: float = 0.85
    entity_confidence_threshold: float = 0.4
    enable_entity_gleaning: bool = True
    
    # Relation extraction settings
    relation_max_rounds: int = 2
    relation_confidence_threshold: float = 0.4
    enable_relation_validation: bool = True
    
    # Deduplication settings
    enable_deduplication: bool = True
    similarity_threshold: float = 0.7
    merge_confidence_threshold: float = 0.5
    enable_llm_validation: bool = True
    
    # Performance settings
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_workers: int = 4
    enable_parallel_processing: bool = True
    
    # Domain and language settings
    domain: str = "general"
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'entity_extraction': {
                'max_rounds': self.entity_max_rounds,
                'target_confidence': self.entity_target_confidence,
                'confidence_threshold': self.entity_confidence_threshold,
                'enable_gleaning': self.enable_entity_gleaning
            },
            'relation_extraction': {
                'max_rounds': self.relation_max_rounds,
                'confidence_threshold': self.relation_confidence_threshold,
                'enable_validation': self.enable_relation_validation
            },
            'deduplication': {
                'enable': self.enable_deduplication,
                'similarity_threshold': self.similarity_threshold,
                'merge_confidence_threshold': self.merge_confidence_threshold,
                'enable_llm_validation': self.enable_llm_validation
            },
            'performance': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'max_workers': self.max_workers,
                'enable_parallel': self.enable_parallel_processing
            },
            'general': {
                'domain': self.domain,
                'language': self.language
            }
        }


class UnifiedExtractionPipeline:
    """Unified extraction pipeline integrating all enhanced components."""
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        api_key: Optional[str] = None,
        model_id: str = "gemini-2.0-flash"
    ):
        """Initialize unified extraction pipeline.
        
        Args:
            config: Pipeline configuration
            api_key: API key for LLM services
            model_id: Model ID for LLM services
        """
        self.config = config or PipelineConfig()
        self.api_key = api_key or self._get_api_key()
        self.model_id = model_id
        
        self.logger = logger.bind(component="unified_extraction_pipeline")
        
        # Initialize components
        self._initialize_components()
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        import os
        return os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
    
    def _initialize_components(self):
        """Initialize extraction components."""
        # Base extractors
        base_entity_extractor = EntityExtractor(
            min_confidence=self.config.entity_confidence_threshold,
            chunk_size=self.config.chunk_size,
            dynamic_types=True,
            language=self.config.language,
            model_id=self.model_id,
            api_key=self.api_key,
            domain=self.config.domain
        )
        
        base_relation_extractor = RelationExtractor(
            min_confidence=self.config.relation_confidence_threshold,
            chunk_size=self.config.chunk_size,
            dynamic_types=True,
            language=self.config.language,
            model_id=self.model_id,
            api_key=self.api_key,
            domain=self.config.domain
        )
        
        # Enhanced extractors
        self.entity_extractor = EnhancedEntityExtractor(
            base_extractor=base_entity_extractor,
            max_rounds=self.config.entity_max_rounds,
            target_confidence=self.config.entity_target_confidence,
            confidence_threshold=self.config.entity_confidence_threshold,
            enable_gleaning=self.config.enable_entity_gleaning
        )
        
        self.relation_extractor = EnhancedRelationExtractor(
            base_extractor=base_relation_extractor,
            max_rounds=self.config.relation_max_rounds,
            confidence_threshold=self.config.relation_confidence_threshold,
            enable_validation=self.config.enable_relation_validation
        )
        
        # Deduplicator
        if self.config.enable_deduplication:
            self.deduplicator = SystematicDeduplicator(
                similarity_threshold=self.config.similarity_threshold,
                merge_confidence_threshold=self.config.merge_confidence_threshold,
                enable_llm_validation=self.config.enable_llm_validation
            )
        else:
            self.deduplicator = None
        
        self.logger.info("Unified extraction pipeline initialized", config=self.config.to_dict())
    
    async def process_document(
        self,
        document: Document,
        chunk_size: Optional[int] = None
    ) -> ProcessingResult:
        """Process document with enhanced extraction pipeline.
        
        Args:
            document: Document to process
            chunk_size: Override default chunk size
            
        Returns:
            ProcessingResult with extracted entities and relations
        """
        start_time = time.time()
        
        self.logger.info(
            "Starting document processing",
            document_id=document.id if hasattr(document, 'id') else 'unknown',
            content_length=len(document.content) if hasattr(document, 'content') else 0
        )
        
        # Step 1: Chunk document
        chunks = await self._chunk_document(document, chunk_size or self.config.chunk_size)
        
        # Step 2: Extract entities and relations with gleaning
        entities_by_chunk, relations_by_chunk = await self._extract_from_chunks(chunks, document)
        
        # Step 3: Cross-chunk deduplication
        if self.config.enable_deduplication and self.deduplicator:
            deduplicated_entities, dedup_result = await self.deduplicator.deduplicate_across_chunks(
                entities_by_chunk
            )
            
            # Update relations based on entity merges
            deduplicated_relations = await self.deduplicator.deduplicate_relations(
                relations_by_chunk, dedup_result.merge_details[0].primary_entity.id if dedup_result.merge_details else {}
            )
        else:
            # Flatten without deduplication
            deduplicated_entities = {
                chunk_id: entities for chunk_id, entities in entities_by_chunk.items()
            }
            deduplicated_relations = relations_by_chunk
            dedup_result = None
        
        # Step 4: Flatten results
        all_entities = []
        all_relations = []
        
        for entities in deduplicated_entities.values():
            all_entities.extend(entities)
        
        for relations in deduplicated_relations.values():
            all_relations.extend(relations)
        
        processing_time = time.time() - start_time
        
        # Create processing metadata
        metadata = self._create_metadata(document, chunks, processing_time)
        
        result = ProcessingResult(
            entities=all_entities,
            relations=all_relations,
            processing_metadata=metadata,
            deduplication_result=dedup_result,
            processing_time=processing_time,
            chunks_processed=len(chunks)
        )
        
        self.logger.info(
            "Document processing completed",
            entities_extracted=len(all_entities),
            relations_extracted=len(all_relations),
            processing_time=f"{processing_time:.2f}s",
            chunks_processed=len(chunks)
        )
        
        return result
    
    async def _chunk_document(
        self,
        document: Document,
        chunk_size: int
    ) -> List[Dict[str, Any]]:
        """Chunk document into smaller pieces for processing."""
        # Simple chunking implementation
        content = document.content if hasattr(document, 'content') else str(document)
        chunks = []
        
        for i in range(0, len(content), chunk_size - self.config.chunk_overlap):
            chunk_content = content[i:i + chunk_size]
            
            if chunk_content.strip():
                chunks.append({
                    'id': f"chunk_{i // (chunk_size - self.config.chunk_overlap)}",
                    'content': chunk_content,
                    'start_offset': i,
                    'end_offset': min(i + chunk_size, len(content)),
                    'document_id': document.id if hasattr(document, 'id') else 'unknown'
                })
        
        return chunks
    
    async def _extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document: Document
    ) -> Tuple[Dict[str, List[Entity]], Dict[str, List[Relation]]]:
        """Extract entities and relations from document chunks."""
        entities_by_chunk = {}
        relations_by_chunk = {}
        
        if self.config.enable_parallel_processing and len(chunks) > 1:
            # Process chunks in parallel
            tasks = [
                self._process_single_chunk(chunk, document)
                for chunk in chunks
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Chunk processing failed: {result}", chunk_id=chunks[i]['id'])
                    entities_by_chunk[chunks[i]['id']] = []
                    relations_by_chunk[chunks[i]['id']] = []
                else:
                    chunk_entities, chunk_relations = result
                    entities_by_chunk[chunks[i]['id']] = chunk_entities
                    relations_by_chunk[chunks[i]['id']] = chunk_relations
        else:
            # Process chunks sequentially
            for chunk in chunks:
                try:
                    chunk_entities, chunk_relations = await self._process_single_chunk(chunk, document)
                    entities_by_chunk[chunk['id']] = chunk_entities
                    relations_by_chunk[chunk['id']] = chunk_relations
                except Exception as e:
                    self.logger.error(f"Chunk processing failed: {e}", chunk_id=chunk['id'])
                    entities_by_chunk[chunk['id']] = []
                    relations_by_chunk[chunk['id']] = []
        
        return entities_by_chunk, relations_by_chunk

    async def _process_single_chunk(
        self,
        chunk: Dict[str, Any],
        document: Document
    ) -> Tuple[List[Entity], List[Relation]]:
        """Process a single chunk to extract entities and relations."""
        chunk_content = chunk['content']
        chunk_id = chunk['id']
        document_id = chunk.get('document_id', 'unknown')

        # Extract entities with gleaning
        entities = await self.entity_extractor.extract_with_gleaning(
            text=chunk_content,
            source_doc_id=document_id
        )

        # Update entity metadata (if fields exist)
        for entity in entities:
            if hasattr(entity, 'chunk_id'):
                entity.chunk_id = chunk_id
            if hasattr(entity, 'start_offset'):
                entity.start_offset = chunk.get('start_offset', 0)
            if hasattr(entity, 'end_offset'):
                entity.end_offset = chunk.get('end_offset', 0)

        # Extract relations with gleaning
        relations = await self.relation_extractor.extract_with_gleaning(
            text=chunk_content,
            entities=entities,
            source_doc_id=document_id
        )

        # Update relation metadata (if fields exist)
        for relation in relations:
            if hasattr(relation, 'chunk_id'):
                relation.chunk_id = chunk_id
            if hasattr(relation, 'start_offset'):
                relation.start_offset = chunk.get('start_offset', 0)
            if hasattr(relation, 'end_offset'):
                relation.end_offset = chunk.get('end_offset', 0)

        self.logger.debug(
            "Chunk processed",
            chunk_id=chunk_id,
            entities_found=len(entities),
            relations_found=len(relations)
        )

        return entities, relations

    def _create_metadata(
        self,
        document: Document,
        chunks: List[Dict[str, Any]],
        processing_time: float
    ) -> Dict[str, Any]:
        """Create processing metadata."""
        return {
            'pipeline_version': '1.0.0',
            'config': self.config.to_dict(),
            'document_info': {
                'id': document.id if hasattr(document, 'id') else 'unknown',
                'content_length': len(document.content) if hasattr(document, 'content') else 0,
                'chunks_created': len(chunks)
            },
            'processing_stats': {
                'total_time': processing_time,
                'avg_time_per_chunk': processing_time / len(chunks) if chunks else 0,
                'chunks_processed': len(chunks)
            },
            'extraction_settings': {
                'entity_gleaning_enabled': self.config.enable_entity_gleaning,
                'relation_validation_enabled': self.config.enable_relation_validation,
                'deduplication_enabled': self.config.enable_deduplication,
                'parallel_processing_enabled': self.config.enable_parallel_processing
            }
        }

    async def process_text(
        self,
        text: str,
        source_id: Optional[str] = None
    ) -> ProcessingResult:
        """Process raw text with enhanced extraction pipeline.

        Args:
            text: Text content to process
            source_id: Optional source identifier

        Returns:
            ProcessingResult with extracted entities and relations
        """
        # Create a simple document object
        class SimpleDocument:
            def __init__(self, content: str, doc_id: str):
                self.content = content
                self.id = doc_id

        document = SimpleDocument(text, source_id or 'text_input')
        return await self.process_document(document)

    async def process_multiple_documents(
        self,
        documents: List[Document],
        enable_cross_document_deduplication: bool = True
    ) -> List[ProcessingResult]:
        """Process multiple documents with optional cross-document deduplication.

        Args:
            documents: List of documents to process
            enable_cross_document_deduplication: Whether to deduplicate across documents

        Returns:
            List of ProcessingResults
        """
        start_time = time.time()

        self.logger.info(
            "Starting multi-document processing",
            document_count=len(documents),
            cross_dedup_enabled=enable_cross_document_deduplication
        )

        # Process each document individually
        if self.config.enable_parallel_processing and len(documents) > 1:
            tasks = [self.process_document(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"Document processing failed: {result}",
                        document_id=documents[i].id if hasattr(documents[i], 'id') else i
                    )
                else:
                    valid_results.append(result)
        else:
            valid_results = []
            for doc in documents:
                try:
                    result = await self.process_document(doc)
                    valid_results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Document processing failed: {e}",
                        document_id=doc.id if hasattr(doc, 'id') else 'unknown'
                    )

        # Cross-document deduplication
        if enable_cross_document_deduplication and self.deduplicator and len(valid_results) > 1:
            valid_results = await self._cross_document_deduplication(valid_results)

        processing_time = time.time() - start_time

        self.logger.info(
            "Multi-document processing completed",
            documents_processed=len(valid_results),
            total_time=f"{processing_time:.2f}s"
        )

        return valid_results

    async def _cross_document_deduplication(
        self,
        results: List[ProcessingResult]
    ) -> List[ProcessingResult]:
        """Perform cross-document entity deduplication."""
        # Collect all entities across documents
        all_entities_by_doc = {}
        all_relations_by_doc = {}

        for i, result in enumerate(results):
            doc_key = f"doc_{i}"
            all_entities_by_doc[doc_key] = result.entities
            all_relations_by_doc[doc_key] = result.relations

        # Perform deduplication
        deduplicated_entities, dedup_result = await self.deduplicator.deduplicate_across_chunks(
            all_entities_by_doc
        )

        # Update relations based on entity merges
        merge_mapping = {}
        if dedup_result.merge_details:
            for merge in dedup_result.merge_details:
                for duplicate in merge.duplicate_entities:
                    merge_mapping[duplicate.id] = merge.primary_entity.id

        deduplicated_relations = await self.deduplicator.deduplicate_relations(
            all_relations_by_doc, merge_mapping
        )

        # Rebuild results with deduplicated entities and relations
        updated_results = []
        for i, result in enumerate(results):
            doc_key = f"doc_{i}"

            # Update result with deduplicated entities and relations
            result.entities = deduplicated_entities.get(doc_key, [])
            result.relations = deduplicated_relations.get(doc_key, [])
            result.deduplication_result = dedup_result

            # Update metadata
            result.processing_metadata['cross_document_deduplication'] = {
                'enabled': True,
                'original_entities': len(all_entities_by_doc[doc_key]),
                'deduplicated_entities': len(result.entities),
                'merges_performed': dedup_result.merges_performed
            }

            updated_results.append(result)

        return updated_results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and configuration."""
        return {
            'config': self.config.to_dict(),
            'components': {
                'entity_extractor': {
                    'type': 'EnhancedEntityExtractor',
                    'gleaning_enabled': self.config.enable_entity_gleaning,
                    'max_rounds': self.config.entity_max_rounds
                },
                'relation_extractor': {
                    'type': 'EnhancedRelationExtractor',
                    'validation_enabled': self.config.enable_relation_validation,
                    'max_rounds': self.config.relation_max_rounds
                },
                'deduplicator': {
                    'type': 'SystematicDeduplicator' if self.deduplicator else None,
                    'enabled': self.config.enable_deduplication,
                    'llm_validation': self.config.enable_llm_validation
                }
            },
            'performance': {
                'parallel_processing': self.config.enable_parallel_processing,
                'max_workers': self.config.max_workers,
                'chunk_size': self.config.chunk_size
            }
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
