"""OpenIE-based relation extractor that integrates all OpenIE components."""

import asyncio
from typing import List, Dict, Any, Optional, NamedTuple
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..models import Entity, Relation, Graph
from ..services.openie_service import OpenIEService, OpenIETriplet
from ..processors.sentence_processor import SentenceProcessor, ProcessedSentence
from ..processors.triplet_processor import TripletProcessor, ValidatedTriplet
from ..normalizers.entity_linker import EntityLinker, EntityMatch
from ..normalizers.predicate_normalizer import PredicateNormalizer, NormalizedPredicate
from ..normalizers.confidence_manager import ConfidenceManager

logger = structlog.get_logger(__name__)


class OpenIEExtractionResult(NamedTuple):
    """Result of OpenIE extraction with all processed components."""
    relations: List[Relation]
    triplets: List[ValidatedTriplet]
    entity_matches: List[EntityMatch]
    normalized_predicates: List[NormalizedPredicate]
    processed_sentences: List[ProcessedSentence]
    metadata: Dict[str, Any]


class OpenIEExtractor:
    """Main OpenIE extractor that integrates all OpenIE components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the OpenIE extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = get_settings()
        
        # Initialize components
        self.openie_service = OpenIEService(config)
        self.sentence_processor = SentenceProcessor(config)
        self.triplet_processor = TripletProcessor(config)
        self.entity_linker = EntityLinker(config)
        self.predicate_normalizer = PredicateNormalizer(config)
        self.confidence_manager = ConfidenceManager(config)
        
        # Configuration
        self.enabled = self.settings.openie_enabled
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.enable_entity_linking = self.settings.openie_enable_entity_linking
        self.enable_predicate_normalization = self.settings.openie_enable_predicate_normalization
        
        logger.info(
            "OpenIE extractor initialized",
            enabled=self.enabled,
            entity_linking=self.enable_entity_linking,
            predicate_normalization=self.enable_predicate_normalization
        )
    
    async def extract_relations(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Relation]:
        """Extract relations using OpenIE pipeline.
        
        Args:
            text: Input text to process
            entities: Optional list of known entities from spaCy NER
            source_doc_id: Optional source document ID
            **kwargs: Additional arguments
            
        Returns:
            List of extracted relations
            
        Raises:
            ProcessingError: If extraction fails
        """
        if not self.enabled:
            logger.debug("OpenIE extractor disabled, returning empty relations")
            return []
        
        if not text or not text.strip():
            return []
        
        try:
            # Perform full extraction
            result = await self.extract_full(text, entities, source_doc_id, **kwargs)
            return result.relations
            
        except Exception as e:
            logger.error(
                "OpenIE relation extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"OpenIE relation extraction failed: {e}")
    
    async def extract_full(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> OpenIEExtractionResult:
        """Perform full OpenIE extraction with all components.
        
        Args:
            text: Input text to process
            entities: Optional list of known entities from spaCy NER
            source_doc_id: Optional source document ID
            **kwargs: Additional arguments
            
        Returns:
            Complete extraction result with all components
            
        Raises:
            ProcessingError: If extraction fails
        """
        if not self.enabled:
            logger.debug("OpenIE extractor disabled, returning empty result")
            return OpenIEExtractionResult(
                relations=[],
                triplets=[],
                entity_matches=[],
                normalized_predicates=[],
                processed_sentences=[],
                metadata={"enabled": False}
            )
        
        if not text or not text.strip():
            return OpenIEExtractionResult(
                relations=[],
                triplets=[],
                entity_matches=[],
                normalized_predicates=[],
                processed_sentences=[],
                metadata={"empty_input": True}
            )
        
        try:
            logger.debug(
                "Starting full OpenIE extraction",
                text_length=len(text),
                entity_count=len(entities) if entities else 0,
                source_doc_id=source_doc_id
            )
            
            # Step 1: Process sentences
            processed_sentences = await self.sentence_processor.process_text(text, source_doc_id)
            
            # Step 2: Extract raw triplets
            raw_triplets = await self.openie_service.extract_triplets(text, source_doc_id)
            
            # Step 3: Process and validate triplets
            validated_triplets = await self.triplet_processor.process_triplets(
                raw_triplets, processed_sentences, source_doc_id
            )
            
            # Step 4: Link entities (if enabled and entities provided)
            entity_matches = []
            if self.enable_entity_linking and entities:
                entity_matches = await self.entity_linker.link_triplet_entities(
                    validated_triplets, entities, source_doc_id
                )
            
            # Step 5: Normalize predicates (if enabled)
            normalized_predicates = []
            if self.enable_predicate_normalization:
                predicates = [t.predicate for t in validated_triplets]
                normalized_predicates = await self.predicate_normalizer.normalize_predicates(
                    predicates, source_doc_id
                )
            
            # Step 6: Convert to relations
            relations = await self._convert_to_relations(
                validated_triplets, entity_matches, normalized_predicates, source_doc_id
            )
            
            # Step 7: Apply confidence filtering
            filtered_relations = await self.confidence_manager.filter_relations(
                relations, self.min_confidence
            )
            
            metadata = {
                "processed_sentences": len(processed_sentences),
                "raw_triplets": len(raw_triplets),
                "validated_triplets": len(validated_triplets),
                "entity_matches": len(entity_matches),
                "normalized_predicates": len(normalized_predicates),
                "final_relations": len(filtered_relations),
                "confidence_threshold": self.min_confidence,
                "entity_linking_enabled": self.enable_entity_linking,
                "predicate_normalization_enabled": self.enable_predicate_normalization
            }
            
            logger.info(
                "OpenIE extraction completed",
                **metadata,
                source_doc_id=source_doc_id
            )
            
            return OpenIEExtractionResult(
                relations=filtered_relations,
                triplets=validated_triplets,
                entity_matches=entity_matches,
                normalized_predicates=normalized_predicates,
                processed_sentences=processed_sentences,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(
                "Full OpenIE extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Full OpenIE extraction failed: {e}")
    
    async def _convert_to_relations(
        self,
        triplets: List[ValidatedTriplet],
        entity_matches: List[EntityMatch],
        normalized_predicates: List[NormalizedPredicate],
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Convert validated triplets to Relation objects.
        
        Args:
            triplets: List of validated triplets
            entity_matches: List of entity matches
            normalized_predicates: List of normalized predicates
            source_doc_id: Optional source document ID
            
        Returns:
            List of Relation objects
        """
        relations = []
        
        # Create lookup dictionaries for efficiency
        entity_match_lookup = {match.openie_entity: match for match in entity_matches}
        predicate_lookup = {pred.original: pred for pred in normalized_predicates}
        
        for triplet in triplets:
            try:
                # Get entity matches
                subject_match = entity_match_lookup.get(triplet.subject)
                object_match = entity_match_lookup.get(triplet.object)
                
                # Get normalized predicate
                predicate_norm = predicate_lookup.get(triplet.predicate)
                
                # Create relation
                relation = Relation(
                    subject=subject_match.spacy_entity.canonical_name if subject_match else triplet.subject,
                    predicate=predicate_norm.canonical_form if predicate_norm else triplet.predicate,
                    object=object_match.spacy_entity.canonical_name if object_match else triplet.object,
                    confidence=triplet.confidence,
                    source_doc_id=source_doc_id,
                    metadata={
                        "extraction_method": "openie",
                        "sentence": triplet.sentence,
                        "sentence_id": triplet.sentence_id,
                        "validation_score": triplet.validation_score,
                        "validation_flags": list(triplet.validation_flags),
                        "subject_linked": subject_match is not None,
                        "object_linked": object_match is not None,
                        "predicate_normalized": predicate_norm is not None,
                        "original_subject": triplet.subject,
                        "original_predicate": triplet.predicate,
                        "original_object": triplet.object
                    }
                )
                
                relations.append(relation)
                
            except Exception as e:
                logger.warning(
                    "Failed to convert triplet to relation",
                    triplet=f"{triplet.subject} | {triplet.predicate} | {triplet.object}",
                    error=str(e)
                )
        
        return relations
    
    async def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the OpenIE extractor components.
        
        Returns:
            Dictionary with component statistics
        """
        stats = {
            "enabled": self.enabled,
            "configuration": {
                "min_confidence": self.min_confidence,
                "entity_linking_enabled": self.enable_entity_linking,
                "predicate_normalization_enabled": self.enable_predicate_normalization
            }
        }
        
        # Get component stats if available
        try:
            if hasattr(self.openie_service, 'get_stats'):
                stats["openie_service"] = await self.openie_service.get_stats()
            if hasattr(self.entity_linker, 'get_stats'):
                stats["entity_linker"] = await self.entity_linker.get_stats()
            if hasattr(self.predicate_normalizer, 'get_stats'):
                stats["predicate_normalizer"] = await self.predicate_normalizer.get_stats()
        except Exception as e:
            logger.warning("Failed to get component stats", error=str(e))
        
        return stats
