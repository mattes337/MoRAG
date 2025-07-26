"""Citation and source tracking for fact-based responses."""

import asyncio
import time
from typing import List, Dict, Any, Optional, NamedTuple, Set
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .fact_scorer import ScoredFact
from .graph_fact_extractor import ExtractedFact

logger = structlog.get_logger(__name__)


class CitationFormat(Enum):
    """Supported citation formats."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    SIMPLE = "simple"
    JSON = "json"


@dataclass
class SourceReference:
    """Detailed source reference information."""
    document_id: str
    document_title: Optional[str]
    chunk_id: Optional[str]
    page_number: Optional[int]
    timestamp: Optional[str]
    confidence: float
    url: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    publisher: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class CitedFact:
    """Fact with complete citation information."""
    fact: ExtractedFact
    score: float
    sources: List[SourceReference]
    citation_text: str
    citation_format: CitationFormat
    verification_status: str
    metadata: Dict[str, Any]


class CitationManager:
    """Comprehensive citation tracking and management system."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the citation manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = get_settings()
        
        # Citation parameters
        self.default_format = CitationFormat(
            self.config.get('default_format', 'simple')
        )
        self.max_sources_per_fact = self.config.get('max_sources_per_fact', 5)
        self.enable_verification = self.config.get('enable_verification', True)
        self.include_timestamps = self.config.get('include_timestamps', True)
        
        # Source tracking
        self.track_document_metadata = self.config.get('track_document_metadata', True)
        self.track_extraction_path = self.config.get('track_extraction_path', True)
        self.track_confidence_scores = self.config.get('track_confidence_scores', True)
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 20)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Initialize components
        self._source_cache = {} if self.enable_caching else None
        self._document_metadata_cache = {}
        
        logger.info(
            "Citation manager initialized",
            default_format=self.default_format.value,
            max_sources_per_fact=self.max_sources_per_fact,
            enable_verification=self.enable_verification,
            track_document_metadata=self.track_document_metadata
        )
    
    async def track_citations(
        self,
        scored_facts: List[ScoredFact],
        citation_format: Optional[CitationFormat] = None
    ) -> List[CitedFact]:
        """Track citations for scored facts.
        
        Args:
            scored_facts: List of scored facts to add citations to
            citation_format: Optional citation format override
            
        Returns:
            List of facts with complete citation information
        """
        if not scored_facts:
            return []
        
        format_to_use = citation_format or self.default_format
        
        try:
            logger.info(
                "Starting citation tracking",
                num_facts=len(scored_facts),
                citation_format=format_to_use.value
            )
            
            cited_facts = []
            
            # Process facts in batches
            for i in range(0, len(scored_facts), self.batch_size):
                batch = scored_facts[i:i + self.batch_size]
                batch_citations = await self._process_citation_batch(
                    batch, format_to_use
                )
                cited_facts.extend(batch_citations)
            
            logger.info(
                "Citation tracking completed",
                total_facts=len(scored_facts),
                cited_facts=len(cited_facts),
                format=format_to_use.value
            )
            
            return cited_facts
            
        except Exception as e:
            logger.error(f"Citation tracking failed: {e}")
            # Return facts with minimal citation info as fallback
            return [
                CitedFact(
                    fact=sf.fact,
                    score=sf.overall_score,
                    sources=[],
                    citation_text="Source information unavailable",
                    citation_format=format_to_use,
                    verification_status="unverified",
                    metadata={'citation_error': str(e)}
                )
                for sf in scored_facts
            ]
    
    async def _process_citation_batch(
        self,
        scored_facts: List[ScoredFact],
        citation_format: CitationFormat
    ) -> List[CitedFact]:
        """Process a batch of facts for citation tracking."""
        cited_facts = []
        
        for scored_fact in scored_facts:
            try:
                cited_fact = await self._create_cited_fact(
                    scored_fact, citation_format
                )
                cited_facts.append(cited_fact)
            except Exception as e:
                logger.warning(
                    f"Failed to create citation for fact {scored_fact.fact.fact_id}: {e}"
                )
                # Add fact with minimal citation as fallback
                fallback_citation = CitedFact(
                    fact=scored_fact.fact,
                    score=scored_fact.overall_score,
                    sources=[],
                    citation_text="Citation unavailable",
                    citation_format=citation_format,
                    verification_status="error",
                    metadata={'citation_error': str(e)}
                )
                cited_facts.append(fallback_citation)
        
        return cited_facts
    
    async def _create_cited_fact(
        self,
        scored_fact: ScoredFact,
        citation_format: CitationFormat
    ) -> CitedFact:
        """Create a cited fact with complete source information."""
        fact = scored_fact.fact
        
        # Extract source references
        sources = await self._extract_source_references(fact)
        
        # Generate citation text
        citation_text = self._generate_citation_text(sources, citation_format)
        
        # Verify sources if enabled
        verification_status = "unverified"
        if self.enable_verification:
            verification_status = await self._verify_sources(sources)
        
        return CitedFact(
            fact=fact,
            score=scored_fact.overall_score,
            sources=sources,
            citation_text=citation_text,
            citation_format=citation_format,
            verification_status=verification_status,
            metadata={
                'citation_generated_at': datetime.now().isoformat(),
                'num_sources': len(sources),
                'scoring_dimensions': scored_fact.scoring_dimensions.__dict__
            }
        )
    
    async def _extract_source_references(
        self,
        fact: ExtractedFact
    ) -> List[SourceReference]:
        """Extract detailed source references from a fact."""
        sources = []
        
        # Process document sources
        for doc_id in fact.source_documents[:self.max_sources_per_fact]:
            try:
                source_ref = await self._create_source_reference_from_document(
                    doc_id, fact
                )
                if source_ref:
                    sources.append(source_ref)
            except Exception as e:
                logger.warning(f"Failed to create source reference for {doc_id}: {e}")
        
        # If no document sources, create references from entities
        if not sources and fact.source_entities:
            for entity_id in fact.source_entities[:self.max_sources_per_fact]:
                try:
                    source_ref = await self._create_source_reference_from_entity(
                        entity_id, fact
                    )
                    if source_ref:
                        sources.append(source_ref)
                except Exception as e:
                    logger.warning(f"Failed to create source reference for entity {entity_id}: {e}")
        
        # If still no sources, create a generic reference
        if not sources:
            sources.append(self._create_generic_source_reference(fact))
        
        return sources
    
    async def _create_source_reference_from_document(
        self,
        document_id: str,
        fact: ExtractedFact
    ) -> Optional[SourceReference]:
        """Create enhanced source reference from document ID."""
        # Check cache first
        if self._source_cache and document_id in self._source_cache:
            cached_ref = self._source_cache[document_id]

            # Extract chunk information from fact context
            chunk_id = self._extract_chunk_id_from_fact(fact, document_id)
            page_number = self._extract_page_number_from_fact(fact, cached_ref)
            timestamp = self._extract_timestamp_from_fact(fact, cached_ref)

            return SourceReference(
                document_id=document_id,
                document_title=cached_ref.get('title', f"Document {document_id}"),
                chunk_id=chunk_id,
                page_number=page_number,
                timestamp=timestamp,
                confidence=self._calculate_source_confidence(fact, cached_ref),
                url=cached_ref.get('url'),
                author=cached_ref.get('author'),
                publication_date=cached_ref.get('publication_date'),
                publisher=cached_ref.get('publisher'),
                metadata=cached_ref.get('metadata', {})
            )
        
        # In a real implementation, this would query the document storage
        # For now, create a basic reference
        return SourceReference(
            document_id=document_id,
            document_title=f"Document {document_id}",
            chunk_id=None,
            page_number=None,
            timestamp=None,
            confidence=fact.confidence,
            metadata={'source_type': 'document'}
        )
    
    async def _create_source_reference_from_entity(
        self,
        entity_id: str,
        fact: ExtractedFact
    ) -> Optional[SourceReference]:
        """Create source reference from entity ID."""
        return SourceReference(
            document_id=f"entity_{entity_id}",
            document_title=f"Knowledge Graph Entity {entity_id}",
            chunk_id=None,
            page_number=None,
            timestamp=None,
            confidence=fact.confidence,
            metadata={'source_type': 'entity', 'entity_id': entity_id}
        )
    
    def _create_generic_source_reference(self, fact: ExtractedFact) -> SourceReference:
        """Create a generic source reference when no specific sources are available."""
        return SourceReference(
            document_id="unknown",
            document_title="Knowledge Graph",
            chunk_id=None,
            page_number=None,
            timestamp=None,
            confidence=fact.confidence,
            metadata={'source_type': 'generic', 'fact_type': fact.fact_type.value}
        )
    
    def _generate_citation_text(
        self,
        sources: List[SourceReference],
        citation_format: CitationFormat
    ) -> str:
        """Generate citation text in the specified format."""
        if not sources:
            return "No sources available"
        
        if citation_format == CitationFormat.SIMPLE:
            return self._generate_simple_citation(sources)
        elif citation_format == CitationFormat.JSON:
            return self._generate_json_citation(sources)
        elif citation_format == CitationFormat.APA:
            return self._generate_apa_citation(sources)
        elif citation_format == CitationFormat.MLA:
            return self._generate_mla_citation(sources)
        elif citation_format == CitationFormat.CHICAGO:
            return self._generate_chicago_citation(sources)
        elif citation_format == CitationFormat.IEEE:
            return self._generate_ieee_citation(sources)
        else:
            return self._generate_simple_citation(sources)
    
    def _generate_simple_citation(self, sources: List[SourceReference]) -> str:
        """Generate simple citation format."""
        citations = []
        for i, source in enumerate(sources, 1):
            citation_parts = []
            
            if source.document_title:
                citation_parts.append(source.document_title)
            
            if source.page_number:
                citation_parts.append(f"p. {source.page_number}")
            
            if source.timestamp:
                citation_parts.append(f"at {source.timestamp}")
            
            if citation_parts:
                citations.append(f"[{i}] {', '.join(citation_parts)}")
            else:
                citations.append(f"[{i}] {source.document_id}")
        
        return "; ".join(citations)
    
    def _generate_json_citation(self, sources: List[SourceReference]) -> str:
        """Generate JSON citation format."""
        import json
        
        citation_data = []
        for source in sources:
            source_dict = {
                'document_id': source.document_id,
                'title': source.document_title,
                'confidence': source.confidence
            }
            
            if source.chunk_id:
                source_dict['chunk_id'] = source.chunk_id
            if source.page_number:
                source_dict['page'] = source.page_number
            if source.timestamp:
                source_dict['timestamp'] = source.timestamp
            if source.url:
                source_dict['url'] = source.url
            if source.author:
                source_dict['author'] = source.author
            
            citation_data.append(source_dict)
        
        return json.dumps(citation_data, indent=2)
    
    def _generate_apa_citation(self, sources: List[SourceReference]) -> str:
        """Generate APA citation format."""
        citations = []
        for source in sources:
            citation_parts = []
            
            if source.author:
                citation_parts.append(source.author)
            
            if source.publication_date:
                citation_parts.append(f"({source.publication_date})")
            
            if source.document_title:
                citation_parts.append(f"{source.document_title}")
            
            if source.publisher:
                citation_parts.append(source.publisher)
            
            citations.append(". ".join(citation_parts) + ".")
        
        return " ".join(citations)
    
    def _generate_mla_citation(self, sources: List[SourceReference]) -> str:
        """Generate MLA citation format."""
        citations = []
        for source in sources:
            citation_parts = []
            
            if source.author:
                citation_parts.append(source.author)
            
            if source.document_title:
                citation_parts.append(f'"{source.document_title}"')
            
            if source.publisher:
                citation_parts.append(source.publisher)
            
            if source.publication_date:
                citation_parts.append(source.publication_date)
            
            citations.append(", ".join(citation_parts) + ".")
        
        return " ".join(citations)
    
    def _generate_chicago_citation(self, sources: List[SourceReference]) -> str:
        """Generate Chicago citation format."""
        # Simplified Chicago format
        return self._generate_apa_citation(sources)
    
    def _generate_ieee_citation(self, sources: List[SourceReference]) -> str:
        """Generate IEEE citation format."""
        citations = []
        for i, source in enumerate(sources, 1):
            citation_parts = []
            
            if source.author:
                citation_parts.append(source.author)
            
            if source.document_title:
                citation_parts.append(f'"{source.document_title}"')
            
            if source.publication_date:
                citation_parts.append(source.publication_date)
            
            citations.append(f"[{i}] " + ", ".join(citation_parts) + ".")
        
        return " ".join(citations)
    
    async def _verify_sources(self, sources: List[SourceReference]) -> str:
        """Verify the availability and accuracy of sources."""
        if not sources:
            return "no_sources"
        
        verified_count = 0
        for source in sources:
            # In a real implementation, this would check document availability
            # For now, assume sources with document_id are verified
            if source.document_id and source.document_id != "unknown":
                verified_count += 1
        
        verification_ratio = verified_count / len(sources)
        
        if verification_ratio >= 0.8:
            return "verified"
        elif verification_ratio >= 0.5:
            return "partially_verified"

    def _extract_chunk_id_from_fact(self, fact: ExtractedFact, document_id: str) -> Optional[str]:
        """Extract chunk ID from fact context."""
        if hasattr(fact, 'context') and fact.context:
            # Look for chunk information in context
            chunk_info = fact.context.get('chunk_id')
            if chunk_info:
                return str(chunk_info)

        if hasattr(fact, 'metadata') and fact.metadata:
            # Look for chunk information in metadata
            chunk_info = fact.metadata.get('chunk_id')
            if chunk_info:
                return str(chunk_info)

        # Generate chunk ID from extraction path if available
        if fact.extraction_path and len(fact.extraction_path) > 0:
            return f"chunk_{document_id}_{hash(''.join(fact.extraction_path)) % 10000}"

        return None

    def _extract_page_number_from_fact(self, fact: ExtractedFact, cached_ref: Dict[str, Any]) -> Optional[int]:
        """Extract page number from fact context or cached reference."""
        # Check fact context first
        if hasattr(fact, 'context') and fact.context:
            page_info = fact.context.get('page_number') or fact.context.get('page')
            if page_info and isinstance(page_info, (int, str)):
                try:
                    return int(page_info)
                except ValueError:
                    pass

        # Check fact metadata
        if hasattr(fact, 'metadata') and fact.metadata:
            page_info = fact.metadata.get('page_number') or fact.metadata.get('page')
            if page_info and isinstance(page_info, (int, str)):
                try:
                    return int(page_info)
                except ValueError:
                    pass

        # Fall back to cached reference
        page_info = cached_ref.get('page')
        if page_info and isinstance(page_info, (int, str)):
            try:
                return int(page_info)
            except ValueError:
                pass

        return None

    def _extract_timestamp_from_fact(self, fact: ExtractedFact, cached_ref: Dict[str, Any]) -> Optional[str]:
        """Extract timestamp from fact context or cached reference."""
        # Check fact metadata for extraction timestamp
        if hasattr(fact, 'metadata') and fact.metadata:
            extraction_time = fact.metadata.get('extraction_timestamp')
            if extraction_time:
                try:
                    dt = datetime.fromtimestamp(extraction_time)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, OSError):
                    pass

        # Check fact context for temporal information
        if hasattr(fact, 'context') and fact.context:
            timestamp_info = fact.context.get('timestamp')
            if timestamp_info:
                return str(timestamp_info)

        # Fall back to cached reference
        return cached_ref.get('timestamp')

    def _calculate_source_confidence(self, fact: ExtractedFact, cached_ref: Dict[str, Any]) -> float:
        """Calculate confidence score for the source reference."""
        base_confidence = fact.confidence

        # Adjust based on source metadata quality
        metadata_quality = 0.0
        if cached_ref.get('title'):
            metadata_quality += 0.1
        if cached_ref.get('author'):
            metadata_quality += 0.1
        if cached_ref.get('publication_date'):
            metadata_quality += 0.1
        if cached_ref.get('url'):
            metadata_quality += 0.05

        # Adjust based on fact context completeness
        context_quality = 0.0
        if hasattr(fact, 'context') and fact.context:
            if fact.context.get('semantic_context'):
                context_quality += 0.05
            if fact.context.get('hop_position') is not None:
                context_quality += 0.05

        final_confidence = base_confidence + metadata_quality + context_quality
        return min(1.0, final_confidence)
        else:
            return "unverified"
