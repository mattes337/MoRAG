"""Citation integration system for seamlessly embedding citations in responses."""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from morag_core.config import get_settings

from .citation_manager import CitationFormat, CitedFact, SourceReference
from .response_generator import GeneratedResponse

logger = structlog.get_logger(__name__)


class CitationStyle(Enum):
    """Citation integration styles."""

    INLINE = "inline"  # Citations within sentences
    FOOTNOTE = "footnote"  # Numbered footnotes
    ENDNOTE = "endnote"  # References at end
    PARENTHETICAL = "parenthetical"  # (Author, Year) style
    SUPERSCRIPT = "superscript"  # Superscript numbers


@dataclass
class CitationOptions:
    """Options for citation integration."""

    style: CitationStyle = CitationStyle.INLINE
    format: CitationFormat = CitationFormat.STRUCTURED
    include_timestamps: bool = True
    include_page_numbers: bool = True
    include_confidence: bool = False
    max_citations_per_sentence: int = 3
    abbreviate_repeated_sources: bool = True
    verify_citations: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class CitedResponse:
    """Response with integrated citations."""

    content: str
    citations_list: List[str]
    citation_map: Dict[str, SourceReference]
    verification_status: str
    integration_time: float
    citation_count: int
    metadata: Dict[str, Any]


@dataclass
class CitationMatch:
    """Represents a match between response content and a fact."""

    fact_id: str
    content_span: Tuple[int, int]
    confidence: float
    citation_text: str
    source_reference: SourceReference


class CitationIntegrator:
    """Integrates citations seamlessly into response text."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the citation integrator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = get_settings()

        # Integration parameters
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.max_citation_length = self.config.get("max_citation_length", 100)
        self.enable_smart_placement = self.config.get("enable_smart_placement", True)

        # Citation formatting
        self.default_style = CitationStyle(self.config.get("default_style", "inline"))
        self.default_format = CitationFormat(
            self.config.get("default_format", "structured")
        )

        # Performance settings
        self.enable_caching = self.config.get("enable_caching", True)
        self.batch_size = self.config.get("batch_size", 10)

        # Initialize components
        self._cache = {} if self.enable_caching else None

        logger.info(
            "Citation integrator initialized",
            similarity_threshold=self.similarity_threshold,
            default_style=self.default_style.value,
            default_format=self.default_format.value,
            enable_smart_placement=self.enable_smart_placement,
        )

    async def integrate_citations(
        self,
        response: GeneratedResponse,
        facts: List[CitedFact],
        options: Optional[CitationOptions] = None,
    ) -> CitedResponse:
        """Integrate citations into response text.

        Args:
            response: Generated response to add citations to
            facts: List of cited facts used in the response
            options: Citation integration options

        Returns:
            Response with integrated citations
        """
        if not facts:
            return self._create_uncited_response(
                response, "No facts provided for citation"
            )

        start_time = time.time()
        options = options or CitationOptions()

        try:
            logger.info(
                "Starting citation integration",
                response_length=len(response.content),
                num_facts=len(facts),
                style=options.style.value,
                format=options.format.value,
            )

            # Find citation matches
            citation_matches = await self._find_citation_matches(
                response, facts, options
            )

            if not citation_matches:
                return self._create_uncited_response(
                    response, "No citation matches found"
                )

            # Integrate citations based on style
            if options.style == CitationStyle.INLINE:
                cited_content = await self._integrate_inline_citations(
                    response.content, citation_matches, options
                )
            elif options.style == CitationStyle.FOOTNOTE:
                cited_content = await self._integrate_footnote_citations(
                    response.content, citation_matches, options
                )
            elif options.style == CitationStyle.ENDNOTE:
                cited_content = await self._integrate_endnote_citations(
                    response.content, citation_matches, options
                )
            elif options.style == CitationStyle.PARENTHETICAL:
                cited_content = await self._integrate_parenthetical_citations(
                    response.content, citation_matches, options
                )
            else:  # SUPERSCRIPT
                cited_content = await self._integrate_superscript_citations(
                    response.content, citation_matches, options
                )

            # Generate citations list
            citations_list = self._generate_citations_list(citation_matches, options)

            # Create citation map
            citation_map = self._create_citation_map(citation_matches)

            # Verify citations if enabled
            verification_status = (
                "verified" if options.verify_citations else "not_verified"
            )
            if options.verify_citations:
                verification_status = await self._verify_citations(citation_matches)

            integration_time = time.time() - start_time

            logger.info(
                "Citation integration completed",
                citation_count=len(citation_matches),
                verification_status=verification_status,
                integration_time=integration_time,
            )

            return CitedResponse(
                content=cited_content,
                citations_list=citations_list,
                citation_map=citation_map,
                verification_status=verification_status,
                integration_time=integration_time,
                citation_count=len(citation_matches),
                metadata={
                    "integration_method": options.style.value,
                    "citation_format": options.format.value,
                    "original_response_length": len(response.content),
                    "cited_response_length": len(cited_content),
                },
            )

        except Exception as e:
            integration_time = time.time() - start_time
            logger.error(f"Citation integration failed: {e}")

            return CitedResponse(
                content=response.content,
                citations_list=[],
                citation_map={},
                verification_status="error",
                integration_time=integration_time,
                citation_count=0,
                metadata={"error": str(e)},
            )

    async def _find_citation_matches(
        self,
        response: GeneratedResponse,
        facts: List[CitedFact],
        options: CitationOptions,
    ) -> List[CitationMatch]:
        """Find matches between response content and facts for citation placement."""
        matches = []
        content = response.content.lower()

        for fact in facts:
            # Simple keyword-based matching
            fact_keywords = self._extract_keywords(fact.fact.content)

            for keyword in fact_keywords:
                if len(keyword) < 3:  # Skip very short keywords
                    continue

                # Find all occurrences of the keyword
                for match in re.finditer(re.escape(keyword.lower()), content):
                    start, end = match.span()

                    # Create citation match
                    citation_match = CitationMatch(
                        fact_id=fact.fact.fact_id,
                        content_span=(start, end),
                        confidence=fact.score,
                        citation_text=self._format_citation_text(fact, options),
                        source_reference=fact.sources[0] if fact.sources else None,
                    )
                    matches.append(citation_match)

        # Remove duplicates and sort by position
        unique_matches = self._deduplicate_matches(matches)
        unique_matches.sort(key=lambda m: m.content_span[0])

        # Limit citations per sentence
        filtered_matches = self._limit_citations_per_sentence(
            unique_matches, response.content, options.max_citations_per_sentence
        )

        return filtered_matches

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for citation matching."""
        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
        }

        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Return unique keywords, prioritizing longer ones
        return sorted(set(keywords), key=len, reverse=True)[:10]

    def _format_citation_text(self, fact: CitedFact, options: CitationOptions) -> str:
        """Format citation text in structured format."""
        if not fact.sources:
            return f"[unknown:unknown:0:fact_id={fact.fact.fact_id}]"

        source = fact.sources[0]
        return self._format_structured_citation(source)

    def _deduplicate_matches(self, matches: List[CitationMatch]) -> List[CitationMatch]:
        """Remove duplicate citation matches."""
        seen_spans = set()
        unique_matches = []

        for match in matches:
            span_key = (match.content_span[0], match.content_span[1])
            if span_key not in seen_spans:
                unique_matches.append(match)
                seen_spans.add(span_key)

        return unique_matches

    def _limit_citations_per_sentence(
        self, matches: List[CitationMatch], content: str, max_per_sentence: int
    ) -> List[CitationMatch]:
        """Limit the number of citations per sentence."""
        sentences = re.split(r"[.!?]+", content)
        sentence_positions = []

        # Find sentence boundaries
        pos = 0
        for sentence in sentences:
            start = pos
            end = pos + len(sentence)
            sentence_positions.append((start, end))
            pos = end + 1  # Account for delimiter

        # Group matches by sentence
        sentence_matches = {}
        for match in matches:
            match_pos = match.content_span[0]

            for i, (start, end) in enumerate(sentence_positions):
                if start <= match_pos <= end:
                    if i not in sentence_matches:
                        sentence_matches[i] = []
                    sentence_matches[i].append(match)
                    break

        # Limit matches per sentence
        filtered_matches = []
        for sentence_idx, sentence_match_list in sentence_matches.items():
            # Sort by confidence and take top N
            top_matches = sorted(
                sentence_match_list, key=lambda m: m.confidence, reverse=True
            )[:max_per_sentence]
            filtered_matches.extend(top_matches)

        return sorted(filtered_matches, key=lambda m: m.content_span[0])

    async def _integrate_inline_citations(
        self, content: str, matches: List[CitationMatch], options: CitationOptions
    ) -> str:
        """Integrate citations inline within the text."""
        # Sort matches by position (reverse order to maintain positions)
        sorted_matches = sorted(matches, key=lambda m: m.content_span[0], reverse=True)

        cited_content = content

        for match in sorted_matches:
            start, end = match.content_span
            citation = f" {match.citation_text}"

            # Insert citation after the matched text
            cited_content = cited_content[:end] + citation + cited_content[end:]

        return cited_content

    async def _integrate_footnote_citations(
        self, content: str, matches: List[CitationMatch], options: CitationOptions
    ) -> str:
        """Integrate citations as footnotes."""
        # Sort matches by position (reverse order)
        sorted_matches = sorted(matches, key=lambda m: m.content_span[0], reverse=True)

        cited_content = content
        footnotes = []

        for i, match in enumerate(reversed(sorted_matches), 1):
            start, end = match.content_span
            footnote_ref = f"[{i}]"

            # Insert footnote reference
            cited_content = cited_content[:end] + footnote_ref + cited_content[end:]

            # Add to footnotes list
            footnotes.append(f"{i}. {match.citation_text}")

        # Add footnotes at the end
        if footnotes:
            cited_content += "\n\n## Footnotes\n" + "\n".join(footnotes)

        return cited_content

    async def _integrate_endnote_citations(
        self, content: str, matches: List[CitationMatch], options: CitationOptions
    ) -> str:
        """Integrate citations as endnotes."""
        # Similar to footnotes but with "References" header
        result = await self._integrate_footnote_citations(content, matches, options)
        return result.replace("## Footnotes", "## References")

    async def _integrate_parenthetical_citations(
        self, content: str, matches: List[CitationMatch], options: CitationOptions
    ) -> str:
        """Integrate citations in parenthetical format."""
        # Sort matches by position (reverse order)
        sorted_matches = sorted(matches, key=lambda m: m.content_span[0], reverse=True)

        cited_content = content

        for match in sorted_matches:
            start, end = match.content_span
            citation = f" {match.citation_text}"

            # Insert citation after the matched text
            cited_content = cited_content[:end] + citation + cited_content[end:]

        return cited_content

    async def _integrate_superscript_citations(
        self, content: str, matches: List[CitationMatch], options: CitationOptions
    ) -> str:
        """Integrate citations as superscript numbers."""
        # Sort matches by position (reverse order)
        sorted_matches = sorted(matches, key=lambda m: m.content_span[0], reverse=True)

        cited_content = content

        for i, match in enumerate(reversed(sorted_matches), 1):
            start, end = match.content_span
            superscript = f"^{i}"

            # Insert superscript reference
            cited_content = cited_content[:end] + superscript + cited_content[end:]

        return cited_content

    def _generate_citations_list(
        self, matches: List[CitationMatch], options: CitationOptions
    ) -> List[str]:
        """Generate list of citations."""
        citations = []
        seen_citations = set()

        for i, match in enumerate(matches, 1):
            citation_text = match.citation_text

            if citation_text not in seen_citations:
                citations.append(f"{i}. {citation_text}")
                seen_citations.add(citation_text)

        return citations

    def _create_citation_map(
        self, matches: List[CitationMatch]
    ) -> Dict[str, SourceReference]:
        """Create mapping from fact IDs to source references."""
        citation_map = {}

        for match in matches:
            if match.source_reference:
                citation_map[match.fact_id] = match.source_reference

        return citation_map

    async def _verify_citations(self, matches: List[CitationMatch]) -> str:
        """Verify citation accuracy and completeness."""
        if not matches:
            return "no_citations"

        verified_count = 0

        for match in matches:
            # Simple verification: check if source reference exists
            if match.source_reference and match.source_reference.document_id:
                verified_count += 1

        verification_ratio = verified_count / len(matches)

        if verification_ratio >= 0.9:
            return "verified"
        elif verification_ratio >= 0.7:
            return "partially_verified"
        else:
            return "unverified"

    def _create_uncited_response(
        self, response: GeneratedResponse, reason: str
    ) -> CitedResponse:
        """Create response without citations when integration fails."""
        return CitedResponse(
            content=response.content,
            citations_list=[],
            citation_map={},
            verification_status="uncited",
            integration_time=0.0,
            citation_count=0,
            metadata={"uncited_reason": reason},
        )

    def _format_structured_citation(self, source: SourceReference) -> str:
        """Format a single source as structured citation: [document_type:filename:chunk_index:metadata]."""
        # Extract filename from document_title or document_id
        filename = source.document_title or source.document_id or "unknown"

        # Get document type, default to 'document' if not specified
        doc_type = source.document_type or "document"

        # Get chunk index, default to 0 if not specified
        chunk_index = source.chunk_id or "0"

        # Build metadata components based on document type
        metadata_parts = []

        if doc_type in ["audio", "video"] and source.timestamp:
            metadata_parts.append(f"timecode={source.timestamp}")
        elif doc_type == "pdf":
            if source.page_number:
                metadata_parts.append(f"page={source.page_number}")
            # Add chapter if available in metadata
            if source.metadata and source.metadata.get("chapter"):
                metadata_parts.append(f"chapter={source.metadata['chapter']}")
        elif doc_type in ["document", "word", "excel", "powerpoint"]:
            if source.page_number:
                metadata_parts.append(f"page={source.page_number}")

        # Add any additional metadata from the metadata dict
        if source.metadata:
            for key, value in source.metadata.items():
                if (
                    key not in ["chapter"] and value is not None
                ):  # chapter already handled above
                    metadata_parts.append(f"{key}={value}")

        # Construct the citation
        citation_base = f"[{doc_type}:{filename}:{chunk_index}"

        if metadata_parts:
            citation = citation_base + ":" + ":".join(metadata_parts) + "]"
        else:
            citation = citation_base + "]"

        return citation
