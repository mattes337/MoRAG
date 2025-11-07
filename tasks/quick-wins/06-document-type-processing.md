# Quick Win 6: Document Type-Specific Processing

## Overview

**Priority**: 📋 **Planned** (2-3 weeks, Medium Impact, Medium ROI)
**Source**: INRAExplorer domain specialization approach
**Expected Impact**: 20-30% improvement in structured knowledge extraction

## Problem Statement

MoRAG currently uses generic document processing for all document types, missing opportunities to extract structured knowledge specific to:
- Academic papers (methodology, findings, citations)
- Reports (executive summaries, recommendations, key metrics)
- Meeting notes (action items, decisions, participants)
- Legal documents (clauses, references, obligations)
- Technical documentation (procedures, specifications, requirements)

This generic approach fails to capture domain-specific structured information that could significantly improve knowledge graph quality.

## Solution Overview

Implement specialized extractors for common document types that identify and extract structured elements as entities and relationships, while maintaining the existing generic processing as a fallback.

## Technical Implementation

### 1. Document Type Classifier

Create `packages/morag-document/src/morag_document/document_classifier.py`:

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter

class DocumentType(Enum):
    ACADEMIC_PAPER = "academic_paper"
    BUSINESS_REPORT = "business_report"
    MEETING_NOTES = "meeting_notes"
    LEGAL_DOCUMENT = "legal_document"
    TECHNICAL_DOC = "technical_doc"
    NEWS_ARTICLE = "news_article"
    EMAIL = "email"
    GENERIC = "generic"

@dataclass
class DocumentClassification:
    document_type: DocumentType
    confidence: float
    indicators: List[str]
    metadata: Dict[str, any]

class DocumentTypeClassifier:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        # Cache for classification results
        self.classification_cache = {}

    async def classify_document(self, content: str, metadata: Dict[str, any] = None, language: str = None) -> DocumentClassification:
        """Classify document type using LLM-based analysis."""
        metadata = metadata or {}

        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        cache_key = f"{content_hash}:{language or 'auto'}"

        if cache_key in self.classification_cache:
            cached_result = self.classification_cache[cache_key]
            return DocumentClassification(
                document_type=DocumentType(cached_result['type']),
                confidence=cached_result['confidence'],
                indicators=cached_result['indicators'],
                metadata=metadata
            )

        # Use LLM for classification
        if self.llm_service:
            classification = await self._llm_classify_document(content, metadata, language)
        else:
            # Fallback to simple heuristics
            classification = self._fallback_classify_document(content, metadata)

        # Cache result
        self.classification_cache[cache_key] = {
            'type': classification.document_type.value,
            'confidence': classification.confidence,
            'indicators': classification.indicators
        }

        return classification

    async def _llm_classify_document(self, content: str, metadata: Dict[str, any], language: str = None) -> DocumentClassification:
        """Use LLM to classify document type."""

        # Truncate content for LLM analysis (first 2000 chars should be sufficient)
        content_sample = content[:2000] + "..." if len(content) > 2000 else content

        prompt = f"""
        Classify the following document into one of these types:

        1. ACADEMIC_PAPER: Research papers, journal articles, conference papers
        2. BUSINESS_REPORT: Business reports, financial reports, market analysis
        3. MEETING_NOTES: Meeting minutes, agenda, action items
        4. LEGAL_DOCUMENT: Contracts, agreements, legal documents
        5. TECHNICAL_DOC: Technical documentation, manuals, specifications
        6. NEWS_ARTICLE: News articles, press releases, journalism
        7. EMAIL: Email messages, correspondence
        8. GENERIC: General documents that don't fit other categories

        Document content:
        {content_sample}

        Metadata: {metadata}
        Language context: {language or "auto-detect"}

        Consider document structure, terminology, formatting, and purpose.

        Respond with JSON:
        {{
            "type": "document_type",
            "confidence": 0.0-1.0,
            "indicators": ["key features that led to this classification"],
            "reasoning": "brief explanation"
        }}
        """

        try:
            response = await self.llm_service.generate(prompt, max_tokens=300)
            result = json.loads(response)

            doc_type = DocumentType(result.get('type', 'GENERIC').lower())
            confidence = float(result.get('confidence', 0.7))
            indicators = result.get('indicators', [])

            return DocumentClassification(
                document_type=doc_type,
                confidence=confidence,
                indicators=indicators,
                metadata=metadata
            )

        except Exception as e:
            # Fallback on LLM failure
            return self._fallback_classify_document(content, metadata)

    def _fallback_classify_document(self, content: str, metadata: Dict[str, any]) -> DocumentClassification:
        """Fallback classification when LLM is unavailable."""

        # Simple heuristics based on file extension and basic patterns
        file_extension = metadata.get('file_extension', '').lower()
        filename = metadata.get('filename', '').lower()

        if file_extension in ['.eml', '.msg'] or 'from:' in content[:500]:
            doc_type = DocumentType.EMAIL
            confidence = 0.8
        elif 'abstract' in content[:1000].lower() and 'references' in content[-1000:].lower():
            doc_type = DocumentType.ACADEMIC_PAPER
            confidence = 0.6
        elif any(word in filename for word in ['minutes', 'meeting', 'agenda']):
            doc_type = DocumentType.MEETING_NOTES
            confidence = 0.6
        else:
            doc_type = DocumentType.GENERIC
            confidence = 0.4

        return DocumentClassification(
            document_type=doc_type,
            confidence=confidence,
            indicators=["fallback_heuristics"],
            metadata=metadata
        )


```

### 2. Specialized Extractors

Create `packages/morag-document/src/morag_document/specialized_extractors/academic_extractor.py`:

```python
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass
class AcademicEntity:
    name: str
    entity_type: str  # author, institution, concept, method, finding
    context: str
    confidence: float
    metadata: Dict[str, any]

class AcademicPaperExtractor:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        # Cache for extraction results
        self.extraction_cache = {}

    async def extract_academic_entities(self, content: str, language: str = None) -> List[AcademicEntity]:
        """Extract academic-specific entities using LLM-based analysis."""

        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        cache_key = f"{content_hash}:{language or 'auto'}"

        if cache_key in self.extraction_cache:
            return self.extraction_cache[cache_key]

        entities = []

        if self.llm_service:
            # Use LLM for comprehensive academic entity extraction
            entities = await self._llm_extract_academic_entities(content, language)
        else:
            # Fallback to basic pattern matching
            entities = self._fallback_extract_academic_entities(content)

        # Cache results
        self.extraction_cache[cache_key] = entities

        return entities

    async def _llm_extract_academic_entities(self, content: str, language: str = None) -> List[AcademicEntity]:
        """Use LLM to extract academic entities."""

        # Truncate content for LLM analysis
        content_sample = content[:4000] + "..." if len(content) > 4000 else content

        prompt = f"""
        Extract academic entities from the following research paper content:

        Extract these types of entities:
        1. AUTHORS: Author names
        2. INSTITUTIONS: Universities, research institutions, organizations
        3. METHODOLOGIES: Research methods, approaches, techniques used
        4. FINDINGS: Key research findings, results, conclusions
        5. CITATIONS: DOIs, ArXiv IDs, reference identifiers
        6. CONCEPTS: Key academic concepts, terminology, technical terms

        Content:
        {content_sample}

        Language context: {language or "auto-detect"}

        Respond with JSON array:
        [
            {{
                "name": "entity name",
                "type": "AUTHOR|INSTITUTION|METHODOLOGY|FINDING|CITATION|CONCEPT",
                "context": "surrounding text snippet",
                "confidence": 0.0-1.0,
                "metadata": {{"additional_info": "value"}}
            }}
        ]

        Only include entities with confidence > 0.6.
        """

        try:
            response = await self.llm_service.generate(prompt, max_tokens=2000)
            entity_data = json.loads(response)

            entities = []
            for item in entity_data:
                if isinstance(item, dict) and item.get('confidence', 0) > 0.6:
                    entities.append(AcademicEntity(
                        name=item.get('name', ''),
                        entity_type=item.get('type', 'concept').lower(),
                        context=item.get('context', ''),
                        confidence=float(item.get('confidence', 0.7)),
                        metadata=item.get('metadata', {'extraction_method': 'llm'})
                    ))

            return entities

        except Exception as e:
            # Fallback on LLM failure
            return self._fallback_extract_academic_entities(content)

    def _fallback_extract_academic_entities(self, content: str) -> List[AcademicEntity]:
        """Fallback extraction when LLM is unavailable."""
        entities = []

        # Simple pattern-based extraction for DOIs and ArXiv IDs
        doi_pattern = r'doi:\s*([^\s]+)'
        for match in re.finditer(doi_pattern, content, re.IGNORECASE):
            entities.append(AcademicEntity(
                name=f"DOI: {match.group(1)}",
                entity_type='citation',
                context=self._get_context(content, match.start(), match.end()),
                confidence=0.9,
                metadata={'citation_type': 'doi', 'extraction_method': 'pattern'}
            ))

        arxiv_pattern = r'arxiv:\s*([^\s]+)'
        for match in re.finditer(arxiv_pattern, content, re.IGNORECASE):
            entities.append(AcademicEntity(
                name=f"ArXiv: {match.group(1)}",
                entity_type='citation',
                context=self._get_context(content, match.start(), match.end()),
                confidence=0.9,
                metadata={'citation_type': 'arxiv', 'extraction_method': 'pattern'}
            ))

        return entities

    def _get_context(self, content: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end].strip()
```

### 3. Meeting Notes Extractor

Create `packages/morag-document/src/morag_document/specialized_extractors/meeting_extractor.py`:

```python
from typing import Dict, List, Optional
import re
from dataclasses import dataclass

@dataclass
class MeetingEntity:
    name: str
    entity_type: str  # participant, action_item, decision, topic
    context: str
    confidence: float
    metadata: Dict[str, any]

class MeetingNotesExtractor:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        # Cache for extraction results
        self.extraction_cache = {}

    async def extract_meeting_entities(self, content: str, language: str = None) -> List[MeetingEntity]:
        """Extract meeting-specific entities using LLM-based analysis."""

        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        cache_key = f"{content_hash}:{language or 'auto'}"

        if cache_key in self.extraction_cache:
            return self.extraction_cache[cache_key]

        entities = []

        if self.llm_service:
            # Use LLM for comprehensive meeting entity extraction
            entities = await self._llm_extract_meeting_entities(content, language)
        else:
            # Fallback to basic pattern matching
            entities = self._fallback_extract_meeting_entities(content)

        # Cache results
        self.extraction_cache[cache_key] = entities

        return entities

    async def _llm_extract_meeting_entities(self, content: str, language: str = None) -> List[MeetingEntity]:
        """Use LLM to extract meeting entities."""

        prompt = f"""
        Extract meeting-related entities from the following meeting notes/minutes:

        Extract these types of entities:
        1. PARTICIPANTS: Names of meeting attendees, participants
        2. ACTION_ITEMS: Tasks assigned, action items, to-dos
        3. DECISIONS: Decisions made, agreements reached
        4. TOPICS: Agenda items, discussion topics, subjects covered

        Content:
        {content}

        Language context: {language or "auto-detect"}

        Respond with JSON array:
        [
            {{
                "name": "entity name/description",
                "type": "PARTICIPANT|ACTION_ITEM|DECISION|TOPIC",
                "context": "surrounding text snippet",
                "confidence": 0.0-1.0,
                "metadata": {{"additional_info": "value"}}
            }}
        ]

        Only include entities with confidence > 0.6.
        """

        try:
            response = await self.llm_service.generate(prompt, max_tokens=1500)
            entity_data = json.loads(response)

            entities = []
            for item in entity_data:
                if isinstance(item, dict) and item.get('confidence', 0) > 0.6:
                    entities.append(MeetingEntity(
                        name=item.get('name', ''),
                        entity_type=item.get('type', 'topic').lower(),
                        context=item.get('context', ''),
                        confidence=float(item.get('confidence', 0.7)),
                        metadata=item.get('metadata', {'extraction_method': 'llm'})
                    ))

            return entities

        except Exception as e:
            # Fallback on LLM failure
            return self._fallback_extract_meeting_entities(content)

    def _fallback_extract_meeting_entities(self, content: str) -> List[MeetingEntity]:
        """Fallback extraction when LLM is unavailable."""
        entities = []

        # Simple pattern-based extraction for common meeting elements
        # Look for attendee lists
        attendee_pattern = r'(?:attendees?|participants?|present):\s*([^:\n]+)'
        attendee_match = re.search(attendee_pattern, content, re.IGNORECASE)
        if attendee_match:
            attendee_text = attendee_match.group(1)
            names = [name.strip() for name in re.split(r'[,;]', attendee_text)]

            for name in names:
                if len(name.split()) >= 2:  # At least first and last name
                    entities.append(MeetingEntity(
                        name=name,
                        entity_type='participant',
                        context=self._get_context(content, attendee_match.start(), attendee_match.end()),
                        confidence=0.8,
                        metadata={'extraction_method': 'pattern_fallback'}
                    ))

        return entities

    def _get_context(self, content: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end].strip()
```

### 4. Integration with Document Processor

Update `packages/morag-document/src/morag_document/document_processor.py`:

```python
from .document_classifier import DocumentTypeClassifier, DocumentType
from .specialized_extractors.academic_extractor import AcademicPaperExtractor
from .specialized_extractors.meeting_extractor import MeetingNotesExtractor

class EnhancedDocumentProcessor:
    def __init__(self, llm_service=None):
        self.classifier = DocumentTypeClassifier(llm_service)
        self.academic_extractor = AcademicPaperExtractor(llm_service)
        self.meeting_extractor = MeetingNotesExtractor(llm_service)
        # Add other specialized extractors as needed

    async def process_document(self, content: str, metadata: Dict[str, any] = None, language: str = None) -> Dict[str, any]:
        """Process document with type-specific extraction."""

        # Classify document type
        classification = await self.classifier.classify_document(content, metadata, language)

        # Extract generic entities (existing pipeline)
        generic_entities = await self._extract_generic_entities(content, language)

        # Extract specialized entities based on type
        specialized_entities = []
        if classification.document_type == DocumentType.ACADEMIC_PAPER:
            specialized_entities = await self.academic_extractor.extract_academic_entities(content, language)
        elif classification.document_type == DocumentType.MEETING_NOTES:
            specialized_entities = await self.meeting_extractor.extract_meeting_entities(content, language)
        # Add other document types as needed

        # Combine and deduplicate entities
        all_entities = self._combine_entities(generic_entities, specialized_entities)

        return {
            'document_classification': classification,
            'generic_entities': generic_entities,
            'specialized_entities': specialized_entities,
            'combined_entities': all_entities,
            'extraction_stats': {
                'total_entities': len(all_entities),
                'generic_count': len(generic_entities),
                'specialized_count': len(specialized_entities),
                'document_type': classification.document_type.value,
                'classification_confidence': classification.confidence
            }
        }

    def _combine_entities(self, generic_entities: List[Dict], specialized_entities: List) -> List[Dict]:
        """Combine and deduplicate generic and specialized entities."""
        # Convert specialized entities to common format
        combined = list(generic_entities)

        for spec_entity in specialized_entities:
            # Convert to common entity format
            entity_dict = {
                'name': spec_entity.name,
                'type': spec_entity.entity_type,
                'confidence': spec_entity.confidence,
                'context': spec_entity.context,
                'extraction_method': 'specialized',
                'metadata': spec_entity.metadata
            }
            combined.append(entity_dict)

        # Simple deduplication by name (could be more sophisticated)
        seen_names = set()
        deduplicated = []
        for entity in combined:
            if entity['name'] not in seen_names:
                seen_names.add(entity['name'])
                deduplicated.append(entity)

        return deduplicated
```

## Configuration

```yaml
# document_type_processing.yml
document_type_processing:
  enabled: true

  classification:
    confidence_threshold: 0.3
    fallback_to_generic: true

  academic_papers:
    enabled: true
    extract_authors: true
    extract_institutions: true
    extract_methodologies: true
    extract_findings: true
    extract_citations: true

  meeting_notes:
    enabled: true
    extract_participants: true
    extract_action_items: true
    extract_decisions: true
    extract_topics: true

  business_reports:
    enabled: true
    extract_metrics: true
    extract_recommendations: true
    extract_executives: true

  legal_documents:
    enabled: false  # Future implementation

  technical_docs:
    enabled: false  # Future implementation
```

## Testing Strategy

```python
# tests/unit/test_document_classification.py
import pytest
from morag_document.document_classifier import DocumentTypeClassifier, DocumentType

class TestDocumentClassification:
    def setup_method(self):
        self.classifier = DocumentTypeClassifier()

    def test_academic_paper_classification(self):
        content = """
        Abstract: This paper presents a novel approach to machine learning.
        Introduction: Recent advances in deep learning have shown...
        Methodology: We used a convolutional neural network...
        Results: Our experiments show significant improvement...
        Conclusion: We have demonstrated that our approach...
        References: [1] Smith, J. et al. (2020)...
        """

        result = self.classifier.classify_document(content)
        assert result.document_type == DocumentType.ACADEMIC_PAPER
        assert result.confidence > 0.7

    def test_meeting_notes_classification(self):
        content = """
        Meeting Minutes - Project Review
        Attendees: John Smith, Jane Doe, Bob Johnson
        Agenda:
        1. Project status update
        2. Budget review
        Action Items:
        - John to complete requirements document
        - Jane to schedule next meeting
        Decisions:
        - Approved budget increase of 10%
        """

        result = self.classifier.classify_document(content)
        assert result.document_type == DocumentType.MEETING_NOTES
        assert result.confidence > 0.7
```

## Monitoring

```python
document_type_metrics = {
    'classification_accuracy': 0.0,
    'document_type_distribution': {
        'academic_paper': 0,
        'meeting_notes': 0,
        'business_report': 0,
        'generic': 0
    },
    'specialized_extraction_quality': {
        'entities_per_document': 0.0,
        'confidence_scores': 0.0,
        'extraction_coverage': 0.0
    }
}
```

## Success Metrics

- **Classification Accuracy**: >85% correct document type identification
- **Extraction Quality**: 20-30% more structured entities extracted
- **Domain Coverage**: Support for 5+ document types
- **Processing Speed**: <10% overhead for specialized processing

## Future Enhancements

1. **ML-based Classification**: Train models for better type detection
2. **More Document Types**: Legal, technical, financial documents
3. **Template Recognition**: Identify and extract from document templates
4. **Cross-Document Linking**: Link entities across related documents
