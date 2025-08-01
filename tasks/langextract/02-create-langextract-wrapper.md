# Task 2: Create LangExtract Wrapper

## Objective
Create a MoRAG-compatible wrapper around LangExtract that maintains the existing API interfaces while leveraging LangExtract's capabilities.

## Prerequisites
- Task 1 completed (LangExtract installed and configured)
- Understanding of current MoRAG Entity/Relation models
- Access to existing extractor interfaces

## Steps

### 1. Create Base LangExtract Extractor

**File**: `packages/morag-graph/src/morag_graph/extraction/langextract_base.py`
```python
"""Base LangExtract extractor for MoRAG integration."""

import asyncio
import structlog
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import langextract as lx

from ..models import Entity, Relation
from ..services.langextract_service import LangExtractService
from ..config.langextract_config import get_langextract_config, get_domain_config
from .base import BaseExtractor

logger = structlog.get_logger(__name__)


class LangExtractBaseExtractor(BaseExtractor, ABC):
    """Base class for LangExtract-based extractors."""
    
    def __init__(
        self,
        domain: str = "general",
        min_confidence: float = 0.7,
        **kwargs
    ):
        """Initialize LangExtract base extractor.
        
        Args:
            domain: Domain for extraction (general, medical, legal, etc.)
            min_confidence: Minimum confidence threshold
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.domain = domain
        self.min_confidence = min_confidence
        self.config = get_langextract_config()
        self.domain_config = get_domain_config(domain)
        self.service = LangExtractService(self.config)
        self.logger = logger.bind(
            extractor=self.__class__.__name__,
            domain=domain
        )
        
        # Load domain-specific examples
        self.examples = self._load_domain_examples()
    
    @abstractmethod
    def _load_domain_examples(self) -> List[lx.data.ExampleData]:
        """Load domain-specific examples for few-shot learning."""
        pass
    
    @abstractmethod
    def _get_prompt_description(self) -> str:
        """Get prompt description for this extractor type."""
        pass
    
    @abstractmethod
    def _convert_extractions_to_models(
        self,
        extractions: List[lx.data.Extraction],
        source_doc_id: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """Convert LangExtract extractions to MoRAG models."""
        pass
    
    async def extract_with_langextract(
        self,
        text: str,
        source_doc_id: Optional[str] = None
    ) -> lx.data.AnnotatedDocument:
        """Extract using LangExtract service.
        
        Args:
            text: Input text
            source_doc_id: Optional source document ID
            
        Returns:
            LangExtract annotated document
        """
        prompt = self._get_prompt_description()
        
        return await self.service.extract_structured_data(
            text=text,
            prompt_description=prompt,
            examples=self.examples,
            source_doc_id=source_doc_id
        )
    
    def _filter_by_confidence(
        self,
        extractions: List[lx.data.Extraction]
    ) -> List[lx.data.Extraction]:
        """Filter extractions by confidence threshold."""
        # LangExtract doesn't provide confidence scores directly
        # We'll use all extractions and apply confidence in conversion
        return extractions
    
    def _create_entity_from_extraction(
        self,
        extraction: lx.data.Extraction,
        source_doc_id: Optional[str] = None
    ) -> Entity:
        """Create MoRAG Entity from LangExtract extraction."""
        # Extract confidence from attributes if available
        confidence = float(extraction.attributes.get('confidence', 0.8))
        
        # Apply domain-specific confidence thresholds
        if self.domain_config:
            class_threshold = self.domain_config.confidence_thresholds.get(
                extraction.extraction_class, self.min_confidence
            )
            confidence = max(confidence, class_threshold)
        
        return Entity(
            name=extraction.extraction_text,
            type=extraction.extraction_class.upper(),
            attributes=extraction.attributes,
            confidence=confidence,
            source_doc_id=source_doc_id,
            # Add source grounding information
            source_offsets=(extraction.start_char, extraction.end_char) if hasattr(extraction, 'start_char') else None
        )
    
    def _create_relation_from_extraction(
        self,
        extraction: lx.data.Extraction,
        entities: List[Entity],
        source_doc_id: Optional[str] = None
    ) -> Optional[Relation]:
        """Create MoRAG Relation from LangExtract extraction."""
        # Extract relation components from attributes
        source_entity = extraction.attributes.get('source_entity')
        target_entity = extraction.attributes.get('target_entity')
        relation_type = extraction.attributes.get('relation_type', extraction.extraction_class)
        
        if not source_entity or not target_entity:
            return None
        
        # Find matching entities
        source_id = self._find_entity_id(source_entity, entities)
        target_id = self._find_entity_id(target_entity, entities)
        
        if not source_id or not target_id:
            return None
        
        confidence = float(extraction.attributes.get('confidence', 0.8))
        
        return Relation(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=relation_type.upper(),
            attributes=extraction.attributes,
            confidence=confidence,
            source_doc_id=source_doc_id
        )
    
    def _find_entity_id(self, entity_name: str, entities: List[Entity]) -> Optional[str]:
        """Find entity ID by name."""
        for entity in entities:
            if entity.name.lower() == entity_name.lower():
                return entity.id
        return None
    
    def generate_visualization(
        self,
        results: List[lx.data.AnnotatedDocument],
        output_path: Optional[str] = None
    ) -> str:
        """Generate HTML visualization for results."""
        return self.service.generate_visualization(results, output_path)
```

### 2. Create LangExtract Entity Extractor

**File**: `packages/morag-graph/src/morag_graph/extraction/langextract_entity_extractor.py`
```python
"""LangExtract-based entity extractor."""

import structlog
from typing import List, Dict, Any, Optional, Tuple
import langextract as lx

from ..models import Entity, Relation
from .langextract_base import LangExtractBaseExtractor

logger = structlog.get_logger(__name__)


class LangExtractEntityExtractor(LangExtractBaseExtractor):
    """Entity extractor using LangExtract."""
    
    def _load_domain_examples(self) -> List[lx.data.ExampleData]:
        """Load domain-specific examples for entity extraction."""
        if self.domain == "medical":
            return self._get_medical_examples()
        elif self.domain == "legal":
            return self._get_legal_examples()
        elif self.domain == "technical":
            return self._get_technical_examples()
        else:
            return self._get_general_examples()
    
    def _get_prompt_description(self) -> str:
        """Get prompt description for entity extraction."""
        return f"""Extract named entities from the text in the {self.domain} domain.
        
        Focus on identifying:
        - People, organizations, and locations
        - Domain-specific concepts and terminology
        - Important objects and substances
        - Events and processes
        
        For each entity, provide:
        - extraction_class: The type of entity (PERSON, ORGANIZATION, CONCEPT, etc.)
        - extraction_text: The exact text as it appears
        - attributes: Additional context like confidence, description, etc.
        
        Use exact text for extractions. Do not paraphrase or modify entity names.
        Provide meaningful attributes to add context and enable better relationship extraction."""
    
    def _convert_extractions_to_models(
        self,
        extractions: List[lx.data.Extraction],
        source_doc_id: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """Convert LangExtract extractions to MoRAG entities."""
        entities = []
        
        for extraction in extractions:
            # Only process entity-type extractions
            if self._is_entity_extraction(extraction):
                entity = self._create_entity_from_extraction(extraction, source_doc_id)
                if entity.confidence >= self.min_confidence:
                    entities.append(entity)
        
        # No relations from entity-only extraction
        return entities, []
    
    def _is_entity_extraction(self, extraction: lx.data.Extraction) -> bool:
        """Check if extraction represents an entity."""
        entity_classes = {
            'person', 'organization', 'location', 'concept', 
            'substance', 'technology', 'event', 'process',
            'disease', 'medication', 'treatment', 'symptom',  # Medical
            'law', 'court', 'contract', 'regulation',  # Legal
            'algorithm', 'framework', 'protocol', 'system'  # Technical
        }
        return extraction.extraction_class.lower() in entity_classes
    
    async def extract(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Entity]:
        """Extract entities from text using LangExtract.
        
        Args:
            text: Input text
            doc_id: Document ID (deprecated, use source_doc_id)
            source_doc_id: Source document ID
            **kwargs: Additional arguments
            
        Returns:
            List of extracted entities
        """
        # Handle backward compatibility
        if source_doc_id is None and doc_id is not None:
            source_doc_id = doc_id
        
        try:
            self.logger.info(
                "Starting entity extraction",
                text_length=len(text),
                domain=self.domain,
                source_doc_id=source_doc_id
            )
            
            # Extract using LangExtract
            result = await self.extract_with_langextract(text, source_doc_id)
            
            # Convert to MoRAG entities
            entities, _ = self._convert_extractions_to_models(
                result.extractions, source_doc_id
            )
            
            self.logger.info(
                "Entity extraction completed",
                entities_found=len(entities),
                source_doc_id=source_doc_id
            )
            
            return entities
            
        except Exception as e:
            self.logger.error(
                "Entity extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                source_doc_id=source_doc_id
            )
            raise
    
    def _get_general_examples(self) -> List[lx.data.ExampleData]:
        """Get general domain examples."""
        return [
            lx.data.ExampleData(
                text="Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="organization",
                        extraction_text="Apple Inc.",
                        attributes={"industry": "technology", "type": "company"}
                    ),
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="Steve Jobs",
                        attributes={"role": "founder", "significance": "high"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Cupertino, California",
                        attributes={"type": "city_state", "country": "USA"}
                    )
                ]
            )
        ]
    
    def _get_medical_examples(self) -> List[lx.data.ExampleData]:
        """Get medical domain examples."""
        return [
            lx.data.ExampleData(
                text="Patient presents with hypertension and was prescribed lisinopril 10mg daily.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="disease",
                        extraction_text="hypertension",
                        attributes={"category": "cardiovascular", "severity": "unspecified"}
                    ),
                    lx.data.Extraction(
                        extraction_class="medication",
                        extraction_text="lisinopril",
                        attributes={"dosage": "10mg", "frequency": "daily", "class": "ACE_inhibitor"}
                    )
                ]
            )
        ]
    
    def _get_legal_examples(self) -> List[lx.data.ExampleData]:
        """Get legal domain examples."""
        return [
            lx.data.ExampleData(
                text="The Supreme Court ruled on the contract dispute under Section 15 of the Commercial Code.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="court",
                        extraction_text="Supreme Court",
                        attributes={"level": "highest", "jurisdiction": "federal"}
                    ),
                    lx.data.Extraction(
                        extraction_class="law",
                        extraction_text="Section 15 of the Commercial Code",
                        attributes={"type": "statute", "area": "commercial"}
                    )
                ]
            )
        ]
    
    def _get_technical_examples(self) -> List[lx.data.ExampleData]:
        """Get technical domain examples."""
        return [
            lx.data.ExampleData(
                text="The system uses TensorFlow framework with BERT algorithm for natural language processing.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="framework",
                        extraction_text="TensorFlow",
                        attributes={"type": "machine_learning", "vendor": "Google"}
                    ),
                    lx.data.Extraction(
                        extraction_class="algorithm",
                        extraction_text="BERT",
                        attributes={"type": "transformer", "application": "NLP"}
                    )
                ]
            )
        ]
```

## Verification Steps

1. **Test Entity Extraction**:
   ```python
   from morag_graph.extraction.langextract_entity_extractor import LangExtractEntityExtractor
   
   extractor = LangExtractEntityExtractor(domain="general")
   entities = await extractor.extract("Apple Inc. was founded by Steve Jobs.")
   print(f"Found {len(entities)} entities")
   ```

2. **Test Domain Examples**:
   ```python
   medical_extractor = LangExtractEntityExtractor(domain="medical")
   examples = medical_extractor._load_domain_examples()
   print(f"Loaded {len(examples)} medical examples")
   ```

3. **Test Visualization**:
   ```python
   result = await extractor.extract_with_langextract("Test text")
   html = extractor.generate_visualization([result])
   print(f"Generated {len(html)} characters of HTML")
   ```

## Success Criteria

- [ ] Base extractor class implemented
- [ ] Entity extractor with domain support
- [ ] Examples for multiple domains
- [ ] Conversion to MoRAG models working
- [ ] Visualization integration
- [ ] Backward compatibility maintained

## Next Steps

After completing this task:
1. Move to Task 3: Create domain examples
2. Test with real documents
3. Implement relation extractor wrapper
