# Task 3: Fact Extractor Implementation

## Objective
Implement the core fact extractor component that analyzes document chunks and extracts structured facts with proper provenance tracking.

## Implementation Plan

### 3.1 Core Fact Extractor Class

**File**: `packages/morag-graph/src/morag_graph/extraction/fact_extractor.py`

```python
class FactExtractor:
    """Extract structured facts from document chunks."""
    
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        min_confidence: float = 0.7,
        max_facts_per_chunk: int = 10,
        domain: str = "general",
        language: str = "en"
    ):
        """Initialize fact extractor with LLM and configuration."""
        
    async def extract_facts(
        self, 
        chunk_text: str, 
        chunk_id: str,
        document_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Fact]:
        """Extract structured facts from a document chunk."""
        
    def _preprocess_chunk(self, text: str) -> str:
        """Clean and prepare text for fact extraction."""
        
    async def _extract_fact_candidates(self, text: str, context: Dict) -> List[Dict]:
        """Use LLM to extract fact candidates from text."""
        
    def _structure_facts(self, candidates: List[Dict], chunk_id: str, document_id: str) -> List[Fact]:
        """Convert LLM output to structured Fact objects."""
        
    def _validate_fact(self, fact: Fact) -> bool:
        """Validate fact quality and completeness."""
        
    def _generate_fact_keywords(self, fact: Fact) -> List[str]:
        """Generate keywords for fact indexing."""
```

### 3.2 Fact Model Definition

**File**: `packages/morag-graph/src/morag_graph/models/fact.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class Fact(BaseModel):
    """Structured fact extracted from document content."""
    
    id: str = Field(description="Unique fact identifier")
    subject: str = Field(description="What the fact is about")
    object: str = Field(description="What is being described or acted upon")
    approach: Optional[str] = Field(default=None, description="How something is done/achieved")
    solution: Optional[str] = Field(default=None, description="What solves a problem/achieves goal")
    remarks: Optional[str] = Field(default=None, description="Additional context/qualifications")
    
    # Provenance
    source_chunk_id: str = Field(description="Source document chunk ID")
    source_document_id: str = Field(description="Source document ID")
    extraction_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in extraction")
    
    # Classification
    fact_type: str = Field(description="Type of fact (research, process, definition, etc.)")
    domain: Optional[str] = Field(default=None, description="Domain/topic area")
    keywords: List[str] = Field(default_factory=list, description="Key terms for indexing")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    language: str = Field(default="en", description="Language of the fact")
    
    def __init__(self, **data):
        """Initialize fact with auto-generated ID if not provided."""
        if 'id' not in data or not data['id']:
            # Generate deterministic ID based on content
            content_hash = hashlib.md5(
                f"{data.get('subject', '')}{data.get('object', '')}{data.get('source_chunk_id', '')}".encode()
            ).hexdigest()[:12]
            data['id'] = f"fact_{content_hash}"
        super().__init__(**data)
    
    def get_neo4j_properties(self) -> Dict[str, Any]:
        """Get properties for Neo4j storage."""
        return {
            "id": self.id,
            "subject": self.subject,
            "object": self.object,
            "approach": self.approach,
            "solution": self.solution,
            "remarks": self.remarks,
            "fact_type": self.fact_type,
            "domain": self.domain,
            "confidence": self.extraction_confidence,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "keywords": ",".join(self.keywords)
        }
```

### 3.3 LLM Prompting System

**File**: `packages/morag-graph/src/morag_graph/extraction/fact_prompts.py`

```python
class FactExtractionPrompts:
    """Prompts for fact extraction using LLMs."""
    
    @staticmethod
    def get_fact_extraction_prompt(domain: str = "general", language: str = "en") -> str:
        """Get the main fact extraction prompt."""
        
    @staticmethod
    def get_fact_validation_prompt() -> str:
        """Get prompt for validating extracted facts."""
        
    @staticmethod
    def get_fact_type_classification_prompt() -> str:
        """Get prompt for classifying fact types."""
```

### 3.4 Quality Validation

**File**: `packages/morag-graph/src/morag_graph/extraction/fact_validator.py`

```python
class FactValidator:
    """Validate quality and completeness of extracted facts."""
    
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        
    def validate_fact(self, fact: Fact) -> Tuple[bool, List[str]]:
        """Validate a fact and return validation result with issues."""
        
    def _check_completeness(self, fact: Fact) -> List[str]:
        """Check if fact has required components."""
        
    def _check_specificity(self, fact: Fact) -> List[str]:
        """Check if fact is specific enough to be useful."""
        
    def _check_actionability(self, fact: Fact) -> List[str]:
        """Check if fact provides actionable information."""
```

## Implementation Tasks

### Task 3.1: Core Infrastructure
- [ ] Create fact model with proper validation
- [ ] Implement fact ID generation system
- [ ] Set up basic fact extractor class structure
- [ ] Create configuration management for fact extraction

### Task 3.2: LLM Integration
- [ ] Design fact extraction prompts for different domains
- [ ] Implement LLM calling logic with error handling
- [ ] Add response parsing and validation
- [ ] Implement retry logic for failed extractions

### Task 3.3: Fact Processing Pipeline
- [ ] Implement text preprocessing for fact extraction
- [ ] Create fact candidate identification logic
- [ ] Build fact structuring and validation pipeline
- [ ] Add keyword generation for fact indexing

### Task 3.4: Quality Assurance
- [ ] Implement fact validation rules
- [ ] Create confidence scoring system
- [ ] Add fact deduplication logic
- [ ] Implement quality metrics collection

### Task 3.5: Testing and Validation
- [ ] Create unit tests for fact extraction
- [ ] Test with various document types and domains
- [ ] Validate fact quality and completeness
- [ ] Performance testing and optimization

## LLM Prompt Design

### Main Extraction Prompt
```
You are a knowledge extraction expert. Extract structured facts from the following text that represent actionable, specific information.

A fact should contain:
- Subject: The main entity or concept the fact is about
- Object: What is being described, studied, or acted upon  
- Approach: The method, technique, or way something is done (optional)
- Solution: The result, outcome, or answer provided (optional)
- Remarks: Important context, limitations, or qualifications (optional)

Extract only facts that are:
1. Specific and actionable (not generic statements)
2. Verifiable from the text
3. Useful for answering questions
4. Complete enough to stand alone

Text: {chunk_text}
Domain: {domain}
Language: {language}

Respond with JSON array of facts:
[
  {
    "subject": "specific subject",
    "object": "what is being described",
    "approach": "how it's done (optional)",
    "solution": "outcome/result (optional)", 
    "remarks": "context/limitations (optional)",
    "fact_type": "research|process|definition|causal|comparative|temporal",
    "confidence": 0.0-1.0,
    "keywords": ["key", "terms"]
  }
]
```

### Validation Prompt
```
Evaluate the quality of this extracted fact:

Fact: {fact_json}

Rate the fact on:
1. Specificity (0-1): Is it specific rather than generic?
2. Actionability (0-1): Does it provide useful, applicable information?
3. Completeness (0-1): Does it contain sufficient context?
4. Verifiability (0-1): Can it be traced to source text?

Respond with JSON:
{
  "overall_score": 0.0-1.0,
  "specificity": 0.0-1.0,
  "actionability": 0.0-1.0,
  "completeness": 0.0-1.0,
  "verifiability": 0.0-1.0,
  "issues": ["list of specific issues"],
  "suggestions": ["improvement suggestions"]
}
```

## Success Criteria

1. **Extraction Quality**: Facts are specific, actionable, and well-structured
2. **Performance**: Can process document chunks efficiently
3. **Validation**: Robust quality checking prevents low-quality facts
4. **Flexibility**: Works across different domains and languages
5. **Integration**: Seamlessly integrates with existing document processing pipeline
