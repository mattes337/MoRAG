# Task 1.3: NLP Pipeline Foundation

**Phase**: 1 - Foundation Infrastructure  
**Priority**: Critical  
**Total Estimated Time**: 8-10 days  
**Dependencies**: None (can run in parallel with Tasks 1.1 and 1.2)

## Overview

This task creates the `morag-nlp` package, which provides natural language processing capabilities for entity recognition, relation extraction, and text analysis required for the graph-augmented RAG system.

## Subtasks

### Task 1.3.1: Create morag-nlp Package
**Priority**: Critical  
**Estimated Time**: 3-4 days  
**Dependencies**: None

#### Implementation Steps

1. **Package Structure Setup**
   - Create package directory structure
   - Set up pyproject.toml with NLP dependencies
   - Initialize core modules and interfaces

2. **Base Extractor Interfaces**
   - Define abstract entity extractor interface
   - Create relation extractor interface
   - Implement common NLP utilities

3. **Configuration and Model Management**
   - Set up model download and caching
   - Create configuration management
   - Implement model versioning

#### Package Structure
```
packages/morag-nlp/
├── src/morag_nlp/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ner_models.py
│   │   ├── relation_models.py
│   │   └── embedding_models.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── entity_extractor.py
│   │   ├── relation_extractor.py
│   │   └── keyword_extractor.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── text_processor.py
│   │   ├── context_processor.py
│   │   └── pipeline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_utils.py
│   │   ├── model_utils.py
│   │   └── validation.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_extractors.py
│   ├── test_processors.py
│   └── test_models.py
├── models/  # Pre-trained model storage
│   ├── spacy/
│   ├── transformers/
│   └── custom/
├── examples/
│   ├── basic_extraction.py
│   ├── custom_training.py
│   └── pipeline_usage.py
├── pyproject.toml
└── README.md
```

#### Code Examples

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "morag-nlp"
version = "0.1.0"
description = "Natural Language Processing for MoRAG"
authors = [{name = "MoRAG Team"}]
requires-python = ">=3.9"
dependencies = [
    "spacy>=3.7.0",
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
    "nltk>=3.8.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "regex>=2023.0.0",
    "sentence-transformers>=2.2.0",
    "spacy-transformers>=1.2.0",
    "datasets>=2.12.0",
    "evaluate>=0.4.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "jupyter>=1.0.0"
]
gpu = [
    "torch[cuda]>=2.0.0",
    "transformers[torch]>=4.30.0"
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 88
```

**Base Extractor Interface**:
```python
# src/morag_nlp/extractors/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

@dataclass
class ExtractedEntity:
    """Represents an entity extracted from text"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExtractedRelation:
    """Represents a relation extracted from text"""
    source_entity: ExtractedEntity
    target_entity: ExtractedEntity
    relation_type: str
    confidence: float = 1.0
    context: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExtractionResult:
    """Complete extraction result from text"""
    text: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    keywords: List[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.metadata is None:
            self.metadata = {}

class BaseEntityExtractor(ABC):
    """Abstract base class for entity extractors"""
    
    @abstractmethod
    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text"""
        pass
    
    @abstractmethod
    async def extract_and_link_entities(
        self, 
        text: str, 
        existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExtractedEntity]:
        """Extract entities and link to existing knowledge base"""
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[str]:
        """Get list of supported entity types"""
        pass

class BaseRelationExtractor(ABC):
    """Abstract base class for relation extractors"""
    
    @abstractmethod
    async def extract_relations(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations between entities in text"""
        pass
    
    @abstractmethod
    def get_supported_relation_types(self) -> List[str]:
        """Get list of supported relation types"""
        pass

class BaseNLPPipeline(ABC):
    """Abstract base class for complete NLP processing pipeline"""
    
    @abstractmethod
    async def process_text(self, text: str) -> ExtractionResult:
        """Process text through complete NLP pipeline"""
        pass
    
    @abstractmethod
    async def process_document(
        self, 
        document_content: str, 
        document_metadata: Dict[str, Any]
    ) -> ExtractionResult:
        """Process a complete document"""
        pass
```

**Configuration Management**:
```python
# src/morag_nlp/config.py
from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
import os

class NLPConfig(BaseSettings):
    """Configuration for NLP processing"""
    
    # Model configurations
    spacy_model: str = "en_core_web_lg"
    transformer_model: str = "bert-base-uncased"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # Processing settings
    max_text_length: int = 10000
    batch_size: int = 32
    confidence_threshold: float = 0.5
    
    # Entity extraction settings
    entity_types: List[str] = Field(default_factory=lambda: [
        "PERSON", "ORG", "GPE", "PRODUCT", "EVENT", 
        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
        "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
    ])
    
    # Relation extraction settings
    relation_types: List[str] = Field(default_factory=lambda: [
        "PART_OF", "LOCATED_IN", "WORKS_FOR", "CREATED_BY",
        "RELATED_TO", "CAUSES", "ENABLES", "REQUIRES",
        "SIMILAR_TO", "OPPOSITE_OF", "TEMPORAL_BEFORE", "TEMPORAL_AFTER"
    ])
    
    # Model storage
    model_cache_dir: str = Field(default_factory=lambda: 
        os.path.join(os.getcwd(), "models", "nlp")
    )
    
    # Performance settings
    use_gpu: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Custom entity patterns
    custom_entity_patterns: Dict[str, List[str]] = Field(default_factory=dict)
    
    class Config:
        env_prefix = "MORAG_NLP_"
        case_sensitive = False

class ModelConfig(BaseSettings):
    """Configuration for specific models"""
    
    # spaCy configuration
    spacy_config: Dict[str, Any] = Field(default_factory=lambda: {
        "disable": [],
        "enable": ["ner", "tagger", "parser", "lemmatizer"],
        "max_length": 1000000
    })
    
    # Transformer configuration
    transformer_config: Dict[str, Any] = Field(default_factory=lambda: {
        "max_length": 512,
        "truncation": True,
        "padding": True,
        "return_tensors": "pt"
    })
    
    # Training configuration
    training_config: Dict[str, Any] = Field(default_factory=lambda: {
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 16,
        "warmup_steps": 500,
        "weight_decay": 0.01
    })
    
    class Config:
        env_prefix = "MORAG_MODEL_"
```

**Model Management Utilities**:
```python
# src/morag_nlp/utils/model_utils.py
import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import spacy
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

class ModelManager:
    """Manages NLP model loading and caching"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, Any] = {}
    
    async def load_spacy_model(self, model_name: str) -> spacy.Language:
        """Load spaCy model with caching"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        try:
            # Try to load the model
            nlp = spacy.load(model_name)
        except OSError:
            # Model not found, try to download
            print(f"Downloading spaCy model: {model_name}")
            os.system(f"python -m spacy download {model_name}")
            nlp = spacy.load(model_name)
        
        self._loaded_models[model_name] = nlp
        return nlp
    
    async def load_transformer_model(
        self, 
        model_name: str, 
        cache_dir: Optional[str] = None
    ) -> tuple:
        """Load transformer model and tokenizer"""
        cache_key = f"transformer_{model_name}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        model_cache_dir = cache_dir or str(self.cache_dir / "transformers")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=model_cache_dir
        )
        model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=model_cache_dir
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        result = (tokenizer, model)
        self._loaded_models[cache_key] = result
        return result
    
    async def load_sentence_transformer(
        self, 
        model_name: str
    ) -> SentenceTransformer:
        """Load sentence transformer model"""
        cache_key = f"sentence_transformer_{model_name}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        model_cache_dir = str(self.cache_dir / "sentence_transformers")
        model = SentenceTransformer(model_name, cache_folder=model_cache_dir)
        
        self._loaded_models[cache_key] = model
        return model
    
    def clear_cache(self):
        """Clear model cache"""
        self._loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self._loaded_models.keys()),
            "cache_dir": str(self.cache_dir),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory 
                         if torch.cuda.is_available() else None
        }

class ModelDownloader:
    """Handles downloading and setup of required models"""
    
    @staticmethod
    async def download_required_models(config: 'NLPConfig'):
        """Download all required models"""
        print("Downloading required NLP models...")
        
        # Download spaCy model
        try:
            spacy.load(config.spacy_model)
            print(f"✓ spaCy model {config.spacy_model} already available")
        except OSError:
            print(f"Downloading spaCy model: {config.spacy_model}")
            os.system(f"python -m spacy download {config.spacy_model}")
        
        # Pre-load transformer models to cache
        manager = ModelManager(config.model_cache_dir)
        
        print(f"Caching transformer model: {config.transformer_model}")
        await manager.load_transformer_model(config.transformer_model)
        
        print(f"Caching sentence transformer: {config.sentence_transformer_model}")
        await manager.load_sentence_transformer(config.sentence_transformer_model)
        
        print("All models downloaded and cached successfully!")
```

#### Deliverables
- [ ] NLP package structure
- [ ] Base interfaces for extractors
- [ ] Configuration management
- [ ] Model management utilities
- [ ] Dependency configuration

---

### Task 1.3.2: Basic Entity Recognition
**Priority**: Critical  
**Estimated Time**: 5-6 days  
**Dependencies**: 1.3.1

#### Implementation Steps

1. **spaCy Integration**
   - Implement spaCy-based entity extractor
   - Handle custom entity types
   - Optimize for performance

2. **Transformer-Based Extraction**
   - Implement BERT/RoBERTa entity recognition
   - Fine-tuning capabilities
   - Ensemble methods

3. **Entity Normalization and Linking**
   - Text cleaning and standardization
   - Alias resolution
   - Confidence scoring

#### Code Examples

**spaCy Entity Extractor**:
```python
# src/morag_nlp/extractors/entity_extractor.py
import asyncio
from typing import List, Dict, Any, Optional
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from .base import BaseEntityExtractor, ExtractedEntity
from ..utils.model_utils import ModelManager
from ..config import NLPConfig

class SpacyEntityExtractor(BaseEntityExtractor):
    """spaCy-based entity extractor"""
    
    def __init__(self, config: NLPConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.nlp: Optional[spacy.Language] = None
        self.matcher: Optional[Matcher] = None
        self._custom_patterns: Dict[str, List[Dict]] = {}
    
    async def initialize(self):
        """Initialize the spaCy model and custom patterns"""
        self.nlp = await self.model_manager.load_spacy_model(self.config.spacy_model)
        self.matcher = Matcher(self.nlp.vocab)
        await self._setup_custom_patterns()
    
    async def _setup_custom_patterns(self):
        """Setup custom entity patterns"""
        for entity_type, patterns in self.config.custom_entity_patterns.items():
            pattern_dicts = []
            for pattern in patterns:
                # Convert string patterns to spaCy pattern format
                pattern_dict = [{"LOWER": token.lower()} for token in pattern.split()]
                pattern_dicts.append(pattern_dict)
            
            self.matcher.add(entity_type, pattern_dicts)
            self._custom_patterns[entity_type] = pattern_dicts
    
    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            await self.initialize()
        
        # Process text with spaCy
        doc = self.nlp(text)
        entities = []
        
        # Extract standard NER entities
        for ent in doc.ents:
            if ent.label_ in self.config.entity_types:
                entity = ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=self._calculate_confidence(ent),
                    metadata={
                        "spacy_label": ent.label_,
                        "lemma": ent.lemma_,
                        "pos": ent.root.pos_,
                        "dependency": ent.root.dep_
                    }
                )
                entities.append(entity)
        
        # Extract custom pattern matches
        custom_entities = await self._extract_custom_entities(doc)
        entities.extend(custom_entities)
        
        # Remove duplicates and overlaps
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    async def _extract_custom_entities(self, doc: Doc) -> List[ExtractedEntity]:
        """Extract entities using custom patterns"""
        entities = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            entity_type = self.nlp.vocab.strings[match_id]
            
            entity = ExtractedEntity(
                text=span.text,
                label=entity_type,
                start=span.start_char,
                end=span.end_char,
                confidence=0.8,  # Custom patterns get lower confidence
                metadata={
                    "extraction_method": "custom_pattern",
                    "pattern_type": entity_type
                }
            )
            entities.append(entity)
        
        return entities
    
    def _calculate_confidence(self, ent: Span) -> float:
        """Calculate confidence score for spaCy entity"""
        # Base confidence from spaCy (if available)
        base_confidence = getattr(ent, 'confidence', 0.8)
        
        # Adjust based on entity characteristics
        length_factor = min(len(ent.text) / 20, 1.0)  # Longer entities more confident
        pos_factor = 1.0 if ent.root.pos_ in ['NOUN', 'PROPN'] else 0.9
        
        return min(base_confidence * length_factor * pos_factor, 1.0)
    
    def _remove_overlapping_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping the one with higher confidence"""
        # Sort by start position
        entities.sort(key=lambda e: e.start)
        
        filtered_entities = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in filtered_entities:
                if (entity.start < existing.end and entity.end > existing.start):
                    # There's an overlap
                    if entity.confidence > existing.confidence:
                        # Replace existing with current
                        filtered_entities.remove(existing)
                        filtered_entities.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    async def extract_and_link_entities(
        self, 
        text: str, 
        existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExtractedEntity]:
        """Extract entities and link to existing knowledge base"""
        entities = await self.extract_entities(text)
        
        if existing_entities:
            # Simple entity linking based on text similarity
            for entity in entities:
                best_match = self._find_best_entity_match(entity, existing_entities)
                if best_match:
                    entity.metadata["linked_entity_id"] = best_match["id"]
                    entity.metadata["link_confidence"] = best_match["similarity"]
        
        return entities
    
    def _find_best_entity_match(
        self, 
        entity: ExtractedEntity, 
        existing_entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find best matching entity in existing knowledge base"""
        best_match = None
        best_similarity = 0.0
        
        for existing in existing_entities:
            # Simple text similarity (can be improved with embeddings)
            similarity = self._calculate_text_similarity(
                entity.text.lower(), 
                existing.get("name", "").lower()
            )
            
            # Check aliases
            for alias in existing.get("aliases", []):
                alias_similarity = self._calculate_text_similarity(
                    entity.text.lower(), 
                    alias.lower()
                )
                similarity = max(similarity, alias_similarity)
            
            if similarity > best_similarity and similarity > 0.8:
                best_similarity = similarity
                best_match = {
                    "id": existing["id"],
                    "similarity": similarity
                }
        
        return best_match
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        if text1 == text2:
            return 1.0
        
        # Simple Jaccard similarity
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def get_supported_entity_types(self) -> List[str]:
        """Get list of supported entity types"""
        return self.config.entity_types + list(self._custom_patterns.keys())
```

**Transformer-Based Entity Extractor**:
```python
# src/morag_nlp/extractors/transformer_entity_extractor.py
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import numpy as np
from .base import BaseEntityExtractor, ExtractedEntity
from ..utils.model_utils import ModelManager
from ..config import NLPConfig

class TransformerEntityExtractor(BaseEntityExtractor):
    """Transformer-based entity extractor using BERT/RoBERTa"""
    
    def __init__(self, config: NLPConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForTokenClassification] = None
        self.ner_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
    
    async def initialize(self, model_name: Optional[str] = None):
        """Initialize the transformer model"""
        model_name = model_name or self.config.transformer_model
        
        # Load pre-trained NER model
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=0 if self.device == "cuda" else -1
        )
    
    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using transformer model"""
        if not self.ner_pipeline:
            await self.initialize()
        
        # Truncate text if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        # Run NER pipeline
        ner_results = self.ner_pipeline(text)
        
        entities = []
        for result in ner_results:
            # Filter by confidence threshold
            if result['score'] < self.config.confidence_threshold:
                continue
            
            # Map transformer labels to our entity types
            entity_type = self._map_transformer_label(result['entity_group'])
            if entity_type not in self.config.entity_types:
                continue
            
            entity = ExtractedEntity(
                text=result['word'],
                label=entity_type,
                start=result['start'],
                end=result['end'],
                confidence=result['score'],
                metadata={
                    "transformer_label": result['entity_group'],
                    "extraction_method": "transformer",
                    "model_name": self.config.transformer_model
                }
            )
            entities.append(entity)
        
        return entities
    
    def _map_transformer_label(self, transformer_label: str) -> str:
        """Map transformer NER labels to our standard entity types"""
        label_mapping = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "ORG": "ORG",
            "ORGANIZATION": "ORG",
            "LOC": "GPE",
            "LOCATION": "GPE",
            "MISC": "PRODUCT",
            "MISCELLANEOUS": "PRODUCT"
        }
        
        # Remove B- and I- prefixes if present
        clean_label = transformer_label.replace("B-", "").replace("I-", "")
        
        return label_mapping.get(clean_label, clean_label)
    
    async def extract_and_link_entities(
        self, 
        text: str, 
        existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExtractedEntity]:
        """Extract entities and link to existing knowledge base"""
        entities = await self.extract_entities(text)
        
        if existing_entities:
            # Use embeddings for better entity linking
            entities = await self._link_entities_with_embeddings(entities, existing_entities)
        
        return entities
    
    async def _link_entities_with_embeddings(
        self, 
        entities: List[ExtractedEntity], 
        existing_entities: List[Dict[str, Any]]
    ) -> List[ExtractedEntity]:
        """Link entities using embedding similarity"""
        # This would use sentence transformers for better similarity
        # For now, using simple text matching
        for entity in entities:
            best_match = self._find_best_entity_match(entity, existing_entities)
            if best_match:
                entity.metadata["linked_entity_id"] = best_match["id"]
                entity.metadata["link_confidence"] = best_match["similarity"]
        
        return entities
    
    def _find_best_entity_match(
        self, 
        entity: ExtractedEntity, 
        existing_entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find best matching entity (placeholder implementation)"""
        # Simplified implementation - would use embeddings in practice
        best_match = None
        best_similarity = 0.0
        
        for existing in existing_entities:
            if existing.get("type") == entity.label:
                # Simple text similarity
                similarity = self._jaccard_similarity(
                    entity.text.lower(), 
                    existing.get("name", "").lower()
                )
                
                if similarity > best_similarity and similarity > 0.7:
                    best_similarity = similarity
                    best_match = {
                        "id": existing["id"],
                        "similarity": similarity
                    }
        
        return best_match
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def get_supported_entity_types(self) -> List[str]:
        """Get list of supported entity types"""
        return self.config.entity_types
```

**Complete NLP Pipeline**:
```python
# src/morag_nlp/processors/pipeline.py
import asyncio
import time
from typing import List, Dict, Any, Optional
from ..extractors.base import BaseNLPPipeline, ExtractionResult
from ..extractors.entity_extractor import SpacyEntityExtractor
from ..extractors.transformer_entity_extractor import TransformerEntityExtractor
from ..utils.model_utils import ModelManager
from ..config import NLPConfig

class ComprehensiveNLPPipeline(BaseNLPPipeline):
    """Complete NLP processing pipeline combining multiple extractors"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.model_manager = ModelManager(config.model_cache_dir)
        self.spacy_extractor = SpacyEntityExtractor(config, self.model_manager)
        self.transformer_extractor = TransformerEntityExtractor(config, self.model_manager)
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        await self.spacy_extractor.initialize()
        await self.transformer_extractor.initialize()
        self._initialized = True
    
    async def process_text(self, text: str) -> ExtractionResult:
        """Process text through complete NLP pipeline"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Extract entities using both methods
        spacy_entities = await self.spacy_extractor.extract_entities(text)
        transformer_entities = await self.transformer_extractor.extract_entities(text)
        
        # Combine and deduplicate entities
        combined_entities = self._combine_entities(spacy_entities, transformer_entities)
        
        # Extract relations (placeholder - will be implemented in Task 2.1)
        relations = []  # TODO: Implement relation extraction
        
        # Extract keywords (simple implementation)
        keywords = await self._extract_keywords(text)
        
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            text=text,
            entities=combined_entities,
            relations=relations,
            keywords=keywords,
            processing_time=processing_time,
            metadata={
                "spacy_entities_count": len(spacy_entities),
                "transformer_entities_count": len(transformer_entities),
                "combined_entities_count": len(combined_entities)
            }
        )
    
    async def process_document(
        self, 
        document_content: str, 
        document_metadata: Dict[str, Any]
    ) -> ExtractionResult:
        """Process a complete document"""
        # For large documents, we might want to process in chunks
        if len(document_content) > self.config.max_text_length:
            return await self._process_large_document(document_content, document_metadata)
        else:
            result = await self.process_text(document_content)
            result.metadata.update(document_metadata)
            return result
    
    async def _process_large_document(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> ExtractionResult:
        """Process large documents in chunks"""
        chunk_size = self.config.max_text_length
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        all_entities = []
        all_relations = []
        all_keywords = []
        total_processing_time = 0.0
        
        for i, chunk in enumerate(chunks):
            chunk_result = await self.process_text(chunk)
            
            # Adjust entity positions for chunk offset
            chunk_offset = i * chunk_size
            for entity in chunk_result.entities:
                entity.start += chunk_offset
                entity.end += chunk_offset
            
            all_entities.extend(chunk_result.entities)
            all_relations.extend(chunk_result.relations)
            all_keywords.extend(chunk_result.keywords)
            total_processing_time += chunk_result.processing_time
        
        # Deduplicate keywords
        unique_keywords = list(set(all_keywords))
        
        return ExtractionResult(
            text=content,
            entities=all_entities,
            relations=all_relations,
            keywords=unique_keywords,
            processing_time=total_processing_time,
            metadata={
                **metadata,
                "chunks_processed": len(chunks),
                "total_entities": len(all_entities)
            }
        )
    
    def _combine_entities(self, spacy_entities, transformer_entities):
        """Combine entities from different extractors, handling duplicates"""
        combined = []
        
        # Add all spaCy entities
        combined.extend(spacy_entities)
        
        # Add transformer entities that don't overlap significantly
        for t_entity in transformer_entities:
            overlaps = False
            for s_entity in spacy_entities:
                if self._entities_overlap(t_entity, s_entity):
                    overlaps = True
                    # If transformer has higher confidence, replace spaCy entity
                    if t_entity.confidence > s_entity.confidence:
                        combined.remove(s_entity)
                        combined.append(t_entity)
                    break
            
            if not overlaps:
                combined.append(t_entity)
        
        return combined
    
    def _entities_overlap(self, entity1, entity2, threshold=0.5):
        """Check if two entities overlap significantly"""
        start1, end1 = entity1.start, entity1.end
        start2, end2 = entity2.start, entity2.end
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return False
        
        overlap_length = overlap_end - overlap_start
        min_length = min(end1 - start1, end2 - start2)
        
        return (overlap_length / min_length) > threshold
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction (can be improved)"""
        # Simple implementation using spaCy
        if not self.spacy_extractor.nlp:
            await self.spacy_extractor.initialize()
        
        doc = self.spacy_extractor.nlp(text)
        keywords = []
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Return unique keywords, sorted by frequency
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]
```

#### Deliverables
- [ ] Working entity extraction with spaCy
- [ ] Transformer-based entity recognition
- [ ] Entity normalization and linking
- [ ] Complete NLP processing pipeline
- [ ] Performance optimization

## Testing Requirements

### Unit Tests
```python
# tests/test_extractors.py
import pytest
import asyncio
from morag_nlp.extractors.entity_extractor import SpacyEntityExtractor
from morag_nlp.extractors.transformer_entity_extractor import TransformerEntityExtractor
from morag_nlp.config import NLPConfig
from morag_nlp.utils.model_utils import ModelManager

@pytest.mark.asyncio
async def test_spacy_entity_extraction():
    """Test spaCy entity extraction"""
    config = NLPConfig()
    model_manager = ModelManager(config.model_cache_dir)
    extractor = SpacyEntityExtractor(config, model_manager)
    
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = await extractor.extract_entities(text)
    
    assert len(entities) > 0
    entity_texts = [e.text for e in entities]
    assert "Apple Inc." in entity_texts or "Apple" in entity_texts
    assert "Steve Jobs" in entity_texts

@pytest.mark.asyncio
async def test_transformer_entity_extraction():
    """Test transformer entity extraction"""
    config = NLPConfig()
    model_manager = ModelManager(config.model_cache_dir)
    extractor = TransformerEntityExtractor(config, model_manager)
    
    text = "Microsoft was founded by Bill Gates in Seattle."
    entities = await extractor.extract_entities(text)
    
    assert len(entities) > 0
    # Check that we get reasonable confidence scores
    for entity in entities:
        assert 0.0 <= entity.confidence <= 1.0

@pytest.mark.asyncio
async def test_entity_linking():
    """Test entity linking functionality"""
    config = NLPConfig()
    model_manager = ModelManager(config.model_cache_dir)
    extractor = SpacyEntityExtractor(config, model_manager)
    
    text = "Apple makes great computers."
    existing_entities = [
        {"id": "apple_inc", "name": "Apple Inc.", "type": "ORG", "aliases": ["Apple"]}
    ]
    
    entities = await extractor.extract_and_link_entities(text, existing_entities)
    
    # Check if Apple was linked
    apple_entities = [e for e in entities if "apple" in e.text.lower()]
    if apple_entities:
        assert "linked_entity_id" in apple_entities[0].metadata
```

### Integration Tests
```python
# tests/test_pipeline.py
import pytest
import asyncio
from morag_nlp.processors.pipeline import ComprehensiveNLPPipeline
from morag_nlp.config import NLPConfig

@pytest.mark.asyncio
async def test_complete_pipeline():
    """Test complete NLP processing pipeline"""
    config = NLPConfig()
    pipeline = ComprehensiveNLPPipeline(config)
    
    text = """
    Apple Inc. is an American multinational technology company headquartered 
    in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, 
    and Ronald Wayne in April 1976.
    """
    
    result = await pipeline.process_text(text)
    
    assert len(result.entities) > 0
    assert len(result.keywords) > 0
    assert result.processing_time > 0
    assert "Apple" in [e.text for e in result.entities] or "Apple Inc." in [e.text for e in result.entities]
```

## Success Criteria

- [ ] NLP package structure created and configured
- [ ] spaCy entity extractor working correctly
- [ ] Transformer-based entity extraction implemented
- [ ] Entity linking and normalization functional
- [ ] Complete NLP pipeline processing documents
- [ ] Custom entity patterns supported
- [ ] Performance optimized for real-time processing
- [ ] Unit tests achieving >85% coverage
- [ ] Integration tests passing
- [ ] Documentation complete

## Next Steps

After completing this task:
1. Proceed to [Task 2.1: Relation Extraction System](./task-2.1-relation-extraction.md)
2. Begin implementing rule-based and ML-based relation extraction
3. Start planning integration with graph construction pipeline

---

**Status**: ⏳ Not Started  
**Assignee**: TBD  
**Last Updated**: December 2024