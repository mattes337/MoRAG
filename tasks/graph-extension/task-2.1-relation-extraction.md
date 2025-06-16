# Task 2.1: Relation Extraction System

**Phase**: 2 - Core Graph Features  
**Priority**: Critical  
**Total Estimated Time**: 10-12 days  
**Dependencies**: Task 1.3 (NLP Pipeline Foundation)

## Overview

This task implements a comprehensive relation extraction system that can identify and extract semantic relationships between entities in text. The system combines rule-based approaches with machine learning models to achieve high accuracy and coverage.

## Subtasks

### Task 2.1.1: Rule-Based Relation Extraction
**Priority**: Critical  
**Estimated Time**: 4-5 days  
**Dependencies**: Task 1.3.1, 1.3.2

#### Implementation Steps

1. **Pattern Matching System**
   - Define relation patterns using spaCy's pattern matching
   - Create templates for common relation types
   - Implement confidence scoring for pattern matches

2. **Dependency Parsing**
   - Use syntactic dependencies to identify relations
   - Extract subject-verb-object triplets
   - Handle complex sentence structures

3. **Linguistic Rules**
   - Implement domain-specific rules
   - Handle negation and modality
   - Process temporal and spatial relations

#### Code Examples

**Rule-Based Relation Extractor**:
```python
# src/morag_nlp/extractors/relation_extractor.py
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import spacy
from spacy.matcher import DependencyMatcher, Matcher
from spacy.tokens import Doc, Token, Span
import re
from .base import BaseRelationExtractor, ExtractedRelation, ExtractedEntity
from ..config import NLPConfig
from ..utils.model_utils import ModelManager

class RuleBasedRelationExtractor(BaseRelationExtractor):
    """Rule-based relation extractor using patterns and dependency parsing"""
    
    def __init__(self, config: NLPConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.nlp: Optional[spacy.Language] = None
        self.dependency_matcher: Optional[DependencyMatcher] = None
        self.pattern_matcher: Optional[Matcher] = None
        self.relation_patterns: Dict[str, List[Dict]] = {}
        self.dependency_patterns: Dict[str, List[Dict]] = {}
    
    async def initialize(self):
        """Initialize the spaCy model and relation patterns"""
        self.nlp = await self.model_manager.load_spacy_model(self.config.spacy_model)
        self.dependency_matcher = DependencyMatcher(self.nlp.vocab)
        self.pattern_matcher = Matcher(self.nlp.vocab)
        await self._setup_relation_patterns()
        await self._setup_dependency_patterns()
    
    async def _setup_relation_patterns(self):
        """Setup text-based relation patterns"""
        # Define common relation patterns
        patterns = {
            "WORKS_FOR": [
                [{"LOWER": {"IN": ["works", "work"]}}, {"LOWER": "for"}],
                [{"LOWER": "employed"}, {"LOWER": "by"}],
                [{"LOWER": {"IN": ["ceo", "president", "director"]}}, {"LOWER": "of"}],
                [{"LOWER": {"IN": ["founded", "started"]}}, {"LOWER": "by"}]
            ],
            "LOCATED_IN": [
                [{"LOWER": "in"}, {"ENT_TYPE": "GPE"}],
                [{"LOWER": "located"}, {"LOWER": "in"}],
                [{"LOWER": "based"}, {"LOWER": "in"}],
                [{"LOWER": "headquartered"}, {"LOWER": "in"}]
            ],
            "PART_OF": [
                [{"LOWER": "part"}, {"LOWER": "of"}],
                [{"LOWER": "division"}, {"LOWER": "of"}],
                [{"LOWER": "subsidiary"}, {"LOWER": "of"}],
                [{"LOWER": "belongs"}, {"LOWER": "to"}]
            ],
            "CREATED_BY": [
                [{"LOWER": "created"}, {"LOWER": "by"}],
                [{"LOWER": "developed"}, {"LOWER": "by"}],
                [{"LOWER": "invented"}, {"LOWER": "by"}],
                [{"LOWER": "designed"}, {"LOWER": "by"}]
            ],
            "TEMPORAL_BEFORE": [
                [{"LOWER": "before"}],
                [{"LOWER": "prior"}, {"LOWER": "to"}],
                [{"LOWER": "earlier"}, {"LOWER": "than"}]
            ],
            "TEMPORAL_AFTER": [
                [{"LOWER": "after"}],
                [{"LOWER": "following"}],
                [{"LOWER": "later"}, {"LOWER": "than"}]
            ],
            "CAUSES": [
                [{"LOWER": "causes"}],
                [{"LOWER": "leads"}, {"LOWER": "to"}],
                [{"LOWER": "results"}, {"LOWER": "in"}],
                [{"LOWER": "triggers"}]
            ],
            "ENABLES": [
                [{"LOWER": "enables"}],
                [{"LOWER": "allows"}],
                [{"LOWER": "facilitates"}],
                [{"LOWER": "supports"}]
            ]
        }
        
        for relation_type, pattern_list in patterns.items():
            self.pattern_matcher.add(relation_type, pattern_list)
            self.relation_patterns[relation_type] = pattern_list
    
    async def _setup_dependency_patterns(self):
        """Setup dependency-based relation patterns"""
        # Define dependency patterns for relation extraction
        dependency_patterns = {
            "WORKS_FOR": [
                # Pattern: "John works for Apple"
                [
                    {"RIGHT_ID": "anchor_verb", "RIGHT_ATTRS": {"LEMMA": "work"}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"LEMMA": "for"}},
                    {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "pobj"}}
                ],
                # Pattern: "Apple employs John"
                [
                    {"RIGHT_ID": "anchor_verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["employ", "hire"]}}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}}
                ]
            ],
            "LOCATED_IN": [
                # Pattern: "Apple is located in California"
                [
                    {"RIGHT_ID": "anchor_verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["locate", "base"]}}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubjpass"}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"LEMMA": "in"}},
                    {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "pobj"}}
                ]
            ],
            "CREATED_BY": [
                # Pattern: "iPhone was created by Apple"
                [
                    {"RIGHT_ID": "anchor_verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["create", "develop", "invent"]}}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubjpass"}},
                    {"LEFT_ID": "anchor_verb", "REL_OP": ">", "RIGHT_ID": "agent", "RIGHT_ATTRS": {"DEP": "agent"}},
                    {"LEFT_ID": "agent", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "pobj"}}
                ]
            ]
        }
        
        for relation_type, patterns in dependency_patterns.items():
            for i, pattern in enumerate(patterns):
                pattern_id = f"{relation_type}_{i}"
                self.dependency_matcher.add(pattern_id, [pattern])
            self.dependency_patterns[relation_type] = patterns
    
    async def extract_relations(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations using rule-based approaches"""
        if not self.nlp:
            await self.initialize()
        
        doc = self.nlp(text)
        relations = []
        
        # Extract relations using dependency patterns
        dependency_relations = await self._extract_dependency_relations(doc, entities)
        relations.extend(dependency_relations)
        
        # Extract relations using text patterns
        pattern_relations = await self._extract_pattern_relations(doc, entities)
        relations.extend(pattern_relations)
        
        # Extract subject-verb-object triplets
        svo_relations = await self._extract_svo_relations(doc, entities)
        relations.extend(svo_relations)
        
        # Remove duplicates and filter by confidence
        relations = self._deduplicate_relations(relations)
        relations = [r for r in relations if r.confidence >= self.config.confidence_threshold]
        
        return relations
    
    async def _extract_dependency_relations(
        self, 
        doc: Doc, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations using dependency patterns"""
        relations = []
        matches = self.dependency_matcher(doc)
        
        for match_id, token_ids in matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            relation_type = pattern_name.split('_')[0] + '_' + pattern_name.split('_')[1]
            
            # Get the matched tokens
            matched_tokens = {}
            for token_id in token_ids:
                token = doc[token_id]
                # Map token roles based on pattern
                if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
                    matched_tokens["subject"] = token
                elif token.dep_ in ["dobj", "pobj"]:
                    matched_tokens["object"] = token
                elif token.pos_ == "VERB":
                    matched_tokens["verb"] = token
            
            # Find corresponding entities
            subject_entity = self._find_entity_for_token(matched_tokens.get("subject"), entities)
            object_entity = self._find_entity_for_token(matched_tokens.get("object"), entities)
            
            if subject_entity and object_entity:
                relation = ExtractedRelation(
                    source_entity=subject_entity,
                    target_entity=object_entity,
                    relation_type=relation_type,
                    confidence=0.8,  # Rule-based gets high confidence
                    context=self._get_relation_context(doc, matched_tokens),
                    metadata={
                        "extraction_method": "dependency_pattern",
                        "pattern_name": pattern_name,
                        "verb": matched_tokens.get("verb", {}).text if matched_tokens.get("verb") else None
                    }
                )
                relations.append(relation)
        
        return relations
    
    async def _extract_pattern_relations(
        self, 
        doc: Doc, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations using text patterns"""
        relations = []
        matches = self.pattern_matcher(doc)
        
        for match_id, start, end in matches:
            relation_type = self.nlp.vocab.strings[match_id]
            pattern_span = doc[start:end]
            
            # Find entities around the pattern
            nearby_entities = self._find_nearby_entities(pattern_span, entities, window=10)
            
            if len(nearby_entities) >= 2:
                # Create relations between nearby entities
                for i in range(len(nearby_entities) - 1):
                    source_entity = nearby_entities[i]
                    target_entity = nearby_entities[i + 1]
                    
                    # Determine direction based on relation type and position
                    if self._should_reverse_relation(relation_type, source_entity, target_entity):
                        source_entity, target_entity = target_entity, source_entity
                    
                    relation = ExtractedRelation(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relation_type=relation_type,
                        confidence=0.7,  # Pattern matching gets medium confidence
                        context=pattern_span.text,
                        metadata={
                            "extraction_method": "text_pattern",
                            "pattern_text": pattern_span.text
                        }
                    )
                    relations.append(relation)
        
        return relations
    
    async def _extract_svo_relations(
        self, 
        doc: Doc, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract subject-verb-object relations"""
        relations = []
        
        for sent in doc.sents:
            # Find the main verb
            main_verb = None
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    main_verb = token
                    break
            
            if not main_verb:
                continue
            
            # Find subject and object
            subject = None
            obj = None
            
            for child in main_verb.children:
                if child.dep_ == "nsubj" or child.dep_ == "nsubjpass":
                    subject = child
                elif child.dep_ == "dobj" or child.dep_ == "pobj":
                    obj = child
            
            if subject and obj:
                # Find corresponding entities
                subject_entity = self._find_entity_for_token(subject, entities)
                object_entity = self._find_entity_for_token(obj, entities)
                
                if subject_entity and object_entity:
                    # Determine relation type from verb
                    relation_type = self._verb_to_relation_type(main_verb.lemma_)
                    
                    if relation_type:
                        relation = ExtractedRelation(
                            source_entity=subject_entity,
                            target_entity=object_entity,
                            relation_type=relation_type,
                            confidence=0.6,  # SVO gets lower confidence
                            context=sent.text,
                            metadata={
                                "extraction_method": "svo_triplet",
                                "verb": main_verb.text,
                                "verb_lemma": main_verb.lemma_
                            }
                        )
                        relations.append(relation)
        
        return relations
    
    def _find_entity_for_token(
        self, 
        token: Optional[Token], 
        entities: List[ExtractedEntity]
    ) -> Optional[ExtractedEntity]:
        """Find entity that contains or matches the given token"""
        if not token:
            return None
        
        # Expand to noun phrase if possible
        if token.pos_ in ["NOUN", "PROPN"]:
            # Get the full noun phrase
            noun_phrase = None
            for chunk in token.doc.noun_chunks:
                if token in chunk:
                    noun_phrase = chunk
                    break
            
            if noun_phrase:
                # Find entity that overlaps with noun phrase
                for entity in entities:
                    if (entity.start <= noun_phrase.start_char < entity.end or
                        entity.start < noun_phrase.end_char <= entity.end):
                        return entity
        
        # Fallback: find entity that contains the token
        for entity in entities:
            if entity.start <= token.idx < entity.end:
                return entity
        
        return None
    
    def _find_nearby_entities(
        self, 
        span: Span, 
        entities: List[ExtractedEntity], 
        window: int = 10
    ) -> List[ExtractedEntity]:
        """Find entities near the given span"""
        nearby = []
        span_start = span.start_char
        span_end = span.end_char
        
        for entity in entities:
            # Check if entity is within window of the span
            distance = min(
                abs(entity.start - span_end),
                abs(entity.end - span_start)
            )
            
            if distance <= window * 10:  # Approximate character distance
                nearby.append(entity)
        
        # Sort by distance from span
        nearby.sort(key=lambda e: min(
            abs(e.start - span_end),
            abs(e.end - span_start)
        ))
        
        return nearby
    
    def _should_reverse_relation(
        self, 
        relation_type: str, 
        source_entity: ExtractedEntity, 
        target_entity: ExtractedEntity
    ) -> bool:
        """Determine if relation direction should be reversed"""
        # Simple heuristics for relation direction
        if relation_type == "WORKS_FOR":
            # Person works for organization
            return (source_entity.label == "ORG" and target_entity.label == "PERSON")
        elif relation_type == "LOCATED_IN":
            # Entity located in place
            return (source_entity.label == "GPE" and target_entity.label in ["ORG", "PERSON"])
        elif relation_type == "CREATED_BY":
            # Product created by person/org
            return (source_entity.label in ["PERSON", "ORG"] and 
                   target_entity.label == "PRODUCT")
        
        return False
    
    def _verb_to_relation_type(self, verb_lemma: str) -> Optional[str]:
        """Map verb lemmas to relation types"""
        verb_mapping = {
            "work": "WORKS_FOR",
            "employ": "WORKS_FOR",
            "hire": "WORKS_FOR",
            "create": "CREATED_BY",
            "develop": "CREATED_BY",
            "invent": "CREATED_BY",
            "design": "CREATED_BY",
            "locate": "LOCATED_IN",
            "base": "LOCATED_IN",
            "headquarter": "LOCATED_IN",
            "cause": "CAUSES",
            "trigger": "CAUSES",
            "enable": "ENABLES",
            "allow": "ENABLES",
            "support": "ENABLES",
            "require": "REQUIRES",
            "need": "REQUIRES"
        }
        
        return verb_mapping.get(verb_lemma)
    
    def _get_relation_context(
        self, 
        doc: Doc, 
        matched_tokens: Dict[str, Token]
    ) -> str:
        """Get context around the relation"""
        if not matched_tokens:
            return ""
        
        # Find the sentence containing the relation
        for sent in doc.sents:
            if any(token in sent for token in matched_tokens.values() if token):
                return sent.text
        
        return ""
    
    def _deduplicate_relations(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """Remove duplicate relations"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # Create a key for deduplication
            key = (
                relation.source_entity.text.lower(),
                relation.target_entity.text.lower(),
                relation.relation_type
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # If we've seen this relation, keep the one with higher confidence
                for i, existing in enumerate(unique_relations):
                    existing_key = (
                        existing.source_entity.text.lower(),
                        existing.target_entity.text.lower(),
                        existing.relation_type
                    )
                    if existing_key == key and relation.confidence > existing.confidence:
                        unique_relations[i] = relation
                        break
        
        return unique_relations
    
    def get_supported_relation_types(self) -> List[str]:
        """Get list of supported relation types"""
        return self.config.relation_types
```

#### Deliverables
- [ ] Rule-based relation extractor
- [ ] Pattern matching system
- [ ] Dependency parsing implementation
- [ ] SVO triplet extraction
- [ ] Relation deduplication

---

### Task 2.1.2: ML-Based Relation Extraction
**Priority**: High  
**Estimated Time**: 6-7 days  
**Dependencies**: Task 2.1.1

#### Implementation Steps

1. **Transformer Model Integration**
   - Implement BERT-based relation classification
   - Fine-tuning pipeline for domain-specific relations
   - Ensemble methods combining multiple models

2. **Training Pipeline**
   - Data preparation and augmentation
   - Model training and validation
   - Hyperparameter optimization

3. **Hybrid Approach**
   - Combine rule-based and ML approaches
   - Confidence weighting and voting
   - Fallback mechanisms

#### Code Examples

**Transformer Relation Extractor**:
```python
# src/morag_nlp/extractors/ml_relation_extractor.py
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
import numpy as np
from sklearn.metrics import classification_report
from .base import BaseRelationExtractor, ExtractedRelation, ExtractedEntity
from ..config import NLPConfig, ModelConfig
from ..utils.model_utils import ModelManager

class TransformerRelationExtractor(BaseRelationExtractor):
    """Transformer-based relation extractor using BERT/RoBERTa"""
    
    def __init__(self, config: NLPConfig, model_manager: ModelManager):
        self.config = config
        self.model_config = ModelConfig()
        self.model_manager = model_manager
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.classifier_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        self.relation_to_id = {}
        self.id_to_relation = {}
    
    async def initialize(self, model_name: Optional[str] = None):
        """Initialize the transformer model"""
        model_name = model_name or self.config.transformer_model
        
        # Setup relation label mappings
        self._setup_relation_mappings()
        
        # Load or create model
        if await self._model_exists():
            await self._load_trained_model()
        else:
            await self._load_pretrained_model(model_name)
    
    def _setup_relation_mappings(self):
        """Setup mappings between relation types and IDs"""
        relations = self.config.relation_types + ["NO_RELATION"]
        self.relation_to_id = {rel: i for i, rel in enumerate(relations)}
        self.id_to_relation = {i: rel for rel, i in self.relation_to_id.items()}
    
    async def _model_exists(self) -> bool:
        """Check if trained model exists"""
        model_path = self.config.model_cache_dir + "/relation_classifier"
        try:
            # Try to load tokenizer to check if model exists
            AutoTokenizer.from_pretrained(model_path)
            return True
        except:
            return False
    
    async def _load_trained_model(self):
        """Load pre-trained relation classification model"""
        model_path = self.config.model_cache_dir + "/relation_classifier"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self.relation_to_id)
        )
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        self.classifier_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True
        )
    
    async def _load_pretrained_model(self, model_name: str):
        """Load pre-trained model for fine-tuning"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.relation_to_id)
        )
        
        if self.device == "cuda":
            self.model = self.model.cuda()
    
    async def extract_relations(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations using transformer model"""
        if not self.model:
            await self.initialize()
        
        relations = []
        
        # Generate entity pairs for classification
        entity_pairs = self._generate_entity_pairs(entities)
        
        for source_entity, target_entity in entity_pairs:
            # Create input text for classification
            input_text = self._create_classification_input(
                text, source_entity, target_entity
            )
            
            # Classify relation
            relation_type, confidence = await self._classify_relation(input_text)
            
            if relation_type != "NO_RELATION" and confidence >= self.config.confidence_threshold:
                relation = ExtractedRelation(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    relation_type=relation_type,
                    confidence=confidence,
                    context=self._extract_context(text, source_entity, target_entity),
                    metadata={
                        "extraction_method": "transformer",
                        "model_name": self.config.transformer_model,
                        "input_text": input_text
                    }
                )
                relations.append(relation)
        
        return relations
    
    def _generate_entity_pairs(
        self, 
        entities: List[ExtractedEntity]
    ) -> List[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Generate all possible entity pairs for relation classification"""
        pairs = []
        
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j:
                    source = entities[i]
                    target = entities[j]
                    
                    # Filter pairs based on entity types and distance
                    if self._is_valid_entity_pair(source, target):
                        pairs.append((source, target))
        
        return pairs
    
    def _is_valid_entity_pair(
        self, 
        source: ExtractedEntity, 
        target: ExtractedEntity
    ) -> bool:
        """Check if entity pair is valid for relation extraction"""
        # Skip if entities are too far apart
        distance = abs(source.start - target.start)
        if distance > 500:  # Character distance threshold
            return False
        
        # Skip if entities are the same
        if source.text.lower() == target.text.lower():
            return False
        
        # Skip certain entity type combinations that rarely have relations
        skip_combinations = [
            ("DATE", "DATE"),
            ("TIME", "TIME"),
            ("PERCENT", "PERCENT"),
            ("MONEY", "MONEY")
        ]
        
        if (source.label, target.label) in skip_combinations:
            return False
        
        return True
    
    def _create_classification_input(
        self, 
        text: str, 
        source_entity: ExtractedEntity, 
        target_entity: ExtractedEntity
    ) -> str:
        """Create input text for relation classification"""
        # Extract sentence containing both entities
        context = self._extract_context(text, source_entity, target_entity)
        
        # Mark entities in the text
        marked_text = context
        
        # Replace entity mentions with marked versions
        # (in reverse order to maintain positions)
        entities_sorted = sorted(
            [source_entity, target_entity], 
            key=lambda e: e.start, 
            reverse=True
        )
        
        for entity in entities_sorted:
            if entity == source_entity:
                marker = f"[E1]{entity.text}[/E1]"
            else:
                marker = f"[E2]{entity.text}[/E2]"
            
            # Adjust positions relative to context
            context_start = max(0, min(source_entity.start, target_entity.start) - 100)
            relative_start = entity.start - context_start
            relative_end = entity.end - context_start
            
            if 0 <= relative_start < len(marked_text):
                marked_text = (
                    marked_text[:relative_start] + 
                    marker + 
                    marked_text[relative_end:]
                )
        
        return marked_text
    
    def _extract_context(
        self, 
        text: str, 
        source_entity: ExtractedEntity, 
        target_entity: ExtractedEntity
    ) -> str:
        """Extract context around the entities"""
        # Find the span that includes both entities with some context
        start_pos = min(source_entity.start, target_entity.start)
        end_pos = max(source_entity.end, target_entity.end)
        
        # Add context window
        context_window = 100
        context_start = max(0, start_pos - context_window)
        context_end = min(len(text), end_pos + context_window)
        
        return text[context_start:context_end]
    
    async def _classify_relation(self, input_text: str) -> Tuple[str, float]:
        """Classify relation for the given input"""
        if self.classifier_pipeline:
            # Use pipeline for inference
            results = self.classifier_pipeline(input_text)
            
            # Find the prediction with highest score
            best_result = max(results, key=lambda x: x['score'])
            relation_id = int(best_result['label'].split('_')[-1])  # Extract ID from label
            relation_type = self.id_to_relation[relation_id]
            confidence = best_result['score']
            
            return relation_type, confidence
        else:
            # Manual inference
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                predicted_id = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_id].item()
                
                relation_type = self.id_to_relation[predicted_id]
                
                return relation_type, confidence
    
    async def train_model(
        self, 
        training_data: List[Dict[str, Any]], 
        validation_data: Optional[List[Dict[str, Any]]] = None
    ):
        """Train the relation classification model"""
        if not self.model:
            await self.initialize()
        
        # Prepare datasets
        train_dataset = self._prepare_dataset(training_data)
        val_dataset = self._prepare_dataset(validation_data) if validation_data else None
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.model_cache_dir + "/relation_classifier",
            num_train_epochs=self.model_config.training_config["num_epochs"],
            per_device_train_batch_size=self.model_config.training_config["batch_size"],
            per_device_eval_batch_size=self.model_config.training_config["batch_size"],
            warmup_steps=self.model_config.training_config["warmup_steps"],
            weight_decay=self.model_config.training_config["weight_decay"],
            learning_rate=self.model_config.training_config["learning_rate"],
            logging_dir=self.config.model_cache_dir + "/logs",
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics if val_dataset else None
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        print("Model training completed and saved!")
    
    def _prepare_dataset(self, data: List[Dict[str, Any]]):
        """Prepare dataset for training"""
        texts = []
        labels = []
        
        for item in data:
            texts.append(item["text"])
            labels.append(self.relation_to_id[item["relation"]])
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset
        class RelationDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return RelationDataset(encodings, labels)
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Convert IDs back to relation names for reporting
        pred_relations = [self.id_to_relation[p] for p in predictions]
        true_relations = [self.id_to_relation[l] for l in labels]
        
        report = classification_report(
            true_relations, pred_relations, output_dict=True
        )
        
        return {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        }
    
    def get_supported_relation_types(self) -> List[str]:
        """Get list of supported relation types"""
        return self.config.relation_types
```

**Hybrid Relation Extractor**:
```python
# src/morag_nlp/extractors/hybrid_relation_extractor.py
import asyncio
from typing import List, Dict, Any, Optional
from .base import BaseRelationExtractor, ExtractedRelation, ExtractedEntity
from .relation_extractor import RuleBasedRelationExtractor
from .ml_relation_extractor import TransformerRelationExtractor
from ..config import NLPConfig
from ..utils.model_utils import ModelManager

class HybridRelationExtractor(BaseRelationExtractor):
    """Hybrid relation extractor combining rule-based and ML approaches"""
    
    def __init__(self, config: NLPConfig, model_manager: ModelManager):
        self.config = config
        self.rule_extractor = RuleBasedRelationExtractor(config, model_manager)
        self.ml_extractor = TransformerRelationExtractor(config, model_manager)
        self.weights = {
            "rule_based": 0.6,
            "transformer": 0.4
        }
    
    async def initialize(self):
        """Initialize both extractors"""
        await self.rule_extractor.initialize()
        await self.ml_extractor.initialize()
    
    async def extract_relations(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """Extract relations using hybrid approach"""
        # Extract relations using both methods
        rule_relations = await self.rule_extractor.extract_relations(text, entities)
        ml_relations = await self.ml_extractor.extract_relations(text, entities)
        
        # Combine and resolve conflicts
        combined_relations = self._combine_relations(rule_relations, ml_relations)
        
        # Apply ensemble voting
        final_relations = self._ensemble_voting(combined_relations)
        
        return final_relations
    
    def _combine_relations(
        self, 
        rule_relations: List[ExtractedRelation], 
        ml_relations: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """Combine relations from different extractors"""
        combined = []
        
        # Group relations by entity pair
        relation_groups = {}
        
        # Add rule-based relations
        for relation in rule_relations:
            key = self._get_relation_key(relation)
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(("rule", relation))
        
        # Add ML relations
        for relation in ml_relations:
            key = self._get_relation_key(relation)
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(("ml", relation))
        
        # Resolve conflicts for each entity pair
        for key, relations in relation_groups.items():
            resolved_relation = self._resolve_relation_conflict(relations)
            if resolved_relation:
                combined.append(resolved_relation)
        
        return combined
    
    def _get_relation_key(self, relation: ExtractedRelation) -> tuple:
        """Get unique key for relation based on entity pair"""
        return (
            relation.source_entity.text.lower(),
            relation.target_entity.text.lower()
        )
    
    def _resolve_relation_conflict(
        self, 
        relations: List[tuple]
    ) -> Optional[ExtractedRelation]:
        """Resolve conflicts between different extraction methods"""
        if not relations:
            return None
        
        if len(relations) == 1:
            return relations[0][1]
        
        # Group by relation type
        type_groups = {}
        for method, relation in relations:
            rel_type = relation.relation_type
            if rel_type not in type_groups:
                type_groups[rel_type] = []
            type_groups[rel_type].append((method, relation))
        
        # If all methods agree on relation type, combine confidences
        if len(type_groups) == 1:
            rel_type = list(type_groups.keys())[0]
            method_relations = type_groups[rel_type]
            
            # Calculate weighted confidence
            total_confidence = 0.0
            total_weight = 0.0
            
            best_relation = None
            best_confidence = 0.0
            
            for method, relation in method_relations:
                weight = self.weights.get(method, 0.5)
                weighted_conf = relation.confidence * weight
                total_confidence += weighted_conf
                total_weight += weight
                
                if relation.confidence > best_confidence:
                    best_confidence = relation.confidence
                    best_relation = relation
            
            if best_relation:
                # Update confidence with weighted average
                best_relation.confidence = total_confidence / total_weight if total_weight > 0 else best_confidence
                best_relation.metadata["extraction_method"] = "hybrid"
                best_relation.metadata["agreement"] = True
                
                return best_relation
        
        # If methods disagree, use the one with higher weighted confidence
        best_relation = None
        best_weighted_confidence = 0.0
        
        for method, relation in relations:
            weight = self.weights.get(method, 0.5)
            weighted_conf = relation.confidence * weight
            
            if weighted_conf > best_weighted_confidence:
                best_weighted_confidence = weighted_conf
                best_relation = relation
        
        if best_relation:
            best_relation.metadata["extraction_method"] = "hybrid"
            best_relation.metadata["agreement"] = False
            best_relation.metadata["conflict_resolved"] = True
        
        return best_relation
    
    def _ensemble_voting(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """Apply ensemble voting to final relations"""
        # Filter by minimum confidence threshold
        filtered_relations = [
            r for r in relations 
            if r.confidence >= self.config.confidence_threshold
        ]
        
        # Sort by confidence
        filtered_relations.sort(key=lambda r: r.confidence, reverse=True)
        
        return filtered_relations
    
    def get_supported_relation_types(self) -> List[str]:
        """Get list of supported relation types"""
        return self.config.relation_types
```

#### Deliverables
- [ ] Transformer-based relation extractor
- [ ] Model training pipeline
- [ ] Hybrid extraction approach
- [ ] Ensemble voting mechanism
- [ ] Performance evaluation

## Testing Requirements

### Unit Tests
```python
# tests/test_relation_extraction.py
import pytest
import asyncio
from morag_nlp.extractors.relation_extractor import RuleBasedRelationExtractor
from morag_nlp.extractors.ml_relation_extractor import TransformerRelationExtractor
from morag_nlp.extractors.hybrid_relation_extractor import HybridRelationExtractor
from morag_nlp.extractors.base import ExtractedEntity
from morag_nlp.config import NLPConfig
from morag_nlp.utils.model_utils import ModelManager

@pytest.mark.asyncio
async def test_rule_based_relation_extraction():
    """Test rule-based relation extraction"""
    config = NLPConfig()
    model_manager = ModelManager(config.model_cache_dir)
    extractor = RuleBasedRelationExtractor(config, model_manager)
    
    text = "Steve Jobs founded Apple Inc. in Cupertino, California."
    entities = [
        ExtractedEntity("Steve Jobs", "PERSON", 0, 10, 0.9),
        ExtractedEntity("Apple Inc.", "ORG", 19, 29, 0.9),
        ExtractedEntity("Cupertino", "GPE", 33, 42, 0.8),
        ExtractedEntity("California", "GPE", 44, 54, 0.8)
    ]
    
    relations = await extractor.extract_relations(text, entities)
    
    assert len(relations) > 0
    # Check for expected relations
    relation_types = [r.relation_type for r in relations]
    assert "CREATED_BY" in relation_types or "WORKS_FOR" in relation_types

@pytest.mark.asyncio
async def test_hybrid_relation_extraction():
    """Test hybrid relation extraction"""
    config = NLPConfig()
    model_manager = ModelManager(config.model_cache_dir)
    extractor = HybridRelationExtractor(config, model_manager)
    
    text = "Microsoft was founded by Bill Gates in Seattle, Washington."
    entities = [
        ExtractedEntity("Microsoft", "ORG", 0, 9, 0.9),
        ExtractedEntity("Bill Gates", "PERSON", 25, 35, 0.9),
        ExtractedEntity("Seattle", "GPE", 39, 46, 0.8),
        ExtractedEntity("Washington", "GPE", 48, 58, 0.8)
    ]
    
    relations = await extractor.extract_relations(text, entities)
    
    assert len(relations) > 0
    # Check that hybrid method produces reasonable results
    for relation in relations:
        assert 0.0 <= relation.confidence <= 1.0
        assert relation.relation_type in config.relation_types
```

### Integration Tests
```python
# tests/test_relation_integration.py
import pytest
import asyncio
from morag_nlp.processors.pipeline import ComprehensiveNLPPipeline
from morag_nlp.config import NLPConfig

@pytest.mark.asyncio
async def test_end_to_end_relation_extraction():
    """Test end-to-end relation extraction in pipeline"""
    config = NLPConfig()
    pipeline = ComprehensiveNLPPipeline(config)
    
    text = """
    Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, 
    and Ronald Wayne. The company is headquartered in Cupertino, California. 
    Tim Cook currently serves as the CEO of Apple.
    """
    
    result = await pipeline.process_text(text)
    
    assert len(result.entities) > 0
    assert len(result.relations) > 0
    
    # Check for expected entity types
    entity_types = [e.label for e in result.entities]
    assert "ORG" in entity_types
    assert "PERSON" in entity_types
    assert "GPE" in entity_types
    
    # Check for expected relation types
    relation_types = [r.relation_type for r in result.relations]
    expected_relations = ["CREATED_BY", "LOCATED_IN", "WORKS_FOR"]
    assert any(rel in relation_types for rel in expected_relations)
```

## Success Criteria

- [ ] Rule-based relation extractor implemented and tested
- [ ] ML-based relation extractor with training pipeline
- [ ] Hybrid approach combining both methods
- [ ] Support for all configured relation types
- [ ] Confidence scoring and threshold filtering
- [ ] Performance optimization for real-time processing
- [ ] Unit tests achieving >80% coverage
- [ ] Integration tests passing
- [ ] Documentation and examples complete

## Performance Targets

- **Accuracy**: >75% precision and recall on test dataset
- **Speed**: <2 seconds for processing 1000-word documents
- **Memory**: <2GB RAM usage during processing
- **Scalability**: Handle 100+ entity pairs per document

## Next Steps

After completing this task:
1. Proceed to [Task 2.2: Graph Construction Pipeline](./task-2.2-graph-construction.md)
2. Begin integrating relation extraction with graph storage
3. Start planning incremental graph updates

---

**Status**: ⏳ Not Started  
**Assignee**: TBD  
**Last Updated**: December 2024