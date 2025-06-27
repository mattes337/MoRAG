# Task 5: Enhanced Entity Extraction with Pattern Matching

## Background

This task combines the best of both worlds: AI-powered entity extraction from PydanticAI with rule-based pattern matching for high-confidence entities. This hybrid approach improves accuracy, coverage, and reliability while providing fallback mechanisms.

### Colemedin's Pattern Matching Insights

The Colemedin approach uses curated lists of known entities (companies, technologies, people) combined with regex patterns for reliable detection. This provides:

1. **High Confidence Detection**: Known entities have 100% accuracy
2. **Fast Processing**: Pattern matching is much faster than LLM calls
3. **Comprehensive Coverage**: Curated lists ensure important entities aren't missed
4. **Fallback Reliability**: Works even when LLM fails

## Implementation Strategy

### Phase 1: Pattern-Based Entity Detection (1 day)

#### 1.1 Entity Pattern Matcher
**File**: `packages/morag-graph/src/morag_graph/extraction/pattern_matcher.py`

```python
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PatternEntityType(str, Enum):
    COMPANY = "COMPANY"
    TECHNOLOGY = "TECHNOLOGY"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    PROGRAMMING_LANGUAGE = "PROGRAMMING_LANGUAGE"
    FRAMEWORK = "FRAMEWORK"

@dataclass
class PatternEntity:
    """Entity detected by pattern matching."""
    name: str
    type: PatternEntityType
    confidence: float
    start_pos: int
    end_pos: int
    pattern_source: str  # Which pattern/list matched

class EntityPatternMatcher:
    """Pattern-based entity matcher with curated knowledge bases."""
    
    def __init__(self):
        self.company_patterns = self._load_company_patterns()
        self.technology_patterns = self._load_technology_patterns()
        self.person_patterns = self._load_person_patterns()
        self.location_patterns = self._load_location_patterns()
        
    def _load_company_patterns(self) -> Dict[str, List[str]]:
        """Load company name patterns and variations."""
        return {
            "tech_giants": [
                "Google", "Alphabet Inc", "Microsoft", "Apple", "Amazon", "Meta", 
                "Facebook", "Tesla", "Netflix", "Uber", "Airbnb", "Spotify",
                "Twitter", "X Corp", "LinkedIn", "Snapchat", "TikTok", "ByteDance"
            ],
            "ai_companies": [
                "OpenAI", "Anthropic", "DeepMind", "Hugging Face", "Stability AI",
                "Cohere", "AI21 Labs", "Inflection AI", "Character.AI", "Midjourney"
            ],
            "cloud_providers": [
                "Amazon Web Services", "AWS", "Microsoft Azure", "Google Cloud",
                "GCP", "Oracle Cloud", "IBM Cloud", "Alibaba Cloud", "DigitalOcean"
            ],
            "enterprise_software": [
                "Salesforce", "Oracle", "SAP", "Adobe", "Atlassian", "ServiceNow",
                "Workday", "Zoom", "Slack", "Notion", "Figma", "Canva"
            ],
            "patterns": [
                r'\b\w+\s+(?:Inc|Corp|Corporation|Ltd|Limited|AG|SE|GmbH|LLC)\b',
                r'\b\w+\s+(?:Technologies|Systems|Solutions|Software|Labs)\b'
            ]
        }
    
    def _load_technology_patterns(self) -> Dict[str, List[str]]:
        """Load technology and framework patterns."""
        return {
            "ai_ml": [
                "artificial intelligence", "AI", "machine learning", "ML", "deep learning",
                "neural network", "transformer", "GPT", "LLM", "large language model",
                "computer vision", "natural language processing", "NLP", "reinforcement learning"
            ],
            "programming_languages": [
                "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
                "Swift", "Kotlin", "Ruby", "PHP", "Scala", "R", "MATLAB", "SQL"
            ],
            "frameworks": [
                "React", "Angular", "Vue.js", "Django", "Flask", "FastAPI", "Spring",
                "Express.js", "Node.js", "TensorFlow", "PyTorch", "Keras", "Scikit-learn"
            ],
            "databases": [
                "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
                "Neo4j", "InfluxDB", "DynamoDB", "Snowflake", "BigQuery", "Redshift"
            ],
            "cloud_tech": [
                "Docker", "Kubernetes", "Terraform", "Jenkins", "GitLab CI", "GitHub Actions",
                "Ansible", "Chef", "Puppet", "Prometheus", "Grafana", "ELK Stack"
            ]
        }
    
    def _load_person_patterns(self) -> Dict[str, List[str]]:
        """Load known person patterns."""
        return {
            "tech_leaders": [
                "Elon Musk", "Jeff Bezos", "Tim Cook", "Satya Nadella", "Sundar Pichai",
                "Mark Zuckerberg", "Sam Altman", "Dario Amodei", "Daniela Amodei",
                "Jensen Huang", "Bill Gates", "Larry Page", "Sergey Brin"
            ],
            "ai_researchers": [
                "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng",
                "Fei-Fei Li", "Demis Hassabis", "Ilya Sutskever", "Andrej Karpathy"
            ],
            "patterns": [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last name pattern
                r'\bDr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Dr. First Last
                r'\bProf\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Prof. First Last
            ]
        }
    
    def _load_location_patterns(self) -> Dict[str, List[str]]:
        """Load location patterns."""
        return {
            "tech_hubs": [
                "Silicon Valley", "San Francisco", "Seattle", "Austin", "Boston",
                "New York", "London", "Tel Aviv", "Singapore", "Beijing", "Shanghai",
                "Tokyo", "Seoul", "Bangalore", "Hyderabad", "Toronto", "Vancouver"
            ],
            "cities": [
                "Mountain View", "Cupertino", "Redmond", "Menlo Park", "Palo Alto",
                "Cambridge", "Dublin", "Amsterdam", "Berlin", "Paris", "Stockholm"
            ]
        }
    
    def extract_entities(self, text: str) -> List[PatternEntity]:
        """Extract entities using pattern matching."""
        entities = []
        
        # Extract companies
        entities.extend(self._extract_companies(text))
        
        # Extract technologies
        entities.extend(self._extract_technologies(text))
        
        # Extract people
        entities.extend(self._extract_people(text))
        
        # Extract locations
        entities.extend(self._extract_locations(text))
        
        # Remove duplicates and overlaps
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_companies(self, text: str) -> List[PatternEntity]:
        """Extract company entities."""
        entities = []
        
        for category, companies in self.company_patterns.items():
            if category == "patterns":
                # Handle regex patterns
                for pattern in companies:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        entities.append(PatternEntity(
                            name=match.group().strip(),
                            type=PatternEntityType.COMPANY,
                            confidence=0.8,  # Lower confidence for pattern matches
                            start_pos=match.start(),
                            end_pos=match.end(),
                            pattern_source=f"regex_{category}"
                        ))
            else:
                # Handle exact name matches
                for company in companies:
                    pattern = r'\b' + re.escape(company) + r'\b'
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        entities.append(PatternEntity(
                            name=match.group(),
                            type=PatternEntityType.COMPANY,
                            confidence=1.0,  # High confidence for exact matches
                            start_pos=match.start(),
                            end_pos=match.end(),
                            pattern_source=category
                        ))
        
        return entities
    
    def _extract_technologies(self, text: str) -> List[PatternEntity]:
        """Extract technology entities."""
        entities = []
        
        for category, technologies in self.technology_patterns.items():
            entity_type = PatternEntityType.PROGRAMMING_LANGUAGE if category == "programming_languages" else PatternEntityType.TECHNOLOGY
            
            for tech in technologies:
                # Use word boundaries for exact matches
                pattern = r'\b' + re.escape(tech) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(PatternEntity(
                        name=match.group(),
                        type=entity_type,
                        confidence=0.95,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        pattern_source=category
                    ))
        
        return entities
    
    def _extract_people(self, text: str) -> List[PatternEntity]:
        """Extract person entities."""
        entities = []
        
        # Extract known people
        for category, people in self.person_patterns.items():
            if category != "patterns":
                for person in people:
                    pattern = r'\b' + re.escape(person) + r'\b'
                    for match in re.finditer(pattern, text):
                        entities.append(PatternEntity(
                            name=match.group(),
                            type=PatternEntityType.PERSON,
                            confidence=1.0,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            pattern_source=category
                        ))
        
        # Extract using name patterns (lower confidence)
        for pattern in self.person_patterns["patterns"]:
            for match in re.finditer(pattern, text):
                # Additional validation for name patterns
                name = match.group().strip()
                if self._is_likely_person_name(name):
                    entities.append(PatternEntity(
                        name=name,
                        type=PatternEntityType.PERSON,
                        confidence=0.7,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        pattern_source="name_pattern"
                    ))
        
        return entities
    
    def _extract_locations(self, text: str) -> List[PatternEntity]:
        """Extract location entities."""
        entities = []
        
        for category, locations in self.location_patterns.items():
            for location in locations:
                pattern = r'\b' + re.escape(location) + r'\b'
                for match in re.finditer(pattern, text):
                    entities.append(PatternEntity(
                        name=match.group(),
                        type=PatternEntityType.LOCATION,
                        confidence=0.9,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        pattern_source=category
                    ))
        
        return entities
    
    def _is_likely_person_name(self, name: str) -> bool:
        """Validate if a string is likely a person name."""
        # Basic validation rules
        words = name.split()
        if len(words) < 2 or len(words) > 4:
            return False
        
        # Check if all words are capitalized
        if not all(word[0].isupper() for word in words):
            return False
        
        # Exclude common non-person patterns
        excluded_patterns = [
            r'\b(Inc|Corp|Ltd|LLC|API|SDK|AI|ML|UI|UX)\b',
            r'\b[A-Z]{2,}\b'  # All caps abbreviations
        ]
        
        for pattern in excluded_patterns:
            if re.search(pattern, name):
                return False
        
        return True
    
    def _deduplicate_entities(self, entities: List[PatternEntity]) -> List[PatternEntity]:
        """Remove duplicate and overlapping entities."""
        # Sort by position
        entities.sort(key=lambda e: (e.start_pos, e.end_pos))
        
        deduplicated = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Overlap detected, keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
```

### Phase 2: Hybrid Entity Extraction (1 day)

#### 2.1 Hybrid Entity Extractor
**File**: `packages/morag-graph/src/morag_graph/extraction/hybrid_entity_extractor.py`

```python
from morag_graph.ai.entity_agent import EntityExtractionAgent
from morag_graph.extraction.pattern_matcher import EntityPatternMatcher, PatternEntity
from morag_core.ai.models.entity_models import ExtractedEntity, EntityType, EntityExtractionResult
from typing import List, Optional, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class HybridEntityExtractor:
    """Hybrid entity extractor combining AI and pattern matching."""
    
    def __init__(self, 
                 ai_confidence_threshold: float = 0.5,
                 pattern_confidence_threshold: float = 0.7,
                 prefer_patterns: bool = True):
        self.ai_agent = EntityExtractionAgent()
        self.pattern_matcher = EntityPatternMatcher()
        self.ai_confidence_threshold = ai_confidence_threshold
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.prefer_patterns = prefer_patterns
    
    async def extract_entities(
        self,
        text: str,
        context: Optional[str] = None,
        use_ai: bool = True,
        use_patterns: bool = True
    ) -> EntityExtractionResult:
        """Extract entities using hybrid approach."""
        
        all_entities = []
        processing_metadata = {
            "ai_enabled": use_ai,
            "patterns_enabled": use_patterns,
            "text_length": len(text)
        }
        
        # Step 1: Pattern-based extraction (fast, high confidence)
        pattern_entities = []
        if use_patterns:
            pattern_results = self.pattern_matcher.extract_entities(text)
            pattern_entities = self._convert_pattern_entities(pattern_results)
            processing_metadata["pattern_entities_found"] = len(pattern_entities)
        
        # Step 2: AI-based extraction
        ai_entities = []
        if use_ai:
            try:
                ai_result = await self.ai_agent.extract_entities(
                    text=text,
                    context=context,
                    min_confidence=self.ai_confidence_threshold
                )
                ai_entities = ai_result.entities
                processing_metadata["ai_entities_found"] = len(ai_entities)
                processing_metadata["ai_metadata"] = ai_result.processing_metadata
            except Exception as e:
                logger.error("AI entity extraction failed", error=str(e))
                processing_metadata["ai_error"] = str(e)
        
        # Step 3: Merge and deduplicate entities
        merged_entities = self._merge_entities(pattern_entities, ai_entities)
        processing_metadata["merged_entities_count"] = len(merged_entities)
        
        return EntityExtractionResult(
            entities=merged_entities,
            total_count=len(merged_entities),
            processing_metadata=processing_metadata
        )
    
    def _convert_pattern_entities(self, pattern_entities: List[PatternEntity]) -> List[ExtractedEntity]:
        """Convert pattern entities to standard format."""
        converted = []
        
        type_mapping = {
            "COMPANY": EntityType.ORGANIZATION,
            "TECHNOLOGY": EntityType.TECHNOLOGY,
            "PERSON": EntityType.PERSON,
            "LOCATION": EntityType.LOCATION,
            "PROGRAMMING_LANGUAGE": EntityType.TECHNOLOGY,
            "FRAMEWORK": EntityType.TECHNOLOGY
        }
        
        for pattern_entity in pattern_entities:
            if pattern_entity.confidence >= self.pattern_confidence_threshold:
                converted.append(ExtractedEntity(
                    name=pattern_entity.name,
                    type=type_mapping.get(pattern_entity.type.value, EntityType.CONCEPT),
                    description=f"Detected by pattern matching ({pattern_entity.pattern_source})",
                    confidence=pattern_entity.confidence,
                    properties={
                        "extraction_method": "pattern_matching",
                        "pattern_source": pattern_entity.pattern_source,
                        "start_pos": pattern_entity.start_pos,
                        "end_pos": pattern_entity.end_pos
                    }
                ))
        
        return converted
    
    def _merge_entities(
        self, 
        pattern_entities: List[ExtractedEntity], 
        ai_entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Merge pattern and AI entities, handling duplicates."""
        
        merged = []
        ai_entities_by_name = {entity.name.lower(): entity for entity in ai_entities}
        
        # Add pattern entities (high priority if prefer_patterns is True)
        for pattern_entity in pattern_entities:
            merged.append(pattern_entity)
            
            # Remove corresponding AI entity to avoid duplicates
            ai_entities_by_name.pop(pattern_entity.name.lower(), None)
        
        # Add remaining AI entities
        for ai_entity in ai_entities_by_name.values():
            # Check for partial matches with pattern entities
            is_duplicate = False
            for pattern_entity in pattern_entities:
                if self._are_similar_entities(pattern_entity.name, ai_entity.name):
                    if self.prefer_patterns:
                        is_duplicate = True
                        break
                    else:
                        # Prefer AI entity, remove pattern entity
                        merged = [e for e in merged if e.name != pattern_entity.name]
                        break
            
            if not is_duplicate:
                merged.append(ai_entity)
        
        # Sort by confidence (descending)
        merged.sort(key=lambda e: e.confidence, reverse=True)
        
        return merged
    
    def _are_similar_entities(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two entity names are similar enough to be considered duplicates."""
        name1_clean = name1.lower().strip()
        name2_clean = name2.lower().strip()
        
        # Exact match
        if name1_clean == name2_clean:
            return True
        
        # One is contained in the other
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True
        
        # Simple similarity check (can be enhanced with fuzzy matching)
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            similarity = intersection / union
            return similarity >= threshold
        
        return False
```

## Testing and Documentation Strategy

### Automated Testing (Each Step)
- Run automated tests in `/tests/test_hybrid_entity_extraction.py` after each implementation step
- Test pattern matching accuracy with curated entity lists
- Test AI + pattern hybrid approach
- Test confidence scoring and entity merging
- Test performance improvements vs baseline
- Test reliability with AI service failures

### Documentation Updates (Mandatory)
- Update `packages/morag-graph/README.md` with hybrid extraction approach
- Create `docs/hybrid-entity-extraction.md` with pattern matching details
- Update API documentation with new hybrid models
- Document curated entity knowledge bases

### Code Cleanup (Mandatory)
- Replace existing entity extraction with hybrid approach
- Remove old single-method extraction code
- Update all entity extraction calls to use hybrid approach
- Remove old entity type classification logic

## Success Criteria

1. ✅ Entity extraction accuracy improved by 15%
2. ✅ High-confidence entities (companies, technologies) have 95%+ accuracy
3. ✅ Processing time reduced by 30% due to pattern matching
4. ✅ ALL old entity extraction code replaced
5. ✅ Comprehensive automated test coverage
6. ✅ All documentation updated
7. ✅ Fallback reliability validated with automated testing

## Dependencies

- Completed PydanticAI foundation
- New entity extraction agent (from Task 2)
- Curated entity knowledge bases (no dependency on old extraction)
