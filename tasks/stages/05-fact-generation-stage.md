# Task 5: Implement fact-generator Stage

## Overview
Implement the fact-generator stage that extracts facts, keywords, entities, and relations from chunks. This completely replaces all existing fact extraction logic.

## Objectives
- Extract entities and normalize them
- Identify relationships between entities
- Generate keywords and domain-specific facts
- Add source attribution with chunk references
- Support dynamic entity and relation types
- **REMOVE ALL LEGACY FACT EXTRACTION CODE**

## Deliverables

### 1. fact-generator Stage Implementation (Complete Replacement)
```python
from morag_stages.models import Stage, StageType, StageResult, StageContext, StageStatus
from morag_services import GeminiLLMService
from morag_graph.services import FactExtractionService, EntityNormalizationService
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json

class FactGeneratorStage(Stage):
    def __init__(self,
                 llm_service: GeminiLLMService,
                 fact_extraction_service: FactExtractionService,
                 entity_normalization_service: EntityNormalizationService):
        super().__init__(StageType.FACT_GENERATOR)
        self.llm_service = llm_service
        self.fact_extraction_service = fact_extraction_service
        self.entity_normalization_service = entity_normalization_service

    async def execute(self,
                     input_files: List[Path],
                     context: StageContext) -> StageResult:
        """Extract facts, entities, and relations from chunks."""
        start_time = time.time()

        try:
            if len(input_files) != 1:
                raise ValueError("Stage 4 requires exactly one chunks.json input file")

            input_file = input_files[0]

            # Load chunk data
            chunk_data = self._load_chunk_data(input_file)

            # Extract facts from chunks
            config = context.config.get('stage4', {})
            facts_data = await self._extract_facts_from_chunks(chunk_data, config)

            # Extract and normalize entities
            entities_data = await self._extract_and_normalize_entities(facts_data, config)

            # Extract relations between entities
            relations_data = await self._extract_relations(entities_data, facts_data, config)

            # Extract keywords
            keywords_data = await self._extract_keywords(chunk_data, config)

            # Create comprehensive facts structure
            facts_output = self._create_facts_output(
                facts_data, entities_data, relations_data, keywords_data,
                chunk_data, input_file
            )

            # Generate output file
            output_file = self._generate_facts_output(
                facts_output, input_file, context
            )

            # Create extraction report
            report_file = self._create_extraction_report(
                facts_output, input_file, context
            )

            execution_time = time.time() - start_time

            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file, report_file],
                metadata={
                    "facts_count": len(facts_data),
                    "entities_count": len(entities_data),
                    "relations_count": len(relations_data),
                    "keywords_count": len(keywords_data),
                    "domain": config.get('domain', 'general'),
                    "processing_time": execution_time
                },
                execution_time=execution_time
            )

        except Exception as e:
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for Stage 4."""
        if len(input_files) != 1:
            return False

        input_file = input_files[0]

        # Check if file exists and is chunks.json
        return input_file.exists() and input_file.name.endswith('.chunks.json')

    def get_dependencies(self) -> List[StageType]:
        """fact-generator depends on chunker."""
        return [StageType.CHUNKER]

    def _load_chunk_data(self, input_file: Path) -> Dict[str, Any]:
        """Load chunk data from JSON file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def _extract_facts_from_chunks(self,
                                        chunk_data: Dict[str, Any],
                                        config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract facts from all chunks."""
        facts = []
        chunks = chunk_data.get('chunks', [])

        # Determine domain for fact extraction
        domain = config.get('domain') or self._detect_domain(chunk_data)

        for chunk in chunks:
            try:
                chunk_facts = await self._extract_facts_from_chunk(
                    chunk, domain, config
                )

                # Add source attribution
                for fact in chunk_facts:
                    fact['source_chunk_index'] = chunk['index']
                    fact['source_metadata'] = chunk.get('source_metadata', {})
                    if 'timestamp' in chunk:
                        fact['timestamp'] = chunk['timestamp']

                facts.extend(chunk_facts)

            except Exception as e:
                logger.warning(f"Failed to extract facts from chunk {chunk['index']}: {e}")
                continue

        return facts

    async def _extract_facts_from_chunk(self,
                                       chunk: Dict[str, Any],
                                       domain: str,
                                       config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract facts from a single chunk."""

        # Prepare fact extraction prompt
        prompt = self._create_fact_extraction_prompt(chunk['content'], domain, config)

        try:
            response = await self.llm_service.generate_text(
                prompt=prompt,
                model=config.get('extraction_model', 'gemini-pro'),
                temperature=0.1,
                max_tokens=2000
            )

            # Parse LLM response to extract structured facts
            facts = self._parse_fact_extraction_response(response, chunk)

            return facts

        except Exception as e:
            logger.error(f"Failed to extract facts from chunk: {e}")
            return []

    def _create_fact_extraction_prompt(self,
                                      content: str,
                                      domain: str,
                                      config: Dict[str, Any]) -> str:
        """Create prompt for fact extraction."""

        return f"""Extract actionable facts from the following content. Focus on concrete, verifiable information rather than meta-commentary.

DOMAIN: {domain}
CONTENT: {content}

Extract facts in the following JSON format:
{{
  "facts": [
    {{
      "statement": "Clear, actionable fact statement",
      "subject": "Main entity or concept",
      "predicate": "Action or relationship",
      "object": "Target entity or value",
      "confidence": 0.95,
      "fact_type": "definition|process|relationship|measurement|etc"
    }}
  ]
}}

REQUIREMENTS:
1. Extract only factual, verifiable information
2. Focus on actionable facts, not meta-commentary
3. Include all relevant entities mentioned
4. Determine fact types dynamically based on content
5. Provide confidence scores (0.0-1.0)
6. Make facts standalone and context-independent

Extract comprehensive facts that capture the essential information from this content."""

    def _parse_fact_extraction_response(self,
                                       response: str,
                                       chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract structured facts."""
        import json
        import re

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                facts = parsed.get('facts', [])

                # Validate and enhance facts
                validated_facts = []
                for fact in facts:
                    if self._validate_fact(fact):
                        # Add metadata
                        fact['chunk_content_preview'] = chunk['content'][:200] + '...'
                        fact['extraction_method'] = 'llm'
                        validated_facts.append(fact)

                return validated_facts

        except Exception as e:
            logger.warning(f"Failed to parse fact extraction response: {e}")

        return []

    def _validate_fact(self, fact: Dict[str, Any]) -> bool:
        """Validate extracted fact structure."""
        required_fields = ['statement', 'subject', 'predicate', 'confidence']

        for field in required_fields:
            if field not in fact or not fact[field]:
                return False

        # Validate confidence score
        confidence = fact.get('confidence', 0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            return False

        return True

    async def _extract_and_normalize_entities(self,
                                             facts_data: List[Dict[str, Any]],
                                             config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and normalize entities from facts."""
        entities = {}

        # Extract entities from facts
        for fact in facts_data:
            # Extract subject and object as entities
            subject = fact.get('subject')
            object_entity = fact.get('object')

            if subject:
                entities[subject] = self._create_entity_object(subject, fact)

            if object_entity:
                entities[object_entity] = self._create_entity_object(object_entity, fact)

        # Normalize entities (merge similar ones)
        if config.get('normalize_entities', True):
            entities = await self._normalize_entities(entities, config)

        return list(entities.values())

    def _create_entity_object(self,
                             entity_name: str,
                             source_fact: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized entity object."""
        return {
            'name': entity_name,
            'normalized_name': entity_name.lower().strip(),
            'entity_type': self._determine_entity_type(entity_name, source_fact),
            'mentions': 1,
            'source_facts': [source_fact.get('statement', '')],
            'confidence': source_fact.get('confidence', 0.5)
        }

    def _determine_entity_type(self,
                              entity_name: str,
                              source_fact: Dict[str, Any]) -> str:
        """Determine entity type dynamically using LLM."""
        # Simple heuristics for now, can be enhanced with LLM
        entity_lower = entity_name.lower()

        if any(word in entity_lower for word in ['company', 'corporation', 'inc', 'ltd']):
            return 'Organization'
        elif any(word in entity_lower for word in ['dr.', 'prof.', 'mr.', 'ms.']):
            return 'Person'
        elif entity_name.isupper() and len(entity_name) <= 10:
            return 'Acronym'
        elif any(char.isdigit() for char in entity_name):
            return 'Measurement'
        else:
            return 'Concept'

    async def _normalize_entities(self,
                                 entities: Dict[str, Dict[str, Any]],
                                 config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize entities by merging similar ones."""
        # Use entity normalization service
        normalized = {}

        for entity_name, entity_data in entities.items():
            normalized_name = await self.entity_normalization_service.normalize_entity(
                entity_name, entity_data.get('entity_type', 'Concept')
            )

            if normalized_name in normalized:
                # Merge with existing entity
                existing = normalized[normalized_name]
                existing['mentions'] += entity_data['mentions']
                existing['source_facts'].extend(entity_data['source_facts'])
                existing['confidence'] = max(existing['confidence'], entity_data['confidence'])
            else:
                entity_data['normalized_name'] = normalized_name
                normalized[normalized_name] = entity_data

        return normalized

    async def _extract_relations(self,
                                entities_data: List[Dict[str, Any]],
                                facts_data: List[Dict[str, Any]],
                                config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relations between entities."""
        relations = []

        for fact in facts_data:
            subject = fact.get('subject')
            predicate = fact.get('predicate')
            object_entity = fact.get('object')

            if subject and predicate and object_entity:
                relation = {
                    'subject': subject,
                    'predicate': self._normalize_predicate(predicate),
                    'object': object_entity,
                    'confidence': fact.get('confidence', 0.5),
                    'source_fact': fact.get('statement', ''),
                    'source_chunk_index': fact.get('source_chunk_index'),
                    'relation_type': self._determine_relation_type(predicate)
                }

                relations.append(relation)

        return relations

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to standard form."""
        # Convert to uppercase and normalize
        normalized = predicate.upper().strip()

        # Map common predicates to standard forms
        predicate_mapping = {
            'IS': 'IS',
            'HAS': 'HAS',
            'CONTAINS': 'CONTAINS',
            'CAUSES': 'CAUSES',
            'SOLVES': 'SOLVES',
            'ADDRESSES': 'ADDRESSES',
            'HANDLES': 'HANDLES',
            'CURES': 'CURES',
            'PREVENTS': 'PREVENTS',
            'LOCATED_IN': 'LOCATED_IN',
            'PART_OF': 'PART_OF'
        }

        return predicate_mapping.get(normalized, normalized)

    def _determine_relation_type(self, predicate: str) -> str:
        """Determine relation type from predicate."""
        predicate_lower = predicate.lower()

        if predicate_lower in ['is', 'are', 'was', 'were']:
            return 'identity'
        elif predicate_lower in ['has', 'contains', 'includes']:
            return 'possession'
        elif predicate_lower in ['causes', 'leads_to', 'results_in']:
            return 'causation'
        elif predicate_lower in ['located_in', 'part_of', 'belongs_to']:
            return 'composition'
        else:
            return 'association'

    async def _extract_keywords(self,
                               chunk_data: Dict[str, Any],
                               config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract keywords from content."""
        if not config.get('extract_keywords', True):
            return []

        # Combine all chunk content
        all_content = ' '.join(chunk['content'] for chunk in chunk_data.get('chunks', []))

        # Use LLM to extract keywords
        prompt = f"""Extract the most important keywords and key phrases from the following content.
        Focus on domain-specific terms, technical concepts, and important entities.

        Content: {all_content[:4000]}  # Limit content length

        Return keywords in JSON format:
        {{
          "keywords": [
            {{
              "term": "keyword or phrase",
              "importance": 0.95,
              "category": "technical|concept|entity|process"
            }}
          ]
        }}
        """

        try:
            response = await self.llm_service.generate_text(
                prompt=prompt,
                model=config.get('extraction_model', 'gemini-pro'),
                temperature=0.1,
                max_tokens=1000
            )

            return self._parse_keywords_response(response)

        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}")
            return []
```

## Implementation Steps

1. **Create fact-generator package structure**
2. **Implement FactGeneratorStage class**
3. **Add fact extraction with LLM prompts**
4. **Implement entity extraction and normalization**
5. **Add relation extraction and validation**
6. **Implement keyword extraction**
7. **Add source attribution and metadata**
8. **REMOVE ALL LEGACY FACT EXTRACTION CODE**
9. **Create comprehensive testing**
10. **Add performance monitoring and error handling**

## Testing Requirements

- Unit tests for fact extraction logic
- Entity normalization tests
- Relation extraction validation
- Keyword extraction tests
- Source attribution verification
- Performance tests for large chunk sets

## Files to Create

- `packages/morag-stages/src/morag_stages/fact_generator/__init__.py`
- `packages/morag-stages/src/morag_stages/fact_generator/implementation.py`
- `packages/morag-stages/src/morag_stages/fact_generator/fact_extraction.py`
- `packages/morag-stages/src/morag_stages/fact_generator/entity_normalization.py`
- `packages/morag-stages/tests/test_fact_generator.py`

## Success Criteria

- Facts are extracted accurately with proper source attribution
- Entities are normalized and deduplicated correctly
- Relations are identified with appropriate types
- Keywords capture important domain concepts
- **ALL LEGACY FACT EXTRACTION CODE IS REMOVED**
- Performance is acceptable for typical chunk sets
- All tests pass with good coverage
