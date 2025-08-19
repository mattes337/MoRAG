"""Fact generator stage implementation."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import structlog

from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

# Import services with graceful fallback
try:
    from morag_core.ai import create_agent, AgentConfig
    from morag_graph.extraction import FactExtractor, EntityNormalizer
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    create_agent = None
    AgentConfig = None
    FactExtractor = None
    EntityNormalizer = None

logger = structlog.get_logger(__name__)


class FactGeneratorStage(Stage):
    """Stage that extracts facts, entities, relations, and keywords from chunks."""
    
    def __init__(self, stage_type: StageType = StageType.FACT_GENERATOR):
        """Initialize fact generator stage."""
        super().__init__(stage_type)
        
        if not SERVICES_AVAILABLE:
            logger.warning("Services not available for fact generation")
        
        self.fact_extractor = FactExtractor() if FactExtractor else None
        self.entity_normalizer = EntityNormalizer() if EntityNormalizer else None
        self.extraction_agent = None
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Execute fact generation on input chunk files.
        
        Args:
            input_files: List of input chunk JSON files
            context: Stage execution context
            
        Returns:
            Stage execution result
        """
        if len(input_files) != 1:
            raise StageValidationError(
                "Fact generator stage requires exactly one input file",
                stage_type=self.stage_type.value,
                invalid_files=[str(f) for f in input_files]
            )
        
        input_file = input_files[0]
        config = context.get_stage_config(self.stage_type)
        
        logger.info("Starting fact generation", 
                   input_file=str(input_file),
                   config=config)
        
        try:
            # Generate output filename
            output_file = context.output_dir / f"{input_file.stem.replace('.chunks', '')}.facts.json"
            context.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Read input chunks
            with open(input_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = chunks_data.get('chunks', [])
            source_metadata = chunks_data.get('metadata', {})
            
            # Extract facts from chunks
            all_entities = []
            all_relations = []
            all_facts = []
            all_keywords = set()
            
            for chunk in chunks:
                chunk_results = await self._extract_from_chunk(chunk, config)
                
                all_entities.extend(chunk_results.get('entities', []))
                all_relations.extend(chunk_results.get('relations', []))
                all_facts.extend(chunk_results.get('facts', []))
                all_keywords.update(chunk_results.get('keywords', []))
            
            # Normalize and deduplicate entities
            if self.entity_normalizer:
                all_entities = await self._normalize_entities(all_entities)
            else:
                all_entities = self._basic_entity_deduplication(all_entities)
            
            # Deduplicate relations and facts
            all_relations = self._deduplicate_relations(all_relations)
            all_facts = self._deduplicate_facts(all_facts)
            
            # Create output data
            output_data = {
                "entities": all_entities,
                "relations": all_relations,
                "facts": all_facts,
                "keywords": sorted(list(all_keywords)),
                "metadata": {
                    "total_entities": len(all_entities),
                    "total_relations": len(all_relations),
                    "total_facts": len(all_facts),
                    "total_keywords": len(all_keywords),
                    "source_chunks": len(chunks),
                    "extraction_config": config,
                    "source_metadata": source_metadata,
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Create metadata
            stage_metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(input_file)],
                output_files=[str(output_file)],
                config_used=config,
                metrics={
                    "total_entities": len(all_entities),
                    "total_relations": len(all_relations),
                    "total_facts": len(all_facts),
                    "total_keywords": len(all_keywords),
                    "chunks_processed": len(chunks),
                    "extraction_enabled": {
                        "entities": config.get('extract_entities', True),
                        "relations": config.get('extract_relations', True),
                        "facts": config.get('extract_facts', True),
                        "keywords": config.get('extract_keywords', True)
                    }
                }
            )
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file],
                metadata=stage_metadata,
                data={
                    "entities_count": len(all_entities),
                    "relations_count": len(all_relations),
                    "facts_count": len(all_facts),
                    "keywords_count": len(all_keywords)
                }
            )
            
        except Exception as e:
            logger.error("Fact generation failed", 
                        input_file=str(input_file), 
                        error=str(e))
            raise StageExecutionError(
                f"Fact generation failed: {e}",
                stage_type=self.stage_type.value,
                original_error=e
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for fact generation.
        
        Args:
            input_files: List of input file paths
            
        Returns:
            True if inputs are valid
        """
        if len(input_files) != 1:
            return False
        
        input_file = input_files[0]
        
        # Check if file exists and is JSON
        if not input_file.exists():
            return False
        
        if not input_file.name.endswith('.chunks.json'):
            return False
        
        # Try to parse JSON
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return 'chunks' in data
        except (json.JSONDecodeError, KeyError):
            return False
    
    def get_dependencies(self) -> List[StageType]:
        """Get stage dependencies.
        
        Returns:
            List containing chunker stage
        """
        return [StageType.CHUNKER]
    
    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Get expected output file paths.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            List of expected output file paths
        """
        if len(input_files) != 1:
            return []
        
        input_file = input_files[0]
        output_file = context.output_dir / f"{input_file.stem.replace('.chunks', '')}.facts.json"
        return [output_file]
    
    async def _extract_from_chunk(self, chunk: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract facts from a single chunk.
        
        Args:
            chunk: Chunk data
            config: Stage configuration
            
        Returns:
            Extraction results
        """
        content = chunk.get('content', '')
        chunk_id = chunk.get('id', 'unknown')
        
        results = {
            'entities': [],
            'relations': [],
            'facts': [],
            'keywords': []
        }
        
        # Use fact extractor if available
        if self.fact_extractor and SERVICES_AVAILABLE:
            try:
                extraction_result = await self.fact_extractor.extract_from_text(
                    content,
                    domain=config.get('domain', 'general'),
                    extract_entities=config.get('extract_entities', True),
                    extract_relations=config.get('extract_relations', True),
                    extract_facts=config.get('extract_facts', True),
                    min_confidence=config.get('min_confidence', 0.7)
                )
                
                # Add source chunk information
                for entity in extraction_result.get('entities', []):
                    entity['source_chunks'] = [chunk_id]
                
                for relation in extraction_result.get('relations', []):
                    relation['source_chunks'] = [chunk_id]
                
                for fact in extraction_result.get('facts', []):
                    fact['source_chunk'] = chunk_id
                
                results.update(extraction_result)
                
            except Exception as e:
                logger.warning("Fact extractor failed, using LLM fallback", error=str(e))
                results = await self._llm_extraction_fallback(content, chunk_id, config)
        else:
            # Use LLM-based extraction
            results = await self._llm_extraction_fallback(content, chunk_id, config)
        
        # Extract keywords if enabled
        if config.get('extract_keywords', True):
            keywords = self._extract_keywords(content)
            results['keywords'] = keywords
        
        return results

    async def _llm_extraction_fallback(self, content: str, chunk_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract facts using LLM as fallback.

        Args:
            content: Chunk content
            chunk_id: Chunk identifier
            config: Stage configuration

        Returns:
            Extraction results
        """
        if not create_agent:
            return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}

        try:
            if not self.extraction_agent:
                agent_config = AgentConfig(
                    model=config.get('model', 'gemini-pro'),
                    temperature=config.get('temperature', 0.1),
                    max_tokens=config.get('max_tokens', 4096)
                )
                self.extraction_agent = create_agent(agent_config)

            # Create extraction prompt
            system_prompt = self._get_extraction_system_prompt(config)
            user_prompt = self._get_extraction_user_prompt(content, config)

            response = await self.extraction_agent.run(user_prompt, system_prompt=system_prompt)
            response_text = response.data if hasattr(response, 'data') else str(response)

            # Parse response (expecting JSON format)
            try:
                results = json.loads(response_text)

                # Add source chunk information
                for entity in results.get('entities', []):
                    entity['source_chunks'] = [chunk_id]

                for relation in results.get('relations', []):
                    relation['source_chunks'] = [chunk_id]

                for fact in results.get('facts', []):
                    fact['source_chunk'] = chunk_id

                return results

            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM extraction response as JSON")
                return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}

        except Exception as e:
            logger.warning("LLM extraction failed", error=str(e))
            return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}

    def _get_extraction_system_prompt(self, config: Dict[str, Any]) -> str:
        """Get system prompt for extraction.

        Args:
            config: Stage configuration

        Returns:
            System prompt string
        """
        domain = config.get('domain', 'general')

        prompt = f"""You are an expert knowledge extraction system. Extract structured information from the provided text in the {domain} domain.

Return your response as valid JSON with the following structure:
{{
    "entities": [
        {{
            "name": "Entity Name",
            "type": "EntityType",
            "normalized_name": "entity_name",
            "confidence": 0.95
        }}
    ],
    "relations": [
        {{
            "subject": "Entity A",
            "predicate": "RELATION_TYPE",
            "object": "Entity B",
            "confidence": 0.88
        }}
    ],
    "facts": [
        {{
            "statement": "Factual statement extracted from text",
            "entities": ["Entity A", "Entity B"],
            "confidence": 0.92
        }}
    ]
}}

Guidelines:
- Extract only factual, verifiable information
- Use specific, descriptive entity types
- Use uppercase relation types (e.g., WORKS_FOR, LOCATED_IN, CAUSES)
- Normalize entity names to lowercase with underscores
- Include confidence scores between 0.0 and 1.0
- Focus on the most important and relevant information"""

        if config.get('entity_types'):
            prompt += f"\n- Prioritize these entity types: {', '.join(config['entity_types'])}"

        if config.get('relation_types'):
            prompt += f"\n- Prioritize these relation types: {', '.join(config['relation_types'])}"

        return prompt

    def _get_extraction_user_prompt(self, content: str, config: Dict[str, Any]) -> str:
        """Get user prompt for extraction.

        Args:
            content: Content to extract from
            config: Stage configuration

        Returns:
            User prompt string
        """
        max_entities = config.get('max_entities_per_chunk', 20)
        max_relations = config.get('max_relations_per_chunk', 15)
        min_confidence = config.get('min_confidence', 0.7)

        return f"""Extract structured information from this text. Limit to {max_entities} entities and {max_relations} relations. Only include items with confidence >= {min_confidence}.

Text:
{content}

Return valid JSON only."""

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content using simple heuristics.

        Args:
            content: Content to extract keywords from

        Returns:
            List of keywords
        """
        import re

        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{3,}\b', content.lower())

        # Filter common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'said',
            'says', 'say', 'get', 'got', 'getting', 'make', 'made', 'making',
            'take', 'took', 'taking', 'come', 'came', 'coming', 'go', 'went',
            'going', 'see', 'saw', 'seeing', 'know', 'knew', 'knowing', 'think',
            'thought', 'thinking', 'want', 'wanted', 'wanting', 'use', 'used',
            'using', 'work', 'worked', 'working', 'way', 'ways', 'time', 'times',
            'year', 'years', 'day', 'days', 'people', 'person', 'man', 'woman',
            'child', 'children', 'life', 'world', 'country', 'state', 'city',
            'place', 'home', 'house', 'school', 'company', 'business', 'job',
            'money', 'number', 'part', 'point', 'hand', 'eye', 'face', 'fact',
            'case', 'group', 'problem', 'question', 'right', 'left', 'good',
            'bad', 'great', 'small', 'large', 'big', 'little', 'long', 'short',
            'high', 'low', 'old', 'new', 'first', 'last', 'next', 'other',
            'same', 'different', 'important', 'public', 'able', 'own', 'sure',
            'such', 'only', 'still', 'also', 'back', 'well', 'just', 'now',
            'here', 'there', 'where', 'when', 'why', 'how', 'what', 'who',
            'which', 'all', 'any', 'each', 'every', 'some', 'many', 'much',
            'more', 'most', 'few', 'little', 'less', 'least', 'very', 'too',
            'quite', 'really', 'actually', 'probably', 'maybe', 'perhaps'
        }

        # Count word frequencies
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:20] if count > 1]

    async def _normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize and deduplicate entities.

        Args:
            entities: List of entity dictionaries

        Returns:
            Normalized and deduplicated entities
        """
        if not self.entity_normalizer:
            return self._basic_entity_deduplication(entities)

        try:
            return await self.entity_normalizer.normalize_entities(entities)
        except Exception as e:
            logger.warning("Entity normalization failed, using basic deduplication", error=str(e))
            return self._basic_entity_deduplication(entities)

    def _basic_entity_deduplication(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic entity deduplication by name.

        Args:
            entities: List of entity dictionaries

        Returns:
            Deduplicated entities
        """
        seen_names = set()
        deduplicated = []

        for entity in entities:
            name = entity.get('name', '').lower()
            if name and name not in seen_names:
                seen_names.add(name)
                deduplicated.append(entity)

        return deduplicated

    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relations by subject-predicate-object.

        Args:
            relations: List of relation dictionaries

        Returns:
            Deduplicated relations
        """
        seen_relations = set()
        deduplicated = []

        for relation in relations:
            subject = relation.get('subject', '').lower()
            predicate = relation.get('predicate', '').upper()
            obj = relation.get('object', '').lower()

            relation_key = (subject, predicate, obj)
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                deduplicated.append(relation)

        return deduplicated

    def _deduplicate_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate facts by statement.

        Args:
            facts: List of fact dictionaries

        Returns:
            Deduplicated facts
        """
        seen_statements = set()
        deduplicated = []

        for fact in facts:
            statement = fact.get('statement', '').lower().strip()
            if statement and statement not in seen_statements:
                seen_statements.add(statement)
                deduplicated.append(fact)

        return deduplicated
