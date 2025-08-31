"""Fact generator stage implementation."""

import json
from datetime import datetime
from typing import List, Dict, Any, TYPE_CHECKING
from pathlib import Path
import structlog

from morag_core.config import FactGeneratorConfig
from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

# Import services with graceful fallback
if TYPE_CHECKING:
    from morag_core.ai import create_agent, AgentConfig
    from morag_graph.extraction import FactExtractor, EntityNormalizer

try:
    from morag_core.ai import create_agent as _create_agent, AgentConfig as _AgentConfig
    from morag_graph.extraction import FactExtractor as _FactExtractor, EntityNormalizer as _EntityNormalizer
    create_agent = _create_agent
    AgentConfig = _AgentConfig
    FactExtractor = _FactExtractor
    EntityNormalizer = _EntityNormalizer
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    create_agent = None  # type: ignore
    AgentConfig = None  # type: ignore
    FactExtractor = None  # type: ignore
    EntityNormalizer = None  # type: ignore

# Import robust LLM response parser from agents framework
try:
    from agents.base import LLMResponseParser
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    LLMResponseParser = None

logger = structlog.get_logger(__name__)


class FactGeneratorStage(Stage):
    """Stage that extracts facts, entities, relations, and keywords from chunks."""

    def __init__(self, stage_type: StageType = StageType.FACT_GENERATOR):
        """Initialize fact generator stage."""
        super().__init__(stage_type)

        if not SERVICES_AVAILABLE:
            logger.warning("Services not available for fact generation")

        # Initialize fact extractor with API key from environment
        if SERVICES_AVAILABLE and FactExtractor is not None:
            import os
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                self.fact_extractor = FactExtractor(
                    model_id=os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.0-flash'),
                    api_key=api_key,
                    min_confidence=float(os.getenv('MORAG_FACT_GENERATOR_MIN_CONFIDENCE', '0.3')),
                    allow_vague_language=os.getenv('MORAG_FACT_GENERATOR_ALLOW_VAGUE_LANGUAGE', 'true').lower() == 'true',
                    require_entities=os.getenv('MORAG_FACT_GENERATOR_REQUIRE_ENTITIES', 'false').lower() == 'true',
                    min_fact_length=int(os.getenv('MORAG_FACT_GENERATOR_MIN_FACT_LENGTH', '20')),
                    strict_validation=os.getenv('MORAG_FACT_GENERATOR_STRICT_VALIDATION', 'true').lower() == 'true'
                )
            else:
                logger.warning("GEMINI_API_KEY not found - fact extraction disabled")
                self.fact_extractor = None
        else:
            self.fact_extractor = None

        # Initialize entity normalizer with API key from environment
        if SERVICES_AVAILABLE and EntityNormalizer is not None:
            import os
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                self.entity_normalizer = EntityNormalizer(
                    model_name=os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.0-flash'),
                    api_key=api_key
                )
            else:
                logger.warning("GEMINI_API_KEY not found - entity normalization disabled")
                self.entity_normalizer = None
        else:
            self.entity_normalizer = None

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

        # Load configuration from environment variables with context overrides
        context_config = context.get_stage_config(self.stage_type)
        config = FactGeneratorConfig.from_env_and_overrides(context_config)
        
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

            logger.info("Processing chunks for fact extraction",
                       total_chunks=len(chunks),
                       fact_extractor_available=bool(self.fact_extractor),
                       services_available=SERVICES_AVAILABLE)

            for i, chunk in enumerate(chunks):
                logger.debug("Processing chunk for fact extraction",
                           chunk_id=chunk.get('id', f'chunk_{i}'),
                           chunk_size=len(chunk.get('content', '')))

                chunk_results = await self._extract_from_chunk(chunk, config)

                # Log results for debugging
                logger.debug("Chunk extraction results",
                           chunk_id=chunk.get('id', f'chunk_{i}'),
                           entities_found=len(chunk_results.get('entities', [])),
                           relations_found=len(chunk_results.get('relations', [])),
                           facts_found=len(chunk_results.get('facts', [])),
                           keywords_found=len(chunk_results.get('keywords', [])))

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
                    "extraction_config": config.model_dump() if hasattr(config, 'model_dump') else config.__dict__,
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
                config_used=config.model_dump() if hasattr(config, 'model_dump') else config.__dict__,
                metrics={
                    "total_entities": len(all_entities),
                    "total_relations": len(all_relations),
                    "total_facts": len(all_facts),
                    "total_keywords": len(all_keywords),
                    "chunks_processed": len(chunks),
                    "extraction_enabled": {
                        "entities": getattr(config, 'extract_entities', True),
                        "relations": getattr(config, 'extract_relations', True),
                        "facts": getattr(config, 'extract_facts', True),
                        "keywords": getattr(config, 'extract_keywords', True)
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
        from ..file_manager import sanitize_filename
        base_name = input_file.stem.replace('.chunks', '')
        sanitized_name = sanitize_filename(base_name)
        output_file = context.output_dir / f"{sanitized_name}.facts.json"
        return [output_file]
    
    async def _extract_from_chunk(self, chunk: Dict[str, Any], config: FactGeneratorConfig) -> Dict[str, Any]:
        """Extract facts from a single chunk.
        
        Args:
            chunk: Chunk data
            config: Stage configuration
            
        Returns:
            Extraction results
        """
        content = chunk.get('content', '')
        chunk_id = chunk.get('id', 'unknown')
        
        results: Dict[str, Any] = {
            'entities': [],
            'relations': [],
            'facts': [],
            'keywords': []
        }
        
        # Use fact extractor if available
        if self.fact_extractor and SERVICES_AVAILABLE:
            try:
                logger.debug("Using fact extractor service", chunk_id=chunk_id, content_length=len(content))

                # Update fact extractor configuration from request config
                if hasattr(config, 'min_confidence'):
                    self.fact_extractor.min_confidence = getattr(config, 'min_confidence', 0.5)
                    self.fact_extractor.validator.min_confidence = getattr(config, 'min_confidence', 0.5)

                if hasattr(config, 'allow_vague_language'):
                    self.fact_extractor.validator.allow_vague_language = getattr(config, 'allow_vague_language', False)

                if hasattr(config, 'require_entities'):
                    self.fact_extractor.validator.require_entities = getattr(config, 'require_entities', True)

                if hasattr(config, 'min_fact_length'):
                    self.fact_extractor.validator.min_fact_length = getattr(config, 'min_fact_length', 20)

                if hasattr(config, 'strict_validation'):
                    self.fact_extractor.validator.strict_validation = getattr(config, 'strict_validation', True)

                # Use the correct method name: extract_facts
                facts = await self.fact_extractor.extract_facts(
                    chunk_text=content,
                    chunk_id=chunk_id,
                    document_id="unknown",  # We don't have document_id in this context
                    context={
                        'domain': getattr(config, 'domain', 'general'),
                        'language': 'en'
                    }
                )

                logger.debug("Fact extractor returned results",
                           chunk_id=chunk_id,
                           facts_count=len(facts) if facts else 0,
                           facts_type=type(facts).__name__)

                # Convert facts to the expected format
                extraction_result = {
                    'entities': [],
                    'relations': [],
                    'facts': [
                        {
                            'id': fact.id,
                            'fact_text': fact.fact_text,
                            'fact_type': fact.fact_type,
                            'confidence': fact.extraction_confidence,
                            'keywords': fact.keywords,
                            'source_chunk': chunk_id,
                            'source_document_id': fact.source_document_id,
                            'source_chunk_id': fact.source_chunk_id,
                            'domain': fact.domain,
                            'language': fact.language,
                            'structured_metadata': fact.structured_metadata.model_dump() if hasattr(fact.structured_metadata, 'model_dump') else fact.structured_metadata
                        }
                        for fact in facts
                    ]
                }
                
                # Add source chunk information
                for entity in extraction_result.get('entities', []):
                    entity['source_chunks'] = [chunk_id]
                
                for relation in extraction_result.get('relations', []):
                    relation['source_chunks'] = [chunk_id]
                
                for fact in extraction_result.get('facts', []):
                    fact['source_chunk'] = chunk_id
                
                results.update(extraction_result)
                
            except Exception as e:
                logger.warning("Fact extractor failed, using LLM fallback",
                             error=str(e),
                             chunk_id=chunk_id,
                             error_type=type(e).__name__)
                results = await self._llm_extraction_fallback(content, chunk_id, config)
        else:
            # Use LLM-based extraction
            logger.debug("Using LLM fallback for fact extraction",
                        chunk_id=chunk_id,
                        reason="fact_extractor_not_available" if not self.fact_extractor else "services_not_available")
            results = await self._llm_extraction_fallback(content, chunk_id, config)
        
        # Extract keywords if enabled
        if getattr(config, 'extract_keywords', True):
            keywords = self._extract_keywords(content)
            results['keywords'] = keywords
        
        return results

    async def _llm_extraction_fallback(self, content: str, chunk_id: str, config: FactGeneratorConfig) -> Dict[str, Any]:
        """Extract facts using LLM as fallback.

        Args:
            content: Chunk content
            chunk_id: Chunk identifier
            config: Stage configuration

        Returns:
            Extraction results
        """
        # We can still use LLM fallback even if services are not available
        # Only return empty if we can't import the LLM client

        logger.debug("Starting LLM extraction fallback",
                    chunk_id=chunk_id,
                    content_length=len(content),
                    config_domain=getattr(config, 'domain', 'unknown'))

        try:
            # Use direct LLM call instead of agent for better JSON control
            try:
                from morag_reasoning.llm import LLMClient, LLMConfig as ReasoningLLMConfig
            except ImportError:
                # Fallback to direct Google AI API if morag_reasoning is not available
                try:
                    import google.generativeai as genai
                    import os

                    # Configure Google AI
                    api_key = os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
                    if not api_key:
                        logger.warning("No Google AI API key found, cannot use LLM fallback")
                        return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}

                    genai.configure(api_key=api_key)

                    # Create model
                    model = genai.GenerativeModel(
                        model_name=getattr(config, 'model', 'gemini-1.5-flash'),
                        generation_config=genai.types.GenerationConfig(
                            temperature=getattr(config, 'temperature', 0.1),
                            max_output_tokens=getattr(config, 'max_tokens', 4000)
                        )
                    )

                    # Create extraction prompt
                    system_prompt = self._get_extraction_system_prompt(config)
                    user_prompt = self._get_extraction_user_prompt(content, config)

                    # Generate response
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    response = model.generate_content(full_prompt)
                    response_text = response.text if response.text else ""

                except ImportError:
                    logger.warning("Neither morag_reasoning nor google.generativeai available, cannot use LLM fallback")
                    return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}
                except Exception as e:
                    logger.warning("Direct Google AI fallback failed", error=str(e))
                    return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}
            else:
                # Use morag_reasoning LLM client
                if not hasattr(self, '_llm_client') or self._llm_client is None:
                    # Get LLM configuration with stage-specific overrides
                    llm_config = config.get_llm_config() if hasattr(config, 'get_llm_config') else config

                    # Convert to reasoning LLMConfig format
                    reasoning_config = ReasoningLLMConfig(
                        provider=getattr(llm_config, 'provider', 'gemini'),
                        model=getattr(llm_config, 'model', 'gemini-2.0-flash'),
                        api_key=getattr(llm_config, 'api_key', ''),
                        temperature=getattr(llm_config, 'temperature', 0.1),
                        max_tokens=getattr(llm_config, 'max_tokens', 4000),
                        max_retries=getattr(llm_config, 'max_retries', 3),
                    )
                    self._llm_client = LLMClient(reasoning_config)

                # Create extraction prompt
                system_prompt = self._get_extraction_system_prompt(config)
                user_prompt = self._get_extraction_user_prompt(content, config)

                # Use LLM client for direct JSON response
                response_text = await self._llm_client.generate_from_messages([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])

            # Parse response (expecting JSON format) using robust parser
            logger.debug("Parsing LLM response",
                        chunk_id=chunk_id,
                        response_length=len(response_text),
                        response_preview=response_text[:200])

            try:
                # Use centralized parser if available, otherwise fallback to basic parsing
                fallback_result = {'entities': [], 'relations': [], 'facts': [], 'keywords': []}

                if PARSER_AVAILABLE and LLMResponseParser:
                    # Use the robust parser from agents framework
                    logger.debug("Using LLMResponseParser", chunk_id=chunk_id)
                    results = LLMResponseParser.parse_json_response(
                        response=response_text,
                        fallback_value=fallback_result,
                        context=f"fact_generator_{chunk_id}"
                    )
                else:
                    # Fallback to basic JSON parsing
                    logger.debug("Using basic JSON parsing", chunk_id=chunk_id)
                    import re
                    try:
                        results = json.loads(response_text.strip())
                        logger.debug("Successfully parsed JSON directly", chunk_id=chunk_id)
                    except json.JSONDecodeError:
                        # Try to extract JSON from markdown code blocks
                        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                        if json_match:
                            try:
                                results = json.loads(json_match.group(1))
                                logger.debug("Successfully parsed JSON from code block", chunk_id=chunk_id)
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse LLM extraction response as JSON",
                                             chunk_id=chunk_id,
                                             response_preview=response_text[:500])
                                results = fallback_result
                        else:
                            logger.warning("No JSON found in LLM response",
                                         chunk_id=chunk_id,
                                         response_preview=response_text[:500])
                            results = fallback_result

                # Ensure results has the expected structure
                if not isinstance(results, dict):
                    logger.warning("LLM response is not a dict, using fallback",
                                 chunk_id=chunk_id,
                                 results_type=type(results).__name__)
                    results = fallback_result

                # Log parsed results for debugging
                logger.debug("Parsed LLM extraction results",
                           chunk_id=chunk_id,
                           entities_count=len(results.get('entities', [])),
                           relations_count=len(results.get('relations', [])),
                           facts_count=len(results.get('facts', [])),
                           keywords_count=len(results.get('keywords', [])))

                # Add source chunk information
                for entity in results.get('entities', []):
                    if isinstance(entity, dict):
                        entity['source_chunks'] = [chunk_id]

                for relation in results.get('relations', []):
                    if isinstance(relation, dict):
                        relation['source_chunks'] = [chunk_id]

                for fact in results.get('facts', []):
                    if isinstance(fact, dict):
                        fact['source_chunk'] = chunk_id

                return results

            except Exception as e:
                logger.warning("Failed to parse LLM extraction response", error=str(e), response_preview=response_text[:500])
                return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}

        except Exception as e:
            logger.warning("LLM extraction failed", error=str(e))
            return {'entities': [], 'relations': [], 'facts': [], 'keywords': []}

    def _get_extraction_system_prompt(self, config: FactGeneratorConfig) -> str:
        """Get system prompt for extraction.

        Args:
            config: Stage configuration

        Returns:
            System prompt string
        """
        domain = config.domain

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

        # Note: Entity and relation type prioritization can be added to FactGeneratorConfig if needed

        return prompt

    def _get_extraction_user_prompt(self, content: str, config: FactGeneratorConfig) -> str:
        """Get user prompt for extraction.

        Args:
            content: Content to extract from
            config: Stage configuration

        Returns:
            User prompt string
        """
        max_entities = getattr(config, 'max_entities_per_chunk', 20)
        max_relations = getattr(config, 'max_relations_per_chunk', 15)
        min_confidence = getattr(config, 'min_confidence', 0.7)

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
        word_counts: Dict[str, int] = {}
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
            # Extract entity names for normalization
            entity_names = [entity.get('name', '') for entity in entities]
            entity_types = [entity.get('type', '') or '' for entity in entities]

            # Use the correct method name: normalize_entities_batch
            normalized_variations = await self.entity_normalizer.normalize_entities_batch(entity_names, entity_types)

            # Update entities with normalized names
            normalized_entities = []
            for i, entity in enumerate(entities):
                if i < len(normalized_variations):
                    normalized_entity = entity.copy()
                    normalized_entity['name'] = normalized_variations[i].normalized
                    normalized_entity['normalization_confidence'] = normalized_variations[i].confidence
                    normalized_entities.append(normalized_entity)
                else:
                    normalized_entities.append(entity)

            return self._basic_entity_deduplication(normalized_entities)
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
