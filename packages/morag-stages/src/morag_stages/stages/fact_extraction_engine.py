"""Core fact extraction engine with AI-powered processing."""

import json
import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import structlog

from morag_core.config import FactGeneratorConfig

logger = structlog.get_logger(__name__)

# Import robust LLM response parser from agents framework
try:
    from agents.base import LLMResponseParser
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    LLMResponseParser = None

if TYPE_CHECKING:
    from morag_core.ai import Agent
    from morag_graph.extraction import FactExtractor, EntityNormalizer


class FactExtractionEngine:
    """Core engine for extracting facts, entities, and relations from text content."""
    
    def __init__(self):
        """Initialize fact extraction engine."""
        self.fact_extractor = None
        self.entity_normalizer = None
        self.agent = None
        self._initialized = False

    async def initialize(self, fact_extractor=None, entity_normalizer=None, agent=None):
        """Initialize the extraction engine with services."""
        self.fact_extractor = fact_extractor
        self.entity_normalizer = entity_normalizer
        self.agent = agent
        self._initialized = True
        logger.info("Fact extraction engine initialized")

    async def extract_from_chunks(self, chunks: List[Dict[str, Any]], config: FactGeneratorConfig) -> Dict[str, Any]:
        """Extract facts from a list of chunks."""
        if not self._initialized:
            await self.initialize()
        
        all_facts = []
        all_entities = []
        all_relations = []
        all_keywords = []
        
        # Process chunks individually or in batches
        if config.enable_batch_processing:
            batch_results = await self._process_chunks_in_batches(chunks, config)
        else:
            batch_results = []
            for i, chunk in enumerate(chunks):
                result = await self._extract_from_chunk(chunk, config)
                batch_results.append(result)
        
        # Aggregate results
        for result in batch_results:
            if result.get('success', False):
                all_facts.extend(result.get('facts', []))
                all_entities.extend(result.get('entities', []))
                all_relations.extend(result.get('relations', []))
                all_keywords.extend(result.get('keywords', []))
        
        # Apply deduplication and normalization
        if config.enable_fact_deduplication:
            all_facts = self._deduplicate_facts(all_facts)
        
        if config.enable_relation_deduplication:
            all_relations = self._deduplicate_relations(all_relations)
        
        if config.enable_entity_normalization and self.entity_normalizer:
            all_entities = await self._normalize_entities(all_entities)
        else:
            all_entities = self._basic_entity_deduplication(all_entities)
        
        return {
            'facts': all_facts,
            'entities': all_entities,
            'relations': all_relations,
            'keywords': all_keywords,
            'processing_metadata': {
                'chunks_processed': len(chunks),
                'extraction_method': 'service' if self.fact_extractor else 'llm_fallback'
            }
        }

    async def _process_chunks_in_batches(self, chunks: List[Dict[str, Any]], config: FactGeneratorConfig) -> List[Dict[str, Any]]:
        """Process chunks in batches for efficiency."""
        batch_size = config.max_chunks_per_batch
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_results = []
            
            for chunk in batch:
                result = await self._extract_from_chunk(chunk, config)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results

    async def _extract_from_chunk(self, chunk: Dict[str, Any], config: FactGeneratorConfig) -> Dict[str, Any]:
        """Extract facts, entities, relations, and keywords from a single chunk."""
        try:
            chunk_text = chunk.get('text', '')
            chunk_id = chunk.get('id', 'unknown')
            
            # Skip very short chunks
            if len(chunk_text.strip()) < config.chunk_size_threshold:
                logger.debug("Skipping short chunk", chunk_id=chunk_id, length=len(chunk_text))
                return {
                    'success': True,
                    'chunk_id': chunk_id,
                    'facts': [],
                    'entities': [],
                    'relations': [],
                    'keywords': []
                }
            
            # Filter out metadata content if needed
            filtered_content = self._filter_metadata_content(chunk_text)
            
            # Try service-based extraction first
            if self.fact_extractor:
                try:
                    service_result = await self.fact_extractor.extract_facts(
                        filtered_content,
                        chunk_metadata=chunk.get('metadata', {})
                    )
                    
                    if service_result and service_result.get('success', False):
                        # Extract keywords
                        keywords = self._extract_keywords(filtered_content) if config.enable_keyword_extraction else []
                        
                        return {
                            'success': True,
                            'chunk_id': chunk_id,
                            'facts': service_result.get('facts', []),
                            'entities': service_result.get('entities', []),
                            'relations': service_result.get('relations', []),
                            'keywords': keywords,
                            'extraction_method': 'service'
                        }
                        
                except Exception as e:
                    logger.warning("Service-based extraction failed, falling back to LLM", 
                                 chunk_id=chunk_id, error=str(e))
            
            # Fallback to LLM-based extraction
            if config.use_llm_fallback:
                fallback_result = await self._llm_extraction_fallback(filtered_content, chunk_id, config)
                if fallback_result.get('success', False):
                    return fallback_result
            
            # If everything fails, return empty results
            logger.warning("All extraction methods failed for chunk", chunk_id=chunk_id)
            return {
                'success': False,
                'chunk_id': chunk_id,
                'facts': [],
                'entities': [],
                'relations': [],
                'keywords': [],
                'error': 'All extraction methods failed'
            }
            
        except Exception as e:
            logger.error("Error extracting from chunk", chunk_id=chunk.get('id', 'unknown'), error=str(e))
            return {
                'success': False,
                'chunk_id': chunk.get('id', 'unknown'),
                'facts': [],
                'entities': [],
                'relations': [],
                'keywords': [],
                'error': str(e)
            }

    def _filter_metadata_content(self, content: str) -> str:
        """Filter out metadata and boilerplate content."""
        # Remove common metadata patterns
        patterns_to_remove = [
            r'Processing timestamp:.*?\n',
            r'File size:.*?\n',
            r'Page \d+ of \d+',
            r'Generated by.*?\n',
            r'Copyright.*?\n',
            r'\[Generated with.*?\]',
            r'Co-Authored-By:.*?\n'
        ]
        
        filtered_content = content
        for pattern in patterns_to_remove:
            filtered_content = re.sub(pattern, '', filtered_content, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        filtered_content = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_content)
        
        return filtered_content.strip()

    async def _llm_extraction_fallback(self, content: str, chunk_id: str, config: FactGeneratorConfig) -> Dict[str, Any]:
        """Fallback extraction using LLM agent."""
        try:
            if not self.agent:
                logger.warning("LLM agent not available for fallback extraction")
                return {'success': False, 'error': 'LLM agent not available'}
            
            # Build extraction prompt
            system_prompt = self._get_extraction_system_prompt(config)
            user_prompt = self._get_extraction_user_prompt(content, config)
            
            # Get LLM response
            response = await self.agent.generate(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            # Parse response
            extracted_data = self._parse_llm_response(response)
            
            if extracted_data:
                # Extract keywords
                keywords = self._extract_keywords(content) if config.enable_keyword_extraction else []
                
                return {
                    'success': True,
                    'chunk_id': chunk_id,
                    'facts': extracted_data.get('facts', []),
                    'entities': extracted_data.get('entities', []),
                    'relations': extracted_data.get('relations', []),
                    'keywords': keywords,
                    'extraction_method': 'llm_fallback'
                }
            else:
                return {
                    'success': False,
                    'chunk_id': chunk_id,
                    'error': 'Failed to parse LLM response'
                }
                
        except Exception as e:
            logger.error("LLM extraction failed", chunk_id=chunk_id, error=str(e))
            return {
                'success': False,
                'chunk_id': chunk_id,
                'error': str(e)
            }

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse structured response from LLM."""
        try:
            # Try to use robust parser if available
            if PARSER_AVAILABLE and LLMResponseParser:
                parser = LLMResponseParser()
                return parser.parse_json_response(response)
            
            # Fallback manual parsing
            # Look for JSON block in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Try parsing entire response as JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # Manual extraction as fallback
            return self._manual_parse_response(response)
            
        except Exception as e:
            logger.error("Error parsing LLM response", error=str(e))
            return None

    def _manual_parse_response(self, response: str) -> Dict[str, Any]:
        """Manually parse response when JSON parsing fails."""
        result = {
            'facts': [],
            'entities': [],
            'relations': []
        }
        
        # Simple pattern matching for facts, entities, relations
        # This is a basic fallback - real implementation would be more sophisticated
        
        # Extract facts (simple heuristic)
        fact_patterns = [
            r'(?:fact|claim):\s*(.+?)(?:\n|$)',
            r'\d+\.\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in fact_patterns:
            facts = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            for fact_text in facts:
                if len(fact_text.strip()) > 10:  # Filter very short facts
                    result['facts'].append({
                        'text': fact_text.strip(),
                        'confidence': 0.5,  # Default confidence for manual parsing
                        'extraction_method': 'manual_pattern'
                    })
        
        return result

    def _get_extraction_system_prompt(self, config: FactGeneratorConfig) -> str:
        """Generate system prompt for fact extraction."""
        return f"""You are an expert fact extractor. Your task is to extract structured information from text content.

Extract the following information in JSON format:
1. Facts: Key claims, statements, or assertions
2. Entities: People, places, organizations, concepts, etc.
3. Relations: Relationships between entities

Requirements:
- Only extract facts with confidence >= {config.min_fact_confidence}
- Only extract relations with confidence >= {config.min_relation_confidence}
- Each fact should be self-contained and meaningful
- Entities should be normalized (e.g., "John Smith" not "john smith")
- Relations should specify subject, relation, and object

Output format:
```json
{{
  "facts": [
    {{"text": "fact text", "confidence": 0.9, "type": "factual_claim"}},
    ...
  ],
  "entities": [
    {{"name": "Entity Name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT", "confidence": 0.8}},
    ...
  ],
  "relations": [
    {{"subject": "entity1", "relation": "relationship", "object": "entity2", "confidence": 0.7}},
    ...
  ]
}}
```"""

    def _get_extraction_user_prompt(self, content: str, config: FactGeneratorConfig) -> str:
        """Generate user prompt for fact extraction."""
        return f"""Extract facts, entities, and relations from the following text content:

{content}

Please provide the extraction results in the specified JSON format."""

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content using simple NLP techniques."""
        # Simple keyword extraction - could be enhanced with proper NLP
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with',
            'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been',
            'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just',
            'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
            'well', 'were', 'what'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:20] if freq >= 2]

    async def _normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize entities using the entity normalizer service."""
        if not self.entity_normalizer:
            return self._basic_entity_deduplication(entities)
        
        try:
            normalized = await self.entity_normalizer.normalize_batch(entities)
            return normalized
        except Exception as e:
            logger.warning("Entity normalization failed, using basic deduplication", error=str(e))
            return self._basic_entity_deduplication(entities)

    def _basic_entity_deduplication(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic entity deduplication by name similarity."""
        if not entities:
            return []
        
        deduplicated = []
        seen_names = set()
        
        for entity in entities:
            name = entity.get('name', '').strip().lower()
            if name and name not in seen_names:
                seen_names.add(name)
                deduplicated.append(entity)
        
        return deduplicated

    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relations by subject-relation-object triplets."""
        if not relations:
            return []
        
        deduplicated = []
        seen_triplets = set()
        
        for relation in relations:
            subject = relation.get('subject', '').strip().lower()
            rel = relation.get('relation', '').strip().lower()
            obj = relation.get('object', '').strip().lower()
            
            triplet = (subject, rel, obj)
            if triplet not in seen_triplets:
                seen_triplets.add(triplet)
                deduplicated.append(relation)
        
        return deduplicated

    def _deduplicate_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate facts by text similarity."""
        if not facts:
            return []
        
        deduplicated = []
        seen_texts = set()
        
        for fact in facts:
            text = fact.get('text', '').strip().lower()
            # Simple deduplication by exact text match
            # Could be enhanced with similarity matching
            if text and text not in seen_texts:
                seen_texts.add(text)
                deduplicated.append(fact)
        
        return deduplicated


__all__ = ["FactExtractionEngine"]