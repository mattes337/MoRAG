"""Extract structured facts from document chunks."""

import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import structlog

from morag_reasoning.llm import LLMClient
from ..models.fact import Fact, FactType
from .fact_prompts import FactExtractionPrompts, FactPromptTemplates
from .fact_validator import FactValidator


class FactExtractor:
    """Extract structured facts from document chunks."""
    
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        min_confidence: float = 0.7,
        max_facts_per_chunk: int = 10,
        domain: str = "general",
        language: str = "en",
        max_workers: int = 5
    ):
        """Initialize fact extractor with LLM and configuration.
        
        Args:
            model_id: LLM model to use for extraction
            api_key: API key for LLM service
            min_confidence: Minimum confidence threshold for facts
            max_facts_per_chunk: Maximum facts to extract per chunk
            domain: Domain context for extraction
            language: Language of the text
            max_workers: Maximum worker threads for parallel processing
        """
        self.model_id = model_id
        self.api_key = api_key
        self.min_confidence = min_confidence
        self.max_facts_per_chunk = max_facts_per_chunk
        self.domain = domain
        self.language = language
        self.max_workers = max_workers
        
        self.logger = structlog.get_logger(__name__)
        self.validator = FactValidator(min_confidence=min_confidence)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize LLM client
        from morag_reasoning.llm import LLMConfig
        llm_config = LLMConfig(
            provider="gemini",
            model=model_id,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2000
        )
        self.llm_client = LLMClient(llm_config)
    
    async def extract_facts(
        self, 
        chunk_text: str, 
        chunk_id: str,
        document_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Fact]:
        """Extract structured facts from a document chunk.
        
        Args:
            chunk_text: Text content to extract facts from
            chunk_id: Unique identifier for the chunk
            document_id: Identifier for the source document
            context: Optional context information (domain, metadata, etc.)
            
        Returns:
            List of validated Fact objects
        """
        if not chunk_text or not chunk_text.strip():
            return []
        
        # Update context from parameters
        extraction_context = {
            'domain': context.get('domain', self.domain) if context else self.domain,
            'language': context.get('language', self.language) if context else self.language,
            'chunk_id': chunk_id,
            'document_id': document_id
        }
        
        self.logger.info(
            "Starting fact extraction",
            chunk_id=chunk_id,
            document_id=document_id,
            text_length=len(chunk_text),
            domain=extraction_context['domain']
        )
        
        try:
            # Preprocess the text
            processed_text = self._preprocess_chunk(chunk_text)
            
            # Extract fact candidates using LLM
            fact_candidates = await self._extract_fact_candidates(processed_text, extraction_context)
            
            if not fact_candidates:
                self.logger.warning(
                    "No fact candidates extracted",
                    chunk_id=chunk_id,
                    text_length=len(chunk_text)
                )
                return []
            
            # Structure facts from candidates
            facts = self._structure_facts(fact_candidates, chunk_id, document_id, extraction_context)
            
            # Validate facts
            validated_facts = []
            for fact in facts:
                is_valid, issues = self.validator.validate_fact(fact)
                if is_valid:
                    validated_facts.append(fact)
                else:
                    self.logger.debug(
                        "Fact validation failed",
                        fact_id=fact.id,
                        issues=issues
                    )
            
            # Generate keywords for validated facts
            for fact in validated_facts:
                if not fact.keywords:
                    fact.keywords = self._generate_fact_keywords(fact)
            
            self.logger.info(
                "Fact extraction completed",
                chunk_id=chunk_id,
                candidates_found=len(fact_candidates),
                facts_structured=len(facts),
                facts_validated=len(validated_facts)
            )
            
            return validated_facts
            
        except Exception as e:
            self.logger.error(
                "Fact extraction failed",
                chunk_id=chunk_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return []
    
    def _preprocess_chunk(self, text: str) -> str:
        """Clean and prepare text for fact extraction.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text ready for extraction
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown artifacts that might confuse extraction
        text = re.sub(r'#{1,6}\s*', '', text)  # Headers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Bold/italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
        
        # Clean up common artifacts
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)  # Images
        
        return text.strip()
    
    async def _extract_fact_candidates(self, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to extract fact candidates from text.
        
        Args:
            text: Preprocessed text
            context: Extraction context
            
        Returns:
            List of fact candidate dictionaries
        """
        domain = context.get('domain', 'general')
        language = context.get('language', 'en')
        
        # Create extraction prompt
        prompt = FactExtractionPrompts.create_extraction_prompt(
            chunk_text=text,
            domain=domain,
            language=language,
            max_facts=self.max_facts_per_chunk
        )
        
        # Enhance prompt for specific domains
        if domain != 'general':
            prompt = FactPromptTemplates.get_domain_prompt(domain, prompt)
        
        try:
            # Call LLM for extraction
            response = await self.llm_client.generate(prompt)
            
            # Parse JSON response
            fact_candidates = self._parse_llm_response(response)
            
            self.logger.debug(
                "LLM fact extraction completed",
                candidates_found=len(fact_candidates),
                domain=domain
            )
            
            return fact_candidates
            
        except Exception as e:
            self.logger.error(
                "LLM fact extraction failed",
                error=str(e),
                domain=domain
            )
            return []
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract fact candidates.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of parsed fact dictionaries
        """
        try:
            # Try to find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                candidates = json.loads(json_str)
                
                if isinstance(candidates, list):
                    return candidates
            
            # Fallback: try to parse entire response as JSON
            candidates = json.loads(response)
            if isinstance(candidates, list):
                return candidates
            elif isinstance(candidates, dict):
                return [candidates]
            
        except json.JSONDecodeError as e:
            self.logger.warning(
                "Failed to parse LLM response as JSON",
                error=str(e),
                response_preview=response[:200]
            )
        
        return []
    
    def _structure_facts(
        self, 
        candidates: List[Dict[str, Any]], 
        chunk_id: str, 
        document_id: str,
        context: Dict[str, Any]
    ) -> List[Fact]:
        """Convert LLM output to structured Fact objects.
        
        Args:
            candidates: List of fact candidate dictionaries
            chunk_id: Source chunk identifier
            document_id: Source document identifier
            context: Extraction context
            
        Returns:
            List of structured Fact objects
        """
        facts = []
        
        for candidate in candidates:
            try:
                # Extract required fields
                subject = candidate.get('subject', '').strip()
                obj = candidate.get('object', '').strip()
                
                if not subject or not obj:
                    continue
                
                # Create fact with all available information
                fact_data = {
                    'subject': subject,
                    'object': obj,
                    'approach': candidate.get('approach', '').strip() or None,
                    'solution': candidate.get('solution', '').strip() or None,
                    'remarks': candidate.get('remarks', '').strip() or None,
                    'source_chunk_id': chunk_id,
                    'source_document_id': document_id,
                    'extraction_confidence': float(candidate.get('confidence', 0.8)),
                    'fact_type': candidate.get('fact_type', FactType.DEFINITION),
                    'domain': context.get('domain'),
                    'language': context.get('language', 'en'),
                    'keywords': candidate.get('keywords', []),
                    # Detailed source metadata from context
                    'source_file_path': context.get('source_file_path'),
                    'source_file_name': context.get('source_file_name'),
                    'page_number': context.get('page_number'),
                    'chapter_title': context.get('chapter_title'),
                    'chapter_index': context.get('chapter_index'),
                    'paragraph_index': context.get('paragraph_index'),
                    'timestamp_start': context.get('timestamp_start'),
                    'timestamp_end': context.get('timestamp_end'),
                    'topic_header': context.get('topic_header'),
                    'speaker_label': context.get('speaker_label'),
                    'source_text_excerpt': context.get('source_text_excerpt')
                }
                
                # Validate fact type
                if fact_data['fact_type'] not in FactType.all_types():
                    fact_data['fact_type'] = FactType.DEFINITION
                
                fact = Fact(**fact_data)
                facts.append(fact)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to structure fact candidate",
                    candidate=candidate,
                    error=str(e)
                )
                continue
        
        return facts
    
    def _generate_fact_keywords(self, fact: Fact) -> List[str]:
        """Generate keywords for fact indexing.
        
        Args:
            fact: Fact to generate keywords for
            
        Returns:
            List of keywords
        """
        # Combine all text content
        text_parts = [fact.subject, fact.object]
        if fact.approach:
            text_parts.append(fact.approach)
        if fact.solution:
            text_parts.append(fact.solution)
        if fact.remarks:
            text_parts.append(fact.remarks)
        
        combined_text = ' '.join(text_parts).lower()
        
        # Extract meaningful words (simple keyword extraction)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'man', 'men', 'oil', 'run', 'say', 'she', 'too', 'use', 'way',
            'with', 'this', 'that', 'they', 'have', 'from', 'will', 'been', 'each',
            'which', 'their', 'said', 'there', 'what', 'were', 'when', 'where'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Remove duplicates and limit count
        unique_keywords = list(dict.fromkeys(keywords))[:10]
        
        return unique_keywords
