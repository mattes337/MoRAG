"""LangExtract-based entity extraction that replaces PydanticAI implementation."""

import structlog
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    lx = None

from ..models import Entity
from .entity_normalizer import LLMEntityNormalizer
from ..utils.retry_utils import retry_with_exponential_backoff
from ..utils.quota_retry import never_fail_extraction, retry_with_quota_handling

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """LangExtract-based entity extractor that replaces the PydanticAI implementation."""
    
    def __init__(
        self,
        min_confidence: float = 0.5,  # Lower threshold for better recall
        chunk_size: int = 800,  # Smaller chunks for better accuracy
        dynamic_types: bool = True,
        entity_types: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_workers: int = 15,  # More workers for parallel processing
        extraction_passes: int = 3,  # More passes for better coverage
        domain: str = "general"  # Default to general domain, configurable by user
    ):
        """Initialize the LangExtract entity extractor.

        Args:
            min_confidence: Minimum confidence threshold for entities
            chunk_size: Maximum characters per chunk
            dynamic_types: Whether to use dynamic entity types
            entity_types: Custom entity types dict
            language: Language code for processing
            model_id: LangExtract model ID
            api_key: API key for LangExtract
            max_workers: Number of parallel workers
            extraction_passes: Number of extraction passes
            domain: Domain for specialized extraction (general, medical, technical, etc.)
        """
        if not LANGEXTRACT_AVAILABLE:
            raise ImportError("LangExtract is not available. Please install it with: pip install langextract")
        
        self.min_confidence = min_confidence
        self.chunk_size = chunk_size
        self.dynamic_types = dynamic_types
        self.entity_types = entity_types or self._get_domain_entity_types(domain)
        self.language = language

        self.model_id = model_id
        self.api_key = api_key or self._get_api_key()
        self.max_workers = max_workers
        self.extraction_passes = extraction_passes
        self.domain = domain
        self.logger = logger.bind(component="langextract_entity_extractor")
        
        # Thread pool for async execution
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Create examples for entity extraction
        self._examples = self._create_entity_examples()
        self._prompt = self._create_entity_prompt()

        # Initialize entity normalizer
        self.normalizer = LLMEntityNormalizer(
            model_name=self.model_id,
            api_key=self.api_key,
            language=self.language or "auto"
        )
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        import os
        return os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    def _get_domain_entity_types(self, domain: str) -> Dict[str, str]:
        """Get entity types for the specified domain."""
        # Return generic entity types to avoid domain-specific bias
        return {
            "person": "Individual person or character",
            "organization": "Company, institution, or group",
            "location": "Place, city, country, or geographic location",
            "concept": "Abstract idea, theory, or principle",
            "object": "Physical item, tool, or artifact",
            "event": "Occurrence, happening, or activity"
        }
    
    def _create_entity_prompt(self) -> str:
        """Create prompt for entity extraction with normalization."""
        base_prompt = """Extract and normalize entities from the text in order of appearance.
        Focus on important entities like people, organizations, locations, concepts, and objects.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes for each entity to add context.

        For each entity:
        1. Extract exact mention from text
        2. Determine canonical form (normalized name)
        3. Identify entity type and attributes
        4. Assess confidence and provide disambiguation context

        IMPORTANT: Create SPECIFIC, DESCRIPTIVE entity types that reflect the actual nature of each entity.
        - Instead of generic types like "PERSON", use specific types like "RESEARCHER", "EXECUTIVE", "PATIENT", "AUTHOR"
        - Instead of generic types like "ORGANIZATION", use specific types like "UNIVERSITY", "HOSPITAL", "COMPANY", "GOVERNMENT_AGENCY"
        - Instead of generic types like "CONCEPT", use specific types like "MEDICAL_CONDITION", "TECHNOLOGY", "METHODOLOGY", "THEORY"
        - Instead of generic types like "LOCATION", use specific types like "CITY", "COUNTRY", "FACILITY", "REGION"

        The entity type should be:
        1. Specific to the domain and context
        2. Descriptive of the entity's role or nature
        3. Uppercase with underscores (e.g., "RESEARCH_INSTITUTION", "PHARMACEUTICAL_DRUG")
        4. Never generic (avoid "ENTITY", "THING", "ITEM", "OBJECT")"""

        if self.entity_types:
            type_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in self.entity_types.items()])
            base_prompt += f"\n\nConsider these domain-specific entity types as inspiration (but create more specific types as needed):\n{type_descriptions}"

        if self.domain and self.domain != "general":
            base_prompt += f"\n\nDomain context: {self.domain}. Generate entity types that are specific to this domain."

        if self.language:
            base_prompt += f"\n\nProcess text in {self.language} language."

        return base_prompt
    
    def _create_entity_examples(self) -> List[Any]:
        """Create few-shot examples for entity extraction."""
        # Use minimal generic examples to avoid domain-specific bias
        if not LANGEXTRACT_AVAILABLE:
            return []

        return [
            lx.data.ExampleData(
                text="The researcher works at the organization in the city.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="researcher",
                        attributes={
                            "role": "professional",
                            "canonical_name": "researcher",
                            "confidence": 0.9,
                            "disambiguation": "professional conducting research"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="organization",
                        extraction_text="organization",
                        attributes={
                            "type": "entity",
                            "canonical_name": "organization",
                            "confidence": 0.8,
                            "disambiguation": "workplace or institution"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="city",
                        attributes={
                            "type": "place",
                            "canonical_name": "city",
                            "confidence": 0.9,
                            "disambiguation": "urban location"
                        }
                    ),
                ]
            )
        ]
    
    async def extract(
        self,
        text: str,
        source_doc_id: Optional[str] = None,
        auto_infer_domain: bool = False
    ) -> List[Entity]:
        """Extract entities from text using LangExtract.

        Args:
            text: Text to extract entities from
            source_doc_id: Optional source document ID
            auto_infer_domain: Whether to automatically infer domain from text content

        Returns:
            List of Entity objects
        """
        
        if not text or not text.strip():
            return []
        
        if not self.api_key:
            self.logger.warning("No API key found for LangExtract. Set LANGEXTRACT_API_KEY or GOOGLE_API_KEY.")
            return []

        # Optionally infer domain from text content
        current_domain = self.domain
        if auto_infer_domain:
            inferred_domain = self._infer_domain_from_text(text)
            if inferred_domain != 'general':
                current_domain = inferred_domain
                self.logger.info(f"Inferred domain '{inferred_domain}' from text content")

        # Define the main extraction function
        async def main_extraction():
            return await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._extract_sync,
                text,
                source_doc_id
            )

        # Define fallback strategies that don't require API calls
        fallback_strategies = [
            lambda: self._create_basic_entities_from_text(text, source_doc_id),
            lambda: self._create_minimal_entities_from_text(text, source_doc_id)
        ]

        try:
            # Use never-fail extraction with quota handling
            result = await never_fail_extraction(
                main_extraction,
                fallback_strategies,
                operation_name="entity extraction"
            )

            # Handle empty result case
            if not result or not hasattr(result, 'extractions'):
                self.logger.warning(
                    "Entity extraction returned empty result, using fallback entities",
                    text_length=len(text),
                    source_doc_id=source_doc_id
                )
                # Return basic entities extracted from text patterns
                return self._create_basic_entities_from_text(text, source_doc_id)

            # Convert LangExtract results to MoRAG Entity objects
            entities = await self._convert_to_entities_async(result, source_doc_id)

            # Filter by confidence
            entities = [e for e in entities if e.confidence >= self.min_confidence]

            self.logger.info(
                "Entity extraction completed",
                text_length=len(text),
                entities_found=len(entities),
                source_doc_id=source_doc_id,
                langextract_extractions=len(result.extractions) if result else 0,
                domain=self.domain
            )

            return entities

        except Exception as e:
            # This should never happen with never_fail_extraction, but just in case
            self.logger.error(
                "Entity extraction failed completely, using emergency fallback",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                domain=self.domain,
                component="langextract_entity_extractor"
            )
            # Return basic entities as last resort
            return self._create_basic_entities_from_text(text, source_doc_id)
    
    def _extract_sync(self, text: str, source_doc_id: Optional[str]) -> Any:
        """Synchronous extraction using LangExtract with robust error handling."""
        try:
            return lx.extract(
                text_or_documents=text,
                prompt_description=self._prompt,
                examples=self._examples,
                model_id=self.model_id,
                api_key=self.api_key,
                max_workers=self.max_workers,
                extraction_passes=self.extraction_passes,
                max_char_buffer=self.chunk_size
            )
        except Exception as e:
            error_msg = str(e).lower()

            # Check if this is a JSON parsing error from langextract
            if any(keyword in error_msg for keyword in [
                'failed to parse content',
                'expecting value',
                'expecting \',\' delimiter',
                'json.decoder.jsondecodeerror',
                'invalid json'
            ]):
                self.logger.warning(
                    "LangExtract JSON parsing error, attempting fallback strategies",
                    error=str(e),
                    error_type=type(e).__name__,
                    text_length=len(text),
                    source_doc_id=source_doc_id
                )

                # Try fallback strategies
                return self._extract_with_fallback(text, source_doc_id, original_error=e)
            else:
                # Re-raise non-JSON parsing errors
                raise

    def _extract_with_fallback(self, text: str, source_doc_id: Optional[str], original_error: Exception) -> Any:
        """Implement fallback strategies when LangExtract JSON parsing fails."""
        fallback_strategies = [
            self._fallback_smaller_chunks,
            self._fallback_reduced_workers,
            self._fallback_single_pass,
            self._fallback_minimal_config
        ]

        for i, strategy in enumerate(fallback_strategies):
            try:
                self.logger.info(
                    f"Attempting fallback strategy {i+1}/{len(fallback_strategies)}",
                    strategy=strategy.__name__,
                    text_length=len(text),
                    source_doc_id=source_doc_id
                )

                result = strategy(text)

                self.logger.info(
                    f"Fallback strategy {i+1} succeeded",
                    strategy=strategy.__name__,
                    extractions_found=len(result.extractions) if result and hasattr(result, 'extractions') else 0
                )

                return result

            except Exception as e:
                self.logger.warning(
                    f"Fallback strategy {i+1} failed",
                    strategy=strategy.__name__,
                    error=str(e),
                    error_type=type(e).__name__
                )
                continue

        # If all fallback strategies fail, return empty result
        self.logger.error(
            "All fallback strategies failed, returning empty result",
            original_error=str(original_error),
            text_length=len(text),
            source_doc_id=source_doc_id
        )

        # Create a minimal result object that matches LangExtract's structure
        class FallbackResult:
            def __init__(self):
                self.extractions = []
                self.text = text

        return FallbackResult()

    def _fallback_smaller_chunks(self, text: str) -> Any:
        """Fallback strategy: Use smaller chunk size to reduce JSON complexity."""
        return lx.extract(
            text_or_documents=text,
            prompt_description=self._prompt,
            examples=self._examples,
            model_id=self.model_id,
            api_key=self.api_key,
            max_workers=self.max_workers,
            extraction_passes=self.extraction_passes,
            max_char_buffer=min(400, self.chunk_size // 2)  # Halve chunk size, minimum 400
        )

    def _fallback_reduced_workers(self, text: str) -> Any:
        """Fallback strategy: Reduce parallel workers to minimize race conditions."""
        return lx.extract(
            text_or_documents=text,
            prompt_description=self._prompt,
            examples=self._examples,
            model_id=self.model_id,
            api_key=self.api_key,
            max_workers=min(3, self.max_workers),  # Reduce to max 3 workers
            extraction_passes=self.extraction_passes,
            max_char_buffer=self.chunk_size
        )

    def _fallback_single_pass(self, text: str) -> Any:
        """Fallback strategy: Use single extraction pass for simplicity."""
        return lx.extract(
            text_or_documents=text,
            prompt_description=self._prompt,
            examples=self._examples,
            model_id=self.model_id,
            api_key=self.api_key,
            max_workers=1,  # Single worker
            extraction_passes=1,  # Single pass
            max_char_buffer=min(600, self.chunk_size)
        )

    def _fallback_minimal_config(self, text: str) -> Any:
        """Fallback strategy: Minimal configuration for maximum stability."""
        return lx.extract(
            text_or_documents=text,
            prompt_description=self._prompt,
            examples=self._examples,
            model_id=self.model_id,
            api_key=self.api_key,
            max_workers=1,
            extraction_passes=1,
            max_char_buffer=300  # Very small chunks
        )

    def _convert_to_entities(self, result: Any, source_doc_id: Optional[str]) -> List[Entity]:
        """Convert LangExtract results to MoRAG Entity objects."""
        entities = []
        
        if not result or not hasattr(result, 'extractions'):
            return entities
        
        for extraction in result.extractions:
            try:
                # Map LangExtract extraction to MoRAG Entity
                entity = Entity(
                    name=extraction.extraction_text,
                    type=extraction.extraction_class.upper(),
                    attributes=extraction.attributes or {},
                    source_doc_id=source_doc_id,
                    confidence=getattr(extraction, 'confidence', 1.0)
                )
                entities.append(entity)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to convert extraction to entity",
                    extraction_text=getattr(extraction, 'extraction_text', 'unknown'),
                    error=str(e)
                )
                continue
        
        return entities

    def _infer_domain_from_text(self, text: str) -> str:
        """Infer domain from text content using keyword analysis.

        Args:
            text: Text to analyze for domain inference

        Returns:
            Inferred domain string
        """
        text_lower = text.lower()

        # Medical/health keywords
        medical_keywords = [
            'patient', 'doctor', 'medical', 'health', 'disease', 'treatment', 'medication',
            'symptom', 'diagnosis', 'therapy', 'clinical', 'hospital', 'medicine',
            'toxin', 'detox', 'vitamin', 'mineral', 'supplement', 'enzyme', 'hormone',
            'schwermetall', 'entgiftung', 'schilddrüse', 'quecksilber', 'aluminium'
        ]

        # Technical keywords
        technical_keywords = [
            'system', 'software', 'database', 'server', 'network', 'algorithm',
            'programming', 'code', 'api', 'framework', 'technology', 'computer'
        ]

        # Legal keywords
        legal_keywords = [
            'court', 'judge', 'law', 'legal', 'contract', 'agreement', 'regulation',
            'statute', 'ruling', 'litigation', 'attorney', 'lawyer'
        ]

        # Business keywords
        business_keywords = [
            'company', 'business', 'market', 'sales', 'revenue', 'profit', 'customer',
            'strategy', 'management', 'finance', 'investment', 'corporate'
        ]

        # Count keyword matches
        medical_score = sum(1 for keyword in medical_keywords if keyword in text_lower)
        technical_score = sum(1 for keyword in technical_keywords if keyword in text_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in text_lower)
        business_score = sum(1 for keyword in business_keywords if keyword in text_lower)

        # Determine domain based on highest score
        scores = {
            'medical': medical_score,
            'technical': technical_score,
            'legal': legal_score,
            'business': business_score
        }

        max_score = max(scores.values())
        if max_score >= 3:  # Minimum threshold for domain inference
            return max(scores, key=scores.get)

        return 'general'  # Default to general if no clear domain

    def _is_generic_entity_name(self, name: str) -> bool:
        """Check if entity name is generic and should be avoided.

        Args:
            name: Entity name to check

        Returns:
            True if the name is generic and should be avoided
        """
        if not name or len(name.strip()) < 2:
            return True

        name_lower = name.lower().strip()

        # Check for exact generic names
        if name_lower in ['entity', 'unknown', 'placeholder']:
            return True

        # Generic patterns to avoid
        generic_patterns = [
            'entity_',
            'entity ',
            'unknown',
            'placeholder',
            'temp_',
            'auto_',
            'generated_'
        ]

        for pattern in generic_patterns:
            if name_lower.startswith(pattern):
                return True

        # Check if it's just "Entity" followed by an ID or hash
        if name_lower.startswith('entity') and ('_' in name or name.isdigit()):
            return True

        return False

    async def _convert_to_entities_async(self, result: Any, source_doc_id: Optional[str]) -> List[Entity]:
        """Convert LangExtract results to MoRAG Entity objects with normalization."""
        entities = []

        if not result or not hasattr(result, 'extractions'):
            return entities

        # First, extract all entity names and types for batch normalization
        entity_data = []
        for extraction in result.extractions:
            try:
                entity_data.append({
                    'extraction': extraction,
                    'name': extraction.extraction_text,
                    'type': extraction.extraction_class.upper()
                })
            except Exception as e:
                self.logger.warning(
                    "Failed to prepare extraction for normalization",
                    extraction_text=getattr(extraction, 'extraction_text', 'unknown'),
                    error=str(e)
                )
                continue

        # Normalize entities in batch
        entity_names = [data['name'] for data in entity_data]
        entity_types = [data['type'] for data in entity_data]

        if entity_names and self.normalizer:
            try:
                normalization_results = await self.normalizer.normalize_entities_batch(
                    entity_names, entity_types
                )
            except Exception as e:
                self.logger.warning(
                    "Batch normalization failed, using original names",
                    error=str(e)
                )
                normalization_results = None
        else:
            normalization_results = None

        # Create Entity objects with normalized names
        for i, data in enumerate(entity_data):
            try:
                extraction = data['extraction']
                original_name = data['name']

                # Use normalized name if available
                if normalization_results and i < len(normalization_results):
                    normalized_result = normalization_results[i]
                    entity_name = normalized_result.normalized
                    # Add normalization info to attributes
                    attributes = extraction.attributes or {}
                    attributes.update({
                        'original_name': original_name,
                        'normalization_confidence': normalized_result.confidence,
                        'normalization_rule': normalized_result.rule_applied
                    })
                else:
                    entity_name = original_name
                    attributes = extraction.attributes or {}

                # Skip entities with generic names
                if self._is_generic_entity_name(entity_name):
                    self.logger.debug(f"Skipping entity with generic name: '{entity_name}'")
                    continue

                # Create Entity object
                entity = Entity(
                    name=entity_name,
                    type=data['type'],
                    attributes=attributes,
                    source_doc_id=source_doc_id,
                    confidence=getattr(extraction, 'confidence', 1.0)
                )
                entities.append(entity)

            except Exception as e:
                self.logger.warning(
                    "Failed to convert extraction to entity",
                    extraction_text=getattr(extraction, 'extraction_text', 'unknown'),
                    error=str(e)
                )
                continue

        return entities

    async def extract_with_context(
        self,
        text: str,
        source_doc_id: Optional[str] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> List[Entity]:
        """Extract entities with additional context information.
        
        Args:
            text: Text to extract entities from
            source_doc_id: ID of the source document
            additional_context: Additional context to help with extraction
            **kwargs: Additional arguments
            
        Returns:
            List of Entity objects with context information
        """
        # Combine text with additional context if provided
        if additional_context:
            combined_text = f"{additional_context}\n\n{text}"
        else:
            combined_text = text
        
        return await self.extract(combined_text, source_doc_id=source_doc_id, **kwargs)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt used by LangExtract."""
        return self._prompt
    
    async def _normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Legacy method for compatibility - LangExtract handles normalization internally."""
        # LangExtract handles normalization internally, so just return entities as-is
        return entities

    def _create_basic_entities_from_text(self, text: str, source_doc_id: Optional[str]) -> List[Entity]:
        """Create basic entities using simple text patterns (no API calls).

        This is a fallback method that extracts entities using basic patterns
        when API calls fail due to quota exhaustion.
        """
        entities = []

        try:
            import re

            # Basic patterns for common entity types
            patterns = {
                "PERSON": [
                    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                    r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. First Last
                    r'\bProf\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Prof. First Last
                ],
                "ORGANIZATION": [
                    r'\b[A-Z][a-z]+ Inc\.\b',  # Company Inc.
                    r'\b[A-Z][a-z]+ Corp\.\b',  # Company Corp.
                    r'\b[A-Z][a-z]+ Ltd\.\b',  # Company Ltd.
                    r'\b[A-Z][a-z]+ GmbH\b',  # Company GmbH
                    r'\b[A-Z][a-z]+ AG\b',  # Company AG
                ],
                "LOCATION": [
                    r'\b[A-Z][a-z]+, [A-Z][a-z]+\b',  # City, State
                    r'\b[A-Z][a-z]+ University\b',  # University names
                ],
                "CONCEPT": [
                    r'\b[A-Z]{2,}\b',  # Acronyms
                    r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # Three-word concepts
                ]
            }

            for entity_type, type_patterns in patterns.items():
                for pattern in type_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        entity_name = match.group().strip()
                        if len(entity_name) > 2 and not self._is_generic_entity_name(entity_name):
                            entity = Entity(
                                name=entity_name,
                                type=entity_type,
                                confidence=0.3,  # Low confidence for pattern-based extraction
                                source_doc_id=source_doc_id,
                                attributes={
                                    "extraction_method": "pattern_fallback",
                                    "pattern": pattern,
                                    "position": match.start()
                                }
                            )
                            entities.append(entity)

            # Remove duplicates
            seen_names = set()
            unique_entities = []
            for entity in entities:
                if entity.name not in seen_names:
                    seen_names.add(entity.name)
                    unique_entities.append(entity)

            self.logger.info(
                "Created basic entities using pattern matching",
                entities_found=len(unique_entities),
                text_length=len(text),
                source_doc_id=source_doc_id
            )

            return unique_entities[:20]  # Limit to 20 entities

        except Exception as e:
            self.logger.warning(
                "Pattern-based entity extraction failed",
                error=str(e),
                text_length=len(text)
            )
            return []

    def _create_minimal_entities_from_text(self, text: str, source_doc_id: Optional[str]) -> List[Entity]:
        """Create minimal entities from capitalized words (absolute fallback).

        This is the most basic fallback that just extracts capitalized words.
        """
        entities = []

        try:
            import re

            # Extract capitalized words (potential proper nouns)
            capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)

            # Filter out common words
            common_words = {
                'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or', 'So',
                'When', 'Where', 'Why', 'How', 'What', 'Who', 'Which', 'While',
                'After', 'Before', 'During', 'Since', 'Until', 'Because', 'Although'
            }

            unique_words = []
            seen = set()
            for word in capitalized_words:
                if word not in common_words and word not in seen and len(word) > 2:
                    seen.add(word)
                    unique_words.append(word)

            # Create entities from unique capitalized words
            for word in unique_words[:10]:  # Limit to 10
                if not self._is_generic_entity_name(word):
                    entity = Entity(
                        name=word,
                        type="UNKNOWN",
                        confidence=0.1,  # Very low confidence
                        source_doc_id=source_doc_id,
                        attributes={
                            "extraction_method": "minimal_fallback",
                            "note": "Extracted from capitalized words"
                        }
                    )
                    entities.append(entity)

            self.logger.info(
                "Created minimal entities from capitalized words",
                entities_found=len(entities),
                text_length=len(text),
                source_doc_id=source_doc_id
            )

            return entities

        except Exception as e:
            self.logger.warning(
                "Minimal entity extraction failed",
                error=str(e),
                text_length=len(text)
            )
            return []
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
