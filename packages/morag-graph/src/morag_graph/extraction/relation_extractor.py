"""LangExtract-based relation extraction that replaces PydanticAI implementation."""

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

from ..models import Entity, Relation
from .langextract_examples import LangExtractExamples, DomainRelationTypes
from ..utils.retry_utils import retry_with_exponential_backoff
from ..utils.quota_retry import never_fail_extraction, retry_with_quota_handling

logger = structlog.get_logger(__name__)


class RelationExtractor:
    """LangExtract-based relation extractor that replaces the PydanticAI implementation."""
    
    def __init__(
        self,
        min_confidence: float = 0.5,  # Lower threshold for better recall
        chunk_size: int = 800,  # Smaller chunks for better accuracy
        dynamic_types: bool = True,
        relation_types: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_workers: int = 15,  # More workers for parallel processing
        extraction_passes: int = 3,  # More passes for better coverage
        domain: str = "general"  # Default to general domain, configurable by user
    ):
        """Initialize the LangExtract relation extractor.

        Args:
            min_confidence: Minimum confidence threshold for relations
            chunk_size: Maximum characters per chunk
            dynamic_types: Whether to use dynamic relation types
            relation_types: Custom relation types dict
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
        self.relation_types = relation_types or self._get_domain_relation_types(domain)
        self.language = language
        self.model_id = model_id
        self.api_key = api_key or self._get_api_key()
        self.max_workers = max_workers
        self.extraction_passes = extraction_passes
        self.domain = domain
        self.logger = logger.bind(component="langextract_relation_extractor")
        
        # Thread pool for async execution
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Create examples for relation extraction
        self._examples = self._create_relation_examples()
        self._prompt = self._create_relation_prompt()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        import os
        return os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    def _get_domain_relation_types(self, domain: str) -> Dict[str, str]:
        """Get relation types for the specified domain."""
        domain_upper = domain.upper()
        if hasattr(DomainRelationTypes, domain_upper):
            return getattr(DomainRelationTypes, domain_upper)
        return DomainRelationTypes.GENERAL
    
    def _create_relation_prompt(self) -> str:
        """Create prompt for relation extraction."""
        base_prompt = """Extract relationships between entities in the text.
        Focus on meaningful connections between people, organizations, locations, and concepts.
        Use exact text spans for relationships. Provide context and attributes.
        Each relationship should connect two specific entities mentioned in the text.

        IMPORTANT: Create SPECIFIC, DESCRIPTIVE relationship types that precisely describe the nature of the relationship.
        - Instead of generic types like "RELATES_TO", use specific types like "EMPLOYS", "RESEARCHES", "TREATS", "DEVELOPS"
        - Instead of generic types like "ASSOCIATED_WITH", use specific types like "COLLABORATES_WITH", "COMPETES_WITH", "SUPPLIES_TO"
        - Instead of generic types like "CONNECTED_TO", use specific types like "MANAGES", "OWNS", "FOUNDED", "ACQUIRED"

        The relationship type should be:
        1. A specific action or connection (e.g., "DIAGNOSES", "PRESCRIBES", "MANUFACTURES")
        2. Descriptive of the exact nature of the relationship
        3. Uppercase with underscores (e.g., "IS_EMPLOYED_BY", "CONDUCTS_RESEARCH_ON", "IS_LOCATED_IN")
        4. Never generic (avoid "RELATES", "CONNECTS", "LINKS", "MENTIONS")
        5. Express the relationship from source to target entity clearly"""

        if self.relation_types:
            type_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in self.relation_types.items()])
            base_prompt += f"\n\nConsider these domain-specific relationship types as inspiration (but create more specific types as needed):\n{type_descriptions}"

        if self.domain and self.domain != "general":
            base_prompt += f"\n\nDomain context: {self.domain}. Generate relationship types that are specific to this domain."

        if self.language:
            base_prompt += f"\n\nProcess text in {self.language} language."

        return base_prompt
    
    def _create_relation_examples(self) -> List[Any]:
        """Create few-shot examples for relation extraction."""
        try:
            return LangExtractExamples.get_relation_examples(self.domain)
        except Exception:
            # Fallback to basic examples if domain examples fail
            return [
                lx.data.ExampleData(
                    text="Dr. Sarah Johnson works as a researcher at Google in Mountain View, California.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="employment",
                            extraction_text="Dr. Sarah Johnson works as a researcher at Google",
                            attributes={
                                "source_entity": "Dr. Sarah Johnson",
                                "target_entity": "Google",
                                "relationship_type": "WORKS_FOR",
                                "role": "researcher"
                            }
                        ),
                    ]
                )
            ]
    
    async def extract(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations from text using LangExtract.

        Args:
            text: Text to extract relations from
            entities: Optional list of known entities
            source_doc_id: Optional source document ID

        Returns:
            List of Relation objects
        """
        
        if not text or not text.strip():
            return []
        
        if not self.api_key:
            self.logger.warning("No API key found for LangExtract. Set LANGEXTRACT_API_KEY or GOOGLE_API_KEY.")
            return []
        
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
            lambda: self._create_basic_relations_from_entities(entities, text, source_doc_id),
            lambda: self._create_minimal_relations_from_text(text, entities, source_doc_id)
        ]

        try:
            # Use never-fail extraction with quota handling
            result = await never_fail_extraction(
                main_extraction,
                fallback_strategies,
                operation_name="relation extraction"
            )

            # Handle empty result case
            if not result or not hasattr(result, 'extractions'):
                self.logger.warning(
                    "Relation extraction returned empty result, using fallback relations",
                    text_length=len(text),
                    num_entities=len(entities) if entities else 0,
                    source_doc_id=source_doc_id
                )
                # Return basic relations from entities
                return self._create_basic_relations_from_entities(entities, text, source_doc_id)

            # Convert LangExtract results to MoRAG Relation objects
            relations = self._convert_to_relations(result, entities, source_doc_id)

            # Filter by confidence
            relations = [r for r in relations if r.confidence >= self.min_confidence]

            self.logger.info(
                "Relation extraction completed",
                text_length=len(text),
                num_entities=len(entities) if entities else 0,
                relations_found=len(relations),
                source_doc_id=source_doc_id,
                langextract_extractions=len(result.extractions) if result else 0,
                domain=self.domain
            )

            return relations

        except Exception as e:
            # This should never happen with never_fail_extraction, but just in case
            self.logger.error(
                "Relation extraction failed completely, using emergency fallback",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                domain=self.domain,
                component="langextract_relation_extractor"
            )
            # Return basic relations as last resort
            return self._create_basic_relations_from_entities(entities, text, source_doc_id)
    
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

    def _convert_to_relations(
        self, 
        result: Any, 
        entities: Optional[List[Entity]], 
        source_doc_id: Optional[str]
    ) -> List[Relation]:
        """Convert LangExtract results to MoRAG Relation objects."""
        relations = []
        
        if not result or not hasattr(result, 'extractions'):
            return relations
        
        # Create comprehensive entity lookup for ID resolution
        # Include both normalized names and original names to handle entity normalization
        entity_lookup = {}
        if entities:
            for entity in entities:
                # Add normalized name (current entity.name)
                entity_lookup[entity.name.lower()] = entity.id

                # Add original name if available from normalization
                if hasattr(entity, 'attributes') and entity.attributes:
                    original_name = entity.attributes.get('original_name')
                    if original_name and original_name.lower() != entity.name.lower():
                        entity_lookup[original_name.lower()] = entity.id
                        self.logger.debug(
                            f"Added entity lookup mapping: '{original_name}' -> '{entity.name}' (ID: {entity.id})",
                            component="langextract_relation_extractor"
                        )
        
        for extraction in result.extractions:
            try:
                attrs = extraction.attributes or {}

                # Extract source and target entities from attributes
                source_entity_name = attrs.get('source_entity', '')
                target_entity_name = attrs.get('target_entity', '')
                relationship_type = attrs.get('relationship_type', extraction.extraction_class.upper())

                if not source_entity_name or not target_entity_name:
                    continue

                # Split comma-separated entities and create multiple relations
                source_entities = self._split_entity_names(source_entity_name)
                target_entities = self._split_entity_names(target_entity_name)

                # Create relations for all combinations of source and target entities
                for source_name in source_entities:
                    for target_name in target_entities:
                        # Try to resolve entity IDs with multiple strategies
                        source_entity_id = self._resolve_entity_id(source_name, entity_lookup)
                        target_entity_id = self._resolve_entity_id(target_name, entity_lookup)

                        # Skip relation if we can't resolve entity IDs properly
                        if not source_entity_id or not target_entity_id:
                            self.logger.debug(
                                f"Skipping relation - could not resolve entity IDs for '{source_name}' -> '{target_name}'",
                                available_entities=list(entity_lookup.keys())[:10],  # Show first 10 for debugging
                                component="langextract_relation_extractor"
                            )
                            continue

                        # Create updated attributes for this specific relation
                        relation_attrs = attrs.copy()
                        relation_attrs['source_entity'] = source_name
                        relation_attrs['target_entity'] = target_name

                        # Create relation
                        relation = Relation(
                            source_entity_id=source_entity_id,
                            target_entity_id=target_entity_id,
                            type=relationship_type,
                            context=extraction.extraction_text,
                            attributes=relation_attrs,
                            source_doc_id=source_doc_id,
                            confidence=getattr(extraction, 'confidence', 1.0)
                        )
                        relations.append(relation)

            except Exception as e:
                self.logger.warning(
                    "Failed to convert extraction to relation",
                    extraction_text=getattr(extraction, 'extraction_text', 'unknown'),
                    error=str(e)
                )
                continue
        
        return relations

    def _resolve_entity_id(self, entity_name: str, entity_lookup: Dict[str, str]) -> Optional[str]:
        """Resolve entity ID with multiple strategies to handle normalization."""
        if not entity_name:
            return None

        # Strategy 1: Direct lookup (exact match)
        entity_id = entity_lookup.get(entity_name.lower())
        if entity_id:
            return entity_id

        # Strategy 2: Try basic normalization (remove common words, trim)
        normalized_name = self._basic_normalize_entity_name(entity_name)
        entity_id = entity_lookup.get(normalized_name.lower())
        if entity_id:
            self.logger.debug(
                f"Resolved entity via basic normalization: '{entity_name}' -> '{normalized_name}'",
                component="langextract_relation_extractor"
            )
            return entity_id

        # Strategy 3: Partial matching (contains or is contained)
        entity_name_lower = entity_name.lower()
        for lookup_name, lookup_id in entity_lookup.items():
            # Check if entity name is contained in lookup name or vice versa
            if (entity_name_lower in lookup_name and len(entity_name_lower) > 3) or \
               (lookup_name in entity_name_lower and len(lookup_name) > 3):
                self.logger.debug(
                    f"Resolved entity via partial matching: '{entity_name}' -> '{lookup_name}'",
                    component="langextract_relation_extractor"
                )
                return lookup_id

        return None

    def _basic_normalize_entity_name(self, entity_name: str) -> str:
        """Apply basic normalization to entity name for matching."""
        normalized = entity_name.strip()

        # Remove common German/English words that might cause mismatches
        words_to_remove = [
            'sich', 'zu', 'der', 'die', 'das', 'den', 'dem', 'des',
            'the', 'a', 'an', 'to', 'of', 'for', 'with', 'by'
        ]

        words = normalized.split()
        filtered_words = [word for word in words if word.lower() not in words_to_remove]

        if filtered_words:
            normalized = ' '.join(filtered_words)

        return normalized

    def _split_entity_names(self, entity_name: str) -> List[str]:
        """Split comma-separated entity names into individual entities.

        Args:
            entity_name: Entity name that may contain multiple comma-separated entities

        Returns:
            List of individual entity names, cleaned and stripped
        """
        if not entity_name:
            return []

        # Split by comma and clean up each entity name
        entities = []
        for name in entity_name.split(','):
            cleaned_name = name.strip()
            if cleaned_name:  # Only add non-empty names
                entities.append(cleaned_name)

        # If no commas found, return the original name as a single-item list
        if not entities:
            entities = [entity_name.strip()]

        return entities

    def _create_basic_relations_from_entities(
        self,
        entities: Optional[List[Entity]],
        text: str,
        source_doc_id: Optional[str]
    ) -> List[Relation]:
        """Create basic relations from entities using simple patterns (no API calls).

        This is a fallback method that creates relations using basic patterns
        when API calls fail due to quota exhaustion.
        """
        relations = []

        if not entities or len(entities) < 2:
            return relations

        try:
            import re

            # Basic relation patterns
            relation_patterns = [
                (r'(\w+(?:\s+\w+)*)\s+(?:works?\s+(?:at|for)|is\s+employed\s+by)\s+(\w+(?:\s+\w+)*)', 'WORKS_FOR'),
                (r'(\w+(?:\s+\w+)*)\s+(?:founded|established|created)\s+(\w+(?:\s+\w+)*)', 'FOUNDED'),
                (r'(\w+(?:\s+\w+)*)\s+(?:owns|possesses)\s+(\w+(?:\s+\w+)*)', 'OWNS'),
                (r'(\w+(?:\s+\w+)*)\s+(?:manages|leads|heads)\s+(\w+(?:\s+\w+)*)', 'MANAGES'),
                (r'(\w+(?:\s+\w+)*)\s+(?:is\s+located\s+in|is\s+based\s+in)\s+(\w+(?:\s+\w+)*)', 'LOCATED_IN'),
                (r'(\w+(?:\s+\w+)*)\s+(?:collaborates?\s+with|partners?\s+with)\s+(\w+(?:\s+\w+)*)', 'COLLABORATES_WITH'),
            ]

            # Create entity name lookup
            entity_names = {entity.name.lower(): entity for entity in entities}

            for pattern, relation_type in relation_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_name = match.group(1).strip()
                    target_name = match.group(2).strip()

                    # Check if both entities exist in our entity list
                    source_entity = entity_names.get(source_name.lower())
                    target_entity = entity_names.get(target_name.lower())

                    if source_entity and target_entity:
                        relation = Relation(
                            source_entity_id=source_entity.id,
                            target_entity_id=target_entity.id,
                            type=relation_type,
                            context=match.group(0),
                            confidence=0.3,  # Low confidence for pattern-based extraction
                            source_doc_id=source_doc_id,
                            attributes={
                                "extraction_method": "pattern_fallback",
                                "pattern": pattern,
                                "position": match.start()
                            }
                        )
                        relations.append(relation)

            # If no pattern-based relations found, create proximity-based relations
            if not relations and len(entities) >= 2:
                # Create relations between entities that appear close to each other
                for i, entity1 in enumerate(entities[:5]):  # Limit to first 5 entities
                    for entity2 in entities[i+1:i+3]:  # Connect to next 2 entities
                        relation = Relation(
                            source_entity_id=entity1.id,
                            target_entity_id=entity2.id,
                            type="MENTIONED_WITH",
                            context=f"{entity1.name} and {entity2.name} mentioned together",
                            confidence=0.2,  # Very low confidence
                            source_doc_id=source_doc_id,
                            attributes={
                                "extraction_method": "proximity_fallback",
                                "note": "Entities mentioned in proximity"
                            }
                        )
                        relations.append(relation)

            self.logger.info(
                "Created basic relations using pattern matching",
                relations_found=len(relations),
                num_entities=len(entities),
                text_length=len(text),
                source_doc_id=source_doc_id
            )

            return relations[:10]  # Limit to 10 relations

        except Exception as e:
            self.logger.warning(
                "Pattern-based relation extraction failed",
                error=str(e),
                text_length=len(text)
            )
            return []

    def _create_minimal_relations_from_text(
        self,
        text: str,
        entities: Optional[List[Entity]],
        source_doc_id: Optional[str]
    ) -> List[Relation]:
        """Create minimal relations from text (absolute fallback).

        This is the most basic fallback that creates simple co-occurrence relations.
        """
        relations = []

        if not entities or len(entities) < 2:
            return relations

        try:
            # Create simple co-occurrence relations between first few entities
            for i, entity1 in enumerate(entities[:3]):  # Limit to first 3 entities
                for entity2 in entities[i+1:i+2]:  # Connect to next entity only
                    relation = Relation(
                        source_entity_id=entity1.id,
                        target_entity_id=entity2.id,
                        type="CO_OCCURS_WITH",
                        context=f"Co-occurrence in document",
                        confidence=0.1,  # Very low confidence
                        source_doc_id=source_doc_id,
                        attributes={
                            "extraction_method": "minimal_fallback",
                            "note": "Basic co-occurrence relation"
                        }
                    )
                    relations.append(relation)

            self.logger.info(
                "Created minimal co-occurrence relations",
                relations_found=len(relations),
                num_entities=len(entities),
                text_length=len(text),
                source_doc_id=source_doc_id
            )

            return relations

        except Exception as e:
            self.logger.warning(
                "Minimal relation extraction failed",
                error=str(e),
                text_length=len(text)
            )
            return []


    def get_system_prompt(self) -> str:
        """Get the system prompt used by LangExtract."""
        return self._prompt
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
