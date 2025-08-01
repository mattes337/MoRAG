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
from .langextract_examples import LangExtractExamples, DomainEntityTypes

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """LangExtract-based entity extractor that replaces the PydanticAI implementation."""
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        chunk_size: int = 1000,  # LangExtract optimal chunk size
        dynamic_types: bool = True,
        entity_types: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        model_id: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_workers: int = 10,
        extraction_passes: int = 2,
        domain: str = "general"
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
        
        # Legacy compatibility
        self.normalizer = None  # LangExtract handles normalization internally
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        import os
        return os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    def _get_domain_entity_types(self, domain: str) -> Dict[str, str]:
        """Get entity types for the specified domain."""
        domain_upper = domain.upper()
        if hasattr(DomainEntityTypes, domain_upper):
            return getattr(DomainEntityTypes, domain_upper)
        return DomainEntityTypes.GENERAL
    
    def _create_entity_prompt(self) -> str:
        """Create prompt for entity extraction."""
        base_prompt = """Extract entities from the text in order of appearance.
        Focus on important entities like people, organizations, locations, concepts, and objects.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes for each entity to add context."""
        
        if self.entity_types:
            type_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in self.entity_types.items()])
            base_prompt += f"\n\nFocus on these entity types:\n{type_descriptions}"
        
        if self.language:
            base_prompt += f"\n\nProcess text in {self.language} language."
        
        return base_prompt
    
    def _create_entity_examples(self) -> List[Any]:
        """Create few-shot examples for entity extraction."""
        try:
            return LangExtractExamples.get_entity_examples(self.domain)
        except Exception:
            # Fallback to basic examples if domain examples fail
            return [
                lx.data.ExampleData(
                    text="Dr. Sarah Johnson works as a researcher at Google in Mountain View, California.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="person",
                            extraction_text="Dr. Sarah Johnson",
                            attributes={"title": "Dr.", "role": "researcher"}
                        ),
                        lx.data.Extraction(
                            extraction_class="organization",
                            extraction_text="Google",
                            attributes={"type": "technology_company"}
                        ),
                        lx.data.Extraction(
                            extraction_class="location",
                            extraction_text="Mountain View",
                            attributes={"type": "city", "state": "California"}
                        ),
                    ]
                )
            ]
    
    async def extract(
        self,
        text: str,
        source_doc_id: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities from text using LangExtract.

        Args:
            text: Text to extract entities from
            source_doc_id: Optional source document ID

        Returns:
            List of Entity objects
        """
        
        if not text or not text.strip():
            return []
        
        if not self.api_key:
            self.logger.warning("No API key found for LangExtract. Set LANGEXTRACT_API_KEY or GOOGLE_API_KEY.")
            return []
        
        try:
            # Run LangExtract in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._extract_sync,
                text,
                source_doc_id
            )
            
            # Convert LangExtract results to MoRAG Entity objects
            entities = self._convert_to_entities(result, source_doc_id)
            
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
            self.logger.error(
                "Entity extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                domain=self.domain
            )
            raise
    
    def _extract_sync(self, text: str, source_doc_id: Optional[str]) -> Any:
        """Synchronous extraction using LangExtract."""
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
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
