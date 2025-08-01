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

logger = structlog.get_logger(__name__)


class RelationExtractor:
    """LangExtract-based relation extractor that replaces the PydanticAI implementation."""
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        chunk_size: int = 1000,
        dynamic_types: bool = True,
        relation_types: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        model_id: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_workers: int = 10,
        extraction_passes: int = 2,
        domain: str = "general"
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
        Each relationship should connect two specific entities mentioned in the text."""
        
        if self.relation_types:
            type_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in self.relation_types.items()])
            base_prompt += f"\n\nFocus on these relationship types:\n{type_descriptions}"
        
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
        
        try:
            # Run LangExtract in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._extract_sync,
                text,
                source_doc_id
            )
            
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
            self.logger.error(
                "Relation extraction failed",
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
        
        # Create entity lookup for ID resolution
        entity_lookup = {}
        if entities:
            for entity in entities:
                entity_lookup[entity.name.lower()] = entity.id
        
        for extraction in result.extractions:
            try:
                attrs = extraction.attributes or {}
                
                # Extract source and target entities from attributes
                source_entity_name = attrs.get('source_entity', '')
                target_entity_name = attrs.get('target_entity', '')
                relationship_type = attrs.get('relationship_type', extraction.extraction_class.upper())
                
                if not source_entity_name or not target_entity_name:
                    continue
                
                # Try to resolve entity IDs
                source_entity_id = entity_lookup.get(source_entity_name.lower())
                target_entity_id = entity_lookup.get(target_entity_name.lower())
                
                # If we can't resolve IDs, create placeholder IDs
                if not source_entity_id:
                    source_entity_id = f"entity_{hash(source_entity_name.lower())}"
                if not target_entity_id:
                    target_entity_id = f"entity_{hash(target_entity_name.lower())}"
                
                # Create relation
                relation = Relation(
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    type=relationship_type,
                    context=extraction.extraction_text,
                    attributes=attrs,
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
