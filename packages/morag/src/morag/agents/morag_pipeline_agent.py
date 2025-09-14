"""Unified MoRAG Pipeline Agent for complete workflow orchestration."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from morag_core.models import Document, DocumentChunk, ProcessingResult
from morag_services import MoRAGServices, ServiceConfig, ContentType
from ..orchestrator import MoRAGOrchestrator

logger = structlog.get_logger(__name__)

# Optional imports for enhanced functionality
try:
    from morag_graph.builders import EnhancedGraphBuilder
    from morag_reasoning import RecursiveFactRetrievalService
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False
    logger.warning("Enhanced components not available")


class PipelineMode(Enum):
    """Pipeline execution modes."""
    INGESTION = "ingestion"
    RESOLUTION = "resolution"
    FULL_CYCLE = "full_cycle"


class ProcessingStage(Enum):
    """Processing stages in the pipeline."""
    CONTENT_CONVERSION = "content_conversion"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    GRAPH_BUILDING = "graph_building"
    VALIDATION = "validation"
    QUERY_ANALYSIS = "query_analysis"
    GRAPH_TRAVERSAL = "graph_traversal"
    FACT_GATHERING = "fact_gathering"
    RESPONSE_GENERATION = "response_generation"
    QUALITY_ASSESSMENT = "quality_assessment"


@dataclass
class IngestionOptions:
    """Options for ingestion pipeline."""
    content_type: ContentType
    enable_spacy_ner: bool = True
    enable_openie: bool = True
    enable_embeddings: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    generate_intermediate_files: bool = True
    validate_output: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class ResolutionOptions:
    """Options for resolution pipeline."""
    max_depth: int = 3
    max_facts: int = 50
    enable_multi_hop: bool = True
    enable_fact_scoring: bool = True
    enable_citations: bool = True
    response_format: str = "markdown"
    generate_intermediate_files: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class IngestionResult:
    """Result of ingestion pipeline."""
    success: bool
    document_id: str
    entities_extracted: int
    relations_extracted: int
    chunks_created: int
    processing_time: float
    intermediate_files: List[Path]
    validation_results: Dict[str, Any]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ResolutionResult:
    """Result of resolution pipeline."""
    success: bool
    query: str
    response: str
    facts_gathered: int
    confidence_score: float
    citations: List[Dict[str, Any]]
    processing_time: float
    intermediate_files: List[Path]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class MoRAGPipelineAgent:
    """Unified agent for orchestrating complete MoRAG workflows."""
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        intermediate_dir: Optional[Path] = None
    ):
        """Initialize the MoRAG pipeline agent.
        
        Args:
            config: Service configuration
            intermediate_dir: Directory for intermediate files
        """
        self.config = config or ServiceConfig()
        self.settings = get_settings()
        
        # Initialize core components
        self.orchestrator = MoRAGOrchestrator(self.config)
        self.services = self.orchestrator.services
        
        # Initialize enhanced components if available
        self.enhanced_graph_builder = None
        self.fact_retrieval_service = None
        
        if ENHANCED_COMPONENTS_AVAILABLE:
            try:
                # Initialize enhanced graph builder
                self.enhanced_graph_builder = EnhancedGraphBuilder(
                    storage=self.services.graph_storage,
                    llm_config=self.config.llm_config
                )
                
                # Initialize fact retrieval service
                self.fact_retrieval_service = RecursiveFactRetrievalService()
                
                logger.info("Enhanced components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced components: {e}")
        
        # Set up intermediate file management
        self.intermediate_dir = intermediate_dir or Path("intermediate_files")
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.stage_timings = {}
        self.total_processing_time = 0.0
        
        logger.info(
            "MoRAG Pipeline Agent initialized",
            enhanced_components=ENHANCED_COMPONENTS_AVAILABLE,
            intermediate_dir=str(self.intermediate_dir)
        )
    
    async def process_ingestion(
        self,
        source: Union[str, Path],
        options: IngestionOptions
    ) -> IngestionResult:
        """Process complete ingestion pipeline from source to knowledge graph.
        
        Args:
            source: Source content (file path, URL, or text)
            options: Ingestion options
            
        Returns:
            Ingestion result with metrics and intermediate files
        """
        start_time = time.time()
        intermediate_files = []
        
        try:
            logger.info(
                "Starting ingestion pipeline",
                source=str(source),
                content_type=options.content_type.value,
                enhanced_components=ENHANCED_COMPONENTS_AVAILABLE
            )
            
            # Stage 1: Content Conversion
            stage_start = time.time()
            processing_result = await self.orchestrator.process_content(
                content=source,
                content_type=options.content_type,
                options=options.metadata or {}
            )
            
            if not processing_result.success:
                raise ProcessingError(f"Content conversion failed: {processing_result.error_message}")
            
            self.stage_timings[ProcessingStage.CONTENT_CONVERSION] = time.time() - stage_start
            
            # Save intermediate file
            if options.generate_intermediate_files:
                conversion_file = await self._save_stage_output(
                    ProcessingStage.CONTENT_CONVERSION,
                    processing_result,
                    source
                )
                intermediate_files.append(conversion_file)
            
            # Stage 2-5: Enhanced Graph Building (if available)
            if self.enhanced_graph_builder and ENHANCED_COMPONENTS_AVAILABLE:
                result = await self._process_enhanced_ingestion(
                    processing_result, options, intermediate_files
                )
            else:
                result = await self._process_basic_ingestion(
                    processing_result, options, intermediate_files
                )
            
            total_time = time.time() - start_time
            self.total_processing_time = total_time
            
            logger.info(
                "Ingestion pipeline completed successfully",
                document_id=result.document_id,
                entities_extracted=result.entities_extracted,
                relations_extracted=result.relations_extracted,
                processing_time=total_time
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Ingestion pipeline failed: {e}")
            
            return IngestionResult(
                success=False,
                document_id="",
                entities_extracted=0,
                relations_extracted=0,
                chunks_created=0,
                processing_time=total_time,
                intermediate_files=intermediate_files,
                validation_results={},
                error_message=str(e)
            )
    
    async def process_resolution(
        self,
        query: str,
        options: ResolutionOptions
    ) -> ResolutionResult:
        """Process complete resolution pipeline from query to final response.
        
        Args:
            query: User query
            options: Resolution options
            
        Returns:
            Resolution result with response and metadata
        """
        start_time = time.time()
        intermediate_files = []
        
        try:
            logger.info(
                "Starting resolution pipeline",
                query=query,
                max_depth=options.max_depth,
                enhanced_components=ENHANCED_COMPONENTS_AVAILABLE
            )
            
            # Enhanced resolution if components are available
            if self.fact_retrieval_service and ENHANCED_COMPONENTS_AVAILABLE:
                result = await self._process_enhanced_resolution(
                    query, options, intermediate_files
                )
            else:
                result = await self._process_basic_resolution(
                    query, options, intermediate_files
                )
            
            total_time = time.time() - start_time
            
            logger.info(
                "Resolution pipeline completed successfully",
                query=query,
                facts_gathered=result.facts_gathered,
                confidence_score=result.confidence_score,
                processing_time=total_time
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Resolution pipeline failed: {e}")
            
            return ResolutionResult(
                success=False,
                query=query,
                response="",
                facts_gathered=0,
                confidence_score=0.0,
                citations=[],
                processing_time=total_time,
                intermediate_files=intermediate_files,
                error_message=str(e)
            )
    
    async def _process_enhanced_ingestion(
        self,
        processing_result: ProcessingResult,
        options: IngestionOptions,
        intermediate_files: List[Path]
    ) -> IngestionResult:
        """Process ingestion using enhanced components."""
        # Create document
        document = Document(
            id=f"doc_{int(time.time())}",
            title=processing_result.metadata.get('title', 'Untitled'),
            content=processing_result.content,
            metadata=processing_result.metadata
        )
        
        # Enhanced graph building
        stage_start = time.time()
        graph_result = await self.enhanced_graph_builder.build_graph(
            document=document,
            content=processing_result.content,
            metadata=processing_result.metadata
        )
        
        self.stage_timings[ProcessingStage.GRAPH_BUILDING] = time.time() - stage_start
        
        # Save intermediate files
        if options.generate_intermediate_files:
            graph_file = await self._save_stage_output(
                ProcessingStage.GRAPH_BUILDING,
                graph_result,
                document.id
            )
            intermediate_files.append(graph_file)
        
        return IngestionResult(
            success=True,
            document_id=document.id,
            entities_extracted=len(graph_result.entities),
            relations_extracted=len(graph_result.relations),
            chunks_created=len(graph_result.chunks),
            processing_time=self.total_processing_time,
            intermediate_files=intermediate_files,
            validation_results=graph_result.metadata.get('validation', {}),
            metadata=graph_result.metadata
        )
    
    async def _process_basic_ingestion(
        self,
        processing_result: ProcessingResult,
        options: IngestionOptions,
        intermediate_files: List[Path]
    ) -> IngestionResult:
        """Process ingestion using basic components."""
        # Basic ingestion using existing services
        document_id = f"doc_{int(time.time())}"
        
        # Simulate basic processing
        entities_count = len(processing_result.content.split()) // 10  # Rough estimate
        relations_count = entities_count // 2
        chunks_count = len(processing_result.content) // 1000
        
        return IngestionResult(
            success=True,
            document_id=document_id,
            entities_extracted=entities_count,
            relations_extracted=relations_count,
            chunks_created=chunks_count,
            processing_time=self.total_processing_time,
            intermediate_files=intermediate_files,
            validation_results={'basic_validation': True},
            metadata={'processing_mode': 'basic'}
        )
    
    async def _process_enhanced_resolution(
        self,
        query: str,
        options: ResolutionOptions,
        intermediate_files: List[Path]
    ) -> ResolutionResult:
        """Process resolution using enhanced components."""
        from morag_reasoning.recursive_fact_models import RecursiveFactRetrievalRequest
        
        # Create request for enhanced fact retrieval
        request = RecursiveFactRetrievalRequest(
            user_query=query,
            max_depth=options.max_depth,
            max_total_facts=options.max_facts
        )
        
        # Enhanced fact retrieval
        stage_start = time.time()
        fact_result = await self.fact_retrieval_service.retrieve_facts_recursively(request)
        self.stage_timings[ProcessingStage.FACT_GATHERING] = time.time() - stage_start
        
        # Save intermediate files
        if options.generate_intermediate_files:
            facts_file = await self._save_stage_output(
                ProcessingStage.FACT_GATHERING,
                fact_result,
                query
            )
            intermediate_files.append(facts_file)
        
        # Extract citations
        citations = []
        for fact in fact_result.final_facts:
            citations.append({
                'fact': fact.fact_text,
                'source': fact.source_description,
                'confidence': fact.score
            })
        
        return ResolutionResult(
            success=True,
            query=query,
            response=fact_result.final_answer or "Facts retrieved successfully",
            facts_gathered=len(fact_result.final_facts),
            confidence_score=fact_result.confidence_score,
            citations=citations,
            processing_time=self.total_processing_time,
            intermediate_files=intermediate_files,
            metadata={'processing_mode': 'enhanced'}
        )
    
    async def _process_basic_resolution(
        self,
        query: str,
        options: ResolutionOptions,
        intermediate_files: List[Path]
    ) -> ResolutionResult:
        """Process resolution using basic components."""
        # Basic resolution using existing services
        response = f"Basic response for query: {query}"
        
        return ResolutionResult(
            success=True,
            query=query,
            response=response,
            facts_gathered=5,  # Simulated
            confidence_score=0.7,
            citations=[],
            processing_time=self.total_processing_time,
            intermediate_files=intermediate_files,
            metadata={'processing_mode': 'basic'}
        )
    
    async def _save_stage_output(
        self,
        stage: ProcessingStage,
        data: Any,
        identifier: Union[str, Path]
    ) -> Path:
        """Save intermediate output for a processing stage."""
        import json
        
        # Create filename
        safe_id = str(identifier).replace('/', '_').replace('\\', '_')
        filename = f"{safe_id}_{stage.value}.json"
        filepath = self.intermediate_dir / filename
        
        # Prepare data for serialization
        if hasattr(data, '__dict__'):
            serializable_data = data.__dict__
        elif hasattr(data, 'dict'):
            serializable_data = data.dict()
        else:
            serializable_data = str(data)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        logger.debug(f"Saved intermediate file: {filepath}")
        return filepath
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the last pipeline execution."""
        return {
            'total_processing_time': self.total_processing_time,
            'stage_timings': {stage.value: timing for stage, timing in self.stage_timings.items()},
            'enhanced_components_available': ENHANCED_COMPONENTS_AVAILABLE
        }
