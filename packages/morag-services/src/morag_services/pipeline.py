"""Pipeline module for MoRAG services.

This module provides classes for building content processing pipelines
that can chain multiple processing steps together.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
import structlog

from morag_core.exceptions import ProcessingError
from .services import MoRAGServices, ProcessingResult, ContentType

logger = structlog.get_logger(__name__)

T = TypeVar('T')
U = TypeVar('U')

class PipelineStepType(str, Enum):
    """Types of pipeline steps."""
    PROCESS = "process"  # Process content (file/URL)
    TRANSFORM = "transform"  # Transform content (e.g., text extraction)
    EMBED = "embed"  # Generate embeddings
    FILTER = "filter"  # Filter content
    CUSTOM = "custom"  # Custom processing function

@dataclass
class PipelineContext:
    """Context for pipeline execution.
    
    This class stores intermediate results and metadata during pipeline execution.
    """
    # Input data
    input_paths: List[str] = field(default_factory=list)
    input_urls: List[str] = field(default_factory=list)
    input_texts: List[str] = field(default_factory=list)
    
    # Processing results
    results: Dict[str, ProcessingResult] = field(default_factory=dict)
    
    # Extracted data
    texts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    
    # Pipeline metadata
    current_step: int = 0
    total_steps: int = 0
    errors: Dict[str, str] = field(default_factory=dict)
    
    def add_error(self, item_id: str, error_message: str):
        """Add error message for an item."""
        self.errors[item_id] = error_message
        logger.error(f"Pipeline error for {item_id}: {error_message}")

@dataclass
class PipelineStep(Generic[T, U]):
    """A step in a processing pipeline.
    
    Each step takes input data, processes it, and produces output data.
    """
    name: str
    step_type: PipelineStepType
    process_fn: Callable[[T, PipelineContext], Awaitable[U]]
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    async def execute(self, data: T, context: PipelineContext) -> U:
        """Execute this pipeline step.
        
        Args:
            data: Input data for this step
            context: Pipeline context
            
        Returns:
            Processed output data
        """
        try:
            logger.info(f"Executing pipeline step: {self.name}")
            result = await self.process_fn(data, context)
            logger.info(f"Completed pipeline step: {self.name}")
            return result
        except Exception as e:
            logger.exception(f"Error in pipeline step: {self.name}", error=str(e))
            if isinstance(data, str):
                context.add_error(data, str(e))
            elif hasattr(data, "__iter__") and not isinstance(data, dict):
                for item in data:
                    if isinstance(item, str):
                        context.add_error(item, str(e))
            raise ProcessingError(f"Error in pipeline step '{self.name}': {str(e)}")

class Pipeline:
    """Content processing pipeline.
    
    A pipeline consists of multiple steps that are executed in sequence.
    Each step takes the output of the previous step as input.
    """
    
    def __init__(self, services: MoRAGServices, name: str = "default"):
        """Initialize pipeline.
        
        Args:
            services: MoRAG services instance
            name: Pipeline name
        """
        self.services = services
        self.name = name
        self.steps: List[PipelineStep] = []
        self.max_concurrent_items = services.config.max_concurrent_tasks
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a step to the pipeline.
        
        Args:
            step: Pipeline step to add
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self
    
    def process_content(self, name: str = "process_content") -> 'Pipeline':
        """Add a content processing step.
        
        This step processes files or URLs based on their content type.
        
        Args:
            name: Step name
            
        Returns:
            Self for method chaining
        """
        async def process_fn(item: str, context: PipelineContext) -> ProcessingResult:
            return await self.services.process_content(item)
        
        step = PipelineStep(
            name=name,
            step_type=PipelineStepType.PROCESS,
            process_fn=process_fn
        )
        
        return self.add_step(step)
    
    def extract_text(self, name: str = "extract_text") -> 'Pipeline':
        """Add a text extraction step.
        
        This step extracts text from processing results.
        
        Args:
            name: Step name
            
        Returns:
            Self for method chaining
        """
        async def process_fn(results: Dict[str, ProcessingResult], context: PipelineContext) -> Dict[str, str]:
            texts = {}
            for item_id, result in results.items():
                if result.success and result.text_content:
                    texts[item_id] = result.text_content
            return texts
        
        step = PipelineStep(
            name=name,
            step_type=PipelineStepType.TRANSFORM,
            process_fn=process_fn,
            input_key="results",
            output_key="texts"
        )
        
        return self.add_step(step)
    
    def extract_metadata(self, name: str = "extract_metadata") -> 'Pipeline':
        """Add a metadata extraction step.
        
        This step extracts metadata from processing results.
        
        Args:
            name: Step name
            
        Returns:
            Self for method chaining
        """
        async def process_fn(results: Dict[str, ProcessingResult], context: PipelineContext) -> Dict[str, Dict[str, Any]]:
            metadata = {}
            for item_id, result in results.items():
                if result.success and result.metadata:
                    metadata[item_id] = result.metadata
            return metadata
        
        step = PipelineStep(
            name=name,
            step_type=PipelineStepType.TRANSFORM,
            process_fn=process_fn,
            input_key="results",
            output_key="metadata"
        )
        
        return self.add_step(step)
    
    def generate_embeddings(self, name: str = "generate_embeddings") -> 'Pipeline':
        """Add an embedding generation step.
        
        This step generates embeddings for extracted texts.
        
        Args:
            name: Step name
            
        Returns:
            Self for method chaining
        """
        async def process_fn(texts: Dict[str, str], context: PipelineContext) -> Dict[str, Union[List[float], float]]:
            embeddings = {}
            
            # Process in batches to avoid overwhelming the embedding service
            items = list(texts.items())
            batch_size = 10  # Adjust based on embedding service capacity
            
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i+batch_size]
                batch_texts = [text for _, text in batch_items]
                batch_ids = [item_id for item_id, _ in batch_items]
                
                # Generate embeddings for batch
                batch_embeddings = await self.services.generate_embeddings(batch_texts)
                
                # Map embeddings back to item IDs
                for j, item_id in enumerate(batch_ids):
                    embeddings[item_id] = batch_embeddings[j]
            
            return embeddings
        
        step = PipelineStep(
            name=name,
            step_type=PipelineStepType.EMBED,
            process_fn=process_fn,
            input_key="texts",
            output_key="embeddings"
        )
        
        return self.add_step(step)
    
    def custom_step(self, 
                   name: str, 
                   process_fn: Callable[[Any, PipelineContext], Awaitable[Any]],
                   input_key: Optional[str] = None,
                   output_key: Optional[str] = None,
                   config: Optional[Dict[str, Any]] = None) -> 'Pipeline':
        """Add a custom processing step.
        
        Args:
            name: Step name
            process_fn: Processing function
            input_key: Key for input data in context
            output_key: Key for output data in context
            config: Step configuration
            
        Returns:
            Self for method chaining
        """
        step = PipelineStep(
            name=name,
            step_type=PipelineStepType.CUSTOM,
            process_fn=process_fn,
            input_key=input_key,
            output_key=output_key,
            config=config or {}
        )
        
        return self.add_step(step)
    
    async def execute(self, 
                     input_paths: Optional[List[str]] = None,
                     input_urls: Optional[List[str]] = None,
                     input_texts: Optional[List[str]] = None) -> PipelineContext:
        """Execute the pipeline.
        
        Args:
            input_paths: List of file paths to process
            input_urls: List of URLs to process
            input_texts: List of text strings to process
            
        Returns:
            Pipeline context with results
        """
        # Initialize context
        context = PipelineContext(
            input_paths=input_paths or [],
            input_urls=input_urls or [],
            input_texts=input_texts or [],
            total_steps=len(self.steps)
        )
        
        # Combine all inputs
        all_inputs = []
        all_inputs.extend(context.input_paths)
        all_inputs.extend(context.input_urls)
        
        # Execute each step
        for i, step in enumerate(self.steps):
            context.current_step = i + 1
            logger.info(f"Starting pipeline step {i+1}/{len(self.steps)}: {step.name}")
            
            try:
                # Determine input data for this step
                if i == 0 and step.step_type == PipelineStepType.PROCESS:
                    # First step with content processing
                    input_data = all_inputs
                elif step.input_key is not None:
                    # Get input from context using input_key
                    if step.input_key in context.__dict__:
                        input_data = getattr(context, step.input_key)
                    else:
                        raise ProcessingError(f"Input key '{step.input_key}' not found in context")
                else:
                    # Use previous step's output
                    input_data = None
                
                # Execute step
                output_data = await step.execute(input_data, context)
                
                # Store output in context
                if step.output_key is not None:
                    setattr(context, step.output_key, output_data)
                
                # Special handling for first processing step
                if i == 0 and step.step_type == PipelineStepType.PROCESS:
                    context.results = output_data
                
                logger.info(f"Completed pipeline step {i+1}/{len(self.steps)}: {step.name}")
            except Exception as e:
                logger.exception(f"Error in pipeline step {step.name}", error=str(e))
                # Continue with next step if possible
        
        return context
    
    async def process_batch(self, items: List[str]) -> PipelineContext:
        """Process a batch of items through the pipeline.
        
        Args:
            items: List of file paths or URLs
            
        Returns:
            Pipeline context with results
        """
        # Categorize items as paths or URLs
        paths = [item for item in items if not item.startswith("http")]
        urls = [item for item in items if item.startswith("http")]
        
        return await self.execute(input_paths=paths, input_urls=urls)
    
    @classmethod
    def create_default_pipeline(cls, services: MoRAGServices) -> 'Pipeline':
        """Create a default pipeline with common processing steps.
        
        Args:
            services: MoRAG services instance
            
        Returns:
            Configured pipeline
        """
        pipeline = cls(services, name="default")
        
        # Add standard processing steps
        pipeline.process_content()
        pipeline.extract_text()
        pipeline.extract_metadata()
        pipeline.generate_embeddings()
        
        return pipeline