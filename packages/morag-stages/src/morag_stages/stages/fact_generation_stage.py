"""Main fact generation stage orchestration."""

import json
from datetime import datetime
from typing import List, Dict, Any, TYPE_CHECKING
from pathlib import Path
import structlog

from morag_core.config import FactGeneratorConfig
from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

from .fact_extraction_engine import FactExtractionEngine

logger = structlog.get_logger(__name__)

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


class FactGeneratorStage(Stage):
    """Stage that extracts facts, entities, relations, and keywords from chunks."""

    def __init__(self, stage_type: StageType = StageType.FACT_GENERATOR):
        """Initialize fact generator stage."""
        super().__init__(stage_type)
        
        # Default configuration
        self.config = {
            'chunk_size_threshold': 100,
            'max_chunks_per_batch': 10,
            'enable_entity_normalization': True,
            'enable_fact_deduplication': True,
            'enable_relation_deduplication': True,
            'enable_keyword_extraction': True,
            'enable_quality_scoring': True,
            'min_fact_confidence': 0.3,
            'min_relation_confidence': 0.2,
            'enable_batch_processing': True,
            'fact_extraction_timeout': 300,
            'use_llm_fallback': True,
            'enable_extraction_caching': True,
            'extraction_cache_ttl': 3600,
        }
        
        # Initialize services
        self.fact_extractor = None
        self.entity_normalizer = None
        self.agent = None
        self.extraction_engine = FactExtractionEngine()

    async def _initialize_services_with_config(self, config: FactGeneratorConfig, context: 'StageContext'):
        """Initialize services with provided configuration."""
        try:
            if SERVICES_AVAILABLE:
                # Initialize fact extractor
                if FactExtractor:
                    self.fact_extractor = FactExtractor()
                    await self.fact_extractor.initialize()
                    logger.info("Fact extractor initialized")
                
                # Initialize entity normalizer
                if EntityNormalizer and config.enable_entity_normalization:
                    self.entity_normalizer = EntityNormalizer()
                    await self.entity_normalizer.initialize()
                    logger.info("Entity normalizer initialized")
                
                # Initialize AI agent
                if create_agent and AgentConfig:
                    agent_config = AgentConfig(
                        model=config.fact_extraction_agent_model,
                        temperature=0.1,
                        max_tokens=4000,
                        timeout=config.fact_extraction_timeout
                    )
                    self.agent = await create_agent(agent_config)
                    logger.info("Fact extraction agent initialized", model=config.fact_extraction_agent_model)
                
                # Initialize extraction engine
                await self.extraction_engine.initialize(
                    fact_extractor=self.fact_extractor,
                    entity_normalizer=self.entity_normalizer,
                    agent=self.agent
                )
                
            else:
                logger.warning("Fact extraction services not available, using fallback methods")
                await self.extraction_engine.initialize()
                
        except Exception as e:
            logger.error("Failed to initialize fact generation services", error=str(e))
            raise StageExecutionError(f"Service initialization failed: {str(e)}")

    async def execute(self, 
                     input_files: List[Path], 
                     output_dir: Path, 
                     context: StageContext = None) -> StageResult:
        """Execute fact generation on chunk files.
        
        Args:
            input_files: List of chunk.json files to process
            output_dir: Directory for output files  
            context: Stage execution context
            
        Returns:
            StageResult with fact generation results
        """
        if context is None:
            context = StageContext()

        # Get configuration
        config = FactGeneratorConfig.from_context(context)
        self.config.update(config.to_dict())

        stage_metadata = StageMetadata(
            stage_type=self.stage_type,
            start_time=datetime.now(),
            input_count=len(input_files),
            config=self.config
        )

        try:
            # Initialize services
            await self._initialize_services_with_config(config, context)
            
            # Validate inputs
            if not self.validate_inputs(input_files):
                return StageResult(
                    status=StageStatus.FAILED,
                    metadata=stage_metadata,
                    error="Input validation failed"
                )

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each chunk file
            results = []
            errors = []
            total_facts = 0
            total_entities = 0
            total_relations = 0
            
            for input_file in input_files:
                try:
                    logger.info("Processing chunk file", file=str(input_file))
                    
                    # Load chunks from JSON file
                    with open(input_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    chunks = data.get('chunks', [])
                    if not chunks:
                        logger.warning("No chunks found in file", file=str(input_file))
                        continue
                    
                    # Generate output filename
                    output_filename = input_file.stem + ".facts.json"
                    output_file = output_dir / output_filename
                    
                    # Extract facts from chunks
                    extracted_results = await self.extraction_engine.extract_from_chunks(chunks, config)
                    
                    # Compile results
                    file_results = {
                        'source_file': str(input_file),
                        'processing_timestamp': datetime.now().isoformat(),
                        'chunk_count': len(chunks),
                        'extraction_results': extracted_results,
                        'statistics': {
                            'total_facts': len(extracted_results.get('facts', [])),
                            'total_entities': len(extracted_results.get('entities', [])),
                            'total_relations': len(extracted_results.get('relations', [])),
                            'total_keywords': len(extracted_results.get('keywords', []))
                        }
                    }
                    
                    # Write results to file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(file_results, f, indent=2, ensure_ascii=False)
                    
                    # Update totals
                    total_facts += file_results['statistics']['total_facts']
                    total_entities += file_results['statistics']['total_entities']
                    total_relations += file_results['statistics']['total_relations']
                    
                    results.append({
                        'input_file': str(input_file),
                        'output_file': str(output_file),
                        'statistics': file_results['statistics']
                    })
                    
                    logger.info("Chunk file processed successfully", 
                              file=str(input_file),
                              output=str(output_file),
                              facts=file_results['statistics']['total_facts'])
                
                except Exception as e:
                    error_msg = f"Failed to process {input_file}: {str(e)}"
                    errors.append({
                        'input_file': str(input_file),
                        'error': error_msg
                    })
                    logger.error("Chunk file processing failed", 
                               file=str(input_file),
                               error=str(e))
            
            # Determine overall status
            total_files = len(input_files)
            successful_files = len(results)
            
            if successful_files == total_files:
                status = StageStatus.COMPLETED
            elif successful_files > 0:
                status = StageStatus.PARTIAL
            else:
                status = StageStatus.FAILED
            
            # Update metadata
            stage_metadata.end_time = datetime.now()
            stage_metadata.output_count = successful_files
            stage_metadata.success_rate = successful_files / total_files if total_files > 0 else 0.0
            stage_metadata.statistics = {
                'total_facts_extracted': total_facts,
                'total_entities_extracted': total_entities,
                'total_relations_extracted': total_relations
            }
            
            return StageResult(
                status=status,
                metadata=stage_metadata,
                outputs=results,
                errors=errors
            )
            
        except Exception as e:
            stage_metadata.end_time = datetime.now()
            logger.error("Stage execution failed", error=str(e))
            
            return StageResult(
                status=StageStatus.FAILED,
                metadata=stage_metadata,
                error=str(e)
            )

    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input chunk files."""
        if not input_files:
            logger.error("No input files provided")
            return False
        
        for input_file in input_files:
            if not input_file.exists():
                logger.error("Input file does not exist", file=str(input_file))
                return False
            
            if not input_file.suffix == '.json':
                logger.error("Input file is not a JSON file", file=str(input_file))
                return False
                
            # Validate file contains chunk data
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'chunks' not in data:
                    logger.error("Input file does not contain chunks", file=str(input_file))
                    return False
                    
            except Exception as e:
                logger.error("Error reading input file", file=str(input_file), error=str(e))
                return False
        
        return True

    def get_dependencies(self) -> List[StageType]:
        """Get list of stage dependencies."""
        return [StageType.CHUNKER]  # Fact generation depends on chunker

    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Get expected output files."""
        output_dir = context.output_dir if context else Path.cwd()
        outputs = []
        
        for input_file in input_files:
            output_filename = input_file.stem + ".facts.json"
            outputs.append(output_dir / output_filename)
        
        return outputs


__all__ = ["FactGeneratorStage"]