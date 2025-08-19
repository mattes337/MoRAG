"""Configuration models for stages."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from .stage import StageType


class StageConfig(BaseModel):
    """Configuration for a single stage."""
    
    stage_type: StageType = Field(description="Type of stage this config applies to")
    enabled: bool = Field(default=True, description="Whether this stage is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific configuration")
    timeout_seconds: Optional[float] = Field(default=None, description="Stage timeout in seconds")
    retry_count: int = Field(default=0, description="Number of retries on failure")
    
    model_config = {
        "json_encoders": {
            StageType: lambda v: v.value
        }
    }


class MarkdownConversionConfig(BaseModel):
    """Configuration for markdown-conversion stage."""
    
    include_timestamps: bool = Field(default=True, description="Include timestamps in transcription")
    preserve_formatting: bool = Field(default=True, description="Preserve original formatting")
    transcription_model: str = Field(default="whisper-large", description="Transcription model to use")
    language: Optional[str] = Field(default=None, description="Language hint for transcription")
    speaker_diarization: bool = Field(default=True, description="Enable speaker diarization")
    topic_segmentation: bool = Field(default=True, description="Enable topic segmentation")
    
    # Document processing
    ocr_enabled: bool = Field(default=True, description="Enable OCR for scanned documents")
    extract_tables: bool = Field(default=True, description="Extract tables from documents")
    extract_images: bool = Field(default=False, description="Extract and describe images")
    
    # Web processing
    follow_links: bool = Field(default=False, description="Follow links in web content")
    max_depth: int = Field(default=1, description="Maximum link following depth")
    
    # Video processing
    extract_thumbnails: bool = Field(default=False, description="Extract video thumbnails")
    thumbnail_interval: int = Field(default=60, description="Thumbnail extraction interval in seconds")


class MarkdownOptimizerConfig(BaseModel):
    """Configuration for markdown-optimizer stage."""
    
    model: str = Field(default="gemini-pro", description="LLM model to use")
    max_tokens: int = Field(default=8192, description="Maximum tokens for LLM")
    temperature: float = Field(default=0.1, description="LLM temperature")
    
    # Optimization settings
    fix_transcription_errors: bool = Field(default=True, description="Fix transcription errors")
    improve_structure: bool = Field(default=True, description="Improve document structure")
    preserve_timestamps: bool = Field(default=True, description="Preserve timestamp information")
    preserve_metadata: bool = Field(default=True, description="Preserve metadata headers")
    
    # Content type specific
    content_type_prompts: Dict[str, str] = Field(
        default_factory=dict, 
        description="Custom prompts for different content types"
    )


class ChunkerConfig(BaseModel):
    """Configuration for chunker stage."""
    
    chunk_strategy: str = Field(default="semantic", description="Chunking strategy")
    chunk_size: int = Field(default=4000, description="Target chunk size in characters")
    overlap: int = Field(default=200, description="Overlap between chunks in characters")
    
    # Summary generation
    generate_summary: bool = Field(default=True, description="Generate document summary")
    summary_max_tokens: int = Field(default=1000, description="Maximum tokens for summary")
    
    # Embedding settings
    embedding_model: str = Field(default="text-embedding-004", description="Embedding model")
    batch_size: int = Field(default=50, description="Batch size for embedding generation")
    
    # Context settings
    include_context: bool = Field(default=True, description="Include surrounding context in chunks")
    context_window: int = Field(default=2, description="Number of surrounding chunks for context")
    
    @field_validator('chunk_strategy')
    @classmethod
    def validate_chunk_strategy(cls, v):
        valid_strategies = ['semantic', 'page-level', 'topic-based', 'fixed-size']
        if v not in valid_strategies:
            raise ValueError(f"chunk_strategy must be one of {valid_strategies}")
        return v


class FactGeneratorConfig(BaseModel):
    """Configuration for fact-generator stage."""
    
    extract_entities: bool = Field(default=True, description="Extract entities")
    extract_relations: bool = Field(default=True, description="Extract relations")
    extract_keywords: bool = Field(default=True, description="Extract keywords")
    extract_facts: bool = Field(default=True, description="Extract factual statements")
    
    # Domain settings
    domain: str = Field(default="general", description="Domain for extraction")
    entity_types: Optional[List[str]] = Field(default=None, description="Specific entity types to extract")
    relation_types: Optional[List[str]] = Field(default=None, description="Specific relation types to extract")
    
    # Quality settings
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")
    max_entities_per_chunk: int = Field(default=20, description="Maximum entities per chunk")
    max_relations_per_chunk: int = Field(default=15, description="Maximum relations per chunk")
    
    # LLM settings
    model: str = Field(default="gemini-pro", description="LLM model for extraction")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens for LLM")


class IngestorConfig(BaseModel):
    """Configuration for ingestor stage."""
    
    databases: List[str] = Field(default=["qdrant"], description="Target databases")
    collection_name: str = Field(default="documents", description="Collection/index name")
    batch_size: int = Field(default=50, description="Batch size for ingestion")
    
    # Deduplication
    enable_deduplication: bool = Field(default=True, description="Enable deduplication")
    dedup_threshold: float = Field(default=0.95, description="Similarity threshold for deduplication")
    
    # Database specific settings
    qdrant_config: Dict[str, Any] = Field(default_factory=dict, description="Qdrant-specific config")
    neo4j_config: Dict[str, Any] = Field(default_factory=dict, description="Neo4j-specific config")
    
    # Conflict resolution
    conflict_resolution: str = Field(default="merge", description="How to handle conflicts")
    
    @field_validator('databases')
    @classmethod
    def validate_databases(cls, v):
        valid_dbs = ['qdrant', 'neo4j', 'elasticsearch', 'pinecone']
        for db in v:
            if db not in valid_dbs:
                raise ValueError(f"Database '{db}' not supported. Valid options: {valid_dbs}")
        return v

    @field_validator('conflict_resolution')
    @classmethod
    def validate_conflict_resolution(cls, v):
        valid_strategies = ['merge', 'replace', 'skip', 'error']
        if v not in valid_strategies:
            raise ValueError(f"conflict_resolution must be one of {valid_strategies}")
        return v


class PipelineConfig(BaseModel):
    """Configuration for entire pipeline."""
    
    stages: List[StageConfig] = Field(description="Configuration for each stage")
    
    # Global settings
    output_dir: Path = Field(description="Base output directory")
    webhook_url: Optional[str] = Field(default=None, description="Webhook for notifications")
    resume_from_existing: bool = Field(default=True, description="Resume from existing files")
    cleanup_intermediate: bool = Field(default=False, description="Clean up intermediate files")
    max_parallel_stages: int = Field(default=1, description="Maximum parallel stages")
    
    # Stage-specific configs
    markdown_conversion: MarkdownConversionConfig = Field(default_factory=MarkdownConversionConfig)
    markdown_optimizer: MarkdownOptimizerConfig = Field(default_factory=MarkdownOptimizerConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    fact_generator: FactGeneratorConfig = Field(default_factory=FactGeneratorConfig)
    ingestor: IngestorConfig = Field(default_factory=IngestorConfig)
    
    model_config = {
        "json_encoders": {
            Path: lambda v: str(v)
        }
    }
    
    def get_stage_config(self, stage_type: StageType) -> Optional[StageConfig]:
        """Get configuration for a specific stage.
        
        Args:
            stage_type: Stage type to get config for
            
        Returns:
            Stage configuration if found, None otherwise
        """
        for stage_config in self.stages:
            if stage_config.stage_type == stage_type:
                return stage_config
        return None
    
    def is_stage_enabled(self, stage_type: StageType) -> bool:
        """Check if a stage is enabled.
        
        Args:
            stage_type: Stage type to check
            
        Returns:
            True if stage is enabled, False otherwise
        """
        stage_config = self.get_stage_config(stage_type)
        return stage_config is None or stage_config.enabled
