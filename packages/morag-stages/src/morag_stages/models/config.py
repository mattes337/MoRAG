"""Configuration models for MoRAG Stages."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class AgentModelConfig(BaseModel):
    """Configuration for agent-specific LLM models."""
    default_model: Optional[str] = Field(default=None, description="Default LLM model for all agents")
    agent_models: Dict[str, str] = Field(default_factory=dict, description="Agent-specific model overrides")


class StageConfig(BaseModel):
    """Base configuration for all stages."""
    enabled: bool = Field(default=True, description="Whether this stage is enabled")
    timeout_seconds: Optional[int] = Field(default=None, description="Timeout for stage execution")
    retry_count: int = Field(default=0, description="Number of retries on failure")
    config: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific configuration")
    agent_model_config: Optional[AgentModelConfig] = Field(default=None, description="Agent model configuration")


class MarkdownConversionConfig(StageConfig):
    """Configuration for markdown conversion stage."""
    # Audio/Video processing
    include_timestamps: bool = Field(default=True, description="Include timestamps in transcription")
    transcription_model: str = Field(default="whisper-large", description="Transcription model to use")
    speaker_diarization: bool = Field(default=True, description="Enable speaker diarization")
    topic_segmentation: bool = Field(default=True, description="Enable topic segmentation")

    # Document processing
    chunk_on_sentences: bool = Field(default=True, description="Chunk documents on sentence boundaries")
    preserve_formatting: bool = Field(default=True, description="Preserve original formatting")

    # Output options
    generate_thumbnails: bool = Field(default=False, description="Generate video thumbnails")
    thumbnail_interval: int = Field(default=60, description="Thumbnail interval in seconds")


class MarkdownOptimizerConfig(StageConfig):
    """Configuration for markdown optimizer stage."""
    fix_transcription_errors: bool = Field(default=True, description="Fix transcription errors")
    improve_readability: bool = Field(default=True, description="Improve text readability")
    preserve_timestamps: bool = Field(default=True, description="Preserve timestamp information")
    enhance_structure: bool = Field(default=True, description="Enhance document structure")

    # LLM settings
    model_name: Optional[str] = Field(default=None, description="LLM model to use")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")


class ChunkerConfig(StageConfig):
    """Configuration for chunker stage."""
    chunk_size: int = Field(default=4000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    strategy: str = Field(default="semantic", description="Chunking strategy (semantic, page, topic)")

    # Semantic chunking
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for semantic chunking")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    max_chunk_size: int = Field(default=8000, description="Maximum chunk size")

    # Document-specific
    respect_boundaries: bool = Field(default=True, description="Respect document boundaries (pages, sections)")
    preserve_metadata: bool = Field(default=True, description="Preserve chunk metadata")


class FactGeneratorConfig(StageConfig):
    """Configuration for fact generator stage."""
    max_facts_per_chunk: int = Field(default=10, description="Maximum facts to extract per chunk")
    confidence_threshold: float = Field(default=0.3, description="Minimum confidence for fact extraction")

    # Entity extraction
    extract_entities: bool = Field(default=True, description="Extract entities from facts")
    entity_types: List[str] = Field(default_factory=list, description="Specific entity types to extract")

    # Relation extraction
    extract_relations: bool = Field(default=True, description="Extract relations between entities")
    relation_confidence: float = Field(default=0.4, description="Minimum confidence for relations")

    # LLM settings
    model_name: Optional[str] = Field(default=None, description="LLM model to use")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")


class IngestorConfig(StageConfig):
    """Configuration for ingestor stage."""
    # Database settings
    databases: List[str] = Field(default_factory=lambda: ["qdrant", "neo4j"], description="Target databases")
    collection_name: Optional[str] = Field(default=None, description="Collection/database name")

    # Processing options
    batch_size: int = Field(default=50, description="Batch size for ingestion")
    enable_deduplication: bool = Field(default=True, description="Enable deduplication")
    overwrite_existing: bool = Field(default=False, description="Overwrite existing documents")

    # Embedding settings
    embedding_model: Optional[str] = Field(default=None, description="Embedding model to use")
    embedding_batch_size: int = Field(default=100, description="Batch size for embeddings")

    # Validation
    validate_before_insert: bool = Field(default=True, description="Validate data before insertion")
    skip_on_error: bool = Field(default=False, description="Skip items that cause errors")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    markdown_conversion: MarkdownConversionConfig = Field(default_factory=MarkdownConversionConfig)
    markdown_optimizer: MarkdownOptimizerConfig = Field(default_factory=MarkdownOptimizerConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    fact_generator: FactGeneratorConfig = Field(default_factory=FactGeneratorConfig)
    ingestor: IngestorConfig = Field(default_factory=IngestorConfig)

    # Global settings
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for notifications")
    output_dir: str = Field(default="./output", description="Output directory")
    temp_dir: str = Field(default="./temp", description="Temporary directory")
    cleanup_temp_files: bool = Field(default=True, description="Clean up temporary files")

    # Execution settings
    max_parallel_stages: int = Field(default=1, description="Maximum parallel stage execution")
    resume_from_existing: bool = Field(default=True, description="Resume from existing files")
    fail_fast: bool = Field(default=False, description="Stop on first stage failure")

    def get_stage_config(self, stage_name: str) -> StageConfig:
        """Get configuration for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Stage configuration

        Raises:
            ValueError: If stage name is not recognized
        """
        stage_configs = {
            "markdown-conversion": self.markdown_conversion,
            "markdown-optimizer": self.markdown_optimizer,
            "chunker": self.chunker,
            "fact-generator": self.fact_generator,
            "ingestor": self.ingestor,
        }

        if stage_name not in stage_configs:
            raise ValueError(f"Unknown stage: {stage_name}")

        return stage_configs[stage_name]
