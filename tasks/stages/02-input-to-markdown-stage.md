# Task 2: Implement markdown-conversion Stage

## Overview
Implement the markdown-conversion stage that converts input files (video/audio/document) to unified markdown format with metadata. This completely replaces all existing input processing.

## Objectives
- **COMPLETELY REPLACE** existing services with new stage interface
- Implement unified markdown output format with metadata headers
- Support all current input types (video, audio, documents, web, YouTube)
- Add stage-specific configuration options
- **REMOVE ALL LEGACY PROCESSING CODE** - no backwards compatibility

## Deliverables

### 1. markdown-conversion Stage Implementation (Complete Replacement)
```python
from morag_stages.models import Stage, StageType, StageResult, StageContext, StageStatus
from morag_services import MoRAGServices
from pathlib import Path
from typing import List, Dict, Any
import time
import asyncio

class MarkdownConversionStage(Stage):
    def __init__(self, services: MoRAGServices):
        super().__init__(StageType.MARKDOWN_CONVERSION)
        self.services = services
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Convert input file to markdown format."""
        start_time = time.time()
        
        try:
            if len(input_files) != 1:
                raise ValueError("Stage 1 requires exactly one input file")
            
            input_file = input_files[0]
            
            # Determine content type
            content_type = self._detect_content_type(input_file)
            
            # Process content using appropriate service
            processing_result = await self._process_content(input_file, content_type, context)
            
            # Generate markdown output
            output_file = self._generate_markdown_output(
                processing_result, input_file, context
            )
            
            # Create metadata file
            metadata_file = self._create_metadata_file(
                processing_result, input_file, context
            )
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file, metadata_file],
                metadata={
                    "content_type": content_type,
                    "source_file": str(input_file),
                    "processing_time": execution_time,
                    "word_count": len(processing_result.content.split()),
                    "character_count": len(processing_result.content)
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.FAILED,
                output_files=[],
                metadata={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for Stage 1."""
        if len(input_files) != 1:
            return False
        
        input_file = input_files[0]
        
        # Check if file exists
        if not input_file.exists():
            return False
        
        # Check if file type is supported
        supported_extensions = {
            '.mp4', '.avi', '.mov', '.mkv',  # Video
            '.mp3', '.wav', '.flac', '.m4a',  # Audio
            '.pdf', '.docx', '.txt', '.md',   # Documents
        }
        
        return input_file.suffix.lower() in supported_extensions or \
               str(input_file).startswith(('http://', 'https://'))
    
    def get_dependencies(self) -> List[StageType]:
        """markdown-conversion has no dependencies."""
        return []
    
    def _detect_content_type(self, input_file: Path) -> str:
        """Detect content type from file extension or URL."""
        if str(input_file).startswith(('http://', 'https://')):
            if 'youtube.com' in str(input_file) or 'youtu.be' in str(input_file):
                return 'youtube'
            return 'web'
        
        extension = input_file.suffix.lower()
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        document_extensions = {'.pdf', '.docx', '.txt', '.md', '.html'}
        
        if extension in video_extensions:
            return 'video'
        elif extension in audio_extensions:
            return 'audio'
        elif extension in document_extensions:
            return 'document'
        else:
            return 'unknown'
    
    async def _process_content(self, 
                              input_file: Path, 
                              content_type: str, 
                              context: StageContext) -> Any:
        """Process content using appropriate service."""
        config = context.config.get('stage1', {})
        
        if content_type == 'video':
            return await self.services.video_service.process_file(
                input_file,
                include_timestamps=config.get('include_timestamps', True),
                transcription_model=config.get('transcription_model', 'whisper-large')
            )
        elif content_type == 'audio':
            return await self.services.audio_service.process_file(
                input_file,
                include_timestamps=config.get('include_timestamps', True),
                transcription_model=config.get('transcription_model', 'whisper-large')
            )
        elif content_type == 'document':
            return await self.services.document_service.process_file(input_file)
        elif content_type == 'web':
            return await self.services.web_service.process_url(str(input_file))
        elif content_type == 'youtube':
            return await self.services.youtube_service.process_video(str(input_file))
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    
    def _generate_markdown_output(self, 
                                 processing_result: Any, 
                                 input_file: Path, 
                                 context: StageContext) -> Path:
        """Generate unified markdown output with metadata header."""
        from morag_stages.file_manager import FileNamingConvention
        
        output_filename = FileNamingConvention.get_stage_output_filename(
            input_file, StageType.MARKDOWN_CONVERSION
        )
        output_path = context.output_dir / output_filename
        
        # Create markdown content with metadata header
        markdown_content = self._create_markdown_with_metadata(
            processing_result, input_file
        )
        
        # Write to file
        output_path.write_text(markdown_content, encoding='utf-8')
        
        return output_path
    
    def _create_markdown_with_metadata(self, 
                                      processing_result: Any, 
                                      input_file: Path) -> str:
        """Create markdown content with YAML metadata header."""
        import yaml
        from datetime import datetime
        
        # Extract metadata from processing result
        metadata = {
            'title': processing_result.metadata.get('title', input_file.stem),
            'source_file': str(input_file),
            'processed_at': datetime.utcnow().isoformat(),
            'content_type': processing_result.metadata.get('content_type', 'unknown'),
            'duration': processing_result.metadata.get('duration'),
            'language': processing_result.metadata.get('language'),
            'word_count': len(processing_result.content.split()),
            'character_count': len(processing_result.content)
        }
        
        # Add content-specific metadata
        if hasattr(processing_result, 'thumbnails'):
            metadata['thumbnails'] = len(processing_result.thumbnails)
        
        if hasattr(processing_result, 'chapters'):
            metadata['chapters'] = len(processing_result.chapters)
        
        # Create YAML frontmatter
        yaml_header = yaml.dump(metadata, default_flow_style=False)
        
        # Combine metadata and content
        markdown_content = f"""---
{yaml_header}---

{processing_result.content}
"""
        
        return markdown_content
    
    def _create_metadata_file(self, 
                             processing_result: Any, 
                             input_file: Path, 
                             context: StageContext) -> Path:
        """Create detailed metadata JSON file."""
        import json
        from morag_stages.file_manager import FileNamingConvention
        
        metadata_filename = FileNamingConvention.get_metadata_filename(
            input_file, StageType.MARKDOWN_CONVERSION
        )
        metadata_path = context.output_dir / metadata_filename
        
        # Comprehensive metadata
        metadata = {
            'stage': 'markdown-conversion',
            'stage_name': 'markdown-conversion',
            'source_file': str(input_file),
            'output_file': str(metadata_path.with_suffix('.md')),
            'processing_result': {
                'success': processing_result.success if hasattr(processing_result, 'success') else True,
                'metadata': processing_result.metadata,
                'content_length': len(processing_result.content),
                'content_preview': processing_result.content[:500] + '...' if len(processing_result.content) > 500 else processing_result.content
            }
        }
        
        # Add thumbnails if available
        if hasattr(processing_result, 'thumbnails'):
            metadata['thumbnails'] = [
                {
                    'timestamp': thumb.timestamp,
                    'size': len(thumb.data) if thumb.data else 0
                }
                for thumb in processing_result.thumbnails
            ]
        
        # Write metadata file
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        return metadata_path
```

### 2. Service Integration
```python
class Stage1ServiceIntegrator:
    """Integrates existing services with Stage 1 interface."""
    
    def __init__(self):
        self.services = {}
    
    async def initialize_services(self):
        """Initialize all required services."""
        from morag_video import VideoService
        from morag_audio import AudioService
        from morag_document import DocumentService
        from morag_web import WebService
        from morag_youtube import YouTubeService
        
        self.services = {
            'video': VideoService(),
            'audio': AudioService(),
            'document': DocumentService(),
            'web': WebService(),
            'youtube': YouTubeService()
        }
        
        # Initialize services
        for service in self.services.values():
            if hasattr(service, 'initialize'):
                await service.initialize()
    
    async def process_by_type(self, 
                             content_type: str, 
                             input_file: Path, 
                             config: Dict[str, Any]) -> Any:
        """Process content using appropriate service."""
        service = self.services.get(content_type)
        if not service:
            raise ValueError(f"No service available for content type: {content_type}")
        
        return await service.process_file(input_file, **config)
```

### 3. Configuration Schema (Complete Replacement)
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class MarkdownConversionConfig(BaseModel):
    """Configuration for markdown-conversion stage."""
    
    # Transcription settings
    include_timestamps: bool = Field(True, description="Include timestamps in transcription")
    transcription_model: str = Field("whisper-large", description="Whisper model to use")
    speaker_diarization: bool = Field(True, description="Enable speaker diarization")
    
    # Document processing
    extract_images: bool = Field(False, description="Extract images from documents")
    preserve_formatting: bool = Field(True, description="Preserve document formatting")
    
    # Video processing
    extract_keyframes: bool = Field(False, description="Extract video keyframes")
    keyframe_interval: int = Field(30, description="Keyframe extraction interval in seconds")
    
    # Web scraping
    follow_links: bool = Field(False, description="Follow internal links")
    max_depth: int = Field(1, description="Maximum link following depth")
    
    # Output settings
    include_metadata_header: bool = Field(True, description="Include YAML metadata header")
    chunk_on_sentences: bool = Field(True, description="Chunk content on sentence boundaries")
    
    # Quality settings
    min_content_length: int = Field(100, description="Minimum content length to process")
    max_content_length: int = Field(1000000, description="Maximum content length to process")
```

## Implementation Steps

1. **Create markdown-conversion package structure**
2. **Implement base MarkdownConversionStage class**
3. **COMPLETELY REPLACE existing services with new interface**
4. **Add unified markdown output format**
5. **Implement metadata header generation**
6. **Add configuration support**
7. **Create file naming utilities**
8. **Add input validation**
9. **Implement error handling**
10. **REMOVE ALL LEGACY CODE and add comprehensive testing**

## Testing Requirements

- Unit tests for Stage 1 implementation
- Integration tests with all service types
- Markdown output format validation
- Metadata header parsing tests
- Error handling and edge case tests
- Performance tests for large files

## Files to Create

- `packages/morag-stages/src/morag_stages/markdown_conversion/__init__.py`
- `packages/morag-stages/src/morag_stages/markdown_conversion/implementation.py`
- `packages/morag-stages/src/morag_stages/markdown_conversion/config.py`
- `packages/morag-stages/src/morag_stages/markdown_conversion/service_integrator.py`
- `packages/morag-stages/tests/test_markdown_conversion.py`

## Success Criteria

- All input types (video, audio, document, web, YouTube) work correctly
- Unified markdown output format is consistent and parseable
- Metadata headers contain all relevant information
- **ALL LEGACY PROCESSING CODE IS REMOVED** - no backwards compatibility
- Performance is equivalent to or better than current implementation
- All tests pass with good coverage
- New stage interface is completely implemented
