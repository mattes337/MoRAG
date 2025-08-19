"""Markdown conversion stage implementation."""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

# Import services with graceful fallback
try:
    from morag_services import MoRAGServices, ContentType
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    MoRAGServices = None
    ContentType = None

logger = structlog.get_logger(__name__)


class MarkdownConversionStage(Stage):
    """Stage that converts input files to unified markdown format."""
    
    def __init__(self, stage_type: StageType = StageType.MARKDOWN_CONVERSION):
        """Initialize markdown conversion stage."""
        super().__init__(stage_type)
        
        if not SERVICES_AVAILABLE:
            raise StageExecutionError(
                "MoRAG services not available for markdown conversion",
                stage_type=self.stage_type.value
            )
        
        self.services = MoRAGServices()
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Execute markdown conversion on input files.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            Stage execution result
        """
        if len(input_files) != 1:
            raise StageValidationError(
                "Markdown conversion stage requires exactly one input file",
                stage_type=self.stage_type.value,
                invalid_files=[str(f) for f in input_files]
            )
        
        input_file = input_files[0]
        config = context.get_stage_config(self.stage_type)
        
        logger.info("Starting markdown conversion", 
                   input_file=str(input_file),
                   config=config)
        
        try:
            # Determine content type
            content_type = self._detect_content_type(input_file)
            
            # Generate output filename
            output_file = context.output_dir / f"{input_file.stem}.md"
            context.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process based on content type
            if content_type == ContentType.VIDEO:
                result_data = await self._process_video(input_file, output_file, config)
            elif content_type == ContentType.AUDIO:
                result_data = await self._process_audio(input_file, output_file, config)
            elif content_type == ContentType.DOCUMENT:
                result_data = await self._process_document(input_file, output_file, config)
            elif content_type == ContentType.WEB:
                result_data = await self._process_web(input_file, output_file, config)
            else:
                result_data = await self._process_text(input_file, output_file, config)
            
            # Create metadata
            metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(input_file)],
                output_files=[str(output_file)],
                config_used=config,
                metrics={
                    "content_type": content_type.value if content_type else "unknown",
                    "input_size_bytes": input_file.stat().st_size if input_file.exists() else 0,
                    "output_size_bytes": output_file.stat().st_size if output_file.exists() else 0,
                    **result_data.get("metrics", {})
                }
            )
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file],
                metadata=metadata,
                data=result_data
            )
            
        except Exception as e:
            logger.error("Markdown conversion failed", 
                        input_file=str(input_file), 
                        error=str(e))
            raise StageExecutionError(
                f"Markdown conversion failed: {e}",
                stage_type=self.stage_type.value,
                original_error=e
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for markdown conversion.
        
        Args:
            input_files: List of input file paths
            
        Returns:
            True if inputs are valid
        """
        if len(input_files) != 1:
            return False
        
        input_file = input_files[0]
        
        # Check if file exists (for local files)
        if not str(input_file).startswith(('http://', 'https://')):
            if not input_file.exists():
                return False
        
        # Check if file type is supported
        content_type = self._detect_content_type(input_file)
        return content_type is not None
    
    def get_dependencies(self) -> List[StageType]:
        """Get stage dependencies.
        
        Returns:
            Empty list - this is the first stage
        """
        return []
    
    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Get expected output file paths.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            List of expected output file paths
        """
        if len(input_files) != 1:
            return []
        
        input_file = input_files[0]
        output_file = context.output_dir / f"{input_file.stem}.md"
        return [output_file]
    
    def _detect_content_type(self, file_path: Path) -> Optional[ContentType]:
        """Detect content type from file path.
        
        Args:
            file_path: File path to analyze
            
        Returns:
            Detected content type or None
        """
        if not ContentType:
            return None
        
        file_str = str(file_path).lower()
        
        # Web URLs
        if file_str.startswith(('http://', 'https://')):
            return ContentType.WEB
        
        # Video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        if file_path.suffix.lower() in video_extensions:
            return ContentType.VIDEO
        
        # Audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        if file_path.suffix.lower() in audio_extensions:
            return ContentType.AUDIO
        
        # Document files
        doc_extensions = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}
        if file_path.suffix.lower() in doc_extensions:
            return ContentType.DOCUMENT
        
        # Text files (default)
        text_extensions = {'.txt', '.md', '.rst', '.html', '.xml', '.json', '.csv'}
        if file_path.suffix.lower() in text_extensions:
            return ContentType.TEXT
        
        # Default to text for unknown types
        return ContentType.TEXT
    
    async def _process_video(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process video file to markdown.
        
        Args:
            input_file: Input video file
            output_file: Output markdown file
            config: Stage configuration
            
        Returns:
            Processing result data
        """
        logger.info("Processing video file", input_file=str(input_file))
        
        # Use video service
        result = await self.services.process_video(
            str(input_file),
            include_timestamps=config.get('include_timestamps', True),
            speaker_diarization=config.get('speaker_diarization', True),
            topic_segmentation=config.get('topic_segmentation', True),
            extract_thumbnails=config.get('extract_thumbnails', False)
        )
        
        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.content,
            metadata={
                "title": result.title or input_file.stem,
                "source": str(input_file),
                "type": "video",
                "duration": result.metadata.get('duration'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )
        
        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')
        
        return {
            "content_type": "video",
            "title": result.title,
            "metadata": result.metadata,
            "metrics": {
                "duration": result.metadata.get('duration', 0),
                "transcript_length": len(result.content),
                "has_timestamps": config.get('include_timestamps', True)
            }
        }

    async def _process_audio(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio file to markdown.

        Args:
            input_file: Input audio file
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        logger.info("Processing audio file", input_file=str(input_file))

        # Use audio service
        result = await self.services.process_audio(
            str(input_file),
            include_timestamps=config.get('include_timestamps', True),
            speaker_diarization=config.get('speaker_diarization', True),
            topic_segmentation=config.get('topic_segmentation', True)
        )

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.content,
            metadata={
                "title": result.title or input_file.stem,
                "source": str(input_file),
                "type": "audio",
                "duration": result.metadata.get('duration'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "audio",
            "title": result.title,
            "metadata": result.metadata,
            "metrics": {
                "duration": result.metadata.get('duration', 0),
                "transcript_length": len(result.content),
                "has_timestamps": config.get('include_timestamps', True)
            }
        }

    async def _process_document(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process document file to markdown.

        Args:
            input_file: Input document file
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        logger.info("Processing document file", input_file=str(input_file))

        # Use document service
        result = await self.services.process_document(
            str(input_file),
            preserve_formatting=config.get('preserve_formatting', True),
            extract_tables=config.get('extract_tables', True),
            extract_images=config.get('extract_images', False),
            ocr_enabled=config.get('ocr_enabled', True)
        )

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.content,
            metadata={
                "title": result.title or input_file.stem,
                "source": str(input_file),
                "type": "document",
                "pages": result.metadata.get('pages'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "document",
            "title": result.title,
            "metadata": result.metadata,
            "metrics": {
                "pages": result.metadata.get('pages', 0),
                "content_length": len(result.content),
                "has_tables": config.get('extract_tables', True)
            }
        }

    async def _process_web(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process web URL to markdown.

        Args:
            input_file: Input URL (as Path object)
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        url = str(input_file)
        logger.info("Processing web URL", url=url)

        # Use web service
        result = await self.services.process_web(
            url,
            follow_links=config.get('follow_links', False),
            max_depth=config.get('max_depth', 1)
        )

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.content,
            metadata={
                "title": result.title or "Web Content",
                "source": url,
                "type": "web",
                "url": url,
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "web",
            "title": result.title,
            "metadata": result.metadata,
            "metrics": {
                "url": url,
                "content_length": len(result.content),
                "links_followed": config.get('follow_links', False)
            }
        }

    async def _process_text(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process text file to markdown.

        Args:
            input_file: Input text file
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        logger.info("Processing text file", input_file=str(input_file))

        # Read text content
        content = input_file.read_text(encoding='utf-8')

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=content,
            metadata={
                "title": input_file.stem,
                "source": str(input_file),
                "type": "text",
                "language": "en",  # Default language
                "created_at": datetime.now().isoformat()
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "text",
            "title": input_file.stem,
            "metadata": {},
            "metrics": {
                "content_length": len(content),
                "original_format": input_file.suffix
            }
        }

    def _create_markdown_with_metadata(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create markdown content with YAML metadata header.

        Args:
            content: Main content
            metadata: Metadata dictionary

        Returns:
            Markdown content with metadata header
        """
        # Create YAML frontmatter
        yaml_lines = ["---"]
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, str):
                    yaml_lines.append(f'{key}: "{value}"')
                else:
                    yaml_lines.append(f'{key}: {value}')
        yaml_lines.append("---")
        yaml_lines.append("")  # Empty line after frontmatter

        # Combine with content
        return "\n".join(yaml_lines) + content
