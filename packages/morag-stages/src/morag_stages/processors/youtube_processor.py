"""YouTube processor wrapper for stage processing."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import structlog

from .interface import StageProcessor, ProcessorResult

logger = structlog.get_logger(__name__)

# Import core exceptions
try:
    from morag_core.exceptions import ProcessingError
except ImportError:
    class ProcessingError(Exception):  # type: ignore
        pass


class YouTubeStageProcessor(StageProcessor):
    """Stage processor for YouTube content using morag_youtube package."""
    
    def __init__(self):
        """Initialize YouTube stage processor."""
        self._youtube_processor = None
        self._youtube_service = None
    
    def _get_youtube_processor(self):
        """Get or create YouTube processor instance."""
        if self._youtube_processor is None:
            try:
                from morag_youtube import YouTubeProcessor
                self._youtube_processor = YouTubeProcessor()
            except ImportError as e:
                raise ProcessingError(f"YouTube processor not available: {e}")
        return self._youtube_processor
    
    def _get_youtube_service(self):
        """Get or create YouTube service instance."""
        if self._youtube_service is None:
            try:
                from morag_youtube.service import YouTubeService
                self._youtube_service = YouTubeService()
            except ImportError as e:
                raise ProcessingError(f"YouTube service not available: {e}")
        return self._youtube_service
    
    def supports_content_type(self, content_type: str) -> bool:
        """Check if this processor supports the given content type."""
        return content_type.upper() == "YOUTUBE"
    
    async def process(
        self, 
        input_file: Path, 
        output_file: Path, 
        config: Dict[str, Any]
    ) -> ProcessorResult:
        """Process YouTube URL to markdown."""
        # Convert Path back to URL string
        url = str(input_file)
        if url.startswith(('C:', 'D:', '/', '\\')):
            # Handle Windows path conversion issues
            url = url.replace('\\', '/')
            if not url.startswith(('http://', 'https://')):
                raise ProcessingError(f"Invalid YouTube URL: {url}")
        
        logger.info("Processing YouTube URL", url=url)
        
        try:
            # Use YouTube processor from morag_youtube package
            processor = self._get_youtube_processor()
            
            # Convert config to YouTubeConfig
            from morag_youtube import YouTubeConfig
            youtube_config = YouTubeConfig(
                extract_metadata=config.get('extract_metadata', True),
                extract_transcript=config.get('extract_transcript', True),
                use_proxy=config.get('use_proxy', True),
                pre_transcribed=config.get('pre_transcribed', False),
                metadata=config.get('metadata'),
                transcript=config.get('transcript'),
                transcript_segments=config.get('transcript_segments')
            )
            
            # Process the URL
            result = await processor.process_url(url, youtube_config)
            
            # Create markdown content
            metadata = {
                "title": result.metadata.get('title', "YouTube Video"),
                "source": url,
                "type": "youtube",
                "url": url,
                "video_id": result.metadata.get('video_id'),
                "uploader": result.metadata.get('uploader'),
                "duration": result.metadata.get('duration'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
            
            # Create content with transcript
            content_lines = []
            if result.transcript:
                content_lines.append(f"\n# Transcript\n\n{result.transcript}")
            
            content = '\n'.join(content_lines)
            markdown_content = self.create_markdown_with_metadata(content, metadata)
            
            # Write to file
            output_file.write_text(markdown_content, encoding='utf-8')
            
            # Generate final filename based on video title if available
            final_output_file = output_file
            if result.metadata.get('title'):
                from ..stages.markdown_conversion import sanitize_filename
                safe_title = sanitize_filename(result.metadata['title'])
                final_output_file = output_file.parent / f"{safe_title}.md"
                if final_output_file != output_file:
                    output_file.rename(final_output_file)
            
            return ProcessorResult(
                content=content,
                metadata=metadata,
                metrics={
                    "url": url,
                    "content_length": len(result.transcript or ""),
                    "transcript_only": config.get('transcript_only', True)
                },
                final_output_file=final_output_file
            )
            
        except Exception as e:
            logger.error("YouTube processing failed", url=url, error=str(e))
            
            # Fallback to YouTube service if available
            try:
                service = self._get_youtube_service()
                video_id = service.transcript_service.extract_video_id(url)
                
                # Try to get transcript using fallback
                transcript_result = await service.transcript_service.get_transcript(video_id)
                
                if transcript_result and transcript_result.segments:
                    transcript_text = '\n'.join([
                        f"[{segment.start:.1f}s] {segment.text}"
                        for segment in transcript_result.segments
                    ])
                    
                    metadata = {
                        "title": f"YouTube Video {video_id}",
                        "source": url,
                        "type": "youtube",
                        "url": url,
                        "video_id": video_id,
                        "created_at": datetime.now().isoformat(),
                        "fallback_processing": True
                    }
                    
                    content = f"\n# Transcript\n\n{transcript_text}"
                    markdown_content = self.create_markdown_with_metadata(content, metadata)
                    output_file.write_text(markdown_content, encoding='utf-8')
                    
                    return ProcessorResult(
                        content=content,
                        metadata=metadata,
                        metrics={
                            "url": url,
                            "content_length": len(transcript_text),
                            "fallback_used": True
                        },
                        final_output_file=output_file
                    )
                
            except Exception as fallback_error:
                logger.error("YouTube fallback processing also failed", 
                           url=url, error=str(fallback_error))
            
            raise ProcessingError(f"YouTube processing failed for {url}: {e}")
