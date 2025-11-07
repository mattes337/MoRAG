"""Conversion processors for different content types."""

import asyncio
from datetime import datetime
from typing import Dict, Any, Union
from pathlib import Path
import structlog

from .converter_factory import ConverterFactory

logger = structlog.get_logger(__name__)

# Import services - these are optional for fallback processing
try:
    from morag_services import MoRAGServices, ContentType
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    # Create placeholder types for when services are not available
    class MoRAGServices:  # type: ignore
        pass
    class ContentType:  # type: ignore
        pass

# Import YouTube processor for fallback when services are not available
try:
    from morag_youtube.service import YouTubeService
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    class YouTubeService:  # type: ignore
        pass

# Import URL path utilities
try:
    from morag.utils.url_path import URLPath, is_url, get_url_string
    URL_PATH_AVAILABLE = True
except ImportError:
    URL_PATH_AVAILABLE = False
    # Fallback implementations
    def is_url(path_like) -> bool:
        return str(path_like).startswith(('http://', 'https://'))

    def get_url_string(path_like) -> str:
        return str(path_like)


class ConversionProcessors:
    """Collection of processors for different content types."""

    def __init__(self):
        """Initialize conversion processors."""
        self.converter_factory = ConverterFactory()
        self._initialized = False

    async def initialize(self):
        """Initialize processors."""
        if self._initialized:
            return

        await self.converter_factory.initialize()
        self._initialized = True

    async def process_file(self,
                          input_file: Union[Path, 'URLPath'],
                          output_file: Path,
                          content_type: Any,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a file based on its content type."""
        if not self._initialized:
            await self.initialize()

        try:
            # Determine processing method based on content type
            converter_type = self.converter_factory.get_converter_type(input_file, content_type)

            if converter_type == 'youtube':
                return await self._process_youtube(input_file, output_file, config)
            elif converter_type == 'video':
                return await self._process_video(input_file, output_file, config)
            elif converter_type == 'audio':
                return await self._process_audio(input_file, output_file, config)
            elif converter_type == 'web':
                return await self._process_web(input_file, output_file, config)
            elif converter_type == 'markitdown':
                return await self._process_with_markitdown(input_file, output_file, config)
            else:
                # Fallback to services if available
                if self.converter_factory.is_services_available():
                    return await self._process_with_services(input_file, output_file, config)
                else:
                    return await self._process_text(input_file, output_file, config)

        except Exception as e:
            logger.error("Processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'processing_time': 0,
                    'processor_type': 'error'
                }
            }

    async def _process_with_markitdown(self, input_file: Union[Path, 'URLPath'], output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process file using markitdown converter."""
        start_time = datetime.now()

        try:
            # Import markitdown - this is optional
            try:
                from markitdown import MarkItDown
                markitdown_available = True
            except ImportError:
                logger.error("Markitdown not available")
                return {
                    'success': False,
                    'error': 'Markitdown library not installed',
                    'metadata': {'processor_type': 'markitdown', 'processing_time': 0}
                }

            # Initialize markitdown
            md = MarkItDown()

            # Process the file
            if is_url(str(input_file)):
                # Process URL
                result = md.convert_url(str(input_file))
            else:
                # Process local file
                result = md.convert(str(input_file))

            # Extract text content
            content = result.text_content if hasattr(result, 'text_content') else str(result)

            # Validate conversion quality
            if not self.converter_factory.validate_conversion_quality(content, input_file):
                logger.warning("Conversion quality validation failed", file=str(input_file))
                return {
                    'success': False,
                    'error': 'Conversion quality too low',
                    'metadata': {
                        'processor_type': 'markitdown',
                        'processing_time': (datetime.now() - start_time).total_seconds(),
                        'content_length': len(content) if content else 0
                    }
                }

            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'content': content,
                'metadata': {
                    'processor_type': 'markitdown',
                    'processing_time': processing_time,
                    'content_length': len(content),
                    'output_file': str(output_file)
                }
            }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error("Markitdown processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': f'Markitdown processing failed: {str(e)}',
                'metadata': {
                    'processor_type': 'markitdown',
                    'processing_time': processing_time
                }
            }

    async def _process_video(self, input_file: Union[Path, 'URLPath'], output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process video file."""
        start_time = datetime.now()

        try:
            # Use MoRAG services if available
            if self.converter_factory.is_services_available():
                services = await self.converter_factory.get_services()
                result = await services.process_video(str(input_file), config)

                if result.success:
                    # Write content to output file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.text_content)

                    processing_time = (datetime.now() - start_time).total_seconds()

                    return {
                        'success': True,
                        'content': result.text_content,
                        'metadata': {
                            'processor_type': 'morag_video',
                            'processing_time': processing_time,
                            'content_length': len(result.text_content),
                            'original_metadata': result.metadata
                        }
                    }
                else:
                    return {
                        'success': False,
                        'error': result.error,
                        'metadata': {
                            'processor_type': 'morag_video',
                            'processing_time': (datetime.now() - start_time).total_seconds()
                        }
                    }
            else:
                return {
                    'success': False,
                    'error': 'Video processing services not available',
                    'metadata': {
                        'processor_type': 'video_fallback',
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
                }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error("Video processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': f'Video processing failed: {str(e)}',
                'metadata': {
                    'processor_type': 'video',
                    'processing_time': processing_time
                }
            }

    async def _process_audio(self, input_file: Union[Path, 'URLPath'], output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio file."""
        start_time = datetime.now()

        try:
            # Use MoRAG services if available
            if self.converter_factory.is_services_available():
                services = await self.converter_factory.get_services()
                result = await services.process_audio(str(input_file), config)

                if result.success:
                    # Write content to output file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.text_content)

                    processing_time = (datetime.now() - start_time).total_seconds()

                    return {
                        'success': True,
                        'content': result.text_content,
                        'metadata': {
                            'processor_type': 'morag_audio',
                            'processing_time': processing_time,
                            'content_length': len(result.text_content),
                            'original_metadata': result.metadata
                        }
                    }
                else:
                    return {
                        'success': False,
                        'error': result.error,
                        'metadata': {
                            'processor_type': 'morag_audio',
                            'processing_time': (datetime.now() - start_time).total_seconds()
                        }
                    }
            else:
                return {
                    'success': False,
                    'error': 'Audio processing services not available',
                    'metadata': {
                        'processor_type': 'audio_fallback',
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
                }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error("Audio processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': f'Audio processing failed: {str(e)}',
                'metadata': {
                    'processor_type': 'audio',
                    'processing_time': processing_time
                }
            }

    async def _process_web(self, input_file: Union[Path, 'URLPath'], output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process web URL."""
        start_time = datetime.now()

        try:
            # Use MoRAG services if available
            if self.converter_factory.is_services_available():
                services = await self.converter_factory.get_services()
                result = await services.process_url(str(input_file), config)

                if result.success:
                    # Write content to output file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.text_content)

                    processing_time = (datetime.now() - start_time).total_seconds()

                    return {
                        'success': True,
                        'content': result.text_content,
                        'metadata': {
                            'processor_type': 'morag_web',
                            'processing_time': processing_time,
                            'content_length': len(result.text_content),
                            'original_metadata': result.metadata
                        }
                    }
                else:
                    return {
                        'success': False,
                        'error': result.error,
                        'metadata': {
                            'processor_type': 'morag_web',
                            'processing_time': (datetime.now() - start_time).total_seconds()
                        }
                    }
            else:
                return {
                    'success': False,
                    'error': 'Web processing services not available',
                    'metadata': {
                        'processor_type': 'web_fallback',
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
                }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error("Web processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': f'Web processing failed: {str(e)}',
                'metadata': {
                    'processor_type': 'web',
                    'processing_time': processing_time
                }
            }

    async def _process_youtube(self, input_file: Union[Path, 'URLPath'], output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process YouTube URL."""
        start_time = datetime.now()

        try:
            # Use MoRAG services if available
            if self.converter_factory.is_services_available():
                services = await self.converter_factory.get_services()
                result = await services.process_youtube(str(input_file), config)

                if result.success:
                    # Write content to output file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.text_content)

                    processing_time = (datetime.now() - start_time).total_seconds()

                    return {
                        'success': True,
                        'content': result.text_content,
                        'metadata': {
                            'processor_type': 'morag_youtube',
                            'processing_time': processing_time,
                            'content_length': len(result.text_content),
                            'original_metadata': result.metadata
                        }
                    }
                else:
                    return {
                        'success': False,
                        'error': result.error,
                        'metadata': {
                            'processor_type': 'morag_youtube',
                            'processing_time': (datetime.now() - start_time).total_seconds()
                        }
                    }
            else:
                # Fallback to direct YouTube service if available
                if YOUTUBE_AVAILABLE:
                    youtube_service = YouTubeService()
                    await youtube_service.initialize()
                    result = await youtube_service.process(str(input_file), config)

                    if result.success:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(result.content)

                        processing_time = (datetime.now() - start_time).total_seconds()

                        return {
                            'success': True,
                            'content': result.content,
                            'metadata': {
                                'processor_type': 'youtube_direct',
                                'processing_time': processing_time,
                                'content_length': len(result.content),
                                'original_metadata': result.metadata
                            }
                        }
                    else:
                        return {
                            'success': False,
                            'error': result.error,
                            'metadata': {
                                'processor_type': 'youtube_direct',
                                'processing_time': (datetime.now() - start_time).total_seconds()
                            }
                        }
                else:
                    return {
                        'success': False,
                        'error': 'YouTube processing services not available',
                        'metadata': {
                            'processor_type': 'youtube_fallback',
                            'processing_time': (datetime.now() - start_time).total_seconds()
                        }
                    }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error("YouTube processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': f'YouTube processing failed: {str(e)}',
                'metadata': {
                    'processor_type': 'youtube',
                    'processing_time': processing_time
                }
            }

    async def _process_with_services(self, input_file: Union[Path, 'URLPath'], output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process using general MoRAG services."""
        start_time = datetime.now()

        try:
            services = await self.converter_factory.get_services()
            result = await services.process_content(str(input_file), config=config)

            if result.success:
                # Write content to output file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.text_content)

                processing_time = (datetime.now() - start_time).total_seconds()

                return {
                    'success': True,
                    'content': result.text_content,
                    'metadata': {
                        'processor_type': 'morag_services',
                        'processing_time': processing_time,
                        'content_length': len(result.text_content),
                        'original_metadata': result.metadata
                    }
                }
            else:
                return {
                    'success': False,
                    'error': result.error,
                    'metadata': {
                        'processor_type': 'morag_services',
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
                }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error("Services processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': f'Services processing failed: {str(e)}',
                'metadata': {
                    'processor_type': 'morag_services',
                    'processing_time': processing_time
                }
            }

    async def _process_text(self, input_file: Union[Path, 'URLPath'], output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback text processing."""
        start_time = datetime.now()

        try:
            if is_url(str(input_file)):
                return {
                    'success': False,
                    'error': 'URL processing not available in fallback mode',
                    'metadata': {
                        'processor_type': 'text_fallback',
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
                }

            # Read text file directly
            input_path = Path(input_file)

            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None

            for encoding in encodings:
                try:
                    with open(input_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                return {
                    'success': False,
                    'error': 'Unable to read file with any supported encoding',
                    'metadata': {
                        'processor_type': 'text_fallback',
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
                }

            # Basic markdown formatting for text files
            if input_path.suffix.lower() in ['.txt', '.md']:
                # Just copy content as-is for markdown files
                formatted_content = content
            else:
                # Add basic formatting for other text files
                formatted_content = f"# {input_path.name}\n\n{content}"

            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_content)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'content': formatted_content,
                'metadata': {
                    'processor_type': 'text_fallback',
                    'processing_time': processing_time,
                    'content_length': len(formatted_content),
                    'encoding_used': encoding
                }
            }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error("Text processing failed", file=str(input_file), error=str(e))
            return {
                'success': False,
                'error': f'Text processing failed: {str(e)}',
                'metadata': {
                    'processor_type': 'text_fallback',
                    'processing_time': processing_time
                }
            }


__all__ = ["ConversionProcessors"]
