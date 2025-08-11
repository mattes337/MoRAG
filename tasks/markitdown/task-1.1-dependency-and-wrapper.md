# Task 1.1: Add Markitdown Dependency and Create Wrapper Service

## Objective
Add the markitdown framework as a dependency and create a wrapper service that provides a clean interface for the MoRAG system to interact with markitdown functionality.

## Scope
- Add markitdown dependency to the project
- Create a wrapper service in morag-document package
- Implement basic configuration management
- Set up logging and error handling
- **MANDATORY**: Test thoroughly before proceeding to Task 1.2

## Implementation Details

### 1. Add Dependency

**File**: `requirements.txt`
```txt
# Add markitdown with all optional dependencies
markitdown[all]>=0.1.2
```

**File**: `packages/morag-document/pyproject.toml`
```toml
[project]
dependencies = [
    # ... existing dependencies
    "markitdown[all]>=0.1.2",
]
```

### 2. Create Wrapper Service

**File**: `packages/morag-document/src/morag_document/services/markitdown_service.py`

```python
"""Markitdown wrapper service for MoRAG document processing."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Union
import structlog
from markitdown import MarkItDown

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError, ConfigurationError

logger = structlog.get_logger(__name__)

class MarkitdownService:
    """Service wrapper for Microsoft's markitdown framework."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize markitdown service.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        self._markitdown = None
        self._initialize_markitdown()
    
    def _initialize_markitdown(self) -> None:
        """Initialize markitdown instance with configuration."""
        try:
            # Basic initialization
            kwargs = {
                'enable_plugins': self.config.get('enable_plugins', False)
            }
            
            # Add Azure Document Intelligence if configured
            if self.config.get('use_azure_doc_intel', False):
                endpoint = self.config.get('azure_doc_intel_endpoint')
                if endpoint:
                    kwargs['docintel_endpoint'] = endpoint
                else:
                    logger.warning("Azure Document Intelligence enabled but no endpoint provided")
            
            # Add LLM client for image descriptions if configured
            if self.config.get('use_llm_image_description', False):
                llm_client = self.config.get('llm_client')
                llm_model = self.config.get('llm_model', 'gpt-4o')
                if llm_client:
                    kwargs['llm_client'] = llm_client
                    kwargs['llm_model'] = llm_model
            
            self._markitdown = MarkItDown(**kwargs)
            logger.info("Markitdown service initialized", config=kwargs)
            
        except Exception as e:
            logger.error("Failed to initialize markitdown service", error=str(e))
            raise ConfigurationError(f"Markitdown initialization failed: {str(e)}")
    
    async def convert_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Convert file to markdown using markitdown.
        
        Args:
            file_path: Path to file to convert
            
        Returns:
            Dictionary containing conversion result
            
        Raises:
            ProcessingError: If conversion fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")
        
        try:
            # Run markitdown conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._markitdown.convert, 
                str(file_path)
            )
            
            return {
                'text_content': result.text_content,
                'metadata': getattr(result, 'metadata', {}),
                'success': True,
                'source_file': str(file_path)
            }
            
        except Exception as e:
            logger.error(
                "Markitdown conversion failed", 
                file_path=str(file_path), 
                error=str(e)
            )
            raise ProcessingError(f"Markitdown conversion failed: {str(e)}")
    
    async def convert_stream(self, file_stream, filename: str) -> Dict[str, Any]:
        """Convert file stream to markdown using markitdown.
        
        Args:
            file_stream: Binary file stream
            filename: Original filename for format detection
            
        Returns:
            Dictionary containing conversion result
            
        Raises:
            ProcessingError: If conversion fails
        """
        try:
            # Run markitdown conversion in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._markitdown.convert_stream, 
                file_stream,
                filename
            )
            
            return {
                'text_content': result.text_content,
                'metadata': getattr(result, 'metadata', {}),
                'success': True,
                'source_file': filename
            }
            
        except Exception as e:
            logger.error(
                "Markitdown stream conversion failed", 
                filename=filename, 
                error=str(e)
            )
            raise ProcessingError(f"Markitdown stream conversion failed: {str(e)}")
    
    def supports_format(self, format_type: str) -> bool:
        """Check if markitdown supports the given format.
        
        Args:
            format_type: File format to check
            
        Returns:
            True if format is supported
        """
        # Markitdown supported formats (as of v0.1.2)
        supported_formats = {
            'pdf', 'docx', 'pptx', 'xlsx', 'xls',
            'html', 'htm', 'csv', 'json', 'xml',
            'txt', 'md', 'markdown', 'zip',
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff',
            'wav', 'mp3', 'mp4', 'avi', 'mov',
            'epub', 'msg'  # Outlook messages
        }
        
        return format_type.lower() in supported_formats
    
    def get_supported_formats(self) -> set:
        """Get set of all supported formats.
        
        Returns:
            Set of supported file extensions
        """
        return {
            'pdf', 'docx', 'pptx', 'xlsx', 'xls',
            'html', 'htm', 'csv', 'json', 'xml',
            'txt', 'md', 'markdown', 'zip',
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff',
            'wav', 'mp3', 'mp4', 'avi', 'mov',
            'epub', 'msg'
        }
```

### 3. Configuration Updates

**File**: `packages/morag-core/src/morag_core/config.py`

Add markitdown-specific configuration:

```python
# Markitdown configuration
MARKITDOWN_ENABLED: bool = Field(default=True, description="Enable markitdown processing")
MARKITDOWN_USE_AZURE_DOC_INTEL: bool = Field(default=False, description="Use Azure Document Intelligence")
MARKITDOWN_AZURE_ENDPOINT: Optional[str] = Field(default=None, description="Azure Document Intelligence endpoint")
MARKITDOWN_USE_LLM_IMAGE_DESCRIPTION: bool = Field(default=False, description="Use LLM for image descriptions")
MARKITDOWN_LLM_MODEL: str = Field(default="gpt-4o", description="LLM model for image descriptions")
MARKITDOWN_ENABLE_PLUGINS: bool = Field(default=False, description="Enable markitdown plugins")
```

## Testing

### Unit Tests

**File**: `packages/morag-document/tests/test_markitdown_service.py`

```python
"""Tests for markitdown service."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from morag_document.services.markitdown_service import MarkitdownService
from morag_core.exceptions import ProcessingError, ConfigurationError

class TestMarkitdownService:
    
    def test_initialization(self):
        """Test service initialization."""
        service = MarkitdownService()
        assert service._markitdown is not None
        assert service.supports_format('pdf')
        assert service.supports_format('docx')
    
    def test_initialization_with_config(self):
        """Test service initialization with custom config."""
        config = {
            'enable_plugins': True,
            'use_azure_doc_intel': True,
            'azure_doc_intel_endpoint': 'https://test.endpoint.com'
        }
        service = MarkitdownService(config)
        assert service.config == config
    
    @pytest.mark.asyncio
    async def test_convert_file_success(self, tmp_path):
        """Test successful file conversion."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        service = MarkitdownService()
        
        with patch.object(service._markitdown, 'convert') as mock_convert:
            mock_result = Mock()
            mock_result.text_content = "# Test content"
            mock_convert.return_value = mock_result
            
            result = await service.convert_file(test_file)
            
            assert result['success'] is True
            assert result['text_content'] == "# Test content"
            assert result['source_file'] == str(test_file)
    
    @pytest.mark.asyncio
    async def test_convert_file_not_found(self):
        """Test conversion with non-existent file."""
        service = MarkitdownService()
        
        with pytest.raises(ProcessingError, match="File not found"):
            await service.convert_file("nonexistent.pdf")
    
    def test_supports_format(self):
        """Test format support checking."""
        service = MarkitdownService()
        
        assert service.supports_format('pdf')
        assert service.supports_format('docx')
        assert service.supports_format('xlsx')
        assert service.supports_format('jpg')
        assert not service.supports_format('unknown')
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        service = MarkitdownService()
        formats = service.get_supported_formats()
        
        assert 'pdf' in formats
        assert 'docx' in formats
        assert 'xlsx' in formats
        assert len(formats) > 10
```

## Acceptance Criteria

- [ ] Markitdown dependency added to project requirements
- [ ] MarkitdownService class implemented with proper error handling
- [ ] Configuration management for markitdown options
- [ ] Async support for non-blocking operations
- [ ] Format support detection functionality
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Proper logging and error handling
- [ ] Documentation for service usage

## Dependencies
- None (this is the foundation task)

## Estimated Effort
- **Development**: 4-6 hours
- **Testing**: 2-3 hours
- **Documentation**: 1-2 hours
- **Total**: 7-11 hours

## Notes
- This task establishes the foundation for all subsequent markitdown integration
- Focus on clean interfaces and proper error handling
- Ensure async compatibility for integration with existing MoRAG services
- Consider future extensibility for additional markitdown features
