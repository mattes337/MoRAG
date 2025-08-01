# Task 1: Install LangExtract and Configure Dependencies

## Objective
Install Google's LangExtract library and configure it for use in MoRAG, replacing the current extraction dependencies.

## Prerequisites
- Python 3.10+ environment
- Google Gemini API key
- Access to modify requirements files

## Steps

### 1. Install LangExtract
```bash
# Add to requirements files
pip install langextract
```

### 2. Update Requirements Files

**File**: `packages/morag-graph/requirements.txt`
```txt
# Add LangExtract
langextract>=1.0.0

# Keep existing dependencies that will remain
neo4j>=5.15.0
pydantic>=2.0.0
structlog>=23.0.0
```

**File**: `packages/morag-graph/pyproject.toml`
```toml
[project]
dependencies = [
    "langextract>=1.0.0",
    # ... other dependencies
]
```

### 3. Environment Configuration

**File**: `.env` (add to existing)
```env
# LangExtract Configuration
LANGEXTRACT_API_KEY=your-gemini-api-key-here
LANGEXTRACT_MODEL=gemini-2.5-flash
LANGEXTRACT_MAX_WORKERS=20
LANGEXTRACT_EXTRACTION_PASSES=3
LANGEXTRACT_MAX_CHAR_BUFFER=1000
```

### 4. Create LangExtract Configuration

**File**: `packages/morag-graph/src/morag_graph/config/langextract_config.py`
```python
"""LangExtract configuration for MoRAG."""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LangExtractConfig(BaseModel):
    """Configuration for LangExtract integration."""
    
    # API Configuration
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGEXTRACT_API_KEY"),
        description="Gemini API key for LangExtract"
    )
    
    model_id: str = Field(
        default_factory=lambda: os.getenv("LANGEXTRACT_MODEL", "gemini-2.5-flash"),
        description="Gemini model to use"
    )
    
    # Performance Configuration
    max_workers: int = Field(
        default_factory=lambda: int(os.getenv("LANGEXTRACT_MAX_WORKERS", "20")),
        description="Maximum concurrent workers"
    )
    
    extraction_passes: int = Field(
        default_factory=lambda: int(os.getenv("LANGEXTRACT_EXTRACTION_PASSES", "3")),
        description="Number of extraction passes for better recall"
    )
    
    max_char_buffer: int = Field(
        default_factory=lambda: int(os.getenv("LANGEXTRACT_MAX_CHAR_BUFFER", "1000")),
        description="Maximum characters per processing buffer"
    )
    
    # Quality Configuration
    min_confidence: float = Field(
        default=0.7,
        description="Minimum confidence threshold for extractions"
    )
    
    enable_visualization: bool = Field(
        default=True,
        description="Enable HTML visualization generation"
    )
    
    # Domain Configuration
    default_domain: str = Field(
        default="general",
        description="Default domain for extraction examples"
    )
    
    custom_domains: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom domain configurations"
    )


class DomainConfig(BaseModel):
    """Configuration for domain-specific extraction."""
    
    name: str = Field(description="Domain name")
    description: str = Field(description="Domain description")
    prompt_template: str = Field(description="Prompt template for this domain")
    examples_file: Optional[str] = Field(
        default=None,
        description="Path to examples file for this domain"
    )
    
    # Domain-specific settings
    extraction_classes: List[str] = Field(
        default_factory=list,
        description="Expected extraction classes for this domain"
    )
    
    required_attributes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Required attributes for each extraction class"
    )
    
    confidence_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence thresholds per extraction class"
    )


def get_langextract_config() -> LangExtractConfig:
    """Get LangExtract configuration instance."""
    return LangExtractConfig()


def get_domain_config(domain: str) -> Optional[DomainConfig]:
    """Get domain-specific configuration."""
    config = get_langextract_config()
    domain_data = config.custom_domains.get(domain)
    
    if domain_data:
        return DomainConfig(**domain_data)
    
    return None
```

### 5. Create Base LangExtract Service

**File**: `packages/morag-graph/src/morag_graph/services/langextract_service.py`
```python
"""LangExtract service for MoRAG integration."""

import asyncio
import structlog
from typing import List, Dict, Any, Optional
import langextract as lx

from ..config.langextract_config import LangExtractConfig, get_langextract_config
from ..models import Entity, Relation

logger = structlog.get_logger(__name__)


class LangExtractService:
    """Service for LangExtract integration with MoRAG."""
    
    def __init__(self, config: Optional[LangExtractConfig] = None):
        """Initialize LangExtract service.
        
        Args:
            config: Optional configuration, defaults to global config
        """
        self.config = config or get_langextract_config()
        self.logger = logger.bind(service="langextract")
        
        # Validate configuration
        if not self.config.api_key:
            raise ValueError("LANGEXTRACT_API_KEY environment variable is required")
    
    async def extract_structured_data(
        self,
        text: str,
        prompt_description: str,
        examples: List[lx.data.ExampleData],
        source_doc_id: Optional[str] = None
    ) -> lx.data.AnnotatedDocument:
        """Extract structured data using LangExtract.
        
        Args:
            text: Input text to process
            prompt_description: Description of what to extract
            examples: Few-shot examples for guidance
            source_doc_id: Optional source document ID
            
        Returns:
            LangExtract annotated document with extractions
        """
        try:
            self.logger.info(
                "Starting LangExtract extraction",
                text_length=len(text),
                num_examples=len(examples),
                source_doc_id=source_doc_id
            )
            
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt_description,
                examples=examples,
                model_id=self.config.model_id,
                api_key=self.config.api_key,
                extraction_passes=self.config.extraction_passes,
                max_workers=self.config.max_workers,
                max_char_buffer=self.config.max_char_buffer
            )
            
            self.logger.info(
                "LangExtract extraction completed",
                num_extractions=len(result.extractions),
                source_doc_id=source_doc_id
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "LangExtract extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                source_doc_id=source_doc_id
            )
            raise
    
    def generate_visualization(
        self,
        results: List[lx.data.AnnotatedDocument],
        output_path: Optional[str] = None
    ) -> str:
        """Generate HTML visualization for extraction results.
        
        Args:
            results: List of annotated documents
            output_path: Optional path to save HTML file
            
        Returns:
            HTML content as string
        """
        if not self.config.enable_visualization:
            return ""
        
        try:
            # Save to temporary JSONL file
            temp_file = "temp_extractions.jsonl"
            lx.io.save_annotated_documents(results, output_name=temp_file)
            
            # Generate visualization
            html_content = lx.visualize(temp_file)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                self.logger.info("Visualization saved", path=output_path)
            
            # Cleanup temp file
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return html_content
            
        except Exception as e:
            self.logger.error(
                "Visualization generation failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return ""
```

### 6. Test Installation

**File**: `tests/test_langextract_installation.py`
```python
"""Test LangExtract installation and basic functionality."""

import pytest
import os
from unittest.mock import patch, MagicMock

from morag_graph.config.langextract_config import get_langextract_config
from morag_graph.services.langextract_service import LangExtractService


class TestLangExtractInstallation:
    """Test LangExtract installation and configuration."""
    
    def test_langextract_import(self):
        """Test that LangExtract can be imported."""
        import langextract as lx
        assert hasattr(lx, 'extract')
        assert hasattr(lx, 'data')
        assert hasattr(lx, 'visualize')
    
    def test_config_creation(self):
        """Test LangExtract configuration creation."""
        config = get_langextract_config()
        assert config is not None
        assert hasattr(config, 'model_id')
        assert hasattr(config, 'max_workers')
    
    @patch.dict(os.environ, {'LANGEXTRACT_API_KEY': 'test-key'})
    def test_service_creation(self):
        """Test LangExtract service creation."""
        service = LangExtractService()
        assert service is not None
        assert service.config.api_key == 'test-key'
    
    def test_service_creation_without_api_key(self):
        """Test service creation fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LANGEXTRACT_API_KEY"):
                LangExtractService()
```

## Verification Steps

1. **Install Package**:
   ```bash
   cd packages/morag-graph
   pip install -e .
   ```

2. **Test Import**:
   ```python
   import langextract as lx
   print(lx.__version__)
   ```

3. **Run Tests**:
   ```bash
   pytest tests/test_langextract_installation.py -v
   ```

4. **Verify Configuration**:
   ```python
   from morag_graph.config.langextract_config import get_langextract_config
   config = get_langextract_config()
   print(f"Model: {config.model_id}")
   print(f"Workers: {config.max_workers}")
   ```

## Success Criteria

- [ ] LangExtract package installed successfully
- [ ] Configuration classes created and working
- [ ] Base service class implemented
- [ ] Environment variables configured
- [ ] Basic tests passing
- [ ] No import errors

## Next Steps

After completing this task:
1. Move to Task 2: Create LangExtract wrapper
2. Ensure API key is properly configured
3. Test basic extraction functionality

## Notes

- Keep existing dependencies until they are explicitly removed in later tasks
- LangExtract requires Python 3.10+ 
- Gemini API key is required for functionality
- Configuration supports both environment variables and direct settings
