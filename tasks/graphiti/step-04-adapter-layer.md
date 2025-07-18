# Step 4: Graphiti Adapter Layer

**Duration**: 4-5 days  
**Phase**: Core Integration  
**Prerequisites**: Steps 1-3 completed, basic search working

## Objective

Create a comprehensive adapter system that bridges MoRAG's existing models and interfaces with Graphiti's episode-based architecture, ensuring seamless integration while maintaining backward compatibility.

## Deliverables

1. Complete adapter architecture with bidirectional conversion
2. Model mapping between MoRAG and Graphiti schemas
3. Transaction and batch processing support
4. Error handling and validation framework
5. Comprehensive test suite with edge case coverage

## Implementation

### 1. Create Core Adapter Architecture

**File**: `packages/morag-graph/src/morag_graph/graphiti/adapters/core.py`

```python
"""Core adapter architecture for MoRAG-Graphiti integration."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')


class AdapterError(Exception):
    """Base exception for adapter-related errors."""
    pass


class ConversionError(AdapterError):
    """Error during model conversion."""
    pass


class ValidationError(AdapterError):
    """Error during validation."""
    pass


class ConversionDirection(Enum):
    """Direction of model conversion."""
    MORAG_TO_GRAPHITI = "morag_to_graphiti"
    GRAPHITI_TO_MORAG = "graphiti_to_morag"


@dataclass
class ConversionResult:
    """Result of a model conversion operation."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class BaseAdapter(ABC, Generic[T, U]):
    """Base class for all model adapters."""
    
    def __init__(self, strict_validation: bool = True):
        self.strict_validation = strict_validation
        self.conversion_stats = {
            "total_conversions": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "warnings_generated": 0
        }
    
    @abstractmethod
    def to_graphiti(self, morag_model: T) -> ConversionResult:
        """Convert MoRAG model to Graphiti format."""
        pass
    
    @abstractmethod
    def from_graphiti(self, graphiti_data: U) -> ConversionResult:
        """Convert Graphiti data to MoRAG model."""
        pass
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate input data before conversion.
        
        Args:
            data: Data to validate
            direction: Conversion direction
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if data is None:
            errors.append("Input data cannot be None")
            return errors
        
        # Subclasses should override this method for specific validation
        return errors
    
    def _record_conversion(self, success: bool, warnings_count: int = 0):
        """Record conversion statistics."""
        self.conversion_stats["total_conversions"] += 1
        if success:
            self.conversion_stats["successful_conversions"] += 1
        else:
            self.conversion_stats["failed_conversions"] += 1
        self.conversion_stats["warnings_generated"] += warnings_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        stats = self.conversion_stats.copy()
        if stats["total_conversions"] > 0:
            stats["success_rate"] = stats["successful_conversions"] / stats["total_conversions"]
        else:
            stats["success_rate"] = 0.0
        return stats


class BatchAdapter(Generic[T, U]):
    """Adapter for batch processing of multiple models."""
    
    def __init__(self, single_adapter: BaseAdapter[T, U], batch_size: int = 100):
        self.single_adapter = single_adapter
        self.batch_size = batch_size
    
    def batch_to_graphiti(self, morag_models: List[T]) -> List[ConversionResult]:
        """Convert multiple MoRAG models to Graphiti format.
        
        Args:
            morag_models: List of MoRAG models
            
        Returns:
            List of conversion results
        """
        results = []
        
        for i in range(0, len(morag_models), self.batch_size):
            batch = morag_models[i:i + self.batch_size]
            batch_results = []
            
            for model in batch:
                try:
                    result = self.single_adapter.to_graphiti(model)
                    batch_results.append(result)
                except Exception as e:
                    error_result = ConversionResult(
                        success=False,
                        error=f"Batch conversion error: {str(e)}"
                    )
                    batch_results.append(error_result)
            
            results.extend(batch_results)
            
            # Log batch progress
            logger.info(f"Processed batch {i//self.batch_size + 1}, items {i+1}-{min(i+self.batch_size, len(morag_models))}")
        
        return results
    
    def batch_from_graphiti(self, graphiti_data_list: List[U]) -> List[ConversionResult]:
        """Convert multiple Graphiti data items to MoRAG models.
        
        Args:
            graphiti_data_list: List of Graphiti data items
            
        Returns:
            List of conversion results
        """
        results = []
        
        for i in range(0, len(graphiti_data_list), self.batch_size):
            batch = graphiti_data_list[i:i + self.batch_size]
            batch_results = []
            
            for data in batch:
                try:
                    result = self.single_adapter.from_graphiti(data)
                    batch_results.append(result)
                except Exception as e:
                    error_result = ConversionResult(
                        success=False,
                        error=f"Batch conversion error: {str(e)}"
                    )
                    batch_results.append(error_result)
            
            results.extend(batch_results)
        
        return results


class AdapterRegistry:
    """Registry for managing different adapter types."""
    
    def __init__(self):
        self._adapters: Dict[str, BaseAdapter] = {}
        self._batch_adapters: Dict[str, BatchAdapter] = {}
    
    def register_adapter(self, name: str, adapter: BaseAdapter):
        """Register a single-item adapter."""
        self._adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")
    
    def register_batch_adapter(self, name: str, batch_adapter: BatchAdapter):
        """Register a batch adapter."""
        self._batch_adapters[name] = batch_adapter
        logger.info(f"Registered batch adapter: {name}")
    
    def get_adapter(self, name: str) -> Optional[BaseAdapter]:
        """Get a single-item adapter by name."""
        return self._adapters.get(name)
    
    def get_batch_adapter(self, name: str) -> Optional[BatchAdapter]:
        """Get a batch adapter by name."""
        return self._batch_adapters.get(name)
    
    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """List all registered adapters with their statistics."""
        result = {
            "single_adapters": {},
            "batch_adapters": {}
        }
        
        for name, adapter in self._adapters.items():
            result["single_adapters"][name] = adapter.get_stats()
        
        for name, batch_adapter in self._batch_adapters.items():
            result["batch_adapters"][name] = batch_adapter.single_adapter.get_stats()
        
        return result


# Global adapter registry instance
adapter_registry = AdapterRegistry()
```

### 2. Create Document Adapter Implementation

**File**: `packages/morag-graph/src/morag_graph/graphiti/adapters/document_adapter.py`

```python
"""Document adapter for MoRAG-Graphiti integration."""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from graphiti_core.nodes import EpisodeType
from morag_graph.models import Document, DocumentChunk
from .core import BaseAdapter, ConversionResult, ConversionDirection, ValidationError

logger = logging.getLogger(__name__)


class DocumentAdapter(BaseAdapter[Document, Dict[str, Any]]):
    """Adapter for converting Documents between MoRAG and Graphiti formats."""
    
    def __init__(self, strict_validation: bool = True, include_chunks: bool = True):
        super().__init__(strict_validation)
        self.include_chunks = include_chunks
        self.episode_type_mapping = {
            'text/plain': EpisodeType.text,
            'text/markdown': EpisodeType.text,
            'application/pdf': EpisodeType.text,
            'text/html': EpisodeType.text,
            'application/json': EpisodeType.json,
            'image/jpeg': EpisodeType.text,  # Graphiti doesn't have image type
            'image/png': EpisodeType.text,
            'default': EpisodeType.text
        }
    
    def to_graphiti(self, morag_model: Document) -> ConversionResult:
        """Convert MoRAG Document to Graphiti episode format.
        
        Args:
            morag_model: MoRAG Document instance
            
        Returns:
            ConversionResult with episode data
        """
        self._record_conversion(True)  # Will update if errors occur
        warnings = []
        
        try:
            # Validate input
            validation_errors = self.validate_input(morag_model, ConversionDirection.MORAG_TO_GRAPHITI)
            if validation_errors and self.strict_validation:
                self._record_conversion(False)
                return ConversionResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_errors)}"
                )
            elif validation_errors:
                warnings.extend(validation_errors)
            
            # Determine episode type
            episode_type = self.episode_type_mapping.get(
                morag_model.mime_type,
                self.episode_type_mapping['default']
            )
            
            # Create episode name
            episode_name = self._generate_episode_name(morag_model)
            
            # Create episode body (placeholder if no chunks)
            episode_body = self._create_episode_body(morag_model)
            
            # Create comprehensive metadata
            metadata = self._create_episode_metadata(morag_model)
            
            # Create source description
            source_description = self._create_source_description(morag_model)
            
            episode_data = {
                'name': episode_name,
                'episode_body': episode_body,
                'source_description': source_description,
                'episode_type': episode_type,
                'metadata': metadata
            }
            
            self._record_conversion(True, len(warnings))
            return ConversionResult(
                success=True,
                data=episode_data,
                warnings=warnings,
                metadata={
                    'original_document_id': morag_model.id,
                    'episode_type': episode_type.value,
                    'conversion_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Document conversion failed: {e}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti episode data back to MoRAG Document.
        
        Args:
            graphiti_data: Episode data from Graphiti
            
        Returns:
            ConversionResult with Document instance
        """
        try:
            # Validate that this is a MoRAG-originated episode
            metadata = graphiti_data.get('metadata', {})
            if not metadata.get('morag_integration'):
                return ConversionResult(
                    success=False,
                    error="Episode is not from MoRAG integration"
                )
            
            # Extract document data from metadata
            document = Document(
                id=metadata.get('morag_document_id'),
                name=metadata.get('file_name') or metadata.get('name'),
                source_file=metadata.get('source_file'),
                file_name=metadata.get('file_name'),
                file_size=metadata.get('file_size'),
                checksum=metadata.get('checksum'),
                mime_type=metadata.get('mime_type'),
                summary=metadata.get('summary'),
                metadata=metadata.get('original_metadata', {}),
                ingestion_timestamp=self._parse_timestamp(metadata.get('ingestion_timestamp')),
                last_modified=self._parse_timestamp(metadata.get('last_modified')),
                model=metadata.get('model')
            )
            
            self._record_conversion(True)
            return ConversionResult(
                success=True,
                data=document,
                metadata={
                    'chunk_count': metadata.get('chunk_count', 0),
                    'entity_count': metadata.get('entity_count', 0),
                    'relation_count': metadata.get('relation_count', 0)
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Graphiti to Document conversion failed: {e}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate input data for conversion."""
        errors = super().validate_input(data, direction)
        
        if direction == ConversionDirection.MORAG_TO_GRAPHITI:
            if not isinstance(data, Document):
                errors.append("Input must be a Document instance")
            elif not data.id:
                errors.append("Document must have an ID")
            elif not data.file_name and not data.name:
                errors.append("Document must have either file_name or name")
        
        elif direction == ConversionDirection.GRAPHITI_TO_MORAG:
            if not isinstance(data, dict):
                errors.append("Input must be a dictionary")
            elif 'metadata' not in data:
                errors.append("Episode data must contain metadata")
        
        return errors
    
    def _generate_episode_name(self, document: Document) -> str:
        """Generate episode name from document."""
        base_name = document.name or document.file_name or "Unknown Document"
        
        if document.mime_type and document.mime_type != 'unknown':
            file_type = document.mime_type.split('/')[-1].upper()
            return f"{base_name} ({file_type})"
        
        return base_name
    
    def _create_episode_body(self, document: Document) -> str:
        """Create episode body from document."""
        parts = []
        
        if document.summary:
            parts.append(f"Document Summary: {document.summary}")
        
        # Add document metadata as context
        if document.source_file:
            parts.append(f"Source: {document.source_file}")
        
        if document.file_size:
            size_mb = document.file_size / (1024 * 1024)
            parts.append(f"Size: {size_mb:.2f}MB")
        
        # Placeholder for actual content (will be replaced when chunks are available)
        parts.append("Document content will be populated from chunks during ingestion.")
        
        return "\n\n".join(parts)
    
    def _create_episode_metadata(self, document: Document) -> Dict[str, Any]:
        """Create comprehensive metadata for episode."""
        return {
            'morag_document_id': document.id,
            'source_file': document.source_file,
            'file_name': document.file_name,
            'file_size': document.file_size,
            'checksum': document.checksum,
            'mime_type': document.mime_type,
            'ingestion_timestamp': document.ingestion_timestamp.isoformat() if document.ingestion_timestamp else None,
            'last_modified': document.last_modified.isoformat() if document.last_modified else None,
            'model': document.model,
            'summary': document.summary,
            'original_metadata': document.metadata,
            'adapter_version': '1.0',
            'conversion_timestamp': datetime.now().isoformat(),
            'morag_integration': True,
            'adapter_type': 'document'
        }
    
    def _create_source_description(self, document: Document) -> str:
        """Create source description for episode."""
        parts = ["MoRAG Document"]
        
        if document.source_file:
            parts.append(f"Source: {Path(document.source_file).name}")
        
        if document.model:
            parts.append(f"Model: {document.model}")
        
        return " | ".join(parts)
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return None
        
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse timestamp: {timestamp_str}")
            return None


class DocumentChunkAdapter(BaseAdapter[DocumentChunk, Dict[str, Any]]):
    """Adapter for converting DocumentChunks between MoRAG and Graphiti formats."""
    
    def to_graphiti(self, morag_model: DocumentChunk) -> ConversionResult:
        """Convert DocumentChunk to Graphiti episode format."""
        try:
            episode_data = {
                'name': f"Chunk {morag_model.chunk_index}",
                'episode_body': morag_model.text,
                'source_description': f"MoRAG Chunk | Document: {morag_model.document_id}",
                'episode_type': EpisodeType.text,
                'metadata': {
                    'morag_chunk_id': morag_model.id,
                    'morag_document_id': morag_model.document_id,
                    'chunk_index': morag_model.chunk_index,
                    'text_length': len(morag_model.text),
                    'original_metadata': morag_model.metadata,
                    'adapter_version': '1.0',
                    'conversion_timestamp': datetime.now().isoformat(),
                    'morag_integration': True,
                    'adapter_type': 'chunk'
                }
            }
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=episode_data)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti episode data back to DocumentChunk."""
        try:
            metadata = graphiti_data.get('metadata', {})
            
            if metadata.get('adapter_type') != 'chunk':
                return ConversionResult(
                    success=False,
                    error="Episode is not a chunk type"
                )
            
            chunk = DocumentChunk(
                id=metadata.get('morag_chunk_id'),
                document_id=metadata.get('morag_document_id'),
                chunk_index=metadata.get('chunk_index'),
                text=graphiti_data.get('episode_body', ''),
                metadata=metadata.get('original_metadata', {})
            )
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=chunk)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate DocumentChunk input."""
        errors = super().validate_input(data, direction)
        
        if direction == ConversionDirection.MORAG_TO_GRAPHITI:
            if not isinstance(data, DocumentChunk):
                errors.append("Input must be a DocumentChunk instance")
            elif not data.text:
                errors.append("DocumentChunk must have text content")
        
        return errors
```

### 3. Create Entity and Relation Adapters

**File**: `packages/morag-graph/src/morag_graph/graphiti/adapters/entity_adapter.py`

```python
"""Entity and Relation adapters for MoRAG-Graphiti integration."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from morag_graph.models import Entity, Relation
from .core import BaseAdapter, ConversionResult, ConversionDirection

logger = logging.getLogger(__name__)


class EntityAdapter(BaseAdapter[Entity, Dict[str, Any]]):
    """Adapter for converting Entities between MoRAG and Graphiti formats."""
    
    def to_graphiti(self, morag_model: Entity) -> ConversionResult:
        """Convert MoRAG Entity to Graphiti metadata format.
        
        Note: Entities in Graphiti are embedded within episodes rather than
        stored as separate nodes. This method creates metadata that can be
        embedded in episode metadata.
        """
        try:
            entity_data = {
                'id': morag_model.id,
                'name': morag_model.name,
                'type': str(morag_model.type),
                'confidence': morag_model.confidence,
                'attributes': morag_model.attributes,
                'source_doc_id': morag_model.source_doc_id,
                'mentioned_in_chunks': morag_model.mentioned_in_chunks,
                'adapter_version': '1.0',
                'conversion_timestamp': datetime.now().isoformat(),
                'morag_integration': True,
                'adapter_type': 'entity'
            }
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=entity_data)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti entity metadata back to MoRAG Entity."""
        try:
            from morag_graph.models import EntityType
            
            entity = Entity(
                id=graphiti_data.get('id'),
                name=graphiti_data.get('name'),
                type=EntityType(graphiti_data.get('type', 'UNKNOWN')),
                confidence=graphiti_data.get('confidence', 0.0),
                attributes=graphiti_data.get('attributes', {}),
                source_doc_id=graphiti_data.get('source_doc_id'),
                mentioned_in_chunks=graphiti_data.get('mentioned_in_chunks', [])
            )
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=entity)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate Entity input."""
        errors = super().validate_input(data, direction)
        
        if direction == ConversionDirection.MORAG_TO_GRAPHITI:
            if not isinstance(data, Entity):
                errors.append("Input must be an Entity instance")
            elif not data.name:
                errors.append("Entity must have a name")
        
        return errors


class RelationAdapter(BaseAdapter[Relation, Dict[str, Any]]):
    """Adapter for converting Relations between MoRAG and Graphiti formats."""
    
    def to_graphiti(self, morag_model: Relation) -> ConversionResult:
        """Convert MoRAG Relation to Graphiti metadata format."""
        try:
            relation_data = {
                'id': morag_model.id,
                'source_entity_id': morag_model.source_entity_id,
                'target_entity_id': morag_model.target_entity_id,
                'relation_type': str(morag_model.relation_type),
                'confidence': morag_model.confidence,
                'attributes': morag_model.attributes,
                'source_doc_id': morag_model.source_doc_id,
                'mentioned_in_chunks': morag_model.mentioned_in_chunks,
                'adapter_version': '1.0',
                'conversion_timestamp': datetime.now().isoformat(),
                'morag_integration': True,
                'adapter_type': 'relation'
            }
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=relation_data)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti relation metadata back to MoRAG Relation."""
        try:
            from morag_graph.models import RelationType
            
            relation = Relation(
                id=graphiti_data.get('id'),
                source_entity_id=graphiti_data.get('source_entity_id'),
                target_entity_id=graphiti_data.get('target_entity_id'),
                relation_type=RelationType(graphiti_data.get('relation_type', 'UNKNOWN')),
                confidence=graphiti_data.get('confidence', 0.0),
                attributes=graphiti_data.get('attributes', {}),
                source_doc_id=graphiti_data.get('source_doc_id'),
                mentioned_in_chunks=graphiti_data.get('mentioned_in_chunks', [])
            )
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=relation)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate Relation input."""
        errors = super().validate_input(data, direction)
        
        if direction == ConversionDirection.MORAG_TO_GRAPHITI:
            if not isinstance(data, Relation):
                errors.append("Input must be a Relation instance")
            elif not data.source_entity_id or not data.target_entity_id:
                errors.append("Relation must have both source and target entity IDs")
        
        return errors
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_graphiti_adapter_layer.py`

```python
"""Unit tests for Graphiti adapter layer."""

import pytest
from datetime import datetime
from morag_graph.models import Document, DocumentChunk, Entity, Relation, EntityType, RelationType
from morag_graph.graphiti.adapters.core import AdapterRegistry, BatchAdapter, ConversionDirection
from morag_graph.graphiti.adapters.document_adapter import DocumentAdapter, DocumentChunkAdapter
from morag_graph.graphiti.adapters.entity_adapter import EntityAdapter, RelationAdapter


class TestDocumentAdapter:
    """Test document adapter functionality."""
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return Document(
            id="doc_123",
            name="Test Document",
            source_file="/path/to/test.pdf",
            file_name="test.pdf",
            file_size=1024000,
            checksum="abc123",
            mime_type="application/pdf",
            summary="Test document summary"
        )
    
    def test_document_to_graphiti_conversion(self, sample_document):
        """Test document to Graphiti conversion."""
        adapter = DocumentAdapter()
        result = adapter.to_graphiti(sample_document)
        
        assert result.success is True
        assert result.data is not None
        assert result.data['name'] == "Test Document (PDF)"
        assert result.data['metadata']['morag_document_id'] == "doc_123"
        assert result.data['metadata']['morag_integration'] is True
    
    def test_graphiti_to_document_conversion(self, sample_document):
        """Test Graphiti to document conversion."""
        adapter = DocumentAdapter()
        
        # First convert to Graphiti
        graphiti_result = adapter.to_graphiti(sample_document)
        assert graphiti_result.success is True
        
        # Then convert back
        episode_data = {
            'metadata': graphiti_result.data['metadata'],
            'episode_body': graphiti_result.data['episode_body']
        }
        
        document_result = adapter.from_graphiti(episode_data)
        assert document_result.success is True
        assert document_result.data.id == sample_document.id
        assert document_result.data.file_name == sample_document.file_name
    
    def test_validation_errors(self):
        """Test validation error handling."""
        adapter = DocumentAdapter(strict_validation=True)
        
        # Test with invalid input
        result = adapter.to_graphiti(None)
        assert result.success is False
        assert "cannot be None" in result.error
        
        # Test with document missing required fields
        invalid_doc = Document(id="", name="", file_name="")
        result = adapter.to_graphiti(invalid_doc)
        assert result.success is False


class TestEntityAdapter:
    """Test entity adapter functionality."""
    
    @pytest.fixture
    def sample_entity(self):
        """Create sample entity."""
        return Entity(
            id="entity_123",
            name="John Doe",
            type=EntityType.PERSON,
            confidence=0.95,
            attributes={"role": "engineer"},
            source_doc_id="doc_123"
        )
    
    def test_entity_to_graphiti_conversion(self, sample_entity):
        """Test entity to Graphiti conversion."""
        adapter = EntityAdapter()
        result = adapter.to_graphiti(sample_entity)
        
        assert result.success is True
        assert result.data['name'] == "John Doe"
        assert result.data['type'] == "PERSON"
        assert result.data['confidence'] == 0.95
        assert result.data['morag_integration'] is True
    
    def test_graphiti_to_entity_conversion(self, sample_entity):
        """Test Graphiti to entity conversion."""
        adapter = EntityAdapter()
        
        # Convert to Graphiti format
        graphiti_result = adapter.to_graphiti(sample_entity)
        
        # Convert back
        entity_result = adapter.from_graphiti(graphiti_result.data)
        assert entity_result.success is True
        assert entity_result.data.name == sample_entity.name
        assert entity_result.data.type == sample_entity.type


class TestBatchAdapter:
    """Test batch processing functionality."""
    
    def test_batch_document_conversion(self):
        """Test batch document conversion."""
        documents = [
            Document(id=f"doc_{i}", name=f"Document {i}", file_name=f"doc{i}.txt")
            for i in range(5)
        ]
        
        single_adapter = DocumentAdapter()
        batch_adapter = BatchAdapter(single_adapter, batch_size=2)
        
        results = batch_adapter.batch_to_graphiti(documents)
        
        assert len(results) == 5
        assert all(result.success for result in results)
        assert all(result.data['metadata']['morag_integration'] for result in results)


class TestAdapterRegistry:
    """Test adapter registry functionality."""
    
    def test_adapter_registration(self):
        """Test adapter registration and retrieval."""
        registry = AdapterRegistry()
        
        doc_adapter = DocumentAdapter()
        registry.register_adapter("document", doc_adapter)
        
        retrieved_adapter = registry.get_adapter("document")
        assert retrieved_adapter is doc_adapter
    
    def test_batch_adapter_registration(self):
        """Test batch adapter registration."""
        registry = AdapterRegistry()
        
        doc_adapter = DocumentAdapter()
        batch_adapter = BatchAdapter(doc_adapter)
        registry.register_batch_adapter("document_batch", batch_adapter)
        
        retrieved_batch = registry.get_batch_adapter("document_batch")
        assert retrieved_batch is batch_adapter
    
    def test_adapter_statistics(self):
        """Test adapter statistics tracking."""
        registry = AdapterRegistry()
        
        doc_adapter = DocumentAdapter()
        registry.register_adapter("document", doc_adapter)
        
        # Perform some conversions
        sample_doc = Document(id="test", name="Test", file_name="test.txt")
        doc_adapter.to_graphiti(sample_doc)
        doc_adapter.to_graphiti(sample_doc)
        
        stats = registry.list_adapters()
        assert "single_adapters" in stats
        assert "document" in stats["single_adapters"]
        assert stats["single_adapters"]["document"]["total_conversions"] == 2


@pytest.mark.integration
class TestAdapterIntegration:
    """Integration tests for adapter layer."""
    
    def test_full_document_lifecycle(self):
        """Test complete document conversion lifecycle."""
        # Create test data
        document = Document(
            id="integration_test",
            name="Integration Test Document",
            file_name="integration.pdf",
            mime_type="application/pdf"
        )
        
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id=document.id,
                chunk_index=0,
                text="First chunk content"
            ),
            DocumentChunk(
                id="chunk_2", 
                document_id=document.id,
                chunk_index=1,
                text="Second chunk content"
            )
        ]
        
        entities = [
            Entity(
                id="entity_1",
                name="Test Entity",
                type=EntityType.ORGANIZATION,
                confidence=0.8
            )
        ]
        
        # Test conversions
        doc_adapter = DocumentAdapter()
        chunk_adapter = DocumentChunkAdapter()
        entity_adapter = EntityAdapter()
        
        # Convert document
        doc_result = doc_adapter.to_graphiti(document)
        assert doc_result.success is True
        
        # Convert chunks
        chunk_results = [chunk_adapter.to_graphiti(chunk) for chunk in chunks]
        assert all(result.success for result in chunk_results)
        
        # Convert entities
        entity_results = [entity_adapter.to_graphiti(entity) for entity in entities]
        assert all(result.success for result in entity_results)
        
        # Test reverse conversions
        doc_back = doc_adapter.from_graphiti({
            'metadata': doc_result.data['metadata'],
            'episode_body': doc_result.data['episode_body']
        })
        assert doc_back.success is True
        assert doc_back.data.id == document.id
```

## Validation Checklist

- [ ] Core adapter architecture implemented with proper error handling
- [ ] Document adapter converts bidirectionally without data loss
- [ ] DocumentChunk adapter preserves all chunk information
- [ ] Entity and Relation adapters handle metadata correctly
- [ ] Batch processing works efficiently for large datasets
- [ ] Adapter registry manages multiple adapter types
- [ ] Validation framework catches input errors appropriately
- [ ] Statistics tracking works for monitoring conversions
- [ ] Unit tests cover all adapter functionality
- [ ] Integration tests validate complete workflows

## Success Criteria

1. **Bidirectional**: All adapters support both MoRAG→Graphiti and Graphiti→MoRAG conversion
2. **Data Integrity**: No data loss during conversion processes
3. **Error Handling**: Robust error handling with meaningful error messages
4. **Performance**: Batch processing handles large datasets efficiently
5. **Extensible**: Easy to add new adapter types for future models

## Next Steps

After completing this step:
1. Validate adapter performance with large datasets
2. Test edge cases and error conditions thoroughly
3. Document adapter usage patterns and best practices
4. Proceed to [Step 5: Entity and Relation Migration](./step-05-entity-relation-migration.md)

## Performance Considerations

- Batch processing reduces overhead for large conversions
- Validation can be disabled for performance-critical operations
- Statistics tracking adds minimal overhead
- Memory usage scales with batch size and model complexity

## Error Recovery

- Failed conversions don't affect batch processing
- Detailed error messages help with debugging
- Conversion statistics help identify problematic patterns
- Rollback capabilities for batch operations
