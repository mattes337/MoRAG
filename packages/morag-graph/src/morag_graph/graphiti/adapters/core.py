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
