"""Stage registry for managing available stages."""

from typing import Dict, List, Optional, Type, Set
import structlog

from .models import Stage, StageType
from .exceptions import StageError, StageDependencyError

logger = structlog.get_logger(__name__)


class StageRegistry:
    """Registry for managing available processing stages."""
    
    def __init__(self):
        """Initialize empty stage registry."""
        self._stages: Dict[StageType, Type[Stage]] = {}
        self._instances: Dict[StageType, Stage] = {}
        
    def register_stage(self, stage_class: Type[Stage]) -> None:
        """Register a stage class.

        Args:
            stage_class: Stage class to register

        Raises:
            StageError: If stage type is already registered
        """
        # Create temporary instance to get stage type
        try:
            temp_instance = stage_class()
            stage_type = temp_instance.stage_type
        except TypeError:
            # Fallback for stages that require stage_type parameter
            # Try to infer stage type from class name
            class_name = stage_class.__name__.lower()
            if 'markdown_conversion' in class_name or 'markdownconversion' in class_name:
                stage_type = StageType.MARKDOWN_CONVERSION
            elif 'markdown_optimizer' in class_name or 'markdownoptimizer' in class_name:
                stage_type = StageType.MARKDOWN_OPTIMIZER
            elif 'chunker' in class_name:
                stage_type = StageType.CHUNKER
            elif 'fact_generator' in class_name or 'factgenerator' in class_name:
                stage_type = StageType.FACT_GENERATOR
            elif 'ingestor' in class_name:
                stage_type = StageType.INGESTOR
            else:
                # Default fallback
                stage_type = StageType.MARKDOWN_CONVERSION

            temp_instance = stage_class(stage_type)
            stage_type = temp_instance.stage_type
        
        if stage_type in self._stages:
            raise StageError(f"Stage type {stage_type.value} is already registered")
        
        self._stages[stage_type] = stage_class
        logger.info("Stage registered", stage_type=stage_type.value, stage_class=stage_class.__name__)
    
    def unregister_stage(self, stage_type: StageType) -> None:
        """Unregister a stage type.
        
        Args:
            stage_type: Stage type to unregister
        """
        if stage_type in self._stages:
            del self._stages[stage_type]
            
        if stage_type in self._instances:
            del self._instances[stage_type]
            
        logger.info("Stage unregistered", stage_type=stage_type.value)
    
    def get_stage(self, stage_type: StageType) -> Stage:
        """Get stage instance for given type.
        
        Args:
            stage_type: Stage type to get instance for
            
        Returns:
            Stage instance
            
        Raises:
            StageError: If stage type is not registered
        """
        if stage_type not in self._stages:
            raise StageError(f"Stage type {stage_type.value} is not registered")
        
        # Return cached instance or create new one
        if stage_type not in self._instances:
            stage_class = self._stages[stage_type]
            try:
                self._instances[stage_type] = stage_class()
            except TypeError:
                # Fallback for stages that require stage_type parameter
                self._instances[stage_type] = stage_class(stage_type)
        
        return self._instances[stage_type]
    
    def is_registered(self, stage_type: StageType) -> bool:
        """Check if stage type is registered.
        
        Args:
            stage_type: Stage type to check
            
        Returns:
            True if registered, False otherwise
        """
        return stage_type in self._stages
    
    def get_registered_stages(self) -> List[StageType]:
        """Get list of all registered stage types.
        
        Returns:
            List of registered stage types
        """
        return list(self._stages.keys())
    
    def validate_stage_chain(self, stage_types: List[StageType]) -> bool:
        """Validate that a stage chain has all dependencies satisfied.
        
        Args:
            stage_types: List of stage types in execution order
            
        Returns:
            True if chain is valid, False otherwise
            
        Raises:
            StageDependencyError: If dependencies are not satisfied
        """
        completed_stages: Set[StageType] = set()
        
        for stage_type in stage_types:
            if not self.is_registered(stage_type):
                raise StageError(f"Stage type {stage_type.value} is not registered")
            
            stage = self.get_stage(stage_type)
            dependencies = stage.get_dependencies()
            
            # Check if all dependencies are satisfied
            missing_deps = [dep for dep in dependencies if dep not in completed_stages]
            if missing_deps:
                missing_names = [dep.value for dep in missing_deps]
                raise StageDependencyError(
                    f"Stage {stage_type.value} has unsatisfied dependencies: {missing_names}",
                    stage_type=stage_type.value,
                    missing_dependencies=missing_names
                )
            
            completed_stages.add(stage_type)
        
        return True
    
    def get_dependency_order(self, stage_types: List[StageType]) -> List[StageType]:
        """Get stages in dependency order.

        Args:
            stage_types: List of stage types to order

        Returns:
            List of stage types in dependency order

        Raises:
            StageDependencyError: If circular dependencies exist
        """
        # Build dependency graph - only consider dependencies within the requested stages
        dependencies: Dict[StageType, Set[StageType]] = {}
        requested_stages = set(stage_types)

        for stage_type in stage_types:
            if not self.is_registered(stage_type):
                raise StageError(f"Stage type {stage_type.value} is not registered")

            stage = self.get_stage(stage_type)
            all_deps = set(stage.get_dependencies())
            # Only include dependencies that are also in the requested stages
            filtered_deps = all_deps.intersection(requested_stages)
            dependencies[stage_type] = filtered_deps

        # Topological sort
        ordered: List[StageType] = []
        remaining = set(stage_types)
        # Keep track of original order for tie-breaking
        original_order = {stage: i for i, stage in enumerate(stage_types)}

        while remaining:
            # Find stages with no unresolved dependencies within the requested set
            ready = [
                stage_type for stage_type in remaining
                if dependencies[stage_type].issubset(set(ordered))
            ]

            if not ready:
                # Check if we have a true circular dependency or just missing external dependencies
                unresolved_deps = set()
                for stage_type in remaining:
                    unresolved_deps.update(dependencies[stage_type] - set(ordered))

                if unresolved_deps.issubset(remaining):
                    # True circular dependency within the requested stages
                    raise StageDependencyError(
                        f"Circular dependency detected in stages: {[s.value for s in remaining]}"
                    )
                else:
                    # External dependencies - pick stages in original order
                    # This preserves user's intended order when no internal dependencies exist
                    ready = [min(remaining, key=lambda x: original_order[x])]

            # Sort ready stages by original order to maintain user's preference
            ready.sort(key=lambda x: original_order[x])

            # Add ready stages to ordered list
            for stage_type in ready:
                ordered.append(stage_type)
                remaining.remove(stage_type)

        return ordered
    
    def get_optional_stages(self) -> List[StageType]:
        """Get list of optional stages.
        
        Returns:
            List of optional stage types
        """
        optional_stages = []
        for stage_type in self._stages:
            stage = self.get_stage(stage_type)
            if stage.is_optional():
                optional_stages.append(stage_type)
        return optional_stages
    
    def get_required_stages(self) -> List[StageType]:
        """Get list of required stages.
        
        Returns:
            List of required stage types
        """
        required_stages = []
        for stage_type in self._stages:
            stage = self.get_stage(stage_type)
            if not stage.is_optional():
                required_stages.append(stage_type)
        return required_stages
    
    def clear(self) -> None:
        """Clear all registered stages."""
        self._stages.clear()
        self._instances.clear()
        logger.info("Stage registry cleared")


# Global registry instance
_global_registry = StageRegistry()


def get_global_registry() -> StageRegistry:
    """Get the global stage registry instance.
    
    Returns:
        Global stage registry
    """
    return _global_registry


def register_stage(stage_class: Type[Stage]) -> None:
    """Register a stage class in the global registry.
    
    Args:
        stage_class: Stage class to register
    """
    _global_registry.register_stage(stage_class)
