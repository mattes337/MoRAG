"""Tests for stage registry."""

import pytest
from pathlib import Path
from typing import List

from morag_stages.models import Stage, StageType, StageResult, StageContext
from morag_stages.registry import StageRegistry
from morag_stages.exceptions import StageError, StageDependencyError


class MockStage(Stage):
    """Mock stage for testing."""
    
    def __init__(self, stage_type: StageType, dependencies: List[StageType] = None):
        super().__init__(stage_type)
        self._dependencies = dependencies or []
    
    async def execute(self, input_files: List[Path], context: StageContext) -> StageResult:
        """Mock execute method."""
        pass
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Mock validate method."""
        return True
    
    def get_dependencies(self) -> List[StageType]:
        """Mock dependencies method."""
        return self._dependencies
    
    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Mock expected outputs method."""
        return []


class TestStageRegistry:
    """Test StageRegistry class."""
    
    def test_register_stage(self):
        """Test registering a stage."""
        registry = StageRegistry()
        
        class TestStage(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION)
        
        # Register stage
        registry.register_stage(TestStage)
        
        # Check it's registered
        assert registry.is_registered(StageType.MARKDOWN_CONVERSION)
        assert StageType.MARKDOWN_CONVERSION in registry.get_registered_stages()
        
        # Get stage instance
        stage = registry.get_stage(StageType.MARKDOWN_CONVERSION)
        assert isinstance(stage, TestStage)
        assert stage.stage_type == StageType.MARKDOWN_CONVERSION
    
    def test_register_duplicate_stage(self):
        """Test registering duplicate stage raises error."""
        registry = StageRegistry()
        
        class TestStage1(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION)
        
        class TestStage2(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION)
        
        # Register first stage
        registry.register_stage(TestStage1)
        
        # Try to register duplicate
        with pytest.raises(StageError, match="already registered"):
            registry.register_stage(TestStage2)
    
    def test_unregister_stage(self):
        """Test unregistering a stage."""
        registry = StageRegistry()
        
        class TestStage(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION)
        
        # Register and then unregister
        registry.register_stage(TestStage)
        assert registry.is_registered(StageType.MARKDOWN_CONVERSION)
        
        registry.unregister_stage(StageType.MARKDOWN_CONVERSION)
        assert not registry.is_registered(StageType.MARKDOWN_CONVERSION)
    
    def test_get_unregistered_stage(self):
        """Test getting unregistered stage raises error."""
        registry = StageRegistry()
        
        with pytest.raises(StageError, match="not registered"):
            registry.get_stage(StageType.MARKDOWN_CONVERSION)
    
    def test_validate_stage_chain_success(self):
        """Test validating successful stage chain."""
        registry = StageRegistry()
        
        # Register stages with dependencies
        class Stage1(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION, [])
        
        class Stage2(MockStage):
            def __init__(self):
                super().__init__(StageType.CHUNKER, [StageType.MARKDOWN_CONVERSION])
        
        class Stage3(MockStage):
            def __init__(self):
                super().__init__(StageType.FACT_GENERATOR, [StageType.CHUNKER])
        
        registry.register_stage(Stage1)
        registry.register_stage(Stage2)
        registry.register_stage(Stage3)
        
        # Test valid chain
        chain = [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER, StageType.FACT_GENERATOR]
        assert registry.validate_stage_chain(chain) is True
    
    def test_validate_stage_chain_missing_dependency(self):
        """Test validating stage chain with missing dependency."""
        registry = StageRegistry()
        
        class Stage1(MockStage):
            def __init__(self):
                super().__init__(StageType.CHUNKER, [StageType.MARKDOWN_CONVERSION])
        
        registry.register_stage(Stage1)
        
        # Test chain missing dependency
        chain = [StageType.CHUNKER]
        with pytest.raises(StageDependencyError, match="unsatisfied dependencies"):
            registry.validate_stage_chain(chain)
    
    def test_get_dependency_order(self):
        """Test getting stages in dependency order."""
        registry = StageRegistry()
        
        # Register stages with dependencies
        class Stage1(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION, [])
        
        class Stage2(MockStage):
            def __init__(self):
                super().__init__(StageType.CHUNKER, [StageType.MARKDOWN_CONVERSION])
        
        class Stage3(MockStage):
            def __init__(self):
                super().__init__(StageType.FACT_GENERATOR, [StageType.CHUNKER])
        
        class Stage4(MockStage):
            def __init__(self):
                super().__init__(StageType.INGESTOR, [StageType.CHUNKER, StageType.FACT_GENERATOR])
        
        registry.register_stage(Stage1)
        registry.register_stage(Stage2)
        registry.register_stage(Stage3)
        registry.register_stage(Stage4)
        
        # Test ordering
        unordered = [StageType.INGESTOR, StageType.FACT_GENERATOR, StageType.MARKDOWN_CONVERSION, StageType.CHUNKER]
        ordered = registry.get_dependency_order(unordered)
        
        expected = [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER, StageType.FACT_GENERATOR, StageType.INGESTOR]
        assert ordered == expected
    
    def test_get_dependency_order_circular(self):
        """Test circular dependency detection."""
        registry = StageRegistry()
        
        # Create circular dependency
        class Stage1(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION, [StageType.CHUNKER])
        
        class Stage2(MockStage):
            def __init__(self):
                super().__init__(StageType.CHUNKER, [StageType.MARKDOWN_CONVERSION])
        
        registry.register_stage(Stage1)
        registry.register_stage(Stage2)
        
        # Test circular dependency detection
        chain = [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER]
        with pytest.raises(StageDependencyError, match="Circular dependency"):
            registry.get_dependency_order(chain)
    
    def test_get_optional_stages(self):
        """Test getting optional stages."""
        registry = StageRegistry()
        
        class RequiredStage(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION)
            
            def is_optional(self) -> bool:
                return False
        
        class OptionalStage(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_OPTIMIZER)
            
            def is_optional(self) -> bool:
                return True
        
        registry.register_stage(RequiredStage)
        registry.register_stage(OptionalStage)
        
        optional_stages = registry.get_optional_stages()
        assert StageType.MARKDOWN_OPTIMIZER in optional_stages
        assert StageType.MARKDOWN_CONVERSION not in optional_stages
    
    def test_get_required_stages(self):
        """Test getting required stages."""
        registry = StageRegistry()
        
        class RequiredStage(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION)
            
            def is_optional(self) -> bool:
                return False
        
        class OptionalStage(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_OPTIMIZER)
            
            def is_optional(self) -> bool:
                return True
        
        registry.register_stage(RequiredStage)
        registry.register_stage(OptionalStage)
        
        required_stages = registry.get_required_stages()
        assert StageType.MARKDOWN_CONVERSION in required_stages
        assert StageType.MARKDOWN_OPTIMIZER not in required_stages
    
    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = StageRegistry()
        
        class TestStage(MockStage):
            def __init__(self):
                super().__init__(StageType.MARKDOWN_CONVERSION)
        
        # Register stage
        registry.register_stage(TestStage)
        assert len(registry.get_registered_stages()) == 1
        
        # Clear registry
        registry.clear()
        assert len(registry.get_registered_stages()) == 0
        assert not registry.is_registered(StageType.MARKDOWN_CONVERSION)
