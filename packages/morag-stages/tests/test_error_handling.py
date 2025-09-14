"""Test error handling decorators."""

import pytest
from unittest.mock import AsyncMock
from pathlib import Path

from morag_stages.error_handling import (
    stage_error_handler,
    validation_error_handler,
    standalone_validation_handler,
)
from morag_stages.exceptions import (
    StageExecutionError,
    StageValidationError,
)


class MockStage:
    """Mock stage class for testing."""
    def __init__(self):
        self.stage_type = "test_stage"


@pytest.mark.asyncio
async def test_stage_error_handler_propagates_stage_errors():
    """Test that stage errors are propagated correctly."""
    
    @stage_error_handler("test_operation")
    async def failing_method(self):
        raise StageExecutionError("Test stage error")
    
    stage = MockStage()
    
    with pytest.raises(StageExecutionError, match="Test stage error"):
        await failing_method(stage)


@pytest.mark.asyncio
async def test_stage_error_handler_wraps_generic_errors():
    """Test that generic errors are wrapped in StageExecutionError."""
    
    @stage_error_handler("test_operation")
    async def failing_method(self):
        raise ValueError("Generic error")
    
    stage = MockStage()
    
    with pytest.raises(StageExecutionError, match="test_operation failed: Generic error"):
        await failing_method(stage)


def test_validation_error_handler_propagates_validation_errors():
    """Test that validation errors are propagated correctly."""
    
    @validation_error_handler("test_validation")
    def failing_validation(self):
        raise StageValidationError("Test validation error")
    
    stage = MockStage()
    
    with pytest.raises(StageValidationError, match="Test validation error"):
        failing_validation(stage)


def test_validation_error_handler_wraps_generic_errors():
    """Test that generic errors are wrapped in StageValidationError."""
    
    @validation_error_handler("test_validation")
    def failing_validation(self):
        raise ValueError("Generic validation error")
    
    stage = MockStage()
    
    with pytest.raises(StageValidationError, match="test_validation validation failed"):
        failing_validation(stage)


def test_standalone_validation_handler_returns_false():
    """Test that standalone validation handler returns False on error."""
    
    @standalone_validation_handler("test_standalone")
    def failing_validation(content: str) -> bool:
        raise ValueError("Something went wrong")
    
    # Should return False instead of raising
    result = failing_validation("test content")
    assert result is False


def test_standalone_validation_handler_allows_success():
    """Test that standalone validation handler allows success."""
    
    @standalone_validation_handler("test_standalone")
    def successful_validation(content: str) -> bool:
        return len(content) > 0
    
    # Should return True for valid input
    result = successful_validation("test content")
    assert result is True


@pytest.mark.asyncio
async def test_stage_error_handler_with_sync_function():
    """Test that stage error handler works with sync functions."""
    
    @stage_error_handler("test_sync_operation")
    def sync_failing_method(self):
        raise ValueError("Sync generic error")
    
    stage = MockStage()
    
    with pytest.raises(StageExecutionError, match="test_sync_operation failed: Sync generic error"):
        sync_failing_method(stage)


def test_error_handlers_preserve_function_metadata():
    """Test that decorators preserve function metadata."""
    
    @stage_error_handler("test_operation")
    async def test_function(self):
        """Test function docstring."""
        pass
    
    assert test_function.__name__ == "test_function"
    assert "Test function docstring" in test_function.__doc__
    
    @validation_error_handler("test_validation")
    def test_validation(self):
        """Test validation docstring."""
        pass
    
    assert test_validation.__name__ == "test_validation"
    assert "Test validation docstring" in test_validation.__doc__
    
    @standalone_validation_handler("test_standalone")
    def test_standalone():
        """Test standalone docstring."""
        pass
    
    assert test_standalone.__name__ == "test_standalone"
    assert "Test standalone docstring" in test_standalone.__doc__