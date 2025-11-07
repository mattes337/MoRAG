# MoRAG Refactoring Task 3: Code Deduplication

## Priority: MEDIUM
## Estimated Time: 6-10 hours
## Impact: Significant codebase reduction and maintainability improvement

## Overview
This task identifies and eliminates duplicate code patterns across the MoRAG codebase. Analysis shows significant duplication in Storage classes (27 files), Processor classes (34 files), and common patterns like initialization (401 `__init__` methods).

## Major Duplication Categories

### 1. Storage Class Pattern Duplication (27 files)
**Files Affected**: All storage implementations across packages
```
morag-graph/storage/: qdrant_storage.py, neo4j_storage.py, json_storage.py
morag-services/: storage.py
morag-core/interfaces/: storage.py
```

**Duplicate Patterns**:
- Connection management boilerplate
- CRUD operation structures
- Error handling patterns
- Health check implementations
- Configuration initialization

**Consolidation Strategy**:
```python
# Create unified base storage class:
# packages/morag-core/src/morag_core/storage/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import structlog

class BaseStorage(ABC):
    """Unified base class for all storage implementations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._connection = None

    async def connect(self) -> bool:
        """Standard connection pattern."""
        try:
            self._connection = await self._establish_connection()
            await self._validate_connection()
            return True
        except Exception as e:
            self.logger.error("Connection failed", error=str(e))
            return False

    @abstractmethod
    async def _establish_connection(self):
        """Subclass-specific connection logic."""
        pass

    @abstractmethod
    async def _validate_connection(self):
        """Subclass-specific validation logic."""
        pass

    # Common CRUD patterns...
    async def health_check(self) -> Dict[str, Any]:
        """Standard health check pattern."""
        return {
            "status": "healthy" if self._connection else "disconnected",
            "connection": bool(self._connection),
            "config": {k: v for k, v in self.config.items() if "password" not in k.lower()}
        }
```

**Estimated Reduction**: 800+ lines across storage files

### 2. Processor Class Pattern Duplication (34 files)
**Files Affected**: All processor implementations
```
morag-audio/: processor.py
morag-video/: processor.py
morag-document/: processor.py
morag-web/: processor.py
morag-stages/processors/: *.py (8 files)
```

**Duplicate Patterns**:
- Processing pipeline structure
- Error handling and logging
- Result formatting
- Configuration management
- Progress tracking

**Consolidation Strategy**:
```python
# packages/morag-core/src/morag_core/processing/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from morag_core.models import ProcessingResult
from morag_core.utils.logging import get_logger

class BaseProcessor(ABC):
    """Unified base processor with common processing patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

    async def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """Standard processing pipeline."""
        try:
            # Pre-processing validation
            validated_input = await self._validate_input(input_data)

            # Progress tracking
            progress_callback = kwargs.get('progress_callback')
            if progress_callback:
                await progress_callback(0, "Starting processing")

            # Main processing
            result = await self._process_implementation(validated_input, **kwargs)

            # Post-processing
            formatted_result = await self._format_result(result)

            if progress_callback:
                await progress_callback(100, "Processing complete")

            return formatted_result

        except Exception as e:
            self.logger.error("Processing failed", error=str(e), input_type=type(input_data))
            raise

    @abstractmethod
    async def _validate_input(self, input_data: Any) -> Any:
        """Subclass-specific input validation."""
        pass

    @abstractmethod
    async def _process_implementation(self, input_data: Any, **kwargs) -> Any:
        """Subclass-specific processing logic."""
        pass

    @abstractmethod
    async def _format_result(self, result: Any) -> ProcessingResult:
        """Subclass-specific result formatting."""
        pass
```

**Estimated Reduction**: 600+ lines across processor files

### 3. Service Class Initialization Patterns (27 service classes)
**Duplicate Patterns**:
- Client initialization (httpx, database clients)
- Configuration loading
- Logger setup
- Health check methods
- Error handling wrappers

**Consolidation Strategy**:
```python
# packages/morag-core/src/morag_core/services/base.py

class BaseService(ABC):
    """Unified base service class."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._load_config(config)
        self.logger = get_logger(self.__class__.__name__)
        self._client = None

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Standard configuration loading pattern."""
        base_config = {
            'timeout': 30,
            'retries': 3,
            'backoff_factor': 0.3
        }
        if config:
            base_config.update(config)
        return base_config

    async def initialize(self) -> bool:
        """Standard service initialization."""
        try:
            self._client = await self._create_client()
            await self._validate_service()
            return True
        except Exception as e:
            self.logger.error("Service initialization failed", error=str(e))
            return False

    @abstractmethod
    async def _create_client(self):
        """Subclass-specific client creation."""
        pass
```

### 4. Error Handling Pattern Duplication
**Found in**: 150+ files with similar try-catch patterns

**Common Pattern**:
```python
# Duplicated across many files:
try:
    result = await some_operation()
    return result
except SomeSpecificError as e:
    logger.error("Operation failed", error=str(e))
    raise ProcessingError(f"Failed to process: {e}")
except Exception as e:
    logger.error("Unexpected error", error=str(e))
    raise InternalError("Internal processing error")
```

**Consolidation Strategy**:
```python
# packages/morag-core/src/morag_core/utils/error_handling.py

from functools import wraps
from typing import Callable, Type, Dict, Any
import structlog

def standard_error_handling(
    logger: structlog.BoundLogger,
    error_mapping: Dict[Type[Exception], Type[Exception]] = None
):
    """Decorator for standard error handling patterns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_type = type(e)
                if error_mapping and error_type in error_mapping:
                    mapped_error = error_mapping[error_type]
                    logger.error(f"{func.__name__} failed",
                               error=str(e), error_type=error_type.__name__)
                    raise mapped_error(f"Failed to {func.__name__}: {e}")
                logger.error(f"Unexpected error in {func.__name__}",
                           error=str(e), error_type=error_type.__name__)
                raise
        return wrapper
    return decorator

# Usage:
@standard_error_handling(logger, {ValueError: ProcessingError})
async def process_data(data):
    # Processing logic
    pass
```

### 5. Configuration Loading Duplication
**Pattern**: Same configuration loading logic in 50+ files

**Consolidation Strategy**:
```python
# packages/morag-core/src/morag_core/config/loader.py

from typing import Dict, Any, Optional, Union
from pathlib import Path
import os
from dotenv import load_dotenv

class ConfigLoader:
    """Unified configuration loading utility."""

    @staticmethod
    def load_service_config(
        service_name: str,
        env_prefix: str = "MORAG",
        defaults: Optional[Dict[str, Any]] = None,
        config_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Load configuration with standard precedence:
        1. Explicit config_file
        2. Environment variables
        3. Default values
        """
        config = defaults or {}

        # Load from environment
        load_dotenv()
        env_config = {}
        prefix = f"{env_prefix}_{service_name.upper()}_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_config[config_key] = value

        config.update(env_config)

        # Load from file if provided
        if config_file:
            file_config = ConfigLoader._load_from_file(config_file)
            config.update(file_config)

        return config
```

## Implementation Plan

### Phase 1: Base Classes Creation (3 hours)
1. **Create base storage class** (1 hour)
   - `morag-core/storage/base.py`
   - Common connection, CRUD, health check patterns

2. **Create base processor class** (1 hour)
   - `morag-core/processing/base.py`
   - Standard processing pipeline

3. **Create base service class** (1 hour)
   - `morag-core/services/base.py`
   - Client initialization and configuration

### Phase 2: Utility Functions (2 hours)
1. **Error handling decorator** (30 minutes)
2. **Configuration loader** (30 minutes)
3. **Logger utility enhancement** (30 minutes)
4. **Progress tracking utility** (30 minutes)

### Phase 3: Migration to Base Classes (4 hours)
1. **Migrate storage classes** (1.5 hours)
   - Update 27 storage files to inherit from BaseStorage
   - Remove duplicated code
   - Test functionality preservation

2. **Migrate processor classes** (1.5 hours)
   - Update 34 processor files to inherit from BaseProcessor
   - Remove duplicated patterns
   - Test processing pipeline

3. **Migrate service classes** (1 hour)
   - Update service classes to inherit from BaseService
   - Remove initialization duplication

### Phase 4: Pattern Replacement (2 hours)
1. **Replace error handling patterns** (1 hour)
   - Apply error handling decorators
   - Remove try-catch boilerplate

2. **Replace configuration patterns** (1 hour)
   - Use ConfigLoader utility
   - Remove custom config loading

## Expected Results

### Code Reduction Metrics
- **Storage files**: 800+ lines removed (30% reduction)
- **Processor files**: 600+ lines removed (25% reduction)
- **Service files**: 400+ lines removed (20% reduction)
- **Error handling**: 300+ lines removed (duplicated try-catch)
- **Configuration**: 200+ lines removed (config loading)

**Total Estimated Reduction**: ~2,300 lines (8-10% of codebase)

### Quality Improvements
- Consistent error handling across all modules
- Standardized logging patterns
- Uniform configuration management
- Easier testing with base class mocking
- Reduced maintenance burden

## Testing Strategy

### Unit Tests for Base Classes
```python
# tests/unit/core/test_base_storage.py
# tests/unit/core/test_base_processor.py
# tests/unit/core/test_base_service.py
# tests/unit/utils/test_error_handling.py
# tests/unit/utils/test_config_loader.py
```

### Migration Validation Tests
```python
# For each migrated class, verify:
# 1. Same functionality as before
# 2. Same API surface
# 3. Same error behavior
# 4. Performance not degraded

# Example test:
async def test_storage_migration_compatibility():
    # Test that QdrantStorage still works identically after migration
    old_behavior = await test_old_qdrant_storage()
    new_behavior = await test_new_qdrant_storage()
    assert old_behavior == new_behavior
```

## Success Criteria

### Quantitative Goals
- [ ] ≥2000 lines of code removed
- [ ] ≥50% reduction in duplicate patterns
- [ ] All tests pass after migration
- [ ] No performance regression >5%

### Qualitative Goals
- [ ] Consistent patterns across all storage/processor/service classes
- [ ] Single source of truth for common functionality
- [ ] Easier to add new storage/processor implementations
- [ ] Improved error messages and logging consistency

## Risk Mitigation

### Gradual Migration Approach
```bash
# Migrate one category at a time:
1. Create base classes first (test in isolation)
2. Migrate 1-2 storage classes (validate approach)
3. Migrate remaining storage classes
4. Repeat for processors and services
```

### Compatibility Testing
```bash
# After each migration batch:
pytest tests/integration/ -v
python tests/cli/test-all.py
```

### Rollback Strategy
```bash
# Each phase gets its own branch:
git checkout -b refactor/dedup-storage-classes
git checkout -b refactor/dedup-processor-classes
git checkout -b refactor/dedup-service-classes
```

## Dependencies
- **Requires**: Completion of file splitting (Task 2) for cleaner base class creation
- **Enables**: Easier dependency cleanup (Task 4) and testing (Task 5) with consistent patterns

## Next Steps After Completion
1. **04-dependency-cleanup.md**: Remove unused dependencies now that patterns are consolidated
2. **05-testing-strategy.md**: Create comprehensive tests for base classes and consolidated patterns
3. **Performance optimization**: Profile reduced codebase for further optimizations
