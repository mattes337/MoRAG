# Task 37: Repository Structure Optimization

## Overview
Optimize repository structure and consolidate scattered components to create a unified, maintainable, and scalable codebase architecture. This task addresses inconsistencies between monolithic and modular structures and establishes clear development patterns.

## Status
- **Current Status**: NEW
- **Priority**: MEDIUM
- **Estimated Effort**: 2-3 days
- **Dependencies**: Task 36 (cleanup completion)

## Problem Statement

### Current Issues

#### 1. Architectural Inconsistency
- **Dual Architecture**: Both monolithic (`src/morag`) and modular (`packages/`) structures coexist
- **Unclear Guidance**: No clear documentation on which approach to use for new development
- **Import Confusion**: Mixed import patterns throughout codebase
- **Development Friction**: Developers unsure which structure to follow

#### 2. Testing Gaps
- **Missing Integration Tests**: No comprehensive tests validating modular architecture works correctly
- **Legacy Test References**: Tests may still reference old import paths
- **Package Isolation**: No validation that packages work independently
- **Cross-Package Dependencies**: Unclear dependency relationships between packages

#### 3. Documentation Inconsistencies
- **Varied Formats**: README files across packages have different structures
- **Installation Variations**: Different installation instructions between packages
- **Missing Guides**: No unified development or deployment documentation
- **Outdated Information**: Documentation may reference obsolete patterns

#### 4. Development Experience Issues
- **Setup Complexity**: Unclear how to set up development environment
- **Contribution Guidelines**: No clear guidelines for contributing to modular architecture
- **Package Development**: Unclear how to develop and test individual packages
- **Release Process**: No defined process for releasing package updates

## Objectives

### Primary Goals
1. **Unified Architecture Documentation**: Clear specification of preferred architecture patterns
2. **Integration Test Suite**: Comprehensive tests validating modular architecture
3. **Consistent Package Documentation**: Standardized documentation across all packages
4. **Development Guidelines**: Clear guidelines for development, testing, and deployment
5. **Migration Strategy**: Clear path for transitioning from monolithic to modular

### Secondary Goals
1. **Automated Validation**: CI/CD checks ensuring architectural consistency
2. **Package Templates**: Templates for creating new packages
3. **Dependency Management**: Clear dependency relationship documentation
4. **Performance Benchmarks**: Performance comparison between architectures

## Technical Requirements

### 1. Repository Structure Documentation

#### Preferred Architecture Specification
```markdown
# MoRAG Architecture Guide

## Preferred Structure: Modular Packages

### Package Organization
- `packages/morag-core/` - Core functionality and shared utilities
- `packages/morag-services/` - AI services and vector storage
- `packages/morag-web/` - Web content processing
- `packages/morag-youtube/` - YouTube processing
- `packages/morag-audio/` - Audio processing
- `packages/morag-video/` - Video processing
- `packages/morag-document/` - Document processing
- `packages/morag-image/` - Image processing

### Import Patterns
```python
# Preferred: Package imports
from morag_core import Settings
from morag_services import VectorStore
from morag_web import WebProcessor

# Deprecated: Monolithic imports
from morag.core.config import Settings  # Don't use
```

### Development Workflow
1. Develop features in appropriate packages
2. Use package imports in all new code
3. Test packages independently
4. Integration test cross-package functionality
```

#### Package Dependency Map
```yaml
# Package Dependencies
morag-core:
  dependencies: []
  description: "Core utilities, no external package dependencies"

morag-services:
  dependencies: [morag-core]
  description: "AI services, vector storage, embeddings"

morag-web:
  dependencies: [morag-core, morag-services]
  description: "Web scraping and content processing"

morag-youtube:
  dependencies: [morag-core, morag-services, morag-audio, morag-video]
  description: "YouTube video processing"

morag-audio:
  dependencies: [morag-core, morag-services]
  description: "Audio processing and transcription"

morag-video:
  dependencies: [morag-core, morag-services, morag-audio]
  description: "Video processing and analysis"

morag-document:
  dependencies: [morag-core, morag-services]
  description: "Document parsing and processing"

morag-image:
  dependencies: [morag-core, morag-services]
  description: "Image processing and OCR"
```

### 2. Integration Test Suite

#### Package Independence Tests
```python
# tests/integration/test_package_independence.py
def test_package_imports_independently():
    """Test that each package can be imported without others."""
    packages = [
        'morag_core',
        'morag_services', 
        'morag_web',
        'morag_youtube',
        'morag_audio',
        'morag_video',
        'morag_document',
        'morag_image'
    ]
    
    for package in packages:
        # Test in isolated environment
        assert_package_imports_cleanly(package)

def test_package_functionality():
    """Test core functionality of each package."""
    # Test each package's main functionality works
    pass

def test_cross_package_integration():
    """Test packages work together correctly."""
    # Test realistic workflows using multiple packages
    pass
```

#### Architecture Validation Tests
```python
# tests/integration/test_architecture_compliance.py
def test_no_monolithic_imports():
    """Ensure no code uses deprecated monolithic imports."""
    forbidden_patterns = [
        'from morag.core',
        'from morag.services',
        'from morag.processors',
        'from morag.converters'
    ]
    
    for pattern in forbidden_patterns:
        assert_no_imports_match_pattern(pattern)

def test_dependency_compliance():
    """Ensure packages only import allowed dependencies."""
    dependency_map = load_dependency_map()
    
    for package, allowed_deps in dependency_map.items():
        actual_deps = get_package_dependencies(package)
        assert actual_deps.issubset(allowed_deps)
```

### 3. Standardized Package Documentation

#### Package README Template
```markdown
# Package Name

## Overview
Brief description of package functionality and purpose.

## Installation
```bash
pip install package-name
```

## Quick Start
```python
from package_name import MainClass

processor = MainClass()
result = processor.process(input_data)
```

## API Reference
Detailed API documentation with examples.

## Configuration
Configuration options and environment variables.

## Testing
How to run tests for this package.

## Contributing
Guidelines for contributing to this package.

## Dependencies
List of dependencies and their purposes.
```

#### Documentation Standards
- **Consistent Structure**: All packages follow same README format
- **API Documentation**: Comprehensive API docs with examples
- **Configuration Guide**: Clear configuration instructions
- **Testing Instructions**: How to test package independently
- **Contribution Guidelines**: How to contribute to specific package

### 4. Development Guidelines

#### Development Workflow Documentation
```markdown
# MoRAG Development Guide

## Setting Up Development Environment
1. Clone repository
2. Install development dependencies
3. Set up pre-commit hooks
4. Configure IDE for package development

## Package Development
1. Choose appropriate package for feature
2. Follow package-specific development patterns
3. Write tests for new functionality
4. Update package documentation

## Testing Strategy
1. Unit tests within packages
2. Integration tests across packages
3. End-to-end tests for complete workflows
4. Performance tests for critical paths

## Release Process
1. Package versioning strategy
2. Dependency update process
3. Release coordination across packages
4. Documentation updates
```

#### Code Quality Standards
- **Import Standards**: Use package imports only
- **Testing Requirements**: >95% coverage per package
- **Documentation Requirements**: All public APIs documented
- **Performance Standards**: No regression in performance

## Implementation Plan

### Phase 1: Documentation and Standards (Day 1)
1. **Create Architecture Documentation**
   - Repository structure guide
   - Package dependency map
   - Import pattern standards
   - Development workflow documentation

2. **Standardize Package Documentation**
   - Create README template
   - Update all package READMEs
   - Ensure consistent formatting
   - Add missing documentation sections

### Phase 2: Integration Testing (Day 2)
1. **Package Independence Tests**
   - Test each package imports independently
   - Validate core functionality works
   - Check for circular dependencies
   - Verify package isolation

2. **Architecture Compliance Tests**
   - Scan for deprecated import patterns
   - Validate dependency compliance
   - Check for architectural violations
   - Test cross-package integration

### Phase 3: Automation and Validation (Day 3)
1. **CI/CD Integration**
   - Add architecture validation to CI
   - Automated documentation checks
   - Package independence validation
   - Performance regression tests

2. **Development Tools**
   - Pre-commit hooks for import validation
   - Package development templates
   - Automated dependency checking
   - Documentation generation tools

## Deliverables

### 1. Documentation
- **File**: `docs/architecture-guide.md`
- **File**: `docs/development-guide.md`
- **File**: `docs/package-development.md`
- **File**: `docs/migration-guide.md`
- **File**: `PACKAGE_README_TEMPLATE.md`

### 2. Integration Tests
- **File**: `tests/integration/test_package_independence.py`
- **File**: `tests/integration/test_architecture_compliance.py`
- **File**: `tests/integration/test_cross_package_integration.py`
- **File**: `tests/integration/test_performance_benchmarks.py`

### 3. Configuration Files
- **File**: `.pre-commit-config.yaml` (updated)
- **File**: `pyproject.toml` (package configuration)
- **File**: `package-dependencies.yaml`
- **File**: `.github/workflows/architecture-validation.yml`

### 4. Development Tools
- **File**: `scripts/validate-architecture.py`
- **File**: `scripts/check-imports.py`
- **File**: `scripts/generate-dependency-map.py`
- **File**: `templates/new-package-template/`

## Testing Strategy

### Architecture Validation
- Import pattern compliance
- Dependency relationship validation
- Package independence verification
- Cross-package integration testing

### Documentation Quality
- README consistency checks
- API documentation completeness
- Example code validation
- Link verification

### Performance Impact
- Package import performance
- Memory usage comparison
- Processing speed benchmarks
- Resource utilization monitoring

## Success Criteria

### Functional Requirements
- ✅ All packages can be imported independently
- ✅ No deprecated import patterns in codebase
- ✅ All packages follow consistent documentation format
- ✅ Integration tests validate modular architecture
- ✅ Development guidelines are clear and complete

### Quality Requirements
- ✅ >95% test coverage maintained across packages
- ✅ No performance regression from modular architecture
- ✅ Documentation is comprehensive and up-to-date
- ✅ CI/CD validates architectural compliance
- ✅ Developer onboarding time reduced by 50%

### Maintainability Requirements
- ✅ Clear separation of concerns between packages
- ✅ Minimal coupling between packages
- ✅ Easy to add new packages following established patterns
- ✅ Automated validation prevents architectural drift

## Future Enhancements

### Advanced Features
- **Package Versioning**: Independent versioning for packages
- **Dependency Management**: Automated dependency updates
- **Performance Monitoring**: Continuous performance tracking
- **Package Metrics**: Usage and performance metrics per package

### Tooling Improvements
- **IDE Integration**: Better IDE support for package development
- **Automated Refactoring**: Tools for migrating to modular patterns
- **Package Generator**: Automated package creation tools
- **Dependency Visualization**: Visual dependency relationship maps

## Migration Strategy

### For Existing Code
1. **Identify Legacy Imports**: Scan for deprecated import patterns
2. **Update Import Statements**: Replace with package imports
3. **Test Functionality**: Ensure no breaking changes
4. **Update Documentation**: Reflect new import patterns

### For New Development
1. **Use Package Imports**: Always use modular package imports
2. **Follow Guidelines**: Adhere to established development patterns
3. **Test Independence**: Ensure packages work independently
4. **Document Changes**: Update relevant documentation

## Notes
- Coordinate with Task 36 (cleanup) to avoid conflicts
- Ensure backward compatibility during transition
- Consider impact on existing deployments
- Plan for gradual migration of legacy code
- Monitor performance impact of modular architecture

---

**Dependencies**: Task 36 (Complete Cleanup and Migration)  
**Blocks**: None  
**Related**: Task 23 (LLM Provider Abstraction), Task 19 (n8n Workflows)
