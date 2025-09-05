# MoRAG Refactoring Plan

## Overview
This directory contains a comprehensive refactoring plan for the MoRAG codebase to achieve the goals of reduced file sizes (<1000 lines each, targeting 500), eliminated code duplication, removed unnecessary dependencies, and comprehensive unit test coverage.

## Analysis Summary

### Current State
- **Codebase Size**: 574 Python files, ~118,000 lines of code
- **File Size Issues**: 9 files exceed 1000 lines (largest: 2958 lines)
- **Test Coverage**: 20.6% (66 test files for 321 production files)
- **Dependencies**: 50+ packages including heavy ML libraries (2.5GB installation)
- **Duplication**: Significant patterns in Storage (27 files), Processor (34 files), and Service classes

### Refactoring Goals
1. **File Size Compliance**: All files ≤1000 lines, targeting ≤500 lines
2. **Zero Duplication**: Eliminate duplicate code patterns through base classes
3. **Minimal Dependencies**: Core installation <300MB, optional heavy dependencies
4. **Comprehensive Testing**: >80% test coverage with fast, reliable unit tests

## Task Execution Order

### Phase 1: Quick Wins (2-4 hours) - START HERE
**File**: `01-quick-fixes.md`
**Priority**: HIGH
**Dependencies**: None

Quick improvements that can be done immediately:
- Standardize on httpx (remove requests)
- Make heavy ML dependencies optional
- Standardize logger initialization
- Split top 3 largest files

### Phase 2: File Splitting (8-12 hours)
**File**: `02-file-splitting.md`
**Priority**: HIGH  
**Dependencies**: Quick fixes completed

Systematic splitting of oversized files:
- `ingestion_coordinator.py` (2958 lines) → 4 files
- `markdown_conversion.py` (1627 lines) → 3 files
- All files >1000 lines split to <500 lines each

### Phase 3: Code Deduplication (6-10 hours)
**File**: `03-deduplication.md`
**Priority**: MEDIUM
**Dependencies**: File splitting completed

Create unified base classes:
- BaseStorage for 27 storage classes
- BaseProcessor for 34 processor classes  
- BaseService for service classes
- Eliminate ~2,300 lines of duplicate code

### Phase 4: Dependency Optimization (4-6 hours)
**File**: `04-dependency-cleanup.md`
**Priority**: MEDIUM-HIGH
**Dependencies**: Deduplication completed

Optimize dependencies:
- Core installation: ~30 packages (200MB)
- Optional extras: audio, video, web, scientific
- Graceful degradation for missing dependencies

### Phase 5: Testing Strategy (10-15 hours)
**File**: `05-testing-strategy.md`
**Priority**: HIGH
**Dependencies**: All previous tasks for comprehensive testing

Achieve comprehensive test coverage:
- 300+ unit tests (80% coverage)
- 50+ integration tests
- 5-10 E2E tests
- Performance and load testing

## Expected Results

### Quantitative Improvements
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max File Size | 2958 lines | <500 lines | 83% reduction |
| Installation Size | 2.5GB | 200MB (core) | 92% reduction |
| Test Coverage | 20.6% | >80% | 4x increase |
| Code Duplication | ~2,300 lines | 0 lines | 100% elimination |
| Import Time | 3-5 seconds | <1 second | 80% faster |

### Qualitative Improvements
- **Maintainability**: Single responsibility files, clear interfaces
- **Reliability**: Comprehensive test coverage with fast feedback
- **Performance**: Faster imports, optional heavy dependencies
- **Developer Experience**: Easy testing, clear error messages
- **Deployment**: Lighter Docker images, flexible installation options

## Getting Started

### Prerequisites
```bash
# Ensure clean working directory
git status  # Should show clean working directory
git checkout -b refactor/initial-analysis

# Run baseline tests
python check_syntax.py --verbose
pytest tests/ -v

# Create baseline metrics
python scripts/analyze_dependencies.py > baseline_metrics.txt
```

### Execution Checklist

- [ ] **Phase 1: Quick Fixes** (`01-quick-fixes.md`)
  - [ ] Standardize HTTP client to httpx
  - [ ] Make ML dependencies optional
  - [ ] Standardize logger initialization  
  - [ ] Split top 3 largest files
  - [ ] All tests pass

- [ ] **Phase 2: File Splitting** (`02-file-splitting.md`)
  - [ ] Split all files >1000 lines
  - [ ] Target <500 lines per file
  - [ ] Maintain clear interfaces
  - [ ] Update imports and tests
  - [ ] No functionality regression

- [ ] **Phase 3: Deduplication** (`03-deduplication.md`)
  - [ ] Create base storage class
  - [ ] Create base processor class
  - [ ] Create base service class
  - [ ] Migrate all implementations
  - [ ] Remove duplicate patterns

- [ ] **Phase 4: Dependencies** (`04-dependency-cleanup.md`)
  - [ ] Create optional dependency groups
  - [ ] Update installation documentation
  - [ ] Test graceful degradation
  - [ ] Verify core installation size

- [ ] **Phase 5: Testing** (`05-testing-strategy.md`)
  - [ ] Create comprehensive unit tests
  - [ ] Add integration tests  
  - [ ] Add performance tests
  - [ ] Achieve >80% coverage
  - [ ] Fast test execution (<5 minutes)

## Safety and Rollback

### Branch Strategy
```bash
# Each phase gets its own branch:
git checkout -b refactor/quick-fixes
git checkout -b refactor/file-splitting
git checkout -b refactor/deduplication
git checkout -b refactor/dependency-cleanup
git checkout -b refactor/testing-strategy
```

### Validation at Each Phase
```bash
# After each major change:
python check_syntax.py --verbose           # No syntax errors
pytest tests/ -v                          # All tests pass  
python tests/cli/test-simple.py          # Basic functionality works
python debug_morag.py                     # System health check
```

### Rollback Procedure
```bash
# If issues occur:
git checkout main
git branch -D refactor/<failed-phase>
# Review issue, adjust approach, retry
```

## Success Metrics

### Immediate Validation
After each phase, run:
```bash
# File size check
find packages/ -name "*.py" -exec wc -l {} + | awk '$1 > 1000 {print $1, $2}'

# Dependency analysis  
pip list | wc -l  # Should decrease over phases

# Test coverage
pytest --cov=packages/ --cov-report=term-missing

# Performance
time python -c "import morag"  # Should be <1 second after optimization
```

### Final Success Criteria
- [ ] No Python files >1000 lines
- [ ] >80% test coverage with >300 unit tests
- [ ] Core installation <300MB 
- [ ] All functionality preserved
- [ ] Import time <1 second
- [ ] Documentation updated

## Timeline Estimate

### Conservative Timeline (40-50 hours)
- **Week 1**: Quick fixes + File splitting (12-16 hours)
- **Week 2**: Deduplication + Dependencies (10-16 hours) 
- **Week 3**: Testing strategy (10-15 hours)
- **Week 4**: Documentation + polish (5-8 hours)

### Aggressive Timeline (30-35 hours)
- Focus on critical issues first
- Parallel testing development
- Streamlined file splitting approach

## Communication and Documentation

### Progress Tracking
Each task file includes:
- Detailed implementation steps
- Success criteria and validation
- Risk mitigation strategies
- Time estimates and dependencies

### Documentation Updates Required
- Update CLAUDE.md with new architecture
- Update README.md with installation options
- Update API documentation for moved classes
- Create migration guide for users

## Post-Refactoring Benefits

### Development Velocity
- Faster test feedback loops
- Easier to locate and modify code
- Reduced cognitive load with smaller files
- Clear separation of concerns

### Operations Benefits  
- Lighter Docker images
- Faster deployments
- Optional feature installation
- Better error messages and debugging

### Maintenance Benefits
- Consistent patterns across codebase
- Comprehensive test coverage
- Clear interfaces and dependencies
- Documented architectural decisions

---

**Start with `01-quick-fixes.md` for immediate improvements with minimal risk.**