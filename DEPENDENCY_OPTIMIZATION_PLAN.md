# MoRAG Dependency Optimization Plan

## Executive Summary

**Current State**: 88 total dependencies (49 core, 39 optional)
**Optimization Goal**: Reduce to ~60 dependencies by removing unused packages and consolidating overlapping functionality

## Analysis Results

### 1. Unused Dependencies (High Priority for Removal)

#### Core Dependencies - Unused (7 packages)
- `bleach==6.2.0` - HTML sanitization (0 usages)
- `html2text==2024.2.26` - HTML to text conversion (0 usages) 
- `kombu==5.3.7` - Celery dependency, likely transitive (0 usages)
- `lxml==5.3.0` - XML/HTML parsing (0 usages in main code)
- `pydantic-settings==2.7.0` - Duplicate with existing config (0 direct usages)
- `python-multipart==0.0.17` - File upload handling (0 usages)

#### Optional Dependencies - Unused (37 packages)
Development tools that should be moved to dev-only:
- `black`, `flake8`, `isort`, `mypy`, `pre-commit` - Code quality tools
- `pytest`, `pytest-asyncio`, `pytest-cov` - Testing tools

Unused ML/processing libraries:
- `deepsearch-glm` - Deep search functionality (0 usages)
- `librosa` - Audio analysis (0 usages)
- `moviepy` - Video processing (0 usages) 
- `newspaper3k` - News article extraction (0 usages)
- `readability-lxml` - Content extraction (0 usages)
- `soundfile` - Audio file I/O (0 usages)
- `speechbrain` - Speech processing (0 usages)
- `transformers` - Transformer models (0 usages)
- `weasel` - spaCy project management (0 usages)
- `xlrd`, `xlwt` - Excel file handling (0 usages)

### 2. Duplicate/Overlapping Dependencies

#### Image Processing
- `Pillow` (requirements.txt) vs `pillow` (package deps) - **Consolidate**
- `opencv-python` appears in multiple packages - **Centralize**

#### HTTP Clients  
- `httpx` (7 usages) vs `aiohttp` (0 usages) - **Remove aiohttp**

#### Text Processing
- `beautifulsoup4` + `lxml` + `html2text` - **Keep only beautifulsoup4**

### 3. Dependencies to Keep (High Value)

#### Core Infrastructure (23 packages)
- `fastapi`, `uvicorn` - Web framework ✅
- `celery`, `redis` - Task queue ✅  
- `qdrant-client` - Vector database ✅
- `pydantic` - Data validation ✅
- `structlog` - Logging ✅
- `google-genai` - LLM integration ✅

#### Document Processing (8 packages)
- `docling` - Enhanced PDF processing ✅
- `pypdf` - PDF fallback ✅
- `python-docx`, `openpyxl`, `python-pptx` - Office formats ✅
- `pytesseract` - OCR ✅

#### Media Processing (6 packages)
- `yt-dlp` - YouTube processing ✅
- `pydub`, `ffmpeg-python` - Audio processing ✅
- `faster-whisper` - Speech recognition ✅

#### Web Processing (3 packages)
- `playwright` - Web scraping ✅
- `trafilatura` - Content extraction ✅
- `markdownify` - HTML to Markdown ✅

## Optimization Recommendations

### Phase 1: Remove Unused Core Dependencies (Save ~7 packages)
```bash
# Remove from requirements.txt
- bleach==6.2.0
- html2text==2024.2.26  
- kombu==5.3.7
- lxml==5.3.0
- python-multipart==0.0.17
```

### Phase 2: Consolidate Overlapping Dependencies (Save ~5 packages)
```bash
# Replace aiohttp with httpx everywhere
# Standardize on Pillow (not pillow)
# Remove redundant text processing libraries
```

### Phase 3: Move Development Tools to dev-only (Save ~10 packages)
```bash
# Move to [project.optional-dependencies.dev]
- black, flake8, isort, mypy, pre-commit
- pytest, pytest-asyncio, pytest-cov
```

### Phase 4: Remove Unused Optional Dependencies (Save ~15 packages)
```bash
# Remove completely unused packages
- deepsearch-glm, librosa, moviepy, newspaper3k
- readability-lxml, soundfile, speechbrain
- transformers, weasel, xlrd, xlwt
```

## Expected Results

**Before**: 88 dependencies (49 core + 39 optional)
**After**: ~60 dependencies (35 core + 25 optional)
**Reduction**: 28 packages (32% decrease)

### Benefits
1. **Faster Installation**: Fewer packages to download and install
2. **Smaller Docker Images**: Reduced image size by ~200-300MB
3. **Fewer Security Vulnerabilities**: Smaller attack surface
4. **Easier Maintenance**: Fewer dependencies to track and update
5. **Clearer Purpose**: Each dependency has a clear, documented use case

## Implementation Priority

### High Priority (Immediate)
1. Remove unused core dependencies
2. Remove completely unused optional dependencies
3. Move dev tools to dev-only section

### Medium Priority (Next Sprint)
1. Consolidate overlapping dependencies
2. Update package imports to use consolidated libraries
3. Test all functionality still works

### Low Priority (Future)
1. Evaluate remaining optional dependencies for actual usage
2. Consider creating more granular optional dependency groups
3. Regular dependency audits (quarterly)

## Risk Assessment

### Low Risk
- Removing unused dependencies (0 usages found)
- Moving dev tools to dev-only section

### Medium Risk  
- Consolidating overlapping dependencies (requires testing)
- Removing transitive dependencies like `kombu`

### Mitigation
- Comprehensive testing after each phase
- Gradual rollout with rollback plan
- Monitor for any missing functionality

## Success Metrics

1. **Dependency Count**: Reduce from 88 to ~60 packages
2. **Docker Image Size**: Reduce by 200-300MB
3. **Installation Time**: Reduce by 20-30%
4. **Functionality**: All existing features continue to work
5. **CI/CD Performance**: Faster build and test times
