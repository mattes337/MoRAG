# MoRAG Refactoring Task 4: Dependency Cleanup & Optimization

## Priority: MEDIUM-HIGH
## Estimated Time: 4-6 hours
## Impact: Reduced installation size, faster imports, optional heavy dependencies

## Overview
This task performs a comprehensive cleanup of dependencies to reduce the codebase footprint, make heavy ML dependencies optional, and improve installation/startup performance. Current requirements.txt is 195 lines with many heavy dependencies that are only used in specific scenarios.

## Current Dependency Analysis

### Heavy Dependencies with Limited Usage
Based on code analysis, several heavy dependencies have minimal usage:

```python
# Heavy ML dependencies (limited usage):
torch>=2.1.0,<2.7.0                    # 25 occurrences, 9 files (audio only)
torchaudio>=2.1.0,<2.7.0               # Audio processing only
pyannote.audio>=3.3.0,<4.0.0           # Audio ML features
sentence-transformers>=3.0.0,<5.0.0    # Limited embedding usage
pytorch-lightning>=2.0.0,<2.5.0        # PyAnnote dependency only
speechbrain>=0.5.0,<1.0.0              # Advanced audio features

# Scientific computing (light usage):
numpy>=2.1.0,<2.2.0                    # 12 occurrences, 11 files
scipy>=1.13.0,<1.15.0                  # Listed but minimal usage found
scikit-learn>=1.5.0,<1.6.0             # Listed but minimal usage found

# Redundant HTTP clients:
requests>=2.32.0                        # 38 files (can be replaced with httpx)
httpx==0.28.1                          # 15 files (preferred client)

# Video processing (specialized usage):
opencv-python>=4.10.0,<4.11.0          # Video processing only
moviepy>=2.1.0,<2.2.0                  # Video processing only
```

### Core Dependencies (Keep)
```python
# Essential for core functionality:
fastapi==0.115.4                       # API framework
uvicorn==0.32.1                         # ASGI server
pydantic>=2.10.0,<2.11.2               # Data validation
pydantic-ai>=0.3.4,<1.0.0              # AI integration
celery==5.3.6                          # Task queue
redis==5.2.1                           # Celery backend
qdrant-client==1.12.1                  # Vector database
google-genai>=1.15.0,<1.18.0           # Primary AI service
structlog==24.4.0                      # Logging
```

## Optimization Strategy

### 1. Create Optional Dependency Groups
Reorganize requirements.txt into optional extras for specialized features:

```ini
# requirements.txt (core dependencies only - ~30 packages)
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "morag"
dependencies = [
    # Core API and Processing
    "fastapi>=0.115.4,<0.116.0",
    "uvicorn>=0.32.1,<0.33.0",
    "pydantic>=2.10.0,<2.11.2",
    "pydantic-ai>=0.3.4,<1.0.0",

    # Task Processing
    "celery>=5.3.6,<5.4.0",
    "redis>=5.2.1,<5.3.0",

    # Storage and AI
    "qdrant-client>=1.12.1,<1.13.0",
    "google-genai>=1.15.0,<1.18.0",
    "httpx>=0.28.1,<0.29.0",

    # Text Processing
    "markitdown>=0.1.2",
    "beautifulsoup4>=4.13.0,<5.0.0",
    "python-dotenv>=1.0.1,<1.1.0",
    "structlog>=24.4.0,<24.5.0",

    # Basic Document Processing
    "pypdf>=5.6.0,<6.0.0",
    "python-docx>=1.1.2",
    "openpyxl>=3.1.5",
]

[project.optional-dependencies]
# Audio processing with ML
audio-ml = [
    "torch>=2.1.0,<2.7.0",
    "torchaudio>=2.1.0,<2.7.0",
    "pyannote.audio>=3.3.0,<4.0.0",
    "faster-whisper>=1.1.0",
    "pydub>=0.25.1",
    "ffmpeg-python>=0.2.0",
]

# Basic audio without heavy ML
audio = [
    "faster-whisper>=1.1.0",
    "pydub>=0.25.1",
    "ffmpeg-python>=0.2.0",
    "soundfile>=0.13.0",
]

# Video processing
video = [
    "opencv-python>=4.10.0,<4.11.0",
    "moviepy>=2.1.0,<2.2.0",
    "imageio>=2.35.0,<2.37.0",
]

# Web scraping
web = [
    "playwright>=1.40.0,<2.0.0",
    "trafilatura>=1.6.0,<2.0.0",
]

# Scientific computing
scientific = [
    "numpy>=2.1.0,<2.2.0",
    "scipy>=1.13.0,<1.15.0",
    "scikit-learn>=1.5.0,<1.6.0",
]

# Advanced ML features
ml-advanced = [
    "sentence-transformers>=3.0.0,<5.0.0",
    "pytorch-lightning>=2.0.0,<2.5.0",
    "speechbrain>=0.5.0,<1.0.0",
]

# Development dependencies
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=1.0.0",
    "black>=25.1.0",
    "isort>=6.0.0",
    "mypy>=1.16.0",
]

# All optional dependencies
all = [
    "morag[audio-ml,video,web,scientific,ml-advanced]"
]
```

### 2. Update Package Structure for Optional Dependencies
Create graceful handling for missing optional dependencies:

```python
# packages/morag-core/src/morag_core/optional_deps.py

"""
Centralized optional dependency management.
Provides graceful degradation when optional packages are missing.
"""

import importlib
from typing import Any, Optional, Dict
import warnings

class OptionalDependency:
    """Manages optional dependencies with graceful fallbacks."""

    def __init__(self, package_name: str, feature_name: str = None):
        self.package_name = package_name
        self.feature_name = feature_name or package_name
        self._module = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                self._module = importlib.import_module(self.package_name)
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    @property
    def module(self) -> Any:
        if not self.available:
            raise ImportError(
                f"Optional dependency '{self.package_name}' not available. "
                f"Install with: pip install 'morag[{self.feature_name}]'"
            )
        return self._module

# Define optional dependencies
TORCH = OptionalDependency("torch", "audio-ml")
OPENCV = OptionalDependency("cv2", "video")
NUMPY = OptionalDependency("numpy", "scientific")
PLAYWRIGHT = OptionalDependency("playwright", "web")

# Convenience functions
def require_torch():
    """Ensure torch is available for audio ML features."""
    return TORCH.module

def require_opencv():
    """Ensure OpenCV is available for video processing."""
    return OPENCV.module

def require_numpy():
    """Ensure numpy is available for scientific computing."""
    return NUMPY.module

def get_available_features() -> Dict[str, bool]:
    """Return dictionary of available optional features."""
    return {
        "audio_ml": TORCH.available,
        "video": OPENCV.available,
        "scientific": NUMPY.available,
        "web": PLAYWRIGHT.available,
    }
```

### 3. Update Components to Handle Optional Dependencies

#### Audio Processing Example:
```python
# packages/morag-audio/src/morag_audio/services/speaker_diarization.py

from morag_core.optional_deps import TORCH, require_torch
import warnings

class SpeakerDiarizationService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ml_available = TORCH.available

        if not self.ml_available:
            warnings.warn(
                "Speaker diarization requires torch. "
                "Install with: pip install 'morag[audio-ml]' "
                "Falling back to basic timestamp segmentation.",
                UserWarning
            )

    async def diarize_speakers(self, audio_path: Path) -> List[Dict]:
        if self.ml_available:
            return await self._ml_diarization(audio_path)
        else:
            return await self._basic_segmentation(audio_path)

    async def _ml_diarization(self, audio_path: Path) -> List[Dict]:
        """Advanced ML-based speaker diarization."""
        torch = require_torch()
        # Use torch-based processing

    async def _basic_segmentation(self, audio_path: Path) -> List[Dict]:
        """Fallback: basic timestamp-based segmentation."""
        # Simple time-based segmentation without ML
```

#### Video Processing Example:
```python
# packages/morag-video/src/morag_video/processor.py

from morag_core.optional_deps import OPENCV, require_opencv

class VideoProcessor:
    def __init__(self):
        self.opencv_available = OPENCV.available

    async def extract_frames(self, video_path: Path) -> List[Path]:
        if self.opencv_available:
            return await self._opencv_extraction(video_path)
        else:
            # Fallback to ffmpeg-based extraction
            return await self._ffmpeg_extraction(video_path)
```

## Implementation Plan

### Phase 1: Dependency Analysis and Requirements Restructure (2 hours)

1. **Audit current usage** (45 minutes):
   ```bash
   # Create dependency usage report
   python scripts/analyze_dependencies.py > dependency_usage_report.txt

   # Find unused imports
   python scripts/check_imports.py --unused-only
   ```

2. **Create new requirements structure** (45 minutes):
   - Split requirements.txt into core + optional extras
   - Test core installation works: `pip install -e .`
   - Test optional extras: `pip install -e '.[audio,video,web]'`

3. **Update setup.py/pyproject.toml** (30 minutes):
   - Define optional dependency groups
   - Update package metadata

### Phase 2: Optional Dependency Infrastructure (2 hours)

1. **Enhance optional_deps.py** (1 hour):
   - Add all optional dependencies
   - Create require_* functions
   - Add feature detection

2. **Update package imports** (1 hour):
   - Modify components to use optional dependencies
   - Add graceful fallbacks
   - Update error messages

### Phase 3: Component Updates (2 hours)

1. **Audio components** (45 minutes):
   - Update to handle missing torch/ML libraries
   - Implement fallback processing methods

2. **Video components** (45 minutes):
   - Handle missing OpenCV gracefully
   - Fallback to ffmpeg for basic operations

3. **Scientific computing** (30 minutes):
   - Make numpy usage optional where possible
   - Use built-in alternatives for simple operations

## Expected Results

### Installation Size Reduction
```bash
# Current full installation:
pip install -e . # ~2.5GB with all ML dependencies

# After optimization:
pip install -e .                    # ~200MB (core only)
pip install -e '.[audio]'           # ~400MB (basic audio)
pip install -e '.[audio-ml]'        # ~2.2GB (with ML)
pip install -e '.[all]'             # ~2.5GB (everything)
```

### Import Performance
```python
# Before (loads all dependencies):
import morag  # ~3-5 seconds with ML libraries

# After (loads only core):
import morag  # ~0.5-1 seconds
```

### Dependency Reduction Summary
- **Core requirements**: 30 packages (was 50+)
- **Optional audio-ml**: 15 packages
- **Optional video**: 8 packages
- **Optional web**: 5 packages
- **Optional scientific**: 8 packages
- **Dev dependencies**: 12 packages

## Testing Strategy

### Installation Testing
```bash
# Test each optional dependency group:
python -m venv test-core && source test-core/bin/activate
pip install -e .
python -c "import morag; print('Core works')"

python -m venv test-audio && source test-audio/bin/activate
pip install -e '.[audio]'
python tests/cli/test-audio.py sample.mp3

python -m venv test-video && source test-video/bin/activate
pip install -e '.[video]'
python tests/cli/test-video.py sample.mp4
```

### Functionality Testing
```bash
# Test graceful degradation:
python -c """
from morag_audio.processor import AudioProcessor
processor = AudioProcessor()
# Should work with warnings about missing ML features
"""

# Test full functionality with optional deps:
pip install -e '.[audio-ml]'
python -c """
from morag_audio.services.speaker_diarization import SpeakerDiarizationService
service = SpeakerDiarizationService({})
# Should work with full ML capabilities
"""
```

## Success Criteria

### Quantitative Goals
- [ ] Core installation size <300MB (vs current ~2.5GB)
- [ ] Core import time <1 second (vs current ~3-5 seconds)
- [ ] ≥60% reduction in required dependencies (30 vs 50+)
- [ ] All optional features work when dependencies installed

### Qualitative Goals
- [ ] Clear error messages when optional dependencies missing
- [ ] Graceful feature degradation without crashes
- [ ] Easy installation with `pip install morag[feature]` syntax
- [ ] Consistent behavior across optional dependency scenarios

## Migration Guide for Users

### Installation Options
```bash
# Minimal installation (basic document processing):
pip install morag

# With audio processing (no ML):
pip install 'morag[audio]'

# With advanced audio (ML features):
pip install 'morag[audio-ml]'

# With video processing:
pip install 'morag[video]'

# Web scraping capabilities:
pip install 'morag[web]'

# Scientific computing features:
pip install 'morag[scientific]'

# Everything (current behavior):
pip install 'morag[all]'

# Development setup:
pip install 'morag[dev]'
```

### Feature Detection
```python
# Check available features programmatically:
from morag_core.optional_deps import get_available_features

features = get_available_features()
if features['audio_ml']:
    # Use advanced audio processing
    pass
elif features['audio_basic']:
    # Use basic audio processing
    pass
else:
    print("Audio processing not available")
```

## Risk Mitigation

### Backward Compatibility
- Maintain `requirements-full.txt` for users wanting old behavior
- Provide migration script to update existing installations
- Clear documentation about new installation options

### Testing Matrix
Test all combinations of optional dependencies:
```bash
# Core functionality matrix:
pytest tests/core/ # No optional deps
pytest tests/core/ --optional=audio
pytest tests/core/ --optional=video
pytest tests/core/ --optional=web
pytest tests/core/ --optional=all
```

### Gradual Rollout
1. Phase 1: Core + optional infrastructure (this task)
2. Phase 2: Update documentation and examples
3. Phase 3: Community feedback and refinement

## Documentation Updates Required

1. **Update CLAUDE.md**: New installation instructions
2. **Update README.md**: Installation matrix and optional features
3. **Update LOCAL_DEVELOPMENT.md**: Development setup with optional deps
4. **Create INSTALLATION.md**: Comprehensive installation guide
5. **Update API documentation**: Feature availability indicators

This task significantly reduces the MoRAG footprint while maintaining full functionality when needed, making it more accessible for lightweight deployments and faster development cycles.
