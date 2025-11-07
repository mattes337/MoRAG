"""Optional dependency management for MoRAG."""

import importlib
import warnings
from typing import Any, Dict, List, Optional

from .utils.logging import get_logger

logger = get_logger(__name__)


class OptionalDependency:
    """Represents an optional dependency with fallback behavior."""

    def __init__(
        self,
        name: str,
        import_path: str,
        feature_name: str,
        fallback_message: Optional[str] = None,
        install_command: Optional[str] = None,
    ):
        self.name = name
        self.import_path = import_path
        self.feature_name = feature_name
        self.fallback_message = fallback_message or f"{feature_name} disabled"
        self.install_command = install_command or f"pip install {name}"
        self._module = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if the dependency is available."""
        if self._available is None:
            try:
                self._module = importlib.import_module(self.import_path)
                self._available = True
                logger.debug(f"{self.feature_name} is available")
            except ImportError:
                self._available = False
                logger.warning(
                    f"{self.fallback_message}",
                    feature=self.feature_name,
                    install_command=self.install_command,
                )
        return self._available

    @property
    def module(self) -> Any:
        """Get the imported module if available."""
        if self.available:
            return self._module
        return None

    def require(self) -> Any:
        """Require the dependency, raising ImportError if not available."""
        if not self.available:
            raise ImportError(
                f"{self.feature_name} requires {self.name}. "
                f"Install with: {self.install_command}"
            )
        return self.module


class OptionalDependencyManager:
    """Manages optional dependencies for MoRAG."""

    def __init__(self):
        self.dependencies: Dict[str, OptionalDependency] = {}
        self._register_dependencies()

    def _register_dependencies(self):
        """Register all optional dependencies."""

        # Web scraping dependencies
        self.dependencies["playwright"] = OptionalDependency(
            name="playwright",
            import_path="playwright",
            feature_name="Dynamic content extraction",
            install_command="pip install playwright && playwright install",
        )

        self.dependencies["trafilatura"] = OptionalDependency(
            name="trafilatura",
            import_path="trafilatura",
            feature_name="Advanced content extraction",
            install_command="pip install trafilatura",
        )

        self.dependencies["readability"] = OptionalDependency(
            name="readability-lxml",
            import_path="readability",
            feature_name="Content cleaning",
            install_command="pip install readability-lxml",
        )

        self.dependencies["newspaper3k"] = OptionalDependency(
            name="newspaper3k",
            import_path="newspaper",
            feature_name="Article extraction",
            install_command="pip install newspaper3k",
        )

        # Audio processing dependencies
        self.dependencies["pyannote"] = OptionalDependency(
            name="pyannote.audio",
            import_path="pyannote.audio",
            feature_name="Speaker diarization",
            install_command="pip install pyannote.audio",
        )

        self.dependencies["faster_whisper"] = OptionalDependency(
            name="faster-whisper",
            import_path="faster_whisper",
            feature_name="Fast speech recognition",
            install_command="pip install faster-whisper",
        )

        # Video processing dependencies
        self.dependencies["opencv"] = OptionalDependency(
            name="opencv-python",
            import_path="cv2",
            feature_name="Video processing",
            install_command="pip install opencv-python",
        )

        self.dependencies["moviepy"] = OptionalDependency(
            name="moviepy",
            import_path="moviepy",
            feature_name="Video editing",
            install_command="pip install moviepy",
        )

        # Image processing dependencies
        self.dependencies["pytesseract"] = OptionalDependency(
            name="pytesseract",
            import_path="pytesseract",
            feature_name="OCR processing",
            install_command="pip install pytesseract",
        )

        self.dependencies["easyocr"] = OptionalDependency(
            name="easyocr",
            import_path="easyocr",
            feature_name="Easy OCR",
            install_command="pip install easyocr",
        )

        # Document processing dependencies
        self.dependencies["docling"] = OptionalDependency(
            name="docling",
            import_path="docling",
            feature_name="Advanced PDF processing",
            install_command="pip install docling",
        )

        # Graph processing dependencies
        self.dependencies["morag_graph"] = OptionalDependency(
            name="morag-graph",
            import_path="morag_graph",
            feature_name="Graph processing",
            install_command="pip install -e packages/morag-graph",
        )

        # NLP dependencies
        self.dependencies["spacy"] = OptionalDependency(
            name="spacy",
            import_path="spacy",
            feature_name="NLP processing",
            install_command="pip install spacy",
        )

        self.dependencies["langdetect"] = OptionalDependency(
            name="langdetect",
            import_path="langdetect",
            feature_name="Language detection",
            install_command="pip install langdetect",
        )

        # ML/Scientific computing dependencies (heavy - should be optional)
        self.dependencies["torch"] = OptionalDependency(
            name="torch",
            import_path="torch",
            feature_name="PyTorch tensor operations",
            install_command="pip install torch>=2.1.0,<2.7.0",
        )

        self.dependencies["torchaudio"] = OptionalDependency(
            name="torchaudio",
            import_path="torchaudio",
            feature_name="PyTorch audio processing",
            install_command="pip install torchaudio>=2.1.0,<2.7.0",
        )

        self.dependencies["scipy"] = OptionalDependency(
            name="scipy",
            import_path="scipy",
            feature_name="Scientific computing",
            install_command="pip install scipy>=1.13.0,<1.15.0",
        )

        self.dependencies["scikit_learn"] = OptionalDependency(
            name="scikit-learn",
            import_path="sklearn",
            feature_name="Machine learning algorithms",
            install_command="pip install scikit-learn>=1.5.0,<1.6.0",
        )

        self.dependencies["sentence_transformers"] = OptionalDependency(
            name="sentence-transformers",
            import_path="sentence_transformers",
            feature_name="Sentence embeddings",
            install_command="pip install sentence-transformers>=3.0.0,<5.0.0",
        )

    def get_dependency(self, name: str) -> OptionalDependency:
        """Get a dependency by name."""
        if name not in self.dependencies:
            raise ValueError(f"Unknown dependency: {name}")
        return self.dependencies[name]

    def is_available(self, name: str) -> bool:
        """Check if a dependency is available."""
        return self.get_dependency(name).available

    def get_module(self, name: str) -> Any:
        """Get a module if available, None otherwise."""
        return self.get_dependency(name).module

    def require(self, name: str) -> Any:
        """Require a dependency, raising ImportError if not available."""
        return self.get_dependency(name).require()

    def get_available_features(self) -> List[str]:
        """Get list of available features."""
        return [dep.feature_name for dep in self.dependencies.values() if dep.available]

    def get_missing_features(self) -> List[Dict[str, str]]:
        """Get list of missing features with install instructions."""
        return [
            {
                "feature": dep.feature_name,
                "package": dep.name,
                "install_command": dep.install_command,
            }
            for dep in self.dependencies.values()
            if not dep.available
        ]

    def print_status(self):
        """Print status of all dependencies."""
        print("\n" + "=" * 60)
        print("  MoRAG Optional Dependencies Status")
        print("=" * 60)

        available = self.get_available_features()
        missing = self.get_missing_features()

        if available:
            print(f"\n✅ Available Features ({len(available)}):")
            for feature in available:
                print(f"   • {feature}")

        if missing:
            print(f"\n❌ Missing Features ({len(missing)}):")
            for item in missing:
                print(f"   • {item['feature']}")
                print(f"     Install: {item['install_command']}")

        print(f"\nTotal: {len(available)} available, {len(missing)} missing")
        print("=" * 60)


# Global instance
optional_deps = OptionalDependencyManager()


def check_optional_dependency(name: str) -> bool:
    """Check if an optional dependency is available."""
    return optional_deps.is_available(name)


def get_optional_module(name: str) -> Any:
    """Get an optional module if available."""
    return optional_deps.get_module(name)


def require_optional_dependency(name: str) -> Any:
    """Require an optional dependency."""
    return optional_deps.require(name)


def safe_import(module_name: str, feature_name: str = None) -> Any:
    """Safely import a module with graceful fallback."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        feature = feature_name or module_name
        logger.warning(f"{feature} not available, feature disabled")
        return None
