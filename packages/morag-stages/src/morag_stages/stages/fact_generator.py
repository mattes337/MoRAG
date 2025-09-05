"""Fact generator stage implementation.

REFACTORED: This module has been split for better maintainability.
The main functionality is now distributed across:
- fact_generation_stage.py: Stage orchestration and coordination
- fact_extraction_engine.py: Core fact extraction logic and AI processing

This file provides backward compatibility.
"""

# Re-export the split components for backward compatibility
from .fact_generation_stage import FactGeneratorStage
from .fact_extraction_engine import FactExtractionEngine

# Maintain backward compatibility by exporting the main class
__all__ = ["FactGeneratorStage", "FactExtractionEngine"]