"""Processing utilities for MoRAG."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class ContentType(Enum):
    """Supported content types."""
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    IMAGE = "image"
    WEB = "web"
    YOUTUBE = "youtube"


class ProcessingMode(Enum):
    """Processing modes."""
    PROCESSING = "processing"  # Immediate results only
    INGESTION = "ingestion"    # Background processing + storage


@dataclass
class ProcessingMetadata:
    """Metadata for processing operations."""
    timestamp: str
    processing_time: float
    content_type: str
    mode: str
    source_path: str
    source_size: int
    model_info: Dict[str, Any]
    options: Dict[str, Any]


def create_processing_metadata(
    content_type: ContentType,
    source_path: str,
    processing_time: float,
    mode: ProcessingMode,
    model_info: Dict[str, Any],
    options: Dict[str, Any]
) -> ProcessingMetadata:
    """Create processing metadata."""
    source_path_obj = Path(source_path)
    return ProcessingMetadata(
        timestamp=datetime.now().isoformat(),
        processing_time=processing_time,
        content_type=content_type.value,
        mode=mode.value,
        source_path=str(source_path),
        source_size=source_path_obj.stat().st_size if source_path_obj.exists() else 0,
        model_info=model_info,
        options=options
    )


def get_output_paths(input_path: Union[str, Path], mode: ProcessingMode) -> Dict[str, Path]:
    """Get standardized output file paths."""
    input_path = Path(input_path)
    stem = input_path.stem
    parent = input_path.parent
    
    if mode == ProcessingMode.PROCESSING:
        return {
            'intermediate_json': parent / f"{stem}_intermediate.json",
            'result_json': parent / f"{stem}_processing_result.json"
        }
    else:  # INGESTION
        return {
            'intermediate_json': parent / f"{stem}_intermediate.json",
            'intermediate_md': parent / f"{stem}_intermediate.md",
            'result_json': parent / f"{stem}_ingestion_result.json"
        }