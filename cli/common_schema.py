#!/usr/bin/env python3
"""
Common Schema Definitions for MoRAG CLI Scripts

This module defines the standardized intermediate JSON and markdown schemas
used across all CLI scripts for consistent processing and ingestion pipelines.
"""

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
class Entity:
    """Extracted entity."""
    id: str
    name: str
    type: str
    confidence: float
    properties: Dict[str, Any]
    source_span: Optional[Dict[str, int]] = None  # start, end positions in text


@dataclass
class Relation:
    """Extracted relation."""
    id: str
    source_entity_id: str
    target_entity_id: str
    type: str
    confidence: float
    properties: Dict[str, Any]
    source_span: Optional[Dict[str, int]] = None


@dataclass
class ProcessingMetadata:
    """Processing metadata."""
    timestamp: str
    processing_time: float
    content_type: str
    mode: str
    source_path: str
    source_size: int
    model_info: Dict[str, Any]
    options: Dict[str, Any]


@dataclass
class IntermediateJSON:
    """Standardized intermediate JSON schema."""
    # Core content
    content_type: str
    source_path: str
    title: str
    text_content: str
    
    # Processing metadata
    metadata: ProcessingMetadata
    
    # Graph extraction results
    entities: List[Entity]
    relations: List[Relation]
    
    # Content-specific data
    segments: Optional[List[Dict[str, Any]]] = None  # For audio/video
    pages: Optional[List[Dict[str, Any]]] = None     # For documents
    frames: Optional[List[Dict[str, Any]]] = None    # For video/images
    links: Optional[List[Dict[str, Any]]] = None     # For web content
    
    # Additional metadata
    custom_metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, file_path: Union[str, Path], indent: int = 2) -> None:
        """Save to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False, default=str)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'IntermediateJSON':
        """Load from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert nested dictionaries back to dataclasses
        if 'metadata' in data:
            data['metadata'] = ProcessingMetadata(**data['metadata'])
        
        if 'entities' in data:
            data['entities'] = [Entity(**entity) for entity in data['entities']]
        
        if 'relations' in data:
            data['relations'] = [Relation(**relation) for relation in data['relations']]
        
        return cls(**data)


class MarkdownGenerator:
    """Generate standardized markdown from intermediate JSON."""
    
    @staticmethod
    def generate(intermediate: IntermediateJSON) -> str:
        """Generate markdown content from intermediate JSON."""
        lines = []
        
        # Header
        lines.append(f"# {intermediate.title}")
        lines.append("")
        
        # Metadata section
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Content Type**: {intermediate.content_type}")
        lines.append(f"- **Source**: {intermediate.source_path}")
        lines.append(f"- **Processed**: {intermediate.metadata.timestamp}")
        lines.append(f"- **Processing Time**: {intermediate.metadata.processing_time:.2f}s")
        lines.append(f"- **Mode**: {intermediate.metadata.mode}")
        lines.append("")
        
        # Main content
        lines.append("## Content")
        lines.append("")
        lines.append(intermediate.text_content)
        lines.append("")
        
        # Entities section
        if intermediate.entities:
            lines.append("## Extracted Entities")
            lines.append("")
            for entity in intermediate.entities:
                lines.append(f"- **{entity.name}** ({entity.type}) - Confidence: {entity.confidence:.2f}")
                if entity.properties:
                    for key, value in entity.properties.items():
                        lines.append(f"  - {key}: {value}")
            lines.append("")
        
        # Relations section
        if intermediate.relations:
            lines.append("## Extracted Relations")
            lines.append("")
            entity_map = {e.id: e.name for e in intermediate.entities}
            for relation in intermediate.relations:
                source_name = entity_map.get(relation.source_entity_id, "Unknown")
                target_name = entity_map.get(relation.target_entity_id, "Unknown")
                lines.append(f"- **{source_name}** --[{relation.type}]--> **{target_name}** (Confidence: {relation.confidence:.2f})")
                if relation.properties:
                    for key, value in relation.properties.items():
                        lines.append(f"  - {key}: {value}")
            lines.append("")
        
        # Content-specific sections
        if intermediate.segments:
            lines.append("## Segments")
            lines.append("")
            for i, segment in enumerate(intermediate.segments[:5]):  # Show first 5
                lines.append(f"### Segment {i+1}")
                if 'start' in segment and 'end' in segment:
                    lines.append(f"**Time**: {segment['start']:.2f}s - {segment['end']:.2f}s")
                if 'text' in segment:
                    lines.append(f"**Text**: {segment['text']}")
                if 'speaker' in segment:
                    lines.append(f"**Speaker**: {segment['speaker']}")
                lines.append("")
        
        if intermediate.pages:
            lines.append("## Pages")
            lines.append("")
            for i, page in enumerate(intermediate.pages[:3]):  # Show first 3
                lines.append(f"### Page {i+1}")
                if 'text' in page:
                    preview = page['text'][:200] + "..." if len(page['text']) > 200 else page['text']
                    lines.append(preview)
                lines.append("")
        
        if intermediate.links:
            lines.append("## Links")
            lines.append("")
            for link in intermediate.links[:10]:  # Show first 10
                if 'url' in link and 'text' in link:
                    lines.append(f"- [{link['text']}]({link['url']})")
            lines.append("")
        
        # Custom metadata
        if intermediate.custom_metadata:
            lines.append("## Additional Metadata")
            lines.append("")
            for key, value in intermediate.custom_metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def save_markdown(intermediate: IntermediateJSON, file_path: Union[str, Path]) -> None:
        """Save markdown to file."""
        markdown_content = MarkdownGenerator.generate(intermediate)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)


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