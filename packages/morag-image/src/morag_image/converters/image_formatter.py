"""Image formatter for LLM documentation compliance."""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog

logger = structlog.get_logger(__name__)


class ImageFormatter:
    """Formats image analysis content according to LLM documentation specifications."""

    def format_image_content(
        self, 
        raw_content: str, 
        file_path: Path, 
        metadata: Dict[str, Any]
    ) -> str:
        """Format image content according to LLM documentation format.

        Args:
            raw_content: Raw markdown content from markitdown/analysis
            file_path: Path to the original image file
            metadata: Image metadata

        Returns:
            Formatted markdown content following LLM specifications
        """
        filename = file_path.name
        
        # Build formatted content
        formatted_parts = []
        
        # Header Section: "Image Analysis: filename.ext"
        formatted_parts.append(f"# Image Analysis: {filename}")
        formatted_parts.append("")
        
        # Image Information Section
        formatted_parts.append("## Image Information")
        
        # Extract metadata for information section
        dimensions = metadata.get('dimensions', 'Unknown')
        file_size = metadata.get('file_size', 'Unknown')
        format_type = metadata.get('format', file_path.suffix.lstrip('.').upper())
        color_space = metadata.get('color_space', 'Unknown')
        
        # Format file size if it's a number
        if isinstance(file_size, (int, float)):
            if file_size < 1024:
                file_size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                file_size_str = f"{file_size / 1024:.1f} KB"
            else:
                file_size_str = f"{file_size / (1024 * 1024):.1f} MB"
        else:
            file_size_str = str(file_size)
        
        # Format dimensions
        if isinstance(dimensions, dict):
            width = dimensions.get('width', 'Unknown')
            height = dimensions.get('height', 'Unknown')
            dimensions_str = f"{width} x {height}"
        elif isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2:
            dimensions_str = f"{dimensions[0]} x {dimensions[1]}"
        else:
            dimensions_str = str(dimensions)
        
        formatted_parts.extend([
            f"- **Dimensions**: {dimensions_str}",
            f"- **File Size**: {file_size_str}",
            f"- **Format**: {format_type}",
            f"- **Color Space**: {color_space}",
            ""
        ])
        
        # Parse and format content sections
        sections = self._parse_content_sections(raw_content)
        
        # Visual Content Section
        visual_content = sections.get('visual', '')
        if visual_content:
            formatted_parts.append("## Visual Content")
            formatted_parts.append("")
            formatted_parts.append(visual_content.strip())
            formatted_parts.append("")
        
        # Text Content (OCR) Section
        ocr_content = sections.get('ocr', '')
        if ocr_content:
            formatted_parts.append("## Text Content (OCR)")
            formatted_parts.append("")
            formatted_parts.append(ocr_content.strip())
            formatted_parts.append("")
        
        # Objects Detected Section
        objects_content = sections.get('objects', '')
        if objects_content:
            formatted_parts.append("## Objects Detected")
            formatted_parts.append("")
            formatted_parts.append(objects_content.strip())
            formatted_parts.append("")
        
        # If no specific sections found, add general content
        if not any(sections.values()) and raw_content.strip():
            formatted_parts.append("## Analysis")
            formatted_parts.append("")
            formatted_parts.append(raw_content.strip())
            formatted_parts.append("")
        
        return "\n".join(formatted_parts).rstrip() + "\n"
    
    def _parse_content_sections(self, raw_content: str) -> Dict[str, str]:
        """Parse raw content into structured sections.

        Args:
            raw_content: Raw content from image analysis

        Returns:
            Dictionary with parsed sections
        """
        sections = {
            'visual': '',
            'ocr': '',
            'objects': ''
        }
        
        if not raw_content.strip():
            return sections
        
        # Try to identify different types of content
        lines = raw_content.split('\n')
        current_section = 'visual'  # Default section
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check for section indicators
            if any(keyword in line_lower for keyword in ['text content', 'ocr', 'extracted text']):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                current_section = 'ocr'
                continue
            elif any(keyword in line_lower for keyword in ['objects detected', 'detected objects', 'object detection']):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                current_section = 'objects'
                continue
            elif any(keyword in line_lower for keyword in ['visual', 'description', 'image shows', 'the image']):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                current_section = 'visual'
                continue
            
            # Add line to current section
            current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Post-process sections
        sections = self._post_process_sections(sections)
        
        return sections
    
    def _post_process_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Post-process sections to improve formatting.

        Args:
            sections: Raw sections dictionary

        Returns:
            Processed sections dictionary
        """
        processed = {}
        
        for section_name, content in sections.items():
            if not content.strip():
                processed[section_name] = ''
                continue
            
            if section_name == 'visual':
                # Format visual description
                processed[section_name] = self._format_visual_description(content)
            elif section_name == 'ocr':
                # Format OCR text
                processed[section_name] = self._format_ocr_text(content)
            elif section_name == 'objects':
                # Format object detection results
                processed[section_name] = self._format_objects_detected(content)
            else:
                processed[section_name] = content.strip()
        
        return processed
    
    def _format_visual_description(self, content: str) -> str:
        """Format visual description content.

        Args:
            content: Raw visual description

        Returns:
            Formatted visual description
        """
        # Clean up and format the description
        content = content.strip()
        
        # Ensure it starts with a descriptive sentence
        if content and not content[0].isupper():
            content = content[0].upper() + content[1:]
        
        # Remove redundant phrases
        content = re.sub(r'^(The image shows?|This image|Image shows?)\s*:?\s*', '', content, flags=re.IGNORECASE)
        
        # Ensure proper sentence structure
        if content and not content.endswith('.'):
            content += '.'
        
        return content
    
    def _format_ocr_text(self, content: str) -> str:
        """Format OCR text content.

        Args:
            content: Raw OCR text

        Returns:
            Formatted OCR text
        """
        if not content.strip():
            return "No text detected in the image."
        
        # Split into lines and format as quoted text
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Format as quoted strings
        formatted_lines = []
        for line in lines:
            if line and not line.startswith('"'):
                formatted_lines.append(f'"{line}"')
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_objects_detected(self, content: str) -> str:
        """Format object detection results.

        Args:
            content: Raw object detection content

        Returns:
            Formatted object detection results
        """
        if not content.strip():
            return "No objects detected in the image."
        
        # Try to format as a list
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        formatted_lines = []
        for line in lines:
            if line:
                # Ensure list format
                if not line.startswith('-') and not line.startswith('*'):
                    formatted_lines.append(f"- {line}")
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def extract_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from image file.

        Args:
            file_path: Path to image file

        Returns:
            Dictionary containing image metadata
        """
        metadata = {}
        
        try:
            # Try to get image dimensions and other metadata
            from PIL import Image
            
            with Image.open(file_path) as img:
                metadata['dimensions'] = {'width': img.width, 'height': img.height}
                metadata['format'] = img.format or file_path.suffix.lstrip('.').upper()
                metadata['color_space'] = img.mode
                
                # Get file size
                metadata['file_size'] = file_path.stat().st_size
                
        except ImportError:
            logger.warning("PIL not available, cannot extract image metadata")
            metadata['dimensions'] = 'Unknown'
            metadata['format'] = file_path.suffix.lstrip('.').upper()
            metadata['color_space'] = 'Unknown'
            metadata['file_size'] = file_path.stat().st_size if file_path.exists() else 'Unknown'
        except Exception as e:
            logger.warning(f"Failed to extract image metadata: {e}")
            metadata['dimensions'] = 'Unknown'
            metadata['format'] = file_path.suffix.lstrip('.').upper()
            metadata['color_space'] = 'Unknown'
            metadata['file_size'] = file_path.stat().st_size if file_path.exists() else 'Unknown'
        
        return metadata
