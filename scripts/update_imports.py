#!/usr/bin/env python3
"""
Script to update import statements from monolithic to modular package imports.

This script systematically updates all import statements in the codebase
to use the new modular package structure instead of the old monolithic imports.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import structlog

logger = structlog.get_logger(__name__)

# Import mapping from old to new
IMPORT_MAPPINGS = {
    # Processors
    'from morag_audio import': 'from morag_audio import',
    'from morag_video import': 'from morag_video import', 
    'from morag_document import': 'from morag_document import',
    'from morag_image import': 'from morag_image import',
    'from morag_web import': 'from morag_web import',
    'from morag_youtube import': 'from morag_youtube import',
    
    # Converters
    'from morag_audio import': 'from morag_audio import',
    'from morag_video import': 'from morag_video import',
    'from morag_document import': 'from morag_document import',
    'from morag_web import': 'from morag_web import',
    'from morag_core.interfaces.converter import': 'from morag_core.interfaces.converter import',
    'from morag_core.models import': 'from morag_core.models import',
    'from morag_services import': 'from morag_services import',
    
    # Services
    'from morag_services.embedding import': 'from morag_services.embedding import',
    'from morag_services.storage import': 'from morag_services.storage import',
    'from morag_services.processing import': 'from morag_services.processing import',
    'from morag_services.processing import': 'from morag_services.processing import',
    'from morag_audio.services import': 'from morag_audio.services import',
    'from morag_audio.services import': 'from morag_audio.services import',
    'from morag_audio.services import': 'from morag_audio.services import',
    'from morag_video.services import': 'from morag_video.services import',
    'from morag_image.services import': 'from morag_image.services import',
    'from morag_image.services import': 'from morag_image.services import',
    
    # Core
    'from morag_core.config import': 'from morag_core.config import',
    'from morag_core.exceptions import': 'from morag_core.exceptions import',
    'from morag_services.celery_app import': 'from morag_services.celery_app import',
    
    # Models
    'from morag_core.models import': 'from morag_core.models import',
    'from morag_core.models import': 'from morag_core.models import',
    
    # Utils
    'from morag_core.utils import': 'from morag_core.utils import',
    'from morag_core.utils import': 'from morag_core.utils import',
    
    # Tasks
    'from morag_services.tasks import': 'from morag_services.tasks import',
    'from morag_audio.tasks import': 'from morag_audio.tasks import',
    'from morag_video.tasks import': 'from morag_video.tasks import',
    'from morag_document.tasks import': 'from morag_document.tasks import',
    'from morag_image.tasks import': 'from morag_image.tasks import',
    'from morag_web.tasks import': 'from morag_web.tasks import',
    'from morag_youtube.tasks import': 'from morag_youtube.tasks import',
}

# Specific class/function mappings that need special handling
CLASS_MAPPINGS = {
    'AudioProcessor': 'morag_audio',
    'VideoProcessor': 'morag_video', 
    'DocumentProcessor': 'morag_document',
    'ImageProcessor': 'morag_image',
    'WebProcessor': 'morag_web',
    'YouTubeProcessor': 'morag_youtube',
    'AudioConverter': 'morag_audio',
    'VideoConverter': 'morag_video',
    'DocumentConverter': 'morag_document',
    'WebConverter': 'morag_web',
    'ConversionOptions': 'morag_core.models',
    'ChunkingStrategy': 'morag_core.models',
}


def update_file_imports(file_path: Path) -> bool:
    """Update imports in a single file.
    
    Args:
        file_path: Path to the file to update
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply import mappings
        for old_import, new_import in IMPORT_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                logger.info(f"Updated import in {file_path}: {old_import} -> {new_import}")
        
        # Handle specific patterns that need more complex replacement
        content = _handle_complex_imports(content, file_path)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to update {file_path}: {e}")
        return False


def _handle_complex_imports(content: str, file_path: Path) -> str:
    """Handle complex import patterns that need special processing."""
    
    # Handle processor instance imports like "from morag_audio import audio_processor"
    processor_pattern = r'from morag\.processors\.(\w+) import (\w+_processor|\w+Processor)'
    def replace_processor(match):
        module = match.group(1)
        processor = match.group(2)
        return f'from morag_{module} import {processor}'
    
    content = re.sub(processor_pattern, replace_processor, content)
    
    # Handle service imports
    service_pattern = r'from morag\.services\.(\w+) import (\w+_service|\w+Service)'
    def replace_service(match):
        module = match.group(1)
        service = match.group(2)
        # Map services to appropriate packages
        service_mappings = {
            'whisper_service': 'morag_audio.services',
            'speaker_diarization': 'morag_audio.services',
            'topic_segmentation': 'morag_audio.services',
            'ffmpeg_service': 'morag_video.services',
            'vision_service': 'morag_image.services',
            'ocr_service': 'morag_image.services',
            'embedding': 'morag_services.embedding',
            'storage': 'morag_services.storage',
            'chunking': 'morag_services.processing',
            'summarization': 'morag_services.processing',
        }
        
        target_package = service_mappings.get(module, 'morag_services')
        return f'from {target_package} import {service}'
    
    content = re.sub(service_pattern, replace_service, content)
    
    return content


def find_files_to_update() -> List[Path]:
    """Find all Python files that need import updates."""
    files_to_update = []
    
    # Directories to scan
    scan_dirs = [
        Path('examples'),
        Path('scripts'),
        Path('tests'),
        Path('src/morag'),  # Only for remaining files
    ]
    
    for scan_dir in scan_dirs:
        if scan_dir.exists():
            for py_file in scan_dir.rglob('*.py'):
                # Skip __pycache__ directories
                if '__pycache__' in str(py_file):
                    continue
                    
                # Check if file contains old imports
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for any old import patterns
                    old_patterns = [
                        'from morag.processors',
                        'from morag.converters',
                        'from morag.services',
                        'from morag.tasks',
                        'from morag.core',
                        'from morag.models',
                        'from morag.utils',
                    ]
                    
                    if any(pattern in content for pattern in old_patterns):
                        files_to_update.append(py_file)
                        
                except Exception as e:
                    logger.warning(f"Could not read {py_file}: {e}")
    
    return files_to_update


def main():
    """Main function to update all imports."""
    logger.info("Starting import update process")
    
    # Find files to update
    files_to_update = find_files_to_update()
    logger.info(f"Found {len(files_to_update)} files to update")
    
    # Update each file
    updated_count = 0
    for file_path in files_to_update:
        if update_file_imports(file_path):
            updated_count += 1
    
    logger.info(f"Updated imports in {updated_count} files")
    
    # Report files that still need manual attention
    remaining_files = []
    for file_path in files_to_update:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            old_patterns = [
                'from morag.processors',
                'from morag.converters', 
                'from morag.services',
                'from morag.tasks',
            ]
            
            if any(pattern in content for pattern in old_patterns):
                remaining_files.append(file_path)
                
        except Exception:
            pass
    
    if remaining_files:
        logger.warning(f"Files still containing old imports: {len(remaining_files)}")
        for file_path in remaining_files:
            logger.warning(f"  - {file_path}")
    else:
        logger.info("All imports successfully updated!")


if __name__ == '__main__':
    main()
