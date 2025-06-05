#!/usr/bin/env python3
"""
Script to fix the remaining specific import issues identified in validation.

This script targets the specific import patterns that are still causing issues
after the main import update script.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import structlog

logger = structlog.get_logger(__name__)

# Specific import mappings for remaining issues
SPECIFIC_MAPPINGS = {
    # Core modules that still exist in src/morag
    'from src.morag.core.ai_error_handlers import': 'from src.morag.core.ai_error_handlers import',
    'from src.morag.core.resilience import': 'from src.morag.core.resilience import',
    
    # Services that still exist in src/morag
    'from src.morag.services.task_manager import': 'from src.morag.services.task_manager import',
    'from src.morag.services.metrics_service import': 'from src.morag.services.metrics_service import',
    'from src.morag.services.status_history import': 'from src.morag.services.status_history import',
    
    # Converters that still exist in src/morag
    'from src.morag.converters.pdf import': 'from src.morag.converters.pdf import',
    'from src.morag.converters.quality import': 'from src.morag.converters.quality import',
    
    # Tasks that still exist in src/morag
    'from src.morag.tasks.base import': 'from src.morag.tasks.base import',
    'from src.morag.tasks.audio_tasks import': 'from src.morag.tasks.audio_tasks import',
    'from src.morag.tasks.video_tasks import': 'from src.morag.tasks.video_tasks import',
    'from src.morag.tasks.document_tasks import': 'from src.morag.tasks.document_tasks import',
    'from src.morag.tasks.image_tasks import': 'from src.morag.tasks.image_tasks import',
    'from src.morag.tasks.web_tasks import': 'from src.morag.tasks.web_tasks import',
    'from src.morag.tasks.youtube_tasks import': 'from src.morag.tasks.youtube_tasks import',
    
    # Models that should go to core
    'from morag_core.models import': 'from morag_core.models import',
    'from morag_core.models import': 'from morag_core.models import',
    
    # Utils that should go to core
    'from morag_core.utils import': 'from morag_core.utils import',
    'from morag_core.utils import': 'from morag_core.utils import',
    
    # Core config and exceptions
    'from morag_core.config import': 'from morag_core.config import',
    'from morag_core.exceptions import': 'from morag_core.exceptions import',
    'from morag_services.celery_app import': 'from morag_services.celery_app import',
}

# Pattern-based replacements for more complex cases
PATTERN_REPLACEMENTS = [
    # Processor imports
    (r'from morag\.processors\.(\w+) import', r'from morag_\1 import'),
    
    # Converter imports (except those that still exist in src)
    (r'from morag\.converters\.audio import', r'from morag_audio import'),
    (r'from morag\.converters\.video import', r'from morag_video import'),
    (r'from morag\.converters\.web import', r'from morag_web import'),
    (r'from morag\.converters\.document import', r'from morag_document import'),
    (r'from morag\.converters\.image import', r'from morag_image import'),
    
    # Service imports for services that moved to packages
    (r'from morag\.services\.embedding import', r'from morag_services.embedding import'),
    (r'from morag\.services\.storage import', r'from morag_services.storage import'),
    (r'from morag\.services\.chunking import', r'from morag_services.processing import'),
    (r'from morag\.services\.summarization import', r'from morag_services.processing import'),
    (r'from morag\.services\.whisper_service import', r'from morag_audio.services import'),
    (r'from morag\.services\.speaker_diarization import', r'from morag_audio.services import'),
    (r'from morag\.services\.topic_segmentation import', r'from morag_audio.services import'),
    (r'from morag\.services\.ffmpeg_service import', r'from morag_video.services import'),
    (r'from morag\.services\.vision_service import', r'from morag_image.services import'),
    (r'from morag\.services\.ocr_service import', r'from morag_image.services import'),
]


def fix_file_imports(file_path: Path) -> bool:
    """Fix imports in a single file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Apply specific mappings first
        for old_import, new_import in SPECIFIC_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes_made.append(f"{old_import} -> {new_import}")
        
        # Apply pattern-based replacements
        for old_pattern, new_pattern in PATTERN_REPLACEMENTS:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                changes_made.append(f"Pattern: {old_pattern} -> {new_pattern}")
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Fixed {file_path}:")
            for change in changes_made:
                print(f"   {change}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Failed to fix {file_path}: {e}")
        return False


def find_files_with_old_imports() -> List[Path]:
    """Find all Python files that still have old import patterns."""
    files_with_issues = []
    
    old_patterns = [
        r'from morag\.processors\.',
        r'from morag\.converters\.',
        r'from morag\.services\.',
        r'from morag\.tasks\.',
        r'from morag\.core\.',
        r'from morag\.models\.',
        r'from morag\.utils\.',
    ]
    
    # Scan relevant directories
    scan_dirs = [
        Path('examples'),
        Path('scripts'),
        Path('tests'),
        Path('src/morag'),
    ]
    
    for scan_dir in scan_dirs:
        if scan_dir.exists():
            for py_file in scan_dir.rglob('*.py'):
                if '__pycache__' in str(py_file):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if file has any old patterns
                    has_old_imports = False
                    for pattern in old_patterns:
                        if re.search(pattern, content):
                            has_old_imports = True
                            break
                    
                    if has_old_imports:
                        files_with_issues.append(py_file)
                        
                except Exception:
                    continue
    
    return files_with_issues


def check_if_import_exists(import_path: str) -> bool:
    """Check if an import path actually exists."""
    try:
        # Convert import path to file path
        if import_path.startswith('from '):
            module_path = import_path.replace('from ', '').replace(' import', '').replace('.', '/')
            
            # Check if it's a src/morag path
            if module_path.startswith('src/morag'):
                file_path = Path(f"{module_path}.py")
                return file_path.exists()
            
            # Check if it's a package path
            if module_path.startswith('morag_'):
                package_dir = Path(f"packages/{module_path.replace('_', '-')}")
                return package_dir.exists()
        
        return False
    except:
        return False


def validate_fixes():
    """Validate that the fixes are correct."""
    print("\n🔍 Validating fixes...")
    
    files_with_issues = find_files_with_old_imports()
    
    if files_with_issues:
        print(f"❌ Still found {len(files_with_issues)} files with old imports:")
        for file_path in files_with_issues[:5]:  # Show first 5
            print(f"   {file_path}")
        if len(files_with_issues) > 5:
            print(f"   ... and {len(files_with_issues) - 5} more")
        return False
    else:
        print("✅ No files with old import patterns found!")
        return True


def main():
    """Main function to fix remaining imports."""
    print("🔧 Fixing remaining import issues...")
    print("=" * 50)
    
    # Find files that still have issues
    files_to_fix = find_files_with_old_imports()
    print(f"\n📁 Found {len(files_to_fix)} files with old import patterns")
    
    if not files_to_fix:
        print("✅ No files need fixing!")
        return
    
    # Fix each file
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_file_imports(file_path):
            fixed_count += 1
    
    print(f"\n📊 Fixed imports in {fixed_count}/{len(files_to_fix)} files")
    
    # Validate the fixes
    if validate_fixes():
        print("\n🎉 All import issues have been resolved!")
    else:
        print("\n⚠️  Some import issues may require manual attention.")


if __name__ == '__main__':
    main()
