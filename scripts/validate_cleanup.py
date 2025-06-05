#!/usr/bin/env python3
"""
Validation script for Task 36 cleanup and Task 37 optimization.

This script validates that the cleanup and migration to modular architecture
has been completed successfully.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json


class CleanupValidator:
    """Validates the cleanup and migration work."""
    
    def __init__(self):
        self.issues = []
        self.successes = []
        
    def validate_all(self) -> Dict[str, any]:
        """Run all validation checks."""
        print("🔍 Validating MoRAG cleanup and migration...")
        print("=" * 60)
        
        # Task 36 validations
        self.validate_obsolete_files_removed()
        self.validate_import_updates()
        self.validate_registry_updates()
        
        # Task 37 validations
        self.validate_test_suite_created()
        self.validate_task_specifications()
        
        # Generate report
        return self.generate_report()
    
    def validate_obsolete_files_removed(self):
        """Validate that obsolete files have been removed."""
        print("\n📁 Checking obsolete file removal...")
        
        obsolete_files = [
            "src/morag/processors/audio.py",
            "src/morag/processors/video.py", 
            "src/morag/processors/document.py",
            "src/morag/processors/image.py",
            "src/morag/processors/web.py",
            "src/morag/processors/youtube.py",
            "src/morag/converters/audio.py",
            "src/morag/converters/video.py",
            "src/morag/converters/web.py",
            "src/morag/services/speaker_diarization.py",
            "src/morag/services/topic_segmentation.py",
            "src/morag/services/whisper_service.py",
            "src/morag/services/ffmpeg_service.py",
            "src/morag/services/vision_service.py",
            "src/morag/services/ocr_service.py",
            "src/morag/services/embedding.py",
            "src/morag/services/storage.py",
            "src/morag/services/chunking.py",
            "src/morag/services/summarization.py",
        ]
        
        removed_count = 0
        for file_path in obsolete_files:
            path = Path(file_path)
            if not path.exists():
                removed_count += 1
                self.successes.append(f"✅ Removed: {file_path}")
            else:
                self.issues.append(f"❌ Still exists: {file_path}")
        
        print(f"   Removed {removed_count}/{len(obsolete_files)} obsolete files")
    
    def validate_import_updates(self):
        """Validate that import statements have been updated."""
        print("\n🔄 Checking import statement updates...")
        
        old_patterns = [
            r'from morag\.processors\.',
            r'from morag\.converters\.',
            r'from morag\.services\.',
            r'from morag\.tasks\.',
            r'from morag\.core\.',
            r'from morag\.models\.',
            r'from morag\.utils\.',
        ]
        
        files_with_old_imports = []
        total_files_checked = 0
        
        # Check relevant directories
        check_dirs = [
            Path("examples"),
            Path("scripts"),
            Path("tests"),
            Path("src/morag"),  # Remaining files
        ]
        
        for check_dir in check_dirs:
            if check_dir.exists():
                for py_file in check_dir.rglob("*.py"):
                    if "__pycache__" in str(py_file):
                        continue
                    
                    total_files_checked += 1
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        for pattern in old_patterns:
                            if re.search(pattern, content):
                                files_with_old_imports.append(str(py_file))
                                break
                    except Exception:
                        continue
        
        if files_with_old_imports:
            print(f"   ❌ Found {len(files_with_old_imports)} files with old imports:")
            for file_path in files_with_old_imports[:5]:  # Show first 5
                self.issues.append(f"Old imports in: {file_path}")
                print(f"      - {file_path}")
            if len(files_with_old_imports) > 5:
                print(f"      ... and {len(files_with_old_imports) - 5} more")
        else:
            print(f"   ✅ All {total_files_checked} files use new import patterns")
            self.successes.append(f"Updated imports in {total_files_checked} files")
    
    def validate_registry_updates(self):
        """Validate that converter registry has been updated."""
        print("\n🔧 Checking converter registry updates...")
        
        registry_file = Path("src/morag/converters/registry.py")
        if not registry_file.exists():
            self.issues.append("❌ Registry file not found")
            return
        
        try:
            with open(registry_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for new modular imports
            modular_imports = [
                "from morag_audio.converters import AudioConverter",
                "from morag_video.converters import VideoConverter",
                "from morag_web import WebConverter",
            ]
            
            found_modular = 0
            for import_stmt in modular_imports:
                if import_stmt in content:
                    found_modular += 1
            
            if found_modular > 0:
                print(f"   ✅ Registry updated with {found_modular} modular imports")
                self.successes.append(f"Registry uses {found_modular} modular imports")
            else:
                print("   ❌ Registry not updated with modular imports")
                self.issues.append("Registry not updated with modular imports")
                
        except Exception as e:
            self.issues.append(f"Error checking registry: {e}")
    
    def validate_test_suite_created(self):
        """Validate that integration test suite has been created."""
        print("\n🧪 Checking integration test suite...")
        
        test_files = [
            "tests/integration/test_package_independence.py",
            "tests/integration/test_architecture_compliance.py", 
            "tests/integration/test_cross_package_integration.py",
        ]
        
        created_count = 0
        for test_file in test_files:
            path = Path(test_file)
            if path.exists():
                created_count += 1
                self.successes.append(f"✅ Created: {test_file}")
                
                # Check file size to ensure it's not empty
                if path.stat().st_size > 1000:  # At least 1KB
                    print(f"   ✅ {test_file} ({path.stat().st_size} bytes)")
                else:
                    print(f"   ⚠️  {test_file} (small file)")
            else:
                self.issues.append(f"❌ Missing: {test_file}")
        
        print(f"   Created {created_count}/{len(test_files)} integration test files")
    
    def validate_task_specifications(self):
        """Validate that task specification files exist."""
        print("\n📋 Checking task specifications...")
        
        task_files = [
            "tasks/36-cleanup-and-migration.md",
            "tasks/37-repository-structure-optimization.md",
        ]
        
        for task_file in task_files:
            path = Path(task_file)
            if path.exists():
                print(f"   ✅ {task_file} ({path.stat().st_size} bytes)")
                self.successes.append(f"Task spec exists: {task_file}")
            else:
                print(f"   ❌ Missing: {task_file}")
                self.issues.append(f"Missing task spec: {task_file}")
    
    def validate_package_structure(self):
        """Validate that package structure is correct."""
        print("\n📦 Checking package structure...")
        
        packages_dir = Path("packages")
        if not packages_dir.exists():
            self.issues.append("❌ Packages directory not found")
            return
        
        expected_packages = [
            "morag-core",
            "morag-services", 
            "morag-web",
            "morag-youtube",
            "morag-audio",
            "morag-video",
            "morag-document",
            "morag-image",
            "morag",  # Main integration package
        ]
        
        found_packages = 0
        for package_name in expected_packages:
            package_dir = packages_dir / package_name
            if package_dir.exists():
                found_packages += 1
                print(f"   ✅ {package_name}")
            else:
                print(f"   ❌ Missing: {package_name}")
                self.issues.append(f"Missing package: {package_name}")
        
        print(f"   Found {found_packages}/{len(expected_packages)} expected packages")
        self.successes.append(f"Found {found_packages} packages")
    
    def generate_report(self) -> Dict[str, any]:
        """Generate validation report."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\n✅ SUCCESSES ({len(self.successes)}):")
        for success in self.successes:
            print(f"   {success}")
        
        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   {issue}")
        else:
            print(f"\n🎉 NO ISSUES FOUND!")
        
        # Overall status
        total_checks = len(self.successes) + len(self.issues)
        success_rate = len(self.successes) / total_checks * 100 if total_checks > 0 else 0
        
        print(f"\n📈 OVERALL STATUS:")
        print(f"   Success Rate: {success_rate:.1f}% ({len(self.successes)}/{total_checks})")
        
        if success_rate >= 90:
            print("   🎯 EXCELLENT - Tasks 36 & 37 are substantially complete!")
        elif success_rate >= 75:
            print("   ✅ GOOD - Most work completed, minor issues remain")
        elif success_rate >= 50:
            print("   ⚠️  PARTIAL - Significant work completed, some issues remain")
        else:
            print("   ❌ NEEDS WORK - Major issues need to be addressed")
        
        return {
            "success_rate": success_rate,
            "successes": self.successes,
            "issues": self.issues,
            "total_checks": total_checks
        }


def main():
    """Main validation function."""
    validator = CleanupValidator()
    
    # Run all validations
    report = validator.validate_all()
    
    # Additional package structure check
    validator.validate_package_structure()
    
    # Save report to file
    report_file = Path("validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_file}")
    
    # Exit with appropriate code
    if report["success_rate"] >= 75:
        print("\n🎉 Validation completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Validation found significant issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
