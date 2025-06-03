#!/usr/bin/env python3
"""
Cleanup script for removing old/unused code after Tasks 25-29 implementation.

This script identifies and removes:
1. Placeholder implementations that have been replaced
2. Unused imports and dependencies
3. Deprecated converter methods
4. Old test files that are no longer relevant
"""

import os
import sys
from pathlib import Path
import re
import ast
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def find_placeholder_implementations():
    """Find and report placeholder implementations that should be cleaned up."""
    placeholder_patterns = [
        r'placeholder.*implementation',
        r'TODO.*implement',
        r'NotImplementedError',
        r'raise NotImplementedError',
        r'# TODO:',
        r'# FIXME:',
        r'# PLACEHOLDER'
    ]
    
    found_placeholders = []
    
    # Search in source files
    src_dir = PROJECT_ROOT / 'src'
    for py_file in src_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in placeholder_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        found_placeholders.append({
                            'file': py_file,
                            'line': i,
                            'content': line.strip(),
                            'pattern': pattern
                        })
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")
    
    return found_placeholders


def find_unused_imports():
    """Find potentially unused imports in Python files."""
    unused_imports = []
    
    src_dir = PROJECT_ROOT / 'src'
    for py_file in src_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find imports
            try:
                tree = ast.parse(content)
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                imports.append(f"{node.module}.{alias.name}")
                
                # Simple heuristic: check if import name appears elsewhere in file
                for imp in imports:
                    imp_name = imp.split('.')[-1]
                    # Count occurrences (excluding the import line itself)
                    occurrences = content.count(imp_name)
                    import_lines = content.count(f"import {imp_name}") + content.count(f"from {imp}")
                    
                    if occurrences <= import_lines:
                        unused_imports.append({
                            'file': py_file,
                            'import': imp,
                            'occurrences': occurrences - import_lines
                        })
                        
            except SyntaxError:
                logger.warning(f"Could not parse {py_file}")
                
        except Exception as e:
            logger.warning(f"Could not process {py_file}: {e}")
    
    return unused_imports


def find_deprecated_methods():
    """Find methods marked as deprecated or old implementations."""
    deprecated_patterns = [
        r'@deprecated',
        r'# DEPRECATED',
        r'def.*_old\(',
        r'def.*_legacy\(',
        r'def.*_placeholder\('
    ]
    
    deprecated_methods = []
    
    src_dir = PROJECT_ROOT / 'src'
    for py_file in src_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in deprecated_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        deprecated_methods.append({
                            'file': py_file,
                            'line': i,
                            'content': line.strip(),
                            'pattern': pattern
                        })
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")
    
    return deprecated_methods


def check_test_coverage():
    """Check if old test files need updating or removal."""
    test_files = []
    tests_dir = PROJECT_ROOT / 'tests'
    
    if tests_dir.exists():
        for test_file in tests_dir.rglob('test_*.py'):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for placeholder or old test patterns
                if any(pattern in content.lower() for pattern in ['placeholder', 'todo', 'fixme', 'skip']):
                    test_files.append({
                        'file': test_file,
                        'reason': 'Contains placeholder or TODO markers'
                    })
                
                # Check if test file is very small (might be placeholder)
                if len(content.split('\n')) < 20:
                    test_files.append({
                        'file': test_file,
                        'reason': 'Very small test file (possible placeholder)'
                    })
                    
            except Exception as e:
                logger.warning(f"Could not read {test_file}: {e}")
    
    return test_files


def generate_cleanup_report():
    """Generate a comprehensive cleanup report."""
    logger.info("Starting code cleanup analysis...")
    
    # Find issues
    placeholders = find_placeholder_implementations()
    unused_imports = find_unused_imports()
    deprecated_methods = find_deprecated_methods()
    test_issues = check_test_coverage()
    
    # Generate report
    report = []
    report.append("# Code Cleanup Report - Tasks 25-29 Implementation")
    report.append("=" * 60)
    report.append("")
    
    # Placeholder implementations
    if placeholders:
        report.append("## Placeholder Implementations Found")
        report.append(f"Found {len(placeholders)} placeholder implementations:")
        report.append("")
        for item in placeholders:
            report.append(f"- **{item['file']}:{item['line']}**")
            report.append(f"  ```python")
            report.append(f"  {item['content']}")
            report.append(f"  ```")
            report.append("")
    else:
        report.append("## ✅ No Placeholder Implementations Found")
        report.append("")
    
    # Unused imports
    if unused_imports:
        report.append("## Potentially Unused Imports")
        report.append(f"Found {len(unused_imports)} potentially unused imports:")
        report.append("")
        for item in unused_imports:
            report.append(f"- **{item['file']}**: `{item['import']}`")
        report.append("")
        report.append("*Note: This is a heuristic analysis. Manual review recommended.*")
        report.append("")
    else:
        report.append("## ✅ No Obviously Unused Imports Found")
        report.append("")
    
    # Deprecated methods
    if deprecated_methods:
        report.append("## Deprecated Methods Found")
        report.append(f"Found {len(deprecated_methods)} deprecated methods:")
        report.append("")
        for item in deprecated_methods:
            report.append(f"- **{item['file']}:{item['line']}**")
            report.append(f"  ```python")
            report.append(f"  {item['content']}")
            report.append(f"  ```")
            report.append("")
    else:
        report.append("## ✅ No Deprecated Methods Found")
        report.append("")
    
    # Test issues
    if test_issues:
        report.append("## Test Files Needing Review")
        report.append(f"Found {len(test_issues)} test files that may need attention:")
        report.append("")
        for item in test_issues:
            report.append(f"- **{item['file']}**: {item['reason']}")
        report.append("")
    else:
        report.append("## ✅ No Test File Issues Found")
        report.append("")
    
    # Summary
    total_issues = len(placeholders) + len(deprecated_methods) + len(test_issues)
    report.append("## Summary")
    report.append("")
    report.append(f"- **Placeholder implementations**: {len(placeholders)}")
    report.append(f"- **Potentially unused imports**: {len(unused_imports)}")
    report.append(f"- **Deprecated methods**: {len(deprecated_methods)}")
    report.append(f"- **Test files needing review**: {len(test_issues)}")
    report.append("")
    report.append(f"**Total issues requiring attention**: {total_issues}")
    
    if total_issues == 0:
        report.append("")
        report.append("🎉 **Excellent! No major cleanup issues found.**")
        report.append("")
        report.append("The codebase appears to be clean after the Tasks 25-29 implementation.")
    else:
        report.append("")
        report.append("📋 **Recommended Actions:**")
        report.append("")
        if placeholders:
            report.append("1. Review and remove placeholder implementations")
        if deprecated_methods:
            report.append("2. Remove or update deprecated methods")
        if test_issues:
            report.append("3. Update or remove outdated test files")
        if unused_imports:
            report.append("4. Review and remove unused imports (manual verification recommended)")
    
    return "\n".join(report)


def main():
    """Main cleanup analysis function."""
    try:
        # Generate report
        report = generate_cleanup_report()
        
        # Save report
        report_file = PROJECT_ROOT / 'cleanup_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Cleanup report generated: {report_file}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("CODE CLEANUP ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Report saved to: {report_file}")
        print("\nPlease review the report for cleanup recommendations.")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Cleanup analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
