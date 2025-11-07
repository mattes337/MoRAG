#!/usr/bin/env python3
"""
Comprehensive static analysis script for Python code.
Detects various code quality issues including:
1. Import errors (missing, unused, circular)
2. Undefined variables
3. Syntax errors
4. Common code smells
5. Type annotation issues
"""

import ast
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse


class StaticAnalyzer:
    """Comprehensive static analyzer for Python code."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors = []
        self.warnings = []

    def run_flake8(self, paths: List[Path]) -> Tuple[List[str], List[str]]:
        """Run flake8 for style and basic error checking."""
        try:
            cmd = ['flake8', '--format=json'] + [str(p) for p in paths]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.stdout:
                try:
                    flake8_results = json.loads(result.stdout)
                    errors = []
                    warnings = []

                    for item in flake8_results:
                        message = f"{item['filename']}:{item['line_number']}:{item['column_number']}: {item['code']} {item['text']}"
                        if item['code'].startswith('E'):
                            errors.append(message)
                        else:
                            warnings.append(message)

                    return errors, warnings
                except json.JSONDecodeError:
                    # Fallback to text parsing
                    lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                    errors = [line for line in lines if ':E' in line]
                    warnings = [line for line in lines if ':W' in line or ':F' in line]
                    return errors, warnings

            return [], []

        except FileNotFoundError:
            return [], ["flake8 not found - install with: pip install flake8"]

    def run_mypy(self, paths: List[Path]) -> Tuple[List[str], List[str]]:
        """Run mypy for type checking."""
        try:
            cmd = ['mypy', '--show-error-codes', '--no-error-summary'] + [str(p) for p in paths]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            errors = []
            warnings = []

            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        if 'error:' in line:
                            errors.append(line)
                        elif 'warning:' in line or 'note:' in line:
                            warnings.append(line)

            return errors, warnings

        except FileNotFoundError:
            return [], ["mypy not found - install with: pip install mypy"]

    def check_syntax(self, filepath: Path) -> List[str]:
        """Check for syntax errors."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            ast.parse(content, filename=str(filepath))
            return []

        except SyntaxError as e:
            return [f"Syntax error in {filepath}:{e.lineno}: {e.msg}"]
        except Exception as e:
            return [f"Error reading {filepath}: {e}"]

    def check_imports_with_ast(self, filepath: Path) -> Tuple[List[str], List[str]]:
        """Use our custom import checker."""
        from check_imports import check_file
        return check_file(filepath)

    def check_common_issues(self, filepath: Path) -> Tuple[List[str], List[str]]:
        """Check for common code issues."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            errors = []
            warnings = []
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Check for common issues
                if 'print(' in line and not line.strip().startswith('#'):
                    warnings.append(f"{filepath}:{i}: Found print statement (consider using logging)")

                if 'TODO' in line.upper() or 'FIXME' in line.upper():
                    warnings.append(f"{filepath}:{i}: Found TODO/FIXME comment")

                if len(line) > 120:
                    warnings.append(f"{filepath}:{i}: Line too long ({len(line)} > 120 characters)")

                # Check for potential security issues
                if 'eval(' in line or 'exec(' in line:
                    errors.append(f"{filepath}:{i}: Dangerous use of eval/exec")

                if 'shell=True' in line:
                    warnings.append(f"{filepath}:{i}: Use of shell=True in subprocess (potential security risk)")

            return errors, warnings

        except Exception as e:
            return [f"Error checking {filepath}: {e}"], []

    def analyze_file(self, filepath: Path) -> Dict[str, Any]:
        """Analyze a single Python file."""
        results = {
            'file': str(filepath),
            'errors': [],
            'warnings': []
        }

        # Check syntax first
        syntax_errors = self.check_syntax(filepath)
        results['errors'].extend(syntax_errors)

        # If syntax is OK, run other checks
        if not syntax_errors:
            # Import checking
            import_errors, import_warnings = self.check_imports_with_ast(filepath)
            results['errors'].extend(import_errors)
            results['warnings'].extend(import_warnings)

            # Common issues
            common_errors, common_warnings = self.check_common_issues(filepath)
            results['errors'].extend(common_errors)
            results['warnings'].extend(common_warnings)

        return results

    def analyze_project(self, paths: List[Path], include_external_tools: bool = True) -> Dict[str, Any]:
        """Analyze the entire project."""
        python_files = []

        for path in paths:
            if path.is_file() and path.suffix == '.py':
                python_files.append(path)
            elif path.is_dir():
                python_files.extend(self._find_python_files(path))

        results = {
            'files_analyzed': len(python_files),
            'file_results': [],
            'summary': {
                'total_errors': 0,
                'total_warnings': 0,
                'files_with_errors': 0,
                'files_with_warnings': 0
            }
        }

        # Analyze individual files
        for filepath in python_files:
            file_result = self.analyze_file(filepath)
            results['file_results'].append(file_result)

            if file_result['errors']:
                results['summary']['files_with_errors'] += 1
                results['summary']['total_errors'] += len(file_result['errors'])

            if file_result['warnings']:
                results['summary']['files_with_warnings'] += 1
                results['summary']['total_warnings'] += len(file_result['warnings'])

        # Run external tools if requested
        if include_external_tools and python_files:
            # Flake8
            flake8_errors, flake8_warnings = self.run_flake8(python_files)
            if flake8_errors or flake8_warnings:
                results['file_results'].append({
                    'file': 'flake8',
                    'errors': flake8_errors,
                    'warnings': flake8_warnings
                })
                results['summary']['total_errors'] += len(flake8_errors)
                results['summary']['total_warnings'] += len(flake8_warnings)

            # MyPy
            mypy_errors, mypy_warnings = self.run_mypy(python_files)
            if mypy_errors or mypy_warnings:
                results['file_results'].append({
                    'file': 'mypy',
                    'errors': mypy_errors,
                    'warnings': mypy_warnings
                })
                results['summary']['total_errors'] += len(mypy_errors)
                results['summary']['total_warnings'] += len(mypy_warnings)

        return results

    def _find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in a directory."""
        exclude_patterns = [
            '__pycache__', '.git', '.venv', 'venv', 'env',
            '.pytest_cache', 'node_modules', '.mypy_cache',
            'build', 'dist', '.tox'
        ]

        python_files = []

        for root, dirs, files in os.walk(directory):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    python_files.append(filepath)

        return python_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Comprehensive static analysis for Python projects')
    parser.add_argument('paths', nargs='*', default=['.'],
                       help='Paths to analyze (files or directories)')
    parser.add_argument('--no-external-tools', action='store_true',
                       help='Skip external tools (flake8, mypy)')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    parser.add_argument('--errors-only', action='store_true',
                       help='Only show errors, not warnings')
    parser.add_argument('--exit-on-error', action='store_true',
                       help='Exit with non-zero code if errors found')

    args = parser.parse_args()

    project_root = Path.cwd()
    analyzer = StaticAnalyzer(project_root)

    paths = [Path(p) for p in args.paths]
    results = analyzer.analyze_project(paths, not args.no_external_tools)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print(f"Static Analysis Results")
        print(f"=" * 50)
        print(f"Files analyzed: {results['files_analyzed']}")
        print(f"Total errors: {results['summary']['total_errors']}")
        print(f"Total warnings: {results['summary']['total_warnings']}")
        print()

        for file_result in results['file_results']:
            if file_result['errors'] or (file_result['warnings'] and not args.errors_only):
                print(f"File: {file_result['file']}")

                if file_result['errors']:
                    print("  ERRORS:")
                    for error in file_result['errors']:
                        print(f"    {error}")

                if file_result['warnings'] and not args.errors_only:
                    print("  WARNINGS:")
                    for warning in file_result['warnings']:
                        print(f"    {warning}")

                print()

    if args.exit_on_error and results['summary']['total_errors'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
