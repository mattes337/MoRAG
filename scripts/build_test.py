#!/usr/bin/env python3
"""
Simple build test script that catches import and syntax errors.
This script tries to import all Python modules to catch runtime import errors.
"""

import sys
import os
import importlib.util
import traceback
from pathlib import Path
from typing import List, Tuple
import argparse


def find_python_modules(directory: Path) -> List[Tuple[Path, str]]:
    """Find all Python modules and their import paths."""
    modules = []
    exclude_patterns = [
        '__pycache__', '.git', '.venv', 'venv', 'env',
        '.pytest_cache', 'node_modules', '.mypy_cache',
        'build', 'dist', '.tox'
    ]

    for py_file in directory.rglob('*.py'):
        # Skip excluded directories
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue

        # Skip __init__.py files for now (they often have complex imports)
        if py_file.name == '__init__.py':
            continue

        # Convert file path to module path
        relative_path = py_file.relative_to(directory)
        module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')

        modules.append((py_file, module_path))

    return modules


def test_module_import(file_path: Path, module_path: str) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        # Try to load the module spec
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None:
            return False, f"Could not create module spec for {module_path}"

        # Try to create the module
        module = importlib.util.module_from_spec(spec)
        if module is None:
            return False, f"Could not create module from spec for {module_path}"

        # Try to execute the module (this will catch import errors)
        spec.loader.exec_module(module)

        return True, ""

    except ImportError as e:
        return False, f"Import error in {module_path}: {e}"
    except SyntaxError as e:
        return False, f"Syntax error in {module_path}:{e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error in {module_path}: {e}"


def test_syntax_only(file_path: Path) -> Tuple[bool, str]:
    """Test only syntax without importing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        compile(content, str(file_path), 'exec')
        return True, ""

    except SyntaxError as e:
        return False, f"Syntax error in {file_path}:{e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading {file_path}: {e}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test Python modules for import and syntax errors')
    parser.add_argument('paths', nargs='*', default=['packages'],
                       help='Paths to test (directories)')
    parser.add_argument('--syntax-only', action='store_true',
                       help='Only check syntax, do not try to import')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose output')

    args = parser.parse_args()

    total_files = 0
    total_errors = 0
    errors = []

    for path_str in args.paths:
        path = Path(path_str)

        if not path.exists():
            print(f"Path does not exist: {path}")
            continue

        if not path.is_dir():
            print(f"Path is not a directory: {path}")
            continue

        print(f"Testing modules in {path}...")

        modules = find_python_modules(path)
        total_files += len(modules)

        for file_path, module_path in modules:
            if args.verbose:
                print(f"  Testing {module_path}...")

            if args.syntax_only:
                success, error_msg = test_syntax_only(file_path)
            else:
                success, error_msg = test_module_import(file_path, module_path)

            if not success:
                total_errors += 1
                errors.append(error_msg)
                if not args.verbose:
                    print(f"  ❌ {module_path}")
                else:
                    print(f"  ❌ {module_path}: {error_msg}")
            elif args.verbose:
                print(f"  ✅ {module_path}")

    print(f"\nResults:")
    print(f"  Files tested: {total_files}")
    print(f"  Errors found: {total_errors}")

    if errors:
        print(f"\nErrors:")
        for error in errors:
            print(f"  {error}")

    if total_errors == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
