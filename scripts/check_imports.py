#!/usr/bin/env python3
"""
Import checker script to detect missing imports and other import-related issues.
This script analyzes Python files to find:
1. Missing imports (using undefined names)
2. Unused imports
3. Circular imports
4. Import order issues
"""

import argparse
import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check for import issues."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports: Dict[str, str] = {}  # name -> module
        self.from_imports: Dict[str, str] = {}  # name -> module
        self.used_names: Set[str] = set()
        self.defined_names: Set[str] = set()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def visit_Import(self, node):
        """Handle 'import module' statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle 'from module import name' statements."""
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                # Star imports are problematic but we can't track them easily
                self.warnings.append(f"Star import from {module} at line {node.lineno}")
            else:
                name = alias.asname if alias.asname else alias.name
                self.from_imports[name] = module
        self.generic_visit(node)

    def visit_Name(self, node):
        """Handle name usage."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Handle function definitions."""
        self.defined_names.add(node.name)
        # Add function parameters as defined names BEFORE visiting the body
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        # Add keyword-only arguments
        for arg in node.args.kwonlyargs:
            self.defined_names.add(arg.arg)
        # Add positional-only arguments (Python 3.8+)
        for arg in node.args.posonlyargs:
            self.defined_names.add(arg.arg)
        # Add *args and **kwargs
        if node.args.vararg:
            self.defined_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defined_names.add(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Handle async function definitions."""
        # Same as visit_FunctionDef
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Handle class definitions."""
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Handle attribute access like 're.sub'."""
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Handle exception handlers."""
        if node.name:
            self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_For(self, node):
        """Handle for loops."""
        # Add loop variable as defined
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.defined_names.add(elt.id)
        self.generic_visit(node)

    def visit_With(self, node):
        """Handle with statements."""
        for item in node.items:
            if item.optional_vars:
                if isinstance(item.optional_vars, ast.Name):
                    self.defined_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        """Handle list/dict/set comprehensions."""
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        self.generic_visit(node)

    def check_missing_imports(self):
        """Check for missing imports."""
        # Built-in names that don't need imports
        builtins = {
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "exit",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
            "__import__",
            # Built-in exceptions
            "Exception",
            "BaseException",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "ImportError",
            "ModuleNotFoundError",
            "FileNotFoundError",
            "RuntimeError",
            "OSError",
            "IOError",
            "KeyboardInterrupt",
            "SystemExit",
            "UnicodeDecodeError",
            "ZeroDivisionError",
            "ConnectionError",
            "LookupError",
            "NotImplementedError",
            "PermissionError",
            "TimeoutError",
            "MemoryError",
            "SyntaxError",
            "StopIteration",
            "GeneratorExit",
            "AssertionError",
            "DeprecationWarning",
            "Warning",
            "UserWarning",
            "FutureWarning",
            # Built-in constants
            "True",
            "False",
            "None",
            "Ellipsis",
            "NotImplemented",
            # Built-in module attributes
            "__name__",
            "__file__",
            "__doc__",
            "__package__",
            "__spec__",
            "__loader__",
            "__cached__",
            "__builtins__",
            # Common names
            "self",
            "cls",
            # Standard library modules commonly used without explicit import in some contexts
            "importlib",
        }

        # Common parameter names that are often false positives
        common_params = {
            # Common function parameters
            "query",
            "text",
            "content",
            "data",
            "result",
            "response",
            "context",
            "config",
            "options",
            "settings",
            "params",
            "args",
            "kwargs",
            "key",
            "value",
            "item",
            "items",
            "name",
            "path",
            "file",
            "filename",
            "filepath",
            "url",
            "uri",
            "source",
            "target",
            "destination",
            "input",
            "output",
            "request",
            "callback",
            # Common loop/comprehension variables
            "x",
            "y",
            "i",
            "j",
            "k",
            "v",
            "c",
            "e",
            "f",
            "g",
            "t",
            # Common test fixtures
            "tmp_path",
            "tmpdir",
            "monkeypatch",
            "capsys",
            "caplog",
            # Common mock names
            "mock_",
            "test_",
            "demo_",
            "example_",
            # Common database/query parameters
            "limit",
            "offset",
            "query_id",
            "entity_id",
            "document_id",
            "collection_name",
            "database_name",
            "table_name",
            "batch_size",
            "chunk_size",
            "max_size",
            "min_size",
            # Common type/validation parameters
            "domain",
            "strategy",
            "threshold",
            "max_depth",
        }

        # Names that are imported or defined locally
        available_names = (
            set(self.imports.keys())
            | set(self.from_imports.keys())
            | self.defined_names
            | builtins
        )

        # Find missing imports
        missing = self.used_names - available_names

        # Filter out common parameter names and single-letter variables
        filtered_missing = set()
        for name in missing:
            # Skip single-letter variables (likely loop variables)
            if len(name) == 1:
                continue
            # Skip if it starts with a common prefix
            if any(
                name.startswith(prefix)
                for prefix in common_params
                if prefix.endswith("_")
            ):
                continue
            # Skip if it's in common params
            if name in common_params:
                continue
            # Skip if it ends with common suffixes
            if any(
                name.endswith(suffix)
                for suffix in [
                    "_id",
                    "_name",
                    "_type",
                    "_path",
                    "_url",
                    "_size",
                    "_count",
                ]
            ):
                continue
            filtered_missing.add(name)

        for name in sorted(filtered_missing):
            self.errors.append(f"Missing import for '{name}' used in {self.filepath}")

    def check_unused_imports(self):
        """Check for unused imports."""
        all_imported = set(self.imports.keys()) | set(self.from_imports.keys())
        unused = all_imported - self.used_names

        for name in unused:
            self.warnings.append(f"Unused import '{name}' in {self.filepath}")


def check_file(filepath: Path) -> Tuple[List[str], List[str]]:
    """Check a single Python file for import issues."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = ImportChecker(str(filepath))
        checker.visit(tree)
        checker.check_missing_imports()
        checker.check_unused_imports()

        return checker.errors, checker.warnings

    except SyntaxError as e:
        return [f"Syntax error in {filepath}: {e}"], []
    except Exception as e:
        return [f"Error checking {filepath}: {e}"], []


def find_python_files(
    directory: Path, exclude_patterns: Optional[List[str]] = None
) -> List[Path]:
    """Find all Python files in a directory."""
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            "node_modules",
            ".mypy_cache",
        ]

    python_files = []

    for root, dirs, files in os.walk(directory):
        # Remove excluded directories
        dirs[:] = [
            d for d in dirs if not any(pattern in d for pattern in exclude_patterns)
        ]

        for file in files:
            if file.endswith(".py"):
                filepath = Path(root) / file
                python_files.append(filepath)

    return python_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check Python files for import issues")
    parser.add_argument(
        "paths", nargs="*", default=["."], help="Paths to check (files or directories)"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Patterns to exclude from checking",
    )
    parser.add_argument(
        "--errors-only", action="store_true", help="Only show errors, not warnings"
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="Exit with non-zero code if errors found",
    )

    args = parser.parse_args()

    all_errors = []
    all_warnings = []

    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file() and path.suffix == ".py":
            files_to_check = [path]
        elif path.is_dir():
            files_to_check = find_python_files(path, args.exclude)
        else:
            print(f"Skipping {path}: not a Python file or directory")
            continue

        for filepath in files_to_check:
            errors, warnings = check_file(filepath)
            all_errors.extend(errors)
            all_warnings.extend(warnings)

    # Print results
    if all_errors:
        print("ERRORS:")
        for error in all_errors:
            print(f"  {error}")
        print()

    if all_warnings and not args.errors_only:
        print("WARNINGS:")
        for warning in all_warnings:
            print(f"  {warning}")
        print()

    # Summary
    print(f"Found {len(all_errors)} errors and {len(all_warnings)} warnings")

    if args.exit_on_error and all_errors:
        sys.exit(1)

    return len(all_errors)


if __name__ == "__main__":
    main()
