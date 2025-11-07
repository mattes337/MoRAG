#!/usr/bin/env python3
"""
Simple syntax checker that focuses on critical import and syntax errors.
This is designed to catch the most common build-breaking issues.
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def check_syntax_errors(filepath: Path) -> List[str]:
    """Check for Python syntax errors."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        ast.parse(content, filename=str(filepath))
        return []

    except SyntaxError as e:
        return [f"Syntax error in {filepath}:{e.lineno}: {e.msg}"]
    except Exception as e:
        return [f"Error reading {filepath}: {e}"]


def check_obvious_import_errors(filepath: Path) -> List[str]:
    """Check for obvious import errors using regex patterns."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        errors = []
        lines = content.split("\n")

        # Get all imports
        imports = set()
        from_imports = set()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Match import statements
            import_match = re.match(r"^import\s+(\w+)", line)
            if import_match:
                imports.add(import_match.group(1))

            # Match from imports
            from_match = re.match(r"^from\s+\S+\s+import\s+(.+)", line)
            if from_match:
                imported_names = from_match.group(1)
                for name in imported_names.split(","):
                    name = name.strip().split(" as ")[0].strip()
                    if name != "*":
                        from_imports.add(name)

        all_imports = imports | from_imports

        # Check for common patterns that indicate missing imports
        for line_num, line in enumerate(lines, 1):
            # Check for re.something without re import
            if re.search(r"\bre\.[a-zA-Z_]", line) and "re" not in all_imports:
                errors.append(
                    f"{filepath}:{line_num}: Using 're' module without import"
                )

            # Check for os.something without os import
            if re.search(r"\bos\.[a-zA-Z_]", line) and "os" not in all_imports:
                errors.append(
                    f"{filepath}:{line_num}: Using 'os' module without import"
                )

            # Check for json.something without json import
            if re.search(r"\bjson\.[a-zA-Z_]", line) and "json" not in all_imports:
                errors.append(
                    f"{filepath}:{line_num}: Using 'json' module without import"
                )

            # Check for sys.something without sys import
            if re.search(r"\bsys\.[a-zA-Z_]", line) and "sys" not in all_imports:
                errors.append(
                    f"{filepath}:{line_num}: Using 'sys' module without import"
                )

            # Check for datetime.something without datetime import
            if (
                re.search(r"\bdatetime\.[a-zA-Z_]", line)
                and "datetime" not in all_imports
            ):
                errors.append(
                    f"{filepath}:{line_num}: Using 'datetime' module without import"
                )

            # Check for Path usage without pathlib import
            if re.search(r"\bPath\(", line) and "Path" not in all_imports:
                errors.append(
                    f"{filepath}:{line_num}: Using 'Path' without import from pathlib"
                )

        return errors

    except Exception as e:
        return [f"Error checking {filepath}: {e}"]


def check_file(filepath: Path) -> Tuple[List[str], List[str]]:
    """Check a single file for critical issues."""
    errors = []
    warnings = []

    # Check syntax first
    syntax_errors = check_syntax_errors(filepath)
    errors.extend(syntax_errors)

    # If syntax is OK, check for obvious import issues
    if not syntax_errors:
        import_errors = check_obvious_import_errors(filepath)
        errors.extend(import_errors)

    return errors, warnings


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory."""
    exclude_patterns = [
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        ".pytest_cache",
        "node_modules",
        ".mypy_cache",
        "build",
        "dist",
        ".tox",
    ]

    python_files = []

    for py_file in directory.rglob("*.py"):
        # Skip excluded directories
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue
        python_files.append(py_file)

    return python_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check Python files for critical syntax and import issues"
    )
    parser.add_argument(
        "paths", nargs="*", default=["."], help="Paths to check (files or directories)"
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
            files_to_check = find_python_files(path)
        else:
            print(f"Skipping {path}: not a Python file or directory")
            continue

        for filepath in files_to_check:
            errors, warnings = check_file(filepath)
            all_errors.extend(errors)
            all_warnings.extend(warnings)

    # Print results
    if all_errors:
        print("CRITICAL ERRORS:")
        for error in all_errors:
            print(f"  {error}")
        print()

    if all_warnings:
        print("WARNINGS:")
        for warning in all_warnings:
            print(f"  {warning}")
        print()

    # Summary
    print(f"Found {len(all_errors)} critical errors and {len(all_warnings)} warnings")

    if args.exit_on_error and all_errors:
        sys.exit(1)

    return len(all_errors)


if __name__ == "__main__":
    main()
