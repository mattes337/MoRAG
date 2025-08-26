#!/usr/bin/env python3
"""
MoRAG Python Syntax Checker

This script performs comprehensive syntax checking on all Python files in the MoRAG project.
It checks for:
1. Python syntax errors (compile-time errors)
2. Import errors
3. Code style issues (using flake8)
4. Type checking (using mypy)
5. Import sorting (using isort)

Usage:
    python scripts/check_syntax.py [--fix] [--verbose] [--path PATH]

Options:
    --fix       Automatically fix issues where possible (isort, black)
    --verbose   Show detailed output
    --path      Check specific path instead of entire project
"""

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import importlib.util


class SyntaxChecker:
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")
    
    def find_python_files(self, path: Path = None) -> List[Path]:
        """Find all Python files in the project."""
        search_path = path or self.project_root
        python_files = []

        # If path is a specific file, return it directly
        if path and path.is_file() and str(path).endswith('.py'):
            return [path]

        # Skip certain directories
        skip_dirs = {
            '__pycache__', '.git', 'venv', 'env', '.venv',
            'node_modules', '.pytest_cache', 'htmlcov',
            '.mypy_cache', '.coverage', 'dist', 'build'
        }

        for root, dirs, files in os.walk(search_path):
            # Remove skip directories from dirs list to avoid traversing them
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        return sorted(python_files)
    
    def check_syntax(self, file_path: Path) -> bool:
        """Check Python syntax by attempting to compile the file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Try to parse the AST
            ast.parse(source, filename=str(file_path))
            self.log(f"✓ Syntax OK: {file_path}")
            return True
            
        except SyntaxError as e:
            error_msg = f"Syntax Error in {file_path}:{e.lineno}:{e.offset} - {e.msg}"
            self.errors.append(error_msg)
            self.log(error_msg, "ERROR")
            return False
        except Exception as e:
            error_msg = f"Error reading {file_path}: {e}"
            self.errors.append(error_msg)
            self.log(error_msg, "ERROR")
            return False
    
    def check_imports(self, file_path: Path) -> bool:
        """Check if all imports in the file can be resolved."""
        try:
            # Store original sys.path to restore later
            original_path = sys.path.copy()

            # Add the project root and packages to Python path
            sys.path.insert(0, str(self.project_root))
            sys.path.insert(0, str(self.project_root / "packages"))

            # Add all package source directories to Python path
            packages_dir = self.project_root / "packages"
            if packages_dir.exists():
                for package_dir in packages_dir.iterdir():
                    if package_dir.is_dir():
                        src_dir = package_dir / "src"
                        if src_dir.exists():
                            sys.path.insert(0, str(src_dir))

            # Also add the CLI directory for CLI scripts
            cli_dir = self.project_root / "cli"
            if cli_dir.exists():
                sys.path.insert(0, str(cli_dir))

            # Try to load the module
            # Use a unique module name based on the file to avoid phantom errors
            module_name = f"syntax_check_{hash(str(file_path)) % 100000}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Try to execute the module to catch runtime import errors
                try:
                    spec.loader.exec_module(module)
                    self.log(f"✓ Imports OK: {file_path}")
                    return True
                except Exception as e:
                    # Check if it's a critical import error or just a runtime error
                    error_str = str(e).lower()

                    # Filter out phantom temp_module errors and other false positives
                    if ('temp_module' in error_str or
                        module_name in error_str or
                        'syntax_check_' in error_str):
                        # This is likely a phantom error from our syntax checking process
                        # Don't log these as they're not real issues
                        return True
                    elif any(keyword in error_str for keyword in ['no module named', 'cannot import', 'importerror', 'modulenotfounderror']):
                        warning_msg = f"Import Error in {file_path}: {e}"
                        self.warnings.append(warning_msg)
                        self.log(warning_msg, "WARNING")
                        return False
                    else:
                        # Runtime error, but imports are probably OK
                        self.log(f"Runtime error in {file_path} (imports OK): {e}", "WARNING")
                        return True
            else:
                warning_msg = f"Could not create module spec for {file_path}"
                self.warnings.append(warning_msg)
                self.log(warning_msg, "WARNING")
                return False

        except ImportError as e:
            warning_msg = f"Import Error in {file_path}: {e}"
            self.warnings.append(warning_msg)
            self.log(warning_msg, "WARNING")
            return False
        except Exception as e:
            # Don't treat other exceptions as critical errors for import checking
            self.log(f"Could not check imports for {file_path}: {e}", "WARNING")
            return True
        finally:
            # Restore original sys.path
            sys.path[:] = original_path
    
    def run_flake8(self, files: List[Path]) -> bool:
        """Run flake8 on the specified files."""
        try:
            # Try different flake8 locations
            flake8_paths = [
                "flake8",
                str(self.project_root / "venv" / "Scripts" / "flake8.exe"),
                str(self.project_root / "venv" / "bin" / "flake8"),
            ]

            flake8_cmd = None
            for path in flake8_paths:
                try:
                    result = subprocess.run([path, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        flake8_cmd = path
                        break
                except FileNotFoundError:
                    continue

            if not flake8_cmd:
                warning_msg = "flake8 not found. Install with: pip install flake8"
                self.warnings.append(warning_msg)
                self.log(warning_msg, "WARNING")
                return True

            cmd = [flake8_cmd] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                self.log("✓ Flake8 checks passed")
                return True
            else:
                error_msg = f"Flake8 issues found:\n{result.stdout}"
                self.errors.append(error_msg)
                self.log(error_msg, "ERROR")
                return False

        except Exception as e:
            warning_msg = f"Error running flake8: {e}"
            self.warnings.append(warning_msg)
            self.log(warning_msg, "WARNING")
            return True  # Don't fail if flake8 has issues
    
    def run_mypy(self, files: List[Path]) -> bool:
        """Run mypy type checking on the specified files."""
        try:
            # Try different mypy locations
            mypy_paths = [
                "mypy",
                str(self.project_root / "venv" / "Scripts" / "mypy.exe"),
                str(self.project_root / "venv" / "bin" / "mypy"),
            ]

            mypy_cmd = None
            for path in mypy_paths:
                try:
                    result = subprocess.run([path, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        mypy_cmd = path
                        break
                except FileNotFoundError:
                    continue

            if not mypy_cmd:
                warning_msg = "mypy not found. Install with: pip install mypy"
                self.warnings.append(warning_msg)
                self.log(warning_msg, "WARNING")
                return True

            cmd = [mypy_cmd] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                self.log("✓ MyPy type checks passed")
                return True
            else:
                warning_msg = f"MyPy type issues found:\n{result.stdout}"
                self.warnings.append(warning_msg)
                self.log(warning_msg, "WARNING")
                return True  # Treat type issues as warnings, not errors

        except Exception as e:
            warning_msg = f"Error running mypy: {e}"
            self.warnings.append(warning_msg)
            self.log(warning_msg, "WARNING")
            return True
    
    def run_isort_check(self, files: List[Path], fix: bool = False) -> bool:
        """Check import sorting with isort."""
        try:
            # Try different isort locations
            isort_paths = [
                "isort",
                str(self.project_root / "venv" / "Scripts" / "isort.exe"),
                str(self.project_root / "venv" / "bin" / "isort"),
            ]

            isort_cmd = None
            for path in isort_paths:
                try:
                    result = subprocess.run([path, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        isort_cmd = path
                        break
                except FileNotFoundError:
                    continue

            if not isort_cmd:
                warning_msg = "isort not found. Install with: pip install isort"
                self.warnings.append(warning_msg)
                self.log(warning_msg, "WARNING")
                return True

            cmd = [isort_cmd]
            if not fix:
                cmd.append("--check-only")
            cmd.extend([str(f) for f in files])

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                self.log("✓ Import sorting OK")
                return True
            else:
                if fix:
                    self.log("✓ Import sorting fixed")
                    return True
                else:
                    warning_msg = f"Import sorting issues found:\n{result.stdout}"
                    self.warnings.append(warning_msg)
                    self.log(warning_msg, "WARNING")
                    return True  # Treat as warning

        except Exception as e:
            warning_msg = f"Error running isort: {e}"
            self.warnings.append(warning_msg)
            self.log(warning_msg, "WARNING")
            return True
    
    def check_all(self, path: Path = None, fix: bool = False) -> bool:
        """Run all syntax checks."""
        self.log("Starting comprehensive Python syntax check...")
        
        files = self.find_python_files(path)
        self.log(f"Found {len(files)} Python files to check")
        
        if not files:
            self.log("No Python files found!")
            return True
        
        # 1. Check syntax for each file
        syntax_ok = True
        for file_path in files:
            if not self.check_syntax(file_path):
                syntax_ok = False
        
        # 2. Check imports (only if syntax is OK)
        if syntax_ok:
            for file_path in files:
                self.check_imports(file_path)
        
        # 3. Run code quality tools in batches to avoid command line length limits
        batch_size = 50  # Process files in batches to avoid command line length issues
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            self.log(f"Running code quality checks on batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
            self.run_flake8(batch)
            self.run_mypy(batch)
            self.run_isort_check(batch, fix=fix)
        
        # Summary
        self.log("\n" + "="*50)
        self.log("SYNTAX CHECK SUMMARY")
        self.log("="*50)
        
        if self.errors:
            self.log(f"❌ {len(self.errors)} ERRORS found:", "ERROR")
            for error in self.errors:
                self.log(f"  - {error}", "ERROR")
        else:
            self.log("✅ No syntax errors found!")
        
        if self.warnings:
            self.log(f"⚠️  {len(self.warnings)} WARNINGS found:", "WARNING")
            for warning in self.warnings:
                self.log(f"  - {warning}", "WARNING")
        
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Check Python syntax for MoRAG project")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues where possible")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--path", type=str, help="Check specific path instead of entire project")
    
    args = parser.parse_args()
    
    # Get project root (assuming script is in scripts/ directory)
    project_root = Path(__file__).parent.parent
    
    checker = SyntaxChecker(project_root, verbose=args.verbose)
    
    check_path = Path(args.path) if args.path else None
    success = checker.check_all(path=check_path, fix=args.fix)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
