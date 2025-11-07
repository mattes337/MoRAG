"""Integration tests for architecture compliance.

This module tests that the codebase follows the modular architecture
patterns and doesn't use deprecated monolithic import patterns.
"""

import pytest
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import yaml


class TestArchitectureCompliance:
    """Test architecture compliance across the codebase."""

    # Forbidden import patterns (old monolithic structure)
    FORBIDDEN_PATTERNS = [
        r'from morag\.core\.',
        r'from morag\.services\.',
        r'from morag\.processors\.',
        r'from morag\.converters\.',
        r'from morag\.tasks\.',
        r'from morag\.models\.',
        r'from morag\.utils\.',
        r'import morag\.core\.',
        r'import morag\.services\.',
        r'import morag\.processors\.',
        r'import morag\.converters\.',
    ]

    # Allowed import patterns (new modular structure)
    ALLOWED_PATTERNS = [
        r'from morag_core',
        r'from morag_services',
        r'from morag_web',
        r'from morag_youtube',
        r'from morag_audio',
        r'from morag_video',
        r'from morag_document',
        r'from morag_image',
        r'import morag_core',
        r'import morag_services',
        r'import morag_web',
        r'import morag_youtube',
        r'import morag_audio',
        r'import morag_video',
        r'import morag_document',
        r'import morag_image',
    ]

    # Package dependency rules
    DEPENDENCY_RULES = {
        'morag_core': [],
        'morag_services': ['morag_core'],
        'morag_web': ['morag_core', 'morag_services'],
        'morag_youtube': ['morag_core', 'morag_services', 'morag_audio', 'morag_video'],
        'morag_audio': ['morag_core', 'morag_services'],
        'morag_video': ['morag_core', 'morag_services', 'morag_audio'],
        'morag_document': ['morag_core', 'morag_services'],
        'morag_image': ['morag_core', 'morag_services'],
    }

    def test_no_forbidden_imports(self):
        """Test that no files use forbidden import patterns."""
        violations = []

        for py_file in self._find_python_files():
            file_violations = self._check_file_for_forbidden_imports(py_file)
            violations.extend(file_violations)

        if violations:
            violation_msg = "\n".join([
                f"{file}:{line}: {pattern}"
                for file, line, pattern in violations
            ])
            pytest.fail(f"Found forbidden import patterns:\n{violation_msg}")

    def test_dependency_compliance(self):
        """Test that packages only import allowed dependencies."""
        violations = []

        for package_name, allowed_deps in self.DEPENDENCY_RULES.items():
            package_violations = self._check_package_dependencies(package_name, allowed_deps)
            violations.extend(package_violations)

        if violations:
            violation_msg = "\n".join([
                f"{package}: imports {imported} (not in allowed: {allowed})"
                for package, imported, allowed in violations
            ])
            pytest.fail(f"Found dependency violations:\n{violation_msg}")

    def test_import_consistency(self):
        """Test that import patterns are consistent across the codebase."""
        inconsistencies = []

        for py_file in self._find_python_files():
            file_inconsistencies = self._check_import_consistency(py_file)
            inconsistencies.extend(file_inconsistencies)

        if inconsistencies:
            inconsistency_msg = "\n".join([
                f"{file}:{line}: {issue}"
                for file, line, issue in inconsistencies
            ])
            pytest.fail(f"Found import inconsistencies:\n{inconsistency_msg}")

    def test_no_circular_imports(self):
        """Test that there are no circular imports between packages."""
        import_graph = self._build_import_graph()
        cycles = self._find_cycles(import_graph)

        if cycles:
            cycle_msg = "\n".join([
                " -> ".join(cycle + [cycle[0]])
                for cycle in cycles
            ])
            pytest.fail(f"Found circular imports:\n{cycle_msg}")

    def test_package_isolation(self):
        """Test that packages maintain proper isolation."""
        violations = []

        # Check that packages don't directly access internal modules of other packages
        for py_file in self._find_python_files():
            file_violations = self._check_package_isolation(py_file)
            violations.extend(file_violations)

        if violations:
            violation_msg = "\n".join([
                f"{file}:{line}: {violation}"
                for file, line, violation in violations
            ])
            pytest.fail(f"Found package isolation violations:\n{violation_msg}")

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []

        # Scan relevant directories
        scan_dirs = [
            Path("src"),
            Path("packages"),
            Path("examples"),
            Path("scripts"),
            Path("tests"),
        ]

        for scan_dir in scan_dirs:
            if scan_dir.exists():
                for py_file in scan_dir.rglob("*.py"):
                    # Skip __pycache__ and .git directories
                    if "__pycache__" not in str(py_file) and ".git" not in str(py_file):
                        python_files.append(py_file)

        return python_files

    def _check_file_for_forbidden_imports(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Check a file for forbidden import patterns."""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern in self.FORBIDDEN_PATTERNS:
                    if re.search(pattern, line):
                        violations.append((str(file_path), line_num, line.strip()))

        except Exception as e:
            # Skip files that can't be read
            pass

        return violations

    def _check_package_dependencies(self, package_name: str, allowed_deps: List[str]) -> List[Tuple[str, str, List[str]]]:
        """Check that a package only imports allowed dependencies."""
        violations = []

        # Find package directory
        package_dir = Path("packages") / package_name.replace('_', '-')
        if not package_dir.exists():
            return violations

        # Check all Python files in the package
        for py_file in package_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse imports
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_package = self._extract_package_name(alias.name)
                            if imported_package and imported_package.startswith('morag_'):
                                if imported_package not in allowed_deps and imported_package != package_name:
                                    violations.append((package_name, imported_package, allowed_deps))

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imported_package = self._extract_package_name(node.module)
                            if imported_package and imported_package.startswith('morag_'):
                                if imported_package not in allowed_deps and imported_package != package_name:
                                    violations.append((package_name, imported_package, allowed_deps))

            except Exception:
                # Skip files that can't be parsed
                continue

        return violations

    def _extract_package_name(self, module_name: str) -> str:
        """Extract the top-level package name from a module path."""
        parts = module_name.split('.')
        if parts and parts[0].startswith('morag_'):
            return parts[0]
        return ""

    def _check_import_consistency(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Check for import consistency issues in a file."""
        inconsistencies = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                # Check for mixed import styles
                if 'from morag.' in line and any(pattern in line for pattern in ['morag_core', 'morag_services']):
                    inconsistencies.append((
                        str(file_path),
                        line_num,
                        "Mixed import styles in same file"
                    ))

        except Exception:
            pass

        return inconsistencies

    def _build_import_graph(self) -> Dict[str, Set[str]]:
        """Build a graph of imports between packages."""
        import_graph = {}

        for package_name in self.DEPENDENCY_RULES.keys():
            import_graph[package_name] = set()

            package_dir = Path("packages") / package_name.replace('_', '-')
            if not package_dir.exists():
                continue

            for py_file in package_dir.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imported_package = self._extract_package_name(alias.name)
                                    if imported_package and imported_package != package_name:
                                        import_graph[package_name].add(imported_package)

                            elif isinstance(node, ast.ImportFrom) and node.module:
                                imported_package = self._extract_package_name(node.module)
                                if imported_package and imported_package != package_name:
                                    import_graph[package_name].add(imported_package)

                except Exception:
                    continue

        return import_graph

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find cycles in the import graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor in graph:  # Only follow edges to known packages
                    dfs(neighbor, path.copy())

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _check_package_isolation(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Check that packages don't access internal modules of other packages."""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                # Check for imports that access internal modules
                # e.g., from morag_audio.internal.something import ...
                if re.search(r'from morag_\w+\.\w+\.\w+', line):
                    # This might be accessing internal modules
                    violations.append((
                        str(file_path),
                        line_num,
                        f"Possible internal module access: {line.strip()}"
                    ))

        except Exception:
            pass

        return violations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
