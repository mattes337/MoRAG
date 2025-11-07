#!/usr/bin/env python3
"""
Build-time check script that runs comprehensive static analysis.
This script should be run before commits and in CI/CD pipelines.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_command(cmd: List[str], cwd: Path = None) -> Dict[str, Any]:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd or Path.cwd()
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command not found: {cmd[0]}",
            "returncode": 127,
        }


def check_python_syntax(project_root: Path) -> Dict[str, Any]:
    """Check Python syntax using py_compile."""
    print("🔍 Checking Python syntax...")

    python_files = []
    for root, dirs, files in project_root.rglob("*.py"):
        if not any(
            exclude in str(root) for exclude in [".git", "__pycache__", ".venv", "venv"]
        ):
            python_files.append(root)

    errors = []
    for py_file in python_files:
        result = run_command([sys.executable, "-m", "py_compile", str(py_file)])
        if not result["success"]:
            errors.append(f"Syntax error in {py_file}: {result['stderr']}")

    return {
        "name": "Python Syntax Check",
        "success": len(errors) == 0,
        "errors": errors,
        "files_checked": len(python_files),
    }


def run_import_check(project_root: Path) -> Dict[str, Any]:
    """Run our custom import checker."""
    print("📦 Checking imports...")

    script_path = project_root / "scripts" / "check_imports.py"
    if not script_path.exists():
        return {
            "name": "Import Check",
            "success": False,
            "errors": ["Import checker script not found"],
            "warnings": [],
        }

    result = run_command(
        [sys.executable, str(script_path), "packages", "--exit-on-error"],
        cwd=project_root,
    )

    errors = []
    warnings = []

    if result["stdout"]:
        lines = result["stdout"].split("\n")
        in_errors = False
        in_warnings = False

        for line in lines:
            line = line.strip()
            if line == "ERRORS:":
                in_errors = True
                in_warnings = False
            elif line == "WARNINGS:":
                in_errors = False
                in_warnings = True
            elif line.startswith("Found "):
                in_errors = False
                in_warnings = False
            elif line and in_errors:
                errors.append(line)
            elif line and in_warnings:
                warnings.append(line)

    return {
        "name": "Import Check",
        "success": result["success"],
        "errors": errors,
        "warnings": warnings,
    }


def run_static_analysis(project_root: Path) -> Dict[str, Any]:
    """Run comprehensive static analysis."""
    print("🔬 Running static analysis...")

    script_path = project_root / "scripts" / "static_analysis.py"
    if not script_path.exists():
        return {
            "name": "Static Analysis",
            "success": False,
            "errors": ["Static analysis script not found"],
            "warnings": [],
        }

    result = run_command(
        [sys.executable, str(script_path), "packages", "--json", "--exit-on-error"],
        cwd=project_root,
    )

    if result["success"] and result["stdout"]:
        try:
            analysis_results = json.loads(result["stdout"])
            return {
                "name": "Static Analysis",
                "success": analysis_results["summary"]["total_errors"] == 0,
                "errors": [
                    error
                    for file_result in analysis_results["file_results"]
                    for error in file_result["errors"]
                ],
                "warnings": [
                    warning
                    for file_result in analysis_results["file_results"]
                    for warning in file_result["warnings"]
                ],
                "summary": analysis_results["summary"],
            }
        except json.JSONDecodeError:
            return {
                "name": "Static Analysis",
                "success": False,
                "errors": ["Failed to parse static analysis results"],
                "warnings": [],
            }
    else:
        return {
            "name": "Static Analysis",
            "success": result["success"],
            "errors": [result["stderr"]] if result["stderr"] else [],
            "warnings": [],
        }


def run_tests(project_root: Path) -> Dict[str, Any]:
    """Run tests if available."""
    print("🧪 Running tests...")

    # Check if pytest is available and there are tests
    test_dirs = [
        project_root / "tests",
        project_root / "test",
        project_root / "packages" / "morag-stages" / "tests",
    ]

    test_dir = None
    for td in test_dirs:
        if td.exists() and any(td.glob("test_*.py")):
            test_dir = td
            break

    if not test_dir:
        return {
            "name": "Tests",
            "success": True,
            "errors": [],
            "warnings": ["No tests found"],
            "skipped": True,
        }

    # Try to run pytest
    result = run_command(
        [sys.executable, "-m", "pytest", str(test_dir), "-v"], cwd=project_root
    )

    return {
        "name": "Tests",
        "success": result["success"],
        "errors": [result["stderr"]]
        if not result["success"] and result["stderr"]
        else [],
        "warnings": [],
        "output": result["stdout"],
    }


def run_build_checks(project_root: Path, skip_tests: bool = False) -> Dict[str, Any]:
    """Run all build checks."""
    print("🚀 Starting build checks...")
    print("=" * 50)

    checks = [
        check_python_syntax,
        run_import_check,
        run_static_analysis,
    ]

    if not skip_tests:
        checks.append(run_tests)

    results = []
    overall_success = True

    for check_func in checks:
        try:
            result = check_func(project_root)
            results.append(result)

            # Print immediate feedback
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['name']}")

            if not result["success"]:
                overall_success = False
                if result.get("errors"):
                    for error in result["errors"][:3]:  # Show first 3 errors
                        print(f"   Error: {error}")
                    if len(result["errors"]) > 3:
                        print(f"   ... and {len(result['errors']) - 3} more errors")

            if result.get("warnings"):
                print(f"   {len(result['warnings'])} warnings")

        except Exception as e:
            print(f"❌ {check_func.__name__} failed: {e}")
            overall_success = False
            results.append(
                {
                    "name": check_func.__name__,
                    "success": False,
                    "errors": [str(e)],
                    "warnings": [],
                }
            )

    print("=" * 50)

    return {
        "overall_success": overall_success,
        "checks": results,
        "summary": {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "total_errors": sum(len(r.get("errors", [])) for r in results),
            "total_warnings": sum(len(r.get("warnings", [])) for r in results),
        },
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run build-time checks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory"
    )

    args = parser.parse_args()

    results = run_build_checks(args.project_root, args.skip_tests)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"Build Check Summary:")
        print(f"  Total checks: {results['summary']['total_checks']}")
        print(f"  Passed: {results['summary']['passed']}")
        print(f"  Failed: {results['summary']['failed']}")
        print(f"  Total errors: {results['summary']['total_errors']}")
        print(f"  Total warnings: {results['summary']['total_warnings']}")

        if results["overall_success"]:
            print("\n🎉 All checks passed!")
        else:
            print("\n💥 Some checks failed!")
            print("\nFailed checks:")
            for check in results["checks"]:
                if not check["success"]:
                    print(f"  - {check['name']}")
                    for error in check.get("errors", []):
                        print(f"    {error}")

    sys.exit(0 if results["overall_success"] else 1)


if __name__ == "__main__":
    main()
