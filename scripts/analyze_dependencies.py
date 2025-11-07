#!/usr/bin/env python3
"""
Comprehensive dependency analysis script for MoRAG project.
Analyzes all dependencies across pyproject.toml, requirements.txt, and package files.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_requirement(req_line: str) -> Tuple[str, str]:
    """Parse a requirement line to extract package name and version spec."""
    req_line = req_line.strip()
    if not req_line or req_line.startswith("#"):
        return None, None

    # Remove comments
    if "#" in req_line:
        req_line = req_line.split("#")[0].strip()

    # Extract package name and version
    if ">=" in req_line:
        parts = req_line.split(">=")
        name = parts[0].strip()
        version = ">=" + parts[1].strip() if len(parts) > 1 else ""
    elif "==" in req_line:
        parts = req_line.split("==")
        name = parts[0].strip()
        version = "==" + parts[1].strip() if len(parts) > 1 else ""
    elif "~=" in req_line:
        parts = req_line.split("~=")
        name = parts[0].strip()
        version = "~=" + parts[1].strip() if len(parts) > 1 else ""
    elif "<" in req_line:
        # Handle < and <=
        if "<=" in req_line:
            parts = req_line.split("<=")
            name = parts[0].strip()
            version = "<=" + parts[1].strip() if len(parts) > 1 else ""
        else:
            parts = req_line.split("<")
            name = parts[0].strip()
            version = "<" + parts[1].strip() if len(parts) > 1 else ""
    else:
        name = req_line.strip()
        version = ""

    # Clean up package name
    name = re.sub(r"\[.*\]", "", name).strip()  # Remove extras like [all]

    return name, version


def analyze_pyproject_toml():
    """Analyze dependencies in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return {}

    dependencies = {}
    current_section = None

    with open(pyproject_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Track sections
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue

        # Parse dependencies
        if current_section == "project" and line.startswith("dependencies = ["):
            # Start of main dependencies
            continue
        elif current_section == "project.optional-dependencies.dev" or (
            current_section and "optional-dependencies" in current_section
        ):
            # Optional dependencies
            if "=" in line and '"' in line:
                # Extract dependency from quoted string
                match = re.search(r'"([^"]+)"', line)
                if match:
                    dep_line = match.group(1)
                    name, version = parse_requirement(dep_line)
                    if name:
                        section_name = (
                            current_section.split(".")[-1]
                            if "." in current_section
                            else "optional"
                        )
                        dependencies[name] = {
                            "version": version,
                            "source": f"pyproject.toml ({section_name})",
                            "category": section_name,
                            "required": False,
                        }
        elif current_section == "project" and '"' in line and "=" not in line:
            # Main dependencies
            match = re.search(r'"([^"]+)"', line)
            if match:
                dep_line = match.group(1)
                name, version = parse_requirement(dep_line)
                if name:
                    dependencies[name] = {
                        "version": version,
                        "source": "pyproject.toml (main)",
                        "category": "core",
                        "required": True,
                    }

    return dependencies


def analyze_requirements_txt():
    """Analyze dependencies in requirements.txt."""
    req_path = Path("requirements.txt")
    if not req_path.exists():
        return {}

    dependencies = {}

    with open(req_path, "r") as f:
        lines = f.readlines()

    current_category = "core"

    for line in lines:
        line = line.strip()

        # Detect category from comments
        if line.startswith("# ") and any(
            keyword in line.lower()
            for keyword in [
                "optional",
                "audio",
                "video",
                "image",
                "web",
                "office",
                "pdf",
                "nlp",
            ]
        ):
            current_category = line.replace("#", "").strip().lower()
            continue

        name, version = parse_requirement(line)
        if name:
            dependencies[name] = {
                "version": version,
                "source": "requirements.txt",
                "category": current_category,
                "required": current_category == "core",
            }

    return dependencies


def analyze_package_dependencies():
    """Analyze dependencies in individual package pyproject.toml files."""
    packages_dir = Path("packages")
    if not packages_dir.exists():
        return {}

    all_deps = {}

    for package_dir in packages_dir.iterdir():
        if not package_dir.is_dir():
            continue

        pyproject_path = package_dir / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        package_name = package_dir.name

        with open(pyproject_path, "r") as f:
            content = f.read()

        # Extract dependencies using regex (simple approach)
        dep_pattern = r'"([^"]+)"'
        dependencies_section = False

        for line in content.split("\n"):
            line = line.strip()

            if "dependencies = [" in line:
                dependencies_section = True
                continue
            elif dependencies_section and line == "]":
                dependencies_section = False
                continue
            elif dependencies_section and '"' in line:
                match = re.search(dep_pattern, line)
                if match:
                    dep_line = match.group(1)
                    name, version = parse_requirement(dep_line)
                    if name:
                        key = f"{package_name}::{name}"
                        all_deps[key] = {
                            "version": version,
                            "source": f"{package_name}/pyproject.toml",
                            "category": package_name,
                            "required": True,
                            "package": package_name,
                        }

    return all_deps


def get_package_usage(package_name: str) -> List[str]:
    """Find where a package is used in the codebase."""
    usage_locations = []

    # Search for imports
    try:
        result = subprocess.run(
            ["grep", "-r", "--include=*.py", f"import {package_name}", "."],
            capture_output=True,
            text=True,
            cwd=".",
        )

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.strip():
                    usage_locations.append(line.split(":")[0])
    except:
        pass

    # Search for from imports
    try:
        result = subprocess.run(
            ["grep", "-r", "--include=*.py", f"from {package_name}", "."],
            capture_output=True,
            text=True,
            cwd=".",
        )

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.strip():
                    usage_locations.append(line.split(":")[0])
    except:
        pass

    return list(set(usage_locations))


def analyze_usage_patterns():
    """Analyze how packages are used in the codebase."""
    # Common package name mappings
    package_mappings = {
        "beautifulsoup4": "bs4",
        "python-multipart": "multipart",
        "python-dotenv": "dotenv",
        "python-docx": "docx",
        "python-pptx": "pptx",
        "opencv-python": "cv2",
        "pillow": "PIL",
        "scikit-learn": "sklearn",
        "google-genai": "google.genai",
        "faster-whisper": "faster_whisper",
        "sentence-transformers": "sentence_transformers",
        "pyannote.audio": "pyannote",
        "qdrant-client": "qdrant_client",
    }

    usage_data = {}

    # Get all unique package names from all sources
    all_deps = {}
    all_deps.update(analyze_pyproject_toml())
    all_deps.update(analyze_requirements_txt())
    all_deps.update(analyze_package_dependencies())

    unique_packages = set()
    for dep_name in all_deps.keys():
        if "::" in dep_name:
            unique_packages.add(dep_name.split("::")[1])
        else:
            unique_packages.add(dep_name)

    for package in unique_packages:
        # Get the import name
        import_name = package_mappings.get(package, package)

        # Find usage
        usage_locations = get_package_usage(import_name)

        # Also try the package name itself if different
        if import_name != package:
            usage_locations.extend(get_package_usage(package))

        usage_data[package] = {
            "import_name": import_name,
            "usage_count": len(usage_locations),
            "usage_locations": usage_locations[:10],  # Limit to first 10
        }

    return usage_data


def generate_dependency_report():
    """Generate a comprehensive dependency report."""
    print("Analyzing MoRAG Dependencies...")
    print("=" * 60)

    # Collect all dependency data
    pyproject_deps = analyze_pyproject_toml()
    requirements_deps = analyze_requirements_txt()
    package_deps = analyze_package_dependencies()
    usage_data = analyze_usage_patterns()

    # Combine all dependencies
    all_deps = {}
    all_deps.update(pyproject_deps)
    all_deps.update(requirements_deps)

    # Add package-specific dependencies
    for key, dep in package_deps.items():
        package_name = key.split("::")[0]
        dep_name = key.split("::")[1]
        if dep_name not in all_deps:
            all_deps[dep_name] = dep
        else:
            # Merge information
            all_deps[dep_name]["source"] += f", {dep['source']}"

    # Generate report
    report = {
        "summary": {
            "total_dependencies": len(all_deps),
            "core_dependencies": len(
                [d for d in all_deps.values() if d.get("required", False)]
            ),
            "optional_dependencies": len(
                [d for d in all_deps.values() if not d.get("required", False)]
            ),
            "categories": list(
                set(d.get("category", "unknown") for d in all_deps.values())
            ),
        },
        "dependencies": {},
    }

    # Analyze each dependency
    for name, dep_info in sorted(all_deps.items()):
        usage_info = usage_data.get(name, {})

        report["dependencies"][name] = {
            "version": dep_info.get("version", ""),
            "source": dep_info.get("source", ""),
            "category": dep_info.get("category", "unknown"),
            "required": dep_info.get("required", False),
            "usage_count": usage_info.get("usage_count", 0),
            "usage_locations": usage_info.get("usage_locations", []),
            "import_name": usage_info.get("import_name", name),
        }

    return report


def print_dependency_table(report):
    """Print a formatted dependency table."""
    deps = report["dependencies"]

    print(f"\nDEPENDENCY ANALYSIS REPORT")
    print(f"Total Dependencies: {report['summary']['total_dependencies']}")
    print(f"Core Dependencies: {report['summary']['core_dependencies']}")
    print(f"Optional Dependencies: {report['summary']['optional_dependencies']}")
    print(f"Categories: {', '.join(report['summary']['categories'])}")
    print("\n" + "=" * 120)

    # Table header
    print(
        f"{'Package Name':<25} {'Version':<20} {'Category':<15} {'Required':<10} {'Usage':<8} {'Source':<30}"
    )
    print("-" * 120)

    # Sort by category and usage
    sorted_deps = sorted(
        deps.items(), key=lambda x: (x[1]["category"], -x[1]["usage_count"], x[0])
    )

    for name, info in sorted_deps:
        usage_count = info["usage_count"]
        usage_indicator = f"{usage_count:>3}" if usage_count > 0 else "  0"
        required = "Yes" if info["required"] else "No"

        print(
            f"{name:<25} {info['version']:<20} {info['category']:<15} {required:<10} {usage_indicator:<8} {info['source']:<30}"
        )


def save_detailed_report(report):
    """Save detailed report to JSON file."""
    with open("dependency_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: dependency_analysis_report.json")


def main():
    """Main function."""
    try:
        report = generate_dependency_report()
        print_dependency_table(report)
        save_detailed_report(report)

        # Print recommendations
        print(f"\n{'RECOMMENDATIONS':<60}")
        print("=" * 60)

        unused_deps = [
            name
            for name, info in report["dependencies"].items()
            if info["usage_count"] == 0
        ]

        if unused_deps:
            print(f"Potentially unused dependencies ({len(unused_deps)}):")
            for dep in unused_deps[:10]:  # Show first 10
                print(f"  - {dep}")
            if len(unused_deps) > 10:
                print(f"  ... and {len(unused_deps) - 10} more")

        print(f"\nOptional dependencies that could be moved to extras:")
        optional_used = [
            name
            for name, info in report["dependencies"].items()
            if not info["required"] and info["usage_count"] > 0
        ]
        for dep in optional_used[:5]:
            print(f"  - {dep}")

    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
