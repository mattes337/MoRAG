#!/usr/bin/env python3
"""Script to toggle MoRAG mock mode on/off."""

import argparse
import os
import sys
from pathlib import Path


def read_env_file(env_path: Path) -> dict:
    """Read environment variables from .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def write_env_file(env_path: Path, env_vars: dict):
    """Write environment variables to .env file."""
    with open(env_path, "w", encoding="utf-8") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")


def update_mock_mode(enable: bool, mock_data_dir: str = "./mock"):
    """Update mock mode settings in .env file."""
    env_path = Path(".env")

    if not env_path.exists():
        print("❌ .env file not found. Please create one first.")
        return False

    # Read current environment variables
    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Update mock mode settings
    mock_mode_found = False
    mock_dir_found = False

    updated_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("MORAG_MOCK_MODE="):
            updated_lines.append(f"MORAG_MOCK_MODE={'true' if enable else 'false'}\n")
            mock_mode_found = True
        elif stripped.startswith("MORAG_MOCK_DATA_DIR="):
            updated_lines.append(f"MORAG_MOCK_DATA_DIR={mock_data_dir}\n")
            mock_dir_found = True
        else:
            updated_lines.append(line)

    # Add missing variables if not found
    if not mock_mode_found:
        # Find a good place to insert (after processing configuration)
        insert_index = len(updated_lines)
        for i, line in enumerate(updated_lines):
            if "Processing Configuration" in line:
                # Find the end of this section
                for j in range(i + 1, len(updated_lines)):
                    if (
                        updated_lines[j].strip().startswith("#")
                        and "=" not in updated_lines[j]
                    ):
                        insert_index = j
                        break
                break

        updated_lines.insert(insert_index, f"\n# Mock Mode Configuration\n")
        updated_lines.insert(
            insert_index + 1, f"MORAG_MOCK_MODE={'true' if enable else 'false'}\n"
        )
        if not mock_dir_found:
            updated_lines.insert(
                insert_index + 2, f"MORAG_MOCK_DATA_DIR={mock_data_dir}\n"
            )
    elif not mock_dir_found:
        # Add mock data dir after mock mode
        for i, line in enumerate(updated_lines):
            if line.strip().startswith("MORAG_MOCK_MODE="):
                updated_lines.insert(i + 1, f"MORAG_MOCK_DATA_DIR={mock_data_dir}\n")
                break

    # Write updated file
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    return True


def check_mock_data(mock_data_dir: str):
    """Check if mock data directory and files exist."""
    mock_path = Path(mock_data_dir)

    if not mock_path.exists():
        print(f"⚠ Mock data directory not found: {mock_path}")
        print("  Run the following to create mock data:")
        print(f"  mkdir -p {mock_path}")
        print("  # Copy mock files from the repository")
        return False

    required_files = [
        "markdown-conversion/sample.md",
        "markdown-optimizer/sample.optimized.md",
        "chunker/sample.chunks.json",
        "fact-generator/sample.facts.json",
        "ingestor/sample.ingestion.json",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = mock_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"⚠ Missing {len(missing_files)} mock files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    print(f"✓ All mock files present in {mock_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Toggle MoRAG mock mode")
    parser.add_argument(
        "action",
        choices=["on", "off", "status"],
        help="Action to perform: on (enable), off (disable), or status (check current state)",
    )
    parser.add_argument(
        "--mock-dir", default="./mock", help="Mock data directory (default: ./mock)"
    )

    args = parser.parse_args()

    print("MoRAG Mock Mode Toggle")
    print("=" * 30)

    if args.action == "status":
        # Check current status
        env_path = Path(".env")
        if not env_path.exists():
            print("❌ .env file not found")
            sys.exit(1)

        env_vars = read_env_file(env_path)
        mock_mode = env_vars.get("MORAG_MOCK_MODE", "false").lower() == "true"
        mock_dir = env_vars.get("MORAG_MOCK_DATA_DIR", "./mock")

        print(f"Mock mode: {'ENABLED' if mock_mode else 'DISABLED'}")
        print(f"Mock data directory: {mock_dir}")

        if mock_mode:
            check_mock_data(mock_dir)

    elif args.action == "on":
        # Enable mock mode
        print("Enabling mock mode...")

        if update_mock_mode(True, args.mock_dir):
            print("✓ Mock mode enabled in .env file")

            # Check mock data
            if check_mock_data(args.mock_dir):
                print("✓ Mock data is ready")
                print("\nMock mode is now ENABLED")
                print("Restart the MoRAG server to apply changes.")
            else:
                print("⚠ Mock mode enabled but mock data is incomplete")
        else:
            print("❌ Failed to enable mock mode")
            sys.exit(1)

    elif args.action == "off":
        # Disable mock mode
        print("Disabling mock mode...")

        if update_mock_mode(False, args.mock_dir):
            print("✓ Mock mode disabled in .env file")
            print("\nMock mode is now DISABLED")
            print("Restart the MoRAG server to apply changes.")
        else:
            print("❌ Failed to disable mock mode")
            sys.exit(1)


if __name__ == "__main__":
    main()
