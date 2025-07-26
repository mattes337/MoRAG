#!/usr/bin/env python3
"""
OpenIE Configuration Validation Script

This script validates the OpenIE configuration and dependencies in MoRAG.
It checks for proper configuration, required dependencies, and system readiness.

Usage:
    python cli/validate-openie-config.py [options]

Examples:
    # Basic validation
    python cli/validate-openie-config.py

    # Verbose validation with detailed output
    python cli/validate-openie-config.py --verbose

    # Check specific components
    python cli/validate-openie-config.py --check config
    python cli/validate-openie-config.py --check dependencies
    python cli/validate-openie-config.py --check services

    # Fix common issues
    python cli/validate-openie-config.py --fix
"""

import argparse
import logging
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(test_name: str, success: bool, details: str = "", fix_suggestion: str = ""):
    """Print validation result with formatting."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"[{status}] {test_name}")
    if details:
        print(f"    {details}")
    if not success and fix_suggestion:
        print(f"    💡 Fix: {fix_suggestion}")


def validate_configuration() -> Tuple[bool, List[str]]:
    """Validate OpenIE configuration."""
    print_section("OpenIE Configuration Validation")
    
    issues = []
    all_passed = True
    
    try:
        from morag_core.config import get_settings
        
        settings = get_settings()
        
        # Check required configuration fields
        config_checks = [
            ("OpenIE Enabled", hasattr(settings, 'openie_enabled'), 
             f"Current: {getattr(settings, 'openie_enabled', 'NOT_SET')}", 
             "Set MORAG_OPENIE_ENABLED=true in environment"),
            
            ("OpenIE Implementation", hasattr(settings, 'openie_implementation'), 
             f"Current: {getattr(settings, 'openie_implementation', 'NOT_SET')}", 
             "Set MORAG_OPENIE_IMPLEMENTATION=stanford in environment"),
            
            ("Confidence Threshold", hasattr(settings, 'openie_confidence_threshold'), 
             f"Current: {getattr(settings, 'openie_confidence_threshold', 'NOT_SET')}", 
             "Set MORAG_OPENIE_CONFIDENCE_THRESHOLD=0.7 in environment"),
            
            ("Max Triplets", hasattr(settings, 'openie_max_triplets_per_sentence'), 
             f"Current: {getattr(settings, 'openie_max_triplets_per_sentence', 'NOT_SET')}", 
             "Set MORAG_OPENIE_MAX_TRIPLETS_PER_SENTENCE=10 in environment"),
            
            ("Entity Linking", hasattr(settings, 'openie_enable_entity_linking'), 
             f"Current: {getattr(settings, 'openie_enable_entity_linking', 'NOT_SET')}", 
             "Set MORAG_OPENIE_ENABLE_ENTITY_LINKING=true in environment"),
            
            ("Predicate Normalization", hasattr(settings, 'openie_enable_predicate_normalization'), 
             f"Current: {getattr(settings, 'openie_enable_predicate_normalization', 'NOT_SET')}", 
             "Set MORAG_OPENIE_ENABLE_PREDICATE_NORMALIZATION=true in environment"),
            
            ("Batch Size", hasattr(settings, 'openie_batch_size'), 
             f"Current: {getattr(settings, 'openie_batch_size', 'NOT_SET')}", 
             "Set MORAG_OPENIE_BATCH_SIZE=100 in environment"),
            
            ("Timeout", hasattr(settings, 'openie_timeout_seconds'), 
             f"Current: {getattr(settings, 'openie_timeout_seconds', 'NOT_SET')}", 
             "Set MORAG_OPENIE_TIMEOUT_SECONDS=30 in environment"),
        ]
        
        for test_name, condition, details, fix_suggestion in config_checks:
            print_result(test_name, condition, details, fix_suggestion)
            if not condition:
                all_passed = False
                issues.append(f"Missing configuration: {test_name}")
        
        # Validate configuration values
        if hasattr(settings, 'openie_enabled') and settings.openie_enabled:
            if hasattr(settings, 'openie_confidence_threshold'):
                threshold_valid = 0.0 <= settings.openie_confidence_threshold <= 1.0
                print_result("Confidence Threshold Range", threshold_valid, 
                           f"Value: {settings.openie_confidence_threshold}", 
                           "Set value between 0.0 and 1.0")
                if not threshold_valid:
                    all_passed = False
                    issues.append("Invalid confidence threshold range")
            
            if hasattr(settings, 'openie_implementation'):
                impl_valid = settings.openie_implementation in ["stanford", "openie5"]
                print_result("Implementation Valid", impl_valid, 
                           f"Value: {settings.openie_implementation}", 
                           "Use 'stanford' or 'openie5'")
                if not impl_valid:
                    all_passed = False
                    issues.append("Invalid OpenIE implementation")
        
        return all_passed, issues
        
    except ImportError as e:
        print_result("Configuration Import", False, f"Import error: {e}", 
                    "Install morag-core package")
        return False, [f"Configuration import failed: {e}"]
    except Exception as e:
        print_result("Configuration Validation", False, f"Error: {e}", 
                    "Check configuration setup")
        return False, [f"Configuration validation failed: {e}"]


def validate_dependencies() -> Tuple[bool, List[str]]:
    """Validate OpenIE dependencies."""
    print_section("OpenIE Dependencies Validation")
    
    issues = []
    all_passed = True
    
    # Check Python packages
    python_deps = [
        ("stanford-openie", "openie", "pip install stanford-openie"),
        ("nltk", "nltk", "pip install nltk"),
        ("structlog", "structlog", "pip install structlog"),
    ]
    
    for dep_name, import_name, install_cmd in python_deps:
        try:
            __import__(import_name)
            print_result(f"Python Package: {dep_name}", True, "Available")
        except ImportError as e:
            print_result(f"Python Package: {dep_name}", False, f"Import error: {e}", install_cmd)
            all_passed = False
            issues.append(f"Missing Python package: {dep_name}")
    
    # Check NLTK data
    try:
        import nltk
        
        nltk_data_checks = [
            ("punkt", "tokenizers/punkt"),
            ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ]
        
        for data_name, data_path in nltk_data_checks:
            try:
                nltk.data.find(data_path)
                print_result(f"NLTK Data: {data_name}", True, "Available")
            except LookupError:
                print_result(f"NLTK Data: {data_name}", False, "Not found", 
                           f"Run: python -c \"import nltk; nltk.download('{data_name}')\"")
                all_passed = False
                issues.append(f"Missing NLTK data: {data_name}")
    
    except ImportError:
        print_result("NLTK Data Check", False, "NLTK not available", "Install NLTK first")
        all_passed = False
        issues.append("NLTK not available for data check")
    
    # Check Java (required for Stanford OpenIE)
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            java_version = result.stderr.split('\n')[0] if result.stderr else "Unknown"
            print_result("Java Runtime", True, f"Version: {java_version}")
        else:
            print_result("Java Runtime", False, "Java not working", 
                        "Install Java 8 or higher")
            all_passed = False
            issues.append("Java runtime not available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_result("Java Runtime", False, "Java not found", 
                    "Install Java 8 or higher and add to PATH")
        all_passed = False
        issues.append("Java not found in PATH")
    except Exception as e:
        print_result("Java Runtime", False, f"Error checking Java: {e}", 
                    "Check Java installation")
        all_passed = False
        issues.append(f"Java check failed: {e}")
    
    return all_passed, issues


def validate_services() -> Tuple[bool, List[str]]:
    """Validate OpenIE services."""
    print_section("OpenIE Services Validation")
    
    issues = []
    all_passed = True
    
    try:
        from morag_graph.services.openie_service import OpenIEService
        
        # Test service initialization
        try:
            service = OpenIEService()
            print_result("Service Initialization", True, "OpenIE service created")
            
            # Check service configuration
            config_ok = (
                hasattr(service, 'enabled') and
                hasattr(service, 'implementation') and
                hasattr(service, 'confidence_threshold')
            )
            print_result("Service Configuration", config_ok, 
                        f"Enabled: {getattr(service, 'enabled', 'Unknown')}")
            
            if not config_ok:
                all_passed = False
                issues.append("Service configuration incomplete")
            
        except Exception as e:
            print_result("Service Initialization", False, f"Error: {e}", 
                        "Check OpenIE configuration and dependencies")
            all_passed = False
            issues.append(f"Service initialization failed: {e}")
        
        # Test extractor
        try:
            from morag_graph.extractors.openie_extractor import OpenIEExtractor
            
            extractor = OpenIEExtractor({'min_confidence': 0.6})
            print_result("Extractor Initialization", True, "OpenIE extractor created")
            
            extractor_ok = (
                hasattr(extractor, 'enabled') and
                hasattr(extractor, 'openie_service') and
                hasattr(extractor, 'extract_relations')
            )
            print_result("Extractor Configuration", extractor_ok, 
                        f"Enabled: {getattr(extractor, 'enabled', 'Unknown')}")
            
            if not extractor_ok:
                all_passed = False
                issues.append("Extractor configuration incomplete")
            
        except Exception as e:
            print_result("Extractor Initialization", False, f"Error: {e}", 
                        "Check OpenIE extractor implementation")
            all_passed = False
            issues.append(f"Extractor initialization failed: {e}")
        
        # Test enhanced graph builder
        try:
            from morag_graph.builders.enhanced_graph_builder import EnhancedGraphBuilder
            
            builder = EnhancedGraphBuilder(
                storage=None,  # Mock for testing
                enable_openie=True,
                openie_config={'min_confidence': 0.6}
            )
            print_result("Enhanced Graph Builder", True, "Builder created with OpenIE")
            
            builder_ok = (
                hasattr(builder, 'openie_enabled') and
                hasattr(builder, 'openie_extractor')
            )
            print_result("Builder OpenIE Integration", builder_ok, 
                        f"OpenIE enabled: {getattr(builder, 'openie_enabled', 'Unknown')}")
            
            if not builder_ok:
                all_passed = False
                issues.append("Enhanced graph builder OpenIE integration incomplete")
            
        except Exception as e:
            print_result("Enhanced Graph Builder", False, f"Error: {e}", 
                        "Check enhanced graph builder implementation")
            all_passed = False
            issues.append(f"Enhanced graph builder failed: {e}")
        
        return all_passed, issues
        
    except ImportError as e:
        print_result("Services Import", False, f"Import error: {e}", 
                    "Install morag-graph package")
        return False, [f"Services import failed: {e}"]
    except Exception as e:
        print_result("Services Validation", False, f"Error: {e}", 
                    "Check services implementation")
        return False, [f"Services validation failed: {e}"]


def fix_common_issues():
    """Attempt to fix common OpenIE configuration issues."""
    print_section("Fixing Common OpenIE Issues")
    
    fixes_applied = []
    
    # Download NLTK data
    try:
        import nltk
        
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print_result("NLTK Data Download", True, "Downloaded punkt and pos_tag data")
        fixes_applied.append("Downloaded NLTK data")
        
    except Exception as e:
        print_result("NLTK Data Download", False, f"Error: {e}")
    
    # Create sample .env file
    try:
        env_file = Path(".env.openie.sample")
        env_content = """# OpenIE Configuration for MoRAG
MORAG_OPENIE_ENABLED=true
MORAG_OPENIE_IMPLEMENTATION=stanford
MORAG_OPENIE_CONFIDENCE_THRESHOLD=0.7
MORAG_OPENIE_MAX_TRIPLETS_PER_SENTENCE=10
MORAG_OPENIE_ENABLE_ENTITY_LINKING=true
MORAG_OPENIE_ENABLE_PREDICATE_NORMALIZATION=true
MORAG_OPENIE_BATCH_SIZE=100
MORAG_OPENIE_TIMEOUT_SECONDS=30

# Graph Processing with OpenIE
MORAG_GRAPH_ENABLED=true
MORAG_GRAPH_LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j Configuration (required for graph storage)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=neo4j
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print_result("Sample Environment File", True, f"Created {env_file}")
        print(f"    💡 Copy {env_file} to .env and update with your values")
        fixes_applied.append(f"Created sample environment file: {env_file}")
        
    except Exception as e:
        print_result("Sample Environment File", False, f"Error: {e}")
    
    return fixes_applied


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate OpenIE configuration and dependencies")
    parser.add_argument("--check", choices=["config", "dependencies", "services", "all"],
                       default="all", help="What to check")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🔍 MoRAG OpenIE Configuration Validation")
    print("=" * 60)
    
    all_results = {}
    all_issues = []
    
    if args.check in ["config", "all"]:
        config_passed, config_issues = validate_configuration()
        all_results["configuration"] = config_passed
        all_issues.extend(config_issues)
    
    if args.check in ["dependencies", "all"]:
        deps_passed, deps_issues = validate_dependencies()
        all_results["dependencies"] = deps_passed
        all_issues.extend(deps_issues)
    
    if args.check in ["services", "all"]:
        services_passed, services_issues = validate_services()
        all_results["services"] = services_passed
        all_issues.extend(services_issues)
    
    if args.fix:
        fixes = fix_common_issues()
        if fixes:
            print(f"\n✅ Applied {len(fixes)} fixes:")
            for fix in fixes:
                print(f"  - {fix}")
    
    # Summary
    print_section("Validation Summary")
    total_checks = len(all_results)
    passed_checks = sum(1 for result in all_results.values() if result)
    
    for check_name, result in all_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {check_name.title()}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
    
    if all_issues:
        print(f"\n⚠️  Issues found:")
        for issue in all_issues:
            print(f"  - {issue}")
    
    if passed_checks == total_checks:
        print("\n🎉 All OpenIE validation checks passed!")
        print("OpenIE is properly configured and ready to use.")
        return 0
    else:
        print("\n⚠️  Some OpenIE validation checks failed.")
        print("Please address the issues above before using OpenIE.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
