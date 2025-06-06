#!/usr/bin/env python3
"""
Simple validation script to check that collection name unification is working.
"""

import os
import re


def check_file_content(filepath, pattern, description):
    """Check if a file contains a specific pattern."""
    if not os.path.exists(filepath):
        print(f"⏭️  {filepath} not found, skipping")
        return True
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if re.search(pattern, content):
            print(f"✅ {filepath}: {description}")
            return True
        else:
            print(f"❌ {filepath}: {description} - PATTERN NOT FOUND")
            return False
    except Exception as e:
        print(f"❌ {filepath}: Error reading file - {e}")
        return False


def check_no_default_values():
    """Check that no files contain default collection name values."""
    print("Checking for removal of default collection name values...")
    
    files_to_check = [
        {
            'path': 'packages/morag-core/src/morag_core/config.py',
            'bad_pattern': r'qdrant_collection_name:\s*str\s*=\s*["\']',
            'good_pattern': r'qdrant_collection_name:\s*Optional\[str\]\s*=\s*None',
            'description': 'No default collection name in core config'
        },
        {
            'path': 'packages/morag-services/src/morag_services/storage.py',
            'bad_pattern': r'collection_name:\s*str\s*=\s*["\']',
            'good_pattern': r'collection_name:\s*Optional\[str\]\s*=\s*None',
            'description': 'No default collection name in storage'
        },
        {
            'path': 'packages/morag-services/src/morag_services/services.py',
            'bad_pattern': r'os\.getenv\(["\']QDRANT_COLLECTION_NAME["\'],\s*["\'][^"\']*["\']\)',
            'good_pattern': r'os\.getenv\(["\']QDRANT_COLLECTION_NAME["\']\)',
            'description': 'No default collection name in services'
        },
        {
            'path': 'packages/morag/src/morag/ingest_tasks.py',
            'bad_pattern': r'os\.getenv\(["\']QDRANT_COLLECTION_NAME["\'],\s*["\'][^"\']*["\']\)',
            'good_pattern': r'os\.getenv\(["\']QDRANT_COLLECTION_NAME["\']\)',
            'description': 'No default collection name in ingest tasks'
        }
    ]
    
    all_good = True
    
    for file_check in files_to_check:
        filepath = file_check['path']
        if not os.path.exists(filepath):
            print(f"⏭️  {filepath} not found, skipping")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for bad patterns (default values)
            if re.search(file_check['bad_pattern'], content):
                print(f"❌ {filepath}: Still contains default collection name")
                all_good = False
            # Check for good patterns (no defaults)
            elif re.search(file_check['good_pattern'], content):
                print(f"✅ {filepath}: {file_check['description']}")
            else:
                print(f"⚠️  {filepath}: Pattern not found, manual check needed")
                
        except Exception as e:
            print(f"❌ {filepath}: Error reading file - {e}")
            all_good = False
    
    return all_good


def check_validation_logic():
    """Check that validation logic is present."""
    print("\nChecking for validation logic...")
    
    validation_checks = [
        {
            'path': 'packages/morag-core/src/morag_core/config.py',
            'pattern': r'field_validator.*qdrant_collection_name',
            'description': 'Core config has collection name validation'
        },
        {
            'path': 'packages/morag-services/src/morag_services/storage.py',
            'pattern': r'if not collection_name:\s*raise ValueError\("collection_name is required',
            'description': 'Storage has collection name validation'
        },
        {
            'path': 'packages/morag-services/src/morag_services/services.py',
            'pattern': r'if not collection_name:\s*raise ValueError\("QDRANT_COLLECTION_NAME environment variable is required"\)',
            'description': 'Services has collection name validation'
        },
        {
            'path': 'packages/morag/src/morag/ingest_tasks.py',
            'pattern': r'if not collection_name_env:\s*raise ValueError\("QDRANT_COLLECTION_NAME environment variable is required"\)',
            'description': 'Ingest tasks has collection name validation'
        }
    ]
    
    all_good = True
    
    for check in validation_checks:
        if not check_file_content(check['path'], check['pattern'], check['description']):
            all_good = False
    
    return all_good


def check_environment_consistency():
    """Check that environment files use consistent collection names."""
    print("\nChecking environment file consistency...")
    
    env_files = [
        '.env.example',
        '.env.prod.example'
    ]
    
    all_good = True
    
    for env_file in env_files:
        if not check_file_content(
            env_file, 
            r'QDRANT_COLLECTION_NAME=morag_documents',
            'Uses morag_documents as collection name'
        ):
            all_good = False
    
    return all_good


def check_documentation_updates():
    """Check that documentation has been updated."""
    print("\nChecking documentation updates...")
    
    doc_checks = [
        {
            'path': 'packages/morag/README.md',
            'pattern': r'morag_documents',
            'description': 'README uses morag_documents'
        },
        {
            'path': 'scripts/install_morag.py',
            'pattern': r'QDRANT_COLLECTION_NAME=morag_documents',
            'description': 'Install script uses morag_documents'
        }
    ]
    
    all_good = True
    
    for check in doc_checks:
        if not check_file_content(check['path'], check['pattern'], check['description']):
            all_good = False
    
    return all_good


def main():
    """Run all validation checks."""
    print("Validating Qdrant Collection Name Unification")
    print("=" * 50)
    
    checks = [
        ("Default Values Removal", check_no_default_values),
        ("Validation Logic", check_validation_logic),
        ("Environment Consistency", check_environment_consistency),
        ("Documentation Updates", check_documentation_updates)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name))
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Collection name unification is complete!")
        print("\nSummary of changes:")
        print("• Removed all default collection name values")
        print("• Added fail-fast validation in all components")
        print("• Unified environment files to use 'morag_documents'")
        print("• Updated documentation and examples")
        print("\nTo use the system, set: QDRANT_COLLECTION_NAME=morag_documents")
    else:
        print("❌ SOME CHECKS FAILED - Please review the issues above")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
