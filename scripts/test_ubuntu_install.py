#!/usr/bin/env python3
"""Test script to validate Ubuntu installation script."""

import sys
import subprocess
import os
import platform
from pathlib import Path

def check_ubuntu():
    """Check if running on Ubuntu."""
    try:
        with open('/etc/os-release', 'r') as f:
            content = f.read()
        return 'Ubuntu' in content
    except:
        return False

def check_script_exists():
    """Check if Ubuntu install script exists."""
    script_path = Path(__file__).parent.parent / "ubuntu-install.sh"
    return script_path.exists(), script_path

def check_script_executable():
    """Check if script is executable."""
    script_exists, script_path = check_script_exists()
    if not script_exists:
        return False
    return os.access(script_path, os.X_OK)

def check_script_content():
    """Check script content for key components."""
    script_exists, script_path = check_script_exists()
    if not script_exists:
        return False, []
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        required_functions = [
            "check_ubuntu_version",
            "install_build_tools", 
            "install_system_dependencies",
            "install_media_dependencies",
            "install_ocr_dependencies",
            "install_web_dependencies",
            "install_python",
            "install_redis",
            "install_gpu_support",
            "install_whisper_alternatives",
            "install_morag_dependencies",
            "install_qdrant",
            "create_environment_config",
            "test_installation"
        ]
        
        found_functions = []
        missing_functions = []
        
        for func in required_functions:
            if func in content:
                found_functions.append(func)
            else:
                missing_functions.append(func)
        
        return len(missing_functions) == 0, {
            'found': found_functions,
            'missing': missing_functions,
            'total': len(required_functions)
        }
        
    except Exception as e:
        return False, {'error': str(e)}

def check_whisper_backends():
    """Check if whisper backend alternatives are included."""
    script_exists, script_path = check_script_exists()
    if not script_exists:
        return False, []
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        whisper_packages = [
            "faster-whisper",
            "vosk", 
            "whispercpp",
            "pywhispercpp",
            "whisper-cpp-python",
            "openai-whisper"
        ]
        
        found_packages = []
        for package in whisper_packages:
            if package in content:
                found_packages.append(package)
        
        return len(found_packages) >= 4, found_packages  # At least 4 alternatives
        
    except Exception as e:
        return False, {'error': str(e)}

def check_ubuntu_specific_features():
    """Check Ubuntu-specific features."""
    script_exists, script_path = check_script_exists()
    if not script_exists:
        return False, []
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        ubuntu_features = [
            "apt update",
            "apt install",
            "systemctl",
            "ubuntu-drivers",
            "lsb_release",
            "CUDA",
            "nvidia-smi",
            "docker-compose"
        ]
        
        found_features = []
        for feature in ubuntu_features:
            if feature in content:
                found_features.append(feature)
        
        return len(found_features) >= 6, found_features
        
    except Exception as e:
        return False, {'error': str(e)}

def simulate_dry_run():
    """Simulate a dry run of the script (check syntax)."""
    script_exists, script_path = check_script_exists()
    if not script_exists:
        return False, "Script not found"
    
    try:
        # Check bash syntax
        result = subprocess.run(['bash', '-n', str(script_path)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, "Syntax check passed"
        else:
            return False, f"Syntax errors: {result.stderr}"
            
    except Exception as e:
        return False, f"Error checking syntax: {e}"

def main():
    """Main test function."""
    print("🐧 UBUNTU INSTALL SCRIPT VALIDATION")
    print("=" * 50)
    
    # Test 1: Check if running on Ubuntu
    is_ubuntu = check_ubuntu()
    print(f"Running on Ubuntu: {'✅' if is_ubuntu else '❌'}")
    if not is_ubuntu:
        print("  Note: This test can run on non-Ubuntu systems for validation")
    
    # Test 2: Check script exists
    script_exists, script_path = check_script_exists()
    print(f"Script exists: {'✅' if script_exists else '❌'}")
    if script_exists:
        print(f"  Path: {script_path}")
    else:
        print("  ❌ ubuntu-install.sh not found")
        return False
    
    # Test 3: Check script is executable
    is_executable = check_script_executable()
    print(f"Script executable: {'✅' if is_executable else '❌'}")
    if not is_executable:
        print("  Run: chmod +x ubuntu-install.sh")
    
    # Test 4: Check script content
    content_ok, content_info = check_script_content()
    print(f"Required functions: {'✅' if content_ok else '❌'}")
    if isinstance(content_info, dict) and 'found' in content_info:
        print(f"  Found: {len(content_info['found'])}/{content_info['total']} functions")
        if content_info['missing']:
            print(f"  Missing: {', '.join(content_info['missing'])}")
    elif 'error' in content_info:
        print(f"  Error: {content_info['error']}")
    
    # Test 5: Check whisper backends
    whisper_ok, whisper_packages = check_whisper_backends()
    print(f"Whisper alternatives: {'✅' if whisper_ok else '❌'}")
    if isinstance(whisper_packages, list):
        print(f"  Found packages: {', '.join(whisper_packages)}")
    elif isinstance(whisper_packages, dict) and 'error' in whisper_packages:
        print(f"  Error: {whisper_packages['error']}")
    
    # Test 6: Check Ubuntu-specific features
    ubuntu_ok, ubuntu_features = check_ubuntu_specific_features()
    print(f"Ubuntu features: {'✅' if ubuntu_ok else '❌'}")
    if isinstance(ubuntu_features, list):
        print(f"  Found features: {', '.join(ubuntu_features)}")
    elif isinstance(ubuntu_features, dict) and 'error' in ubuntu_features:
        print(f"  Error: {ubuntu_features['error']}")
    
    # Test 7: Syntax check
    syntax_ok, syntax_msg = simulate_dry_run()
    print(f"Syntax check: {'✅' if syntax_ok else '❌'}")
    print(f"  {syntax_msg}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    tests = [script_exists, content_ok, whisper_ok, ubuntu_ok, syntax_ok]
    passed = sum(tests)
    total = len(tests)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("🎉 Ubuntu install script is ready to use!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("Some issues need to be addressed")
    
    print("\nTo use the script:")
    print("1. Make executable: chmod +x ubuntu-install.sh")
    print("2. Run: ./ubuntu-install.sh")
    print("3. Follow the prompts for Qdrant installation")
    
    print("\nFeatures included:")
    print("- ✅ Full Ubuntu package management (apt)")
    print("- ✅ GPU support detection and CUDA installation")
    print("- ✅ Multiple whisper backend alternatives")
    print("- ✅ Optional local Qdrant installation with Docker")
    print("- ✅ Comprehensive dependency installation")
    print("- ✅ Environment configuration for Ubuntu")
    print("- ✅ Installation testing and validation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
