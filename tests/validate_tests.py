#!/usr/bin/env python3
"""
Validation script for MoRAG test suite.

This script validates the test structure and ensures all test files
are properly formatted and importable.
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple


class TestValidator:
    """Validates test files and structure."""
    
    def __init__(self, test_root: Path):
        self.test_root = test_root
        self.validation_results = {
            "total_files": 0,
            "valid_files": 0,
            "syntax_errors": [],
            "import_errors": [],
            "structure_issues": []
        }
    
    def validate_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Validate Python syntax of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return True, "OK"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    def validate_test_structure(self, file_path: Path) -> List[str]:
        """Validate test file structure and conventions."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check for test classes and functions
            has_test_class = False
            has_test_function = False
            has_docstring = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith('Test'):
                        has_test_class = True
                        # Check if class has docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                            has_docstring = True
                
                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        has_test_function = True
            
            # Check for module docstring
            if (tree.body and 
                isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)):
                has_docstring = True
            
            # Validate structure
            if not has_test_class and not has_test_function:
                issues.append("No test classes or functions found")
            
            if not has_docstring:
                issues.append("Missing docstring")
                
        except Exception as e:
            issues.append(f"Error analyzing structure: {e}")
        
        return issues
    
    def validate_imports(self, file_path: Path) -> List[str]:
        """Validate imports in test file."""
        import_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check for pytest import
            has_pytest = False
            has_mock = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == 'pytest':
                            has_pytest = True
                        elif 'mock' in alias.name:
                            has_mock = True
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module == 'pytest':
                        has_pytest = True
                    elif node.module and 'mock' in node.module:
                        has_mock = True
            
            # Check if file needs pytest but doesn't import it
            if 'test_' in file_path.name and not has_pytest:
                import_issues.append("Missing pytest import")
                
        except Exception as e:
            import_issues.append(f"Error checking imports: {e}")
        
        return import_issues
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single test file."""
        result = {
            "file": str(file_path),
            "valid": True,
            "issues": []
        }
        
        # Validate syntax
        syntax_valid, syntax_msg = self.validate_syntax(file_path)
        if not syntax_valid:
            result["valid"] = False
            result["issues"].append(f"Syntax: {syntax_msg}")
            self.validation_results["syntax_errors"].append({
                "file": str(file_path),
                "error": syntax_msg
            })
        
        # Validate structure
        structure_issues = self.validate_test_structure(file_path)
        if structure_issues:
            result["issues"].extend([f"Structure: {issue}" for issue in structure_issues])
            self.validation_results["structure_issues"].extend([
                {"file": str(file_path), "issue": issue} for issue in structure_issues
            ])
        
        # Validate imports
        import_issues = self.validate_imports(file_path)
        if import_issues:
            result["issues"].extend([f"Import: {issue}" for issue in import_issues])
            self.validation_results["import_errors"].extend([
                {"file": str(file_path), "issue": issue} for issue in import_issues
            ])
        
        if not result["issues"]:
            self.validation_results["valid_files"] += 1
        
        return result
    
    def validate_all_tests(self) -> Dict[str, Any]:
        """Validate all test files in the test directory."""
        test_files = list(self.test_root.rglob("test_*.py"))
        
        print(f"Found {len(test_files)} test files to validate...")
        
        results = []
        for test_file in test_files:
            self.validation_results["total_files"] += 1
            result = self.validate_file(test_file)
            results.append(result)
            
            # Print progress
            status = "✅" if result["valid"] else "❌"
            print(f"{status} {test_file.name}")
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"    - {issue}")
        
        # Update final results
        self.validation_results["results"] = results
        return self.validation_results
    
    def print_summary(self):
        """Print validation summary."""
        results = self.validation_results
        
        print("\n" + "="*60)
        print("TEST VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Total test files: {results['total_files']}")
        print(f"Valid test files: {results['valid_files']}")
        print(f"Files with issues: {results['total_files'] - results['valid_files']}")
        
        if results['syntax_errors']:
            print(f"\nSyntax errors: {len(results['syntax_errors'])}")
            for error in results['syntax_errors']:
                print(f"  - {Path(error['file']).name}: {error['error']}")
        
        if results['import_errors']:
            print(f"\nImport issues: {len(results['import_errors'])}")
            for error in results['import_errors']:
                print(f"  - {Path(error['file']).name}: {error['issue']}")
        
        if results['structure_issues']:
            print(f"\nStructure issues: {len(results['structure_issues'])}")
            for issue in results['structure_issues']:
                print(f"  - {Path(issue['file']).name}: {issue['issue']}")
        
        success_rate = (results['valid_files'] / results['total_files']) * 100 if results['total_files'] > 0 else 0
        print(f"\nValidation success rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 All tests passed validation!")
        elif success_rate >= 80:
            print("⚠️  Most tests are valid, but some issues need attention.")
        else:
            print("❌ Significant issues found. Please review and fix.")


def main():
    """Main validation function."""
    test_root = Path(__file__).parent
    
    print("MoRAG Test Suite Validator")
    print(f"Validating tests in: {test_root}")
    print("-" * 60)
    
    validator = TestValidator(test_root)
    validator.validate_all_tests()
    validator.print_summary()
    
    # Exit with error code if validation failed
    success_rate = (validator.validation_results['valid_files'] / 
                   validator.validation_results['total_files']) * 100
    
    if success_rate < 100:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()