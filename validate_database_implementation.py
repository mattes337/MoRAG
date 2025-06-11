#!/usr/bin/env python3
"""
Database Implementation Validation Script

This script validates that all database layer components are properly implemented
and can be imported without errors.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the packages to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))

def test_imports():
    """Test that all new components can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test API Keys
        from morag_core.api_keys import (
            ApiKeyService, ApiKeyMiddleware, 
            ApiKeyCreate, ApiKeyResponse,
            get_api_key_user, require_api_key
        )
        print("✅ API Keys module imported successfully")
        
        # Test Tenancy
        from morag_core.tenancy import (
            TenantService, TenantMiddleware,
            TenantInfo, ResourceType
        )
        print("✅ Tenancy module imported successfully")
        
        # Test Database Session
        from morag_core.database.session import get_session_context
        print("✅ Database session imported successfully")
        
        # Test Database Models
        from morag_core.database.models import (
            User, UserSettings, Document, Job, ApiKey,
            create_user, create_api_key, create_document, create_job
        )
        print("✅ Database models imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_database_operations():
    """Test basic database operations."""
    print("\n🔍 Testing database operations...")
    
    try:
        from morag_core.database import DatabaseManager
        from morag_core.database.session import get_session_context
        from morag_core.database.models import create_user, create_api_key
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        db_url = f"sqlite:///{db_path}"
        db_manager = DatabaseManager(db_url)
        db_manager.create_tables()
        
        # Test user creation
        with get_session_context(db_manager) as session:
            user = create_user(
                session,
                name="Test User",
                email="test@example.com"
            )
            
            # Test API key creation
            api_key = create_api_key(
                session,
                user_id=user.id,
                name="Test API Key",
                key="test_key_hash"
            )
            
            assert user.id is not None
            assert api_key.id is not None
            assert api_key.user_id == user.id
        
        # Cleanup
        os.unlink(db_path)
        
        print("✅ Database operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

def test_service_initialization():
    """Test that services can be initialized."""
    print("\n🔍 Testing service initialization...")
    
    try:
        from morag_core.api_keys import ApiKeyService
        from morag_core.tenancy import TenantService
        
        # Test service initialization
        api_key_service = ApiKeyService()
        tenant_service = TenantService()
        
        print("✅ Services initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

def test_model_validation():
    """Test that Pydantic models work correctly."""
    print("\n🔍 Testing model validation...")
    
    try:
        from morag_core.api_keys.models import ApiKeyCreate, ApiKeyPermission
        from morag_core.tenancy.models import ResourceType, ResourceQuota
        
        # Test API key model
        api_key_data = ApiKeyCreate(
            name="Test Key",
            description="Test description",
            permissions=[ApiKeyPermission.READ, ApiKeyPermission.WRITE]
        )
        assert api_key_data.name == "Test Key"
        
        # Test quota model
        quota = ResourceQuota(
            resource_type=ResourceType.DOCUMENTS,
            limit=1000,
            used=50
        )
        assert quota.usage_percentage == 0.05
        assert not quota.is_exceeded
        
        print("✅ Model validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model validation test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Database Implementation Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_service_initialization,
        test_model_validation,
        test_database_operations,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database implementation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
