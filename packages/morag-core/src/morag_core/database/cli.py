#!/usr/bin/env python3
"""Database management CLI for MoRAG."""

import argparse
import sys
from typing import Optional
import structlog

from morag_core.config import get_settings
from .initialization import DatabaseInitializer
from .manager import get_database_manager

logger = structlog.get_logger(__name__)


def init_database(args):
    """Initialize database with tables."""
    try:
        initializer = DatabaseInitializer(args.database_url)
        success = initializer.initialize_database(drop_existing=args.drop_existing)
        
        if success:
            print("✅ Database initialized successfully")
            
            if args.create_admin:
                admin_id = initializer.create_admin_user(
                    name=args.admin_name or "Admin User",
                    email=args.admin_email or "admin@morag.dev"
                )
                if admin_id:
                    print(f"✅ Admin user created: {args.admin_email or 'admin@morag.dev'}")
                else:
                    print("❌ Failed to create admin user")
                    return 1
            
            if args.setup_dev_data:
                success = initializer.setup_development_data()
                if success:
                    print("✅ Development data setup completed")
                else:
                    print("❌ Failed to setup development data")
                    return 1
        else:
            print("❌ Database initialization failed")
            return 1
            
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def check_database(args):
    """Check database status and schema."""
    try:
        initializer = DatabaseInitializer(args.database_url)
        info = initializer.get_database_info()
        
        print("📊 Database Information:")
        print(f"  Connection: {'✅ OK' if info['connection_ok'] else '❌ Failed'}")
        print(f"  Schema: {'✅ Valid' if info['schema_valid'] else '❌ Invalid'}")
        print(f"  Tables: {len(info['tables'])}")
        print(f"  Users: {info['user_count']}")
        print(f"  Documents: {info['document_count']}")
        
        if info['tables']:
            print("  Table list:")
            for table in info['tables']:
                print(f"    - {table}")
        
        if 'error' in info:
            print(f"  Error: {info['error']}")
            return 1
            
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def reset_database(args):
    """Reset database by dropping and recreating all tables."""
    try:
        if not args.confirm:
            response = input("⚠️  This will delete ALL data. Are you sure? (yes/no): ")
            if response.lower() != 'yes':
                print("Operation cancelled")
                return 0
        
        initializer = DatabaseInitializer(args.database_url)
        success = initializer.reset_database()
        
        if success:
            print("✅ Database reset successfully")
            return 0
        else:
            print("❌ Database reset failed")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def create_user(args):
    """Create a new user."""
    try:
        initializer = DatabaseInitializer(args.database_url)
        
        if args.admin:
            user_id = initializer.create_admin_user(
                name=args.name,
                email=args.email,
                avatar=args.avatar
            )
        else:
            user_id = initializer.create_test_user(
                name=args.name,
                email=args.email,
                avatar=args.avatar
            )
        
        if user_id:
            role = "admin" if args.admin else "user"
            print(f"✅ {role.title()} user created: {args.email} (ID: {user_id})")
            return 0
        else:
            print("❌ Failed to create user")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MoRAG Database Management CLI")
    parser.add_argument(
        "--database-url",
        help="Database URL (overrides config)",
        default=None
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating"
    )
    init_parser.add_argument(
        "--create-admin",
        action="store_true",
        help="Create admin user after initialization"
    )
    init_parser.add_argument(
        "--admin-name",
        default="Admin User",
        help="Admin user name"
    )
    init_parser.add_argument(
        "--admin-email",
        default="admin@morag.dev",
        help="Admin user email"
    )
    init_parser.add_argument(
        "--setup-dev-data",
        action="store_true",
        help="Setup development data"
    )
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check database status")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset database")
    reset_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Create user command
    user_parser = subparsers.add_parser("create-user", help="Create a new user")
    user_parser.add_argument("name", help="User name")
    user_parser.add_argument("email", help="User email")
    user_parser.add_argument("--avatar", help="User avatar URL")
    user_parser.add_argument(
        "--admin",
        action="store_true",
        help="Create admin user"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Execute command
    if args.command == "init":
        return init_database(args)
    elif args.command == "check":
        return check_database(args)
    elif args.command == "reset":
        return reset_database(args)
    elif args.command == "create-user":
        return create_user(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
