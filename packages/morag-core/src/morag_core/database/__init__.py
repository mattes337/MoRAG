"""Database package for MoRAG core."""

from .models import (
    Base,
    User,
    UserSettings,
    DatabaseServer,
    Database,
    Document,
    ApiKey,
    Job,
    UserRole,
    Theme,
    DocumentState,
    DatabaseType,
    JobStatus,
    create_user,
    create_database_server,
    create_database,
    create_document,
    create_api_key,
    create_job,
    update_document_count,
    update_job_progress,
)
from .manager import DatabaseManager, get_database_manager, reset_database_manager
from .session import get_session, SessionLocal
from .initialization import get_session_context
from .initialization import DatabaseInitializer
from .migrations import MigrationManager

__all__ = [
    # Base and models
    "Base",
    "User",
    "UserSettings",
    "DatabaseServer",
    "Database",
    "Document",
    "ApiKey",
    "Job",
    # Enums
    "UserRole",
    "Theme",
    "DocumentState",
    "DatabaseType",
    "JobStatus",
    # Manager and session
    "DatabaseManager",
    "get_database_manager",
    "reset_database_manager",
    "get_session",
    "get_session_context",
    "SessionLocal",
    # Initialization
    "DatabaseInitializer",
    # Migrations
    "MigrationManager",
    # Utility functions
    "create_user",
    "create_database_server",
    "create_database",
    "create_document",
    "create_api_key",
    "create_job",
    "update_document_count",
    "update_job_progress",
]
