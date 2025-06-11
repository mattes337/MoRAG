"""Database connection and session management."""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Optional
import structlog

from morag_core.config import get_settings
from .models import Base

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Enhanced database manager with connection pooling."""

    def __init__(self, database_url: Optional[str] = None):
        settings = get_settings()
        self.database_url = database_url or settings.database_url

        # Configure engine with connection pooling
        engine_kwargs = {
            "poolclass": QueuePool,
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_max_overflow,
            "pool_pre_ping": True,
            "echo": settings.database_echo,
        }

        # Handle SQLite special case (no connection pooling)
        if self.database_url.startswith("sqlite"):
            engine_kwargs = {
                "echo": settings.database_echo,
                "connect_args": {"check_same_thread": False},
            }

        self.engine = create_engine(self.database_url, **engine_kwargs)

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        logger.info("Database manager initialized", url=self._safe_url())

    def _safe_url(self) -> str:
        """Return database URL with password masked for logging."""
        if "://" not in self.database_url:
            return self.database_url

        try:
            # Split URL to mask password
            parts = self.database_url.split("://")
            if len(parts) != 2:
                return self.database_url

            scheme = parts[0]
            rest = parts[1]

            if "@" in rest:
                auth_part, host_part = rest.split("@", 1)
                if ":" in auth_part:
                    user, _ = auth_part.split(":", 1)
                    return f"{scheme}://{user}:***@{host_part}"
                else:
                    return f"{scheme}://{auth_part}:***@{host_part}"
            else:
                return self.database_url
        except Exception:
            return "***"

    def create_tables(self):
        """Create all tables in the database."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise

    def drop_tables(self):
        """Drop all tables in the database."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e))
            raise

    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()

    def close_session(self, session):
        """Close a database session."""
        try:
            session.close()
        except Exception as e:
            logger.warning("Error closing database session", error=str(e))

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            return False

    def get_table_names(self) -> list:
        """Get list of table names in the database."""
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            logger.error("Failed to get table names", error=str(e))
            return []

    def verify_schema(self) -> bool:
        """Verify database schema matches expected structure."""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            expected_tables = [
                "users",
                "user_settings",
                "database_servers",
                "databases",
                "documents",
                "api_keys",
                "jobs",
            ]

            missing_tables = [table for table in expected_tables if table not in tables]
            if missing_tables:
                logger.warning("Missing database tables", missing=missing_tables)
                return False

            logger.info("Database schema verification successful", tables=tables)
            return True
        except Exception as e:
            logger.error("Schema verification failed", error=str(e))
            return False


# Global database manager instance
_database_manager_instance = None


def get_database_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """Get the global database manager instance, creating it if necessary."""
    global _database_manager_instance
    if _database_manager_instance is None or database_url is not None:
        _database_manager_instance = DatabaseManager(database_url)
    return _database_manager_instance


def reset_database_manager():
    """Reset the global database manager instance."""
    global _database_manager_instance
    _database_manager_instance = None
