"""Database session handling and dependency injection."""

from contextlib import contextmanager
from typing import Generator
import structlog

from .manager import get_database_manager

logger = structlog.get_logger(__name__)


def get_session():
    """Get a database session for dependency injection."""
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    except Exception as e:
        logger.error("Database session error", error=str(e))
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_session_context():
    """Get a database session with context manager."""
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error("Database session error", error=str(e))
        session.rollback()
        raise
    finally:
        session.close()


# For backward compatibility
SessionLocal = None


def init_session_local():
    """Initialize SessionLocal for backward compatibility."""
    global SessionLocal
    db_manager = get_database_manager()
    SessionLocal = db_manager.SessionLocal
    return SessionLocal
