"""Database initialization and setup utilities."""

from typing import Optional
import structlog

from .manager import DatabaseManager
from .models import User, UserSettings, UserRole, Theme
from contextlib import contextmanager

logger = structlog.get_logger(__name__)


@contextmanager
def get_session_context(db_manager):
    """Get a database session with context manager."""
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


class DatabaseInitializer:
    """Database initialization and setup manager."""

    def __init__(self, database_url: Optional[str] = None):
        self.db_manager = DatabaseManager(database_url)

    def initialize_database(self, drop_existing: bool = False) -> bool:
        """Initialize database with tables and default data."""
        try:
            logger.info("Starting database initialization", drop_existing=drop_existing)

            # Test connection first
            if not self.db_manager.test_connection():
                logger.error("Database connection failed")
                return False

            # Drop existing tables if requested
            if drop_existing:
                logger.info("Dropping existing tables")
                self.db_manager.drop_tables()

            # Create tables
            logger.info("Creating database tables")
            self.db_manager.create_tables()

            # Verify schema
            if not self.verify_schema():
                logger.error("Schema verification failed")
                return False

            logger.info("Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            return False

    def verify_schema(self) -> bool:
        """Verify database schema matches expected structure."""
        return self.db_manager.verify_schema()

    def create_admin_user(
        self, name: str, email: str, avatar: Optional[str] = None
    ) -> Optional[str]:
        """Create an admin user and return user ID."""
        try:
            with get_session_context(self.db_manager) as session:
                # Check if user already exists
                existing_user = session.query(User).filter(User.email == email).first()
                if existing_user:
                    logger.warning("User already exists", email=email, user_id=existing_user.id)
                    return existing_user.id

                # Create admin user
                user = User(name=name, email=email, role=UserRole.ADMIN, avatar=avatar)
                session.add(user)
                session.flush()  # Get the user ID

                # Create default user settings
                user_settings = UserSettings(
                    user_id=user.id,
                    theme=Theme.LIGHT,
                    language="en",
                    notifications=True,
                    auto_save=True,
                )
                session.add(user_settings)

                logger.info("Admin user created", email=email, user_id=user.id)
                return user.id

        except Exception as e:
            logger.error("Failed to create admin user", error=str(e), email=email)
            return None

    def setup_development_data(self) -> bool:
        """Set up development data for testing."""
        try:
            logger.info("Setting up development data")

            # Create admin user
            admin_id = self.create_admin_user(
                name="Admin User", email="admin@morag.dev", avatar=None
            )

            if not admin_id:
                logger.error("Failed to create admin user")
                return False

            # Create test user
            test_user_id = self.create_test_user(
                name="Test User", email="test@morag.dev", avatar=None
            )

            if not test_user_id:
                logger.error("Failed to create test user")
                return False

            logger.info(
                "Development data setup completed",
                admin_id=admin_id,
                test_user_id=test_user_id,
            )
            return True

        except Exception as e:
            logger.error("Failed to setup development data", error=str(e))
            return False

    def create_test_user(
        self, name: str, email: str, avatar: Optional[str] = None
    ) -> Optional[str]:
        """Create a test user and return user ID."""
        try:
            with get_session_context(self.db_manager) as session:
                # Check if user already exists
                existing_user = session.query(User).filter(User.email == email).first()
                if existing_user:
                    logger.warning("User already exists", email=email, user_id=existing_user.id)
                    return existing_user.id

                # Create test user
                user = User(name=name, email=email, role=UserRole.USER, avatar=avatar)
                session.add(user)
                session.flush()  # Get the user ID

                # Create default user settings
                user_settings = UserSettings(
                    user_id=user.id,
                    theme=Theme.SYSTEM,
                    language="en",
                    notifications=True,
                    auto_save=True,
                )
                session.add(user_settings)

                logger.info("Test user created", email=email, user_id=user.id)
                return user.id

        except Exception as e:
            logger.error("Failed to create test user", error=str(e), email=email)
            return None

    def reset_database(self) -> bool:
        """Reset database by dropping and recreating all tables."""
        try:
            logger.info("Resetting database")
            return self.initialize_database(drop_existing=True)
        except Exception as e:
            logger.error("Failed to reset database", error=str(e))
            return False

    def get_database_info(self) -> dict:
        """Get database information and statistics."""
        try:
            with get_session_context(self.db_manager) as session:
                user_count = session.query(User).count()
                document_count = session.query(User).count()  # Will be updated when Document model is available
                
                return {
                    "tables": self.db_manager.get_table_names(),
                    "user_count": user_count,
                    "document_count": document_count,
                    "schema_valid": self.verify_schema(),
                    "connection_ok": self.db_manager.test_connection(),
                }
        except Exception as e:
            logger.error("Failed to get database info", error=str(e))
            return {
                "tables": [],
                "user_count": 0,
                "document_count": 0,
                "schema_valid": False,
                "connection_ok": False,
                "error": str(e),
            }
