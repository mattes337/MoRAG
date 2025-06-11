"""User management service."""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import structlog

from morag_core.database import User, UserSettings, get_database_manager, get_session_context
from morag_core.database.models import UserRole as DBUserRole, Theme as DBTheme
from .models import (
    UserCreate, UserUpdate, UserSettingsUpdate, UserResponse, 
    UserSettingsResponse, PasswordChangeRequest
)
from .security import PasswordManager, JWTManager
from morag_core.exceptions import (
    AuthenticationError, ValidationError, NotFoundError, ConflictError, DatabaseError
)

logger = structlog.get_logger(__name__)


class UserService:
    """User management service."""

    def __init__(self, database_url: Optional[str] = None):
        self.db_manager = get_database_manager(database_url)
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager()
    
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user."""
        try:
            with get_session_context(self.db_manager) as session:
                # Check if user already exists
                existing_user = session.query(User).filter_by(email=user_data.email).first()
                if existing_user:
                    raise ConflictError(f"User with email {user_data.email} already exists")
                
                # Hash password
                hashed_password = self.password_manager.hash_password(user_data.password)
                
                # Create user
                user = User(
                    name=user_data.name,
                    email=user_data.email,
                    password_hash=hashed_password,
                    avatar=user_data.avatar,
                    role=DBUserRole.USER,
                    is_active=True
                )
                
                session.add(user)
                session.flush()  # Get user ID
                
                # Create default user settings
                user_settings = UserSettings(
                    user_id=user.id,
                    theme=DBTheme.LIGHT,
                    language="en",
                    notifications=True,
                    auto_save=True
                )
                session.add(user_settings)
                
                logger.info("User created", user_id=user.id, email=user.email)
                return self._user_to_response(user)
                
        except IntegrityError as e:
            logger.error("Database integrity error during user creation", error=str(e))
            raise ConflictError(f"User with email {user_data.email} already exists")
        except Exception as e:
            logger.error("Failed to create user", error=str(e))
            raise DatabaseError(f"Failed to create user: {str(e)}")
    
    def authenticate_user(self, email: str, password: str) -> Optional[UserResponse]:
        """Authenticate a user by email and password."""
        try:
            with get_session_context(self.db_manager) as session:
                user = session.query(User).filter_by(email=email, is_active=True).first()
                if not user or not user.password_hash:
                    return None
                
                if not self.password_manager.verify_password(password, user.password_hash):
                    return None
                
                logger.info("User authenticated", user_id=user.id, email=email)
                return self._user_to_response(user)
                
        except Exception as e:
            logger.error("Authentication error", email=email, error=str(e))
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID."""
        try:
            with get_session_context(self.db_manager) as session:
                user = session.query(User).filter_by(id=user_id, is_active=True).first()
                if user:
                    return self._user_to_response(user)
                return None
        except Exception as e:
            logger.error("Failed to get user by ID", user_id=user_id, error=str(e))
            return None
    
    def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get user by email."""
        try:
            with get_session_context(self.db_manager) as session:
                user = session.query(User).filter_by(email=email, is_active=True).first()
                if user:
                    return self._user_to_response(user)
                return None
        except Exception as e:
            logger.error("Failed to get user by email", email=email, error=str(e))
            return None
    
    def update_user(self, user_id: str, user_data: UserUpdate) -> UserResponse:
        """Update user information."""
        try:
            with get_session_context(self.db_manager) as session:
                user = session.query(User).filter_by(id=user_id, is_active=True).first()
                if not user:
                    raise NotFoundError(f"User {user_id} not found")
                
                # Update fields
                if user_data.name is not None:
                    user.name = user_data.name
                if user_data.email is not None:
                    # Check email uniqueness
                    existing = session.query(User).filter_by(email=user_data.email).first()
                    if existing and existing.id != user_id:
                        raise ConflictError(f"Email {user_data.email} already in use")
                    user.email = user_data.email
                if user_data.avatar is not None:
                    user.avatar = user_data.avatar
                
                logger.info("User updated", user_id=user_id)
                return self._user_to_response(user)
                
        except (NotFoundError, ConflictError):
            raise
        except Exception as e:
            logger.error("Failed to update user", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to update user: {str(e)}")
    
    def change_password(self, user_id: str, password_data: PasswordChangeRequest) -> bool:
        """Change user password."""
        try:
            with get_session_context(self.db_manager) as session:
                user = session.query(User).filter_by(id=user_id, is_active=True).first()
                if not user:
                    raise NotFoundError(f"User {user_id} not found")
                
                # Verify current password
                if not user.password_hash or not self.password_manager.verify_password(
                    password_data.current_password, user.password_hash
                ):
                    raise AuthenticationError("Current password is incorrect")
                
                # Hash new password
                user.password_hash = self.password_manager.hash_password(password_data.new_password)
                
                logger.info("Password changed", user_id=user_id)
                return True
                
        except (NotFoundError, AuthenticationError):
            raise
        except Exception as e:
            logger.error("Failed to change password", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to change password: {str(e)}")
    
    def get_user_settings(self, user_id: str) -> Optional[UserSettingsResponse]:
        """Get user settings."""
        try:
            with get_session_context(self.db_manager) as session:
                settings = session.query(UserSettings).filter_by(user_id=user_id).first()
                if settings:
                    return self._settings_to_response(settings)
                return None
        except Exception as e:
            logger.error("Failed to get user settings", user_id=user_id, error=str(e))
            return None
    
    def update_user_settings(self, user_id: str, settings_data: UserSettingsUpdate) -> UserSettingsResponse:
        """Update user settings."""
        try:
            with get_session_context(self.db_manager) as session:
                settings = session.query(UserSettings).filter_by(user_id=user_id).first()
                if not settings:
                    raise NotFoundError(f"User settings for {user_id} not found")
                
                # Update settings
                if settings_data.theme is not None:
                    settings.theme = DBTheme(settings_data.theme.value)
                if settings_data.language is not None:
                    settings.language = settings_data.language
                if settings_data.notifications is not None:
                    settings.notifications = settings_data.notifications
                if settings_data.auto_save is not None:
                    settings.auto_save = settings_data.auto_save
                if settings_data.default_database is not None:
                    settings.default_database = settings_data.default_database
                
                logger.info("User settings updated", user_id=user_id)
                return self._settings_to_response(settings)
                
        except NotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update user settings", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to update user settings: {str(e)}")
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user (soft delete by setting inactive)."""
        try:
            with get_session_context(self.db_manager) as session:
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    return False
                
                # Soft delete
                user.is_active = False
                logger.info("User deleted", user_id=user_id)
                return True
                
        except Exception as e:
            logger.error("Failed to delete user", user_id=user_id, error=str(e))
            return False
    
    def list_users(self, skip: int = 0, limit: int = 100) -> List[UserResponse]:
        """List all active users (admin only)."""
        try:
            with get_session_context(self.db_manager) as session:
                users = session.query(User).filter_by(is_active=True).offset(skip).limit(limit).all()
                return [self._user_to_response(user) for user in users]
        except Exception as e:
            logger.error("Failed to list users", error=str(e))
            return []
    
    def _user_to_response(self, user: User) -> UserResponse:
        """Convert User model to UserResponse."""
        return UserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            avatar=user.avatar,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    def _settings_to_response(self, settings: UserSettings) -> UserSettingsResponse:
        """Convert UserSettings to UserSettingsResponse."""
        return UserSettingsResponse(
            theme=settings.theme.value,
            language=settings.language,
            notifications=settings.notifications,
            auto_save=settings.auto_save,
            default_database=settings.default_database,
            created_at=settings.created_at,
            updated_at=settings.updated_at
        )
