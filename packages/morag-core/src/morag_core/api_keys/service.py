"""API Key service for MoRAG core."""

import secrets
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import structlog

from morag_core.database import (
    get_database_manager, 
    ApiKey, 
    User,
    get_session_context
)
from morag_core.exceptions import NotFoundError, ConflictError, ValidationError
from .models import (
    ApiKeyCreate,
    ApiKeyUpdate,
    ApiKeyResponse,
    ApiKeyCreateResponse,
    ApiKeyUsage,
    ApiKeyUsageStats,
    ApiKeySearchRequest,
    ApiKeySearchResponse,
    ApiKeyStatus,
    ApiKeyPermission,
)

logger = structlog.get_logger(__name__)


class ApiKeyService:
    """Service for managing API keys."""

    def __init__(self):
        self.db_manager = get_database_manager()

    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        # Generate 32 bytes of random data and encode as hex
        return f"mk_{secrets.token_hex(32)}"

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_api_key(self, api_key_data: ApiKeyCreate, user_id: str) -> ApiKeyCreateResponse:
        """Create a new API key."""
        try:
            with get_session_context(self.db_manager) as session:
                # Check if user exists
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    raise NotFoundError(f"User not found: {user_id}")

                # Check for duplicate name
                existing = session.query(ApiKey).filter_by(
                    user_id=user_id, 
                    name=api_key_data.name
                ).first()
                if existing:
                    raise ConflictError(f"API key with name '{api_key_data.name}' already exists")

                # Generate API key
                secret_key = self.generate_api_key()
                key_hash = self.hash_api_key(secret_key)

                # Create API key record
                api_key = ApiKey(
                    user_id=user_id,
                    name=api_key_data.name,
                    key=key_hash,
                    created=datetime.now(timezone.utc),
                    last_used=None
                )

                session.add(api_key)
                session.commit()

                logger.info("API key created", 
                           api_key_id=api_key.id,
                           user_id=user_id,
                           name=api_key_data.name)

                # Convert to response model
                api_key_response = self._to_response_model(api_key, api_key_data)

                return ApiKeyCreateResponse(
                    api_key=api_key_response,
                    secret_key=secret_key
                )

        except Exception as e:
            logger.error("Failed to create API key", 
                        user_id=user_id,
                        name=api_key_data.name,
                        error=str(e))
            raise

    def get_api_key(self, api_key_id: str, user_id: str) -> Optional[ApiKeyResponse]:
        """Get API key by ID."""
        try:
            with get_session_context(self.db_manager) as session:
                api_key = session.query(ApiKey).filter_by(
                    id=api_key_id,
                    user_id=user_id
                ).first()

                if not api_key:
                    return None

                return self._to_response_model(api_key)

        except Exception as e:
            logger.error("Failed to get API key",
                        api_key_id=api_key_id,
                        user_id=user_id,
                        error=str(e))
            raise

    def update_api_key(self, api_key_id: str, api_key_data: ApiKeyUpdate, user_id: str) -> ApiKeyResponse:
        """Update an API key."""
        try:
            with get_session_context(self.db_manager) as session:
                api_key = session.query(ApiKey).filter_by(
                    id=api_key_id,
                    user_id=user_id
                ).first()

                if not api_key:
                    raise NotFoundError(f"API key not found: {api_key_id}")

                # Update fields
                if api_key_data.name is not None:
                    # Check for duplicate name
                    existing = session.query(ApiKey).filter_by(
                        user_id=user_id,
                        name=api_key_data.name
                    ).filter(ApiKey.id != api_key_id).first()
                    if existing:
                        raise ConflictError(f"API key with name '{api_key_data.name}' already exists")
                    api_key.name = api_key_data.name

                session.commit()

                logger.info("API key updated",
                           api_key_id=api_key_id,
                           user_id=user_id)

                return self._to_response_model(api_key, api_key_data)

        except Exception as e:
            logger.error("Failed to update API key",
                        api_key_id=api_key_id,
                        user_id=user_id,
                        error=str(e))
            raise

    def delete_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Delete an API key."""
        try:
            with get_session_context(self.db_manager) as session:
                api_key = session.query(ApiKey).filter_by(
                    id=api_key_id,
                    user_id=user_id
                ).first()

                if not api_key:
                    raise NotFoundError(f"API key not found: {api_key_id}")

                session.delete(api_key)
                session.commit()

                logger.info("API key deleted",
                           api_key_id=api_key_id,
                           user_id=user_id)

                return True

        except Exception as e:
            logger.error("Failed to delete API key",
                        api_key_id=api_key_id,
                        user_id=user_id,
                        error=str(e))
            raise

    def list_api_keys(self, search_request: ApiKeySearchRequest, user_id: str) -> ApiKeySearchResponse:
        """List API keys for a user."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(ApiKey).filter_by(user_id=user_id)

                # Apply filters
                if search_request.name_contains:
                    query = query.filter(ApiKey.name.contains(search_request.name_contains))

                if search_request.created_after:
                    query = query.filter(ApiKey.created_at >= search_request.created_after)

                if search_request.created_before:
                    query = query.filter(ApiKey.created_at <= search_request.created_before)

                if search_request.last_used_after:
                    query = query.filter(ApiKey.last_used >= search_request.last_used_after)

                if search_request.last_used_before:
                    query = query.filter(ApiKey.last_used <= search_request.last_used_before)

                # Get total count
                total = query.count()

                # Apply pagination
                api_keys = query.offset(search_request.skip).limit(search_request.limit).all()

                # Convert to response models
                api_key_responses = [self._to_response_model(api_key) for api_key in api_keys]

                return ApiKeySearchResponse(
                    api_keys=api_key_responses,
                    total=total,
                    skip=search_request.skip,
                    limit=search_request.limit
                )

        except Exception as e:
            logger.error("Failed to list API keys",
                        user_id=user_id,
                        error=str(e))
            raise

    def authenticate_api_key(self, api_key: str) -> Optional[str]:
        """Authenticate an API key and return user ID."""
        try:
            key_hash = self.hash_api_key(api_key)

            with get_session_context(self.db_manager) as session:
                api_key_record = session.query(ApiKey).filter_by(key=key_hash).first()

                if not api_key_record:
                    return None

                # Update last used timestamp
                api_key_record.last_used = datetime.now(timezone.utc)
                session.commit()

                logger.debug("API key authenticated",
                           api_key_id=api_key_record.id,
                           user_id=api_key_record.user_id)

                return api_key_record.user_id

        except Exception as e:
            logger.error("Failed to authenticate API key", error=str(e))
            return None

    def _to_response_model(self, api_key: ApiKey, create_data: Optional[ApiKeyCreate] = None) -> ApiKeyResponse:
        """Convert database model to response model."""
        return ApiKeyResponse(
            id=api_key.id,
            name=api_key.name,
            description=create_data.description if create_data else None,
            key_prefix=api_key.key[:8] if api_key.key else "",
            permissions=create_data.permissions if create_data else [ApiKeyPermission.READ],
            status=ApiKeyStatus.ACTIVE,  # Default status
            expires_at=create_data.expires_at if create_data else None,
            rate_limit_per_hour=create_data.rate_limit_per_hour if create_data else 1000,
            allowed_ips=create_data.allowed_ips if create_data else None,
            created_at=api_key.created_at,
            last_used=api_key.last_used,
            usage_count=0,  # TODO: Implement usage tracking
            user_id=api_key.user_id,
            metadata=create_data.metadata if create_data else None
        )
