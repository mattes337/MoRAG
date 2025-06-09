"""API key authentication service for HTTP remote workers - No Redis Required."""

import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
import structlog

logger = structlog.get_logger(__name__)

class APIKeyService:
    """Service for managing API keys and user authentication - In-memory storage."""

    def __init__(self):
        # In-memory storage for HTTP workers (no Redis required)
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_prefix = "morag:api_keys:"
        self.user_prefix = "morag:users:"

    async def create_api_key(self, user_id: str, description: str = "",
                           expires_days: Optional[int] = None) -> str:
        """Create a new API key for a user."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        key_data = {
            "user_id": user_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=expires_days)).isoformat() if expires_days else None,
            "active": True
        }

        # Store API key data in memory
        self._api_keys[key_hash] = key_data

        logger.info("API key created", user_id=user_id, description=description)
        return api_key

    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = self._api_keys.get(key_hash)

        if not key_data:
            return None

        # Check if key is active
        if not key_data.get("active", False):
            return None

        # Check expiration
        if key_data.get("expires_at"):
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if datetime.utcnow() > expires_at:
                return None

        return key_data

    def get_user_worker_id(self, user_id: str, worker_type: str = "gpu") -> str:
        """Get worker ID for user and worker type (HTTP workers don't use queues)."""
        return f"{worker_type}-worker-{user_id}"

    def get_cpu_worker_id(self, user_id: str) -> str:
        """Get CPU worker ID for user."""
        return f"cpu-worker-{user_id}"

    def get_default_worker_id(self) -> str:
        """Get default worker ID for anonymous processing."""
        return "http-worker"

    async def list_user_api_keys(self, user_id: str) -> list:
        """List all API keys for a user."""
        keys = []

        for key_hash, key_data in self._api_keys.items():
            if key_data.get("user_id") == user_id:
                # Don't include the actual key hash in response
                safe_data = {k: v for k, v in key_data.items() if k != "key_hash"}
                keys.append(safe_data)

        return keys

    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        key_data = self._api_keys.get(key_hash)
        if not key_data:
            return False

        key_data["active"] = False

        logger.info("API key revoked", user_id=key_data.get("user_id"))
        return True
