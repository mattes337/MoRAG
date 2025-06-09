"""API key authentication service for remote workers."""

import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis
import json
import structlog

logger = structlog.get_logger(__name__)

class APIKeyService:
    """Service for managing API keys and user authentication."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
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

        # Store API key data
        self.redis.setex(
            f"{self.key_prefix}{key_hash}",
            int(timedelta(days=expires_days or 365).total_seconds()),
            json.dumps(key_data)
        )

        logger.info("API key created", user_id=user_id, description=description)
        return api_key

    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data_json = self.redis.get(f"{self.key_prefix}{key_hash}")

        if not key_data_json:
            return None

        key_data = json.loads(key_data_json)

        # Check if key is active
        if not key_data.get("active", False):
            return None

        # Check expiration
        if key_data.get("expires_at"):
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if datetime.utcnow() > expires_at:
                return None

        return key_data

    def get_user_queue_name(self, user_id: str, worker_type: str = "gpu") -> str:
        """Get queue name for user and worker type."""
        return f"{worker_type}-tasks-{user_id}"

    def get_cpu_queue_name(self, user_id: str) -> str:
        """Get CPU queue name for user."""
        return f"cpu-tasks-{user_id}"

    def get_default_queue_name(self) -> str:
        """Get default queue name for anonymous processing."""
        return "celery"

    async def list_user_api_keys(self, user_id: str) -> list:
        """List all API keys for a user."""
        keys = []
        pattern = f"{self.key_prefix}*"
        
        for key in self.redis.scan_iter(match=pattern):
            key_data_json = self.redis.get(key)
            if key_data_json:
                key_data = json.loads(key_data_json)
                if key_data.get("user_id") == user_id:
                    # Don't include the actual key hash in response
                    safe_data = {k: v for k, v in key_data.items() if k != "key_hash"}
                    keys.append(safe_data)
        
        return keys

    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_name = f"{self.key_prefix}{key_hash}"
        
        key_data_json = self.redis.get(key_name)
        if not key_data_json:
            return False
            
        key_data = json.loads(key_data_json)
        key_data["active"] = False
        
        # Update the key data
        self.redis.set(key_name, json.dumps(key_data))
        
        logger.info("API key revoked", user_id=key_data.get("user_id"))
        return True
