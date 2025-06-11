"""Security utilities for authentication."""

import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import structlog
import secrets

from morag_core.config import get_settings

logger = structlog.get_logger(__name__)


class PasswordManager:
    """Password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error("Password verification failed", error=str(e))
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a secure random password."""
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))


class JWTManager:
    """JWT token management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.jwt_secret_key
        self.algorithm = self.settings.jwt_algorithm
        self.expiration_hours = self.settings.jwt_expiration_hours
    
    def create_access_token(self, user_id: str, email: str, role: str) -> Dict[str, Any]:
        """Create a JWT access token."""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.expiration_hours)
        
        payload = {
            "sub": user_id,
            "email": email,
            "role": role,
            "iat": now,
            "exp": expires_at,
            "type": "access"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self.expiration_hours * 3600,
            "expires_at": expires_at.isoformat()
        }
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get("type") != "access":
                logger.warning("Invalid token type", token_type=payload.get("type"))
                return None
                
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None
    
    def refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Refresh a JWT token if it's still valid."""
        payload = self.decode_token(token)
        if payload:
            return self.create_access_token(
                payload["sub"], 
                payload["email"], 
                payload["role"]
            )
        return None
    
    def extract_user_id(self, token: str) -> Optional[str]:
        """Extract user ID from token without full validation."""
        try:
            # Decode without verification for user ID extraction
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload.get("sub")
        except Exception:
            return None
