"""Authentication and user management models."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    USER = "USER"
    VIEWER = "VIEWER"


class Theme(str, Enum):
    LIGHT = "LIGHT"
    DARK = "DARK"
    SYSTEM = "SYSTEM"


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    avatar: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    avatar: Optional[str]
    role: str  # Changed from UserRole to str for serialization
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=255)
    email: Optional[EmailStr] = None
    avatar: Optional[str] = None


class UserSettingsUpdate(BaseModel):
    theme: Optional[Theme] = None
    language: Optional[str] = Field(None, max_length=10)
    notifications: Optional[bool] = None
    auto_save: Optional[bool] = None
    default_database: Optional[str] = None


class UserSettingsResponse(BaseModel):
    theme: Theme
    language: str
    notifications: bool
    auto_save: bool
    default_database: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
