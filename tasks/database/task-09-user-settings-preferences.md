# Task 09: User Settings and Preferences

## 📋 Task Overview

**Objective**: Implement comprehensive user settings and preferences management, allowing users to customize their MoRAG experience with personalized configurations, themes, and processing preferences.

**Priority**: Medium - Enhances user experience
**Estimated Time**: 1 week
**Dependencies**: Task 08 (Multi-tenancy Implementation)

## 🎯 Goals

1. Enhance user settings management system
2. Add processing preferences and defaults
3. Implement theme and UI customization
4. Create notification preferences
5. Add workspace and organization settings
6. Implement settings import/export
7. Create settings management API endpoints

## 📊 Current State Analysis

### Existing User Settings Model
- **Fields**: theme, language, notifications, auto_save, default_database
- **Features**: Basic user preferences
- **Limitations**: Limited customization options

### Required Enhancements
- **Processing Preferences**: Default chunk sizes, models, strategies
- **Notification Settings**: Granular notification control
- **Workspace Settings**: Custom workflows and templates
- **Integration Settings**: External service configurations

## 🔧 Implementation Plan

### Step 1: Enhance User Settings Models

**Files to Create/Modify**:
```
packages/morag-core/src/morag_core/
├── settings/
│   ├── __init__.py
│   ├── models.py          # Enhanced settings models
│   ├── service.py         # Settings management service
│   ├── defaults.py        # Default settings and templates
│   ├── validation.py      # Settings validation
│   └── migration.py       # Settings migration utilities
```

**Implementation Details**:

1. **Enhanced Settings Models**:
```python
# packages/morag-core/src/morag_core/settings/models.py
"""Enhanced user settings and preferences models."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class Theme(str, Enum):
    LIGHT = "LIGHT"
    DARK = "DARK"
    SYSTEM = "SYSTEM"
    HIGH_CONTRAST = "HIGH_CONTRAST"

class Language(str, Enum):
    EN = "en"
    DE = "de"
    FR = "fr"
    ES = "es"
    IT = "it"
    JA = "ja"
    ZH = "zh"

class ChunkingStrategy(str, Enum):
    CHARACTER = "CHARACTER"
    WORD = "WORD"
    SENTENCE = "SENTENCE"
    PARAGRAPH = "PARAGRAPH"
    CHAPTER = "CHAPTER"
    PAGE = "PAGE"

class NotificationChannel(str, Enum):
    EMAIL = "EMAIL"
    WEBHOOK = "WEBHOOK"
    IN_APP = "IN_APP"

class ProcessingPreferences(BaseModel):
    # Document Processing
    default_chunk_size: int = Field(default=4000, ge=500, le=16000)
    default_chunk_overlap: int = Field(default=200, ge=0, le=1000)
    default_chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    use_docling_by_default: bool = Field(default=False)
    
    # Audio Processing
    default_audio_model: str = Field(default="base")
    enable_speaker_diarization: bool = Field(default=True)
    enable_topic_segmentation: bool = Field(default=True)
    audio_language: Optional[str] = Field(default="auto")
    
    # Video Processing
    extract_thumbnails: bool = Field(default=False)
    video_quality: str = Field(default="medium")
    
    # Image Processing
    enable_ocr: bool = Field(default=True)
    image_quality: str = Field(default="high")
    
    # General Processing
    auto_process_uploads: bool = Field(default=True)
    enable_quality_scoring: bool = Field(default=True)
    max_concurrent_jobs: int = Field(default=3, ge=1, le=10)

class NotificationPreferences(BaseModel):
    # Job Notifications
    job_completed: Dict[NotificationChannel, bool] = Field(default_factory=lambda: {
        NotificationChannel.EMAIL: True,
        NotificationChannel.IN_APP: True,
        NotificationChannel.WEBHOOK: False
    })
    job_failed: Dict[NotificationChannel, bool] = Field(default_factory=lambda: {
        NotificationChannel.EMAIL: True,
        NotificationChannel.IN_APP: True,
        NotificationChannel.WEBHOOK: False
    })
    
    # System Notifications
    quota_warning: Dict[NotificationChannel, bool] = Field(default_factory=lambda: {
        NotificationChannel.EMAIL: True,
        NotificationChannel.IN_APP: True,
        NotificationChannel.WEBHOOK: False
    })
    quota_exceeded: Dict[NotificationChannel, bool] = Field(default_factory=lambda: {
        NotificationChannel.EMAIL: True,
        NotificationChannel.IN_APP: True,
        NotificationChannel.WEBHOOK: True
    })
    
    # Security Notifications
    login_alerts: Dict[NotificationChannel, bool] = Field(default_factory=lambda: {
        NotificationChannel.EMAIL: True,
        NotificationChannel.IN_APP: False,
        NotificationChannel.WEBHOOK: False
    })
    api_key_usage: Dict[NotificationChannel, bool] = Field(default_factory=lambda: {
        NotificationChannel.EMAIL: False,
        NotificationChannel.IN_APP: True,
        NotificationChannel.WEBHOOK: False
    })
    
    # Notification Settings
    email_digest_frequency: str = Field(default="daily")  # never, daily, weekly
    quiet_hours_start: Optional[str] = Field(default="22:00")
    quiet_hours_end: Optional[str] = Field(default="08:00")
    timezone: str = Field(default="UTC")

class WorkspacePreferences(BaseModel):
    # Dashboard Settings
    default_view: str = Field(default="documents")  # documents, jobs, analytics
    items_per_page: int = Field(default=25, ge=10, le=100)
    show_thumbnails: bool = Field(default=True)
    auto_refresh_interval: int = Field(default=30, ge=10, le=300)  # seconds
    
    # Search Settings
    search_suggestions: bool = Field(default=True)
    search_history_size: int = Field(default=50, ge=0, le=200)
    default_search_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # File Management
    auto_organize_uploads: bool = Field(default=False)
    default_upload_folder: Optional[str] = None
    file_naming_pattern: str = Field(default="{original_name}")
    
    # Collaboration
    share_processing_results: bool = Field(default=False)
    allow_public_links: bool = Field(default=False)

class IntegrationPreferences(BaseModel):
    # External Services
    webhook_endpoints: List[Dict[str, str]] = Field(default_factory=list)
    api_rate_limits: Dict[str, int] = Field(default_factory=dict)
    
    # Cloud Storage
    cloud_backup_enabled: bool = Field(default=False)
    cloud_provider: Optional[str] = None
    backup_frequency: str = Field(default="weekly")
    
    # Third-party Tools
    slack_integration: Dict[str, Any] = Field(default_factory=dict)
    teams_integration: Dict[str, Any] = Field(default_factory=dict)
    zapier_integration: Dict[str, Any] = Field(default_factory=dict)

class UserSettingsComplete(BaseModel):
    # Basic Settings
    theme: Theme = Theme.LIGHT
    language: Language = Language.EN
    timezone: str = Field(default="UTC")
    
    # Database Settings
    default_database_id: Optional[str] = None
    auto_save: bool = Field(default=True)
    
    # Processing Preferences
    processing: ProcessingPreferences = Field(default_factory=ProcessingPreferences)
    
    # Notification Preferences
    notifications: NotificationPreferences = Field(default_factory=NotificationPreferences)
    
    # Workspace Preferences
    workspace: WorkspacePreferences = Field(default_factory=WorkspacePreferences)
    
    # Integration Preferences
    integrations: IntegrationPreferences = Field(default_factory=IntegrationPreferences)
    
    # Custom Settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)

class UserSettingsUpdate(BaseModel):
    theme: Optional[Theme] = None
    language: Optional[Language] = None
    timezone: Optional[str] = None
    default_database_id: Optional[str] = None
    auto_save: Optional[bool] = None
    processing: Optional[ProcessingPreferences] = None
    notifications: Optional[NotificationPreferences] = None
    workspace: Optional[WorkspacePreferences] = None
    integrations: Optional[IntegrationPreferences] = None
    custom_settings: Optional[Dict[str, Any]] = None

class SettingsTemplate(BaseModel):
    name: str
    description: str
    category: str  # "processing", "notifications", "workspace", "complete"
    settings: Dict[str, Any]
    is_public: bool = Field(default=False)
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = Field(default=0)

class SettingsExport(BaseModel):
    user_id: str
    export_date: datetime
    settings: UserSettingsComplete
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

2. **Settings Service**:
```python
# packages/morag-core/src/morag_core/settings/service.py
"""Enhanced user settings management service."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import structlog
import json
from datetime import datetime

from morag_core.database import UserSettings, get_database_manager
from .models import (
    UserSettingsComplete, UserSettingsUpdate, ProcessingPreferences,
    NotificationPreferences, WorkspacePreferences, IntegrationPreferences,
    SettingsTemplate, SettingsExport
)
from .defaults import get_default_settings
from morag_core.exceptions import NotFoundError, ValidationError

logger = structlog.get_logger(__name__)

class UserSettingsService:
    """Enhanced user settings management service."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def get_user_settings(self, user_id: str) -> UserSettingsComplete:
        """Get complete user settings with defaults."""
        with self.db_manager.get_session() as session:
            settings = session.query(UserSettings).filter_by(user_id=user_id).first()
            
            if not settings:
                # Create default settings
                return self.create_default_settings(user_id)
            
            # Parse stored settings
            try:
                stored_settings = json.loads(settings.settings_json) if hasattr(settings, 'settings_json') else {}
            except (json.JSONDecodeError, AttributeError):
                stored_settings = {}
            
            # Merge with defaults
            default_settings = get_default_settings()
            merged_settings = self._merge_settings(default_settings, stored_settings)
            
            # Add basic fields from UserSettings model
            merged_settings.update({
                'theme': settings.theme.value if settings.theme else 'LIGHT',
                'language': settings.language or 'en',
                'default_database_id': settings.default_database,
                'auto_save': settings.auto_save if settings.auto_save is not None else True,
                'last_updated': settings.updated_at
            })
            
            return UserSettingsComplete(**merged_settings)
    
    def update_user_settings(self, user_id: str, settings_update: UserSettingsUpdate) -> UserSettingsComplete:
        """Update user settings."""
        with self.db_manager.get_session() as session:
            settings = session.query(UserSettings).filter_by(user_id=user_id).first()
            
            if not settings:
                # Create new settings record
                settings = UserSettings(user_id=user_id)
                session.add(settings)
            
            # Update basic fields
            if settings_update.theme is not None:
                from morag_core.database.models import Theme as DBTheme
                settings.theme = DBTheme(settings_update.theme.value)
            
            if settings_update.language is not None:
                settings.language = settings_update.language.value
            
            if settings_update.default_database_id is not None:
                settings.default_database = settings_update.default_database_id
            
            if settings_update.auto_save is not None:
                settings.auto_save = settings_update.auto_save
            
            # Update extended settings
            current_settings = self.get_user_settings(user_id)
            
            if settings_update.processing is not None:
                current_settings.processing = settings_update.processing
            
            if settings_update.notifications is not None:
                current_settings.notifications = settings_update.notifications
            
            if settings_update.workspace is not None:
                current_settings.workspace = settings_update.workspace
            
            if settings_update.integrations is not None:
                current_settings.integrations = settings_update.integrations
            
            if settings_update.custom_settings is not None:
                current_settings.custom_settings.update(settings_update.custom_settings)
            
            # Store extended settings as JSON
            extended_settings = {
                'processing': current_settings.processing.dict(),
                'notifications': current_settings.notifications.dict(),
                'workspace': current_settings.workspace.dict(),
                'integrations': current_settings.integrations.dict(),
                'custom_settings': current_settings.custom_settings,
                'version': current_settings.version + 1
            }
            
            # TODO: Add settings_json field to UserSettings model
            # settings.settings_json = json.dumps(extended_settings)
            
            logger.info("User settings updated", user_id=user_id)
            return self.get_user_settings(user_id)
    
    def create_default_settings(self, user_id: str) -> UserSettingsComplete:
        """Create default settings for new user."""
        default_settings = get_default_settings()
        
        with self.db_manager.get_session() as session:
            settings = UserSettings(
                user_id=user_id,
                theme=default_settings['theme'],
                language=default_settings['language'],
                notifications=default_settings['notifications'],
                auto_save=default_settings['auto_save']
            )
            session.add(settings)
            
            logger.info("Default settings created", user_id=user_id)
        
        return UserSettingsComplete(**default_settings)
    
    def export_settings(self, user_id: str) -> SettingsExport:
        """Export user settings for backup or transfer."""
        settings = self.get_user_settings(user_id)
        
        return SettingsExport(
            user_id=user_id,
            export_date=datetime.utcnow(),
            settings=settings,
            metadata={
                'export_version': '1.0',
                'morag_version': '0.1.0'
            }
        )
    
    def import_settings(self, user_id: str, settings_export: SettingsExport) -> UserSettingsComplete:
        """Import settings from export."""
        # Validate import
        if settings_export.user_id != user_id:
            raise ValidationError("Settings export user ID mismatch")
        
        # Convert to update format
        settings_update = UserSettingsUpdate(
            theme=settings_export.settings.theme,
            language=settings_export.settings.language,
            timezone=settings_export.settings.timezone,
            default_database_id=settings_export.settings.default_database_id,
            auto_save=settings_export.settings.auto_save,
            processing=settings_export.settings.processing,
            notifications=settings_export.settings.notifications,
            workspace=settings_export.settings.workspace,
            integrations=settings_export.settings.integrations,
            custom_settings=settings_export.settings.custom_settings
        )
        
        logger.info("Settings imported", user_id=user_id)
        return self.update_user_settings(user_id, settings_update)
    
    def reset_settings(self, user_id: str, category: Optional[str] = None) -> UserSettingsComplete:
        """Reset settings to defaults."""
        if category:
            # Reset specific category
            current_settings = self.get_user_settings(user_id)
            defaults = get_default_settings()
            
            if category == "processing":
                current_settings.processing = ProcessingPreferences()
            elif category == "notifications":
                current_settings.notifications = NotificationPreferences()
            elif category == "workspace":
                current_settings.workspace = WorkspacePreferences()
            elif category == "integrations":
                current_settings.integrations = IntegrationPreferences()
            
            settings_update = UserSettingsUpdate(**current_settings.dict())
            return self.update_user_settings(user_id, settings_update)
        else:
            # Reset all settings
            with self.db_manager.get_session() as session:
                settings = session.query(UserSettings).filter_by(user_id=user_id).first()
                if settings:
                    session.delete(settings)
            
            return self.create_default_settings(user_id)
    
    def _merge_settings(self, defaults: Dict[str, Any], stored: Dict[str, Any]) -> Dict[str, Any]:
        """Merge stored settings with defaults."""
        merged = defaults.copy()
        
        for key, value in stored.items():
            if key in merged:
                if isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key] = self._merge_settings(merged[key], value)
                else:
                    merged[key] = value
            else:
                merged[key] = value
        
        return merged
```

3. **Default Settings**:
```python
# packages/morag-core/src/morag_core/settings/defaults.py
"""Default settings and templates."""

from typing import Dict, Any
from .models import (
    Theme, Language, ProcessingPreferences, NotificationPreferences,
    WorkspacePreferences, IntegrationPreferences, NotificationChannel
)

def get_default_settings() -> Dict[str, Any]:
    """Get default user settings."""
    return {
        'theme': Theme.LIGHT,
        'language': Language.EN,
        'timezone': 'UTC',
        'default_database_id': None,
        'auto_save': True,
        'processing': ProcessingPreferences().dict(),
        'notifications': NotificationPreferences().dict(),
        'workspace': WorkspacePreferences().dict(),
        'integrations': IntegrationPreferences().dict(),
        'custom_settings': {},
        'version': 1
    }

def get_settings_templates() -> Dict[str, Dict[str, Any]]:
    """Get predefined settings templates."""
    return {
        'researcher': {
            'name': 'Researcher',
            'description': 'Optimized for academic research and document analysis',
            'processing': {
                'default_chunk_size': 6000,
                'default_chunking_strategy': 'PARAGRAPH',
                'use_docling_by_default': True,
                'enable_quality_scoring': True
            },
            'workspace': {
                'default_view': 'analytics',
                'items_per_page': 50,
                'show_thumbnails': False
            }
        },
        'content_creator': {
            'name': 'Content Creator',
            'description': 'Optimized for multimedia content processing',
            'processing': {
                'extract_thumbnails': True,
                'enable_speaker_diarization': True,
                'enable_topic_segmentation': True,
                'auto_process_uploads': True
            },
            'workspace': {
                'show_thumbnails': True,
                'auto_refresh_interval': 15
            }
        },
        'developer': {
            'name': 'Developer',
            'description': 'Optimized for API usage and automation',
            'notifications': {
                'job_completed': {
                    NotificationChannel.WEBHOOK: True,
                    NotificationChannel.EMAIL: False,
                    NotificationChannel.IN_APP: False
                }
            },
            'integrations': {
                'webhook_endpoints': [
                    {'name': 'Development Webhook', 'url': 'http://localhost:3000/webhook'}
                ]
            }
        }
    }
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_user_settings.py
import pytest
from morag_core.settings import UserSettingsService
from morag_core.settings.models import UserSettingsUpdate, ProcessingPreferences

def test_default_settings_creation():
    """Test default settings creation."""
    service = UserSettingsService()
    settings = service.create_default_settings("user123")
    
    assert settings.theme.value == "LIGHT"
    assert settings.language.value == "en"
    assert settings.processing.default_chunk_size == 4000
    assert settings.auto_save is True

def test_settings_update():
    """Test settings update."""
    service = UserSettingsService()
    
    # Create default settings
    settings = service.create_default_settings("user123")
    
    # Update processing preferences
    update = UserSettingsUpdate(
        processing=ProcessingPreferences(
            default_chunk_size=8000,
            enable_speaker_diarization=False
        )
    )
    
    updated_settings = service.update_user_settings("user123", update)
    assert updated_settings.processing.default_chunk_size == 8000
    assert updated_settings.processing.enable_speaker_diarization is False

def test_settings_export_import():
    """Test settings export and import."""
    service = UserSettingsService()
    
    # Create and modify settings
    settings = service.create_default_settings("user123")
    update = UserSettingsUpdate(theme="DARK")
    service.update_user_settings("user123", update)
    
    # Export settings
    export = service.export_settings("user123")
    assert export.user_id == "user123"
    assert export.settings.theme.value == "DARK"
    
    # Reset and import
    service.reset_settings("user123")
    imported_settings = service.import_settings("user123", export)
    assert imported_settings.theme.value == "DARK"
```

## 📋 Acceptance Criteria

- [ ] Enhanced user settings model implemented
- [ ] Processing preferences management working
- [ ] Notification preferences functional
- [ ] Workspace customization available
- [ ] Settings templates and defaults created
- [ ] Settings import/export functionality
- [ ] Settings validation and migration
- [ ] API endpoints for settings management
- [ ] Comprehensive unit tests passing
- [ ] Settings persistence and retrieval working

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 10: Testing and Validation](./task-10-testing-validation.md)
2. Add settings UI components
3. Test settings with different user scenarios
4. Implement settings synchronization across devices

## 📝 Notes

- Ensure backward compatibility with existing settings
- Add validation for all settings values
- Implement settings versioning for future migrations
- Consider settings inheritance for organization-level defaults
- Add comprehensive logging for settings changes
