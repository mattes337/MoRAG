"""Migration management for MoRAG database."""

import os
from pathlib import Path
from typing import Optional
import structlog
from alembic.config import Config
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from morag_core.config import get_settings
from ..manager import get_database_manager

logger = structlog.get_logger(__name__)


class MigrationManager:
    """Manage database migrations using Alembic."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_settings().database_url
        self.db_manager = get_database_manager(database_url)
        
        # Set up Alembic configuration
        self.migrations_dir = Path(__file__).parent
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", str(self.migrations_dir))
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)

    def init_migrations(self) -> bool:
        """Initialize migration environment."""
        try:
            # Create alembic.ini if it doesn't exist
            alembic_ini_path = self.migrations_dir / "alembic.ini"
            if not alembic_ini_path.exists():
                self._create_alembic_ini()

            # Initialize Alembic if not already initialized
            versions_dir = self.migrations_dir / "versions"
            if not versions_dir.exists():
                command.init(self.alembic_cfg, str(self.migrations_dir))
                logger.info("Migration environment initialized")

            return True
        except Exception as e:
            logger.error("Failed to initialize migrations", error=str(e))
            return False

    def create_migration(self, message: str, auto_generate: bool = True) -> Optional[str]:
        """Create a new migration."""
        try:
            if auto_generate:
                # Auto-generate migration based on model changes
                revision = command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True
                )
            else:
                # Create empty migration
                revision = command.revision(
                    self.alembic_cfg,
                    message=message
                )

            logger.info("Migration created", message=message, revision=revision)
            return revision
        except Exception as e:
            logger.error("Failed to create migration", error=str(e), message=message)
            return None

    def upgrade_database(self, revision: str = "head") -> bool:
        """Upgrade database to specified revision."""
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info("Database upgraded", revision=revision)
            return True
        except Exception as e:
            logger.error("Failed to upgrade database", error=str(e), revision=revision)
            return False

    def downgrade_database(self, revision: str) -> bool:
        """Downgrade database to specified revision."""
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info("Database downgraded", revision=revision)
            return True
        except Exception as e:
            logger.error("Failed to downgrade database", error=str(e), revision=revision)
            return False

    def get_current_revision(self) -> Optional[str]:
        """Get current database revision."""
        try:
            with self.db_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            logger.error("Failed to get current revision", error=str(e))
            return None

    def get_migration_history(self) -> list:
        """Get migration history."""
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = []
            
            for revision in script.walk_revisions():
                revisions.append({
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "message": revision.doc,
                    "branch_labels": revision.branch_labels,
                })
            
            return revisions
        except Exception as e:
            logger.error("Failed to get migration history", error=str(e))
            return []

    def check_migration_status(self) -> dict:
        """Check migration status."""
        try:
            current = self.get_current_revision()
            script = ScriptDirectory.from_config(self.alembic_cfg)
            head = script.get_current_head()
            
            pending_migrations = []
            if current != head:
                # Get pending migrations
                for revision in script.iterate_revisions(current, head):
                    if revision.revision != current:
                        pending_migrations.append({
                            "revision": revision.revision,
                            "message": revision.doc
                        })

            return {
                "current_revision": current,
                "head_revision": head,
                "up_to_date": current == head,
                "pending_migrations": pending_migrations
            }
        except Exception as e:
            logger.error("Failed to check migration status", error=str(e))
            return {
                "current_revision": None,
                "head_revision": None,
                "up_to_date": False,
                "pending_migrations": [],
                "error": str(e)
            }

    def _create_alembic_ini(self):
        """Create alembic.ini configuration file."""
        alembic_ini_content = """# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = %(here)s

# template used to generate migration files
# file_template = %%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
# defaults to the current working directory.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python-dateutil library that can be
# installed by adding `alembic[tz]` to the pip requirements
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version number format
version_num_format = %04d

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
# Valid values for version_path_separator are:
#
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = driver://user:pass@localhost/dbname


[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79 REVISION_SCRIPT_FILENAME

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        
        alembic_ini_path = self.migrations_dir / "alembic.ini"
        with open(alembic_ini_path, "w") as f:
            f.write(alembic_ini_content)
