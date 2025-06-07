#!/usr/bin/env python3
"""Migration script to help transition from monolithic to modular MoRAG architecture."""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoRAGMigrator:
    """Handles migration from monolithic to modular architecture."""
    
    def __init__(self, source_dir: Path, backup_dir: Path):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.migration_log = []
    
    def create_backup(self) -> None:
        """Create backup of existing installation."""
        logger.info(f"Creating backup in {self.backup_dir}")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        shutil.copytree(self.source_dir, self.backup_dir)
        logger.info("Backup created successfully")
    
    def migrate_configuration(self) -> Dict[str, Any]:
        """Migrate configuration files to new format."""
        logger.info("Migrating configuration files")
        
        old_config_paths = [
            self.source_dir / "config.json",
            self.source_dir / "settings.py",
            self.source_dir / ".env"
        ]
        
        new_config = {
            "gemini_api_key": "",
            "qdrant_host": "localhost",
            "qdrant_port": 6333,
            "qdrant_collection_name": "morag_vectors",
            "redis_url": "redis://localhost:6379/0",
            "max_workers": 4,
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        # Try to extract configuration from old files
        for config_path in old_config_paths:
            if config_path.exists():
                logger.info(f"Found old config file: {config_path}")
                
                if config_path.suffix == ".json":
                    with open(config_path) as f:
                        old_config = json.load(f)
                        new_config.update(old_config)
                
                elif config_path.name == ".env":
                    with open(config_path) as f:
                        for line in f:
                            if "=" in line and not line.startswith("#"):
                                key, value = line.strip().split("=", 1)
                                key = key.lower()
                                if key in new_config:
                                    new_config[key] = value.strip('"\'')
        
        # Save new configuration
        new_config_path = self.source_dir / "morag_config.json"
        with open(new_config_path, "w") as f:
            json.dump(new_config, f, indent=2)
        
        logger.info(f"New configuration saved to {new_config_path}")
        return new_config
    
    def migrate_data(self) -> None:
        """Migrate data files and databases."""
        logger.info("Migrating data files")
        
        # Create data directory structure
        data_dirs = [
            "data/vectors",
            "data/documents", 
            "data/cache",
            "data/logs"
        ]
        
        for dir_path in data_dirs:
            (self.source_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Move existing data files
        old_data_paths = [
            "vectors.db",
            "documents.db",
            "cache/",
            "logs/",
            "temp/"
        ]
        
        for old_path in old_data_paths:
            old_full_path = self.source_dir / old_path
            if old_full_path.exists():
                if old_full_path.is_file():
                    new_path = self.source_dir / "data" / old_path
                    shutil.move(str(old_full_path), str(new_path))
                    logger.info(f"Moved {old_path} to data/{old_path}")
                elif old_full_path.is_dir():
                    new_path = self.source_dir / "data" / old_path
                    if new_path.exists():
                        shutil.rmtree(new_path)
                    shutil.move(str(old_full_path), str(new_path))
                    logger.info(f"Moved {old_path}/ to data/{old_path}/")
    
    def create_docker_setup(self) -> None:
        """Create Docker setup files."""
        logger.info("Creating Docker setup files")
        
        # Create .env file for Docker Compose
        env_content = """# MoRAG Environment Configuration
GEMINI_API_KEY=your-gemini-api-key-here
QDRANT_HOST=qdrant
QDRANT_PORT=6333
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO

# Optional: Qdrant Cloud Configuration
# QDRANT_API_KEY=your-qdrant-cloud-api-key
# QDRANT_HOST=your-cluster-url.qdrant.io

# Optional: Custom Redis Configuration
# REDIS_URL=redis://your-redis-host:6379/0
"""
        
        with open(self.source_dir / ".env", "w") as f:
            f.write(env_content)
        
        # Create docker-compose.override.yml for local development
        override_content = """version: '3.8'

services:
  morag-api:
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - DEBUG=true
    
  morag-worker:
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - DEBUG=true
"""
        
        with open(self.source_dir / "docker-compose.override.yml", "w") as f:
            f.write(override_content)
        
        logger.info("Docker setup files created")
    
    def cleanup_old_files(self) -> None:
        """Remove obsolete files and directories."""
        logger.info("Cleaning up obsolete files")
        
        obsolete_paths = [
            "src/morag/processors/",
            "src/morag/converters/",
            "src/morag/tasks/",
            "src/morag/services/",
            "old_main.py",
            "legacy_api.py",
            "requirements_old.txt"
        ]
        
        for path in obsolete_paths:
            full_path = self.source_dir / path
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                    logger.info(f"Removed file: {path}")
                elif full_path.is_dir():
                    shutil.rmtree(full_path)
                    logger.info(f"Removed directory: {path}")
    
    def create_migration_report(self) -> None:
        """Create a migration report."""
        logger.info("Creating migration report")
        
        report = {
            "migration_date": str(Path.cwd()),
            "source_directory": str(self.source_dir),
            "backup_directory": str(self.backup_dir),
            "migration_steps": self.migration_log,
            "next_steps": [
                "1. Review the new configuration in morag_config.json",
                "2. Update your API keys and service URLs",
                "3. Install the new modular packages: pip install morag",
                "4. Test the new system: morag health",
                "5. Start services: docker-compose up -d",
                "6. Migrate your existing data if needed",
                "7. Update your application code to use the new API"
            ],
            "breaking_changes": [
                "Import paths have changed - update your code",
                "Configuration format has changed",
                "API endpoints may have different signatures",
                "Some legacy features may not be available"
            ]
        }
        
        with open(self.source_dir / "migration_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("Migration report saved to migration_report.json")
    
    def run_migration(self) -> None:
        """Run the complete migration process."""
        logger.info("Starting MoRAG migration to modular architecture")
        
        try:
            self.create_backup()
            self.migration_log.append("Created backup")
            
            config = self.migrate_configuration()
            self.migration_log.append("Migrated configuration")
            
            self.migrate_data()
            self.migration_log.append("Migrated data files")
            
            self.create_docker_setup()
            self.migration_log.append("Created Docker setup")
            
            self.cleanup_old_files()
            self.migration_log.append("Cleaned up obsolete files")
            
            self.create_migration_report()
            self.migration_log.append("Created migration report")
            
            logger.info("Migration completed successfully!")
            logger.info("Please review migration_report.json for next steps")
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            logger.info(f"Backup is available at: {self.backup_dir}")
            raise


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(description="Migrate MoRAG to modular architecture")
    parser.add_argument("--source", "-s", default=".", help="Source directory (default: current)")
    parser.add_argument("--backup", "-b", default="./backup_pre_migration", help="Backup directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source).resolve()
    backup_dir = Path(args.backup).resolve()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info(f"Would migrate: {source_dir}")
        logger.info(f"Would create backup: {backup_dir}")
        return
    
    # Confirm migration
    print(f"This will migrate MoRAG installation at: {source_dir}")
    print(f"Backup will be created at: {backup_dir}")
    print("\nThis is a potentially destructive operation.")
    
    confirm = input("Do you want to continue? (yes/no): ").lower().strip()
    if confirm not in ["yes", "y"]:
        print("Migration cancelled")
        return
    
    migrator = MoRAGMigrator(source_dir, backup_dir)
    migrator.run_migration()


if __name__ == "__main__":
    main()
