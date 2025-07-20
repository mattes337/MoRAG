"""Production configuration for Graphiti integration."""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..config import GraphitiConfig

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production-specific configuration for Graphiti."""
    
    # Environment settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    batch_size: int = 500
    cache_ttl: int = 3600  # 1 hour
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 60
    
    # Backup settings
    enable_backup: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    
    # Security settings
    enable_ssl: bool = True
    require_authentication: bool = True
    api_key_required: bool = True
    
    # Resource limits
    max_memory_mb: int = 8192
    max_cpu_cores: int = 4
    max_disk_usage_gb: int = 100


class ProductionConfigManager:
    """Manager for production configuration and environment setup."""
    
    def __init__(self):
        self.config = self._load_production_config()
        self.graphiti_config = self._create_graphiti_config()
    
    def _load_production_config(self) -> ProductionConfig:
        """Load production configuration from environment variables."""
        return ProductionConfig(
            environment=os.getenv("MORAG_ENVIRONMENT", "production"),
            debug_mode=os.getenv("MORAG_DEBUG", "false").lower() == "true",
            log_level=os.getenv("MORAG_LOG_LEVEL", "INFO"),
            
            max_concurrent_requests=int(os.getenv("MORAG_MAX_CONCURRENT_REQUESTS", "100")),
            request_timeout=int(os.getenv("MORAG_REQUEST_TIMEOUT", "30")),
            batch_size=int(os.getenv("MORAG_BATCH_SIZE", "500")),
            cache_ttl=int(os.getenv("MORAG_CACHE_TTL", "3600")),
            
            enable_metrics=os.getenv("MORAG_ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("MORAG_METRICS_PORT", "9090")),
            health_check_interval=int(os.getenv("MORAG_HEALTH_CHECK_INTERVAL", "60")),
            
            enable_backup=os.getenv("MORAG_ENABLE_BACKUP", "true").lower() == "true",
            backup_interval_hours=int(os.getenv("MORAG_BACKUP_INTERVAL_HOURS", "24")),
            backup_retention_days=int(os.getenv("MORAG_BACKUP_RETENTION_DAYS", "30")),
            
            enable_ssl=os.getenv("MORAG_ENABLE_SSL", "true").lower() == "true",
            require_authentication=os.getenv("MORAG_REQUIRE_AUTH", "true").lower() == "true",
            api_key_required=os.getenv("MORAG_API_KEY_REQUIRED", "true").lower() == "true",
            
            max_memory_mb=int(os.getenv("MORAG_MAX_MEMORY_MB", "8192")),
            max_cpu_cores=int(os.getenv("MORAG_MAX_CPU_CORES", "4")),
            max_disk_usage_gb=int(os.getenv("MORAG_MAX_DISK_USAGE_GB", "100"))
        )
    
    def _create_graphiti_config(self) -> GraphitiConfig:
        """Create Graphiti configuration for production."""
        return GraphitiConfig(
            neo4j_uri=os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("GRAPHITI_NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("GRAPHITI_NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("GRAPHITI_NEO4J_DATABASE", "morag_graphiti"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "default_key"),
            enable_telemetry=os.getenv("GRAPHITI_TELEMETRY_ENABLED", "false").lower() == "true"
        )
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate production configuration."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required environment variables
        required_vars = [
            "GRAPHITI_NEO4J_PASSWORD",
            "OPENAI_API_KEY"
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                validation_results["errors"].append(f"Missing required environment variable: {var}")
                validation_results["valid"] = False
        
        # Check resource limits
        if self.config.max_memory_mb < 4096:
            validation_results["warnings"].append("Memory limit below recommended 4GB")
        
        if self.config.max_cpu_cores < 2:
            validation_results["warnings"].append("CPU cores below recommended minimum of 2")
        
        # Check Neo4j connectivity
        try:
            # This would test actual Neo4j connection
            # For now, we'll just validate the URI format
            if not self.graphiti_config.neo4j_uri.startswith(("bolt://", "neo4j://")):
                validation_results["errors"].append("Invalid Neo4j URI format")
                validation_results["valid"] = False
        except Exception as e:
            validation_results["errors"].append(f"Neo4j configuration error: {str(e)}")
            validation_results["valid"] = False
        
        return validation_results
    
    def setup_logging(self):
        """Setup production logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('/var/log/morag/graphiti.log') if os.path.exists('/var/log/morag') else logging.NullHandler()
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('graphiti_core').setLevel(logging.WARNING)
        logging.getLogger('neo4j').setLevel(logging.WARNING)
        
        if not self.config.debug_mode:
            logging.getLogger('urllib3').setLevel(logging.WARNING)
            logging.getLogger('requests').setLevel(logging.WARNING)


# Global production config instance (lazy initialization)
_production_config = None

def get_production_config() -> ProductionConfigManager:
    """Get or create the global production config instance."""
    global _production_config
    if _production_config is None:
        _production_config = ProductionConfigManager()
    return _production_config

# For backward compatibility
production_config = get_production_config
