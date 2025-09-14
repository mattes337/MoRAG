# Quick Win 9: Configuration-Based Tuning

## Overview

**Priority**: 🔄 **Maintenance** (1 week, Low Impact, Low ROI)  
**Source**: General system flexibility principles  
**Expected Impact**: Easier optimization for different domains and use cases

## Problem Statement

MoRAG currently has many hard-coded parameters throughout the system:
- Entity extraction confidence thresholds
- Graph traversal depth limits
- Similarity thresholds for retrieval
- Chunk size and overlap parameters
- Caching policies and TTL values
- Query processing timeouts

This makes it difficult to optimize the system for different domains, use cases, or performance requirements without code changes.

## Solution Overview

Externalize key parameters into configuration files with environment-specific overrides, runtime parameter adjustment, and validation to enable easy tuning and optimization without code deployment.

## Technical Implementation

### 1. Configuration Management System

Create `packages/morag-core/src/morag_core/config/config_manager.py`:

```python
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
import os
from enum import Enum

class ConfigScope(Enum):
    GLOBAL = "global"
    COLLECTION = "collection"
    USER = "user"
    SESSION = "session"

@dataclass
class ConfigParameter:
    name: str
    value: Any
    default_value: Any
    description: str
    data_type: type
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    scope: ConfigScope = ConfigScope.GLOBAL
    requires_restart: bool = False

@dataclass
class ConfigSection:
    name: str
    description: str
    parameters: Dict[str, ConfigParameter] = field(default_factory=dict)

class ConfigManager:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.sections = {}
        self.overrides = {}
        self.environment_configs = {}
        
        # Load configuration schema
        self._load_config_schema()
        
        # Load base configurations
        self._load_base_configs()
        
        # Apply environment-specific overrides
        self._load_environment_overrides()

    def get(self, section: str, parameter: str, scope_context: Dict[str, str] = None) -> Any:
        """Get configuration value with scope-aware resolution."""
        
        # Build lookup key with scope context
        lookup_keys = [f"{section}.{parameter}"]
        
        if scope_context:
            # Add scoped lookup keys in priority order
            if 'session_id' in scope_context:
                lookup_keys.insert(0, f"session.{scope_context['session_id']}.{section}.{parameter}")
            if 'user_id' in scope_context:
                lookup_keys.insert(0, f"user.{scope_context['user_id']}.{section}.{parameter}")
            if 'collection_name' in scope_context:
                lookup_keys.insert(0, f"collection.{scope_context['collection_name']}.{section}.{parameter}")
        
        # Try each lookup key in priority order
        for key in lookup_keys:
            if key in self.overrides:
                return self.overrides[key]
        
        # Fall back to base configuration
        if section in self.sections and parameter in self.sections[section].parameters:
            return self.sections[section].parameters[parameter].value
        
        # Return default if available
        if section in self.sections and parameter in self.sections[section].parameters:
            return self.sections[section].parameters[parameter].default_value
        
        raise KeyError(f"Configuration parameter {section}.{parameter} not found")

    def set(self, section: str, parameter: str, value: Any, 
            scope_context: Dict[str, str] = None, validate: bool = True) -> bool:
        """Set configuration value with validation."""
        
        # Validate parameter exists
        if section not in self.sections or parameter not in self.sections[section].parameters:
            raise KeyError(f"Configuration parameter {section}.{parameter} not found")
        
        param_config = self.sections[section].parameters[parameter]
        
        # Validate value if requested
        if validate:
            if not self._validate_value(param_config, value):
                raise ValueError(f"Invalid value {value} for parameter {section}.{parameter}")
        
        # Build override key
        if scope_context:
            if 'session_id' in scope_context:
                key = f"session.{scope_context['session_id']}.{section}.{parameter}"
            elif 'user_id' in scope_context:
                key = f"user.{scope_context['user_id']}.{section}.{parameter}"
            elif 'collection_name' in scope_context:
                key = f"collection.{scope_context['collection_name']}.{section}.{parameter}"
            else:
                key = f"{section}.{parameter}"
        else:
            key = f"{section}.{parameter}"
        
        # Set override
        self.overrides[key] = value
        
        return not param_config.requires_restart

    def get_section(self, section_name: str, scope_context: Dict[str, str] = None) -> Dict[str, Any]:
        """Get all parameters in a section."""
        if section_name not in self.sections:
            raise KeyError(f"Configuration section {section_name} not found")
        
        result = {}
        for param_name in self.sections[section_name].parameters:
            result[param_name] = self.get(section_name, param_name, scope_context)
        
        return result

    def list_parameters(self, section: str = None) -> Dict[str, Dict[str, Any]]:
        """List all available parameters with their metadata."""
        result = {}
        
        sections_to_process = [section] if section else self.sections.keys()
        
        for section_name in sections_to_process:
            if section_name in self.sections:
                result[section_name] = {}
                for param_name, param_config in self.sections[section_name].parameters.items():
                    result[section_name][param_name] = {
                        'current_value': self.get(section_name, param_name),
                        'default_value': param_config.default_value,
                        'description': param_config.description,
                        'data_type': param_config.data_type.__name__,
                        'min_value': param_config.min_value,
                        'max_value': param_config.max_value,
                        'allowed_values': param_config.allowed_values,
                        'scope': param_config.scope.value,
                        'requires_restart': param_config.requires_restart
                    }
        
        return result

    def save_overrides(self, file_path: str = None):
        """Save current overrides to file."""
        if not file_path:
            file_path = self.config_dir / "overrides.yaml"
        
        with open(file_path, 'w') as f:
            yaml.dump(self.overrides, f, default_flow_style=False)

    def load_overrides(self, file_path: str):
        """Load overrides from file."""
        with open(file_path, 'r') as f:
            loaded_overrides = yaml.safe_load(f)
            self.overrides.update(loaded_overrides)

    def reset_to_defaults(self, section: str = None, parameter: str = None):
        """Reset configuration to defaults."""
        if section and parameter:
            # Reset specific parameter
            key = f"{section}.{parameter}"
            if key in self.overrides:
                del self.overrides[key]
        elif section:
            # Reset entire section
            keys_to_remove = [k for k in self.overrides.keys() if k.startswith(f"{section}.")]
            for key in keys_to_remove:
                del self.overrides[key]
        else:
            # Reset all
            self.overrides.clear()

    def _load_config_schema(self):
        """Load configuration schema defining all available parameters."""
        
        # Entity Extraction Configuration
        entity_section = ConfigSection(
            name="entity_extraction",
            description="Entity extraction and normalization parameters"
        )
        
        entity_section.parameters["confidence_threshold"] = ConfigParameter(
            name="confidence_threshold",
            value=0.4,
            default_value=0.4,
            description="Minimum confidence threshold for entity extraction",
            data_type=float,
            min_value=0.0,
            max_value=1.0,
            scope=ConfigScope.COLLECTION
        )
        
        entity_section.parameters["max_entities_per_chunk"] = ConfigParameter(
            name="max_entities_per_chunk",
            value=50,
            default_value=50,
            description="Maximum number of entities to extract per chunk",
            data_type=int,
            min_value=1,
            max_value=200,
            scope=ConfigScope.COLLECTION
        )
        
        entity_section.parameters["normalization_enabled"] = ConfigParameter(
            name="normalization_enabled",
            value=True,
            default_value=True,
            description="Enable entity normalization",
            data_type=bool,
            scope=ConfigScope.COLLECTION
        )
        
        self.sections["entity_extraction"] = entity_section
        
        # Graph Traversal Configuration
        graph_section = ConfigSection(
            name="graph_traversal",
            description="Graph traversal and reasoning parameters"
        )
        
        graph_section.parameters["max_hops"] = ConfigParameter(
            name="max_hops",
            value=3,
            default_value=3,
            description="Maximum number of hops for graph traversal",
            data_type=int,
            min_value=1,
            max_value=10,
            scope=ConfigScope.COLLECTION
        )
        
        graph_section.parameters["similarity_threshold"] = ConfigParameter(
            name="similarity_threshold",
            value=0.7,
            default_value=0.7,
            description="Similarity threshold for entity matching",
            data_type=float,
            min_value=0.0,
            max_value=1.0,
            scope=ConfigScope.COLLECTION
        )
        
        graph_section.parameters["max_results"] = ConfigParameter(
            name="max_results",
            value=10,
            default_value=10,
            description="Maximum number of results to return",
            data_type=int,
            min_value=1,
            max_value=100,
            scope=ConfigScope.SESSION
        )
        
        self.sections["graph_traversal"] = graph_section
        
        # Query Processing Configuration
        query_section = ConfigSection(
            name="query_processing",
            description="Query processing and response generation parameters"
        )
        
        query_section.parameters["timeout_seconds"] = ConfigParameter(
            name="timeout_seconds",
            value=30,
            default_value=30,
            description="Query processing timeout in seconds",
            data_type=int,
            min_value=5,
            max_value=300,
            scope=ConfigScope.GLOBAL
        )
        
        query_section.parameters["max_context_tokens"] = ConfigParameter(
            name="max_context_tokens",
            value=4000,
            default_value=4000,
            description="Maximum context tokens for LLM",
            data_type=int,
            min_value=1000,
            max_value=32000,
            scope=ConfigScope.COLLECTION
        )
        
        query_section.parameters["enable_caching"] = ConfigParameter(
            name="enable_caching",
            value=True,
            default_value=True,
            description="Enable query result caching",
            data_type=bool,
            scope=ConfigScope.GLOBAL
        )
        
        self.sections["query_processing"] = query_section
        
        # Performance Configuration
        perf_section = ConfigSection(
            name="performance",
            description="Performance and optimization parameters"
        )
        
        perf_section.parameters["batch_size"] = ConfigParameter(
            name="batch_size",
            value=32,
            default_value=32,
            description="Batch size for processing operations",
            data_type=int,
            min_value=1,
            max_value=256,
            scope=ConfigScope.GLOBAL,
            requires_restart=True
        )
        
        perf_section.parameters["worker_threads"] = ConfigParameter(
            name="worker_threads",
            value=4,
            default_value=4,
            description="Number of worker threads",
            data_type=int,
            min_value=1,
            max_value=32,
            scope=ConfigScope.GLOBAL,
            requires_restart=True
        )
        
        self.sections["performance"] = perf_section

    def _load_base_configs(self):
        """Load base configuration files."""
        config_files = [
            "entity_extraction.yml",
            "graph_traversal.yml", 
            "query_processing.yml",
            "performance.yml"
        ]
        
        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self._apply_config_data(config_data)

    def _load_environment_overrides(self):
        """Load environment-specific configuration overrides."""
        env = os.getenv('MORAG_ENV', 'development')
        env_file = self.config_dir / f"{env}.yml"
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_config = yaml.safe_load(f)
                self._apply_config_data(env_config)

    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to parameters."""
        for section_name, section_data in config_data.items():
            if section_name in self.sections:
                for param_name, param_value in section_data.items():
                    if param_name in self.sections[section_name].parameters:
                        self.sections[section_name].parameters[param_name].value = param_value

    def _validate_value(self, param_config: ConfigParameter, value: Any) -> bool:
        """Validate a configuration value."""
        
        # Type check
        if not isinstance(value, param_config.data_type):
            try:
                value = param_config.data_type(value)
            except (ValueError, TypeError):
                return False
        
        # Range check for numeric values
        if param_config.min_value is not None and value < param_config.min_value:
            return False
        if param_config.max_value is not None and value > param_config.max_value:
            return False
        
        # Allowed values check
        if param_config.allowed_values is not None and value not in param_config.allowed_values:
            return False
        
        return True
```

### 2. Configuration API Endpoints

Create `packages/morag-services/src/morag_services/api/config_endpoints.py`:

```python
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from pydantic import BaseModel
from ..config_manager import ConfigManager

router = APIRouter(prefix="/config", tags=["configuration"])

class ConfigUpdateRequest(BaseModel):
    section: str
    parameter: str
    value: Any
    collection_name: Optional[str] = None
    user_id: Optional[str] = None

@router.get("/sections")
async def list_sections():
    """List all configuration sections."""
    config_manager = ConfigManager()
    return list(config_manager.sections.keys())

@router.get("/sections/{section_name}")
async def get_section(section_name: str, 
                     collection_name: Optional[str] = Query(None),
                     user_id: Optional[str] = Query(None)):
    """Get all parameters in a configuration section."""
    config_manager = ConfigManager()
    
    scope_context = {}
    if collection_name:
        scope_context['collection_name'] = collection_name
    if user_id:
        scope_context['user_id'] = user_id
    
    try:
        return config_manager.get_section(section_name, scope_context)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/parameters")
async def list_all_parameters():
    """List all available configuration parameters with metadata."""
    config_manager = ConfigManager()
    return config_manager.list_parameters()

@router.get("/parameters/{section_name}/{parameter_name}")
async def get_parameter(section_name: str, 
                       parameter_name: str,
                       collection_name: Optional[str] = Query(None),
                       user_id: Optional[str] = Query(None)):
    """Get a specific configuration parameter."""
    config_manager = ConfigManager()
    
    scope_context = {}
    if collection_name:
        scope_context['collection_name'] = collection_name
    if user_id:
        scope_context['user_id'] = user_id
    
    try:
        value = config_manager.get(section_name, parameter_name, scope_context)
        return {"value": value}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/parameters")
async def update_parameter(request: ConfigUpdateRequest):
    """Update a configuration parameter."""
    config_manager = ConfigManager()
    
    scope_context = {}
    if request.collection_name:
        scope_context['collection_name'] = request.collection_name
    if request.user_id:
        scope_context['user_id'] = request.user_id
    
    try:
        can_apply_immediately = config_manager.set(
            request.section, 
            request.parameter, 
            request.value, 
            scope_context
        )
        
        return {
            "success": True,
            "requires_restart": not can_apply_immediately,
            "message": "Parameter updated successfully"
        }
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reset")
async def reset_configuration(section: Optional[str] = None, 
                            parameter: Optional[str] = None):
    """Reset configuration to defaults."""
    config_manager = ConfigManager()
    config_manager.reset_to_defaults(section, parameter)
    
    return {"success": True, "message": "Configuration reset to defaults"}

@router.post("/save")
async def save_configuration():
    """Save current configuration overrides."""
    config_manager = ConfigManager()
    config_manager.save_overrides()
    
    return {"success": True, "message": "Configuration saved"}
```

### 3. Configuration CLI Tool

Create `cli/config-manager.py`:

```python
import argparse
import asyncio
import yaml
from morag_core.config.config_manager import ConfigManager

def main():
    parser = argparse.ArgumentParser(description='MoRAG Configuration Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List configuration parameters')
    list_parser.add_argument('--section', help='Specific section to list')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get configuration value')
    get_parser.add_argument('section', help='Configuration section')
    get_parser.add_argument('parameter', help='Parameter name')
    get_parser.add_argument('--collection', help='Collection context')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set configuration value')
    set_parser.add_argument('section', help='Configuration section')
    set_parser.add_argument('parameter', help='Parameter name')
    set_parser.add_argument('value', help='Parameter value')
    set_parser.add_argument('--collection', help='Collection context')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset to defaults')
    reset_parser.add_argument('--section', help='Section to reset')
    reset_parser.add_argument('--parameter', help='Parameter to reset')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    if args.command == 'list':
        parameters = config_manager.list_parameters(args.section)
        print(yaml.dump(parameters, default_flow_style=False))
        
    elif args.command == 'get':
        scope_context = {'collection_name': args.collection} if args.collection else None
        try:
            value = config_manager.get(args.section, args.parameter, scope_context)
            print(f"{args.section}.{args.parameter} = {value}")
        except KeyError as e:
            print(f"Error: {e}")
            
    elif args.command == 'set':
        scope_context = {'collection_name': args.collection} if args.collection else None
        try:
            # Try to convert value to appropriate type
            value = args.value
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
            
            can_apply = config_manager.set(args.section, args.parameter, value, scope_context)
            print(f"Set {args.section}.{args.parameter} = {value}")
            if not can_apply:
                print("Warning: This change requires a system restart to take effect")
        except (KeyError, ValueError) as e:
            print(f"Error: {e}")
            
    elif args.command == 'reset':
        config_manager.reset_to_defaults(args.section, args.parameter)
        print("Configuration reset to defaults")
        
    elif args.command == 'validate':
        # Validate current configuration
        print("Configuration validation not yet implemented")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### 4. Integration with Services

Update services to use configuration manager:

```python
# packages/morag-graph/src/morag_graph/entity_extractor.py

from morag_core.config.config_manager import ConfigManager

class EntityExtractor:
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
    
    async def extract_entities(self, text: str, collection_name: str = None) -> List[Dict[str, Any]]:
        """Extract entities using configurable parameters."""
        
        scope_context = {'collection_name': collection_name} if collection_name else None
        
        # Get configuration values
        confidence_threshold = self.config.get('entity_extraction', 'confidence_threshold', scope_context)
        max_entities = self.config.get('entity_extraction', 'max_entities_per_chunk', scope_context)
        normalization_enabled = self.config.get('entity_extraction', 'normalization_enabled', scope_context)
        
        # Use configuration in extraction logic
        entities = await self._extract_entities_internal(text, confidence_threshold, max_entities)
        
        if normalization_enabled:
            entities = await self._normalize_entities(entities)
        
        return entities
```

## Configuration Files

Create base configuration files:

```yaml
# configs/entity_extraction.yml
entity_extraction:
  confidence_threshold: 0.7
  max_entities_per_chunk: 50
  normalization_enabled: true
  extraction_timeout: 30

# configs/graph_traversal.yml  
graph_traversal:
  max_hops: 3
  similarity_threshold: 0.7
  max_results: 10
  traversal_timeout: 15

# configs/query_processing.yml
query_processing:
  timeout_seconds: 30
  max_context_tokens: 4000
  enable_caching: true
  response_streaming: false

# configs/performance.yml
performance:
  batch_size: 32
  worker_threads: 4
  memory_limit_mb: 2048
  enable_gpu: true
```

## Testing Strategy

```python
# tests/unit/test_config_manager.py
import pytest
from morag_core.config.config_manager import ConfigManager

class TestConfigManager:
    def setup_method(self):
        self.config = ConfigManager()

    def test_get_default_value(self):
        value = self.config.get('entity_extraction', 'confidence_threshold')
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_set_and_get_value(self):
        self.config.set('entity_extraction', 'confidence_threshold', 0.8)
        value = self.config.get('entity_extraction', 'confidence_threshold')
        assert value == 0.8

    def test_scope_context(self):
        # Test collection-specific configuration
        self.config.set('entity_extraction', 'confidence_threshold', 0.9, 
                       {'collection_name': 'test_collection'})
        
        # Global value should be unchanged
        global_value = self.config.get('entity_extraction', 'confidence_threshold')
        assert global_value != 0.9
        
        # Collection-specific value should be set
        collection_value = self.config.get('entity_extraction', 'confidence_threshold',
                                         {'collection_name': 'test_collection'})
        assert collection_value == 0.9
```

## Success Metrics

- **Configuration Coverage**: >80% of tunable parameters externalized
- **Deployment Flexibility**: Zero-downtime parameter updates for 90% of settings
- **Domain Optimization**: Easy optimization for different use cases
- **Operational Efficiency**: 50% reduction in configuration-related deployments

## Future Enhancements

1. **Web UI**: Configuration management interface
2. **A/B Testing**: Automatic parameter optimization
3. **Configuration Validation**: Advanced validation rules and dependencies
4. **Hot Reloading**: Dynamic configuration updates without restart
