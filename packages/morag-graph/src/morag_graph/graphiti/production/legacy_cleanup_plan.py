"""Legacy cleanup plan for Neo4j components after Graphiti migration."""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LegacyCleanupPlan:
    """Comprehensive plan for cleaning up legacy Neo4j components."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def analyze_legacy_components(self) -> Dict[str, Any]:
        """Analyze legacy Neo4j components that can be cleaned up."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "safe_to_remove": [],
                "migration_dependent": [],
                "keep_for_compatibility": [],
                "requires_review": []
            },
            "files_to_modify": [],
            "dependencies_to_remove": [],
            "configuration_changes": []
        }
        
        # Components safe to remove after Graphiti migration
        analysis["components"]["safe_to_remove"] = [
            {
                "component": "Direct Neo4j usage in ingestion",
                "files": [
                    "packages/morag/src/morag/ingestion_coordinator.py",
                    "packages/morag-services/src/morag_services/graph_processor.py"
                ],
                "description": "Direct Neo4j storage usage that can be replaced with Graphiti",
                "impact": "Low - Graphiti provides equivalent functionality"
            },
            {
                "component": "Neo4j-specific test utilities",
                "files": [
                    "check_neo4j.py"
                ],
                "description": "Standalone Neo4j testing scripts",
                "impact": "None - development utilities only"
            }
        ]
        
        # Components that depend on migration completion
        analysis["components"]["migration_dependent"] = [
            {
                "component": "Neo4j storage backend",
                "files": [
                    "packages/morag-graph/src/morag_graph/storage/neo4j_storage.py"
                ],
                "description": "Core Neo4j storage implementation",
                "impact": "High - Only remove after complete migration validation",
                "prerequisite": "All data migrated and validated in Graphiti"
            },
            {
                "component": "Neo4j configuration",
                "files": [
                    "packages/morag-graph/src/morag_graph/storage/__init__.py",
                    "packages/morag/src/morag/database_factory.py"
                ],
                "description": "Neo4j configuration and factory methods",
                "impact": "Medium - Remove after migration tools no longer needed"
            }
        ]
        
        # Components to keep for backward compatibility
        analysis["components"]["keep_for_compatibility"] = [
            {
                "component": "Neo4j imports in dependencies",
                "files": [
                    "packages/morag/src/morag/dependencies.py"
                ],
                "description": "Graceful degradation imports",
                "impact": "None - Maintains backward compatibility",
                "reason": "Allows existing code to work without Neo4j"
            },
            {
                "component": "Migration utilities",
                "files": [
                    "packages/morag-graph/src/morag_graph/graphiti/migration_utils.py"
                ],
                "description": "Neo4j to Graphiti migration tools",
                "impact": "None - May be needed for future migrations",
                "reason": "Useful for data recovery or re-migration"
            }
        ]
        
        # Components requiring manual review
        analysis["components"]["requires_review"] = [
            {
                "component": "Test backward compatibility",
                "files": [
                    "packages/morag-graph/tests/test_graphiti_integration.py"
                ],
                "description": "Tests that verify Neo4j imports still work",
                "impact": "Low - May need updating based on cleanup decisions",
                "action": "Review and update tests based on what's kept"
            }
        ]
        
        # Files that need modification (not removal)
        analysis["files_to_modify"] = [
            {
                "file": "packages/morag-graph/pyproject.toml",
                "changes": [
                    "Consider making neo4j dependency optional",
                    "Update version constraints if needed"
                ],
                "impact": "Low - Dependency management"
            },
            {
                "file": "packages/morag/src/morag/ingestion_coordinator.py",
                "changes": [
                    "Replace Neo4j storage with Graphiti integration",
                    "Update database initialization methods",
                    "Modify graph data writing to use Graphiti"
                ],
                "impact": "High - Core functionality change"
            }
        ]
        
        # Dependencies that can be removed
        analysis["dependencies_to_remove"] = [
            {
                "dependency": "neo4j>=5.15.0,<6.0.0",
                "file": "packages/morag-graph/pyproject.toml",
                "condition": "After migration complete and Neo4j storage removed",
                "impact": "Reduces package size and dependencies"
            }
        ]
        
        # Configuration changes needed
        analysis["configuration_changes"] = [
            {
                "change": "Remove Neo4j environment variables from documentation",
                "files": ["README.md", "docs/configuration.md"],
                "impact": "Documentation cleanup"
            },
            {
                "change": "Update default configuration to use Graphiti",
                "files": ["packages/morag/src/morag/config.py"],
                "impact": "Changes default behavior for new installations"
            }
        ]
        
        return analysis
    
    def create_cleanup_phases(self) -> List[Dict[str, Any]]:
        """Create phased cleanup plan."""
        phases = []
        
        # Phase 1: Safe removals and updates
        phase1 = {
            "phase": 1,
            "name": "Safe Component Cleanup",
            "description": "Remove components that are definitely obsolete",
            "duration_hours": 4,
            "risk_level": "Low",
            "actions": [
                "Remove standalone Neo4j test scripts",
                "Update ingestion coordinator to use Graphiti",
                "Update graph processor to use Graphiti",
                "Update documentation to reflect Graphiti usage"
            ],
            "validation": [
                "All tests pass with Graphiti",
                "Ingestion works with Graphiti backend",
                "No broken imports or references"
            ]
        }
        phases.append(phase1)
        
        # Phase 2: Migration-dependent cleanup
        phase2 = {
            "phase": 2,
            "name": "Migration-Dependent Cleanup",
            "description": "Remove components after migration validation",
            "duration_hours": 6,
            "risk_level": "Medium",
            "prerequisites": [
                "All data migrated to Graphiti",
                "Migration validation completed",
                "Rollback procedures tested"
            ],
            "actions": [
                "Remove Neo4j storage backend",
                "Remove Neo4j configuration from factories",
                "Update storage imports and exports",
                "Make Neo4j dependency optional"
            ],
            "validation": [
                "System works without Neo4j dependency",
                "All functionality available through Graphiti",
                "Performance meets requirements"
            ]
        }
        phases.append(phase2)
        
        # Phase 3: Final cleanup and optimization
        phase3 = {
            "phase": 3,
            "name": "Final Cleanup and Optimization",
            "description": "Complete cleanup and optimize for Graphiti-only operation",
            "duration_hours": 3,
            "risk_level": "Low",
            "actions": [
                "Remove Neo4j dependency entirely (if desired)",
                "Clean up any remaining Neo4j references",
                "Optimize imports and reduce package size",
                "Update all documentation"
            ],
            "validation": [
                "Package installs and works without Neo4j",
                "All tests pass",
                "Documentation is accurate and complete"
            ]
        }
        phases.append(phase3)
        
        return phases
    
    def generate_cleanup_script(self, phase: int) -> str:
        """Generate cleanup script for a specific phase."""
        if phase == 1:
            return self._generate_phase1_script()
        elif phase == 2:
            return self._generate_phase2_script()
        elif phase == 3:
            return self._generate_phase3_script()
        else:
            raise ValueError(f"Invalid phase: {phase}")
    
    def _generate_phase1_script(self) -> str:
        """Generate Phase 1 cleanup script."""
        return """#!/bin/bash
# Phase 1: Safe Component Cleanup

echo "Starting Phase 1 cleanup..."

# Remove standalone Neo4j test scripts
echo "Removing standalone test scripts..."
rm -f check_neo4j.py

# Update ingestion coordinator (manual step)
echo "Manual step: Update ingestion_coordinator.py to use Graphiti"
echo "  - Replace Neo4j storage initialization with Graphiti"
echo "  - Update _write_to_neo4j method to use Graphiti"

# Update graph processor (manual step)  
echo "Manual step: Update graph_processor.py to use Graphiti"
echo "  - Replace Neo4j storage creation with Graphiti"

# Run tests to validate changes
echo "Running tests..."
python -m pytest packages/morag-graph/tests/ -v

echo "Phase 1 cleanup complete!"
"""
    
    def _generate_phase2_script(self) -> str:
        """Generate Phase 2 cleanup script."""
        return """#!/bin/bash
# Phase 2: Migration-Dependent Cleanup

echo "Starting Phase 2 cleanup..."
echo "Prerequisites: Ensure migration is complete and validated"

# Make Neo4j dependency optional in pyproject.toml
echo "Manual step: Update pyproject.toml"
echo "  - Move neo4j to optional dependencies"
echo "  - Update dependency constraints"

# Update storage module exports
echo "Manual step: Update storage/__init__.py"
echo "  - Make Neo4j imports conditional"
echo "  - Add graceful degradation"

# Run comprehensive tests
echo "Running comprehensive tests..."
python -m pytest packages/morag-graph/tests/ -v
python -m pytest packages/morag/tests/ -v

echo "Phase 2 cleanup complete!"
"""
    
    def _generate_phase3_script(self) -> str:
        """Generate Phase 3 cleanup script."""
        return """#!/bin/bash
# Phase 3: Final Cleanup and Optimization

echo "Starting Phase 3 cleanup..."

# Remove Neo4j dependency entirely (optional)
echo "Optional: Remove Neo4j dependency completely"
echo "  - Remove from pyproject.toml dependencies"
echo "  - Update all imports to handle missing dependency"

# Clean up documentation
echo "Updating documentation..."
echo "  - Remove Neo4j configuration examples"
echo "  - Update setup instructions for Graphiti-only"

# Final test run
echo "Running final tests..."
python -m pytest packages/morag-graph/tests/ -v

echo "Phase 3 cleanup complete!"
echo "Legacy cleanup finished successfully!"
"""
