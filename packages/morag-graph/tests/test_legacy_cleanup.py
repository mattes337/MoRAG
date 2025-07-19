"""Tests for legacy cleanup planning and execution."""

import pytest
from morag_graph.graphiti.production.legacy_cleanup_plan import LegacyCleanupPlan


class TestLegacyCleanupPlan:
    """Test legacy cleanup planning functionality."""
    
    @pytest.fixture
    def cleanup_plan(self):
        """Create cleanup plan instance."""
        return LegacyCleanupPlan()
    
    def test_analyze_legacy_components(self, cleanup_plan):
        """Test legacy component analysis."""
        analysis = cleanup_plan.analyze_legacy_components()
        
        # Check structure
        assert "timestamp" in analysis
        assert "components" in analysis
        assert "files_to_modify" in analysis
        assert "dependencies_to_remove" in analysis
        assert "configuration_changes" in analysis
        
        # Check component categories
        components = analysis["components"]
        assert "safe_to_remove" in components
        assert "migration_dependent" in components
        assert "keep_for_compatibility" in components
        assert "requires_review" in components
        
        # Verify we have identified some components in each category
        assert len(components["safe_to_remove"]) > 0
        assert len(components["migration_dependent"]) > 0
        assert len(components["keep_for_compatibility"]) > 0
        
        # Check component structure
        for category in components.values():
            for component in category:
                assert "component" in component
                assert "files" in component
                assert "description" in component
                assert "impact" in component
    
    def test_create_cleanup_phases(self, cleanup_plan):
        """Test cleanup phase creation."""
        phases = cleanup_plan.create_cleanup_phases()
        
        # Should have 3 phases
        assert len(phases) == 3
        
        # Check phase structure
        for i, phase in enumerate(phases, 1):
            assert phase["phase"] == i
            assert "name" in phase
            assert "description" in phase
            assert "duration_hours" in phase
            assert "risk_level" in phase
            assert "actions" in phase
            assert "validation" in phase
            
            # Check that actions and validation are non-empty
            assert len(phase["actions"]) > 0
            assert len(phase["validation"]) > 0
        
        # Check risk levels are appropriate
        risk_levels = [phase["risk_level"] for phase in phases]
        assert "Low" in risk_levels
        assert "Medium" in risk_levels  # Phase 2 should be medium risk
    
    def test_generate_cleanup_scripts(self, cleanup_plan):
        """Test cleanup script generation."""
        # Test all three phases
        for phase in [1, 2, 3]:
            script = cleanup_plan.generate_cleanup_script(phase)
            
            assert isinstance(script, str)
            assert len(script) > 0
            assert f"Phase {phase}" in script
            assert "echo" in script  # Should be a bash script
        
        # Test invalid phase
        with pytest.raises(ValueError):
            cleanup_plan.generate_cleanup_script(4)
    
    def test_phase1_script_content(self, cleanup_plan):
        """Test Phase 1 script content."""
        script = cleanup_plan.generate_cleanup_script(1)
        
        # Should mention safe removals
        assert "check_neo4j.py" in script
        assert "ingestion_coordinator.py" in script
        assert "graph_processor.py" in script
        assert "pytest" in script
    
    def test_phase2_script_content(self, cleanup_plan):
        """Test Phase 2 script content."""
        script = cleanup_plan.generate_cleanup_script(2)
        
        # Should mention migration-dependent changes
        assert "pyproject.toml" in script
        assert "storage/__init__.py" in script
        assert "Prerequisites" in script
        assert "migration" in script.lower()
    
    def test_phase3_script_content(self, cleanup_plan):
        """Test Phase 3 script content."""
        script = cleanup_plan.generate_cleanup_script(3)
        
        # Should mention final cleanup
        assert "documentation" in script.lower()
        assert "final" in script.lower()
        assert "Optional" in script
    
    def test_component_categorization(self, cleanup_plan):
        """Test that components are properly categorized."""
        analysis = cleanup_plan.analyze_legacy_components()
        components = analysis["components"]
        
        # Check that critical components are in migration_dependent
        migration_dependent_files = []
        for component in components["migration_dependent"]:
            migration_dependent_files.extend(component["files"])
        
        # Neo4j storage should be migration dependent
        assert any("neo4j_storage.py" in file for file in migration_dependent_files)
        
        # Check that safe components are properly identified
        safe_to_remove_files = []
        for component in components["safe_to_remove"]:
            safe_to_remove_files.extend(component["files"])
        
        # Test scripts should be safe to remove
        assert any("check_neo4j.py" in file for file in safe_to_remove_files)
        
        # Check that compatibility components are preserved
        keep_for_compatibility_files = []
        for component in components["keep_for_compatibility"]:
            keep_for_compatibility_files.extend(component["files"])
        
        # Dependencies.py should be kept for compatibility
        assert any("dependencies.py" in file for file in keep_for_compatibility_files)
    
    def test_dependency_removal_plan(self, cleanup_plan):
        """Test dependency removal planning."""
        analysis = cleanup_plan.analyze_legacy_components()
        dependencies = analysis["dependencies_to_remove"]
        
        # Should identify Neo4j dependency for removal
        assert len(dependencies) > 0
        
        neo4j_dep = next((dep for dep in dependencies if "neo4j" in dep["dependency"]), None)
        assert neo4j_dep is not None
        assert "condition" in neo4j_dep
        assert "impact" in neo4j_dep
    
    def test_configuration_changes_plan(self, cleanup_plan):
        """Test configuration changes planning."""
        analysis = cleanup_plan.analyze_legacy_components()
        config_changes = analysis["configuration_changes"]
        
        # Should have configuration changes planned
        assert len(config_changes) > 0
        
        # Should include documentation updates
        doc_changes = [change for change in config_changes if "documentation" in change["change"].lower()]
        assert len(doc_changes) > 0
    
    def test_files_to_modify_plan(self, cleanup_plan):
        """Test files to modify planning."""
        analysis = cleanup_plan.analyze_legacy_components()
        files_to_modify = analysis["files_to_modify"]
        
        # Should identify files that need modification
        assert len(files_to_modify) > 0
        
        # Should include ingestion coordinator
        ingestion_file = next((file for file in files_to_modify 
                             if "ingestion_coordinator.py" in file["file"]), None)
        assert ingestion_file is not None
        assert "changes" in ingestion_file
        assert len(ingestion_file["changes"]) > 0


class TestLegacyCleanupIntegration:
    """Integration tests for legacy cleanup."""
    
    def test_full_cleanup_analysis_workflow(self):
        """Test complete cleanup analysis workflow."""
        cleanup_plan = LegacyCleanupPlan()
        
        # Analyze components
        analysis = cleanup_plan.analyze_legacy_components()
        assert analysis is not None
        
        # Create phases
        phases = cleanup_plan.create_cleanup_phases()
        assert len(phases) == 3
        
        # Generate scripts for all phases
        scripts = {}
        for phase_num in [1, 2, 3]:
            scripts[phase_num] = cleanup_plan.generate_cleanup_script(phase_num)
            assert len(scripts[phase_num]) > 0
        
        # Verify scripts are different
        assert scripts[1] != scripts[2]
        assert scripts[2] != scripts[3]
        assert scripts[1] != scripts[3]
    
    def test_cleanup_plan_completeness(self):
        """Test that cleanup plan covers all major components."""
        cleanup_plan = LegacyCleanupPlan()
        analysis = cleanup_plan.analyze_legacy_components()
        
        # Collect all files mentioned in the analysis
        all_files = set()
        
        for category in analysis["components"].values():
            for component in category:
                all_files.update(component["files"])
        
        for file_info in analysis["files_to_modify"]:
            all_files.add(file_info["file"])
        
        # Should cover major Neo4j-related files
        expected_files = [
            "neo4j_storage.py",
            "ingestion_coordinator.py", 
            "graph_processor.py",
            "dependencies.py",
            "pyproject.toml"
        ]
        
        for expected_file in expected_files:
            assert any(expected_file in file for file in all_files), f"Missing {expected_file} in cleanup plan"
