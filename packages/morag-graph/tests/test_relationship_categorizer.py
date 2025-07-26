"""Tests for relationship categorizer."""

import pytest
from morag_graph.normalizers.relationship_categorizer import (
    RelationshipCategorizer, CategorizedRelationship, RelationshipCategory,
    RelationshipTaxonomy
)
from morag_graph.normalizers.predicate_normalizer import NormalizedPredicate, RelationshipType


class TestRelationshipCategorizer:
    """Test cases for relationship categorizer."""
    
    @pytest.fixture
    def categorizer(self):
        """Create relationship categorizer instance for testing."""
        config = {
            'enable_semantic_weighting': True,
            'enable_domain_detection': True,
            'min_confidence_threshold': 0.6,
            'prefer_specific_categories': True
        }
        return RelationshipCategorizer(config=config)
    
    @pytest.fixture
    def sample_normalized_predicates(self):
        """Create sample normalized predicates for testing."""
        return [
            NormalizedPredicate(
                original="works for",
                normalized="works for",
                canonical_form="works_at",
                relationship_type=RelationshipType.EMPLOYMENT,
                confidence=0.9,
                language="en"
            ),
            NormalizedPredicate(
                original="is a",
                normalized="is",
                canonical_form="is",
                relationship_type=RelationshipType.IDENTITY,
                confidence=0.8,
                language="en"
            ),
            NormalizedPredicate(
                original="located in",
                normalized="located in",
                canonical_form="located_in",
                relationship_type=RelationshipType.LOCATION,
                confidence=0.85,
                language="en"
            ),
            NormalizedPredicate(
                original="unknown predicate",
                normalized="unknown predicate",
                canonical_form="unknown_predicate",
                relationship_type=RelationshipType.OTHER,
                confidence=0.6,
                language="en"
            )
        ]
    
    def test_init(self, categorizer):
        """Test categorizer initialization."""
        assert categorizer.enable_semantic_weighting is True
        assert categorizer.enable_domain_detection is True
        assert categorizer.min_confidence_threshold == 0.6
        assert categorizer.prefer_specific_categories is True
        
        # Check that taxonomy is initialized
        assert categorizer.taxonomy is not None
        assert isinstance(categorizer.taxonomy, RelationshipTaxonomy)
    
    @pytest.mark.asyncio
    async def test_categorize_empty_predicates(self, categorizer):
        """Test categorizing empty predicate list."""
        result = await categorizer.categorize_relationships([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_categorize_known_predicates(self, categorizer, sample_normalized_predicates):
        """Test categorizing known predicates."""
        # Use only the first three predicates (known ones)
        known_predicates = sample_normalized_predicates[:3]
        
        result = await categorizer.categorize_relationships(known_predicates)
        
        assert len(result) == 3
        
        # Check that all results are CategorizedRelationship objects
        for rel in result:
            assert isinstance(rel, CategorizedRelationship)
            assert isinstance(rel.relationship_type, RelationshipType)
            assert isinstance(rel.relationship_category, RelationshipCategory)
            assert 0.0 <= rel.confidence <= 1.0
            assert 0.0 <= rel.semantic_weight <= 1.0
            assert rel.directionality in ["directed", "undirected", "bidirectional"]
            assert rel.domain_specificity in ["general", "domain_specific", "technical"]
        
        # Check specific categorizations
        predicates_by_canonical = {rel.predicate: rel for rel in result}
        
        # Employment relationship
        if "works_at" in predicates_by_canonical:
            works_rel = predicates_by_canonical["works_at"]
            assert works_rel.relationship_type == RelationshipType.EMPLOYMENT
            assert works_rel.relationship_category == RelationshipCategory.FUNCTIONAL
        
        # Identity relationship
        if "is" in predicates_by_canonical:
            is_rel = predicates_by_canonical["is"]
            assert is_rel.relationship_type == RelationshipType.IDENTITY
            assert is_rel.relationship_category == RelationshipCategory.STRUCTURAL
        
        # Location relationship
        if "located_in" in predicates_by_canonical:
            location_rel = predicates_by_canonical["located_in"]
            assert location_rel.relationship_type == RelationshipType.LOCATION
            assert location_rel.relationship_category == RelationshipCategory.SPATIAL
    
    @pytest.mark.asyncio
    async def test_categorize_unknown_predicate(self, categorizer, sample_normalized_predicates):
        """Test categorizing unknown predicate."""
        # Use only the unknown predicate
        unknown_predicate = [sample_normalized_predicates[3]]
        
        result = await categorizer.categorize_relationships(unknown_predicate)
        
        assert len(result) == 1
        rel = result[0]
        
        # Should be categorized as OTHER with fallback category
        assert rel.relationship_type == RelationshipType.OTHER
        assert rel.relationship_category == RelationshipCategory.DESCRIPTIVE
        assert rel.confidence < 0.8  # Should have reduced confidence
        assert "fallback" in rel.metadata.get("categorization_method", "")
    
    def test_build_predicate_lookup(self, categorizer):
        """Test predicate lookup building."""
        lookup = categorizer._predicate_to_type
        
        # Check that common predicates are in lookup
        assert "works_at" in lookup
        assert "is" in lookup
        assert "located_in" in lookup
        assert "manages" in lookup
        
        # Check lookup structure
        works_type, works_category = lookup["works_at"]
        assert works_type == RelationshipType.EMPLOYMENT
        assert works_category == RelationshipCategory.FUNCTIONAL
        
        is_type, is_category = lookup["is"]
        assert is_type == RelationshipType.IDENTITY
        assert is_category == RelationshipCategory.STRUCTURAL
    
    def test_build_type_category_lookup(self, categorizer):
        """Test type to category lookup building."""
        lookup = categorizer._type_to_category
        
        # Check mappings
        assert lookup[RelationshipType.EMPLOYMENT] == RelationshipCategory.FUNCTIONAL
        assert lookup[RelationshipType.IDENTITY] == RelationshipCategory.STRUCTURAL
        assert lookup[RelationshipType.LOCATION] == RelationshipCategory.SPATIAL
        assert lookup[RelationshipType.TEMPORAL] == RelationshipCategory.TEMPORAL
    
    def test_get_category_statistics(self, categorizer, sample_normalized_predicates):
        """Test category statistics generation."""
        # Create some categorized relationships manually
        categorized = [
            CategorizedRelationship(
                predicate="works_at",
                relationship_type=RelationshipType.EMPLOYMENT,
                relationship_category=RelationshipCategory.FUNCTIONAL,
                confidence=0.9,
                semantic_weight=0.8,
                directionality="directed",
                domain_specificity="general"
            ),
            CategorizedRelationship(
                predicate="is",
                relationship_type=RelationshipType.IDENTITY,
                relationship_category=RelationshipCategory.STRUCTURAL,
                confidence=0.8,
                semantic_weight=0.9,
                directionality="directed",
                domain_specificity="general"
            ),
            CategorizedRelationship(
                predicate="located_in",
                relationship_type=RelationshipType.LOCATION,
                relationship_category=RelationshipCategory.SPATIAL,
                confidence=0.85,
                semantic_weight=0.8,
                directionality="directed",
                domain_specificity="general"
            )
        ]
        
        stats = categorizer.get_category_statistics(categorized)
        
        # Check structure
        assert "total_relationships" in stats
        assert "category_distribution" in stats
        assert "type_distribution" in stats
        assert "average_confidence_by_category" in stats
        assert "overall_average_confidence" in stats
        assert "average_semantic_weight" in stats
        assert "directionality_distribution" in stats
        
        # Check values
        assert stats["total_relationships"] == 3
        assert stats["category_distribution"]["functional"] == 1
        assert stats["category_distribution"]["structural"] == 1
        assert stats["category_distribution"]["spatial"] == 1
        
        # Check averages
        assert 0.0 <= stats["overall_average_confidence"] <= 1.0
        assert 0.0 <= stats["average_semantic_weight"] <= 1.0
    
    def test_get_taxonomy_info(self, categorizer):
        """Test taxonomy information retrieval."""
        taxonomy_info = categorizer.get_taxonomy_info()
        
        # Check structure
        assert "categories" in taxonomy_info
        assert "total_types" in taxonomy_info
        assert "total_predicates" in taxonomy_info
        
        # Check that all categories are present
        categories = taxonomy_info["categories"]
        assert "structural" in categories
        assert "functional" in categories
        assert "spatial" in categories
        assert "temporal" in categories
        assert "causal" in categories
        assert "informational" in categories
        assert "descriptive" in categories
        
        # Check category structure
        structural = categories["structural"]
        assert "types" in structural
        assert "type_count" in structural
        assert "predicate_count" in structural
        
        # Check type structure
        identity_type = structural["types"]["identity"]
        assert "predicates" in identity_type
        assert "predicate_count" in identity_type
        assert "semantic_weight" in identity_type
        assert "directionality" in identity_type
        assert "domain_specificity" in identity_type
        assert "description" in identity_type
        
        # Check counts
        assert taxonomy_info["total_types"] > 0
        assert taxonomy_info["total_predicates"] > 0
    
    def test_relationship_taxonomy(self):
        """Test relationship taxonomy structure."""
        taxonomy = RelationshipTaxonomy()
        
        # Check that taxonomy is built
        assert taxonomy.taxonomy is not None
        assert isinstance(taxonomy.taxonomy, dict)
        
        # Check that all categories are present
        assert RelationshipCategory.STRUCTURAL in taxonomy.taxonomy
        assert RelationshipCategory.FUNCTIONAL in taxonomy.taxonomy
        assert RelationshipCategory.SPATIAL in taxonomy.taxonomy
        assert RelationshipCategory.TEMPORAL in taxonomy.taxonomy
        assert RelationshipCategory.CAUSAL in taxonomy.taxonomy
        assert RelationshipCategory.INFORMATIONAL in taxonomy.taxonomy
        assert RelationshipCategory.DESCRIPTIVE in taxonomy.taxonomy
        
        # Check structural category
        structural = taxonomy.taxonomy[RelationshipCategory.STRUCTURAL]
        assert RelationshipType.IDENTITY in structural
        assert RelationshipType.POSSESSION in structural
        assert RelationshipType.MEMBERSHIP in structural
        
        # Check identity type
        identity = structural[RelationshipType.IDENTITY]
        assert "predicates" in identity
        assert "semantic_weight" in identity
        assert "directionality" in identity
        assert "domain_specificity" in identity
        assert "description" in identity
        
        # Check predicates
        assert "is" in identity["predicates"]
        assert "are" in identity["predicates"]
    
    def test_relationship_categories_enum(self):
        """Test relationship categories enumeration."""
        # Check that all expected categories exist
        assert RelationshipCategory.STRUCTURAL
        assert RelationshipCategory.FUNCTIONAL
        assert RelationshipCategory.SPATIAL
        assert RelationshipCategory.TEMPORAL
        assert RelationshipCategory.CAUSAL
        assert RelationshipCategory.SOCIAL
        assert RelationshipCategory.INFORMATIONAL
        assert RelationshipCategory.TRANSACTIONAL
        assert RelationshipCategory.COMPARATIVE
        assert RelationshipCategory.DESCRIPTIVE
        
        # Check that values are strings
        assert isinstance(RelationshipCategory.STRUCTURAL.value, str)
        assert isinstance(RelationshipCategory.FUNCTIONAL.value, str)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, categorizer):
        """Test error handling in categorization."""
        # Create a problematic predicate
        problematic_predicate = NormalizedPredicate(
            original="",
            normalized="",
            canonical_form="",
            relationship_type=RelationshipType.OTHER,
            confidence=0.0,
            language=None
        )
        
        # Should not crash
        result = await categorizer.categorize_relationships([problematic_predicate])
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_close(self, categorizer):
        """Test categorizer cleanup."""
        # Should not raise any exceptions
        await categorizer.close()
    
    @pytest.mark.asyncio
    async def test_metadata_generation(self, categorizer, sample_normalized_predicates):
        """Test metadata generation in categorized relationships."""
        result = await categorizer.categorize_relationships([sample_normalized_predicates[0]])
        
        assert len(result) >= 1
        rel = result[0]
        metadata = rel.metadata
        
        # Check required metadata fields
        assert "original_predicate" in metadata
        assert "normalized_predicate" in metadata
        assert "language" in metadata
        assert "taxonomy_description" in metadata
        assert "categorization_method" in metadata
        
        # Check values
        assert metadata["original_predicate"] == "works for"
        assert metadata["language"] == "en"
        assert metadata["categorization_method"] in ["direct_lookup", "type_mapping", "fallback"]


if __name__ == "__main__":
    pytest.main([__file__])
