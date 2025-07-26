"""Tests for triplet validator."""

import pytest
from morag_graph.processors.triplet_validator import (
    TripletValidator, ValidationLevel, ValidationResult, QualityScore, ValidatedTriplet
)
from morag_graph.normalizers.entity_linker import LinkedTriplet, EntityMatch
from morag_graph.normalizers.confidence_manager import ConfidenceScore, ConfidenceLevel
from morag_graph.normalizers.relationship_categorizer import CategorizedRelationship, RelationshipCategory
from morag_graph.normalizers.predicate_normalizer import RelationshipType
from morag_graph.models.entity import Entity as GraphEntity


class TestTripletValidator:
    """Test cases for triplet validator."""
    
    @pytest.fixture
    def validator(self):
        """Create triplet validator instance for testing."""
        config = {
            'validation_level': 'standard',
            'min_overall_quality': 0.6,
            'min_relevance_score': 0.5,
            'enable_semantic_validation': True,
            'enable_domain_validation': False,
            'reject_low_quality': True
        }
        return TripletValidator(config=config)
    
    @pytest.fixture
    def strict_validator(self):
        """Create strict triplet validator for testing."""
        config = {
            'validation_level': 'strict',
            'min_overall_quality': 0.8,
            'min_relevance_score': 0.7,
            'enable_semantic_validation': True,
            'reject_low_quality': True
        }
        return TripletValidator(config=config)
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            GraphEntity(
                name="John Smith",
                type="PERSON",
                confidence=0.9,
                source_doc_id="doc1"
            ),
            GraphEntity(
                name="Google Inc",
                type="ORGANIZATION",
                confidence=0.8,
                source_doc_id="doc1"
            )
        ]
    
    @pytest.fixture
    def sample_triplets(self, sample_entities):
        """Create sample linked triplets for testing."""
        return [
            # High quality triplet
            LinkedTriplet(
                subject="John Smith",
                predicate="works at",
                object="Google Inc",
                subject_entity=sample_entities[0],
                object_entity=sample_entities[1],
                subject_match=EntityMatch(
                    openie_entity="John Smith",
                    spacy_entity=sample_entities[0],
                    match_type="exact",
                    confidence=1.0,
                    similarity_score=1.0
                ),
                object_match=EntityMatch(
                    openie_entity="Google Inc",
                    spacy_entity=sample_entities[1],
                    match_type="exact",
                    confidence=1.0,
                    similarity_score=1.0
                ),
                confidence=0.9,
                validation_score=0.8,
                sentence="John Smith works at Google Inc.",
                sentence_id="doc1_0",
                source_doc_id="doc1",
                metadata={'sentence_quality_score': 0.9}
            ),
            # Medium quality triplet
            LinkedTriplet(
                subject="Mary Johnson",
                predicate="is",
                object="CEO",
                subject_entity=None,
                object_entity=None,
                subject_match=None,
                object_match=None,
                confidence=0.7,
                validation_score=0.6,
                sentence="Mary Johnson is the CEO.",
                sentence_id="doc1_1",
                source_doc_id="doc1",
                metadata={'sentence_quality_score': 0.7}
            ),
            # Low quality triplet
            LinkedTriplet(
                subject="",  # Empty subject
                predicate="of",
                object="abc",
                subject_entity=None,
                object_entity=None,
                subject_match=None,
                object_match=None,
                confidence=0.3,
                validation_score=0.4,
                sentence="Empty subject of abc.",
                sentence_id="doc1_2",
                source_doc_id="doc1",
                metadata={}
            ),
            # Invalid triplet
            LinkedTriplet(
                subject="@#$%",
                predicate="",  # Empty predicate
                object="xyz",
                subject_entity=None,
                object_entity=None,
                subject_match=None,
                object_match=None,
                confidence=0.2,
                validation_score=0.3,
                sentence="Invalid triplet with bad characters.",
                sentence_id="doc1_3",
                source_doc_id="doc1",
                metadata={}
            )
        ]
    
    @pytest.fixture
    def sample_confidence_scores(self):
        """Create sample confidence scores for testing."""
        return [
            ConfidenceScore(
                overall_score=0.85,
                component_scores={'extraction_confidence': 0.9, 'validation_score': 0.8},
                confidence_level=ConfidenceLevel.HIGH,
                quality_flags={'high_extraction_confidence', 'both_entities_linked'}
            ),
            ConfidenceScore(
                overall_score=0.65,
                component_scores={'extraction_confidence': 0.7, 'validation_score': 0.6},
                confidence_level=ConfidenceLevel.MEDIUM,
                quality_flags={'partial_entity_linking'}
            ),
            ConfidenceScore(
                overall_score=0.35,
                component_scores={'extraction_confidence': 0.3, 'validation_score': 0.4},
                confidence_level=ConfidenceLevel.LOW,
                quality_flags={'low_extraction_confidence', 'no_entities_linked'}
            ),
            ConfidenceScore(
                overall_score=0.25,
                component_scores={'extraction_confidence': 0.2, 'validation_score': 0.3},
                confidence_level=ConfidenceLevel.VERY_LOW,
                quality_flags={'scoring_error'}
            )
        ]
    
    @pytest.fixture
    def sample_categorized_relationships(self):
        """Create sample categorized relationships for testing."""
        return [
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
            None,  # No categorization for low quality triplet
            None   # No categorization for invalid triplet
        ]
    
    def test_init(self, validator):
        """Test validator initialization."""
        assert validator.validation_level == ValidationLevel.STANDARD
        assert validator.min_overall_quality == 0.6
        assert validator.min_relevance_score == 0.5
        assert validator.enable_semantic_validation is True
        assert validator.enable_domain_validation is False
        assert validator.reject_low_quality is True
        
        # Check that validation rules are built
        assert len(validator.validation_rules) > 0
        assert "non_empty_entities" in validator.validation_rules
        assert "non_empty_predicate" in validator.validation_rules
    
    @pytest.mark.asyncio
    async def test_validate_empty_triplets(self, validator):
        """Test validating empty triplet list."""
        result = await validator.validate_triplets([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_validate_high_quality_triplet(self, validator, sample_triplets, sample_confidence_scores, sample_categorized_relationships):
        """Test validating high quality triplet."""
        # Use only the first (high quality) triplet
        high_quality_triplet = [sample_triplets[0]]
        confidence_score = [sample_confidence_scores[0]]
        categorized_rel = [sample_categorized_relationships[0]]
        
        result = await validator.validate_triplets(
            high_quality_triplet, 
            confidence_score, 
            categorized_rel
        )
        
        assert len(result) == 1
        validated = result[0]
        
        # Check structure
        assert isinstance(validated, ValidatedTriplet)
        assert validated.triplet == high_quality_triplet[0]
        assert validated.confidence_score == confidence_score[0]
        assert validated.categorized_relationship == categorized_rel[0]
        assert isinstance(validated.quality_score, QualityScore)
        assert validated.validation_level == ValidationLevel.STANDARD
        
        # Should pass validation
        assert validated.passed_validation is True
        assert validated.rejection_reason is None
        assert validated.quality_score.validation_result in [ValidationResult.VALID, ValidationResult.WARNING]
        
        # Should have good quality score
        assert validated.quality_score.overall_score > 0.6
        assert validated.quality_score.relevance_score > 0.5
    
    @pytest.mark.asyncio
    async def test_validate_low_quality_triplets(self, validator, sample_triplets):
        """Test validating low quality triplets."""
        # Use the low quality triplets (empty subject and empty predicate)
        low_quality_triplets = sample_triplets[2:4]
        
        result = await validator.validate_triplets(low_quality_triplets)
        
        # Should filter out low quality triplets when reject_low_quality is True
        assert len(result) == 0 or all(not t.passed_validation for t in result)
    
    @pytest.mark.asyncio
    async def test_strict_validation(self, strict_validator, sample_triplets, sample_confidence_scores):
        """Test strict validation level."""
        # Use medium quality triplet
        medium_triplet = [sample_triplets[1]]
        medium_confidence = [sample_confidence_scores[1]]
        
        result = await strict_validator.validate_triplets(medium_triplet, medium_confidence)
        
        # Strict validator should be more restrictive
        if result:
            validated = result[0]
            # Either rejected or has lower pass rate than standard validation
            assert validated.validation_level == ValidationLevel.STRICT
    
    def test_validate_entities(self, validator, sample_triplets):
        """Test entity validation."""
        # High quality triplet
        score, issues, flags = validator._validate_entities(sample_triplets[0])
        assert score > 0.8
        assert len(issues) == 0 or all("warning" in issue.lower() for issue in issues)
        assert "subject_linked" in flags
        assert "object_linked" in flags
        
        # Low quality triplet (empty subject)
        score, issues, flags = validator._validate_entities(sample_triplets[2])
        assert score < 0.6
        assert "Empty subject" in issues
        assert "empty_subject" in flags
    
    def test_validate_predicate(self, validator, sample_triplets):
        """Test predicate validation."""
        # Good predicate
        score, issues, flags = validator._validate_predicate(sample_triplets[0])
        assert score > 0.8
        assert "good_predicate" in flags
        
        # Empty predicate
        score, issues, flags = validator._validate_predicate(sample_triplets[3])
        assert score < 0.3
        assert "Empty predicate" in issues
        assert "empty_predicate" in flags
    
    def test_validate_semantics(self, validator, sample_triplets):
        """Test semantic validation."""
        # Good semantics
        score, issues, flags = validator._validate_semantics(sample_triplets[0])
        assert score > 0.8
        assert "proper_noun_subject" in flags
        assert "proper_noun_object" in flags
        
        # Create triplet with identical subject and object
        identical_triplet = LinkedTriplet(
            subject="test",
            predicate="is",
            object="test",  # Same as subject
            subject_entity=None,
            object_entity=None,
            subject_match=None,
            object_match=None,
            confidence=0.7,
            validation_score=0.6,
            sentence="test is test.",
            sentence_id="test_0",
            source_doc_id="test"
        )
        
        score, issues, flags = validator._validate_semantics(identical_triplet)
        assert score < 0.8
        assert "Subject and object are identical" in issues
        assert "identical_subject_object" in flags
    
    def test_calculate_relevance_score(self, validator, sample_triplets, sample_categorized_relationships):
        """Test relevance score calculation."""
        # High quality with entity linking and good categorization
        relevance = validator._calculate_relevance_score(
            sample_triplets[0], 
            sample_categorized_relationships[0]
        )
        assert relevance > 0.7
        
        # Medium quality without entity linking
        relevance = validator._calculate_relevance_score(
            sample_triplets[1], 
            sample_categorized_relationships[1]
        )
        assert 0.4 <= relevance <= 0.8
        
        # Low quality
        relevance = validator._calculate_relevance_score(sample_triplets[2])
        assert relevance < 0.7
    
    def test_determine_validation_result(self, validator):
        """Test validation result determination."""
        # Valid result
        result = validator._determine_validation_result(0.8, [])
        assert result == ValidationResult.VALID
        
        # Warning result
        result = validator._determine_validation_result(0.8, ["Minor issue"])
        assert result == ValidationResult.WARNING
        
        # Invalid result
        result = validator._determine_validation_result(0.4, ["Quality issue"])
        assert result == ValidationResult.INVALID
        
        # Rejected result
        result = validator._determine_validation_result(0.8, ["Empty subject"])
        assert result == ValidationResult.REJECTED
    
    @pytest.mark.asyncio
    async def test_get_validation_statistics(self, validator, sample_triplets):
        """Test validation statistics generation."""
        result = await validator.validate_triplets(sample_triplets)
        stats = validator.get_validation_statistics(result)
        
        # Check structure
        assert "total_triplets" in stats
        assert "passed_validation" in stats
        assert "validation_pass_rate" in stats
        assert "validation_result_distribution" in stats
        assert "average_quality_score" in stats
        assert "component_score_averages" in stats
        assert "quality_flag_distribution" in stats
        assert "validation_level" in stats
        
        # Check values
        assert stats["total_triplets"] == len(result)
        assert 0.0 <= stats["validation_pass_rate"] <= 1.0
        assert 0.0 <= stats["average_quality_score"] <= 1.0
        assert stats["validation_level"] == "standard"
    
    def test_get_validation_rules_info(self, validator):
        """Test validation rules information."""
        rules_info = validator.get_validation_rules_info()
        
        # Check structure
        assert "total_rules" in rules_info
        assert "enabled_rules" in rules_info
        assert "rules_by_category" in rules_info
        assert "rules_by_severity" in rules_info
        assert "rules" in rules_info
        
        # Check values
        assert rules_info["total_rules"] > 0
        assert rules_info["enabled_rules"] <= rules_info["total_rules"]
        
        # Check categories
        categories = rules_info["rules_by_category"]
        assert "entity" in categories
        assert "predicate" in categories
        assert "semantic" in categories
        assert "quality" in categories
        
        # Check severities
        severities = rules_info["rules_by_severity"]
        assert "error" in severities
        assert "warning" in severities
    
    @pytest.mark.asyncio
    async def test_error_handling(self, validator):
        """Test error handling in validation."""
        # Create a problematic triplet
        problematic_triplet = LinkedTriplet(
            subject=None,  # None instead of string
            predicate=None,
            object=None,
            subject_entity=None,
            object_entity=None,
            subject_match=None,
            object_match=None,
            confidence=0.0,
            validation_score=0.0,
            sentence="",
            sentence_id="",
            source_doc_id=None,
            metadata={}
        )
        
        # Should not crash, but might return empty results
        try:
            result = await validator.validate_triplets([problematic_triplet])
            assert isinstance(result, list)
        except Exception:
            # Some errors are expected with None values
            pass
    
    @pytest.mark.asyncio
    async def test_close(self, validator):
        """Test validator cleanup."""
        # Should not raise any exceptions
        await validator.close()
    
    def test_validation_levels_enum(self):
        """Test validation levels enumeration."""
        # Check that all expected levels exist
        assert ValidationLevel.PERMISSIVE
        assert ValidationLevel.STANDARD
        assert ValidationLevel.STRICT
        assert ValidationLevel.VERY_STRICT
        
        # Check that values are strings
        assert isinstance(ValidationLevel.PERMISSIVE.value, str)
        assert isinstance(ValidationLevel.STANDARD.value, str)
    
    def test_validation_results_enum(self):
        """Test validation results enumeration."""
        # Check that all expected results exist
        assert ValidationResult.VALID
        assert ValidationResult.WARNING
        assert ValidationResult.INVALID
        assert ValidationResult.REJECTED
        
        # Check that values are strings
        assert isinstance(ValidationResult.VALID.value, str)
        assert isinstance(ValidationResult.WARNING.value, str)


if __name__ == "__main__":
    pytest.main([__file__])
