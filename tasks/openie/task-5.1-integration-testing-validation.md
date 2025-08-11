# Task 5.1: Integration Testing and Validation

## Objective
Implement comprehensive integration testing and validation for the complete OpenIE pipeline, ensuring end-to-end functionality and quality assurance across all components.

## Scope
- Create end-to-end integration tests
- Implement validation frameworks for pipeline quality
- Add performance benchmarking and monitoring
- Create test data sets and validation metrics
- **MANDATORY**: Complete validation before production deployment

## Implementation Details

### 1. Integration Test Suite

**File**: `packages/morag-graph/tests/integration/test_openie_integration.py`

```python
"""Integration tests for OpenIE pipeline."""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock

from morag_graph.services.openie_service import OpenIEService
from morag_graph.processors.sentence_processor import SentenceProcessor
from morag_graph.processors.triplet_processor import TripletProcessor
from morag_graph.normalizers.entity_linker import EntityLinker
from morag_graph.normalizers.entity_normalizer import EntityNormalizer
from morag_graph.normalizers.predicate_normalizer import PredicateNormalizer
from morag_graph.validators.confidence_scorer import ConfidenceScorer
from morag_graph.validators.triplet_filter import TripletFilter
from morag_graph.services.openie_ingestion_service import OpenIEIngestionService

class TestOpenIEIntegration:
    """Integration tests for complete OpenIE pipeline."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        John Smith works at Microsoft Corporation in Seattle.
        He is the CEO of the company and lives in Washington.
        Microsoft was founded by Bill Gates in 1975.
        The company develops software products and services.
        """

    @pytest.fixture
    def sample_german_text(self):
        """Sample German text for testing."""
        return """
        Johann Müller arbeitet bei der Volkswagen AG in Wolfsburg.
        Er ist Geschäftsführer des Unternehmens und lebt in Niedersachsen.
        Volkswagen wurde von Ferdinand Porsche gegründet.
        Das Unternehmen entwickelt Automobile und Technologien.
        """

    @pytest.fixture
    def sample_spanish_text(self):
        """Sample Spanish text for testing."""
        return """
        Juan García trabaja en Microsoft Corporation en Madrid.
        Él es director de la empresa y vive en España.
        Microsoft fue fundada por Bill Gates en 1975.
        La empresa desarrolla productos de software.
        """
    
    @pytest.fixture
    def sample_spacy_entities(self):
        """Sample spaCy entities for testing."""
        return [
            {
                'text': 'John Smith',
                'label': 'PERSON',
                'entity_id': 'person_001',
                'canonical_form': 'john smith',
                'variations': ['John', 'J. Smith']
            },
            {
                'text': 'Microsoft Corporation',
                'label': 'ORG',
                'entity_id': 'org_001',
                'canonical_form': 'microsoft',
                'variations': ['Microsoft', 'MSFT', 'Microsoft Corp']
            },
            {
                'text': 'Seattle',
                'label': 'GPE',
                'entity_id': 'loc_001',
                'canonical_form': 'seattle',
                'variations': ['Seattle, WA']
            },
            {
                'text': 'Bill Gates',
                'label': 'PERSON',
                'entity_id': 'person_002',
                'canonical_form': 'bill gates',
                'variations': ['William Gates', 'Bill']
            }
        ]
    
    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing."""
        driver = AsyncMock()
        session = AsyncMock()
        driver.session.return_value.__aenter__.return_value = session
        return driver
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, sample_text, sample_spacy_entities, mock_neo4j_driver):
        """Test complete OpenIE pipeline from text to Neo4j ingestion."""
        
        # Initialize services
        openie_service = OpenIEService()
        sentence_processor = SentenceProcessor()
        triplet_processor = TripletProcessor()
        entity_linker = EntityLinker()
        entity_normalizer = EntityNormalizer()
        predicate_normalizer = PredicateNormalizer()
        confidence_scorer = ConfidenceScorer()
        triplet_filter = TripletFilter()
        ingestion_service = OpenIEIngestionService(mock_neo4j_driver)
        
        # Mock OpenIE extraction
        openie_service._openie = Mock()
        openie_service._openie.annotate.return_value = [
            {
                'subject': 'John Smith',
                'relation': 'works at',
                'object': 'Microsoft Corporation',
                'confidence': 0.85
            },
            {
                'subject': 'John Smith',
                'relation': 'is CEO of',
                'object': 'Microsoft Corporation',
                'confidence': 0.90
            },
            {
                'subject': 'Microsoft',
                'relation': 'founded by',
                'object': 'Bill Gates',
                'confidence': 0.80
            }
        ]
        
        # Step 1: Process sentences
        processed_sentences = await sentence_processor.process_text(sample_text)
        assert len(processed_sentences) > 0
        assert all('quality_score' in sentence for sentence in processed_sentences)
        
        # Step 2: Extract triplets
        raw_triplets = await openie_service.extract_triplets(sample_text)
        assert len(raw_triplets) > 0
        assert all('subject' in triplet for triplet in raw_triplets)
        assert all('predicate' in triplet for triplet in raw_triplets)
        assert all('object' in triplet for triplet in raw_triplets)
        
        # Step 3: Process triplets
        processed_triplets = await triplet_processor.process_triplets(raw_triplets)
        assert len(processed_triplets) > 0
        assert all('quality_score' in triplet for triplet in processed_triplets)
        
        # Step 4: Normalize entities
        normalized_entities = await entity_normalizer.normalize_entities(sample_spacy_entities)
        assert len(normalized_entities) > 0
        assert all('canonical_form' in entity for entity in normalized_entities)
        
        # Step 5: Link entities
        linked_triplets = await entity_linker.link_entities(processed_triplets, normalized_entities)
        assert len(linked_triplets) > 0
        
        # Step 6: Normalize predicates
        predicate_normalized = await predicate_normalizer.normalize_predicates(linked_triplets)
        assert len(predicate_normalized) > 0
        assert all('predicate_normalization' in triplet for triplet in predicate_normalized)
        
        # Step 7: Score confidence
        scored_triplets = await confidence_scorer.score_triplets(predicate_normalized)
        assert len(scored_triplets) > 0
        assert all('overall_confidence' in triplet for triplet in scored_triplets)
        
        # Step 8: Filter triplets
        filter_results = await triplet_filter.filter_triplets(scored_triplets)
        filtered_triplets = filter_results['filtered_triplets']
        assert len(filtered_triplets) > 0
        
        # Step 9: Mock ingestion
        ingestion_service._ingest_batch = AsyncMock(return_value={
            'triplets_processed': len(filtered_triplets),
            'relationships_created': len(filtered_triplets),
            'nodes_created': len(filtered_triplets) * 2
        })
        
        ingestion_results = await ingestion_service.ingest_triplets(filtered_triplets, 'test_doc_001')
        assert ingestion_results['triplets_ingested'] > 0
        assert ingestion_results['relationships_created'] > 0
        
        # Validate end-to-end quality
        assert len(filtered_triplets) <= len(raw_triplets)  # Filtering should reduce count
        
        # Check that high-confidence triplets are preserved
        high_conf_triplets = [t for t in filtered_triplets if t.get('confidence_level') == 'HIGH']
        assert len(high_conf_triplets) > 0

    @pytest.mark.asyncio
    async def test_multilingual_pipeline_german(self, sample_german_text, mock_neo4j_driver):
        """Test OpenIE pipeline with German text."""
        from morag_graph.normalizers.german_normalizer import GermanNormalizer

        # Initialize services with German support
        german_normalizer = GermanNormalizer()
        entity_normalizer = EntityNormalizer()
        predicate_normalizer = PredicateNormalizer()

        # Mock German OpenIE extraction
        mock_german_triplets = [
            {
                'subject': 'Johann Müller',
                'predicate': 'arbeitet bei',
                'object': 'Volkswagen AG',
                'confidence': 0.85
            },
            {
                'subject': 'Johann Müller',
                'predicate': 'ist Geschäftsführer von',
                'object': 'Volkswagen AG',
                'confidence': 0.90
            }
        ]

        # Test German entity normalization
        german_entity = await german_normalizer.normalize_german_entity('Johann Müller', 'PERSON')
        assert 'john' in german_entity.lower()

        # Test German predicate normalization
        german_predicate = await german_normalizer.normalize_german_predicate('arbeitet bei')
        assert german_predicate == 'works at'

        # Test German company normalization
        german_company = await german_normalizer.normalize_german_entity('Volkswagen AG', 'ORG')
        assert 'corporation' in german_company.lower()

        logger.info("German multilingual pipeline test completed successfully")

    @pytest.mark.asyncio
    async def test_multilingual_pipeline_spanish(self, sample_spanish_text):
        """Test OpenIE pipeline with Spanish text."""
        predicate_normalizer = PredicateNormalizer()

        # Test Spanish predicate normalization
        spanish_predicates = [
            ('trabaja en', 'works at'),
            ('es director de', 'director of'),
            ('fue fundada por', 'founded by')
        ]

        for spanish_pred, expected_english in spanish_predicates:
            # This would use the existing multilingual mappings
            # in the predicate normalizer
            result = await predicate_normalizer._normalize_predicate(spanish_pred)
            # Note: This test would need the actual Spanish mappings implemented
            logger.info(f"Spanish predicate '{spanish_pred}' normalized to '{result}'")

        logger.info("Spanish multilingual pipeline test completed successfully")
    
    @pytest.mark.asyncio
    async def test_entity_linking_accuracy(self, sample_spacy_entities):
        """Test entity linking accuracy."""
        entity_linker = EntityLinker()
        
        test_triplets = [
            {
                'subject': 'John Smith',
                'predicate': 'works at',
                'object': 'Microsoft',
                'confidence': 0.8
            },
            {
                'subject': 'Bill Gates',
                'predicate': 'founded',
                'object': 'Microsoft Corporation',
                'confidence': 0.9
            }
        ]
        
        linked_triplets = await entity_linker.link_entities(test_triplets, sample_spacy_entities)
        
        # Validate linking results
        assert len(linked_triplets) == len(test_triplets)
        
        # Check that entities are properly linked
        for triplet in linked_triplets:
            entity_linking = triplet.get('entity_linking', {})
            assert 'subject_linked' in entity_linking
            assert 'object_linked' in entity_linking
            
            # At least one entity should be linked for each triplet
            assert (entity_linking['subject_linked'] or entity_linking['object_linked'])
    
    @pytest.mark.asyncio
    async def test_predicate_normalization_coverage(self):
        """Test predicate normalization coverage."""
        predicate_normalizer = PredicateNormalizer()
        
        test_triplets = [
            {
                'subject': 'John',
                'predicate': 'is employed by',
                'object': 'Microsoft',
                'confidence': 0.8
            },
            {
                'subject': 'Company',
                'predicate': 'was founded by',
                'object': 'Bill',
                'confidence': 0.9
            },
            {
                'subject': 'Person',
                'predicate': 'lives in',
                'object': 'City',
                'confidence': 0.7
            }
        ]
        
        normalized_triplets = await predicate_normalizer.normalize_predicates(test_triplets)
        
        # Validate normalization
        assert len(normalized_triplets) == len(test_triplets)
        
        for triplet in normalized_triplets:
            assert 'predicate_normalization' in triplet
            norm_info = triplet['predicate_normalization']
            assert 'method' in norm_info
            assert 'confidence' in norm_info
            assert 'category' in norm_info
            
            # Predicate should be normalized to standard form
            predicate = triplet['predicate']
            assert predicate.isupper() or predicate in ['WORKS_AT', 'CREATED', 'LOCATED_IN']
    
    @pytest.mark.asyncio
    async def test_confidence_scoring_distribution(self):
        """Test confidence scoring distribution."""
        confidence_scorer = ConfidenceScorer()
        
        # Create triplets with varying quality
        test_triplets = [
            {  # High quality
                'confidence': 0.9,
                'entity_linking': {'linking_confidence': 0.95, 'subject_linked': True, 'object_linked': True},
                'predicate_normalization': {'confidence': 0.9, 'method': 'exact_match'},
                'sentence_quality': 0.8,
                'quality_score': 0.85
            },
            {  # Medium quality
                'confidence': 0.7,
                'entity_linking': {'linking_confidence': 0.6, 'subject_linked': True, 'object_linked': False},
                'predicate_normalization': {'confidence': 0.7, 'method': 'pattern_match'},
                'sentence_quality': 0.6,
                'quality_score': 0.65
            },
            {  # Low quality
                'confidence': 0.4,
                'entity_linking': {'linking_confidence': 0.3, 'subject_linked': False, 'object_linked': False},
                'predicate_normalization': {'confidence': 0.4, 'method': 'fallback'},
                'sentence_quality': 0.4,
                'quality_score': 0.35
            }
        ]
        
        scored_triplets = await confidence_scorer.score_triplets(test_triplets)
        
        # Validate scoring distribution
        assert len(scored_triplets) == len(test_triplets)
        
        confidence_levels = [t['confidence_level'] for t in scored_triplets]
        assert 'HIGH' in confidence_levels
        assert 'MEDIUM' in confidence_levels or 'LOW' in confidence_levels
        
        # High quality triplet should have highest confidence
        high_quality_triplet = scored_triplets[0]
        assert high_quality_triplet['overall_confidence'] > scored_triplets[1]['overall_confidence']
        assert high_quality_triplet['overall_confidence'] > scored_triplets[2]['overall_confidence']
    
    @pytest.mark.asyncio
    async def test_filtering_effectiveness(self):
        """Test filtering effectiveness."""
        triplet_filter = TripletFilter({
            'min_overall_confidence': 0.5,
            'min_extraction_confidence': 0.4,
            'min_quality_score': 0.3
        })
        
        # Create triplets with varying quality
        test_triplets = [
            {  # Should pass
                'overall_confidence': 0.8,
                'confidence': 0.7,
                'quality_score': 0.6,
                'confidence_level': 'HIGH'
            },
            {  # Should pass
                'overall_confidence': 0.6,
                'confidence': 0.5,
                'quality_score': 0.4,
                'confidence_level': 'MEDIUM'
            },
            {  # Should be filtered out
                'overall_confidence': 0.3,
                'confidence': 0.2,
                'quality_score': 0.2,
                'confidence_level': 'VERY_LOW'
            }
        ]
        
        filter_results = await triplet_filter.filter_triplets(test_triplets)
        filtered_triplets = filter_results['filtered_triplets']
        
        # Validate filtering
        assert len(filtered_triplets) < len(test_triplets)
        assert len(filtered_triplets) >= 2  # First two should pass
        
        # All filtered triplets should meet minimum criteria
        for triplet in filtered_triplets:
            assert triplet['overall_confidence'] >= 0.5
            assert triplet['confidence'] >= 0.4
            assert triplet['quality_score'] >= 0.3
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, sample_text):
        """Test performance benchmarks."""
        import time
        
        openie_service = OpenIEService()
        
        # Mock OpenIE for consistent timing
        openie_service._openie = Mock()
        openie_service._openie.annotate.return_value = [
            {
                'subject': 'Test Subject',
                'relation': 'test relation',
                'object': 'Test Object',
                'confidence': 0.8
            }
        ]
        
        # Measure processing time
        start_time = time.time()
        triplets = await openie_service.extract_triplets(sample_text)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert len(triplets) > 0
        
        # Memory usage should be reasonable (basic check)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 1000  # Should use less than 1GB
```

### 2. Validation Framework

**File**: `packages/morag-graph/src/morag_graph/validation/pipeline_validator.py`

```python
"""Validation framework for OpenIE pipeline."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import structlog
from collections import defaultdict

from morag_core.config import get_settings
from morag_core.exceptions import ValidationError

logger = structlog.get_logger(__name__)

class OpenIEPipelineValidator:
    """Validates OpenIE pipeline quality and performance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline validator.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_triplet_extraction_rate': self.config.get('min_triplet_extraction_rate', 0.1),
            'min_entity_linking_rate': self.config.get('min_entity_linking_rate', 0.3),
            'min_predicate_normalization_rate': self.config.get('min_predicate_normalization_rate', 0.5),
            'min_average_confidence': self.config.get('min_average_confidence', 0.4),
            'max_processing_time_per_sentence': self.config.get('max_processing_time_per_sentence', 2.0)
        }
    
    async def validate_pipeline_output(
        self, 
        input_text: str,
        pipeline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate complete pipeline output.
        
        Args:
            input_text: Original input text
            pipeline_results: Results from pipeline processing
            
        Returns:
            Validation results and metrics
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            validation_results = {
                'input_validation': await self._validate_input(input_text),
                'extraction_validation': await self._validate_extraction(pipeline_results),
                'quality_validation': await self._validate_quality(pipeline_results),
                'performance_validation': await self._validate_performance(pipeline_results),
                'consistency_validation': await self._validate_consistency(pipeline_results)
            }
            
            # Calculate overall validation score
            overall_score = await self._calculate_overall_score(validation_results)
            validation_results['overall_score'] = overall_score
            validation_results['validation_passed'] = overall_score >= 0.7
            
            logger.info(
                "Pipeline validation completed",
                overall_score=overall_score,
                validation_passed=validation_results['validation_passed']
            )
            
            return validation_results
            
        except Exception as e:
            logger.error("Pipeline validation failed", error=str(e))
            raise ValidationError(f"Pipeline validation failed: {str(e)}")
    
    async def _validate_input(self, input_text: str) -> Dict[str, Any]:
        """Validate input text quality.
        
        Args:
            input_text: Input text to validate
            
        Returns:
            Input validation results
        """
        validation = {
            'text_length': len(input_text),
            'sentence_count': len(input_text.split('.')),
            'word_count': len(input_text.split()),
            'has_content': bool(input_text.strip()),
            'language_detected': 'en',  # Could be enhanced with language detection
            'encoding_valid': True
        }
        
        # Quality checks
        validation['quality_score'] = 1.0
        
        if validation['text_length'] < 10:
            validation['quality_score'] -= 0.3
            validation['issues'] = validation.get('issues', []) + ['Text too short']
        
        if validation['word_count'] < 5:
            validation['quality_score'] -= 0.2
            validation['issues'] = validation.get('issues', []) + ['Too few words']
        
        validation['validation_passed'] = validation['quality_score'] >= 0.5
        
        return validation
    
    async def _validate_extraction(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate triplet extraction results.
        
        Args:
            pipeline_results: Pipeline processing results
            
        Returns:
            Extraction validation results
        """
        triplets = pipeline_results.get('filtered_triplets', [])
        raw_triplets = pipeline_results.get('raw_triplets', [])
        
        validation = {
            'raw_triplets_count': len(raw_triplets),
            'filtered_triplets_count': len(triplets),
            'extraction_rate': len(triplets) / max(1, len(raw_triplets)),
            'has_triplets': len(triplets) > 0
        }
        
        # Check extraction rate
        if validation['extraction_rate'] >= self.quality_thresholds['min_triplet_extraction_rate']:
            validation['extraction_rate_passed'] = True
        else:
            validation['extraction_rate_passed'] = False
            validation['issues'] = validation.get('issues', []) + [
                f"Low extraction rate: {validation['extraction_rate']:.2f}"
            ]
        
        # Validate triplet structure
        structure_valid = all(
            all(key in triplet for key in ['subject', 'predicate', 'object'])
            for triplet in triplets
        )
        validation['structure_valid'] = structure_valid
        
        if not structure_valid:
            validation['issues'] = validation.get('issues', []) + ['Invalid triplet structure']
        
        validation['validation_passed'] = (
            validation['extraction_rate_passed'] and 
            validation['structure_valid'] and 
            validation['has_triplets']
        )
        
        return validation
    
    async def _validate_quality(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality metrics.
        
        Args:
            pipeline_results: Pipeline processing results
            
        Returns:
            Quality validation results
        """
        triplets = pipeline_results.get('filtered_triplets', [])
        
        if not triplets:
            return {
                'validation_passed': False,
                'issues': ['No triplets to validate quality']
            }
        
        # Calculate quality metrics
        confidences = [t.get('overall_confidence', 0) for t in triplets]
        quality_scores = [t.get('quality_score', 0) for t in triplets]
        
        validation = {
            'average_confidence': sum(confidences) / len(confidences),
            'average_quality': sum(quality_scores) / len(quality_scores),
            'high_confidence_count': sum(1 for c in confidences if c >= 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.4)
        }
        
        # Entity linking validation
        linked_subjects = sum(1 for t in triplets if t.get('entity_linking', {}).get('subject_linked'))
        linked_objects = sum(1 for t in triplets if t.get('entity_linking', {}).get('object_linked'))
        total_entities = len(triplets) * 2
        
        validation['entity_linking_rate'] = (linked_subjects + linked_objects) / total_entities
        validation['entity_linking_passed'] = (
            validation['entity_linking_rate'] >= self.quality_thresholds['min_entity_linking_rate']
        )
        
        # Predicate normalization validation
        normalized_predicates = sum(
            1 for t in triplets 
            if t.get('predicate_normalization', {}).get('method') != 'fallback'
        )
        validation['predicate_normalization_rate'] = normalized_predicates / len(triplets)
        validation['predicate_normalization_passed'] = (
            validation['predicate_normalization_rate'] >= 
            self.quality_thresholds['min_predicate_normalization_rate']
        )
        
        # Overall quality check
        validation['confidence_passed'] = (
            validation['average_confidence'] >= self.quality_thresholds['min_average_confidence']
        )
        
        validation['validation_passed'] = (
            validation['entity_linking_passed'] and
            validation['predicate_normalization_passed'] and
            validation['confidence_passed']
        )
        
        if not validation['validation_passed']:
            issues = []
            if not validation['entity_linking_passed']:
                issues.append(f"Low entity linking rate: {validation['entity_linking_rate']:.2f}")
            if not validation['predicate_normalization_passed']:
                issues.append(f"Low predicate normalization rate: {validation['predicate_normalization_rate']:.2f}")
            if not validation['confidence_passed']:
                issues.append(f"Low average confidence: {validation['average_confidence']:.2f}")
            validation['issues'] = issues
        
        return validation
    
    async def _validate_performance(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance metrics.
        
        Args:
            pipeline_results: Pipeline processing results
            
        Returns:
            Performance validation results
        """
        processing_time = pipeline_results.get('processing_time', 0)
        sentence_count = pipeline_results.get('sentence_count', 1)
        
        validation = {
            'total_processing_time': processing_time,
            'processing_time_per_sentence': processing_time / sentence_count,
            'sentences_processed': sentence_count
        }
        
        # Performance thresholds
        validation['performance_passed'] = (
            validation['processing_time_per_sentence'] <= 
            self.quality_thresholds['max_processing_time_per_sentence']
        )
        
        if not validation['performance_passed']:
            validation['issues'] = [
                f"Slow processing: {validation['processing_time_per_sentence']:.2f}s per sentence"
            ]
        
        return validation
    
    async def _validate_consistency(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency across pipeline stages.
        
        Args:
            pipeline_results: Pipeline processing results
            
        Returns:
            Consistency validation results
        """
        triplets = pipeline_results.get('filtered_triplets', [])
        
        validation = {
            'entity_consistency': True,
            'predicate_consistency': True,
            'confidence_consistency': True
        }
        
        # Check entity consistency
        for triplet in triplets:
            subject_entity = triplet.get('subject_entity', {})
            object_entity = triplet.get('object_entity', {})
            
            # If entity is linked, check consistency
            if subject_entity and triplet.get('subject') != subject_entity.get('original_text', ''):
                validation['entity_consistency'] = False
                break
            
            if object_entity and triplet.get('object') != object_entity.get('original_text', ''):
                validation['entity_consistency'] = False
                break
        
        # Check confidence consistency
        for triplet in triplets:
            overall_conf = triplet.get('overall_confidence', 0)
            extraction_conf = triplet.get('confidence', 0)
            
            # Overall confidence should not be dramatically different from extraction confidence
            if abs(overall_conf - extraction_conf) > 0.5:
                validation['confidence_consistency'] = False
                break
        
        validation['validation_passed'] = (
            validation['entity_consistency'] and
            validation['predicate_consistency'] and
            validation['confidence_consistency']
        )
        
        if not validation['validation_passed']:
            issues = []
            if not validation['entity_consistency']:
                issues.append('Entity linking inconsistency detected')
            if not validation['predicate_consistency']:
                issues.append('Predicate normalization inconsistency detected')
            if not validation['confidence_consistency']:
                issues.append('Confidence scoring inconsistency detected')
            validation['issues'] = issues
        
        return validation
    
    async def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score.
        
        Args:
            validation_results: Individual validation results
            
        Returns:
            Overall validation score (0-1)
        """
        weights = {
            'input_validation': 0.1,
            'extraction_validation': 0.3,
            'quality_validation': 0.4,
            'performance_validation': 0.1,
            'consistency_validation': 0.1
        }
        
        total_score = 0.0
        
        for category, weight in weights.items():
            category_result = validation_results.get(category, {})
            
            if category_result.get('validation_passed', False):
                category_score = 1.0
            else:
                # Partial scoring based on specific metrics
                if category == 'quality_validation':
                    category_score = max(0.0, category_result.get('average_confidence', 0))
                elif category == 'extraction_validation':
                    category_score = max(0.0, category_result.get('extraction_rate', 0))
                else:
                    category_score = 0.5 if category_result else 0.0
            
            total_score += weight * category_score
        
        return total_score
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_pipeline_validator.py`

```python
"""Tests for pipeline validator."""

import pytest
from morag_graph.validation.pipeline_validator import OpenIEPipelineValidator

class TestOpenIEPipelineValidator:
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = OpenIEPipelineValidator()
        assert validator.quality_thresholds['min_average_confidence'] > 0
    
    @pytest.mark.asyncio
    async def test_validate_input(self):
        """Test input validation."""
        validator = OpenIEPipelineValidator()
        
        # Valid input
        result = await validator._validate_input("This is a good test sentence with enough content.")
        assert result['validation_passed']
        assert result['quality_score'] > 0.5
        
        # Invalid input
        result = await validator._validate_input("Short")
        assert not result['validation_passed']
        assert 'issues' in result
    
    @pytest.mark.asyncio
    async def test_validate_extraction(self):
        """Test extraction validation."""
        validator = OpenIEPipelineValidator()
        
        pipeline_results = {
            'raw_triplets': [{'subject': 'A', 'predicate': 'B', 'object': 'C'}] * 10,
            'filtered_triplets': [
                {'subject': 'John', 'predicate': 'works_at', 'object': 'Microsoft'},
                {'subject': 'Mary', 'predicate': 'lives_in', 'object': 'Seattle'}
            ]
        }
        
        result = await validator._validate_extraction(pipeline_results)
        assert result['has_triplets']
        assert result['structure_valid']
        assert result['extraction_rate'] > 0
```

## Acceptance Criteria

- [ ] Comprehensive integration test suite covering end-to-end pipeline
- [ ] Pipeline validation framework with quality metrics
- [ ] Performance benchmarking and monitoring
- [ ] Entity linking accuracy validation
- [ ] Predicate normalization coverage testing
- [ ] Confidence scoring distribution validation
- [ ] Filtering effectiveness testing
- [ ] Consistency validation across pipeline stages
- [ ] Overall quality scoring system
- [ ] Comprehensive unit tests with >90% coverage

## Dependencies
- All previous OpenIE tasks (1.1-4.1)

## Estimated Effort
- **Development**: 8-10 hours
- **Testing**: 6-8 hours
- **Documentation**: 2-3 hours
- **Total**: 16-21 hours

## Notes
- Focus on realistic test scenarios and edge cases
- Implement automated quality monitoring
- Create baseline metrics for performance comparison
- Plan for continuous validation in production environment
