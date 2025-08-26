"""Tests for processing agents."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test-key"

from agents.processing.chunking import ChunkingAgent
from agents.processing.classification import ClassificationAgent
from agents.processing.validation import ValidationAgent
from agents.processing.filtering import FilteringAgent
from agents.processing.models import (
    ChunkingResult,
    ClassificationResult,
    ValidationResult,
    FilteringResult
)
from agents.base.config import AgentConfig


class TestChunkingAgent:
    """Test chunking agent."""
    
    @pytest.fixture
    def chunking_agent(self):
        """Create a chunking agent for testing."""
        config = AgentConfig(name="chunking")
        return ChunkingAgent(config)
    
    def test_agent_initialization(self, chunking_agent):
        """Test agent initialization."""
        assert chunking_agent.config.name == "chunking"
        assert chunking_agent.config.model.provider == "gemini"
    
    @pytest.mark.asyncio
    async def test_semantic_chunking(self, chunking_agent):
        """Test semantic chunking."""
        text = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.
        
        Types of Machine Learning
        
        There are three main types: supervised learning, unsupervised learning, and reinforcement learning.
        
        Applications
        
        Machine learning is used in image recognition, natural language processing, and recommendation systems.
        """
        
        with patch.object(chunking_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "chunks": [
                    {
                        "content": "Introduction to Machine Learning\n\nMachine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
                        "chunk_type": "introduction",
                        "start_pos": 0,
                        "end_pos": 150,
                        "topic": "ML definition",
                        "importance": 0.9
                    },
                    {
                        "content": "Types of Machine Learning\n\nThere are three main types: supervised learning, unsupervised learning, and reinforcement learning.",
                        "chunk_type": "content",
                        "start_pos": 151,
                        "end_pos": 280,
                        "topic": "ML types",
                        "importance": 0.85
                    },
                    {
                        "content": "Applications\n\nMachine learning is used in image recognition, natural language processing, and recommendation systems.",
                        "chunk_type": "content",
                        "start_pos": 281,
                        "end_pos": 400,
                        "topic": "ML applications",
                        "importance": 0.8
                    }
                ],
                "chunk_strategy": "semantic",
                "total_chunks": 3
            }
            
            result = await chunking_agent.chunk_text(text, strategy="semantic")
            
            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) == 3
            assert result.chunk_strategy == "semantic"
            assert result.chunks[0].chunk_type == ChunkType.INTRODUCTION
            assert all(chunk.importance > 0.7 for chunk in result.chunks)
    
    @pytest.mark.asyncio
    async def test_fixed_size_chunking(self, chunking_agent):
        """Test fixed size chunking."""
        text = "This is a long text that needs to be split into fixed-size chunks for processing. " * 10
        
        with patch.object(chunking_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "chunks": [
                    {
                        "content": text[:200],
                        "chunk_type": "content",
                        "start_pos": 0,
                        "end_pos": 200,
                        "topic": "general",
                        "importance": 0.7
                    },
                    {
                        "content": text[200:400],
                        "chunk_type": "content",
                        "start_pos": 200,
                        "end_pos": 400,
                        "topic": "general",
                        "importance": 0.7
                    }
                ],
                "chunk_strategy": "fixed_size",
                "total_chunks": 2
            }
            
            result = await chunking_agent.chunk_text(text, strategy="fixed_size", chunk_size=200)
            
            assert result.chunk_strategy == "fixed_size"
            assert len(result.chunks) == 2
            assert all(len(chunk.content) <= 200 for chunk in result.chunks)


class TestClassificationAgent:
    """Test classification agent."""
    
    @pytest.fixture
    def classification_agent(self):
        """Create a classification agent for testing."""
        config = AgentConfig(name="classification")
        return ClassificationAgent(config)
    
    @pytest.mark.asyncio
    async def test_content_classification(self, classification_agent):
        """Test content classification."""
        text = """
        A 65-year-old male patient presents with chest pain and shortness of breath.
        ECG shows ST-elevation in leads II, III, and aVF.
        Troponin levels are elevated at 15.2 ng/mL.
        Diagnosis: Acute myocardial infarction.
        """
        
        with patch.object(classification_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "primary_category": "medical_record",
                "subcategories": ["cardiology", "emergency_medicine"],
                "confidence": 0.95,
                "classification_features": [
                    "patient_demographics",
                    "clinical_symptoms",
                    "diagnostic_tests",
                    "medical_diagnosis"
                ],
                "domain_specificity": 0.9
            }
            
            result = await classification_agent.classify_content(text)
            
            assert isinstance(result, ClassificationResult)
            assert result.primary_category == ClassificationCategory.MEDICAL_RECORD
            assert "cardiology" in result.subcategories
            assert result.confidence > 0.9
            assert "diagnostic_tests" in result.classification_features
    
    @pytest.mark.asyncio
    async def test_research_paper_classification(self, classification_agent):
        """Test research paper classification."""
        text = """
        Abstract: This study investigates the performance of transformer models
        on natural language understanding tasks. We evaluate BERT, RoBERTa, and GPT
        on the GLUE benchmark, achieving state-of-the-art results.
        
        Keywords: transformers, BERT, natural language processing, GLUE
        """
        
        with patch.object(classification_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "primary_category": "research_paper",
                "subcategories": ["computer_science", "natural_language_processing"],
                "confidence": 0.9,
                "classification_features": [
                    "abstract_section",
                    "keywords_section",
                    "technical_terminology",
                    "benchmark_evaluation"
                ],
                "domain_specificity": 0.95
            }
            
            result = await classification_agent.classify_content(text)
            
            assert result.primary_category == ClassificationCategory.RESEARCH_PAPER
            assert "natural_language_processing" in result.subcategories
            assert result.domain_specificity > 0.9


class TestValidationAgent:
    """Test validation agent."""
    
    @pytest.fixture
    def validation_agent(self):
        """Create a validation agent for testing."""
        config = AgentConfig(name="validation")
        return ValidationAgent(config)
    
    @pytest.mark.asyncio
    async def test_fact_validation(self, validation_agent):
        """Test fact validation."""
        fact = {
            "subject": "Aspirin",
            "object": "heart attack prevention",
            "approach": "daily low-dose administration",
            "solution": "reduced cardiovascular events",
            "confidence": 0.9
        }
        
        with patch.object(validation_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "validation_status": "valid",
                "confidence": 0.9,
                "validation_checks": [
                    {"check": "completeness", "passed": True, "score": 0.95},
                    {"check": "consistency", "passed": True, "score": 0.9},
                    {"check": "plausibility", "passed": True, "score": 0.85}
                ],
                "issues_found": [],
                "suggestions": ["Consider adding dosage information"]
            }
            
            result = await validation_agent.validate_fact(fact)
            
            assert isinstance(result, ValidationResult)
            assert result.validation_status == ValidationStatus.VALID
            assert result.confidence > 0.8
            assert len(result.validation_checks) == 3
            assert all(check["passed"] for check in result.validation_checks)
    
    @pytest.mark.asyncio
    async def test_invalid_fact_validation(self, validation_agent):
        """Test validation of invalid fact."""
        fact = {
            "subject": "Water",
            "object": "cancer cure",
            "approach": "drinking large amounts",
            "solution": "complete cancer elimination",
            "confidence": 0.3
        }
        
        with patch.object(validation_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "validation_status": "invalid",
                "confidence": 0.95,
                "validation_checks": [
                    {"check": "completeness", "passed": True, "score": 0.8},
                    {"check": "consistency", "passed": False, "score": 0.2},
                    {"check": "plausibility", "passed": False, "score": 0.1}
                ],
                "issues_found": [
                    "Implausible medical claim",
                    "Lacks scientific evidence",
                    "Potentially harmful misinformation"
                ],
                "suggestions": ["Remove or correct this claim"]
            }
            
            result = await validation_agent.validate_fact(fact)
            
            assert result.validation_status == ValidationStatus.INVALID
            assert len(result.issues_found) > 0
            assert "implausible" in result.issues_found[0].lower()


class TestFilteringAgent:
    """Test filtering agent."""
    
    @pytest.fixture
    def filtering_agent(self):
        """Create a filtering agent for testing."""
        config = AgentConfig(name="filtering")
        return FilteringAgent(config)
    
    @pytest.mark.asyncio
    async def test_relevance_filtering(self, filtering_agent):
        """Test relevance-based filtering."""
        query = "diabetes treatment"
        candidates = [
            {"text": "Metformin is first-line treatment for type 2 diabetes", "score": 0.9},
            {"text": "Insulin therapy for type 1 diabetes management", "score": 0.85},
            {"text": "Weather forecast for tomorrow", "score": 0.1},
            {"text": "Dietary recommendations for diabetic patients", "score": 0.8}
        ]
        
        with patch.object(filtering_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "filtered_items": [
                    {"text": "Metformin is first-line treatment for type 2 diabetes", "relevance": 0.95, "keep": True},
                    {"text": "Insulin therapy for type 1 diabetes management", "relevance": 0.9, "keep": True},
                    {"text": "Dietary recommendations for diabetic patients", "relevance": 0.85, "keep": True}
                ],
                "filter_criteria": "diabetes_relevance",
                "threshold_applied": 0.7,
                "items_kept": 3,
                "items_filtered": 1
            }
            
            result = await filtering_agent.filter_content(query, candidates, criteria="relevance")
            
            assert isinstance(result, FilteringResult)
            assert len(result.filtered_items) == 3
            assert result.items_kept == 3
            assert result.items_filtered == 1
            assert all(item["relevance"] > 0.8 for item in result.filtered_items)
    
    @pytest.mark.asyncio
    async def test_quality_filtering(self, filtering_agent):
        """Test quality-based filtering."""
        candidates = [
            {"text": "Well-researched medical article with citations", "quality": 0.9},
            {"text": "Blog post with personal opinions only", "quality": 0.3},
            {"text": "Peer-reviewed research study", "quality": 0.95},
            {"text": "Social media post with unverified claims", "quality": 0.2}
        ]
        
        with patch.object(filtering_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "filtered_items": [
                    {"text": "Peer-reviewed research study", "quality": 0.95, "keep": True},
                    {"text": "Well-researched medical article with citations", "quality": 0.9, "keep": True}
                ],
                "filter_criteria": "quality_threshold",
                "threshold_applied": 0.8,
                "items_kept": 2,
                "items_filtered": 2
            }
            
            result = await filtering_agent.filter_content("", candidates, criteria="quality", threshold=0.8)
            
            assert len(result.filtered_items) == 2
            assert result.threshold_applied == 0.8
            assert all(item["quality"] > 0.8 for item in result.filtered_items)


class TestProcessingAgentsIntegration:
    """Test integration between processing agents."""
    
    @pytest.mark.asyncio
    async def test_processing_pipeline(self):
        """Test complete processing pipeline."""
        raw_text = """
        Medical Research on Diabetes Treatment
        
        Type 2 diabetes affects millions worldwide. Current treatments include metformin as first-line therapy.
        Insulin therapy is reserved for advanced cases. Lifestyle modifications remain crucial.
        
        Recent Studies
        
        New research shows promising results with GLP-1 agonists. These medications improve glucose control
        while promoting weight loss. Side effects are generally mild.
        
        Conclusion
        
        Multiple treatment options exist for diabetes management. Personalized approaches yield best outcomes.
        """
        
        # Initialize agents
        chunking_config = AgentConfig(name="chunking")
        classification_config = AgentConfig(name="classification")
        validation_config = AgentConfig(name="validation")
        filtering_config = AgentConfig(name="filtering")
        
        chunking_agent = ChunkingAgent(chunking_config)
        classification_agent = ClassificationAgent(classification_config)
        validation_agent = ValidationAgent(validation_config)
        filtering_agent = FilteringAgent(filtering_config)
        
        # Mock responses
        with patch.object(chunking_agent, '_call_model') as mock_chunking, \
             patch.object(classification_agent, '_call_model') as mock_classification, \
             patch.object(filtering_agent, '_call_model') as mock_filtering:
            
            mock_chunking.return_value = {
                "chunks": [
                    {
                        "content": "Medical Research on Diabetes Treatment\n\nType 2 diabetes affects millions worldwide...",
                        "chunk_type": "introduction",
                        "start_pos": 0,
                        "end_pos": 200,
                        "topic": "diabetes_overview",
                        "importance": 0.9
                    },
                    {
                        "content": "Recent Studies\n\nNew research shows promising results with GLP-1 agonists...",
                        "chunk_type": "content",
                        "start_pos": 201,
                        "end_pos": 400,
                        "topic": "new_treatments",
                        "importance": 0.85
                    }
                ],
                "chunk_strategy": "semantic",
                "total_chunks": 2
            }
            
            mock_classification.return_value = {
                "primary_category": "medical_research",
                "subcategories": ["endocrinology", "diabetes"],
                "confidence": 0.9,
                "classification_features": ["medical_terminology", "research_content"],
                "domain_specificity": 0.95
            }
            
            mock_filtering.return_value = {
                "filtered_items": [
                    {"content": "chunk1", "relevance": 0.9, "keep": True},
                    {"content": "chunk2", "relevance": 0.85, "keep": True}
                ],
                "filter_criteria": "medical_relevance",
                "threshold_applied": 0.7,
                "items_kept": 2,
                "items_filtered": 0
            }
            
            # Run processing pipeline
            chunking_result = await chunking_agent.chunk_text(raw_text)
            classification_result = await classification_agent.classify_content(raw_text)
            filtering_result = await filtering_agent.filter_content(
                "diabetes treatment", 
                [{"text": chunk.content} for chunk in chunking_result.chunks]
            )
            
            # Verify results
            assert len(chunking_result.chunks) == 2
            assert classification_result.primary_category == ClassificationCategory.MEDICAL_RESEARCH
            assert filtering_result.items_kept == 2
            
            print("✅ Processing pipeline test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
