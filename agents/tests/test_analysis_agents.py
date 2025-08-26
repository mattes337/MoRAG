"""Tests for analysis agents."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test-key"

from agents.analysis.query_analysis import QueryAnalysisAgent
from agents.analysis.content_analysis import ContentAnalysisAgent
from agents.analysis.sentiment_analysis import SentimentAnalysisAgent
from agents.analysis.topic_analysis import TopicAnalysisAgent
from agents.analysis.models import (
    QueryAnalysisResult, QueryType, ComplexityLevel, ConfidenceLevel,
    ContentAnalysisResult,
    SentimentAnalysisResult, SentimentPolarity,
    TopicAnalysisResult
)
from agents.base.config import AgentConfig


class TestQueryAnalysisAgent:
    """Test query analysis agent."""
    
    @pytest.fixture
    def query_agent(self):
        """Create a query analysis agent for testing."""
        config = AgentConfig(name="query_analysis")
        return QueryAnalysisAgent(config)
    
    def test_agent_initialization(self, query_agent):
        """Test agent initialization."""
        assert query_agent.config.name == "query_analysis"
        assert query_agent.config.model.provider == "gemini"
    
    @pytest.mark.asyncio
    async def test_simple_query_analysis(self, query_agent):
        """Test analysis of a simple query."""
        query = "What is machine learning?"
        
        with patch.object(query_agent, '_call_model') as mock_llm:
            mock_llm.return_value = """{
                "intent": "question",
                "entities": ["machine learning"],
                "keywords": ["machine learning", "definition"],
                "query_type": "factual",
                "complexity": "simple",
                "confidence": "high",
                "metadata": {
                    "original_query": "What is machine learning?",
                    "query_length": 25,
                    "word_count": 4,
                    "has_context": false,
                    "analysis_method": "llm"
                }
            }"""
            
            result = await query_agent.analyze_query(query)
            
            assert isinstance(result, QueryAnalysisResult)
            assert result.query_type == QueryType.FACTUAL
            assert result.complexity == ComplexityLevel.SIMPLE
            assert "machine learning" in result.entities
            assert result.confidence == ConfidenceLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_complex_query_analysis(self, query_agent):
        """Test analysis of a complex query."""
        query = "How do convolutional neural networks compare to transformer models for image classification tasks in terms of accuracy and computational efficiency?"
        
        with patch.object(query_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "query_type": "comparative",
                "complexity": "complex",
                "intent": "comparison_request",
                "entities": ["convolutional neural networks", "transformer models", "image classification"],
                "keywords": ["CNN", "transformers", "accuracy", "efficiency"],
                "domain": "machine_learning",
                "confidence": 0.95
            }
            
            result = await query_agent.analyze_query(query)
            
            assert result.query_type == QueryType.COMPARATIVE
            assert result.complexity == QueryComplexity.COMPLEX
            assert len(result.entities) >= 3
            assert result.domain == "machine_learning"


class TestContentAnalysisAgent:
    """Test content analysis agent."""
    
    @pytest.fixture
    def content_agent(self):
        """Create a content analysis agent for testing."""
        config = AgentConfig(name="content_analysis")
        return ContentAnalysisAgent(config)
    
    @pytest.mark.asyncio
    async def test_research_paper_analysis(self, content_agent):
        """Test analysis of research paper content."""
        content = """
        Abstract: This paper presents a novel approach to image classification using 
        deep convolutional neural networks. We achieve 95% accuracy on the ImageNet dataset.
        
        Introduction: Computer vision has made significant advances with deep learning...
        
        Methodology: We used a ResNet-50 architecture with data augmentation...
        
        Results: Our model achieved state-of-the-art performance...
        """
        
        with patch.object(content_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "content_type": "research_paper",
                "domain": "computer_vision",
                "structure": {
                    "abstract": True,
                    "introduction": True,
                    "methodology": True,
                    "results": True
                },
                "key_concepts": ["deep learning", "CNN", "image classification"],
                "complexity": "high",
                "quality_score": 0.9
            }
            
            result = await content_agent.analyze_content(content)
            
            assert isinstance(result, ContentAnalysisResult)
            assert result.content_type == ContentType.RESEARCH_PAPER
            assert result.domain == "computer_vision"
            assert "deep learning" in result.key_concepts
            assert result.quality_score > 0.8
    
    @pytest.mark.asyncio
    async def test_medical_content_analysis(self, content_agent):
        """Test analysis of medical content."""
        content = """
        Patient presents with acute chest pain and shortness of breath.
        Diagnosis: Myocardial infarction.
        Treatment: Administered aspirin and nitroglycerin.
        Outcome: Patient stabilized after 2 hours.
        """
        
        with patch.object(content_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "content_type": "medical_record",
                "domain": "cardiology",
                "structure": {
                    "symptoms": True,
                    "diagnosis": True,
                    "treatment": True,
                    "outcome": True
                },
                "key_concepts": ["chest pain", "myocardial infarction", "aspirin"],
                "complexity": "medium",
                "quality_score": 0.85
            }
            
            result = await content_agent.analyze_content(content)
            
            assert result.content_type == ContentType.MEDICAL_RECORD
            assert result.domain == "cardiology"
            assert "myocardial infarction" in result.key_concepts


class TestSentimentAnalysisAgent:
    """Test sentiment analysis agent."""
    
    @pytest.fixture
    def sentiment_agent(self):
        """Create a sentiment analysis agent for testing."""
        config = AgentConfig(name="sentiment_analysis")
        return SentimentAnalysisAgent(config)
    
    @pytest.mark.asyncio
    async def test_positive_sentiment(self, sentiment_agent):
        """Test positive sentiment analysis."""
        text = "This new treatment is absolutely amazing! It completely cured my symptoms and I feel fantastic."
        
        with patch.object(sentiment_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "sentiment": "positive",
                "confidence": 0.95,
                "intensity": 0.9,
                "emotions": ["joy", "relief", "satisfaction"],
                "key_phrases": ["absolutely amazing", "completely cured", "feel fantastic"]
            }
            
            result = await sentiment_agent.analyze_sentiment(text)
            
            assert isinstance(result, SentimentAnalysisResult)
            assert result.sentiment == SentimentType.POSITIVE
            assert result.confidence > 0.9
            assert result.intensity > 0.8
            assert "joy" in result.emotions
    
    @pytest.mark.asyncio
    async def test_negative_sentiment(self, sentiment_agent):
        """Test negative sentiment analysis."""
        text = "This treatment was terrible. It made my symptoms worse and caused severe side effects."
        
        with patch.object(sentiment_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "sentiment": "negative",
                "confidence": 0.9,
                "intensity": 0.8,
                "emotions": ["anger", "disappointment", "frustration"],
                "key_phrases": ["terrible", "made worse", "severe side effects"]
            }
            
            result = await sentiment_agent.analyze_sentiment(text)
            
            assert result.sentiment == SentimentType.NEGATIVE
            assert result.confidence > 0.8
            assert "anger" in result.emotions


class TestTopicAnalysisAgent:
    """Test topic analysis agent."""
    
    @pytest.fixture
    def topic_agent(self):
        """Create a topic analysis agent for testing."""
        config = AgentConfig(name="topic_analysis")
        return TopicAnalysisAgent(config)
    
    @pytest.mark.asyncio
    async def test_medical_topic_analysis(self, topic_agent):
        """Test medical topic analysis."""
        text = """
        Cardiovascular disease remains the leading cause of death worldwide.
        Risk factors include hypertension, diabetes, smoking, and obesity.
        Prevention strategies focus on lifestyle modifications and medication.
        """
        
        with patch.object(topic_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "primary_topic": "cardiovascular_disease",
                "category": "medical",
                "subtopics": ["risk_factors", "prevention", "epidemiology"],
                "confidence": 0.9,
                "keywords": ["cardiovascular", "hypertension", "diabetes", "prevention"],
                "domain_specificity": 0.95
            }
            
            result = await topic_agent.analyze_topics(text)
            
            assert isinstance(result, TopicAnalysisResult)
            assert result.primary_topic == "cardiovascular_disease"
            assert result.category == TopicCategory.MEDICAL
            assert "risk_factors" in result.subtopics
            assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_technology_topic_analysis(self, topic_agent):
        """Test technology topic analysis."""
        text = """
        Artificial intelligence and machine learning are transforming industries.
        Deep learning models achieve unprecedented accuracy in computer vision.
        Natural language processing enables better human-computer interaction.
        """
        
        with patch.object(topic_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "primary_topic": "artificial_intelligence",
                "category": "technology",
                "subtopics": ["machine_learning", "computer_vision", "nlp"],
                "confidence": 0.95,
                "keywords": ["AI", "deep learning", "computer vision", "NLP"],
                "domain_specificity": 0.9
            }
            
            result = await topic_agent.analyze_topics(text)
            
            assert result.primary_topic == "artificial_intelligence"
            assert result.category == TopicCategory.TECHNOLOGY
            assert "machine_learning" in result.subtopics


class TestAnalysisAgentsIntegration:
    """Test integration between analysis agents."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_pipeline(self):
        """Test comprehensive analysis pipeline."""
        text = "I'm really excited about this new AI breakthrough in medical diagnosis!"
        
        # Initialize agents
        query_config = AgentConfig(name="query_analysis")
        content_config = AgentConfig(name="content_analysis")
        sentiment_config = AgentConfig(name="sentiment_analysis")
        topic_config = AgentConfig(name="topic_analysis")
        
        query_agent = QueryAnalysisAgent(query_config)
        content_agent = ContentAnalysisAgent(content_config)
        sentiment_agent = SentimentAnalysisAgent(sentiment_config)
        topic_agent = TopicAnalysisAgent(topic_config)
        
        # Mock responses
        with patch.object(sentiment_agent, '_call_model') as mock_sentiment, \
             patch.object(topic_agent, '_call_model') as mock_topic, \
             patch.object(content_agent, '_call_model') as mock_content:
            
            mock_sentiment.return_value = {
                "sentiment": "positive",
                "confidence": 0.9,
                "intensity": 0.8,
                "emotions": ["excitement"],
                "key_phrases": ["really excited"]
            }
            
            mock_topic.return_value = {
                "primary_topic": "medical_ai",
                "category": "technology",
                "subtopics": ["AI", "medical_diagnosis"],
                "confidence": 0.9,
                "keywords": ["AI", "medical", "diagnosis"],
                "domain_specificity": 0.85
            }
            
            mock_content.return_value = {
                "content_type": "social_media",
                "domain": "technology",
                "structure": {"informal": True},
                "key_concepts": ["AI", "medical diagnosis"],
                "complexity": "low",
                "quality_score": 0.7
            }
            
            # Run analysis pipeline
            sentiment_result = await sentiment_agent.analyze_sentiment(text)
            topic_result = await topic_agent.analyze_topics(text)
            content_result = await content_agent.analyze_content(text)
            
            # Verify results
            assert sentiment_result.sentiment == SentimentType.POSITIVE
            assert topic_result.category == TopicCategory.TECHNOLOGY
            assert content_result.content_type == ContentType.SOCIAL_MEDIA
            
            print("✅ Analysis pipeline test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
