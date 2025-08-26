"""Tests for reasoning agents."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test-key"

from agents.reasoning.path_selection import PathSelectionAgent
from agents.reasoning.reasoning import ReasoningAgent
from agents.reasoning.decision_making import DecisionMakingAgent
from agents.reasoning.context_analysis import ContextAnalysisAgent
from agents.reasoning.models import (
    PathSelectionResult, ReasoningPath, PathType,
    ReasoningResult, ReasoningStep, ReasoningType,
    DecisionResult, DecisionOption, DecisionCriteria,
    ContextAnalysisResult, ContextType, ContextRelevance
)
from agents.base.config import AgentConfig


class TestPathSelectionAgent:
    """Test path selection agent."""
    
    @pytest.fixture
    def path_agent(self):
        """Create a path selection agent for testing."""
        config = AgentConfig(name="path_selection")
        return PathSelectionAgent(config)
    
    def test_agent_initialization(self, path_agent):
        """Test agent initialization."""
        assert path_agent.config.name == "path_selection"
        assert path_agent.config.model.provider == "gemini"
    
    @pytest.mark.asyncio
    async def test_simple_path_selection(self, path_agent):
        """Test simple path selection."""
        query = "What causes diabetes?"
        available_paths = [
            {"path_id": "direct_search", "description": "Direct medical database search"},
            {"path_id": "graph_traversal", "description": "Knowledge graph traversal"},
            {"path_id": "semantic_search", "description": "Semantic similarity search"}
        ]
        
        with patch.object(path_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_paths": [
                    {
                        "path_id": "direct_search",
                        "path_type": "direct",
                        "confidence": 0.9,
                        "reasoning": "Medical query best served by direct database search",
                        "expected_quality": 0.85
                    },
                    {
                        "path_id": "graph_traversal",
                        "path_type": "graph",
                        "confidence": 0.7,
                        "reasoning": "Graph can provide related concepts",
                        "expected_quality": 0.75
                    }
                ],
                "primary_path": "direct_search",
                "strategy": "medical_focused"
            }
            
            result = await path_agent.select_paths(query, available_paths)
            
            assert isinstance(result, PathSelectionResult)
            assert len(result.selected_paths) == 2
            assert result.primary_path == "direct_search"
            assert result.selected_paths[0].confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_complex_path_selection(self, path_agent):
        """Test complex path selection with multiple criteria."""
        query = "How do machine learning algorithms compare in terms of accuracy and efficiency for medical diagnosis?"
        available_paths = [
            {"path_id": "comparative_analysis", "description": "Comparative analysis engine"},
            {"path_id": "performance_metrics", "description": "Performance metrics database"},
            {"path_id": "research_papers", "description": "Research paper analysis"},
            {"path_id": "case_studies", "description": "Medical case studies"}
        ]
        
        with patch.object(path_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_paths": [
                    {
                        "path_id": "comparative_analysis",
                        "path_type": "analytical",
                        "confidence": 0.95,
                        "reasoning": "Perfect for comparison queries",
                        "expected_quality": 0.9
                    },
                    {
                        "path_id": "performance_metrics",
                        "path_type": "data",
                        "confidence": 0.85,
                        "reasoning": "Provides quantitative data",
                        "expected_quality": 0.8
                    }
                ],
                "primary_path": "comparative_analysis",
                "strategy": "multi_source_comparison"
            }
            
            result = await path_agent.select_paths(query, available_paths)
            
            assert result.primary_path == "comparative_analysis"
            assert len(result.selected_paths) == 2
            assert result.strategy == "multi_source_comparison"


class TestReasoningAgent:
    """Test reasoning agent."""
    
    @pytest.fixture
    def reasoning_agent(self):
        """Create a reasoning agent for testing."""
        config = AgentConfig(name="reasoning")
        return ReasoningAgent(config)
    
    @pytest.mark.asyncio
    async def test_deductive_reasoning(self, reasoning_agent):
        """Test deductive reasoning."""
        premises = [
            "All patients with diabetes have elevated blood glucose",
            "John has diabetes",
            "Therefore, John has elevated blood glucose"
        ]
        
        with patch.object(reasoning_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "reasoning_type": "deductive",
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Identify major premise",
                        "content": "All patients with diabetes have elevated blood glucose",
                        "confidence": 0.95
                    },
                    {
                        "step_number": 2,
                        "description": "Identify minor premise",
                        "content": "John has diabetes",
                        "confidence": 0.9
                    },
                    {
                        "step_number": 3,
                        "description": "Draw logical conclusion",
                        "content": "John has elevated blood glucose",
                        "confidence": 0.9
                    }
                ],
                "conclusion": "John has elevated blood glucose",
                "validity": True,
                "confidence": 0.9
            }
            
            result = await reasoning_agent.reason(premises)
            
            assert isinstance(result, ReasoningResult)
            assert result.reasoning_type == ReasoningType.DEDUCTIVE
            assert len(result.steps) == 3
            assert result.validity == True
            assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_inductive_reasoning(self, reasoning_agent):
        """Test inductive reasoning."""
        observations = [
            "Patient A with hypertension developed heart disease",
            "Patient B with hypertension developed heart disease",
            "Patient C with hypertension developed heart disease"
        ]
        
        with patch.object(reasoning_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "reasoning_type": "inductive",
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Analyze pattern in observations",
                        "content": "Multiple patients with hypertension developed heart disease",
                        "confidence": 0.8
                    },
                    {
                        "step_number": 2,
                        "description": "Generalize from specific cases",
                        "content": "Hypertension may be a risk factor for heart disease",
                        "confidence": 0.75
                    }
                ],
                "conclusion": "Hypertension is likely a risk factor for heart disease",
                "validity": True,
                "confidence": 0.75
            }
            
            result = await reasoning_agent.reason(observations)
            
            assert result.reasoning_type == ReasoningType.INDUCTIVE
            assert len(result.steps) == 2
            assert "risk factor" in result.conclusion


class TestDecisionMakingAgent:
    """Test decision making agent."""
    
    @pytest.fixture
    def decision_agent(self):
        """Create a decision making agent for testing."""
        config = AgentConfig(name="decision_making")
        return DecisionMakingAgent(config)
    
    @pytest.mark.asyncio
    async def test_treatment_decision(self, decision_agent):
        """Test medical treatment decision making."""
        context = "Patient with type 2 diabetes, BMI 32, age 55, no kidney disease"
        options = [
            {"option": "metformin", "description": "First-line diabetes medication"},
            {"option": "insulin", "description": "Injectable glucose control"},
            {"option": "lifestyle_only", "description": "Diet and exercise only"}
        ]
        criteria = ["effectiveness", "side_effects", "patient_compliance", "cost"]
        
        with patch.object(decision_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "recommended_option": "metformin",
                "confidence": 0.9,
                "reasoning": "First-line treatment for type 2 diabetes with good safety profile",
                "option_scores": [
                    {
                        "option": "metformin",
                        "total_score": 0.85,
                        "criteria_scores": {
                            "effectiveness": 0.8,
                            "side_effects": 0.9,
                            "patient_compliance": 0.85,
                            "cost": 0.9
                        }
                    },
                    {
                        "option": "insulin",
                        "total_score": 0.7,
                        "criteria_scores": {
                            "effectiveness": 0.95,
                            "side_effects": 0.6,
                            "patient_compliance": 0.5,
                            "cost": 0.7
                        }
                    }
                ],
                "risk_assessment": "low_risk"
            }
            
            result = await decision_agent.make_decision(context, options, criteria)
            
            assert isinstance(result, DecisionResult)
            assert result.recommended_option == "metformin"
            assert result.confidence > 0.8
            assert len(result.option_scores) == 2
            assert result.risk_assessment == "low_risk"


class TestContextAnalysisAgent:
    """Test context analysis agent."""
    
    @pytest.fixture
    def context_agent(self):
        """Create a context analysis agent for testing."""
        config = AgentConfig(name="context_analysis")
        return ContextAnalysisAgent(config)
    
    @pytest.mark.asyncio
    async def test_medical_context_analysis(self, context_agent):
        """Test medical context analysis."""
        query = "What medication should I take?"
        context_info = {
            "patient_age": 65,
            "medical_history": ["hypertension", "diabetes"],
            "current_medications": ["lisinopril", "metformin"],
            "allergies": ["penicillin"]
        }
        
        with patch.object(context_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "context_type": "medical_consultation",
                "relevance": "high",
                "key_factors": [
                    "patient_age",
                    "medical_history",
                    "drug_interactions",
                    "allergies"
                ],
                "risk_factors": ["age_related_sensitivity", "drug_interactions"],
                "recommendations": [
                    "Consider age-appropriate dosing",
                    "Check for drug interactions",
                    "Avoid penicillin-based medications"
                ],
                "confidence": 0.9
            }
            
            result = await context_agent.analyze_context(query, context_info)
            
            assert isinstance(result, ContextAnalysisResult)
            assert result.context_type == ContextType.MEDICAL_CONSULTATION
            assert result.relevance == ContextRelevance.HIGH
            assert "drug_interactions" in result.key_factors
            assert len(result.recommendations) >= 3


class TestReasoningAgentsIntegration:
    """Test integration between reasoning agents."""
    
    @pytest.mark.asyncio
    async def test_reasoning_pipeline(self):
        """Test complete reasoning pipeline."""
        query = "Should this patient receive surgery?"
        context = {
            "patient_age": 75,
            "condition": "coronary artery disease",
            "comorbidities": ["diabetes", "hypertension"],
            "surgical_risk": "moderate"
        }
        
        # Initialize agents
        path_config = AgentConfig(name="path_selection")
        reasoning_config = AgentConfig(name="reasoning")
        decision_config = AgentConfig(name="decision_making")
        context_config = AgentConfig(name="context_analysis")
        
        path_agent = PathSelectionAgent(path_config)
        reasoning_agent = ReasoningAgent(reasoning_config)
        decision_agent = DecisionMakingAgent(decision_config)
        context_agent = ContextAnalysisAgent(context_config)
        
        # Mock responses
        with patch.object(context_agent, '_call_llm') as mock_context, \
             patch.object(decision_agent, '_call_llm') as mock_decision:
            
            mock_context.return_value = {
                "context_type": "medical_consultation",
                "relevance": "high",
                "key_factors": ["age", "comorbidities", "surgical_risk"],
                "risk_factors": ["advanced_age", "multiple_comorbidities"],
                "recommendations": ["Consider non-surgical options"],
                "confidence": 0.85
            }
            
            mock_decision.return_value = {
                "recommended_option": "conservative_treatment",
                "confidence": 0.8,
                "reasoning": "High surgical risk due to age and comorbidities",
                "option_scores": [],
                "risk_assessment": "high_risk"
            }
            
            # Run reasoning pipeline
            context_result = await context_agent.analyze_context(query, context)
            decision_result = await decision_agent.make_decision(
                str(context), 
                [{"option": "surgery"}, {"option": "conservative_treatment"}],
                ["effectiveness", "risk", "quality_of_life"]
            )
            
            # Verify results
            assert context_result.relevance == ContextRelevance.HIGH
            assert decision_result.recommended_option == "conservative_treatment"
            assert decision_result.risk_assessment == "high_risk"
            
            print("✅ Reasoning pipeline test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
