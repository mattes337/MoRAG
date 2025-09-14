"""Tests for generation agents."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test-key"

from agents.generation.summarization import SummarizationAgent
from agents.generation.response_generation import ResponseGenerationAgent
from agents.generation.explanation import ExplanationAgent
from agents.generation.synthesis import SynthesisAgent
from agents.generation.models import (
    SummarizationResult,
    ResponseGenerationResult,
    ExplanationResult,
    SynthesisResult
)
from agents.base.config import AgentConfig


class TestSummarizationAgent:
    """Test summarization agent."""
    
    @pytest.fixture
    def summary_agent(self):
        """Create a summarization agent for testing."""
        config = AgentConfig(name="summarization")
        return SummarizationAgent(config)
    
    def test_agent_initialization(self, summary_agent):
        """Test agent initialization."""
        assert summary_agent.config.name == "summarization"
        assert summary_agent.config.model.provider == "gemini"
    
    @pytest.mark.asyncio
    async def test_research_paper_summarization(self, summary_agent):
        """Test research paper summarization."""
        text = """
        This study investigates the effectiveness of deep learning models for medical image analysis.
        We trained convolutional neural networks on a dataset of 10,000 chest X-rays.
        The model achieved 95% accuracy in detecting pneumonia cases.
        Our results demonstrate that AI can significantly improve diagnostic accuracy.
        The implications for clinical practice are substantial, potentially reducing misdiagnosis rates.
        """
        
        with patch.object(summary_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "summary": "Study shows deep learning models achieve 95% accuracy in pneumonia detection from chest X-rays, demonstrating AI's potential to improve medical diagnosis.",
                "summary_type": "abstractive",
                "key_points": [
                    "Deep learning for medical image analysis",
                    "95% accuracy in pneumonia detection",
                    "Potential to reduce misdiagnosis"
                ],
                "compression_ratio": 0.3,
                "confidence": "high"
            }
            
            result = await summary_agent.summarize(text, summary_type="abstractive")
            
            assert isinstance(result, SummarizationResult)
            assert result.summary_type == "abstractive"
            assert "95% accuracy" in result.summary
            assert len(result.key_points) == 3
            assert result.compression_ratio < 0.5
    
    @pytest.mark.asyncio
    async def test_extractive_summarization(self, summary_agent):
        """Test extractive summarization."""
        text = """
        Machine learning algorithms are transforming healthcare.
        Deep neural networks can analyze medical images with high precision.
        Natural language processing helps extract insights from clinical notes.
        These technologies promise to improve patient outcomes significantly.
        """
        
        with patch.object(summary_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "summary": "Machine learning algorithms are transforming healthcare. These technologies promise to improve patient outcomes significantly.",
                "summary_type": "extractive",
                "key_points": [
                    "ML transforming healthcare",
                    "Improved patient outcomes"
                ],
                "compression_ratio": 0.5,
                "confidence": "high"
            }
            
            result = await summary_agent.summarize(text, summary_type="extractive")
            
            assert result.summary_type == "extractive"
            assert "Machine learning algorithms" in result.summary
            assert result.compression_ratio == 0.5


class TestResponseGenerationAgent:
    """Test response generation agent."""
    
    @pytest.fixture
    def response_agent(self):
        """Create a response generation agent for testing."""
        config = AgentConfig(name="response_generation")
        return ResponseGenerationAgent(config)
    
    @pytest.mark.asyncio
    async def test_informative_response(self, response_agent):
        """Test informative response generation."""
        query = "What are the symptoms of diabetes?"
        context = [
            "Diabetes is a metabolic disorder characterized by high blood glucose",
            "Common symptoms include excessive thirst, frequent urination, fatigue",
            "Type 1 and Type 2 diabetes have similar symptoms but different causes"
        ]
        
        with patch.object(response_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "response": "Diabetes symptoms include excessive thirst (polydipsia), frequent urination (polyuria), unexplained fatigue, and blurred vision. These symptoms occur because high blood glucose levels affect normal body functions.",
                "response_type": "informative",
                "sources": [],
                "confidence": "high",
                "citations": [],
                "metadata": {
                    "sources_used": 2,
                    "completeness": 0.85
                }
            }
            
            result = await response_agent.generate_response(query, context)
            
            assert isinstance(result, ResponseGenerationResult)
            assert result.response_type == "informative"
            assert "polydipsia" in result.response
            assert result.confidence == "high"
            assert result.metadata.get("sources_used", 0) >= 2
    
    @pytest.mark.asyncio
    async def test_explanatory_response(self, response_agent):
        """Test explanatory response generation."""
        query = "Why does diabetes cause frequent urination?"
        context = [
            "High blood glucose levels exceed kidney filtration capacity",
            "Excess glucose is excreted in urine, drawing water with it",
            "This osmotic effect leads to increased urine production"
        ]
        
        with patch.object(response_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "response": "Diabetes causes frequent urination through an osmotic mechanism. When blood glucose levels are high, the kidneys cannot reabsorb all the glucose, so it spills into the urine. Glucose in urine draws water with it through osmosis, resulting in increased urine volume and frequency.",
                "response_type": "explanatory",
                "sources": [],
                "confidence": "high",
                "citations": [],
                "metadata": {
                    "sources_used": 3,
                    "completeness": 0.9
                }
            }
            
            result = await response_agent.generate_response(query, context, response_type="explanatory")
            
            assert result.response_type == "explanatory"
            assert "osmotic" in result.response
            assert result.metadata.get("completeness", 0) > 0.8


class TestExplanationAgent:
    """Test explanation agent."""
    
    @pytest.fixture
    def explanation_agent(self):
        """Create an explanation agent for testing."""
        config = AgentConfig(name="explanation")
        return ExplanationAgent(config)
    
    @pytest.mark.asyncio
    async def test_causal_explanation(self, explanation_agent):
        """Test causal explanation."""
        phenomenon = "Why do antibiotics not work against viral infections?"
        context = [
            "Antibiotics target bacterial cell structures",
            "Viruses lack cell walls and ribosomes",
            "Viral replication uses host cell machinery"
        ]
        
        with patch.object(explanation_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "explanation": "Antibiotics don't work against viruses because they target specific bacterial structures like cell walls and ribosomes that viruses don't possess. Viruses are much simpler organisms that hijack host cell machinery for replication, making them immune to antibiotic mechanisms.",
                "explanation_type": "causal",
                "reasoning_steps": [
                    "Antibiotics target bacterial structures",
                    "Viruses lack these target structures",
                    "Therefore, antibiotics cannot affect viruses"
                ],
                "examples": [],
                "confidence": "high",
                "metadata": {
                    "clarity_score": 0.9
                }
            }
            
            result = await explanation_agent.explain(phenomenon, context)
            
            assert isinstance(result, ExplanationResult)
            assert result.explanation_type == "causal"
            assert "cell walls" in result.explanation
            assert len(result.reasoning_steps) == 3
            assert result.confidence == "high"
    
    @pytest.mark.asyncio
    async def test_mechanistic_explanation(self, explanation_agent):
        """Test mechanistic explanation."""
        phenomenon = "How does insulin regulate blood glucose?"
        context = [
            "Insulin binds to cell surface receptors",
            "This triggers glucose transporter activation",
            "Glucose uptake by cells increases"
        ]
        
        with patch.object(explanation_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "explanation": "Insulin regulates blood glucose through a receptor-mediated mechanism. When insulin binds to insulin receptors on cell surfaces, it triggers a cascade that activates glucose transporters (GLUT4), allowing cells to take up glucose from the bloodstream, thereby lowering blood glucose levels.",
                "explanation_type": "mechanistic",
                "reasoning_steps": [
                    "Insulin binds to receptors",
                    "Signaling cascade activates",
                    "GLUT4 transporters activated",
                    "Glucose uptake increases"
                ],
                "examples": [],
                "confidence": "high",
                "metadata": {
                    "clarity_score": 0.85
                }
            }
            
            result = await explanation_agent.explain(phenomenon, "mechanistic", context=context)
            
            assert result.explanation_type == "mechanistic"
            assert "GLUT4" in result.explanation
            assert len(result.reasoning_steps) == 4


class TestSynthesisAgent:
    """Test synthesis agent."""
    
    @pytest.fixture
    def synthesis_agent(self):
        """Create a synthesis agent for testing."""
        config = AgentConfig(name="synthesis")
        return SynthesisAgent(config)
    
    @pytest.mark.asyncio
    async def test_comparative_synthesis(self, synthesis_agent):
        """Test comparative synthesis."""
        sources = [
            "Study A: Drug X shows 80% efficacy in treating condition Y",
            "Study B: Drug X demonstrates 75% success rate with minimal side effects",
            "Study C: Drug X effective in 85% of cases but causes nausea in 20% of patients"
        ]
        
        with patch.object(synthesis_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "synthesis": "Multiple studies demonstrate Drug X's effectiveness for condition Y, with efficacy rates ranging from 75-85%. While the drug shows consistent therapeutic benefit, side effects including nausea occur in approximately 20% of patients, requiring careful risk-benefit assessment.",
                "sources_integrated": 3,
                "coherence_score": 0.9,
                "confidence": "high",
                "metadata": {
                    "synthesis_type": "comparative",
                    "key_insights": [
                        "Consistent efficacy across studies (75-85%)",
                        "Notable side effect profile (20% nausea)",
                        "Positive risk-benefit ratio"
                    ],
                    "source_coverage": 1.0
                }
            }
            
            result = await synthesis_agent.synthesize(sources, synthesis_type="comparative")
            
            assert isinstance(result, SynthesisResult)
            assert result.metadata.get("synthesis_type") == "comparative"
            assert "75-85%" in result.synthesis
            assert len(result.metadata.get("key_insights", [])) == 3
            assert result.metadata.get("source_coverage") == 1.0
    
    @pytest.mark.asyncio
    async def test_integrative_synthesis(self, synthesis_agent):
        """Test integrative synthesis."""
        sources = [
            "Genetic factors contribute to diabetes risk",
            "Environmental factors like diet affect diabetes development",
            "Lifestyle interventions can prevent type 2 diabetes"
        ]
        
        with patch.object(synthesis_agent, '_call_model') as mock_llm:
            mock_llm.return_value = {
                "synthesis": "Diabetes development involves complex interactions between genetic predisposition and environmental factors. While genetic factors establish baseline risk, environmental influences like diet and lifestyle play crucial roles in disease manifestation, suggesting that targeted lifestyle interventions can effectively prevent type 2 diabetes even in genetically susceptible individuals.",
                "sources_integrated": 3,
                "coherence_score": 0.85,
                "confidence": "high",
                "metadata": {
                    "synthesis_type": "integrative",
                    "key_insights": [
                        "Gene-environment interaction model",
                        "Lifestyle interventions overcome genetic risk",
                        "Prevention possible through behavior modification"
                    ],
                    "source_coverage": 1.0
                }
            }
            
            result = await synthesis_agent.synthesize(sources, synthesis_type="integrative")
            
            assert result.metadata.get("synthesis_type") == "integrative"
            assert "genetic" in result.synthesis.lower() and "environmental" in result.synthesis.lower()
            assert result.confidence == "high"


class TestGenerationAgentsIntegration:
    """Test integration between generation agents."""
    
    @pytest.mark.asyncio
    async def test_generation_pipeline(self):
        """Test complete generation pipeline."""
        query = "Explain the relationship between diabetes and cardiovascular disease"
        raw_content = [
            "Diabetes increases cardiovascular disease risk by 2-4 fold",
            "High glucose damages blood vessel walls",
            "Insulin resistance promotes inflammation",
            "Diabetic patients often have hypertension and dyslipidemia"
        ]
        
        # Initialize agents
        summary_config = AgentConfig(name="summarization")
        response_config = AgentConfig(name="response_generation")
        explanation_config = AgentConfig(name="explanation")
        synthesis_config = AgentConfig(name="synthesis")
        
        summary_agent = SummarizationAgent(summary_config)
        response_agent = ResponseGenerationAgent(response_config)
        explanation_agent = ExplanationAgent(explanation_config)
        synthesis_agent = SynthesisAgent(synthesis_config)
        
        # Mock responses
        with patch.object(synthesis_agent, '_call_model') as mock_synthesis, \
             patch.object(explanation_agent, '_call_model') as mock_explanation, \
             patch.object(response_agent, '_call_model') as mock_response:
            
            mock_synthesis.return_value = {
                "synthesis": "Diabetes significantly increases cardiovascular disease risk through multiple mechanisms including vascular damage, inflammation, and metabolic dysfunction.",
                "sources_integrated": 4,
                "coherence_score": 0.9,
                "confidence": "high",
                "metadata": {
                    "synthesis_type": "integrative",
                    "key_insights": ["Vascular damage", "Inflammation", "Metabolic dysfunction"],
                    "source_coverage": 1.0
                }
            }
            
            mock_explanation.return_value = {
                "explanation": "Diabetes causes cardiovascular disease through hyperglycemia-induced endothelial damage, chronic inflammation from insulin resistance, and associated metabolic abnormalities.",
                "explanation_type": "causal",
                "reasoning_steps": ["Hyperglycemia damages vessels", "Insulin resistance causes inflammation"],
                "examples": [],
                "confidence": "high",
                "metadata": {
                    "clarity_score": 0.85
                }
            }

            mock_response.return_value = {
                "response": "Diabetes and cardiovascular disease are closely linked through multiple pathophysiological mechanisms, resulting in 2-4 fold increased risk for diabetic patients.",
                "response_type": "explanatory",
                "sources": [],
                "confidence": "high",
                "citations": [],
                "metadata": {
                    "sources_used": 4,
                    "completeness": 0.9
                }
            }
            
            # Run generation pipeline
            synthesis_result = await synthesis_agent.synthesize(raw_content)
            explanation_result = await explanation_agent.explain(query, raw_content)
            response_result = await response_agent.generate_response(query, raw_content)
            
            # Verify results
            assert "cardiovascular" in synthesis_result.synthesis
            assert explanation_result.explanation_type == "causal"
            assert response_result.confidence == "high"
            
            print("✅ Generation pipeline test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
