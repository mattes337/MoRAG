"""Tests for OpenIE service."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from morag_graph.services.openie_service import OpenIEService, OpenIETriplet


class TestOpenIEService:
    """Test cases for OpenIE service."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'enabled': True,
            'implementation': 'stanford',
            'confidence_threshold': 0.5,
            'max_triplets_per_sentence': 5,
            'batch_size': 10,
            'timeout_seconds': 10
        }
    
    @pytest.fixture
    def openie_service(self, mock_config):
        """Create OpenIE service instance for testing."""
        return OpenIEService(config=mock_config)
    
    def test_init(self, openie_service):
        """Test service initialization."""
        assert openie_service.enabled is True
        assert openie_service.implementation == 'stanford'
        assert openie_service.confidence_threshold == 0.5
        assert openie_service.max_triplets_per_sentence == 5
        assert openie_service.batch_size == 10
        assert openie_service.timeout_seconds == 10
    
    def test_init_disabled(self):
        """Test service initialization when disabled."""
        config = {'enabled': False}
        service = OpenIEService(config=config)
        assert service.enabled is False
    
    @pytest.mark.asyncio
    async def test_extract_triplets_disabled(self):
        """Test triplet extraction when service is disabled."""
        config = {'enabled': False}
        service = OpenIEService(config=config)
        
        triplets = await service.extract_triplets("John loves Mary.")
        assert triplets == []
    
    @pytest.mark.asyncio
    async def test_extract_triplets_empty_text(self, openie_service):
        """Test triplet extraction with empty text."""
        triplets = await openie_service.extract_triplets("")
        assert triplets == []
        
        triplets = await openie_service.extract_triplets("   ")
        assert triplets == []
    
    @pytest.mark.asyncio
    @patch('morag_graph.services.openie_service.StanfordOpenIE')
    async def test_extract_triplets_success(self, mock_stanford_openie, openie_service):
        """Test successful triplet extraction."""
        # Mock Stanford OpenIE client
        mock_client = Mock()
        mock_client.annotate.return_value = [
            {
                'subject': 'John',
                'relation': 'loves',
                'object': 'Mary',
                'confidence': 0.8
            },
            {
                'subject': 'Mary',
                'relation': 'works at',
                'object': 'Google',
                'confidence': 0.9
            }
        ]
        mock_stanford_openie.return_value = mock_client
        
        # Mock sentence splitting
        with patch.object(openie_service, '_split_sentences', return_value=['John loves Mary.', 'Mary works at Google.']):
            triplets = await openie_service.extract_triplets("John loves Mary. Mary works at Google.")
        
        assert len(triplets) == 2
        assert triplets[0].subject == 'John'
        assert triplets[0].predicate == 'loves'
        assert triplets[0].object == 'Mary'
        assert triplets[0].confidence == 0.8
        
        assert triplets[1].subject == 'Mary'
        assert triplets[1].predicate == 'works at'
        assert triplets[1].object == 'Google'
        assert triplets[1].confidence == 0.9
    
    @pytest.mark.asyncio
    @patch('morag_graph.services.openie_service.StanfordOpenIE')
    async def test_extract_triplets_confidence_filtering(self, mock_stanford_openie, openie_service):
        """Test confidence-based filtering of triplets."""
        # Mock Stanford OpenIE client
        mock_client = Mock()
        mock_client.annotate.return_value = [
            {
                'subject': 'John',
                'relation': 'loves',
                'object': 'Mary',
                'confidence': 0.8  # Above threshold (0.5)
            },
            {
                'subject': 'Mary',
                'relation': 'likes',
                'object': 'pizza',
                'confidence': 0.3  # Below threshold (0.5)
            }
        ]
        mock_stanford_openie.return_value = mock_client
        
        # Mock sentence splitting
        with patch.object(openie_service, '_split_sentences', return_value=['John loves Mary and Mary likes pizza.']):
            triplets = await openie_service.extract_triplets("John loves Mary and Mary likes pizza.")
        
        # Only the high-confidence triplet should be returned
        assert len(triplets) == 1
        assert triplets[0].subject == 'John'
        assert triplets[0].confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_split_sentences_simple(self, openie_service):
        """Test simple sentence splitting."""
        text = "John loves Mary. She works at Google! How are you?"
        sentences = await openie_service._split_sentences(text)
        
        assert len(sentences) >= 2  # At least 2 meaningful sentences
        assert any('John loves Mary' in s for s in sentences)
        assert any('works at Google' in s for s in sentences)
    
    @pytest.mark.asyncio
    async def test_split_sentences_filters_short(self, openie_service):
        """Test that very short sentences are filtered out."""
        text = "Hi. John loves Mary. Ok."
        sentences = await openie_service._split_sentences(text)
        
        # Should filter out very short sentences like "Hi" and "Ok"
        assert all(len(s.strip()) > 10 for s in sentences)
        assert any('John loves Mary' in s for s in sentences)
    
    @pytest.mark.asyncio
    async def test_close(self, openie_service):
        """Test service cleanup."""
        # Should not raise any exceptions
        await openie_service.close()
    
    def test_openie_triplet_creation(self):
        """Test OpenIETriplet creation."""
        triplet = OpenIETriplet(
            subject="John",
            predicate="loves",
            object="Mary",
            confidence=0.8,
            sentence="John loves Mary."
        )
        
        assert triplet.subject == "John"
        assert triplet.predicate == "loves"
        assert triplet.object == "Mary"
        assert triplet.confidence == 0.8
        assert triplet.sentence == "John loves Mary."
        assert triplet.start_pos == 0  # Default value
        assert triplet.end_pos == 0  # Default value
    
    @pytest.mark.asyncio
    @patch('morag_graph.services.openie_service.StanfordOpenIE')
    async def test_initialization_error_handling(self, mock_stanford_openie):
        """Test error handling during initialization."""
        mock_stanford_openie.side_effect = Exception("Initialization failed")
        
        service = OpenIEService()
        
        with pytest.raises(Exception):  # Should raise ConfigurationError
            await service.initialize()
    
    @pytest.mark.asyncio
    async def test_unsupported_implementation(self):
        """Test error handling for unsupported implementation."""
        config = {'implementation': 'unsupported'}
        service = OpenIEService(config=config)
        
        with pytest.raises(Exception):  # Should raise ConfigurationError
            await service.initialize()


if __name__ == "__main__":
    pytest.main([__file__])
