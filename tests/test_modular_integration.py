"""Integration tests for the modular MoRAG system."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

# Test imports for all packages
try:
    from morag import MoRAGAPI, MoRAGOrchestrator
    from morag_core.models import Document, DocumentChunk, ProcessingResult
    from morag_services import MoRAGServices, ServiceConfig, ContentType
    from morag_web import WebProcessor, WebConverter
    from morag_youtube import YouTubeProcessor
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Package imports failed: {IMPORT_ERROR}")
class TestModularIntegration:
    """Test integration between modular packages."""
    
    @pytest.fixture
    def service_config(self):
        """Create test service configuration."""
        return ServiceConfig(
            gemini_api_key="test-key",
            qdrant_host="localhost",
            qdrant_port=6333,
            redis_url="redis://localhost:6379/0"
        )
    
    @pytest.fixture
    def morag_api(self, service_config):
        """Create MoRAG API instance."""
        return MoRAGAPI(service_config)
    
    @pytest.fixture
    def morag_orchestrator(self, service_config):
        """Create MoRAG orchestrator instance."""
        return MoRAGOrchestrator(service_config)
    
    def test_package_imports(self):
        """Test that all packages can be imported."""
        # Core package
        from morag_core.interfaces.processor import BaseProcessor
        from morag_core.interfaces.converter import BaseConverter
        from morag_core.models import Document
        
        # Services package
        from morag_services import MoRAGServices
        from morag_services.storage import QdrantVectorStorage
        from morag_services.embedding import GeminiEmbeddingService
        
        # Web package
        from morag_web import WebProcessor, WebConverter
        
        # YouTube package
        from morag_youtube import YouTubeProcessor
        
        # Main package
        from morag import MoRAGAPI, MoRAGOrchestrator
        
        assert True  # If we get here, all imports succeeded
    
    def test_service_config_creation(self, service_config):
        """Test service configuration creation."""
        assert service_config.gemini_api_key == "test-key"
        assert service_config.qdrant_host == "localhost"
        assert service_config.qdrant_port == 6333
    
    def test_morag_api_initialization(self, morag_api):
        """Test MoRAG API initialization."""
        assert morag_api is not None
        assert morag_api.orchestrator is not None
    
    def test_morag_orchestrator_initialization(self, morag_orchestrator):
        """Test MoRAG orchestrator initialization."""
        assert morag_orchestrator is not None
        assert morag_orchestrator.services is not None
        assert morag_orchestrator.web_processor is not None
        assert morag_orchestrator.youtube_processor is not None
    
    def test_content_type_detection(self, morag_api):
        """Test content type auto-detection."""
        # Web URLs
        assert morag_api._detect_content_type("https://example.com") == "web"
        assert morag_api._detect_content_type("http://test.org") == "web"
        
        # YouTube URLs
        assert morag_api._detect_content_type("https://youtube.com/watch?v=123") == "youtube"
        assert morag_api._detect_content_type("https://youtu.be/123") == "youtube"
        
        # File paths
        assert morag_api._detect_content_type_from_file(Path("test.pdf")) == "document"
        assert morag_api._detect_content_type_from_file(Path("test.mp3")) == "audio"
        assert morag_api._detect_content_type_from_file(Path("test.mp4")) == "video"
    
    @pytest.mark.asyncio
    async def test_web_processing_integration(self, morag_orchestrator):
        """Test web processing integration."""
        with patch.object(morag_orchestrator.web_processor, 'process') as mock_process:
            # Mock web processing result
            mock_result = Mock()
            mock_result.content.markdown_content = "# Test Content"
            mock_result.content.content = "Test content"
            mock_result.content.title = "Test Page"
            mock_result.content.metadata = {"source": "web"}
            mock_result.processing_time = 1.0
            mock_result.success = True
            mock_result.error_message = None
            mock_process.return_value = mock_result
            
            result = await morag_orchestrator._process_web_content("https://example.com", {})
            
            assert result.success
            assert result.content == "# Test Content"
            assert result.metadata["source_type"] == "web"
            assert result.metadata["title"] == "Test Page"
    
    @pytest.mark.asyncio
    async def test_youtube_processing_integration(self, morag_orchestrator):
        """Test YouTube processing integration."""
        with patch.object(morag_orchestrator.youtube_processor, 'process') as mock_process:
            # Mock YouTube processing result
            mock_metadata = Mock()
            mock_metadata.id = "test123"
            mock_metadata.title = "Test Video"
            mock_metadata.description = "Test description"
            mock_metadata.uploader = "Test Channel"
            mock_metadata.duration = 120.0
            mock_metadata.view_count = 1000
            
            mock_result = Mock()
            mock_result.metadata = mock_metadata
            mock_result.video_path = None
            mock_result.audio_path = None
            mock_result.subtitle_paths = []
            mock_result.thumbnail_paths = []
            mock_result.processing_time = 2.0
            mock_result.success = True
            mock_process.return_value = mock_result
            
            result = await morag_orchestrator._process_youtube_content("https://youtube.com/watch?v=123", {})
            
            assert result.success
            assert "Test Video" in result.content
            assert result.metadata["source_type"] == "youtube"
            assert result.metadata["video_id"] == "test123"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, morag_orchestrator):
        """Test batch processing functionality."""
        with patch.object(morag_orchestrator, 'process_content') as mock_process:
            # Mock individual processing results
            mock_result1 = ProcessingResult(
                content="Content 1",
                metadata={"source": "item1"},
                processing_time=1.0,
                success=True
            )
            mock_result2 = ProcessingResult(
                content="Content 2", 
                metadata={"source": "item2"},
                processing_time=1.5,
                success=True
            )
            mock_process.side_effect = [mock_result1, mock_result2]
            
            items = [
                {"content": "https://example1.com", "content_type": "web"},
                {"content": "https://example2.com", "content_type": "web"}
            ]
            
            results = await morag_orchestrator.process_batch(items)
            
            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].content == "Content 1"
            assert results[1].content == "Content 2"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, morag_orchestrator):
        """Test error handling in processing."""
        with patch.object(morag_orchestrator.web_processor, 'process') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            with pytest.raises(Exception, match="Test error"):
                await morag_orchestrator._process_web_content("https://example.com", {})
    
    @pytest.mark.asyncio
    async def test_cleanup(self, morag_api):
        """Test cleanup functionality."""
        with patch.object(morag_api.orchestrator, 'cleanup') as mock_cleanup:
            await morag_api.cleanup()
            mock_cleanup.assert_called_once()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Package imports failed: {IMPORT_ERROR}")
class TestPackageInterfaces:
    """Test interfaces between packages."""
    
    def test_web_processor_interface(self):
        """Test web processor implements correct interface."""
        from morag_core.interfaces.processor import BaseProcessor
        from morag_web import WebProcessor
        
        processor = WebProcessor()
        assert isinstance(processor, BaseProcessor)
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'supports_format')
    
    def test_web_converter_interface(self):
        """Test web converter implements correct interface."""
        from morag_core.interfaces.converter import BaseConverter
        from morag_web import WebConverter
        
        converter = WebConverter()
        assert isinstance(converter, BaseConverter)
        assert hasattr(converter, 'convert')
        assert hasattr(converter, 'supports_format')
    
    def test_youtube_processor_interface(self):
        """Test YouTube processor implements correct interface."""
        from morag_core.interfaces.processor import BaseProcessor
        from morag_youtube import YouTubeProcessor
        
        processor = YouTubeProcessor()
        assert isinstance(processor, BaseProcessor)
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'supports_format')
    
    def test_storage_interface(self):
        """Test storage service implements correct interface."""
        from morag_core.interfaces.storage import BaseVectorStorage
        from morag_services.storage import QdrantVectorStorage
        
        storage = QdrantVectorStorage()
        assert isinstance(storage, BaseVectorStorage)
        assert hasattr(storage, 'store_vectors')
        assert hasattr(storage, 'search_similar')
    
    def test_embedding_interface(self):
        """Test embedding service implements correct interface."""
        from morag_core.interfaces.embedding import BaseEmbeddingService
        from morag_services.embedding import GeminiEmbeddingService
        
        service = GeminiEmbeddingService("test-key")
        assert isinstance(service, BaseEmbeddingService)
        assert hasattr(service, 'generate_embedding')


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Package imports failed: {IMPORT_ERROR}")
class TestConfigurationManagement:
    """Test configuration management across packages."""
    
    def test_service_config_serialization(self):
        """Test service configuration can be serialized."""
        from morag_services import ServiceConfig
        
        config = ServiceConfig(
            gemini_api_key="test-key",
            qdrant_host="localhost",
            qdrant_port=6333
        )
        
        # Test dict conversion
        config_dict = config.model_dump()
        assert config_dict["gemini_api_key"] == "test-key"
        assert config_dict["qdrant_host"] == "localhost"
        
        # Test JSON serialization
        config_json = config.model_dump_json()
        assert "test-key" in config_json
        
        # Test reconstruction
        new_config = ServiceConfig.model_validate(config_dict)
        assert new_config.gemini_api_key == config.gemini_api_key
    
    def test_config_file_loading(self):
        """Test loading configuration from file."""
        from morag_services import ServiceConfig
        
        config_data = {
            "gemini_api_key": "file-key",
            "qdrant_host": "remote-host",
            "qdrant_port": 6333,
            "redis_url": "redis://remote:6379/0"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Load from file
            with open(config_file) as f:
                loaded_data = json.load(f)
            
            config = ServiceConfig.model_validate(loaded_data)
            assert config.gemini_api_key == "file-key"
            assert config.qdrant_host == "remote-host"
            
        finally:
            Path(config_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
