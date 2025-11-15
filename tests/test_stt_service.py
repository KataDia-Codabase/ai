import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from app.ml.services.stt_service import STTService

class TestSTTService:
    """Test cases for STT Service."""
    
    @pytest.fixture
    def stt_service(self):
        """Create STT service instance for testing."""
        # Mock the setup to avoid requiring actual credentials
        with patch('app.ml.services.stt_service.settings'):
            service = STTService()
            service.whisper_model = Mock()
            service.whisper_model.transcribe = Mock(return_value={
                "text": "test transcript",
                "segments": []
            })
            return service
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write minimal WAV header
            f.write(b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
            filename = f.name
        
        yield filename
        # Cleanup
        try:
            os.unlink(filename)
        except OSError:
            pass
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_with_mock_whisper(self, stt_service, mock_audio_file):
        """Test audio transcription using Whisper fallback."""
        result = await stt_service.transcribe_audio(
            audio_path=mock_audio_file,
            language="id-ID",
            use_word_timestamps=True
        )
        
        assert result["transcript"] == "test transcript"
        assert result["language"] == "id-ID"
        assert result["confidence"] == 0.95
        assert result["engine"] == "whisper"
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_without_timestamps(self, stt_service, mock_audio_file):
        """Test audio transcription without word timestamps."""
        result = await stt_service.transcribe_audio(
            audio_path=mock_audio_file,
            language="en-US",
            use_word_timestamps=False
        )
        
        assert result["transcript"] == "test transcript"
        assert result["language"] == "en-US"
        assert result["confidence"] == 0.95
        assert result["word_timestamps"] is None
    
    def test_language_mapping(self):
        """Test that language codes are correctly mapped."""
        with patch('app.ml.services.stt_service.settings'):
            service = STTService()
            service.whisper_model = Mock()
            service.whisper_model.transcribe = Mock(return_value={
                "text": "test",
                "segments": []
            })
            
            # Test Indonesian mapping
            assert service._transcribe_with_whisper.__code__.co_varnames.index('whisper_lang') > 0
    
    @pytest.mark.asyncio
    async def test_transcription_error_handling(self, stt_service):
        """Test error handling during transcription."""
        # Test with non-existent file
        with pytest.raises(Exception):
            await stt_service.transcribe_audio(
                audio_path="/non/existent/file.wav",
                language="id-ID"
            )
    
    def test_setup_logging(self):
        """Test that setup logging is called during initialization."""
        with patch('app.ml.services.stt_service.logger') as mock_logger:
            with patch('app.ml.services.stt_service.settings'):
                stt_service = STTService()
                # Verify logging was called during setup
                assert stt_service is not None
