from google.cloud import speech
import whisper
import librosa
import torch
from typing import Optional, Dict, List
from pathlib import Path
import io
import structlog
from app.core.config import settings
import asyncio

logger = structlog.get_logger()

class STTService:
    """Speech-to-Text service with Google Cloud STT and Whisper fallback."""
    
    def __init__(self):
        self.speech_client = None
        self.whisper_model = None
        self._setup_services()
    
    def _setup_services(self):
        """Initialize Google Cloud STT and Whisper models."""
        try:
            # Initialize Google Cloud STT
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                import os
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.GOOGLE_APPLICATION_CREDENTIALS
                self.speech_client = speech.SpeechClient()
                logger.info("Google Cloud STT client initialized")
            else:
                logger.warning("Google Cloud credentials not provided, using Whisper only")
                
        except Exception as e:
            logger.error("Failed to initialize Google Cloud STT", error=str(e))
        
        try:
            # Initialize Whisper model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Whisper model on {device}")
            self.whisper_model = whisper.load_model(
                settings.WHISPER_MODEL_SIZE, 
                device=device
            )
            logger.info(f"Whisper model '{settings.WHISPER_MODEL_SIZE}' loaded successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Whisper", error=str(e))
            raise
    
    async def transcribe_audio(
        self, 
        audio_path: str, 
        language: str,
        use_word_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (id-ID or en-US)
            use_word_timestamps: Whether to extract word-level timing
            
        Returns:
            Dictionary containing transcript, confidence, and word timestamps
        """
        try:
            # Try Google Cloud STT first
            if self.speech_client:
                result = await self._transcribe_with_google(
                    audio_path, language, use_word_timestamps
                )
                if result:
                    logger.info("Transcription completed with Google Cloud STT")
                    return result
            
            # Fallback to Whisper
            logger.info("Falling back to Whisper for transcription")
            return await self._transcribe_with_whisper(
                audio_path, language, use_word_timestamps
            )
            
        except Exception as e:
            logger.error(
                "Transcription failed", 
                audio_path=audio_path, 
                language=language, 
                error=str(e)
            )
            raise
    
    async def _transcribe_with_google(
        self, 
        audio_path: str, 
        language: str,
        use_word_timestamps: bool
    ) -> Optional[Dict]:
        """Transcribe using Google Cloud STT."""
        try:
            # Read audio file
            with io.open(audio_path, "rb") as audio_file:
                content = audio_file.read()
            
            # Configure recognition
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                language_code=language,
                enable_word_time_offsets=use_word_timestamps,
                enable_automatic_punctuation=True,
                model="latest_short",
                audio_channel_count=1,
                sample_rate_hertz=16000
            )
            
            # Perform recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Parse response
            transcript = ""
            word_timestamps = []
            confidence_sum = 0
            word_count = 0
            
            for result in response.results:
                transcript += result.alternatives[0].transcript
                
                if use_word_timestamps and result.alternatives[0].words:
                    for word_info in result.alternatives[0].words:
                        word_timestamps.append({
                            "word": word_info.word,
                            "start_time": word_info.start_time.total_seconds(),
                            "end_time": word_info.end_time.total_seconds(),
                            "confidence": word_info.confidence if hasattr(word_info, 'confidence') else 0.95
                        })
                        confidence_sum += word_info.confidence if hasattr(word_info, 'confidence') else 0.95
                        word_count += 1
            
            # Calculate average confidence
            avg_confidence = confidence_sum / word_count if word_count > 0 else 0.0
            
            return {
                "transcript": transcript.strip(),
                "confidence": avg_confidence,
                "language": language,
                "word_timestamps": word_timestamps if use_word_timestamps else None,
                "engine": "google_cloud_stt"
            }
            
        except Exception as e:
            logger.error("Google Cloud STT failed", error=str(e))
            return None
    
    async def _transcribe_with_whisper(
        self, 
        audio_path: str, 
        language: str,
        use_word_timestamps: bool
    ) -> Dict:
        """Transcribe using Whisper model."""
        try:
            # Map language codes to Whisper format
            lang_map = {
                "id-ID": "id",
                "en-US": "en"
            }
            whisper_lang = lang_map.get(language, "en")
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.whisper_model.transcribe,
                audio_path,
                whisper_lang,
                False  # without timestamps initially
            )
            
            transcript = result["text"].strip()
            
            # Extract word timestamps if needed
            word_timestamps = None
            if use_word_timestamps:
                # Run transcription with timestamps
                result_with_timestamps = await loop.run_in_executor(
                    None,
                    self.whisper_model.transcribe,
                    audio_path,
                    whisper_lang,
                    True  # with timestamps
                )
                
                word_timestamps = []
                for segment in result_with_timestamps.get("segments", []):
                    for word, timing in segment.get("words", []):
                        word_timestamps.append({
                            "word": word,
                            "start_time": timing["start"],
                            "end_time": timing["end"], 
                            "confidence": timing.get("probability", 0.95)
                        })
            
            # Whisper doesn't provide overall confidence, use average of word confidences
            avg_confidence = 0.95  # Default confidence for Whisper
            
            return {
                "transcript": transcript,
                "confidence": avg_confidence,
                "language": language,
                "word_timestamps": word_timestamps if use_word_timestamps else None,
                "engine": "whisper"
            }
            
        except Exception as e:
            logger.error("Whisper transcription failed", error=str(e))
            raise
    
    async def transcribe_real_time(
        self, 
        audio_stream, 
        language: str
    ) -> str:
        """
        Real-time transcription from audio stream.
        This will be implemented in later sprints.
        """
        # Placeholder for real-time transcription
        logger.info("Real-time transcription not yet implemented")
        return "Real-time transcription coming soon"
