from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
from app.core.logging import get_logger
from app.core.cache import get_cache, CacheKeys, generate_hash
from app.ml.services.stt_service import STTService
import aiofiles
import os
import tempfile
from datetime import datetime
import structlog
import hashlib

logger = get_logger()
router = APIRouter()

# Initialize STT service
stt_service = STTService()

class TranscriptionRequest(BaseModel):
    language: str  # id-ID or en-US

class TranscriptionResponse(BaseModel):
    transcript: str
    confidence: float
    language: str
    word_timestamps: Optional[dict] = None
    cache_hit: bool = False  # Redis cache indicator

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    language: str = Body(...),
    audio_file: UploadFile = File(...)
):
    """
    Transcribe audio file to text.
    
    This endpoint will be fully implemented in Sprint 1 with:
    - Google Cloud STT integration
    - Whisper fallback
    - Word-level timing extraction
    - Language detection
    
    For now, returns placeholder data.
    """
    
    # Validate file type - accept any file, will check by extension later if needed
    # Bypass strict content-type check since different clients send different types
    
    # Validate language
    if language not in ["id-ID", "en-US"]:
        raise HTTPException(
            status_code=400,
            detail="Language must be 'id-ID' or 'en-US'"
        )
    
    try:
        # Read audio content for hashing and cache key generation
        try:
            audio_content = await audio_file.read()
        except Exception as read_error:
            logger.error(f"Error reading audio file: {read_error}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(read_error)}")
        
        # Generate cache key based on audio hash and language
        try:
            audio_hash = hashlib.sha256(audio_content).hexdigest()[:16]
            cache_key = CacheKeys.transcription_key(audio_hash, language)
        except Exception as hash_error:
            logger.error(f"Error generating hash: {hash_error}")
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(hash_error)}")
        
        # Initialize cache
        try:
            cache = get_cache()
        except Exception as cache_error:
            logger.error(f"Error initializing cache: {cache_error}")
            cache = None
        
        # Try to get from cache first
        cached_result = None
        if cache:
            try:
                cached_result = cache.get(cache_key)
            except Exception as cache_get_error:
                logger.warning(f"Error getting from cache: {cache_get_error}")
                cached_result = None
        
        if cached_result is not None:
            logger.info(
                "Cache HIT for transcription",
                language=language,
                cache_key=cache_key
            )
            cached_result["cache_hit"] = True
            return TranscriptionResponse(**cached_result)
        
        logger.info(
            "Cache MISS for transcription",
            language=language,
            cache_key=cache_key
        )
        
        # Save uploaded audio temporarily
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"transcription_{timestamp}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        async with aiofiles.open(audio_path, 'wb') as f:
            await f.write(audio_content)
        
        logger.info(f"Audio saved temporarily to {audio_path}")
        
        # Transcribe using STT service
        try:
            transcription_result = await stt_service.transcribe_audio(
                audio_path=audio_path,
                language=language,
                use_word_timestamps=True
            )
        except Exception as stt_error:
            logger.warning(f"STT service error, using placeholder: {stt_error}")
            # Fallback to placeholder result when STT fails
            transcription_result = {
                "transcript": "[transcription unavailable]",
                "confidence": 0.0,
                "language": language,
                "word_timestamps": None,
                "engine": "placeholder"
            }
        
        # Convert word_timestamps from list to dict if needed
        word_timestamps = transcription_result.get("word_timestamps")
        if isinstance(word_timestamps, list):
            # Convert list of word timing objects to dict format
            word_timestamps_dict = {}
            for i, word_ts in enumerate(word_timestamps):
                if isinstance(word_ts, dict):
                    word_timestamps_dict[word_ts.get('word', f'word_{i}')] = {
                        'start': word_ts.get('start', 0),
                        'end': word_ts.get('end', 0),
                        'confidence': word_ts.get('confidence', 0)
                    }
            word_timestamps = word_timestamps_dict if word_timestamps_dict else None
        
        # Convert to response format
        response = TranscriptionResponse(
            transcript=transcription_result["transcript"],
            confidence=transcription_result["confidence"],
            language=transcription_result["language"],
            word_timestamps=word_timestamps
        )
        
        # Cache the result (24 hours TTL)
        try:
            cache.set(
                cache_key,
                response.dict(),
                ttl_seconds=CacheKeys.TRANSCRIPTION_TTL
            )
            logger.info("Transcription result cached", cache_key=cache_key)
        except Exception as cache_error:
            logger.warning("Failed to cache transcription result", error=str(cache_error))
        
        # Cleanup temporary file
        try:
            os.remove(audio_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
        
        logger.info(
            "Audio transcription completed",
            language=language,
            confidence=response.confidence,
            transcript_length=len(response.transcript),
            engine=transcription_result.get("engine", "unknown")
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Error during audio transcription",
            language=language,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during transcription"
        )
