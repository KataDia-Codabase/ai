from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
from app.core.logging import get_logger
from app.ml.services.stt_service import STTService
import aiofiles
import os
import tempfile
from datetime import datetime
import structlog

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
    
    # Validate file type
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an audio file"
        )
    
    # Validate language
    if language not in ["id-ID", "en-US"]:
        raise HTTPException(
            status_code=400,
            detail="Language must be 'id-ID' or 'en-US'"
        )
    
    try:
        # Save uploaded audio temporarily
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"transcription_{timestamp}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        async with aiofiles.open(audio_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)
        
        logger.info(f"Audio saved temporarily to {audio_path}")
        
        # Transcribe using STT service
        transcription_result = await stt_service.transcribe_audio(
            audio_path=audio_path,
            language=language,
            use_word_timestamps=True
        )
        
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
