from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.core.logging import get_logger
from app.ml.services.stt_service import STTService
from app.ml.services.phoneme_service import PhonemeService
from app.ml.services.scoring_service import ScoringService, ScoringResult
from app.ml.services.enhanced_scoring import EnhancedScoringService
from app.ml.services.gemini_feedback import GeminiFeedbackService
from app.ml.services.cefr_assessment import EnglishCEFRAssessment, CEFRAssessmentResult
from app.core.config import settings
import structlog
import aiofiles
import os
import tempfile
from datetime import datetime

logger = get_logger()
router = APIRouter()

# Initialize ML services
stt_service = STTService()
phoneme_service = PhonemeService()
scoring_service = ScoringService()
enhanced_scoring = EnhancedScoringService()
gemini_feedback = GeminiFeedbackService()
cefr_assessment = EnglishCEFRAssessment()

class ScoreRequest(BaseModel):
    transcript: str
    language: str  # id-ID or en-US
    user_id: str
    session_id: str
    audio_url: Optional[str] = None  # Provided by mobile backend
    lesson_vocab_id: Optional[int] = None

class ErrorDetail(BaseModel):
    type: str  # substitution, deletion, insertion
    expected: str
    actual: str
    position: int
    confidence: float
    phoneme_index: Optional[int] = None

class ScoreResponse(BaseModel):
    overall_score: float
    dimensions: Dict[str, float]  # accuracy, fluency, prosody, stress
    errors: List[ErrorDetail]
    cefr_level: Optional[str]
    feedback: Optional[str]
    processing_time: Optional[float] = None
    analysis_level: Optional[str] = None  # basic, comprehensive
    features: Optional[Dict] = None  # Detailed feature analysis
    generated_transcript: Optional[str] = None
    error_summary: Optional[Dict[str, Any]] = None
    phoneme_insights: Optional[Dict[str, Any]] = None
    audio_url: Optional[str] = None
    lesson_vocab_id: Optional[int] = None
    user_id: Optional[str] = None

@router.post("/score", response_model=ScoreResponse)
async def score_pronunciation(
    transcript: str = Body(...),
    language: str = Body(...),
    user_id: str = Body(...),
    session_id: str = Body(...),
    audio_url: Optional[str] = Body(None),
    lesson_vocab_id: Optional[int] = Body(None),
    audio_file: UploadFile = File(...)
):
    """
    Score pronunciation from uploaded audio file.
    
    This endpoint will be fully implemented in Sprint 1-2 with:
    - Audio processing and transcription
    - Phoneme extraction and alignment
    - Multi-dimensional scoring
    - Error detection
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
        # Start timing
        start_time = datetime.now()
        
        # Log the request
        logger.info(
            "Pronunciation scoring request",
            user_id=user_id,
            session_id=session_id,
            language=language,
            transcript=transcript[:50] if transcript else ""
        )
        
        # Save uploaded audio temporarily
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"pronunciation_{user_id}_{timestamp}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        async with aiofiles.open(audio_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)
        
        logger.info(f"Audio saved temporarily to {audio_path}")
        
        # Step 1: Transcribe audio (for verification)
        transcription_result = await stt_service.transcribe_audio(
            audio_path, language
        )
        
        # Step 2: Extract phonemes using alignment
        alignment_result = await phoneme_service.extract_phonemes(
            audio_path, transcript, language
        )
        
        # Step 3: Calculate pronunciation score using enhanced scoring when available
        use_enhanced = enhanced_scoring.supports_language(language)
        comprehensive_result = None
        gemini_feedback_result = None
        cefr_result = None

        if use_enhanced:
            comprehensive_result = await enhanced_scoring.calculate_comprehensive_score(
                audio_path=audio_path,
                transcript=transcript,
                language=language,
                alignment=alignment_result
            )

            scoring_result = ScoringResult(
                overall_score=comprehensive_result["overall_score"],
                dimensions=comprehensive_result["dimensions"],
                errors=comprehensive_result.get("errors", []),
                confidence=comprehensive_result.get("confidence", 0.0),
                details={
                    "features": comprehensive_result.get("features"),
                    "feedback": comprehensive_result.get("feedback"),
                    "recognized_transcript": comprehensive_result.get("recognized_transcript"),
                    "audio_profile": comprehensive_result.get("audio_profile"),
                    "error_summary": comprehensive_result.get("error_summary"),
                    "phoneme_insights": comprehensive_result.get("phoneme_insights"),
                },
            )

            if language == "en-US":
                gemini_feedback_result = await gemini_feedback.generate_feedback(
                    comprehensive_result=comprehensive_result,
                    language=language
                )
                cefr_result = await cefr_assessment.assess_cefr_level(
                    comprehensive_score=comprehensive_result
                )
        else:
            expected_phonemes = [p.phoneme for p in alignment_result.phonemes]
            actual_phonemes = [p.phoneme for p in alignment_result.phonemes]

            scoring_result = await scoring_service.calculate_gop_score(
                expected_phonemes, actual_phonemes
            )
        
        # Step 4: Convert to response format
        response_errors = []
        for error in scoring_result.errors:
            response_errors.append(ErrorDetail(
                type=error.type,
                expected=error.expected,
                actual=error.actual,
                position=error.position,
                confidence=error.confidence,
                phoneme_index=getattr(error, 'phoneme_index', None)
            ))
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Determine response format based on enhanced scoring
        if comprehensive_result:
            # Extract CEFR level if available
            cefr_level = cefr_result.cefr_level.value if cefr_result else None
            base_features = comprehensive_result.get('features') or {}
            extended_features = {
                **base_features,
                'audio_profile': comprehensive_result.get('audio_profile'),
                'error_summary': comprehensive_result.get('error_summary'),
                'recognized_transcript': comprehensive_result.get('recognized_transcript'),
                'phoneme_insights': comprehensive_result.get('phoneme_insights')
            }
            if cefr_result:
                extended_features['cefr_assessment'] = {
                    'level': cefr_level,
                    'confidence': cefr_result.confidence,
                    'strengths': cefr_result.strengths,
                    'weaknesses': cefr_result.weaknesses
                }
            
            result_response = ScoreResponse(
                overall_score=scoring_result.overall_score,
                dimensions=scoring_result.dimensions,
                errors=response_errors,
                cefr_level=cefr_level,
                feedback=gemini_feedback_result.get('feedback') if gemini_feedback_result else None,
                processing_time=processing_time,
                analysis_level=comprehensive_result.get('analysis_level'),
                features=extended_features,
                generated_transcript=comprehensive_result.get('recognized_transcript'),
                error_summary=comprehensive_result.get('error_summary'),
                phoneme_insights=comprehensive_result.get('phoneme_insights'),
                audio_url=audio_url,
                lesson_vocab_id=lesson_vocab_id,
                user_id=user_id
            )
        else:
            result_response = ScoreResponse(
                overall_score=scoring_result.overall_score,
                dimensions=scoring_result.dimensions,
                errors=response_errors,
                cefr_level=None,
                feedback=None,
                processing_time=processing_time,
                analysis_level="basic",
                features=None,
                generated_transcript=transcript,
                error_summary=None,
                phoneme_insights=None,
                audio_url=audio_url,
                lesson_vocab_id=lesson_vocab_id,
                user_id=user_id
            )
        
        # Step 5: Cleanup temporary file
        try:
            os.remove(audio_path)
            phoneme_service.cleanup_temp_files(audio_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
        
        logger.info(
            "Pronunciation scoring completed",
            user_id=user_id,
            session_id=session_id,
            score=result_response.overall_score,
            processing_time=processing_time,
            engine_used=transcription_result.get("engine", "unknown")
        )
        
        return result_response
        
    except Exception as e:
        logger.error(
            "Error during pronunciation scoring",
            user_id=user_id,
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during scoring"
        )
