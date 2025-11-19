from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import aiofiles

from app.core.logging import get_logger
from app.ml.services.stt_service import STTService
from app.ml.services.phoneme_service import PhonemeService
from app.ml.services.scoring_service import ScoringResult
from app.ml.services.enhanced_scoring import EnhancedScoringService
from app.ml.services.cefr_assessment import EnglishCEFRAssessment
from app.ml.services.gemini_feedback import GeminiFeedbackService
BASE_DIR = Path(__file__).resolve().parents[3]

logger = get_logger()
router = APIRouter()

# Initialize ML services
stt_service = STTService()
phoneme_service = PhonemeService()
enhanced_scoring = EnhancedScoringService()
cefr_assessment = EnglishCEFRAssessment()
feedback_service = GeminiFeedbackService()

UPLOAD_ROOT = BASE_DIR / "temp_audio" / "scoring_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
class ScoreRequest(BaseModel):
    transcript: str
    language: str  # currently only en-US supported
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
    language_code: str
    overall_score: float
    accuracy_score: Optional[float] = None
    fluency_score: Optional[float] = None
    prosody_score: Optional[float] = None
    stress_score: Optional[float] = None
    dimensions: Dict[str, float]  # accuracy, fluency, prosody, stress
    errors: List[ErrorDetail]
    phoneme_errors_json: Optional[List[Dict[str, Any]]] = None
    cefr_level: Optional[str]
    cefr_level_assessment: Optional[str] = None
    personalized_feedback: Optional[str] = None
    feedback_details: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    analysis_level: Optional[str] = None  # basic, comprehensive
    features: Optional[Dict] = None  # Detailed feature analysis
    generated_transcript: Optional[str] = None
    error_summary: Optional[Dict[str, Any]] = None
    phoneme_insights: Optional[Dict[str, Any]] = None
    audio_url: Optional[str] = None
    lesson_vocab_id: Optional[int] = None
    user_id: Optional[str] = None
    created_at: datetime

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
    
    # Simply accept any uploaded file - validation happens during processing
    # No strict validation needed here
    
    # Validate language (enhanced scoring focus)
    if language != "en-US":
        raise HTTPException(
            status_code=400,
            detail="Enhanced scoring currently supports only English (en-US)."
        )
    if not enhanced_scoring.supports_language(language):
        raise HTTPException(
            status_code=400,
            detail=f"Enhanced scoring model not available for {language}."
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
        
        # Save uploaded audio for reproducible scoring
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"pronunciation_{user_id}_{timestamp}.wav"
        audio_path = UPLOAD_ROOT / audio_filename
        
        async with aiofiles.open(audio_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)
        
        logger.info(
            "Audio stored for scoring",
            path=str(audio_path),
            size_bytes=len(content)
        )

        # Use provided audio_url if present, otherwise reference the stored local path
        audio_reference = audio_url or f"file://{audio_path.as_posix()}"
        
        # Step 1: Transcribe audio (for verification)
        transcription_result = await stt_service.transcribe_audio(
            audio_path, language
        )
        
        # Step 2: Extract phonemes using alignment
        alignment_result = await phoneme_service.extract_phonemes(
            audio_path, transcript, language
        )
        
        # Step 3: Always use enhanced scoring for English
        comprehensive_result = await enhanced_scoring.calculate_comprehensive_score(
            audio_path=str(audio_path),
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

        cefr_result = await cefr_assessment.assess_cefr_level(
            comprehensive_score=comprehensive_result
        )

        # Step 3b: Generate personalized feedback for DB consumers
        feedback_details = await feedback_service.generate_feedback(
            comprehensive_result,
            language,
            {
                "user_id": user_id,
                "session_id": session_id,
                "lesson_vocab_id": lesson_vocab_id
            }
        ) if language == "en-US" else None
        personalized_feedback = None
        if feedback_details:
            personalized_feedback = feedback_details.get("overall_assessment") or \
                feedback_details.get("ai_feedback")
        
        # Step 4: Convert to response format
        response_errors = []
        phoneme_errors_json = []
        for error in scoring_result.errors:
            response_errors.append(ErrorDetail(
                type=error.type,
                expected=error.expected,
                actual=error.actual,
                position=error.position,
                confidence=error.confidence,
                phoneme_index=getattr(error, 'phoneme_index', None)
            ))
            phoneme_errors_json.append(asdict(error))
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Determine response format based on enhanced scoring
        # Build enriched response
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

        dimensions = scoring_result.dimensions or {}
        created_at = datetime.now(timezone.utc)

        result_response = ScoreResponse(
            language_code=language,
            overall_score=scoring_result.overall_score,
            accuracy_score=dimensions.get('accuracy'),
            fluency_score=dimensions.get('fluency'),
            prosody_score=dimensions.get('prosody'),
            stress_score=dimensions.get('stress'),
            dimensions=scoring_result.dimensions,
            errors=response_errors,
            phoneme_errors_json=phoneme_errors_json,
            cefr_level=cefr_level,
            cefr_level_assessment=cefr_level,
            personalized_feedback=personalized_feedback,
            feedback_details=feedback_details,
            processing_time=processing_time,
            analysis_level=comprehensive_result.get('analysis_level'),
            features=extended_features,
            generated_transcript=comprehensive_result.get('recognized_transcript'),
            error_summary=comprehensive_result.get('error_summary'),
            phoneme_insights=comprehensive_result.get('phoneme_insights'),
            audio_url=audio_reference,
            lesson_vocab_id=lesson_vocab_id,
            user_id=user_id,
            created_at=created_at
        )
        
        # Step 5: Cleanup auxiliary temp files (keep uploaded audio for traceability)
        try:
            phoneme_service.cleanup_temp_files(str(audio_path))
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary alignment file: {cleanup_error}")
        
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
