from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.core.logging import get_logger
from app.core.cache import get_cache, CacheKeys, generate_hash
import structlog
import json

from app.ml.services.gemini_feedback import GeminiFeedbackService

logger = get_logger()
router = APIRouter()
feedback_service = GeminiFeedbackService()

class ErrorDetail(BaseModel):
    type: str
    expected: str
    actual: str
    position: int
    confidence: float

class PronunciationData(BaseModel):
    overall_score: float
    dimensions: dict
    errors: List[ErrorDetail]
    cefr_level: Optional[str] = None

class FeedbackRequest(BaseModel):
    pronunciation_data: PronunciationData
    user_id: str
    language: str  # id-ID or en-US
    session_history: Optional[List[dict]] = None

class FeedbackResponse(BaseModel):
    ai_feedback: str
    specific_errors: List[ErrorDetail]
    improvement_suggestions: List[str]
    next_steps: List[str]
    cache_hit: bool = False  # Redis cache indicator

@router.post("/feedback", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    """
    Generate personalized pronunciation feedback.
    
    This endpoint will be fully implemented in Sprint 3 with:
    - GPT-4 integration for contextual feedback
    - Error pattern analysis
    - Personalized improvement suggestions
    - Bilingual feedback generation
    
    For now, returns placeholder data.
    """
    
    try:
        # Generate cache key based on feedback data
        feedback_data_str = json.dumps(request.pronunciation_data.dict(), sort_keys=True)
        feedback_hash = generate_hash(feedback_data_str)
        cache_key = CacheKeys.feedback_key(request.user_id, feedback_hash)
        
        # Initialize cache
        cache = get_cache()
        
        # Try to get from cache first
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                "Cache HIT for feedback",
                user_id=request.user_id,
                cache_key=cache_key
            )
            cached_result["cache_hit"] = True
            return FeedbackResponse(**cached_result)
        
        logger.info(
            "Cache MISS for feedback",
            user_id=request.user_id,
            cache_key=cache_key
        )
        
        logger.info(
            "Generating personalized feedback",
            user_id=request.user_id,
            language=request.language,
            score=request.pronunciation_data.overall_score
        )
        
        default_feedback = {
            "overall_assessment": "Bagus! Anda sudah bisa mengucapkan kata dengan cukup jelas. Fokus pada huruf 't' yang terkadang terdengar seperti 'd'.",
            "practice_suggestions": [
                "Latih pengucapan huruf 't' dengan meletakkan ujung lidah di belakang gigi atas",
                "Praktik dengan kata-kata yang mengandung huruf 't' seperti 'terima kasih'",
                "Rekam suara Anda dan bandingkan dengan native speaker"
            ],
            "next_steps": [
                "Lanjut ke pelajaran berikutnya tentang konsonan",
                "Lakukan latihan pengucapan harian selama 15 menit",
                "Coba praktik dengan kalimat yang lebih panjang"
            ]
        }

        comprehensive_payload: Dict[str, Any] = {
            "overall_score": request.pronunciation_data.overall_score,
            "dimensions": request.pronunciation_data.dimensions,
            "errors": [error.dict() for error in request.pronunciation_data.errors],
            "features": {},
            "feedback": {}
        }

        feedback_details = await feedback_service.generate_feedback(
            comprehensive_payload,
            request.language,
            {
                "user_id": request.user_id,
                "session_history": request.session_history
            }
        )

        final_feedback = feedback_details or default_feedback

        placeholder_feedback = FeedbackResponse(
            ai_feedback=final_feedback.get("overall_assessment", default_feedback["overall_assessment"]),
            specific_errors=request.pronunciation_data.errors,
            improvement_suggestions=final_feedback.get("practice_suggestions", default_feedback["practice_suggestions"]),
            next_steps=final_feedback.get("next_steps", default_feedback["next_steps"])
        )
        
        # Cache the result (7 days TTL)
        try:
            cache.set(
                cache_key,
                placeholder_feedback.dict(),
                ttl_seconds=CacheKeys.FEEDBACK_TTL
            )
            logger.info("Feedback result cached", cache_key=cache_key)
        except Exception as cache_error:
            logger.warning("Failed to cache feedback result", error=str(cache_error))
        
        logger.info(
            "Feedback generation completed",
            user_id=request.user_id,
            feedback_length=len(placeholder_feedback.ai_feedback)
        )
        
        return placeholder_feedback
        
    except Exception as e:
        logger.error(
            "Error during feedback generation",
            user_id=request.user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during feedback generation"
        )
