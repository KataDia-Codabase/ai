"""
Analytics dan Adaptive Learning endpoints untuk tracking progress dan rekomendasi latihan.
"""

from fastapi import APIRouter, Body, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import structlog

from app.ml.services.analytics import PronunciationAnalyticsService, ProgressAnalytics
from app.ml.services.adaptive_learning import AdaptiveLearningService, PracticeRecommendation, DifficultyLevel

logger = structlog.get_logger()
router = APIRouter()

# Initialize services
analytics_service = PronunciationAnalyticsService()
adaptive_learning_service = AdaptiveLearningService()


class ProgressAnalyticsRequest(BaseModel):
    """Request untuk analytics endpoint."""
    user_id: str
    language: str = "en-US"


class ProgressAnalyticsResponse(BaseModel):
    """Response dari analytics endpoint."""
    user_id: str
    total_attempts: int
    average_score: float
    improvement_rate: float
    consistency_score: float  # 0-100, higher = more consistent
    learning_velocity: float  # points per hour
    retention_rate: float  # 0-1
    strong_areas: List[str]
    weak_areas: List[str]
    time_spent_hours: float
    last_update: datetime


class NativeSpeakerComparisonResponse(BaseModel):
    """Response dari native speaker comparison."""
    user_score: float
    native_mean: float
    difference: float
    percentile: float  # 0-100
    z_score: float
    insight: str


class PracticeRecommendationResponse(BaseModel):
    """Response dari practice recommendation endpoint."""
    difficulty_level: str
    focus_areas: List[str]
    practice_items: List[Dict[str, Any]]
    session_duration_minutes: int
    next_review_date: datetime
    rationale: str


@router.post("/analytics", response_model=ProgressAnalyticsResponse)
async def get_pronunciation_analytics(request: ProgressAnalyticsRequest):
    """
    Dapatkan analytics komprehensif untuk progress pembelajaran pronunciation.
    
    **Fitur:**
    - Total attempts dan average score
    - Improvement rate dengan analisis trend
    - Consistency score untuk stabilitas performa
    - Learning velocity (kecepatan belajar per jam)
    - Retention rate untuk durabilitas skill
    - Identifikasi strong dan weak areas
    - Total time spent practicing
    
    **Menggunakan:**
    - Linear regression untuk menghitung improvement trend
    - Standard deviation untuk consistency analysis
    - Time-based metrics untuk learning velocity
    - Session grouping untuk retention analysis
    
    **Response:**
    - Semua metrics dalam satu response komprehensif
    - Berguna untuk dashboard dan progress tracking
    """
    try:
        # In real implementation, would fetch from database
        # For now, return default analytics
        analytics = ProgressAnalytics(
            user_id=request.user_id,
            total_attempts=0,
            average_score=0,
            improvement_rate=0,
            consistency_score=50,
            learning_velocity=0,
            retention_rate=0.5,
            strong_areas=[],
            weak_areas=["general_pronunciation"],
            time_spent_hours=0,
            last_update=datetime.now()
        )
        
        logger.info(
            "Retrieved analytics",
            user_id=request.user_id,
            total_attempts=analytics.total_attempts
        )
        
        return ProgressAnalyticsResponse(**analytics.__dict__)
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail="Analytics generation failed")


@router.post("/analytics/native-comparison")
async def compare_with_native_speakers(
    user_id: str = Body(...),
    dimensions: Dict[str, float] = Body(...)
):
    """
    Bandingkan pronunciation user dengan standar native speaker.
    
    **Input:**
    - `dimensions`: Dict berisi skor per dimensi
      - accuracy, fluency, prosody, stress (0-100)
    
    **Analisis:**
    - Perbandingan dengan benchmark native speaker
    - Perhitungan z-score untuk setiap dimensi
    - Percentile ranking (0-100)
    - Insights tentang gap dengan native standard
    
    **Response:**
    - User score vs native mean
    - Percentile ranking
    - Actionable insights per dimensi
    - Overall similarity score
    """
    try:
        comparison = await analytics_service.native_comparator.compare_with_native(dimensions)
        
        logger.info(
            "Completed native speaker comparison",
            user_id=user_id,
            overall_similarity=comparison.get("overall_native_similarity")
        )
        
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to compare with native speakers: {e}")
        raise HTTPException(status_code=500, detail="Comparison failed")


@router.post("/practice/recommendations", response_model=PracticeRecommendationResponse)
async def get_practice_recommendations(
    user_id: str = Body(...),
    current_level: str = Body(...),
    recent_performance: List[Dict] = Body(...)
):
    """
    Dapatkan rekomendasi latihan yang dipersonalisasi berdasarkan performa.
    
    **Adaptive Learning Features:**
    - Analisis performance trend dari recent attempts
    - Adjustment kesulitan berdasarkan performa (± 25%)
    - Identifikasi weak areas untuk targeted practice
    - Spaced repetition scheduling untuk optimal review timing
    - Rekomendasi durasi sesi yang optimal
    
    **Input:**
    - `recent_performance`: List of recent scoring results
    - `current_level`: CEFR level (A1-C2)
    - `user_id`: Untuk tracking
    
    **Logika:**
    1. Jika accuracy > 85% dan improving → tingkatkan difficulty
    2. Jika accuracy < 65% dan declining → turunkan difficulty
    3. Identifikasi error patterns untuk focus areas
    4. Pilih practice items yang relevan
    5. Hitung optimal session duration
    
    **Response:**
    - Difficulty level berikutnya
    - Focus areas untuk latihan
    - Practice items yang specific
    - Recommended session duration
    - Next review date dengan spaced repetition
    - Rationale untuk rekomendasi
    """
    try:
        recommendation = await adaptive_learning_service.recommend_next_practice(
            user_id=user_id,
            current_level=current_level,
            recent_performance=recent_performance
        )
        
        logger.info(
            "Generated practice recommendation",
            user_id=user_id,
            difficulty=recommendation.difficulty_level,
            focus_areas=recommendation.focus_areas
        )
        
        return PracticeRecommendationResponse(
            difficulty_level=recommendation.difficulty_level.value,
            focus_areas=recommendation.focus_areas,
            practice_items=recommendation.practice_items,
            session_duration_minutes=recommendation.session_duration_minutes,
            next_review_date=recommendation.next_review_date,
            rationale=recommendation.rationale
        )
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(status_code=500, detail="Recommendation generation failed")


@router.post("/practice/schedule-review")
async def schedule_next_review(
    item_id: str = Body(...),
    quality_score: float = Body(...)  # 0-5 scale
):
    """
    Schedule next review menggunakan SM-2 spaced repetition algorithm.
    
    **SM-2 Algorithm:**
    - Mempertimbangkan quality of response (0-5)
    - Ease factor yang dinamis
    - Interval yang meningkat exponentially
    - Optimal untuk retention dan recall
    
    **Quality Scale:**
    - 5: Sempurna, recall tanpa kesulitan
    - 4: Correct response dengan hesitation
    - 3: Correct response dengan serious effort
    - 2: Incorrect response, correct info teringat sebelumnya
    - 1: Completely incorrect or forgotten
    - 0: Blank, completely forgotten
    
    **Response:**
    - Next review datetime
    - Interval (days)
    - Updated ease factor
    """
    try:
        next_review = adaptive_learning_service.practice_scheduler.schedule_next_review(
            item_id, quality=quality_score
        )
        
        logger.info(
            "Scheduled next review",
            item_id=item_id,
            quality=quality_score,
            next_review=next_review
        )
        
        return {
            "next_review_date": next_review,
            "message": f"Review dijadwalkan untuk {next_review.strftime('%Y-%m-%d')}"
        }
        
    except Exception as e:
        logger.error(f"Failed to schedule review: {e}")
        raise HTTPException(status_code=500, detail="Review scheduling failed")


@router.get("/practice/weak-areas/{user_id}")
async def identify_weak_areas(user_id: str):
    """
    Identifikasi weak areas dari recent performance history.
    
    **Analisis:**
    - Error pattern distribution
    - Low-scoring dimensions
    - Frequency of specific error types
    - Prioritized by frequency dan impact
    
    **Response:**
    - List weak areas terurut by priority
    - Error frequency counts
    - Dimension scores
    """
    try:
        # In real implementation, would fetch from database
        return {
            "user_id": user_id,
            "weak_areas": ["general_pronunciation"],
            "error_patterns": {},
            "dimension_scores": {
                "accuracy": 0,
                "fluency": 0,
                "prosody": 0,
                "stress": 0
            }
        }
    except Exception as e:
        logger.error(f"Failed to identify weak areas: {e}")
        raise HTTPException(status_code=500, detail="Weak area identification failed")
