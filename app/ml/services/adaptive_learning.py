"""
Adaptive Learning Service for personalized pronunciation practice recommendations.
Implements spaced repetition, difficulty adjustment, and weak area identification.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from enum import Enum

logger = structlog.get_logger()


class DifficultyLevel(str, Enum):
    """Difficulty levels for practice items."""
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    UPPER_INTERMEDIATE = "upper_intermediate"
    ADVANCED = "advanced"
    PROFICIENCY = "proficiency"


@dataclass
class PracticeRecommendation:
    """Recommendation for next practice session."""
    difficulty_level: DifficultyLevel
    focus_areas: List[str]
    practice_items: List[Dict]
    session_duration_minutes: int
    next_review_date: datetime
    rationale: str


class SpacedRepetitionScheduler:
    """Implements spaced repetition algorithm for optimal review scheduling."""
    
    def __init__(self):
        self.initial_interval = 1  # days
        self.ease_factor = 2.5
        self.intervals = {}
    
    def schedule_next_review(self, item_id: str, quality: float) -> datetime:
        """
        Schedule next review based on SM-2 algorithm.
        
        Args:
            item_id: Unique identifier for practice item
            quality: Quality of response (0-5)
            
        Returns:
            Datetime for next review
        """
        if item_id not in self.intervals:
            self.intervals[item_id] = {"interval": 1, "ease": self.ease_factor, "repetitions": 0}
        
        item_data = self.intervals[item_id]
        
        # Update ease factor
        new_ease = item_data["ease"] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        new_ease = max(1.3, new_ease)
        
        # Calculate next interval
        if quality >= 3:
            if item_data["repetitions"] == 0:
                next_interval = 1
            elif item_data["repetitions"] == 1:
                next_interval = 3
            else:
                next_interval = int(item_data["interval"] * new_ease)
        else:
            next_interval = 1
        
        # Update data
        item_data["interval"] = next_interval
        item_data["ease"] = new_ease
        item_data["repetitions"] += 1
        
        # Calculate next review date
        next_review = datetime.now() + timedelta(days=next_interval)
        
        logger.info(
            "Scheduled next review",
            item_id=item_id,
            interval=next_interval,
            next_review=next_review
        )
        
        return next_review


class AdaptiveLearningService:
    """Service for adaptive learning recommendations."""
    
    def __init__(self):
        self.practice_scheduler = SpacedRepetitionScheduler()
        self.performance_threshold_good = 0.85
        self.performance_threshold_poor = 0.65
    
    async def recommend_next_practice(
        self,
        user_id: str,
        current_level: str,
        recent_performance: List[Dict],
        user_history: Optional[List[Dict]] = None
    ) -> PracticeRecommendation:
        """
        Generate personalized practice recommendations based on performance.
        
        Args:
            user_id: User identifier
            current_level: Current CEFR level
            recent_performance: Recent pronunciation attempts
            user_history: Historical performance data
            
        Returns:
            PracticeRecommendation with detailed suggestions
        """
        try:
            # Analyze performance trends
            performance_trend = self._analyze_performance_trend(recent_performance)
            
            # Adjust difficulty based on performance
            difficulty_adjustment = self._calculate_difficulty_adjustment(
                performance_trend, current_level
            )
            
            # Identify weak areas for targeted practice
            weak_areas = self._identify_weak_areas(recent_performance)
            
            # Determine next difficulty level
            next_difficulty = self._determine_next_difficulty(
                current_level, difficulty_adjustment
            )
            
            # Select practice items
            practice_items = await self._select_practice_items(
                user_id, weak_areas, next_difficulty
            )
            
            # Calculate optimal session duration
            session_duration = self._calculate_optimal_session_duration(
                recent_performance, len(weak_areas)
            )
            
            # Schedule next review
            next_review = self.practice_scheduler.schedule_next_review(
                f"user_{user_id}_session", quality=3
            )
            
            # Generate rationale
            rationale = self._generate_rationale(
                performance_trend, weak_areas, difficulty_adjustment
            )
            
            recommendation = PracticeRecommendation(
                difficulty_level=next_difficulty,
                focus_areas=weak_areas,
                practice_items=practice_items,
                session_duration_minutes=session_duration,
                next_review_date=next_review,
                rationale=rationale
            )
            
            logger.info(
                "Generated practice recommendation",
                user_id=user_id,
                difficulty=next_difficulty,
                focus_areas=weak_areas,
                duration=session_duration
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Failed to generate recommendation: {e}")
            return self._get_default_recommendation(current_level)
    
    def _analyze_performance_trend(self, recent_performance: List[Dict]) -> Dict:
        """Analyze performance trends from recent attempts."""
        if not recent_performance:
            return {
                "average_accuracy": 0,
                "improvement_rate": 0,
                "consistency": 0,
                "recent_scores": []
            }
        
        scores = [p.get("overall_score", 0) for p in recent_performance]
        recent_scores = scores[-5:]  # Last 5 attempts
        
        # Calculate average
        avg_accuracy = sum(scores) / len(scores) if scores else 0
        
        # Calculate improvement rate
        if len(scores) >= 2:
            improvement_rate = (scores[-1] - scores[-5]) / 5 if len(scores) >= 5 else (scores[-1] - scores[0])
        else:
            improvement_rate = 0
        
        # Calculate consistency (standard deviation)
        if len(scores) > 1:
            variance = sum((x - avg_accuracy) ** 2 for x in scores) / len(scores)
            consistency = 100 - (variance ** 0.5)  # Higher is more consistent
        else:
            consistency = 50
        
        return {
            "average_accuracy": avg_accuracy,
            "improvement_rate": improvement_rate,
            "consistency": max(0, consistency),
            "recent_scores": recent_scores,
            "total_attempts": len(scores)
        }
    
    def _calculate_difficulty_adjustment(self, trend: Dict, current_level: str) -> float:
        """
        Calculate difficulty adjustment factor.
        
        Returns:
            Positive value = increase difficulty
            Negative value = decrease difficulty
            Zero = maintain difficulty
        """
        avg_accuracy = trend["average_accuracy"]
        improvement_rate = trend["improvement_rate"]
        
        # Increase difficulty if performing well and improving
        if avg_accuracy > self.performance_threshold_good and improvement_rate > 0:
            return 0.25  # Increase by 25%
        
        # Maintain if performing OK
        elif self.performance_threshold_poor <= avg_accuracy <= self.performance_threshold_good:
            return 0.0
        
        # Decrease difficulty if struggling
        elif avg_accuracy < self.performance_threshold_poor and improvement_rate < 0:
            return -0.20  # Decrease by 20%
        
        return 0.0
    
    def _identify_weak_areas(self, recent_performance: List[Dict]) -> List[str]:
        """Identify weak areas from recent performance."""
        weak_areas = []
        
        if not recent_performance:
            return ["general_pronunciation"]
        
        # Analyze error patterns
        error_patterns = {}
        dimension_scores = []
        
        for attempt in recent_performance:
            # Track dimension scores
            dimensions = attempt.get("dimensions", {})
            dimension_scores.append(dimensions)
            
            # Track error types
            errors = attempt.get("errors", [])
            for error in errors:
                error_type = error.get("type", "unknown")
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        # Identify low-scoring dimensions
        if dimension_scores:
            avg_dimensions = {}
            for dim in ["accuracy", "fluency", "prosody", "stress"]:
                scores = [d.get(dim, 0) for d in dimension_scores]
                avg_dimensions[dim] = sum(scores) / len(scores) if scores else 0
            
            # Add dimensions scoring < 70 to weak areas
            for dim, score in avg_dimensions.items():
                if score < 70:
                    weak_areas.append(dim)
        
        # Add top error patterns
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:2]:
            if count > 1:
                weak_areas.append(f"error_{error_type}")
        
        return weak_areas if weak_areas else ["general_pronunciation"]
    
    def _determine_next_difficulty(self, current_level: str, adjustment: float) -> DifficultyLevel:
        """Determine next difficulty level based on current level and adjustment."""
        level_mapping = {
            "A1": DifficultyLevel.BEGINNER,
            "A2": DifficultyLevel.ELEMENTARY,
            "B1": DifficultyLevel.INTERMEDIATE,
            "B2": DifficultyLevel.UPPER_INTERMEDIATE,
            "C1": DifficultyLevel.ADVANCED,
            "C2": DifficultyLevel.PROFICIENCY
        }
        
        current_difficulty = level_mapping.get(current_level, DifficultyLevel.INTERMEDIATE)
        
        if adjustment > 0.15:
            # Move to next difficulty level
            difficulty_levels = list(DifficultyLevel)
            current_index = difficulty_levels.index(current_difficulty)
            if current_index < len(difficulty_levels) - 1:
                return difficulty_levels[current_index + 1]
        elif adjustment < -0.15:
            # Move to previous difficulty level
            difficulty_levels = list(DifficultyLevel)
            current_index = difficulty_levels.index(current_difficulty)
            if current_index > 0:
                return difficulty_levels[current_index - 1]
        
        return current_difficulty
    
    async def _select_practice_items(
        self,
        user_id: str,
        weak_areas: List[str],
        difficulty: DifficultyLevel
    ) -> List[Dict]:
        """Select practice items based on weak areas and difficulty."""
        # This would typically fetch from a practice items database
        # For now, return template items
        practice_items = []
        
        for area in weak_areas[:3]:  # Focus on top 3 weak areas
            if "accuracy" in area:
                practice_items.append({
                    "id": f"practice_accuracy_{difficulty.value}",
                    "type": "pronunciation",
                    "focus": "accuracy",
                    "difficulty": difficulty.value,
                    "word": "difficult",
                    "description": "Practice accurate pronunciation of challenging words"
                })
            elif "fluency" in area:
                practice_items.append({
                    "id": f"practice_fluency_{difficulty.value}",
                    "type": "fluency",
                    "focus": "fluency",
                    "difficulty": difficulty.value,
                    "sentence": "The quick brown fox jumps over the lazy dog",
                    "description": "Practice smooth and continuous speech"
                })
            elif "prosody" in area:
                practice_items.append({
                    "id": f"practice_prosody_{difficulty.value}",
                    "type": "prosody",
                    "focus": "prosody",
                    "difficulty": difficulty.value,
                    "sentence": "Really? You didn't know that?",
                    "description": "Practice intonation and stress patterns"
                })
            elif "stress" in area:
                practice_items.append({
                    "id": f"practice_stress_{difficulty.value}",
                    "type": "stress",
                    "focus": "stress",
                    "difficulty": difficulty.value,
                    "word_pairs": [("RECORD", "reCORD"), ("PRODUCE", "proDUCE")],
                    "description": "Practice word stress patterns"
                })
        
        return practice_items
    
    def _calculate_optimal_session_duration(
        self,
        recent_performance: List[Dict],
        num_weak_areas: int
    ) -> int:
        """Calculate optimal practice session duration in minutes."""
        # Base duration: 15 minutes
        base_duration = 15
        
        # Add time for each weak area (5 minutes each, max 15 additional)
        area_time = min(num_weak_areas * 5, 15)
        
        # Adjust based on consistency
        if recent_performance:
            trend = self._analyze_performance_trend(recent_performance)
            consistency = trend.get("consistency", 50)
            
            # Lower consistency = need more practice
            if consistency < 40:
                area_time += 10
            elif consistency > 80:
                area_time -= 5
        
        total_duration = base_duration + area_time
        
        # Cap at reasonable limits
        return min(max(total_duration, 15), 60)
    
    def _generate_rationale(
        self,
        performance_trend: Dict,
        weak_areas: List[str],
        difficulty_adjustment: float
    ) -> str:
        """Generate explanation for the recommendation."""
        avg_accuracy = performance_trend.get("average_accuracy", 0)
        improvement_rate = performance_trend.get("improvement_rate", 0)
        
        rationale_parts = []
        
        # Performance assessment
        if avg_accuracy > 85:
            rationale_parts.append("Anda menunjukkan kinerja yang sangat baik")
        elif avg_accuracy > 70:
            rationale_parts.append("Anda menunjukkan kinerja yang baik")
        else:
            rationale_parts.append("Anda perlu lebih banyak latihan")
        
        # Improvement trend
        if improvement_rate > 5:
            rationale_parts.append("dan sedang mengalami peningkatan yang konsisten")
        elif improvement_rate < -5:
            rationale_parts.append("namun performa menunjukkan tren menurun")
        else:
            rationale_parts.append("dengan performa yang konsisten")
        
        # Weak areas
        if weak_areas:
            rationale_parts.append(f". Fokus pada: {', '.join(weak_areas[:2])}")
        
        # Difficulty adjustment
        if difficulty_adjustment > 0.15:
            rationale_parts.append(". Kami meningkatkan kesulitan untuk tantangan lebih")
        elif difficulty_adjustment < -0.15:
            rationale_parts.append(". Kami menurunkan kesulitan untuk membangun kepercayaan diri")
        
        return "".join(rationale_parts)
    
    def _get_default_recommendation(self, current_level: str) -> PracticeRecommendation:
        """Get default recommendation when analysis fails."""
        return PracticeRecommendation(
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            focus_areas=["general_pronunciation"],
            practice_items=[
                {
                    "id": "default_practice_1",
                    "type": "pronunciation",
                    "word": "pronunciation",
                    "description": "Basic pronunciation practice"
                }
            ],
            session_duration_minutes=20,
            next_review_date=datetime.now() + timedelta(days=1),
            rationale="Mulai dengan latihan dasar untuk membangun fondasi yang kuat"
        )
