"""
Pronunciation Analytics Service for tracking progress and generating insights.
Provides progress tracking, native speaker comparison, and learning analytics.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import structlog
from enum import Enum

logger = structlog.get_logger()


@dataclass
class ProgressMetric:
    """Single progress metric data point."""
    timestamp: datetime
    score: float
    dimension: str
    category: str


@dataclass
class ProgressAnalytics:
    """Comprehensive progress analytics."""
    user_id: str
    total_attempts: int
    average_score: float
    improvement_rate: float
    consistency_score: float
    learning_velocity: float
    retention_rate: float
    strong_areas: List[str]
    weak_areas: List[str]
    time_spent_hours: float
    last_update: datetime


class NativeSpeakerComparator:
    """Compares user pronunciation with native speaker benchmarks."""
    
    def __init__(self):
        self.native_benchmarks = self._load_native_benchmarks()
    
    def _load_native_benchmarks(self) -> Dict:
        """Load native speaker benchmark data."""
        return {
            "accuracy": {"mean": 95, "std": 2},
            "fluency": {"mean": 92, "std": 3},
            "prosody": {"mean": 90, "std": 4},
            "stress": {"mean": 88, "std": 5}
        }
    
    async def compare_with_native(
        self,
        user_dimensions: Dict[str, float]
    ) -> Dict:
        """
        Compare user performance with native speaker benchmarks.
        
        Returns:
            Dict with comparison metrics and insights
        """
        comparison = {}
        insights = []
        
        for dimension, user_score in user_dimensions.items():
            if dimension in self.native_benchmarks:
                benchmark = self.native_benchmarks[dimension]
                mean = benchmark["mean"]
                std = benchmark["std"]
                
                # Calculate distance from native standard
                z_score = (user_score - mean) / std if std > 0 else 0
                percentile = self._z_score_to_percentile(z_score)
                
                comparison[dimension] = {
                    "user_score": user_score,
                    "native_mean": mean,
                    "difference": user_score - mean,
                    "percentile": percentile,
                    "z_score": z_score
                }
                
                # Generate insights
                if user_score >= mean - std:
                    insights.append(f"✓ {dimension.capitalize()}: Mendekati standar native speaker")
                elif user_score >= mean - 2 * std:
                    insights.append(f"→ {dimension.capitalize()}: Butuh perbaikan untuk mencapai native level")
                else:
                    insights.append(f"⚠ {dimension.capitalize()}: Masih jauh dari native speaker standard")
        
        return {
            "comparison": comparison,
            "insights": insights,
            "overall_native_similarity": self._calculate_overall_similarity(comparison)
        }
    
    def _z_score_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile."""
        # Approximation using error function
        from math import erf
        percentile = 50 * (1 + erf(z_score / (2 ** 0.5)))
        return min(100, max(0, percentile))
    
    def _calculate_overall_similarity(self, comparison: Dict) -> float:
        """Calculate overall similarity to native speaker."""
        if not comparison:
            return 0
        
        percentiles = [c.get("percentile", 0) for c in comparison.values()]
        return sum(percentiles) / len(percentiles) if percentiles else 0


class PronunciationAnalyticsService:
    """Service for comprehensive pronunciation analytics and insights."""
    
    def __init__(self):
        self.native_comparator = NativeSpeakerComparator()
    
    async def generate_comprehensive_analytics(
        self,
        user_id: str,
        pronunciation_history: List[Dict],
        user_metadata: Optional[Dict] = None
    ) -> ProgressAnalytics:
        """
        Generate comprehensive analytics for user pronunciation progress.
        
        Args:
            user_id: User identifier
            pronunciation_history: List of pronunciation attempts with scores
            user_metadata: Optional user metadata
            
        Returns:
            ProgressAnalytics with detailed insights
        """
        try:
            if not pronunciation_history:
                return self._get_default_analytics(user_id)
            
            # Calculate basic statistics
            total_attempts = len(pronunciation_history)
            scores = [p.get("overall_score", 0) for p in pronunciation_history]
            average_score = np.mean(scores) if scores else 0
            
            # Calculate improvement rate
            improvement_rate = self._calculate_improvement_rate(scores)
            
            # Calculate consistency
            consistency_score = self._calculate_consistency(scores)
            
            # Calculate learning velocity
            learning_velocity = self._calculate_learning_velocity(
                pronunciation_history
            )
            
            # Calculate retention rate
            retention_rate = self._calculate_retention_rate(pronunciation_history)
            
            # Identify strong and weak areas
            strong_areas, weak_areas = self._identify_areas(pronunciation_history)
            
            # Calculate time spent
            time_spent_hours = self._calculate_time_spent(pronunciation_history)
            
            analytics = ProgressAnalytics(
                user_id=user_id,
                total_attempts=total_attempts,
                average_score=average_score,
                improvement_rate=improvement_rate,
                consistency_score=consistency_score,
                learning_velocity=learning_velocity,
                retention_rate=retention_rate,
                strong_areas=strong_areas,
                weak_areas=weak_areas,
                time_spent_hours=time_spent_hours,
                last_update=datetime.now()
            )
            
            logger.info(
                "Generated comprehensive analytics",
                user_id=user_id,
                total_attempts=total_attempts,
                average_score=average_score,
                improvement_rate=improvement_rate
            )
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate analytics: {e}")
            return self._get_default_analytics(user_id)
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """
        Calculate improvement rate using linear regression.
        
        Returns:
            Points per attempt
        """
        if len(scores) < 2:
            return 0
        
        # Use linear regression to find trend
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Calculate slope
        n = len(scores)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x ** 2) - (np.sum(x)) ** 2
        )
        
        return float(slope)
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """
        Calculate consistency score (inverse of standard deviation).
        Higher score = more consistent
        """
        if len(scores) < 2:
            return 50
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        # Convert std to 0-100 scale (100 = perfectly consistent)
        consistency = 100 - min(std * 10, 100)
        
        return float(max(0, consistency))
    
    def _calculate_learning_velocity(self, history: List[Dict]) -> float:
        """
        Calculate learning velocity (rate of improvement per time unit).
        """
        if len(history) < 3:
            return 0
        
        # Get recent attempts (last 10 or all if less)
        recent = history[-10:]
        
        if not recent:
            return 0
        
        # Get scores and timestamps
        scores = [p.get("overall_score", 0) for p in recent]
        timestamps = [p.get("created_at", datetime.now()) for p in recent]
        
        # Calculate time span
        if timestamps:
            time_span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            if time_span_hours == 0:
                time_span_hours = 0.1
            
            # Calculate score improvement
            score_improvement = scores[-1] - scores[0]
            
            # Learning velocity = improvement per hour
            velocity = score_improvement / time_span_hours
            
            return float(velocity)
        
        return 0
    
    def _calculate_retention_rate(self, history: List[Dict]) -> float:
        """
        Calculate retention rate (consistency of maintaining skill level).
        """
        if len(history) < 5:
            return 0.5
        
        scores = [p.get("overall_score", 0) for p in history[-20:]]  # Last 20 attempts
        
        if len(scores) < 2:
            return 0.5
        
        # Group into sessions
        sessions = []
        current_session = [scores[0]]
        
        for i in range(1, len(scores)):
            # If more than 5 points difference, it's a new session
            if abs(scores[i] - scores[i-1]) > 15:
                sessions.append(current_session)
                current_session = [scores[i]]
            else:
                current_session.append(scores[i])
        
        sessions.append(current_session)
        
        # Calculate retention between sessions
        if len(sessions) < 2:
            return 0.7
        
        retentions = []
        for i in range(1, len(sessions)):
            prev_avg = np.mean(sessions[i-1])
            curr_avg = np.mean(sessions[i])
            retention = curr_avg / prev_avg if prev_avg > 0 else 0.5
            retentions.append(min(retention, 1.0))
        
        return float(np.mean(retentions)) if retentions else 0.5
    
    def _identify_areas(
        self,
        history: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        """Identify strong and weak areas."""
        dimension_scores = {"accuracy": [], "fluency": [], "prosody": [], "stress": []}
        
        for attempt in history:
            dimensions = attempt.get("dimensions", {})
            for dim, score in dimensions.items():
                if dim in dimension_scores:
                    dimension_scores[dim].append(score)
        
        # Calculate averages
        avg_scores = {}
        for dim, scores in dimension_scores.items():
            if scores:
                avg_scores[dim] = np.mean(scores)
        
        # Identify strong (>75) and weak (<70)
        strong_areas = [dim for dim, score in avg_scores.items() if score > 75]
        weak_areas = [dim for dim, score in avg_scores.items() if score < 70]
        
        return strong_areas, weak_areas
    
    def _calculate_time_spent(self, history: List[Dict]) -> float:
        """Calculate total time spent practicing."""
        if not history or len(history) < 2:
            return 0
        
        timestamps = [p.get("created_at", datetime.now()) for p in history]
        if len(timestamps) < 2:
            return 0
        
        total_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        total_hours = total_seconds / 3600
        
        return float(total_hours)
    
    async def analyze_progress_over_time(
        self,
        user_id: str,
        pronunciation_history: List[Dict],
        interval_days: int = 7
    ) -> Dict:
        """
        Analyze progress over time with aggregation by interval.
        
        Args:
            user_id: User identifier
            pronunciation_history: Complete pronunciation history
            interval_days: Interval for aggregation
            
        Returns:
            Dict with progress metrics by time interval
        """
        progress_intervals = {}
        
        for attempt in pronunciation_history:
            created_at = attempt.get("created_at", datetime.now())
            
            # Calculate interval
            days_ago = (datetime.now() - created_at).days
            interval_index = days_ago // interval_days
            
            interval_key = f"interval_{interval_index}"
            
            if interval_key not in progress_intervals:
                progress_intervals[interval_key] = {
                    "attempts": 0,
                    "total_score": 0,
                    "dimensions": {"accuracy": 0, "fluency": 0, "prosody": 0, "stress": 0},
                    "count": {dim: 0 for dim in ["accuracy", "fluency", "prosody", "stress"]}
                }
            
            progress_intervals[interval_key]["attempts"] += 1
            progress_intervals[interval_key]["total_score"] += attempt.get("overall_score", 0)
            
            dimensions = attempt.get("dimensions", {})
            for dim, score in dimensions.items():
                if dim in progress_intervals[interval_key]["dimensions"]:
                    progress_intervals[interval_key]["dimensions"][dim] += score
                    progress_intervals[interval_key]["count"][dim] += 1
        
        # Calculate averages
        for interval, data in progress_intervals.items():
            if data["attempts"] > 0:
                data["average_score"] = data["total_score"] / data["attempts"]
            
            for dim in data["dimensions"]:
                if data["count"][dim] > 0:
                    data["dimensions"][dim] /= data["count"][dim]
        
        return progress_intervals
    
    def _get_default_analytics(self, user_id: str) -> ProgressAnalytics:
        """Get default analytics when no history available."""
        return ProgressAnalytics(
            user_id=user_id,
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
