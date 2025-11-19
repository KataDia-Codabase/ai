"""
A/B Testing Service for ML model experimentation and optimization.
Manages experiments, variant assignment, and analytics collection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid
import structlog
import json

logger = structlog.get_logger()


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantStatus(str, Enum):
    """Status of a variant."""
    ACTIVE = "active"
    PAUSED = "paused"
    CONTROL = "control"


@dataclass
class Variant:
    """Experiment variant configuration."""
    name: str
    model_name: str
    weight: float  # Traffic allocation percentage
    status: VariantStatus


@dataclass
class ExperimentMetric:
    """Metric data point for experiment."""
    experiment_id: str
    variant_name: str
    user_id: str
    metric_name: str
    metric_value: float
    timestamp: datetime


@dataclass
class ExperimentResult:
    """Result of an experiment."""
    experiment_id: str
    variant_name: str
    metric_name: str
    sample_size: int
    average_value: float
    std_dev: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool


class ExperimentManager:
    """Manages A/B testing experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.user_assignments: Dict[str, Dict] = {}
        self.metrics: List[ExperimentMetric] = []
    
    async def create_experiment(self, experiment_config: Dict) -> str:
        """
        Create new experiment.
        
        Args:
            experiment_config: {
                "name": str,
                "description": str,
                "variants": [{"name": str, "model": str, "weight": float}],
                "success_metrics": List[str],
                "duration_days": int,
                "target_sample_size": Optional[int]
            }
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        # Validate weights sum to 1.0
        total_weight = sum(v.get("weight", 0) for v in experiment_config.get("variants", []))
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
        
        # Create variants
        variants = []
        for i, v in enumerate(experiment_config.get("variants", [])):
            variant = Variant(
                name=v.get("name", f"variant_{i}"),
                model_name=v.get("model", ""),
                weight=v.get("weight", 1.0 / len(experiment_config.get("variants", [1]))),
                status=VariantStatus.CONTROL if i == 0 else VariantStatus.ACTIVE
            )
            variants.append(variant)
        
        # Store experiment
        self.experiments[experiment_id] = {
            "id": experiment_id,
            "name": experiment_config.get("name", ""),
            "description": experiment_config.get("description", ""),
            "variants": [
                {
                    "name": v.name,
                    "model": v.model_name,
                    "weight": v.weight,
                    "status": v.status.value
                }
                for v in variants
            ],
            "success_metrics": experiment_config.get("success_metrics", []),
            "status": ExperimentStatus.ACTIVE.value,
            "created_at": datetime.now(),
            "end_date": datetime.now() + timedelta(days=experiment_config.get("duration_days", 7)),
            "target_sample_size": experiment_config.get("target_sample_size")
        }
        
        logger.info(
            "Created new experiment",
            experiment_id=experiment_id,
            name=experiment_config.get("name"),
            variants=[v.name for v in variants]
        )
        
        return experiment_id
    
    async def get_assigned_variant(
        self,
        user_id: str,
        experiment_id: str
    ) -> Dict:
        """
        Get assigned variant for user in experiment.
        
        Returns:
            Dict with variant info: {"name": str, "model": str, "weight": float}
        """
        # Check if user already assigned
        assignment_key = f"{experiment_id}_{user_id}"
        if assignment_key in self.user_assignments:
            return self.user_assignments[assignment_key]
        
        # Get experiment
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        variants = experiment.get("variants", [])
        
        if not variants:
            raise ValueError(f"No variants in experiment {experiment_id}")
        
        # Assign variant based on weights
        assigned_variant = self._assign_variant_by_weight(user_id, variants)
        
        # Store assignment
        self.user_assignments[assignment_key] = assigned_variant
        
        logger.info(
            "Assigned user to variant",
            user_id=user_id,
            experiment_id=experiment_id,
            variant=assigned_variant["name"]
        )
        
        return assigned_variant
    
    def _assign_variant_by_weight(
        self,
        user_id: str,
        variants: List[Dict]
    ) -> Dict:
        """Assign variant to user based on weights."""
        # Use deterministic hash of user_id for consistent assignment
        user_hash = hash(user_id) % 10000
        cumulative_weight = 0
        normalized_hash = user_hash / 10000
        
        for variant in variants:
            cumulative_weight += variant.get("weight", 0)
            if normalized_hash <= cumulative_weight:
                return variant
        
        # Fallback to last variant
        return variants[-1]
    
    async def record_metric(
        self,
        experiment_id: str,
        variant_name: str,
        user_id: str,
        metric_name: str,
        metric_value: float
    ) -> None:
        """Record metric for experiment."""
        metric = ExperimentMetric(
            experiment_id=experiment_id,
            variant_name=variant_name,
            user_id=user_id,
            metric_name=metric_name,
            metric_value=metric_value,
            timestamp=datetime.now()
        )
        
        self.metrics.append(metric)
    
    async def analyze_experiment(
        self,
        experiment_id: str
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Analyze experiment results.
        
        Returns:
            Dict mapping metric names to results for each variant
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        metrics = experiment.get("success_metrics", [])
        
        results = {}
        
        for metric_name in metrics:
            results[metric_name] = []
            
            # Get data for each variant
            for variant in experiment.get("variants", []):
                variant_metrics = [
                    m for m in self.metrics
                    if m.experiment_id == experiment_id
                    and m.variant_name == variant["name"]
                    and m.metric_name == metric_name
                ]
                
                if variant_metrics:
                    values = [m.metric_value for m in variant_metrics]
                    result = self._calculate_metric_stats(
                        experiment_id, variant["name"], metric_name, values
                    )
                    results[metric_name].append(result)
        
        return results
    
    def _calculate_metric_stats(
        self,
        experiment_id: str,
        variant_name: str,
        metric_name: str,
        values: List[float]
    ) -> ExperimentResult:
        """Calculate statistics for metric."""
        import numpy as np
        from scipy import stats
        
        values_array = np.array(values)
        n = len(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        # 95% confidence interval
        ci = stats.t.interval(
            0.95,
            n - 1,
            loc=mean,
            scale=std / np.sqrt(n) if n > 1 else 0
        )
        
        # Placeholder for p-value (would need control group comparison)
        p_value = 0.05
        is_significant = p_value < 0.05
        
        return ExperimentResult(
            experiment_id=experiment_id,
            variant_name=variant_name,
            metric_name=metric_name,
            sample_size=n,
            average_value=float(mean),
            std_dev=float(std),
            confidence_interval=(float(ci[0]), float(ci[1])),
            p_value=p_value,
            is_significant=is_significant
        )
    
    async def pause_experiment(self, experiment_id: str) -> None:
        """Pause an experiment."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = ExperimentStatus.PAUSED.value
            logger.info("Paused experiment", experiment_id=experiment_id)
    
    async def complete_experiment(self, experiment_id: str) -> None:
        """Complete an experiment."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = ExperimentStatus.COMPLETED.value
            logger.info("Completed experiment", experiment_id=experiment_id)
    
    async def get_experiment_status(self, experiment_id: str) -> Dict:
        """Get current status of experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        sample_sizes = {}
        
        for variant in experiment.get("variants", []):
            variant_metrics = [
                m for m in self.metrics
                if m.experiment_id == experiment_id
                and m.variant_name == variant["name"]
            ]
            sample_sizes[variant["name"]] = len(set(m.user_id for m in variant_metrics))
        
        return {
            "id": experiment_id,
            "name": experiment.get("name"),
            "status": experiment.get("status"),
            "created_at": experiment.get("created_at").isoformat(),
            "end_date": experiment.get("end_date").isoformat(),
            "sample_sizes": sample_sizes,
            "variants": experiment.get("variants")
        }


class ABTestingService:
    """Service for managing A/B tests with ML models."""
    
    def __init__(self):
        self.experiment_manager = ExperimentManager()
    
    async def setup_scoring_model_ab_test(self) -> str:
        """Setup A/B test for comparing pronunciation scoring models."""
        experiment_config = {
            "name": "pronunciation_scoring_model_comparison",
            "description": "Compare baseline vs enhanced pronunciation scoring models",
            "variants": [
                {
                    "name": "control",
                    "model": "wav2vec2_baseline",
                    "weight": 0.5
                },
                {
                    "name": "treatment",
                    "model": "wav2vec2_enhanced",
                    "weight": 0.5
                }
            ],
            "success_metrics": [
                "user_satisfaction_score",
                "pronunciation_improvement",
                "practice_completion_rate",
                "average_score_accuracy"
            ],
            "duration_days": 14,
            "target_sample_size": 500
        }
        
        experiment_id = await self.experiment_manager.create_experiment(experiment_config)
        
        logger.info(
            "Created scoring model A/B test",
            experiment_id=experiment_id,
            duration_days=14
        )
        
        return experiment_id
    
    async def route_scoring_request(
        self,
        user_id: str,
        experiment_id: str,
        request_data: Dict
    ) -> Tuple[str, Dict]:
        """
        Route scoring request to appropriate model variant.
        
        Returns:
            Tuple of (model_name, variant_config)
        """
        variant = await self.experiment_manager.get_assigned_variant(
            user_id, experiment_id
        )
        
        return variant["model"], variant
    
    async def record_experiment_result(
        self,
        experiment_id: str,
        variant_name: str,
        user_id: str,
        result: Dict
    ) -> None:
        """Record result from scoring request for experiment analysis."""
        
        # Record relevant metrics
        await self.experiment_manager.record_metric(
            experiment_id,
            variant_name,
            user_id,
            "average_score_accuracy",
            result.get("overall_score", 0)
        )
        
        # Record dimension scores
        dimensions = result.get("dimensions", {})
        for dim, score in dimensions.items():
            await self.experiment_manager.record_metric(
                experiment_id,
                variant_name,
                user_id,
                f"score_{dim}",
                score
            )
    
    async def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get current results of experiment."""
        status = await self.experiment_manager.get_experiment_status(experiment_id)
        
        try:
            analysis = await self.experiment_manager.analyze_experiment(experiment_id)
        except Exception as e:
            logger.warning(f"Could not analyze experiment: {e}")
            analysis = {}
        
        return {
            "status": status,
            "analysis": analysis
        }
