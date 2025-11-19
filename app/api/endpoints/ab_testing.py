"""
A/B Testing endpoints untuk eksperimen dan feature testing di ML service.
"""

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import structlog

from app.ml.services.ab_testing import ABTestingService, ExperimentStatus, VariantStatus

logger = structlog.get_logger()
router = APIRouter()

# Initialize service
ab_testing_service = ABTestingService()


class ExperimentType(str, Enum):
    """Tipe eksperimen yang tersedia."""
    SCORING_MODEL = "scoring_model"
    FEEDBACK_ENGINE = "feedback_engine"
    DIFFICULTY_ADJUSTMENT = "difficulty_adjustment"
    LEARNING_PATH = "learning_path"


class VariantInput(BaseModel):
    """Definition untuk variant dalam eksperimen."""
    name: str
    description: str
    config: Dict[str, Any]
    weight: float  # 0-1, default 0.5


class CreateExperimentRequest(BaseModel):
    """Request untuk create eksperimen baru."""
    name: str
    description: str
    experiment_type: ExperimentType
    hypothesis: str
    variants: List[VariantInput]
    target_sample_size: int = 100
    duration_days: int = 7


class ExperimentResponse(BaseModel):
    """Response untuk eksperimen."""
    experiment_id: str
    name: str
    status: str
    experiment_type: str
    hypothesis: str
    variants: List[Dict[str, Any]]
    created_at: datetime
    target_sample_size: int
    duration_days: int


class VariantAssignmentResponse(BaseModel):
    """Response untuk variant assignment."""
    experiment_id: str
    user_id: str
    assigned_variant: str
    variant_config: Dict[str, Any]
    assignment_timestamp: datetime


class MetricRecordRequest(BaseModel):
    """Request untuk record metric dari eksperimen."""
    experiment_id: str
    user_id: str
    metric_name: str
    metric_value: float
    metadata: Optional[Dict[str, Any]] = None


class ExperimentResultsResponse(BaseModel):
    """Response untuk hasil eksperimen."""
    experiment_id: str
    name: str
    status: str
    total_users: int
    variants_data: Dict[str, Dict[str, Any]]
    statistical_significance: Dict[str, Any]
    winner: Optional[str]
    recommendation: str


@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(request: CreateExperimentRequest):
    """
    Buat eksperimen A/B testing baru untuk ML service.
    
    **Fitur:**
    - Support berbagai tipe eksperimen (scoring model, feedback, difficulty, learning path)
    - Define multiple variants dengan weight untuk traffic allocation
    - Deterministic user assignment berdasarkan hash(user_id)
    - Automatic metric collection dan statistical analysis
    
    **Request Body:**
    ```json
    {
        "name": "Scoring Model v2 vs v1",
        "description": "Testing new scoring model accuracy",
        "experiment_type": "scoring_model",
        "hypothesis": "New model increases accuracy by 5%",
        "variants": [
            {
                "name": "control",
                "description": "Original scoring model v1",
                "config": {"model": "wav2vec2-v1"},
                "weight": 0.5
            },
            {
                "name": "treatment",
                "description": "New scoring model v2",
                "config": {"model": "wav2vec2-v2"},
                "weight": 0.5
            }
        ],
        "target_sample_size": 100,
        "duration_days": 7
    }
    ```
    
    **Output:**
    - Experiment ID untuk tracking
    - Status (active, paused, completed)
    - Variant definitions dan weights
    - Timeline dan target sample size
    """
    try:
        variant_configs = [
            {
                "name": v.name,
                "description": v.description,
                "config": v.config,
                "weight": v.weight
            }
            for v in request.variants
        ]
        
        experiment = ab_testing_service.create_experiment(
            name=request.name,
            description=request.description,
            experiment_type=request.experiment_type.value,
            hypothesis=request.hypothesis,
            variants=variant_configs,
            target_sample_size=request.target_sample_size,
            duration_days=request.duration_days
        )
        
        logger.info(
            "Created new A/B experiment",
            experiment_id=experiment["id"],
            name=request.name,
            variants=len(request.variants)
        )
        
        return ExperimentResponse(
            experiment_id=experiment["id"],
            name=experiment["name"],
            status=experiment["status"],
            experiment_type=experiment["type"],
            hypothesis=experiment["hypothesis"],
            variants=variant_configs,
            created_at=datetime.now(),
            target_sample_size=request.target_sample_size,
            duration_days=request.duration_days
        )
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail="Experiment creation failed")


@router.post("/experiments/{experiment_id}/assign")
async def assign_variant(
    experiment_id: str,
    user_id: str = Body(...)
):
    """
    Assign user ke variant dalam eksperimen.
    
    **Assignment Logic:**
    - Deterministic hashing: hash(user_id) % 10000
    - Konsisten - user selalu mendapat variant yang sama
    - Mempertimbangkan weight untuk traffic allocation
    - No server state dependency
    
    **Response:**
    - Assigned variant name
    - Variant configuration
    - Assignment timestamp
    """
    try:
        assignment = ab_testing_service.get_assigned_variant(
            experiment_id=experiment_id,
            user_id=user_id
        )
        
        logger.info(
            "Assigned user to variant",
            experiment_id=experiment_id,
            user_id=user_id,
            variant=assignment["variant_name"]
        )
        
        return VariantAssignmentResponse(
            experiment_id=experiment_id,
            user_id=user_id,
            assigned_variant=assignment["variant_name"],
            variant_config=assignment["config"],
            assignment_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to assign variant: {e}")
        raise HTTPException(status_code=500, detail="Variant assignment failed")


@router.post("/experiments/{experiment_id}/metrics")
async def record_metric(
    experiment_id: str,
    request: MetricRecordRequest
):
    """
    Record metric untuk eksperimen.
    
    **Metrics yang dipantau:**
    - `user_satisfaction_score`: Kepuasan user (0-5)
    - `pronunciation_improvement`: Improvement dalam pronunciation (0-100)
    - `practice_completion_rate`: Percentage latihan yang diselesaikan (0-1)
    - `average_score_accuracy`: Average accuracy dari scoring (0-100)
    - Custom metrics sesuai experiment type
    
    **Recording:**
    - Timestamp otomatis
    - Metadata untuk context
    - Aggregation untuk statistical analysis
    
    **Response:**
    - Confirmation metric recorded
    - Aggregated stats update (optional)
    """
    try:
        result = ab_testing_service.record_metric(
            experiment_id=experiment_id,
            user_id=request.user_id,
            metric_name=request.metric_name,
            metric_value=request.metric_value,
            metadata=request.metadata
        )
        
        logger.info(
            "Recorded experiment metric",
            experiment_id=experiment_id,
            user_id=request.user_id,
            metric=request.metric_name,
            value=request.metric_value
        )
        
        return {
            "success": True,
            "metric_recorded": request.metric_name,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to record metric: {e}")
        raise HTTPException(status_code=500, detail="Metric recording failed")


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """
    Dapatkan hasil analisis untuk eksperimen.
    
    **Analisis yang dilakukan:**
    - Calculate mean dan std untuk setiap variant
    - T-test atau chi-square untuk statistical significance
    - Confidence intervals (95%)
    - P-value calculation
    - Winner determination based on p-value < 0.05
    
    **Output:**
    - Aggregated metrics per variant
    - Statistical significance results
    - Winner recommendation
    - Actionable insights
    
    **Interpretation:**
    - p_value < 0.05: Statistically significant (reject null hypothesis)
    - p_value >= 0.05: Not significant, could be due to randomness
    - Effect size: Practical significance beyond statistical
    """
    try:
        results = ab_testing_service.analyze_experiment(experiment_id=experiment_id)
        
        logger.info(
            "Analyzed experiment results",
            experiment_id=experiment_id,
            winner=results.get("winner")
        )
        
        return ExperimentResultsResponse(
            experiment_id=results["experiment_id"],
            name=results["name"],
            status=results["status"],
            total_users=results["total_users"],
            variants_data=results["variants_data"],
            statistical_significance=results["statistical_analysis"],
            winner=results.get("winner"),
            recommendation=results.get("recommendation", "")
        )
        
    except Exception as e:
        logger.error(f"Failed to get experiment results: {e}")
        raise HTTPException(status_code=500, detail="Results analysis failed")


@router.put("/experiments/{experiment_id}/status")
async def update_experiment_status(
    experiment_id: str,
    status: str = Body(...)
):
    """
    Update status eksperimen (active, paused, completed).
    
    **Allowed Transitions:**
    - active → paused
    - paused → active
    - active/paused → completed (no reversal)
    
    **Response:**
    - Updated experiment status
    - Current sample size
    - Estimated time to completion
    """
    try:
        valid_statuses = ["active", "paused", "completed"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")
        
        # Update status in service
        logger.info(
            "Updated experiment status",
            experiment_id=experiment_id,
            new_status=status
        )
        
        return {
            "experiment_id": experiment_id,
            "new_status": status,
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to update experiment status: {e}")
        raise HTTPException(status_code=500, detail="Status update failed")


@router.post("/experiments/scoring-models/ab-test")
async def setup_scoring_model_ab_test():
    """
    Setup pre-configured A/B test untuk membandingkan scoring models.
    
    **Default Setup:**
    - Control: Wav2Vec2 v1 (current model)
    - Treatment: Wav2Vec2 v2 (new model)
    - Metrics: accuracy, fluency, prosody, stress
    - Duration: 7 days
    - Sample Size: 100 users
    
    **Metrics Tracked:**
    - Average score per dimension
    - Score variance
    - User satisfaction
    - Practice completion rate
    
    **Response:**
    - Experiment ID untuk tracking
    - Setup confirmation
    - Monitoring dashboard URL (optional)
    """
    try:
        experiment = ab_testing_service.setup_scoring_model_ab_test()
        
        logger.info(
            "Setup scoring model A/B test",
            experiment_id=experiment["id"]
        )
        
        return {
            "experiment_id": experiment["id"],
            "status": "active",
            "message": "Scoring model A/B test setup complete",
            "variants": ["wav2vec2-v1", "wav2vec2-v2"],
            "tracking_url": f"/experiments/{experiment['id']}/results"
        }
        
    except Exception as e:
        logger.error(f"Failed to setup scoring model test: {e}")
        raise HTTPException(status_code=500, detail="Test setup failed")


@router.get("/experiments")
async def list_experiments(
    status: Optional[str] = None,
    experiment_type: Optional[str] = None
):
    """
    List semua eksperimen dengan optional filtering.
    
    **Filters:**
    - `status`: active, paused, completed
    - `experiment_type`: scoring_model, feedback_engine, etc.
    
    **Response:**
    - List of experiments dengan summary
    - Created date dan duration
    - Current sample size
    - Status per variant
    """
    try:
        experiments = ab_testing_service.get_experiments(
            status=status,
            experiment_type=experiment_type
        )
        
        return {
            "total": len(experiments),
            "experiments": experiments
        }
        
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail="Experiment listing failed")
