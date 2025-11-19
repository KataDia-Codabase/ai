"""
Production Optimizer API Endpoints

Endpoints for:
- Model optimization (quantization, ONNX, pruning)
- Performance profiling and benchmarking
- Optimization metrics and recommendations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import structlog
import asyncio
import torch
from pathlib import Path

from app.ml.optimization.production_optimizer import (
    EnglishModelOptimizer,
    ProductionOptimizationService,
    ProductionOptimizedModelLoader,
    OptimizationConfig,
    ENGLISH_OPTIMIZATION_CONFIGS,
    LatencyOptimization,
    PruningOptimizer,
    DistillationOptimizer,
    EnhancedProductionOptimizer,
    BatchOptimization
)

logger = structlog.get_logger()

# Router initialization
router = APIRouter(prefix="/optimizer", tags=["optimizer"])

# ==================== REQUEST/RESPONSE MODELS ====================

class OptimizationRequest(BaseModel):
    """Request model for optimization."""
    model_path: str = Field(..., description="Path to model file")
    optimization_type: str = Field("comprehensive", description="Type of optimization")
    quantization: bool = Field(True, description="Enable quantization")
    onnx_conversion: bool = Field(True, description="Enable ONNX conversion")
    pruning: bool = Field(False, description="Enable pruning")
    distillation: bool = Field(False, description="Enable knowledge distillation")
    profile_mode: str = Field("production", description="Optimization profile")


class ProfilingRequest(BaseModel):
    """Request model for profiling."""
    model_path: str = Field(..., description="Path to model file")
    input_shape: List[int] = Field(default=[1, 16000], description="Input shape for dummy data")
    num_iterations: int = Field(100, description="Number of profiling iterations")
    detailed: bool = Field(False, description="Include detailed metrics")


class QuantizationRequest(BaseModel):
    """Request model for quantization."""
    model_path: str = Field(..., description="Path to model file")
    quantization_type: str = Field("int8", description="Quantization type (int8, fp16, dynamic)")
    output_path: str = Field(..., description="Output path for quantized model")
    calibration_data_size: int = Field(100, description="Size of calibration data")


class PruningRequest(BaseModel):
    """Request model for pruning."""
    model_path: str = Field(..., description="Path to model file")
    pruning_rate: float = Field(0.3, description="Pruning rate (0.0-1.0)")
    pruning_type: str = Field("unstructured", description="Pruning type")
    output_path: str = Field(..., description="Output path for pruned model")


class OptimizationResponse(BaseModel):
    """Response model for optimization."""
    success: bool
    optimization_type: str
    original_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvements: Dict[str, float]
    recommendations: List[str]
    output_paths: Dict[str, str]
    duration_seconds: float


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    model_size_mb: float
    latency_ms: float
    throughput_qps: float
    memory_usage_mb: float
    quantization_type: str


class BottleneckAnalysis(BaseModel):
    """Bottleneck analysis result."""
    stage: str
    latency_ms: float
    percentage: float


class LatencyAnalysisResponse(BaseModel):
    """Response model for latency analysis."""
    bottlenecks: List[BottleneckAnalysis]
    recommendations: List[str]
    total_latency_ms: float


# ==================== OPTIMIZATION ENDPOINTS ====================

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_model(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Optimize a model for production deployment.
    
    Supports:
    - Dynamic quantization (INT8, FP16)
    - ONNX conversion
    - Structured/unstructured pruning
    - Knowledge distillation
    - TorchScript compilation
    
    Args:
        request: Optimization parameters
        background_tasks: Background task runner
        
    Returns:
        Optimization results with metrics and recommendations
    """
    try:
        logger.info(
            "Optimization request received",
            model_path=request.model_path,
            optimization_type=request.optimization_type
        )
        
        # Validate model path
        model_path = Path(request.model_path)
        if not model_path.exists():
            # Try to provide helpful error message
            parent = model_path.parent
            if parent.exists():
                available = [f.name for f in parent.iterdir()]
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model not found at: {request.model_path}. Available in {parent}: {', '.join(available[:5])}"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Model path not found: {request.model_path}. Directory does not exist.")
        
        # Get optimization config
        profile = ENGLISH_OPTIMIZATION_CONFIGS.get(
            request.profile_mode,
            ENGLISH_OPTIMIZATION_CONFIGS["production"]
        )
        
        # Update config based on request
        profile.quantization = request.quantization
        profile.onnx_conversion = request.onnx_conversion
        
        # Initialize optimizer
        optimizer = EnglishModelOptimizer(profile)
        
        # Create output directory
        output_dir = f"./models/optimized_{request.optimization_type}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run optimization
        results = await optimizer.optimize_for_production(
            model_path=request.model_path,
            output_path=output_dir,
            model_type="wav2vec2"
        )
        
        # Prepare response
        original_metrics = results.get("original_metrics", {})
        best_model = results.get("recommended_model", {})
        best_metrics = best_model.get("metrics", {})
        
        improvements = {}
        if original_metrics.get("avg_inference_time") and best_metrics.get("avg_inference_time"):
            improvements["latency_improvement"] = (
                (original_metrics["avg_inference_time"] - best_metrics["avg_inference_time"]) /
                original_metrics["avg_inference_time"] * 100
            )
        
        logger.info(
            "Optimization completed successfully",
            improvements=improvements,
            recommended_type=best_model.get("type")
        )
        
        return {
            "success": True,
            "optimization_type": request.optimization_type,
            "original_metrics": original_metrics,
            "optimized_metrics": best_metrics,
            "improvements": improvements,
            "recommendations": results.get("recommendations", []),
            "output_paths": {best_model.get("type", "best"): best_model.get("path", "")},
            "duration_seconds": 0  # Would calculate from timestamps
        }
        
    except Exception as e:
        logger.error("Optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/quantize", response_model=MetricsResponse)
async def quantize_model(request: QuantizationRequest) -> Dict[str, Any]:
    """
    Quantize a model to reduce size and improve latency.
    
    Supported quantization types:
    - int8: 8-bit integer quantization
    - fp16: 16-bit floating point quantization
    - dynamic: Dynamic quantization at inference time
    
    Args:
        request: Quantization parameters
        
    Returns:
        Quantized model metrics
    """
    try:
        logger.info(
            "Quantization request received",
            model_path=request.model_path,
            quantization_type=request.quantization_type
        )
        
        if not Path(request.model_path).exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
        
        # Create output directory
        Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        config = OptimizationConfig(quantization=True)
        optimizer = EnglishModelOptimizer(config)
        
        # Load and quantize model
        model = optimizer._load_model(request.model_path, "wav2vec2")
        quantized_model = await optimizer._quantize_model(model, request.output_path)
        
        # Benchmark quantized model
        metrics = await optimizer._benchmark_model(quantized_model, "wav2vec2")
        
        logger.info(
            "Quantization completed",
            output_path=request.output_path,
            metrics=metrics
        )
        
        return {
            "model_size_mb": metrics.get("model_size_mb", 0),
            "latency_ms": metrics.get("avg_inference_time", 0) * 1000,
            "throughput_qps": 1000 / (metrics.get("avg_inference_time", 1) * 1000),
            "memory_usage_mb": metrics.get("peak_memory_mb", 0),
            "quantization_type": request.quantization_type
        }
        
    except Exception as e:
        logger.error("Quantization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Quantization failed: {str(e)}")


@router.post("/prune", response_model=MetricsResponse)
async def prune_model(request: PruningRequest) -> Dict[str, Any]:
    """
    Prune a model to reduce size and improve inference speed.
    
    Supports:
    - Unstructured pruning: Remove individual weights
    - Structured pruning: Remove entire filters/channels
    
    Args:
        request: Pruning parameters
        
    Returns:
        Pruned model metrics
    """
    try:
        logger.info(
            "Pruning request received",
            model_path=request.model_path,
            pruning_rate=request.pruning_rate
        )
        
        if not Path(request.model_path).exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
        
        if not (0.0 <= request.pruning_rate <= 1.0):
            raise HTTPException(status_code=400, detail="Pruning rate must be between 0.0 and 1.0")
        
        # Create output directory
        Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        optimizer = EnglishModelOptimizer()
        
        # Load and prune model
        model = optimizer._load_model(request.model_path, "wav2vec2")
        
        if request.pruning_type == "structured":
            pruned_model = PruningOptimizer.structured_pruning(model, request.pruning_rate)
        else:
            pruned_model = PruningOptimizer.unstructured_pruning(model, request.pruning_rate)
        
        # Save pruned model
        pruned_model.save_pretrained(request.output_path)
        
        # Benchmark pruned model
        metrics = await optimizer._benchmark_model(pruned_model, "wav2vec2")
        
        logger.info(
            "Pruning completed",
            output_path=request.output_path,
            pruning_rate=request.pruning_rate
        )
        
        return {
            "model_size_mb": metrics.get("model_size_mb", 0),
            "latency_ms": metrics.get("avg_inference_time", 0) * 1000,
            "throughput_qps": 1000 / (metrics.get("avg_inference_time", 1) * 1000),
            "memory_usage_mb": metrics.get("peak_memory_mb", 0),
            "quantization_type": "pruned"
        }
        
    except Exception as e:
        logger.error("Pruning failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pruning failed: {str(e)}")


# ==================== PROFILING ENDPOINTS ====================

@router.post("/profile", response_model=MetricsResponse)
async def profile_model(request: ProfilingRequest) -> Dict[str, Any]:
    """
    Profile a model to measure performance metrics.
    
    Measures:
    - Average inference latency
    - P95/P99 latency
    - Memory usage
    - Model size
    
    Args:
        request: Profiling parameters
        
    Returns:
        Performance metrics
    """
    try:
        logger.info(
            "Profiling request received",
            model_path=request.model_path,
            num_iterations=request.num_iterations
        )
        
        if not Path(request.model_path).exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
        
        # Initialize optimizer
        optimizer = EnglishModelOptimizer()
        
        # Load model
        model = optimizer._load_model(request.model_path, "wav2vec2")
        
        # Profile model
        metrics = await optimizer._benchmark_model(
            model,
            "wav2vec2",
            num_samples=request.num_iterations
        )
        
        logger.info("Profiling completed", metrics=metrics)
        
        response = {
            "model_size_mb": metrics.get("model_size_mb", 0),
            "latency_ms": metrics.get("avg_inference_time", 0) * 1000,
            "throughput_qps": 1000 / (metrics.get("avg_inference_time", 1) * 1000),
            "memory_usage_mb": metrics.get("peak_memory_mb", 0),
            "quantization_type": "fp32"
        }
        
        if request.detailed:
            response["p95_latency_ms"] = metrics.get("p95_inference_time", 0) * 1000
            response["p99_latency_ms"] = metrics.get("p99_inference_time", 0) * 1000
        
        return response
        
    except Exception as e:
        logger.error("Profiling failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")


@router.post("/latency-analysis", response_model=LatencyAnalysisResponse)
async def analyze_latency(request: ProfilingRequest) -> Dict[str, Any]:
    """
    Analyze model latency to identify bottlenecks.
    
    Breaks down latency into:
    - Preprocessing
    - Forward pass
    - Postprocessing
    
    Args:
        request: Analysis parameters
        
    Returns:
        Bottleneck analysis and recommendations
    """
    try:
        logger.info(
            "Latency analysis request received",
            model_path=request.model_path
        )
        
        if not Path(request.model_path).exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
        
        # Initialize components
        optimizer = EnglishModelOptimizer()
        latency_analyzer = LatencyOptimization()
        
        # Load model
        model = optimizer._load_model(request.model_path, "wav2vec2")
        
        # Create dummy input
        import torch
        dummy_input = torch.randn(*request.input_shape)
        
        # Analyze latency
        bottlenecks = await latency_analyzer.identify_bottlenecks(model, dummy_input)
        
        # Calculate total latency
        stages = await latency_analyzer.profile_latency_breakdown(model, dummy_input)
        total_latency = sum(stages.values())
        
        logger.info(
            "Latency analysis completed",
            bottlenecks=len(bottlenecks),
            total_latency_ms=total_latency * 1000
        )
        
        return {
            "bottlenecks": [
                {
                    "stage": b["stage"],
                    "latency_ms": b["latency_ms"] * 1000,
                    "percentage": b["percentage"]
                }
                for b in bottlenecks
            ],
            "recommendations": latency_analyzer._generate_recommendations(bottlenecks),
            "total_latency_ms": total_latency * 1000
        }
        
    except Exception as e:
        logger.error("Latency analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Latency analysis failed: {str(e)}")


# ==================== COMPARISON ENDPOINTS ====================

@router.get("/compare/{model_id}")
async def compare_models(model_id: str) -> Dict[str, Any]:
    """
    Compare original and optimized model metrics.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Comparison of original vs optimized metrics
    """
    try:
        logger.info("Model comparison request", model_id=model_id)
        
        # TODO: Load metadata from optimization results
        # Compare original vs best optimized
        
        return {
            "model_id": model_id,
            "original": {
                "size_mb": 0,
                "latency_ms": 0,
                "memory_mb": 0
            },
            "optimized": {
                "size_mb": 0,
                "latency_ms": 0,
                "memory_mb": 0
            },
            "improvements": {
                "size_reduction": "0%",
                "latency_speedup": "1.0x",
                "memory_reduction": "0%"
            }
        }
        
    except Exception as e:
        logger.error("Comparison failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# ==================== OPTIMIZATION PROFILES ====================

@router.get("/profiles")
async def get_optimization_profiles() -> Dict[str, Dict]:
    """
    Get available optimization profiles.
    
    Returns:
        Available optimization configurations
    """
    try:
        profiles = {}
        
        for profile_name, config in ENGLISH_OPTIMIZATION_CONFIGS.items():
            profiles[profile_name] = {
                "quantization": config.quantization,
                "onnx_conversion": config.onnx_conversion,
                "torch_script": config.torch_script,
                "model_compression": config.model_compression,
                "inference_time_target": config.inference_time_target,
                "memory_limit_mb": config.memory_limit_mb
            }
        
        logger.info("Profiles retrieved", count=len(profiles))
        return profiles
        
    except Exception as e:
        logger.error("Failed to retrieve profiles", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profiles: {str(e)}")


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check optimizer service health.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "production_optimizer",
        "version": "1.0.0",
        "supported_formats": ["pytorch", "onnx", "torchscript"],
        "supported_optimizations": [
            "quantization",
            "onnx_conversion",
            "pruning",
            "distillation",
            "torchscript"
        ]
    }


@router.get("/available-models")
async def get_available_models() -> Dict[str, Any]:
    """
    Get list of available models that can be optimized.
    
    Returns:
        List of available model directories
    """
    try:
        models_dir = Path("./models")
        if not models_dir.exists():
            return {
                "success": False,
                "error": "Models directory not found at ./models",
                "models": []
            }
        
        models = [
            {
                "name": d.name,
                "path": str(d),
                "type": "pytorch"  # Default type
            }
            for d in models_dir.iterdir() 
            if d.is_dir()
        ]
        
        logger.info(f"Available models: {len(models)}")
        
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return {
            "success": False,
            "error": str(e),
            "models": []
        }

