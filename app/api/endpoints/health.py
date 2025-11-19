from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.logging import get_logger
from app.ml.services.optimized_model_loader import get_loader_info
import structlog

logger = get_logger()
router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "katadia-ai-ml",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check including ML components."""
    model_info = get_loader_info()
    
    health_status = {
        "status": "healthy",
        "service": "katadia-ai-ml",
        "version": "1.0.0",
        "components": {
            "database": "not_implemented",
            "stt_service": "not_implemented", 
            "ml_models": "not_implemented",
            "cache": "not_implemented"
        },
        "model_optimization": model_info
    }
    
    logger.info("Detailed health check requested", health_status=health_status)
    
    return health_status

@router.get("/health/models")
async def model_status():
    """Check model optimization and availability status."""
    model_info = get_loader_info()
    
    return {
        "status": "active",
        "base_model_path": model_info.get("base_model_path", "not_initialized"),
        "optimization_directory": model_info.get("optimization_dir", "not_initialized"),
        "device": model_info.get("device", "cpu"),
        "available_optimizations": list(model_info.get("metadata", {}).get("models", {}).keys()),
        "metadata": model_info.get("metadata", {}),
        "message": "Optimized models are automatically loaded when available"
    }
