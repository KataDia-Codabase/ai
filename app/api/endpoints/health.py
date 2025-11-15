from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.logging import get_logger
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
    # This will be expanded in later sprints to include:
    # - Database connectivity
    # - ML model status
    # - External API status
    # - Performance metrics
    
    health_status = {
        "status": "healthy",
        "service": "katadia-ai-ml",
        "version": "1.0.0",
        "components": {
            "database": "not_implemented",
            "stt_service": "not_implemented", 
            "ml_models": "not_implemented",
            "cache": "not_implemented"
        }
    }
    
    logger.info("Detailed health check requested", health_status=health_status)
    
    return health_status
