"""
Cache management endpoints for Redis operations.
Provides monitoring, debugging, and cache control.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional
from app.core.logging import get_logger
from app.core.cache import get_cache, CacheKeys

logger = get_logger()
router = APIRouter(prefix="/cache", tags=["cache"])


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    status: str
    used_memory_mb: Optional[float] = None
    connected_clients: Optional[int] = None
    total_commands_processed: Optional[int] = None
    keyspace_hits: Optional[int] = None
    keyspace_misses: Optional[int] = None
    redis_version: Optional[str] = None
    hit_rate: Optional[float] = None


class CacheInvalidateResponse(BaseModel):
    """Cache invalidation response."""
    success: bool
    keys_deleted: int
    pattern: str


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get cache statistics and performance metrics.
    
    Returns:
        - Memory usage
        - Connected clients
        - Hit/miss rates
        - Redis version
    """
    try:
        cache = get_cache()
        stats = cache.get_stats()
        
        # Calculate hit rate if available
        hits = stats.get("keyspace_hits", 0)
        misses = stats.get("keyspace_misses", 0)
        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
        
        return CacheStatsResponse(
            status=stats.get("status", "unknown"),
            used_memory_mb=stats.get("used_memory_mb", 0),
            connected_clients=stats.get("connected_clients", 0),
            total_commands_processed=stats.get("total_commands_processed", 0),
            keyspace_hits=stats.get("keyspace_hits", 0),
            keyspace_misses=stats.get("keyspace_misses", 0),
            redis_version=stats.get("redis_version", "unknown"),
            hit_rate=hit_rate
        )
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.post("/invalidate")
async def invalidate_user_cache(user_id: str = Query(...)):
    """
    Invalidate all cache entries for a specific user.
    
    Args:
        user_id: User ID to invalidate cache for
        
    Returns:
        Number of keys deleted
    """
    try:
        cache = get_cache()
        
        # Delete all scores for user
        scores_deleted = cache.delete_pattern(CacheKeys.user_scores_pattern(user_id))
        
        # Delete all feedback for user
        feedback_deleted = cache.delete_pattern(CacheKeys.user_feedback_pattern(user_id))
        
        total_deleted = scores_deleted + feedback_deleted
        
        logger.info(
            "User cache invalidated",
            user_id=user_id,
            scores_deleted=scores_deleted,
            feedback_deleted=feedback_deleted,
            total_deleted=total_deleted
        )
        
        return CacheInvalidateResponse(
            success=True,
            keys_deleted=total_deleted,
            pattern=f"user:{user_id}:*"
        )
    except Exception as e:
        logger.error("Failed to invalidate user cache", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")


@router.delete("/clear")
async def clear_all_cache():
    """
    Clear entire cache (development/testing only).
    
    WARNING: This will delete all cached data!
    """
    try:
        cache = get_cache()
        success = cache.clear_all()
        
        if success:
            logger.warning("Cache cleared completely")
            return {"success": True, "message": "Cache cleared"}
        else:
            raise Exception("Failed to clear cache")
            
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/health")
async def cache_health():
    """
    Check Redis cache health.
    
    Returns:
        - Available: True if Redis is accessible
        - Status: Connection status
    """
    try:
        cache = get_cache()
        available = cache.is_available()
        
        return {
            "available": available,
            "status": "healthy" if available else "unavailable",
            "service": "redis_cache"
        }
    except Exception as e:
        logger.error("Cache health check failed", error=str(e))
        return {
            "available": False,
            "status": "error",
            "error": str(e)
        }
