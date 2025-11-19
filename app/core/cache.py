"""
Redis caching service for KataDia AI-ML application.
Provides caching layer for transcriptions, scoring results, and feedback.
"""

import json
import hashlib
from typing import Any, Optional, Dict
from datetime import timedelta
import redis
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


class RedisCache:
    """Redis cache wrapper with TTL and key pattern management."""
    
    def __init__(self):
        """Initialize Redis connection from settings."""
        try:
            # Parse Redis URL: redis://localhost:6379/0
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established", url=settings.REDIS_URL)
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if self.redis_client is None:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning("Redis unavailable", error=str(e))
            return False
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time to live in seconds (default 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Convert value to JSON
            serialized = json.dumps(value) if not isinstance(value, str) else value
            
            # Set with expiration
            self.redis_client.setex(
                key,
                ttl_seconds,
                serialized
            )
            
            logger.debug("Cache SET", key=key, ttl_seconds=ttl_seconds)
            return True
            
        except Exception as e:
            logger.warning("Cache SET failed", key=key, error=str(e))
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_available():
            return None
        
        try:
            value = self.redis_client.get(key)
            
            if value is None:
                logger.debug("Cache MISS", key=key)
                return None
            
            # Try to parse as JSON
            try:
                result = json.loads(value)
                logger.debug("Cache HIT", key=key)
                return result
            except json.JSONDecodeError:
                # Return as is if not JSON
                logger.debug("Cache HIT (raw)", key=key)
                return value
                
        except Exception as e:
            logger.warning("Cache GET failed", key=key, error=str(e))
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            self.redis_client.delete(key)
            logger.debug("Cache DELETE", key=key)
            return True
        except Exception as e:
            logger.warning("Cache DELETE failed", key=key, error=str(e))
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "score:user_123:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.debug("Cache DELETE_PATTERN", pattern=pattern, deleted=deleted)
                return deleted
            return 0
        except Exception as e:
            logger.warning("Cache DELETE_PATTERN failed", pattern=pattern, error=str(e))
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_available():
            return False
        
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.warning("Cache EXISTS check failed", key=key, error=str(e))
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get TTL in seconds for key (-1 if no expiry, -2 if not exists)."""
        if not self.is_available():
            return -2
        
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            logger.warning("Cache TTL check failed", key=key, error=str(e))
            return -2
    
    def clear_all(self) -> bool:
        """Clear all cache (use with caution)."""
        if not self.is_available():
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("Cache CLEAR_ALL executed")
            return True
        except Exception as e:
            logger.error("Cache CLEAR_ALL failed", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_available():
            return {"status": "unavailable"}
        
        try:
            info = self.redis_client.info()
            return {
                "status": "available",
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "redis_version": info.get("redis_version", "unknown")
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"status": "error", "error": str(e)}


# Key pattern constants
class CacheKeys:
    """Cache key patterns and factory methods."""
    
    TRANSCRIPTION_TTL = 86400  # 24 hours
    SCORING_TTL = 604800  # 7 days
    FEEDBACK_TTL = 604800  # 7 days
    
    @staticmethod
    def transcription_key(audio_hash: str, language: str) -> str:
        """Generate transcription cache key."""
        return f"transcription:{language}:{audio_hash}"
    
    @staticmethod
    def scoring_key(user_id: str, transcript_hash: str, language: str) -> str:
        """Generate scoring cache key."""
        return f"score:{user_id}:{language}:{transcript_hash}"
    
    @staticmethod
    def feedback_key(user_id: str, feedback_hash: str) -> str:
        """Generate feedback cache key."""
        return f"feedback:{user_id}:{feedback_hash}"
    
    @staticmethod
    def user_scores_pattern(user_id: str) -> str:
        """Get pattern for all scores by user."""
        return f"score:{user_id}:*"
    
    @staticmethod
    def user_feedback_pattern(user_id: str) -> str:
        """Get pattern for all feedback by user."""
        return f"feedback:{user_id}:*"


def generate_hash(data: Any) -> str:
    """Generate SHA256 hash for data (for cache keys)."""
    data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get or create global Redis cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance
