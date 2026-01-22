"""Redis caching layer for FraudGuard API.

Provides low-latency caching for repeated predictions,
reducing inference load and improving response times.

Example:
    >>> cache = PredictionCache()
    >>> cache.set("TX123", {"is_fraud": True, "prob": 0.87})
    >>> result = cache.get("TX123")
"""

import hashlib
import json
from functools import lru_cache
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionCache:
    """Redis-based prediction cache.

    Features:
    - TTL-based expiration
    - Feature-based cache keys
    - JSON serialization
    - Graceful degradation if Redis unavailable

    Example:
        >>> cache = PredictionCache(host="localhost", port=6379)
        >>> cache.set("TX123", {"is_fraud": True})
        >>> result = cache.get("TX123")
    """

    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        ttl_seconds: int = 3600,
        prefix: str = "fraudguard:",
    ):
        """Initialize cache connection.

        Args:
            host: Redis host
            port: Redis port
            ttl_seconds: Default TTL for cached items
            prefix: Key prefix for namespacing
        """
        self.ttl = ttl_seconds
        self.prefix = prefix
        self._client = None

        try:
            import redis

            self._client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            self._client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Caching disabled.")
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def _make_key(self, transaction_id: str) -> str:
        """Create cache key from transaction ID."""
        return f"{self.prefix}pred:{transaction_id}"

    def _hash_features(self, features: Dict[str, Any]) -> str:
        """Create hash of features for deduplication."""
        sorted_features = json.dumps(features, sort_keys=True)
        return hashlib.md5(sorted_features.encode()).hexdigest()[:12]

    def get(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction.

        Args:
            transaction_id: Transaction identifier

        Returns:
            Cached prediction dict or None if not found
        """
        if not self._client:
            return None

        try:
            key = self._make_key(transaction_id)
            cached = self._client.get(key)
            if cached:
                logger.debug(f"Cache hit: {transaction_id}")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(
        self,
        transaction_id: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a prediction.

        Args:
            transaction_id: Transaction identifier
            prediction: Prediction result to cache
            ttl: Optional custom TTL in seconds

        Returns:
            True if cached successfully
        """
        if not self._client:
            return False

        try:
            key = self._make_key(transaction_id)
            value = json.dumps(prediction)
            self._client.setex(key, ttl or self.ttl, value)
            logger.debug(f"Cached: {transaction_id}")
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    def delete(self, transaction_id: str) -> bool:
        """Delete cached prediction.

        Args:
            transaction_id: Transaction identifier

        Returns:
            True if deleted successfully
        """
        if not self._client:
            return False

        try:
            key = self._make_key(transaction_id)
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")
            return False

    def clear_all(self) -> int:
        """Clear all cached predictions.

        Returns:
            Number of keys deleted
        """
        if not self._client:
            return 0

        try:
            pattern = f"{self.prefix}*"
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self._client:
            return {"status": "disconnected"}

        try:
            info = self._client.info("stats")
            pattern = f"{self.prefix}*"
            key_count = len(self._client.keys(pattern))

            return {
                "status": "connected",
                "total_keys": key_count,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0)
                / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


@lru_cache(maxsize=1)
def get_prediction_cache() -> PredictionCache:
    """Get singleton cache instance."""
    import os

    return PredictionCache(
        host=os.environ.get("REDIS_HOST", "redis"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
    )
