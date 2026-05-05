"""Caching layer for LLM responses to reduce costs and improve speed."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LLMCache:
    """Cache for LLM responses.
    
    Supports multiple backends: memory, file system, and Redis.
    
    Args:
        backend: Cache backend ("memory", "file", "redis")
        ttl: Time-to-live in seconds (None = no expiration)
        cache_dir: Directory for file-based cache
        redis_url: Redis connection URL
    
    Example:
        >>> cache = LLMCache(backend="file", ttl=3600)
        >>> cache.set("key", "response")
        >>> response = cache.get("key")
    """
    
    def __init__(
        self,
        backend: str = "memory",
        ttl: Optional[int] = None,
        cache_dir: str = ".cache/rag_eval",
        redis_url: Optional[str] = None
    ):
        """Initialize cache.
        
        Args:
            backend: Cache backend type
            ttl: Time-to-live in seconds
            cache_dir: Directory for file cache
            redis_url: Redis connection URL
        """
        self.backend = backend
        self.ttl = ttl
        self.cache_dir = Path(cache_dir)
        self.redis_url = redis_url
        
        # Initialize backend
        if backend == "memory":
            self._cache: Dict[str, tuple[Any, Optional[datetime]]] = {}
        elif backend == "file":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        elif backend == "redis":
            self._init_redis()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        logger.info(f"Initialized LLMCache with backend: {backend}")
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            import redis
            self._redis = redis.from_url(
                self.redis_url or "redis://localhost:6379/0"
            )
            # Test connection
            self._redis.ping()
            logger.info("Connected to Redis")
        except ImportError:
            raise ImportError("redis package required. Install with: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if self.backend == "memory":
            return self._get_memory(key)
        elif self.backend == "file":
            return self._get_file(key)
        elif self.backend == "redis":
            return self._get_redis(key)
    
    def set(self, key: str, value: str) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if self.backend == "memory":
            self._set_memory(key, value)
        elif self.backend == "file":
            self._set_file(key, value)
        elif self.backend == "redis":
            self._set_redis(key, value)
    
    def _get_memory(self, key: str) -> Optional[str]:
        """Get from memory cache."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if expiry is None or datetime.utcnow() < expiry:
                logger.debug(f"Cache hit (memory): {key[:20]}...")
                return value
            else:
                # Expired
                del self._cache[key]
                logger.debug(f"Cache expired (memory): {key[:20]}...")
        return None
    
    def _set_memory(self, key: str, value: str) -> None:
        """Set in memory cache."""
        expiry = None
        if self.ttl:
            expiry = datetime.utcnow() + timedelta(seconds=self.ttl)
        self._cache[key] = (value, expiry)
        logger.debug(f"Cache set (memory): {key[:20]}...")
    
    def _get_file(self, key: str) -> Optional[str]:
        """Get from file cache."""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                
                value, expiry = data["value"], data.get("expiry")
                
                if expiry is None or datetime.utcnow() < expiry:
                    logger.debug(f"Cache hit (file): {key[:20]}...")
                    return value
                else:
                    # Expired
                    cache_file.unlink()
                    logger.debug(f"Cache expired (file): {key[:20]}...")
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        return None
    
    def _set_file(self, key: str, value: str) -> None:
        """Set in file cache."""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        expiry = None
        if self.ttl:
            expiry = datetime.utcnow() + timedelta(seconds=self.ttl)
        
        data = {"value": value, "expiry": expiry}
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Cache set (file): {key[:20]}...")
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")
    
    def _get_redis(self, key: str) -> Optional[str]:
        """Get from Redis cache."""
        try:
            value = self._redis.get(key)
            if value:
                logger.debug(f"Cache hit (redis): {key[:20]}...")
                return value.decode("utf-8")
        except Exception as e:
            logger.warning(f"Error reading from Redis: {e}")
        return None
    
    def _set_redis(self, key: str, value: str) -> None:
        """Set in Redis cache."""
        try:
            if self.ttl:
                self._redis.setex(key, self.ttl, value)
            else:
                self._redis.set(key, value)
            logger.debug(f"Cache set (redis): {key[:20]}...")
        except Exception as e:
            logger.warning(f"Error writing to Redis: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if self.backend == "memory":
            self._cache.clear()
            logger.info("Memory cache cleared")
        elif self.backend == "file":
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            logger.info("File cache cleared")
        elif self.backend == "redis":
            self._redis.flushdb()
            logger.info("Redis cache cleared")
    
    def size(self) -> int:
        """Get number of cached entries.
        
        Returns:
            Number of cache entries
        """
        if self.backend == "memory":
            return len(self._cache)
        elif self.backend == "file":
            return len(list(self.cache_dir.glob("*.cache")))
        elif self.backend == "redis":
            return self._redis.dbsize()
        return 0
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash key for file names.
        
        Args:
            key: Original key
            
        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    @staticmethod
    def create_cache_key(prompt: str, model: str, temperature: float) -> str:
        """Create cache key from LLM parameters.
        
        Args:
            prompt: LLM prompt
            model: Model name
            temperature: Temperature parameter
            
        Returns:
            Cache key
        """
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


class CachedLLMMetric:
    """Wrapper to add caching to LLM-based metrics.
    
    Args:
        metric: Base LLM metric
        cache: LLM cache instance
    
    Example:
        >>> from rag_eval.metrics.llm_metrics import FaithfulnessMetric
        >>> cache = LLMCache(backend="file", ttl=3600)
        >>> metric = FaithfulnessMetric()
        >>> cached_metric = CachedLLMMetric(metric, cache)
    """
    
    def __init__(self, metric: Any, cache: LLMCache):
        """Initialize cached metric.
        
        Args:
            metric: Base metric to wrap
            cache: Cache instance
        """
        self.metric = metric
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    def compute(self, sample: Any) -> Any:
        """Compute metric with caching.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            Metric result
        """
        # Create cache key from sample
        cache_key = self._create_sample_key(sample)
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.cache_hits += 1
            logger.debug(f"Cache hit for {self.metric.name}")
            # Deserialize result
            from rag_eval.dataset.schema import MetricResult
            result_dict = json.loads(cached_result)
            return MetricResult(**result_dict)
        
        # Cache miss - compute metric
        self.cache_misses += 1
        logger.debug(f"Cache miss for {self.metric.name}")
        result = self.metric.compute(sample)
        
        # Cache result
        if result.score is not None:
            result_json = json.dumps(result.model_dump(), default=str)
            self.cache.set(cache_key, result_json)
        
        return result
    
    def _create_sample_key(self, sample: Any) -> str:
        """Create cache key from sample.
        
        Args:
            sample: Evaluation sample
            
        Returns:
            Cache key
        """
        key_data = {
            "metric": self.metric.name,
            "query": sample.query,
            "answer": sample.generated_answer,
            "contexts": sample.retrieved_docs,
            "ground_truth": sample.ground_truth
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache hits and misses
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": hit_rate
        }
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped metric."""
        return getattr(self.metric, name)
