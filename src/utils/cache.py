"""
Redis cache utilities.
"""
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import redis
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# Redis connection parameters
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Cache TTL in seconds
DEFAULT_TTL = 3600  # 1 hour

# Type variable for generic return type
T = TypeVar("T")

class RedisCache:
    """Redis cache client."""
    
    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        password: Optional[str] = REDIS_PASSWORD,
        prefix: str = "inventory:",
    ):
        """Initialize Redis cache client.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            prefix: Key prefix
        """
        self.prefix = prefix
        self.client = None
        self.enabled = True
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Redis cache disabled")
            self.enabled = False
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        if not self.enabled or not self.client:
            return None
            
        try:
            full_key = f"{self.prefix}{key}"
            value = self.client.get(full_key)
            
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
            
    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
            
        try:
            full_key = f"{self.prefix}{key}"
            serialized = json.dumps(value)
            return bool(self.client.setex(full_key, ttl, serialized))
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
            
        try:
            full_key = f"{self.prefix}{key}"
            return bool(self.client.delete(full_key))
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
            
    def flush(self, pattern: str = "*") -> bool:
        """Flush cache keys matching pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
            
        try:
            full_pattern = f"{self.prefix}{pattern}"
            keys = self.client.keys(full_pattern)
            
            if keys:
                return bool(self.client.delete(*keys))
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
            
# Create a global cache instance
cache = RedisCache()

def cached(key_prefix: str, ttl: int = DEFAULT_TTL):
    """Decorator to cache function results.
    
    Args:
        key_prefix: Prefix for cache key
        ttl: Time to live in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            key_parts = [key_prefix]
            
            # Add args to key
            for arg in args:
                if hasattr(arg, "__dict__"):
                    # For objects, use their string representation
                    key_parts.append(str(arg))
                else:
                    # For simple types, use their value
                    key_parts.append(str(arg))
                    
            # Add kwargs to key
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
                
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cast(T, cached_value)
                
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
    
def setup_fastapi_cache(app: Any) -> None:
    """Set up FastAPI cache middleware.
    
    Args:
        app: FastAPI application
    """
    @app.middleware("http")
    async def cache_middleware(request: Request, call_next: Callable) -> Response:
        """Cache middleware for FastAPI.
        
        Args:
            request: Request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response: Response from next middleware or endpoint
        """
        # Skip non-GET requests
        if request.method != "GET":
            return await call_next(request)
            
        # Skip requests with query params that should bypass cache
        skip_cache = request.query_params.get("skip_cache", "").lower() == "true"
        if skip_cache:
            return await call_next(request)
            
        # Generate cache key
        cache_key = f"api:{request.url.path}:{sorted(dict(request.query_params).items())}"
        
        # Try to get from cache
        cached_response = cache.get(cache_key)
        if cached_response:
            # Create response from cached data
            content_type = cached_response.get("content_type", "application/json")
            return Response(
                content=cached_response.get("content", ""),
                status_code=cached_response.get("status_code", 200),
                headers=cached_response.get("headers", {}),
                media_type=content_type,
            )
            
        # Call next middleware or endpoint
        response = await call_next(request)
        
        # Cache response if status code is 200 OK
        if response.status_code == 200:
            # Get response content
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
                
            # Create response object for caching
            headers = dict(response.headers)
            
            cached_data = {
                "content": response_body.decode(),
                "status_code": response.status_code,
                "headers": headers,
                "content_type": response.media_type,
            }
            
            # Cache response
            cache.set(cache_key, cached_data)
            
            # Return new response with the content
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
            
        return response 