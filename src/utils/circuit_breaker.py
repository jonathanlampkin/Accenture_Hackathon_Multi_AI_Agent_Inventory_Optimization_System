"""
Circuit breaker pattern implementation for external service calls.

This module provides a circuit breaker pattern implementation that helps
prevent cascading failures and provides resilience to the system by
temporarily disabling calls to failing external services.
"""
import asyncio
import functools
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Circuit is closed, requests are allowed through
    OPEN = "open"  # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Circuit is testing if the service has recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit is open."""

    def __init__(self, service_name: str, failure_count: int, recovery_time: float):
        """Initialize CircuitBreakerError.
        
        Args:
            service_name: Name of the failing service
            failure_count: Number of consecutive failures
            recovery_time: Time in seconds until next recovery attempt
        """
        self.service_name = service_name
        self.failure_count = failure_count
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit breaker for {service_name} is open "
            f"after {failure_count} consecutive failures. "
            f"Next recovery attempt in {recovery_time:.2f} seconds."
        )


class CircuitBreaker:
    """Circuit breaker for handling external service calls.
    
    Implements the circuit breaker pattern to prevent cascading failures
    when an external service is unavailable or experiencing issues.
    """

    # Class-level registry of circuit breakers
    _registry: Dict[str, "CircuitBreaker"] = {}

    @classmethod
    def get_or_create(cls, name: str, **kwargs) -> "CircuitBreaker":
        """Get an existing circuit breaker or create a new one.
        
        Args:
            name: Name of the circuit breaker
            **kwargs: Additional arguments for the circuit breaker constructor
            
        Returns:
            CircuitBreaker: The circuit breaker instance
        """
        if name not in cls._registry:
            cls._registry[name] = CircuitBreaker(name, **kwargs)
        return cls._registry[name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers to closed state."""
        for circuit in cls._registry.values():
            circuit.reset()

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        exception_types: Optional[Union[Type[Exception], tuple]] = None,
    ):
        """Initialize CircuitBreaker.
        
        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Time in seconds to wait before trying recovery
            half_open_max_calls: Maximum number of test calls in half-open state
            exception_types: Exception types to count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        # Default to all exceptions if not specified
        self.exception_types = exception_types or Exception

        # State variables
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock() if asyncio.get_event_loop_policy().get_event_loop().is_running() else None

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' has been reset")

    def get_state(self) -> CircuitState:
        """Get the current state of the circuit breaker.
        
        Returns:
            CircuitState: Current state
        """
        # Auto-transition from OPEN to HALF_OPEN after recovery_timeout
        if (
            self._state == CircuitState.OPEN
            and time.time() - self._last_failure_time >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.info(
                f"Circuit breaker '{self.name}' transitioned from OPEN to HALF_OPEN"
            )

        return self._state

    def record_success(self) -> None:
        """Record a successful call and potentially close the circuit."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            
            # If we've had enough successful test calls, close the circuit
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' transitioned from HALF_OPEN to CLOSED"
                )
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' transitioned from CLOSED to OPEN "
                f"after {self._failure_count} consecutive failures"
            )
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' transitioned from HALF_OPEN to OPEN "
                f"after test call failure"
            )

    def _can_execute(self) -> bool:
        """Check if execution is allowed based on current state.
        
        Returns:
            bool: True if execution is allowed, False otherwise
        """
        state = self.get_state()
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN and self._half_open_calls < self.half_open_max_calls:
            return True
        
        return False

    def _handle_error(self, exc: Exception) -> bool:
        """Handle an exception and determine if it counts as a failure.
        
        Args:
            exc: The exception to handle
            
        Returns:
            bool: True if exception should count as a failure, False otherwise
        """
        if isinstance(exc, self.exception_types):
            self.record_failure()
            return True
        return False

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            T: Function return value
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        if not self._can_execute():
            time_remaining = (
                self.recovery_timeout - (time.time() - self._last_failure_time)
            )
            raise CircuitBreakerError(
                self.name, self._failure_count, max(0, time_remaining)
            )

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            if self._handle_error(exc):
                # Re-raise the original exception
                raise
            # If not a tracked exception type, just pass it through
            raise

    async def call_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute an async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function return value
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        if self._lock:
            async with self._lock:
                if not self._can_execute():
                    time_remaining = (
                        self.recovery_timeout - (time.time() - self._last_failure_time)
                    )
                    raise CircuitBreakerError(
                        self.name, self._failure_count, max(0, time_remaining)
                    )
        else:
            if not self._can_execute():
                time_remaining = (
                    self.recovery_timeout - (time.time() - self._last_failure_time)
                )
                raise CircuitBreakerError(
                    self.name, self._failure_count, max(0, time_remaining)
                )

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            if self._handle_error(exc):
                # Re-raise the original exception
                raise
            # If not a tracked exception type, just pass it through
            raise


def circuit_breaker(
    name: str = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_max_calls: int = 1,
    exception_types: Optional[Union[Type[Exception], tuple]] = None,
) -> Callable:
    """Decorator for adding circuit breaker functionality to a function.
    
    Args:
        name: Name of the circuit breaker (defaults to function name)
        failure_threshold: Number of consecutive failures before opening circuit
        recovery_timeout: Time in seconds to wait before trying recovery
        half_open_max_calls: Maximum number of test calls in half-open state
        exception_types: Exception types to count as failures
        
    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable) -> Callable:
        breaker_name = name or func.__qualname__
        cb = CircuitBreaker.get_or_create(
            breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            exception_types=exception_types,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cb.call(func, *args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await cb.call_async(func, *args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator 