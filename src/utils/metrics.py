"""
Prometheus metrics utilities.
"""
import time
import asyncio
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

# Define metrics
API_REQUESTS = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

API_REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0, float("inf")),
)

FORECAST_PROCESSING_TIME = Histogram(
    "forecast_processing_time_seconds",
    "Time taken to generate forecasts",
    ["product_id", "model_type"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")),
)

FORECAST_ACCURACY = Gauge(
    "forecast_accuracy",
    "Accuracy of forecasts",
    ["product_id", "model_type", "metric"],
)

INVENTORY_LEVELS = Gauge(
    "inventory_levels",
    "Current inventory levels",
    ["product_id", "location"],
)

INVENTORY_ALERTS = Counter(
    "inventory_alerts_total",
    "Total number of inventory alerts",
    ["product_id", "alert_type"],
)

INVENTORY_METRICS = Gauge(
    "inventory_metrics",
    "Inventory optimization metrics",
    ["product_id", "metric_type"],
)

TASK_STARTED = Counter(
    "task_started_total",
    "Total number of tasks started",
    ["task_name"],
)

TASK_COMPLETED = Counter(
    "task_completed_total",
    "Total number of tasks completed successfully",
    ["task_name"],
)

TASK_FAILED = Counter(
    "task_failed_total",
    "Total number of tasks that failed",
    ["task_name"],
)

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API request metrics."""
    
    def __init__(self, app: FastAPI):
        """
        Initialize metrics middleware
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.request_counts = {}
        self.response_times = {}
        self.logger = logging.getLogger(__name__ + ".MetricsMiddleware")
        self.logger.info("Metrics middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, collecting metrics
        
        Args:
            request: The incoming request
            call_next: The next middleware or endpoint handler
            
        Returns:
            The response
        """
        # Record start time
        start_time = time.time()
        
        # Continue with request processing
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # In a real implementation, we would export these metrics to Prometheus
        # For this simplified implementation, we'll just log them
        self.logger.info(
            "Request to %s completed in %.3f seconds with status %d",
            request.url.path,
            duration,
            response.status_code
        )
        
        # Update internal metrics (these would normally be exported)
        path = request.url.path
        method = request.method
        status = response.status_code
        
        # Update request count
        key = f"{method}:{path}"
        self.request_counts[key] = self.request_counts.get(key, 0) + 1
        
        # Update response times
        status_key = f"{method}:{path}:{status}"
        if status_key not in self.response_times:
            self.response_times[status_key] = []
        self.response_times[status_key].append(duration)
        
        # Keep only the last 100 response times to avoid memory issues
        if len(self.response_times[status_key]) > 100:
            self.response_times[status_key] = self.response_times[status_key][-100:]
        
        return response

def record_forecast_metrics(product_id: str, model_type: str, rmse: float, mae: float, r2: float, processing_time: float) -> None:
    """
    Record forecast metrics.
    
    Args:
        product_id: Product ID
        model_type: Model type used for forecasting
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2: R-squared value
        processing_time: Processing time in seconds
    """
    FORECAST_ACCURACY.labels(product_id=product_id, model_type=model_type, metric="rmse").set(rmse)
    FORECAST_ACCURACY.labels(product_id=product_id, model_type=model_type, metric="mae").set(mae)
    FORECAST_ACCURACY.labels(product_id=product_id, model_type=model_type, metric="r2").set(r2)
    FORECAST_PROCESSING_TIME.labels(product_id=product_id, model_type=model_type).observe(processing_time)

def record_inventory_metrics(product_id: str, safety_stock: float, reorder_point: float, eoq: float, service_level: float) -> None:
    """
    Record inventory optimization metrics.
    
    Args:
        product_id: Product ID
        safety_stock: Safety stock level
        reorder_point: Reorder point
        eoq: Economic order quantity
        service_level: Target service level
    """
    INVENTORY_METRICS.labels(product_id=product_id, metric_type="safety_stock").set(safety_stock)
    INVENTORY_METRICS.labels(product_id=product_id, metric_type="reorder_point").set(reorder_point)
    INVENTORY_METRICS.labels(product_id=product_id, metric_type="eoq").set(eoq)
    INVENTORY_METRICS.labels(product_id=product_id, metric_type="service_level").set(service_level)

def record_inventory_level(product_id: str, location: str, quantity: float) -> None:
    """
    Record inventory level for a product at a location.
    
    Args:
        product_id: Product ID
        location: Location identifier
        quantity: Current inventory quantity
    """
    INVENTORY_LEVELS.labels(product_id=product_id, location=location).set(quantity)

def record_inventory_alert(product_id: str, alert_type: str) -> None:
    """
    Record inventory alert.
    
    Args:
        product_id: Product ID
        alert_type: Type of alert (e.g., "stockout", "low_stock")
    """
    INVENTORY_ALERTS.labels(product_id=product_id, alert_type=alert_type).inc()

# Endpoint to expose Prometheus metrics
def metrics_endpoint() -> Response:
    """
    Endpoint to expose Prometheus metrics.
    
    Returns:
        Response: Metrics in Prometheus format
    """
    return Response(content=generate_latest(), media_type="text/plain")

# Register middleware and endpoints
def add_metrics(app: FastAPI) -> None:
    """
    Add metrics middleware to FastAPI app
    
    Args:
        app: FastAPI application
    """
    app.add_middleware(MetricsMiddleware)
    logger.info("Metrics middleware added to application")

# Decorator to measure function execution time
def time_execution(name: str, labels: Dict[str, str] = None):
    """
    Decorator to measure function execution time.
    
    Args:
        name: Metric name
        labels: Metric labels
        
    Returns:
        Callable: Decorated function
    """
    if labels is None:
        labels = {}
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            API_REQUEST_DURATION.labels(method="func", endpoint=name).observe(duration)
            
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            API_REQUEST_DURATION.labels(method="func", endpoint=name).observe(duration)
            
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator 