"""
Locust load test for API rate limiting and circuit breaker.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
"""
import json
import logging
import random
import time
from typing import Dict, List, Optional, Union

from locust import HttpUser, TaskSet, between, events, task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("locust")


# Create a listener for failures that might be due to rate limiting
@events.request_failure.add_listener
def handle_request_failure(request_type, name, response_time, exception, **kwargs):
    """Log possible rate limiting responses."""
    if hasattr(exception, "response") and exception.response is not None:
        if exception.response.status_code == 429:
            retry_after = exception.response.headers.get("Retry-After", "unknown")
            logger.warning(
                f"Rate limit exceeded: {name}, Retry-After: {retry_after}"
            )


class InventoryAPIUser(HttpUser):
    """Simulated user for inventory API load testing."""
    
    # Wait between 1-5 seconds between tasks
    wait_time = between(1, 5)
    
    # Store some state between tasks
    product_ids = []
    location_ids = []
    
    def on_start(self):
        """Initialize user state by fetching some IDs."""
        # Try to login if authentication is enabled
        try:
            response = self.client.post(
                "/api/auth/login",
                json={"username": "testuser", "password": "testpassword"},
            )
            if response.status_code == 200:
                token = response.json().get("access_token")
                self.client.headers.update({"Authorization": f"Bearer {token}"})
                logger.info("Successfully logged in")
            else:
                logger.warning("Login failed, proceeding without authentication")
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
        
        # Load some product IDs
        try:
            response = self.client.get("/api/products?limit=100")
            if response.status_code == 200:
                data = response.json()
                self.product_ids = [product["id"] for product in data.get("items", [])]
                logger.info(f"Loaded {len(self.product_ids)} product IDs")
            else:
                logger.warning("Failed to load product IDs")
        except Exception as e:
            logger.error(f"Error loading product IDs: {str(e)}")
        
        # Load some location IDs
        try:
            response = self.client.get("/api/locations?limit=50")
            if response.status_code == 200:
                data = response.json()
                self.location_ids = [location["id"] for location in data.get("items", [])]
                logger.info(f"Loaded {len(self.location_ids)} location IDs")
            else:
                logger.warning("Failed to load location IDs")
        except Exception as e:
            logger.error(f"Error loading location IDs: {str(e)}")
    
    @task(10)
    def get_products(self):
        """Get product list with pagination and filtering."""
        params = {
            "limit": random.choice([10, 20, 50, 100]),
            "offset": random.choice([0, 10, 20, 50]),
        }
        
        # Add random filter 30% of the time
        if random.random() < 0.3:
            category = random.choice(["electronics", "clothing", "food", "beverage", "household"])
            params["category"] = category
        
        self.client.get("/api/products", params=params, name="/api/products")
    
    @task(5)
    def get_product_detail(self):
        """Get details for a specific product."""
        if not self.product_ids:
            return
        
        product_id = random.choice(self.product_ids)
        self.client.get(f"/api/products/{product_id}", name="/api/products/{id}")
    
    @task(3)
    def get_inventory_by_location(self):
        """Get inventory for a specific location."""
        if not self.location_ids:
            return
        
        location_id = random.choice(self.location_ids)
        self.client.get(
            f"/api/inventory/location/{location_id}",
            name="/api/inventory/location/{id}"
        )
    
    @task(2)
    def get_forecasts(self):
        """Get forecast list."""
        params = {
            "limit": random.choice([10, 20, 50]),
            "offset": random.choice([0, 10, 20]),
        }
        self.client.get("/api/forecasts", params=params, name="/api/forecasts")
    
    @task(1)
    def generate_forecast(self):
        """Request a new forecast generation."""
        if not self.product_ids:
            return
        
        product_id = random.choice(self.product_ids)
        data = {
            "product_id": product_id,
            "horizon": random.choice([7, 14, 30, 90]),
            "method": random.choice(["auto", "sarima", "prophet", "exponential_smoothing"]),
        }
        
        self.client.post(
            "/api/forecasts/generate",
            json=data,
            name="/api/forecasts/generate"
        )
    
    @task(1)
    def get_dashboard(self):
        """Get dashboard data."""
        self.client.get("/api/dashboard", name="/api/dashboard")


class CircuitBreakerTest(HttpUser):
    """User that specifically tests the circuit breaker pattern."""
    
    wait_time = between(0.1, 1)  # Aggressive timing to trigger circuit breaker
    
    @task(10)
    def call_external_service(self):
        """Call endpoint that uses circuit breaker to access external service."""
        self.client.get(
            "/api/external/weather",
            name="/api/external/weather (circuit breaker test)"
        )
    
    @task(5)
    def call_external_service_with_retry(self):
        """Call endpoint with retry mechanism."""
        self.client.get(
            "/api/external/market-data",
            name="/api/external/market-data (with retry)"
        )


class RateLimitTest(HttpUser):
    """User that aggressively tests rate limits."""
    
    wait_time = between(0.05, 0.1)  # Very aggressive timing to trigger rate limiting
    
    @task(10)
    def login_attempt(self):
        """Repeatedly attempt login to trigger rate limiting."""
        username = f"user{random.randint(1, 1000)}"
        password = f"password{random.randint(1, 1000)}"
        
        self.client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
            name="/api/auth/login (rate limit test)"
        )
    
    @task(5)
    def rapid_api_calls(self):
        """Make rapid API calls to test general API rate limiting."""
        endpoint = random.choice([
            "/api/products",
            "/api/locations",
            "/api/inventory",
            "/api/forecasts",
        ])
        
        self.client.get(
            endpoint,
            name=f"{endpoint} (rate limit test)"
        ) 