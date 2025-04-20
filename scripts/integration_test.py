#!/usr/bin/env python
"""
Integration Test Script for Inventory Optimization System.

This script tests the integration between different components of the system:
- Database connectivity and model validation
- Redis cache functionality
- RabbitMQ message passing
- MLflow tracking
- Authentication and authorization
- API endpoints
- Celery task execution

Example usage:
    python scripts/integration_test.py --all
    python scripts/integration_test.py --database --redis --rabbitmq
    python scripts/integration_test.py --api --auth
"""
import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
log_dir = project_root / "logs" / "integration_tests"
os.makedirs(log_dir, exist_ok=True)

log_file = log_dir / f"integration_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("integration_test")


class IntegrationTest:
    """Integration tests for Inventory Optimization System."""

    def __init__(
        self,
        db_url: str,
        redis_url: str,
        rabbitmq_url: str,
        api_base_url: str,
        admin_username: str,
        admin_password: str,
    ):
        """Initialize integration test.
        
        Args:
            db_url: Database connection URL
            redis_url: Redis connection URL
            rabbitmq_url: RabbitMQ connection URL
            api_base_url: API base URL
            admin_username: Admin username for authentication
            admin_password: Admin password for authentication
        """
        self.db_url = db_url
        self.redis_url = redis_url
        self.rabbitmq_url = rabbitmq_url
        self.api_base_url = api_base_url
        self.admin_username = admin_username
        self.admin_password = admin_password
        
        # Access token for API requests
        self.access_token = None
        
        # Import modules dynamically to avoid circular imports and ensure proper initialization
        try:
            # Database models
            from src.models.database import get_db
            from src.models.inventory import Product, Location, Inventory, InventoryTransaction
            from src.models.forecast import ForecastModel, Forecast
            from src.models.user import User, Role
            
            self.db_models = {
                "Product": Product,
                "Location": Location,
                "Inventory": Inventory,
                "InventoryTransaction": InventoryTransaction,
                "ForecastModel": ForecastModel,
                "Forecast": Forecast,
                "User": User,
                "Role": Role,
            }
            
            self.get_db = get_db
            
            # Redis cache
            from src.utils.redis_cache import RedisCache
            self.redis_cache = RedisCache(redis_url=redis_url)
            
            # RabbitMQ
            from src.utils.rabbitmq_producer import RabbitMQProducer
            self.rabbitmq_producer = RabbitMQProducer(rabbitmq_url=rabbitmq_url)
            
            # Successfully imported all required components
            logger.info("Successfully imported all required components")
        
        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.info("Some tests may be skipped due to missing components")
    
    def run_all_tests(self):
        """Run all integration tests."""
        test_results = {}
        
        # Run database tests
        logger.info("Running database tests...")
        test_results["database"] = self.test_database_connection()
        
        # Run Redis tests
        logger.info("Running Redis tests...")
        test_results["redis"] = self.test_redis_cache()
        
        # Run RabbitMQ tests
        logger.info("Running RabbitMQ tests...")
        test_results["rabbitmq"] = self.test_rabbitmq()
        
        # Run authentication tests
        logger.info("Running authentication tests...")
        test_results["auth"] = self.test_authentication()
        
        # Run API tests
        if test_results["auth"]:
            logger.info("Running API tests...")
            test_results["api"] = self.test_api_endpoints()
        else:
            logger.warning("Skipping API tests due to authentication failure")
            test_results["api"] = False
        
        # Run MLflow tests
        logger.info("Running MLflow tests...")
        test_results["mlflow"] = self.test_mlflow()
        
        # Run Celery tests
        logger.info("Running Celery tests...")
        test_results["celery"] = self.test_celery()
        
        # Summarize results
        logger.info("Integration test summary:")
        for component, result in test_results.items():
            status = "PASSED" if result else "FAILED"
            logger.info(f"  {component}: {status}")
        
        # Overall pass/fail
        overall_result = all(test_results.values())
        logger.info(f"Overall integration test result: {'PASSED' if overall_result else 'FAILED'}")
        
        return overall_result
    
    def test_database_connection(self) -> bool:
        """Test database connection and model validation.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create SQLAlchemy engine and session
            engine = create_engine(self.db_url)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            session = SessionLocal()
            
            # Check if tables exist
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            logger.info(f"Database tables found: {table_names}")
            
            # If tables exist, try querying each one
            for table in table_names:
                count = session.execute(f"SELECT COUNT(*) FROM {table}").scalar()
                logger.info(f"Table {table} has {count} rows")
            
            # Test session with imported get_db function if available
            if hasattr(self, 'get_db'):
                logger.info("Testing get_db function...")
                db = next(self.get_db())
                try:
                    # Try to use the session
                    product_count = db.query(self.db_models["Product"]).count()
                    logger.info(f"Product count from get_db: {product_count}")
                finally:
                    db.close()
            
            # Close the original session
            session.close()
            
            logger.info("Database connection test passed")
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def test_redis_cache(self) -> bool:
        """Test Redis cache functionality.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self, 'redis_cache'):
                logger.warning("Redis cache component not available, skipping test")
                return False
            
            # Test set and get operations
            test_key = f"integration_test_{uuid.uuid4()}"
            test_value = {
                "timestamp": datetime.datetime.now().isoformat(),
                "test_id": str(uuid.uuid4()),
                "random_value": random.randint(1, 1000)
            }
            
            # Set value in cache
            self.redis_cache.set(test_key, test_value, ttl=300)
            
            # Get value from cache
            retrieved_value = self.redis_cache.get(test_key)
            
            if retrieved_value == test_value:
                logger.info("Redis cache set/get test passed")
            else:
                logger.error(f"Redis cache test failed: values don't match. Original: {test_value}, Retrieved: {retrieved_value}")
                return False
            
            # Test deletion
            self.redis_cache.delete(test_key)
            after_delete = self.redis_cache.get(test_key)
            
            if after_delete is None:
                logger.info("Redis cache delete test passed")
            else:
                logger.error(f"Redis cache delete test failed: key still exists with value {after_delete}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache test failed: {str(e)}")
            return False
    
    def test_rabbitmq(self) -> bool:
        """Test RabbitMQ message passing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self, 'rabbitmq_producer'):
                logger.warning("RabbitMQ component not available, skipping test")
                return False
            
            # Send test message
            test_message = {
                "event_type": "integration_test",
                "timestamp": datetime.datetime.now().isoformat(),
                "test_id": str(uuid.uuid4()),
                "data": {
                    "random_value": random.randint(1, 1000)
                }
            }
            
            # Use a test-specific queue
            test_queue = "integration_test_queue"
            
            # Publish message
            self.rabbitmq_producer.publish(
                message=test_message,
                routing_key=test_queue
            )
            
            logger.info(f"Published test message to RabbitMQ queue '{test_queue}'")
            
            # Note: For a complete test, we would need a consumer to verify message receipt
            # In a real integration test, we might check logs or use a separate consumer script
            # For now, we'll consider the test successful if no exceptions were raised
            
            logger.info("RabbitMQ test passed (message sent successfully)")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ test failed: {str(e)}")
            return False
    
    def test_authentication(self) -> bool:
        """Test authentication and authorization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Login request to obtain access token
            login_url = f"{self.api_base_url}/auth/login"
            login_data = {
                "username": self.admin_username,
                "password": self.admin_password
            }
            
            # Make login request
            logger.info(f"Attempting login with username: {self.admin_username}")
            response = requests.post(login_url, json=login_data)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get("access_token")
                
                if self.access_token:
                    logger.info("Authentication successful, received access token")
                    
                    # Test token by making a request to a protected endpoint
                    me_url = f"{self.api_base_url}/users/me"
                    headers = {"Authorization": f"Bearer {self.access_token}"}
                    
                    me_response = requests.get(me_url, headers=headers)
                    
                    if me_response.status_code == 200:
                        user_data = me_response.json()
                        logger.info(f"Token validation successful, user: {user_data.get('username')}")
                        return True
                    else:
                        logger.error(f"Token validation failed with status code {me_response.status_code}")
                        logger.error(f"Response: {me_response.text}")
                        return False
                else:
                    logger.error("Authentication failed: No access token in response")
                    logger.error(f"Response: {token_data}")
                    return False
            else:
                logger.error(f"Authentication failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Authentication test failed: {str(e)}")
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test key API endpoints.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.access_token:
                logger.warning("No access token available, skipping API tests")
                return False
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Test endpoints and track success/failure
            endpoint_tests = []
            
            # 1. Test products endpoint
            products_url = f"{self.api_base_url}/products"
            response = requests.get(products_url, headers=headers)
            products_success = response.status_code == 200
            endpoint_tests.append(("products", products_success))
            
            if products_success:
                products_data = response.json()
                logger.info(f"Products endpoint successful, found {len(products_data)} products")
            else:
                logger.error(f"Products endpoint failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # 2. Test locations endpoint
            locations_url = f"{self.api_base_url}/locations"
            response = requests.get(locations_url, headers=headers)
            locations_success = response.status_code == 200
            endpoint_tests.append(("locations", locations_success))
            
            if locations_success:
                locations_data = response.json()
                logger.info(f"Locations endpoint successful, found {len(locations_data)} locations")
            else:
                logger.error(f"Locations endpoint failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # 3. Test inventory endpoint
            inventory_url = f"{self.api_base_url}/inventory"
            response = requests.get(inventory_url, headers=headers)
            inventory_success = response.status_code == 200
            endpoint_tests.append(("inventory", inventory_success))
            
            if inventory_success:
                inventory_data = response.json()
                logger.info(f"Inventory endpoint successful, found {len(inventory_data)} inventory records")
            else:
                logger.error(f"Inventory endpoint failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # 4. Test forecasts endpoint
            forecasts_url = f"{self.api_base_url}/forecasts"
            response = requests.get(forecasts_url, headers=headers)
            forecasts_success = response.status_code == 200
            endpoint_tests.append(("forecasts", forecasts_success))
            
            if forecasts_success:
                forecasts_data = response.json()
                logger.info(f"Forecasts endpoint successful, found {len(forecasts_data)} forecasts")
            else:
                logger.error(f"Forecasts endpoint failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # 5. Test health endpoint
            health_url = f"{self.api_base_url}/health"
            response = requests.get(health_url)  # Health endpoint typically doesn't require auth
            health_success = response.status_code == 200
            endpoint_tests.append(("health", health_success))
            
            if health_success:
                health_data = response.json()
                logger.info(f"Health endpoint successful: {health_data}")
            else:
                logger.error(f"Health endpoint failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # Calculate overall success based on individual endpoint tests
            endpoint_successes = [success for _, success in endpoint_tests]
            all_endpoints_success = all(endpoint_successes)
            
            if all_endpoints_success:
                logger.info("All API endpoints tested successfully")
            else:
                failed_endpoints = [endpoint for endpoint, success in endpoint_tests if not success]
                logger.error(f"The following API endpoints failed: {', '.join(failed_endpoints)}")
            
            return all_endpoints_success
            
        except Exception as e:
            logger.error(f"API endpoints test failed: {str(e)}")
            return False
    
    def test_mlflow(self) -> bool:
        """Test MLflow tracking.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import MLflow
            import mlflow
            
            # Set tracking URI (typically from an environment variable)
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Start a test run
            test_experiment_name = "integration_test"
            mlflow.set_experiment(test_experiment_name)
            
            with mlflow.start_run(run_name=f"integration_test_{uuid.uuid4()}") as run:
                run_id = run.info.run_id
                
                # Log some test metrics and parameters
                mlflow.log_param("test_datetime", datetime.datetime.now().isoformat())
                mlflow.log_metric("random_metric", random.random())
                mlflow.log_metric("test_accuracy", 0.95)
                
                # Log a test artifact
                test_artifact_path = log_dir / "test_artifact.json"
                test_data = {
                    "test_id": str(uuid.uuid4()),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "random_values": [random.random() for _ in range(5)]
                }
                
                with open(test_artifact_path, "w") as f:
                    json.dump(test_data, f)
                
                mlflow.log_artifact(test_artifact_path)
                
                logger.info(f"MLflow test run created with ID: {run_id}")
            
            logger.info("MLflow tracking test passed")
            return True
            
        except ImportError:
            logger.warning("MLflow not available, skipping test")
            return False
        except Exception as e:
            logger.error(f"MLflow tracking test failed: {str(e)}")
            return False
    
    def test_celery(self) -> bool:
        """Test Celery task execution.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import Celery tasks module
            from src.tasks.forecasting import generate_forecast
            
            # Submit a test task
            task_id = str(uuid.uuid4())
            
            # Create a test payload
            test_payload = {
                "product_id": 1,
                "location_id": 1,
                "horizon": 30,
                "model_id": 1,
                "test_run": True  # Flag to indicate this is a test
            }
            
            # Submit the task asynchronously
            result = generate_forecast.apply_async(
                args=[test_payload],
                task_id=task_id
            )
            
            logger.info(f"Celery task submitted with ID: {result.id}")
            
            # Wait for task completion with timeout
            try:
                task_result = result.get(timeout=30)  # 30-second timeout
                logger.info(f"Celery task completed with result: {task_result}")
                return True
            except Exception as e:
                logger.error(f"Error waiting for Celery task: {str(e)}")
                return False
            
        except ImportError:
            logger.warning("Celery tasks not available, skipping test")
            return False
        except Exception as e:
            logger.error(f"Celery task execution test failed: {str(e)}")
            return False


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run integration tests for Inventory Optimization System."
    )
    
    # Connection arguments
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inventory"),
        help="Database connection URL",
    )
    
    parser.add_argument(
        "--redis-url",
        type=str,
        default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis connection URL",
    )
    
    parser.add_argument(
        "--rabbitmq-url",
        type=str,
        default=os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F"),
        help="RabbitMQ connection URL",
    )
    
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1"),
        help="API base URL",
    )
    
    # Authentication credentials
    parser.add_argument(
        "--admin-username",
        type=str,
        default=os.environ.get("ADMIN_USERNAME", "admin"),
        help="Admin username for authentication",
    )
    
    parser.add_argument(
        "--admin-password",
        type=str,
        default=os.environ.get("ADMIN_PASSWORD", "password"),
        help="Admin password for authentication",
    )
    
    # Test selection arguments
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all integration tests",
    )
    
    parser.add_argument(
        "--database",
        action="store_true",
        help="Run database tests",
    )
    
    parser.add_argument(
        "--redis",
        action="store_true",
        help="Run Redis cache tests",
    )
    
    parser.add_argument(
        "--rabbitmq",
        action="store_true",
        help="Run RabbitMQ tests",
    )
    
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Run authentication tests",
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API tests",
    )
    
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Run MLflow tests",
    )
    
    parser.add_argument(
        "--celery",
        action="store_true",
        help="Run Celery tests",
    )
    
    return parser.parse_args()


def main():
    """Run the script."""
    args = parse_args()
    
    logger.info("Starting integration tests")
    
    # Initialize the integration test with connection details
    integration_test = IntegrationTest(
        db_url=args.db_url,
        redis_url=args.redis_url,
        rabbitmq_url=args.rabbitmq_url,
        api_base_url=args.api_base_url,
        admin_username=args.admin_username,
        admin_password=args.admin_password,
    )
    
    # Determine which tests to run
    if args.all:
        success = integration_test.run_all_tests()
    else:
        # Run selected tests
        success = True
        
        if args.database:
            logger.info("Running database tests...")
            db_success = integration_test.test_database_connection()
            logger.info(f"Database tests: {'PASSED' if db_success else 'FAILED'}")
            success = success and db_success
        
        if args.redis:
            logger.info("Running Redis tests...")
            redis_success = integration_test.test_redis_cache()
            logger.info(f"Redis tests: {'PASSED' if redis_success else 'FAILED'}")
            success = success and redis_success
        
        if args.rabbitmq:
            logger.info("Running RabbitMQ tests...")
            rabbitmq_success = integration_test.test_rabbitmq()
            logger.info(f"RabbitMQ tests: {'PASSED' if rabbitmq_success else 'FAILED'}")
            success = success and rabbitmq_success
        
        if args.auth:
            logger.info("Running authentication tests...")
            auth_success = integration_test.test_authentication()
            logger.info(f"Authentication tests: {'PASSED' if auth_success else 'FAILED'}")
            success = success and auth_success
        
        if args.api:
            logger.info("Running API tests...")
            api_success = integration_test.test_api_endpoints()
            logger.info(f"API tests: {'PASSED' if api_success else 'FAILED'}")
            success = success and api_success
        
        if args.mlflow:
            logger.info("Running MLflow tests...")
            mlflow_success = integration_test.test_mlflow()
            logger.info(f"MLflow tests: {'PASSED' if mlflow_success else 'FAILED'}")
            success = success and mlflow_success
        
        if args.celery:
            logger.info("Running Celery tests...")
            celery_success = integration_test.test_celery()
            logger.info(f"Celery tests: {'PASSED' if celery_success else 'FAILED'}")
            success = success and celery_success
        
        # If no tests were specified, run all tests
        if not any([args.database, args.redis, args.rabbitmq, args.auth, args.api, args.mlflow, args.celery]):
            logger.info("No specific tests selected, running all tests...")
            success = integration_test.run_all_tests()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 