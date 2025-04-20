#!/usr/bin/env python3
"""
API Test Script for Inventory Optimization System

This script tests the basic functionality of the inventory optimization API
by sending requests to various endpoints and validating the responses.
"""

import requests
import json
import logging
import os
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000"
TEST_DATA_PATH = "../data/demand_data.csv"

def test_health_endpoint():
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'healthy':
            logger.info("Health check endpoint: PASSED")
            return True
        else:
            logger.error(f"Health check endpoint returned unexpected status: {data}")
            return False
    except Exception as e:
        logger.error(f"Health check endpoint test failed: {str(e)}")
        return False

def test_file_upload():
    """Test file upload functionality."""
    try:
        if not os.path.exists(TEST_DATA_PATH):
            logger.error(f"Test data file not found: {TEST_DATA_PATH}")
            return False
        
        with open(TEST_DATA_PATH, 'rb') as file:
            files = {'file': ('test_demand_data.csv', file, 'text/csv')}
            response = requests.post(f"{API_URL}/upload", files=files)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success':
                logger.info(f"File upload test: PASSED (uploaded {data.get('rows')} rows)")
                return True
            else:
                logger.error(f"File upload test returned unexpected response: {data}")
                return False
    except Exception as e:
        logger.error(f"File upload test failed: {str(e)}")
        return False

def test_file_list():
    """Test file listing functionality."""
    try:
        response = requests.get(f"{API_URL}/files")
        response.raise_for_status()
        data = response.json()
        
        if 'files' in data:
            logger.info(f"File list test: PASSED (found {len(data['files'])} files)")
            return True, data['files']
        else:
            logger.error(f"File list test returned unexpected response: {data}")
            return False, []
    except Exception as e:
        logger.error(f"File list test failed: {str(e)}")
        return False, []

def test_forecast_generation(files):
    """Test forecast generation for a file."""
    if not files:
        logger.warning("Skipping forecast test: No files available")
        return False
    
    try:
        # Get the first file
        test_file = files[0]['filename']
        
        # Prepare form data
        data = {
            'filename': test_file,
            'horizon': 30,
            'test_proportion': 0.2
        }
        
        logger.info(f"Generating forecasts for {test_file}...")
        response = requests.post(f"{API_URL}/forecast", data=data)
        response.raise_for_status()
        result = response.json()
        
        if result.get('status') == 'success':
            forecast_count = result.get('forecast_count', 0)
            logger.info(f"Forecast test: PASSED (generated {forecast_count} forecasts)")
            
            # Print summary information
            if 'summary' in result and result['summary']:
                logger.info("Forecast Summary:")
                for item in result['summary']:
                    logger.info(f"  Product {item.get('Product ID')}: {item.get('Method')} - "
                              f"Mean: {item.get('Mean Forecast')}, RMSE: {item.get('RMSE')}")
            
            return True
        else:
            logger.error(f"Forecast test returned unexpected response: {result}")
            return False
    except Exception as e:
        logger.error(f"Forecast test failed: {str(e)}")
        return False

def test_product_forecast(files):
    """Test single product forecast functionality."""
    if not files:
        logger.warning("Skipping product forecast test: No files available")
        return False
    
    try:
        # Get the first file
        test_file = files[0]['filename']
        
        # Test product ID (assuming 101 exists in the data)
        product_id = 101
        
        # Make request
        data = {'filename': test_file}
        response = requests.post(
            f"{API_URL}/forecast-product?product_id={product_id}&horizon=30",
            data=data
        )
        response.raise_for_status()
        result = response.json()
        
        if 'product_id' in result and result['product_id'] == product_id:
            logger.info(f"Product forecast test: PASSED (generated forecast for product {product_id})")
            logger.info(f"  Method: {result.get('method')}")
            logger.info(f"  RMSE: {result.get('rmse')}")
            
            if 'visualization' in result:
                logger.info(f"  Visualization: {result.get('visualization')}")
            
            return True
        else:
            logger.error(f"Product forecast test returned unexpected response: {result}")
            return False
    except Exception as e:
        logger.error(f"Product forecast test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all API tests and report results."""
    results = {}
    
    logger.info("Starting API tests...")
    
    # Test health endpoint
    results['health'] = test_health_endpoint()
    
    # Test file upload
    results['upload'] = test_file_upload()
    
    # Test file listing
    results['file_list'], files = test_file_list()
    
    # Test forecast generation
    results['forecast'] = test_forecast_generation(files)
    
    # Test product forecast
    results['product_forecast'] = test_product_forecast(files)
    
    # Report results
    logger.info("\nTest Results:")
    all_passed = True
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        all_passed = all_passed and result
    
    return all_passed

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        sys.exit(1) 