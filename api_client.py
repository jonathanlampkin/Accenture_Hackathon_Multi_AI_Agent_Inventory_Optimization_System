#!/usr/bin/env python3
"""
API Client for Inventory Optimization System

This script demonstrates how to interact with the Inventory Optimization API
programmatically. It shows how to:
1. Check API health
2. List available files
3. Generate forecasts for specific products
4. Generate forecasts for all products in a file
"""

import requests
import json
import sys
import os
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def check_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking API health: {e}")
        return None

def list_files():
    """List all available files in the API."""
    try:
        response = requests.get(f"{BASE_URL}/files")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error listing files: {e}")
        return None

def upload_file(file_path):
    """Upload a file to the API."""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def forecast_product(filename, product_id, horizon=30):
    """Generate forecast for a specific product."""
    try:
        data = {
            'filename': filename,
            'product_id': product_id,
            'horizon': horizon
        }
        response = requests.post(f"{BASE_URL}/forecast-product", data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error generating forecast for product {product_id}: {e}")
        return None

def generate_forecasts(filename, horizon=30, test_proportion=0.2):
    """Generate forecasts for all products in a file."""
    try:
        data = {
            'filename': filename,
            'horizon': horizon,
            'test_proportion': test_proportion
        }
        response = requests.post(f"{BASE_URL}/forecast", data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error generating forecasts: {e}")
        return None

def save_forecast_image(image_url, output_path):
    """Download and save a forecast image."""
    try:
        # Extract the filename from the path
        image_filename = os.path.basename(image_url)
        
        # For the API, we need to query the /results/{filename} endpoint
        response = requests.get(f"{BASE_URL}/results/{image_filename}")
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Image saved to {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False

def display_results(result, save_dir=None):
    """Display the forecast results."""
    if not result:
        return
    
    print("\n=== Forecast Results ===")
    print(f"Product ID: {result.get('product_id')}")
    print(f"Method: {result.get('method')}")
    
    # Display metrics if available
    if 'rmse' in result and result['rmse'] is not None:
        print(f"RMSE: {result['rmse']:.2f}")
    if 'mae' in result and result['mae'] is not None:
        print(f"MAE: {result['mae']:.2f}")
    if 'r2' in result and result['r2'] is not None:
        print(f"R²: {result['r2']:.2f}")
    
    # Get forecast statistics
    forecast = result.get('forecast', [])
    if forecast:
        print(f"\nForecast Statistics:")
        print(f"  Mean: {sum(forecast) / len(forecast):.2f}")
        print(f"  Min:  {min(forecast):.2f}")
        print(f"  Max:  {max(forecast):.2f}")
    
    # Save visualization if available
    if 'visualization' in result and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        image_filename = os.path.basename(result['visualization'])
        output_path = os.path.join(save_dir, image_filename)
        save_forecast_image(result['visualization'], output_path)

def main():
    """Main function to demonstrate API usage."""
    print("=== Inventory Optimization API Client ===")
    
    # Check if the API is healthy
    health = check_health()
    if not health or health.get('status') != 'healthy':
        print("Error: API is not available or healthy")
        sys.exit(1)
    
    print("API is healthy and available!")
    
    # List available files
    print("\n=== Available Files ===")
    files_response = list_files()
    if files_response and 'files' in files_response:
        for file in files_response['files']:
            print(f"File: {file['filename']}")
            print(f"  Rows: {file.get('rows', 'N/A')}")
            print(f"  Columns: {file.get('columns', 'N/A')}")
            print(f"  Size: {file.get('size_bytes', 0) / 1024:.2f} KB")
            print("---")
    
    # Create output directory for results
    output_dir = f"api_client_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if 'demand_data.csv' exists, if not use the first available file or exit
    available_files = [file['filename'] for file in files_response.get('files', [])]
    if not available_files:
        print("No files available. Please upload a file first.")
        sys.exit(1)
    
    data_file = 'demand_data.csv' if 'demand_data.csv' in available_files else available_files[0]
    print(f"\nUsing data file: {data_file}")
    
    # Generate forecast for specific products
    products_to_forecast = [101, 102, 103, 104, 105]
    
    for product_id in products_to_forecast:
        print(f"\n=== Generating forecast for Product {product_id} ===")
        result = forecast_product(data_file, product_id)
        display_results(result, output_dir)
    
    print(f"\nAll results saved to {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main() 