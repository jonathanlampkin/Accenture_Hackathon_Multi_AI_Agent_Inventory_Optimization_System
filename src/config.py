"""
Configuration settings for the Multi-Agent Inventory Optimization System.
Contains paths, parameters, feature flags, and optimization goals.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Data file paths with fallbacks
def get_data_file_path(filename):
    """
    Get the path to a data file, checking multiple possible locations.
    
    Args:
        filename (str): Name of the data file
        
    Returns:
        str: Full path to the data file
    """
    # Check possible locations in order of preference
    possible_locations = [
        os.path.join(DATA_DIR, filename),  # Preferred location in data/ directory
        os.path.join(BASE_DIR, filename),  # Root directory
        os.path.join(os.path.dirname(BASE_DIR), filename)  # Parent directory
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            return location
    
    # If not found, return the default location (data/ directory)
    return os.path.join(DATA_DIR, filename)

# Data files
INVENTORY_DATA = get_data_file_path("inventory_monitoring.csv")
DEMAND_DATA = get_data_file_path("demand_forecasting.csv")
PRICING_DATA = get_data_file_path("pricing_optimization.csv")

# Create output and model directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Model parameters
FORECAST_HORIZON = 30  # days
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
STOCKOUT_RISK_THRESHOLD = 0.15  # 15% chance of stockout

# Agent parameters
MAX_ITERATIONS = 10
COMMUNICATION_INTERVAL = 2

# Feature flags
USE_ADVANCED_MODELS = True  # Set to False for environments with limited dependencies

# Optimization goals and weights
OPTIMIZATION_GOALS = {
    "cost": {
        "inventory_carrying_cost": 0.7,
        "stockout_cost": 0.2,
        "order_cost": 0.1
    },
    "availability": {
        "inventory_carrying_cost": 0.2,
        "stockout_cost": 0.7,
        "order_cost": 0.1
    },
    "balanced": {
        "inventory_carrying_cost": 0.4,
        "stockout_cost": 0.4,
        "order_cost": 0.2
    }
}

# Environment variable for GPU usage
USE_GPU = os.environ.get("USE_GPU", "0") == "1"

# Default configuration
DEFAULT_CONFIG = {
    "simple_mode": False,
    "optimization_target": "balanced",  # 'cost', 'availability', or 'balanced'
    "product_id": None,
    "store_id": None,
    "iterations": 5,
    "output_dir": "output",
    "use_gpu": USE_GPU
}

def parse_args():
    """Parse command line arguments and update config."""
    parser = argparse.ArgumentParser(description='Multi-Agent Inventory Optimization System')
    parser.add_argument('--simple-mode', action='store_true', help='Run in simple mode without advanced ML models')
    parser.add_argument('--optimize-for', choices=['cost', 'availability', 'balanced'], default='balanced',
                        help='Optimization target (default: balanced)')
    parser.add_argument('--product-id', type=str, help='Focus on a specific product ID')
    parser.add_argument('--store-id', type=str, help='Focus on a specific store ID')
    parser.add_argument('--iterations', type=int, default=5, help='Number of optimization iterations')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for results')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "simple_mode": args.simple_mode,
        "optimization_target": args.optimize_for,
        "product_id": args.product_id,
        "store_id": args.store_id,
        "iterations": args.iterations,
        "output_dir": args.output_dir,
        "use_gpu": USE_GPU
    })
    
    return config

def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") 