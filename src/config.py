"""
Configuration settings for the Multi-Agent Inventory Optimization System.
Contains paths, parameters, feature flags, and optimization goals.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

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

# Global configuration flags
USE_ADVANCED_MODELS = True
USE_GPU = True

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

def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments for the inventory optimization system.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Inventory Optimization System"
    )
    
    parser.add_argument(
        "--optimization-target",
        type=str,
        choices=["cost", "availability", "balanced"],
        default="balanced",
        help="Target optimization goal"
    )
    
    parser.add_argument(
        "--product-id",
        type=str,
        help="Specific product ID to focus on"
    )
    
    parser.add_argument(
        "--store-id",
        type=str,
        help="Specific store ID to focus on"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum number of optimization iterations"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration if available"
    )
    
    parser.add_argument(
        "--simple-mode",
        action="store_true",
        help="Run in simple mode without advanced ML models"
    )
    
    parser.add_argument(
        "--use-crewai",
        action="store_true",
        default=True,
        help="Use the new crewAI-based system for optimization"
    )
    
    args = parser.parse_args()
    
    return {
        "optimization_target": args.optimization_target,
        "product_id": args.product_id,
        "store_id": args.store_id,
        "iterations": args.iterations,
        "output_dir": args.output_dir,
        "use_gpu": args.use_gpu,
        "simple_mode": args.simple_mode,
        "use_crewai": args.use_crewai
    }

def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") 