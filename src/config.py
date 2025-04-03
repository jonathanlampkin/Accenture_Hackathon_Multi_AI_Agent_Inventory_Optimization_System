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
LOG_DIR = os.path.join(BASE_DIR, "logs")

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
    """Parse command line arguments and return configuration"""
    parser = argparse.ArgumentParser(description='Multi-Agent Inventory Optimization System')
    
    parser.add_argument('--optimize-for', type=str, default='balanced',
                      choices=['cost', 'service', 'balanced'],
                      help='Optimization target (default: balanced)')
    
    # Updated LLM/Ollama arguments
    parser.add_argument('--model-name', type=str, default=os.getenv('OLLAMA_MODEL', 'llama3'), # Default to llama3, allow env var override
                      help='Ollama model name to use (e.g., llama3, mistral, codellama). Can also be set via OLLAMA_MODEL env var.')
    
    parser.add_argument('--ollama-base-url', type=str, default=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                      help='Base URL for the Ollama server. Can also be set via OLLAMA_BASE_URL env var.')
    
    # Keep existing arguments
    parser.add_argument('--use-gpu', action='store_true', default=os.getenv('USE_GPU', '0') == '1',
                      help='Enable GPU support (primarily for Ollama). Can also be set via USE_GPU=1 env var.')
    
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save results (default: output)')
    
    parser.add_argument('--iterations', type=int, default=5,
                      help='Number of optimization iterations (default: 5 - Note: This might not be directly used by CrewAI\'s process)')
    
    # Add data file argument
    parser.add_argument('--data-file', type=str, default='data/inventory_data.csv', 
                      help='Path to the inventory data CSV file (default: data/inventory_data.csv)')
    
    args = parser.parse_args()
    
    # Convert to dictionary
    config = vars(args)
    
    # Ensure output directory exists
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # No need to re-add env vars here as they are handled by `default` in argparse
    
    return config

# Default configuration (less relevant now with argparse defaults)
DEFAULT_CONFIG = {
    'optimize_for': 'balanced',
    'model_name': 'llama3',
    'ollama_base_url': 'http://localhost:11434',
    'use_gpu': False,
    'output_dir': 'output',
    'data_file': 'data/inventory_data.csv',
    'iterations': 5
}

def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") 