"""
API Module for Inventory Optimization System

This package contains the API endpoints for the inventory optimization system,
including forecasting APIs, inventory management APIs, and reporting APIs.
"""

__version__ = "1.0.0"

import os

# Create required directories
def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "api/uploads",
        "api/results",
        "api/static",
        "api/templates",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Run directory initialization
ensure_directories() 