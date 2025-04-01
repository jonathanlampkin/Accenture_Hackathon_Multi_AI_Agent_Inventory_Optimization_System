"""
Utility modules for the Multi-Agent Inventory Optimization System
"""

import os
import sys

# Add the src directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from src.utils.data_loader import (
    load_inventory_data,
    load_demand_data,
    load_pricing_data,
    get_product_data,
    get_store_data,
    get_data_summary
)

from src.utils.visualizer import (
    plot_inventory_levels,
    plot_stockout_risk,
    plot_sales_trend,
    plot_price_elasticity,
    plot_price_vs_sales,
    generate_inventory_dashboard
)

def check_data_files():
    """
    Check if all required data files exist.
    
    Returns:
        bool: True if all files exist, False otherwise
    """
    files_to_check = [
        config.INVENTORY_DATA,
        config.DEMAND_DATA,
        config.PRICING_DATA
    ]
    
    missing_files = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: The following required data files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease make sure these files are in the root directory.")
        return False
    
    return True 