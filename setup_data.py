#!/usr/bin/env python3
"""
Setup script to copy data files to the correct location.
This helps ensure the Multi-Agent Inventory Optimization System can find the data files.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataSetup")

def setup_data_files():
    """
    Copy data files to the data directory if they exist in the root directory
    but not in the data directory.
    """
    # Get the base directory (project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir)
    
    # Files to check
    data_files = [
        "inventory_monitoring.csv",
        "demand_forecasting.csv",
        "pricing_optimization.csv"
    ]
    
    # Copy files from root to data directory if needed
    files_copied = 0
    for filename in data_files:
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(data_dir, filename)
        
        # Check if file exists in root but not in data directory
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            logger.info(f"Copying {filename} to data directory")
            try:
                shutil.copy2(src_path, dst_path)
                files_copied += 1
            except Exception as e:
                logger.error(f"Error copying {filename}: {e}")
        elif not os.path.exists(src_path) and not os.path.exists(dst_path):
            logger.warning(f"Data file {filename} not found in either root or data directory")
        elif os.path.exists(dst_path):
            logger.info(f"File {filename} already exists in data directory")
    
    if files_copied > 0:
        logger.info(f"Successfully copied {files_copied} data files to {data_dir}")
    else:
        logger.info("No files needed to be copied")
    
    # Check if all required files are present
    missing_files = []
    for filename in data_files:
        if not os.path.exists(os.path.join(data_dir, filename)):
            missing_files.append(filename)
    
    if missing_files:
        logger.warning("The following data files are still missing:")
        for filename in missing_files:
            logger.warning(f"  - {filename}")
        logger.warning("Please ensure these files are placed in the data directory before running the system")
    else:
        logger.info("All required data files are present in the data directory")
        
    return len(missing_files) == 0

if __name__ == "__main__":
    logger.info("Setting up data files for the Multi-Agent Inventory Optimization System")
    all_files_present = setup_data_files()
    
    if all_files_present:
        logger.info("Data setup completed successfully. The system is ready to run.")
        sys.exit(0)
    else:
        logger.warning("Data setup completed with warnings. Some files may be missing.")
        sys.exit(1) 