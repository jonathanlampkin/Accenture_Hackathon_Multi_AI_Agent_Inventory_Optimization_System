#!/usr/bin/env python3
"""
Multi-Agent Inventory Optimization System - Main Entry Point

This script initializes and runs the Multi-Agent Inventory Optimization System,
which uses a collaborative approach among specialized agents to optimize
inventory management, demand forecasting, and pricing strategies.
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"inventory_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("InventorySystem")

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system components
try:
    from setup_data import setup_data_files
    from src.utils import check_data_files, get_data_summary
    from src.config import parse_args, USE_ADVANCED_MODELS, USE_GPU
    from src.coordinator import MultiAgentCoordinator
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please make sure you've installed all dependencies from requirements.txt")
    sys.exit(1)

def main():
    """Main entry point for the Multi-Agent Inventory Optimization System"""
    config = parse_args()
    
    logger.info("Starting Multi-Agent Inventory Optimization System")
    logger.info(f"GPU support: {'Enabled' if config['use_gpu'] else 'Disabled'}")
    
    # Set up data files
    logger.info("Setting up data files...")
    setup_data_files()
    
    # Check if required data files exist
    if not check_data_files():
        logger.error("Required data files are missing. Exiting.")
        sys.exit(1)
    
    # Display data summary
    logger.info("Generating data summary...")
    summary = get_data_summary()
    
    logger.info("=== Data Summary ===")
    for dataset, stats in summary.items():
        logger.info(f"{dataset.capitalize()} Dataset:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    # Override advanced models setting if simple mode is requested
    if config["simple_mode"]:
        from src import config as cfg
        cfg.USE_ADVANCED_MODELS = False
        logger.info("Running in simple mode without advanced ML models")
    
    try:
        # Initialize the coordinator with all agents
        coordinator = MultiAgentCoordinator(
            optimization_target=config["optimization_target"],
            product_id=config["product_id"],
            store_id=config["store_id"],
            max_iterations=config["iterations"],
            output_dir=config["output_dir"],
            use_gpu=config["use_gpu"]
        )
        
        # Run the optimization process
        results = coordinator.run_optimization()
        
        # Display summary of results
        logger.info("Optimization completed successfully")
        logger.info(f"Optimization target: {config['optimization_target']}")
        logger.info(f"Total optimization cycles: {config['iterations']}")
        
        # Print key metrics from results
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info(f"Detailed results and visualizations saved to: {coordinator.output_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 