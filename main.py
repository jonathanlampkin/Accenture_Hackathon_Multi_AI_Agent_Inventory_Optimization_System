#!/usr/bin/env python3
"""
Multi-Agent Inventory Optimization System - Main Entry Point

This script initializes and runs the Multi-Agent Inventory Optimization System,
which uses a collaborative approach among specialized agents to optimize
inventory management, demand forecasting, and pricing strategies.
"""

import os
import sys
import argparse
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
    from src.config import USE_ADVANCED_MODELS
    from src.coordinator import MultiAgentCoordinator
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please make sure you've installed all dependencies from requirements.txt")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Agent Inventory Optimization System')
    parser.add_argument('--simple-mode', action='store_true', 
                        help='Run in simple mode without advanced ML models')
    parser.add_argument('--optimize-for', choices=['cost', 'availability', 'balanced'],
                        default='balanced', help='Optimization target')
    parser.add_argument('--product-id', type=str, help='Focus on a specific product ID')
    parser.add_argument('--store-id', type=str, help='Focus on a specific store ID')
    parser.add_argument('--iterations', type=int, default=5, 
                        help='Number of optimization iterations')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    parser.add_argument('--setup-only', action='store_true',
                        help='Only set up data files without running the optimization')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only display data summary without running the optimization')
    return parser.parse_args()

def main():
    """Main entry point for the Multi-Agent Inventory Optimization System"""
    args = parse_arguments()
    
    logger.info("Starting Multi-Agent Inventory Optimization System")
    
    # Set up data files
    logger.info("Setting up data files...")
    setup_data_files()
    
    # Check if required data files exist
    if not check_data_files():
        logger.error("Required data files are missing. Exiting.")
        sys.exit(1)
    
    # If setup-only flag is set, exit after setup
    if args.setup_only:
        logger.info("Setup completed successfully. Exiting as requested (--setup-only).")
        sys.exit(0)
    
    # Display data summary
    logger.info("Generating data summary...")
    summary = get_data_summary()
    
    logger.info("=== Data Summary ===")
    for dataset, stats in summary.items():
        logger.info(f"{dataset.capitalize()} Dataset:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    # If summary-only flag is set, exit after displaying summary
    if args.summary_only:
        logger.info("Summary generated successfully. Exiting as requested (--summary-only).")
        sys.exit(0)
    
    # Override advanced models setting if simple mode is requested
    if args.simple_mode:
        from src import config
        config.USE_ADVANCED_MODELS = False
        logger.info("Running in simple mode without advanced ML models")
    
    try:
        # Initialize the coordinator with all agents
        coordinator = MultiAgentCoordinator(
            optimization_target=args.optimize_for,
            product_id=args.product_id,
            store_id=args.store_id,
            max_iterations=args.iterations,
            output_dir=args.output_dir
        )
        
        # Run the optimization process
        results = coordinator.run_optimization()
        
        # Display summary of results
        logger.info("Optimization completed successfully")
        logger.info(f"Optimization target: {args.optimize_for}")
        logger.info(f"Total optimization cycles: {args.iterations}")
        
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