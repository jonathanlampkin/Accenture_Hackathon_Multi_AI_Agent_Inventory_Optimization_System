#!/usr/bin/env python
"""
Main script to run the Multi-Agent Inventory Optimization System.

This script initializes and executes the inventory optimization process
using the CrewAI-based agent system. It handles configuration loading,
system execution, and result visualization.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.crew_manager import InventoryCrewManager, run_inventory_optimization
from src.utils.visualization import create_dashboard, plot_optimization_results
from src.utils.data_loader import load_inventory_data, load_demand_data


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file or use defaults.
    
    Args:
        config_path: Path to a JSON or YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default configuration
        config = {
            "data": {
                "demand_data_path": "data/demand_data.csv",
                "inventory_data_path": "data/inventory_data.csv"
            },
            "output_dir": "./output/inventory_optimization",
            "verbose": 2,
            "process": "sequential",
            "forecasting": {
                "horizon": 30,
                "confidence_level": 0.95,
                "test_proportion": 0.2
            },
            "optimization": {
                "service_level_target": 0.95,
                "holding_cost_rate": 0.25
            },
            "scenario_planning": {
                "scenarios": {
                    "base": {"demand_factor": 1.0, "lead_time_factor": 1.0},
                    "high_demand": {"demand_factor": 1.5, "lead_time_factor": 1.0},
                    "supply_disruption": {"demand_factor": 1.0, "lead_time_factor": 2.0},
                    "worst_case": {"demand_factor": 1.3, "lead_time_factor": 1.7}
                }
            },
            "monitoring": {
                "kpi_targets": {
                    "service_level": 0.95,
                    "fill_rate": 0.97,
                    "inventory_turnover": 12.0
                }
            },
            "anomaly_detection": {
                "threshold": 3.0,
                "method": "zscore" 
            }
        }
        return config
    
    # Load from file
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    elif config_path.endswith((".yaml", ".yml")):
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must be JSON or YAML format")


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("inventory_optimization.log")
        ]
    )


def generate_visualizations(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate visualizations from the optimization results.
    
    Args:
        results: Dictionary containing results from each agent
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dashboard visualization
    dashboard_path = os.path.join(output_dir, "inventory_dashboard.html")
    create_dashboard(results, dashboard_path)
    
    # Generate individual plots
    plot_optimization_results(results, output_dir)
    
    logging.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run the inventory optimization system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Multi-Agent Inventory Optimization System")
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON or YAML)")
    parser.add_argument("--output-dir", type=str, help="Directory to save output files")
    parser.add_argument("--demand-data", type=str, help="Path to demand data CSV file")
    parser.add_argument("--inventory-data", type=str, help="Path to inventory data CSV file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting Multi-Agent Inventory Optimization System")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.demand_data:
        config["data"]["demand_data_path"] = args.demand_data
    if args.inventory_data:
        config["data"]["inventory_data_path"] = args.inventory_data
    
    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Save the configuration
    config_path = os.path.join(config["output_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Validate input data
    try:
        demand_df = load_demand_data(config["data"]["demand_data_path"])
        logging.info(f"Loaded demand data with {len(demand_df)} records")
        
        if "inventory_data_path" in config["data"]:
            inventory_df = load_inventory_data(config["data"]["inventory_data_path"])
            logging.info(f"Loaded inventory data with {len(inventory_df)} records")
    except Exception as e:
        logging.error(f"Error loading input data: {str(e)}")
        return 1
    
    try:
        # Run the inventory optimization
        logging.info("Running inventory optimization...")
        results = run_inventory_optimization(config_dict=config)
        
        # Generate visualizations
        logging.info("Generating visualizations...")
        generate_visualizations(results, config["output_dir"])
        
        logging.info(f"Inventory optimization completed. Results saved to {config['output_dir']}")
        return 0
        
    except Exception as e:
        logging.error(f"Error during inventory optimization: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 