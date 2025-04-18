#!/usr/bin/env python3
"""
Multi-Agent Inventory Optimization System - Main Entry Point

Refactored to use CrewAI with Ollama integration for improved performance and structure.
"""

import argparse
import os
import logging
import time
import sys
import signal
import pandas as pd
from datetime import datetime
import sqlite3 # Import sqlite3

# Assuming project structure allows this import path
from src.config import LOG_DIR, DATA_DIR
from src.agents import InventoryAgents
from src.tasks import InventoryTasks
# Correct the import for data loader - Import both functions
from src.utils.data_loader import load_inventory_data, load_demand_data
# Import logger utils
from src.utils.db_logger import initialize_db, log_result # Correct import

# Set up logging
# --- Start Logging Setup ---
# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"inventory_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure root logger
logging.basicConfig(level=logging.DEBUG, # Set to DEBUG to capture all levels
                    format='%(asctime)s - %(thread)d - %(filename)s-%(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename), # Log to file
                        logging.StreamHandler(sys.stdout)  # Log to console
                    ])

# Configure specific loggers (optional, can adjust levels for libraries)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.INFO)
logging.getLogger("crewai").setLevel(logging.INFO) # Set crewai to INFO to reduce verbosity slightly if needed

logger = logging.getLogger(__name__) # Get logger for the main module
# --- End Logging Setup ---

def timeout_handler(signum, frame):
    logger.error("Script timed out!")
    raise TimeoutError("Script execution timed out")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the Multi-Agent Inventory Optimization System.')
    parser.add_argument('--model-name', type=str, default='llama3', help='Name of the Ollama model to use (e.g., llama3, mistral)')
    parser.add_argument('--ollama-base-url', type=str, default='http://localhost:11434', help='Base URL for the Ollama server')
    # Use a default data file, assuming it's in the expected DATA_DIR
    default_data_file = os.path.join(DATA_DIR, "inventory_monitoring.csv") 
    parser.add_argument('--data-file', type=str, default=default_data_file, help='Path to the inventory data CSV file')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output files and logs')
    
    args = parser.parse_args()
    # Ensure output directory exists (config.py might also do this, but good to have here too)
    os.makedirs(args.output_dir, exist_ok=True)
    
    return vars(args)

def main():
    """Main execution flow for the Inventory Optimization Crew."""
    # Set timeout (consider if still needed with CrewAI's internal timeouts)
    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(3600)  # 1 hour - adjust as needed

    start_time = time.time()
    config = None
    crew_result = None # Initialize crew_result
    tasks_list = [] # Initialize tasks_list
    
    try:
        config = parse_args()
        logger.info("Configuration loaded:")
        # Log key config items
        logger.info(f"  Ollama Model: {config['model_name']}")
        logger.info(f"  Ollama URL: {config['ollama_base_url']}")
        logger.info(f"  Data File: {config['data_file']}")
        logger.info(f"  Output Dir: {config['output_dir']}")
        logger.info(f"  Log File: {log_filename}")

        # Initialize the results database
        logger.info("Initializing results database...")
        initialize_db()
        logger.info("Database initialized.")

        # Load Data
        logger.info(f"Loading inventory data...")
        inventory_data = load_inventory_data()
        if inventory_data is None or inventory_data.empty:
             logger.error("Failed to load inventory data or data is empty. Exiting.")
             sys.exit(1)
        logger.info("Inventory data loaded successfully.")

        logger.info(f"Loading demand data...") # Add demand data loading
        demand_data = load_demand_data()
        if demand_data is None or demand_data.empty:
             logger.error("Failed to load demand data or data is empty. Exiting.")
             sys.exit(1)
        logger.info("Demand data loaded successfully.")

        # Initialize Agents
        logger.info("Initializing agents...")
        agent_provider = InventoryAgents(
            model_name=config['model_name'], 
            ollama_base_url=config['ollama_base_url']
        )
        # Updated agents_dict - removed pricing_analyst
        agents_dict = {
            "demand_analyst": agent_provider.create_demand_analyst(),
            "inventory_optimizer": agent_provider.create_inventory_optimizer(),
            "supply_chain_analyst": agent_provider.create_supply_chain_analyst(),
            # "pricing_analyst": agent_provider.create_pricing_analyst(), # Removed
            "risk_analyst": agent_provider.create_risk_analyst()
        }
        logger.info("Agents initialized.")
        
        # Initialize Tasks - Pass both dataframes
        logger.info("Initializing tasks...")
        task_provider = InventoryTasks(inventory_data=inventory_data, demand_data=demand_data) 
        tasks_list = task_provider.get_all_tasks(agents_dict)
        logger.info(f"Tasks created: {[task.description[:50] + '...' for task in tasks_list]}") # Log brief task descriptions
        
        # Create and Run Crew
        logger.info("Creating and kicking off the crew...")
        # Need to import Crew here or at top level if used earlier
        from crewai import Crew 
        crew = Crew(agents=list(agents_dict.values()), tasks=tasks_list, verbose=2) # Use verbose=2 for detailed execution
        crew_result = crew.kickoff()
        
        logger.info("\n===== Crew Execution Finished ====")
        logger.info(f"Final Result:\n{crew_result}")

    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        sys.exit(1) # Exit if critical error during setup or execution
    finally:
        # Log the final result to the database
        if crew_result and tasks_list: # Check if crew ran and tasks exist
            try:
                last_task = tasks_list[-1]
                # Simple summary for now, could parse crew_result if needed
                result_summary = str(crew_result)[:500] + ("..." if len(str(crew_result)) > 500 else "") 
                log_result(
                    task_description=last_task.description,
                    agent_role=last_task.agent.role, # Get role from agent assigned to the last task
                    result_summary=result_summary,
                    raw_output=str(crew_result)
                )
                logger.info("Final result logged to database.")
            except Exception as log_e:
                logger.error(f"Failed to log final result to database: {log_e}", exc_info=True)
        elif not crew_result:
             logger.warning("Crew did not produce a result, nothing logged to DB.")

        # signal.alarm(0)  # Disable alarm
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 