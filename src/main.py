"""
Main script for the Multi-Agent Inventory Optimization System.
This script initializes and runs the multi-agent system with all specialized agents
and the coordination agent.
"""

import os
import sys
import json
import pandas as pd
import logging
import argparse
from datetime import datetime
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import agent classes
from src.agents import (
    InventoryAgent,
    DemandAgent,
    PricingAgent,
    CoordinationAgent
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.error_handler import EnhancedErrorHandler
from utils.task_manager import TaskManager
from utils.knowledge_manager import KnowledgeManager
from utils.memory_manager import MemoryManager
from utils.monitoring import MonitoringSystem
from utils.security import SecurityManager
from utils.resource_manager import ResourceManager
from utils.communication import CommunicationManager
from utils.training import TrainingManager
from utils.integration import IntegrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'system.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Main")

def setup_environment():
    """
    Set up the environment by creating necessary directories.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    logger.info(f"Environment setup completed. Output directory: {config.OUTPUT_DIR}")

def initialize_agents():
    """
    Initialize all agents in the system.
    
    Returns:
        tuple: Initialized agents (inventory_agent, demand_agent, pricing_agent, coordination_agent)
    """
    logger.info("Initializing agents...")
    
    # Initialize specialized agents
    inventory_agent = InventoryAgent()
    demand_agent = DemandAgent()
    pricing_agent = PricingAgent()
    
    # Initialize coordination agent with references to specialized agents
    coordination_agent = CoordinationAgent(
        inventory_agent=inventory_agent,
        demand_agent=demand_agent,
        pricing_agent=pricing_agent
    )
    
    logger.info("All agents initialized successfully")
    
    return inventory_agent, demand_agent, pricing_agent, coordination_agent

def run_analysis(inventory_agent, demand_agent, pricing_agent, coordination_agent):
    """
    Run analysis on all agents.
    
    Args:
        inventory_agent: InventoryAgent instance
        demand_agent: DemandAgent instance
        pricing_agent: PricingAgent instance
        coordination_agent: CoordinationAgent instance
    """
    logger.info("Starting analysis on all agents...")
    
    # Run analysis on specialized agents
    logger.info("Running inventory analysis...")
    inventory_analysis = inventory_agent.analyze()
    
    logger.info("Running demand analysis...")
    demand_analysis = demand_agent.analyze()
    
    logger.info("Running pricing analysis...")
    pricing_analysis = pricing_agent.analyze()
    
    # Run coordination analysis
    logger.info("Running coordination analysis...")
    coordination_analysis = coordination_agent.analyze()
    
    logger.info("All analyses completed successfully")
    
    return {
        'inventory': inventory_analysis,
        'demand': demand_analysis,
        'pricing': pricing_analysis,
        'coordination': coordination_analysis
    }

def generate_recommendations(inventory_agent, demand_agent, pricing_agent, coordination_agent):
    """
    Generate recommendations from all agents.
    
    Args:
        inventory_agent: InventoryAgent instance
        demand_agent: DemandAgent instance
        pricing_agent: PricingAgent instance
        coordination_agent: CoordinationAgent instance
    """
    logger.info("Generating recommendations from all agents...")
    
    # Generate recommendations from specialized agents
    logger.info("Generating inventory recommendations...")
    inventory_recommendations = inventory_agent.make_recommendation()
    
    logger.info("Generating demand recommendations...")
    demand_recommendations = demand_agent.make_recommendation()
    
    logger.info("Generating pricing recommendations...")
    pricing_recommendations = pricing_agent.make_recommendation()
    
    # Generate integrated recommendations
    logger.info("Generating integrated recommendations...")
    integrated_recommendations = coordination_agent.make_recommendation()
    
    logger.info("All recommendations generated successfully")
    
    return {
        'inventory': inventory_recommendations,
        'demand': demand_recommendations,
        'pricing': pricing_recommendations,
        'integrated': integrated_recommendations
    }

def save_results(analysis_results, recommendation_results):
    """
    Save analysis and recommendation results to JSON files.
    
    Args:
        analysis_results: Dictionary with analysis results from all agents
        recommendation_results: Dictionary with recommendation results from all agents
    """
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save analysis results
    analysis_file = os.path.join(config.OUTPUT_DIR, f"analysis_results_{timestamp}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save recommendation results
    recommendation_file = os.path.join(config.OUTPUT_DIR, f"recommendation_results_{timestamp}.json")
    with open(recommendation_file, 'w') as f:
        json.dump(recommendation_results, f, indent=2)
    
    logger.info(f"Results saved to {analysis_file} and {recommendation_file}")

def print_summary(recommendation_results):
    """
    Print a summary of the high-priority recommendations.
    
    Args:
        recommendation_results: Dictionary with recommendation results from all agents
    """
    print("\n" + "="*80)
    print("MULTI-AGENT INVENTORY OPTIMIZATION SYSTEM - SUMMARY")
    print("="*80)
    
    # Print high-priority integrated recommendations
    print("\nHIGH PRIORITY RECOMMENDATIONS:")
    high_priority = recommendation_results['integrated']['high_priority']
    for i, rec in enumerate(high_priority):
        print(f"{i+1}. [{rec['source'].upper()}] {rec['action']}")
        print(f"   Impact: {rec['impact']} | Confidence: {rec['confidence']:.2f} | Priority: {rec['priority_score']}")
    
    # Print conflict resolutions
    print("\nCONFLICT RESOLUTIONS:")
    conflict_resolutions = recommendation_results['integrated']['conflict_resolutions']
    for i, rec in enumerate(conflict_resolutions):
        print(f"{i+1}. {rec['action']}")
        print(f"   Rationale: {rec['rationale']}")
    
    print("\n" + "="*80)
    print(f"Total recommendations generated: {sum(len(recs) for recs in recommendation_results['integrated'].values() if isinstance(recs, list))}")
    print(f"Results saved to {config.OUTPUT_DIR}")
    print("="*80 + "\n")

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the Multi-Agent Inventory Optimization System')
    parser.add_argument('--analysis-only', action='store_true', help='Run only the analysis phase, not recommendations')
    parser.add_argument('--recommendations-only', action='store_true', help='Run only the recommendations phase, not analysis')
    parser.add_argument('--no-summary', action='store_true', help='Do not print the summary of recommendations')
    return parser.parse_args()

def main():
    """
    Main function to run the Multi-Agent Inventory Optimization System.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up environment
    setup_environment()
    
    # Initialize agents
    inventory_agent, demand_agent, pricing_agent, coordination_agent = initialize_agents()
    
    analysis_results = None
    recommendation_results = None
    
    # Run analysis if requested or if both analysis and recommendations are requested
    if not args.recommendations_only:
        analysis_results = run_analysis(inventory_agent, demand_agent, pricing_agent, coordination_agent)
    
    # Generate recommendations if requested or if both analysis and recommendations are requested
    if not args.analysis_only:
        recommendation_results = generate_recommendations(inventory_agent, demand_agent, pricing_agent, coordination_agent)
    
    # Save results if both analysis and recommendations were run
    if analysis_results and recommendation_results:
        save_results(analysis_results, recommendation_results)
    
    # Print summary if requested
    if recommendation_results and not args.no_summary:
        print_summary(recommendation_results)
    
    logger.info("Multi-Agent Inventory Optimization System execution completed")

# Load configuration
def load_config() -> Dict[str, Any]:
    config_file = Path("config.json")
    if not config_file.exists():
        raise FileNotFoundError("Configuration file not found")
        
    with open(config_file) as f:
        return json.load(f)

# Initialize managers
config = load_config()

error_handler = EnhancedErrorHandler()
task_manager = TaskManager()
knowledge_manager = KnowledgeManager()
memory_manager = MemoryManager()
monitoring_system = MonitoringSystem(
    interval=config["monitoring"]["interval"],
    metrics_dir=config["monitoring"]["metrics_dir"],
    alerts=config["monitoring"]["alerts"]
)
security_manager = SecurityManager(
    secret_key=config["security"]["secret_key"],
    algorithm=config["security"]["algorithm"]
)
resource_manager = ResourceManager(
    cpu_limit=config["resources"]["cpu_limit"],
    memory_limit=config["resources"]["memory_limit"],
    disk_limit=config["resources"]["disk_limit"],
    network_limit=config["resources"]["network_limit"]
)
communication_manager = CommunicationManager()
training_manager = TrainingManager(
    batch_size=config["training"]["batch_size"],
    epochs=config["training"]["epochs"],
    learning_rate=config["training"]["learning_rate"],
    validation_split=config["training"]["validation_split"]
)
integration_manager = IntegrationManager()

# Create FastAPI app
app = FastAPI(
    title="Multi-AI Agent Inventory Optimization System",
    description="A comprehensive system for optimizing inventory management using multiple AI agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Error handler endpoint
@app.get("/api/v1/errors")
async def get_errors():
    return error_handler.get_error_metrics()

# Task manager endpoint
@app.get("/api/v1/tasks")
async def get_tasks():
    return task_manager.get_task_metrics()

# Knowledge manager endpoint
@app.get("/api/v1/knowledge")
async def get_knowledge():
    return knowledge_manager.get_knowledge_metrics()

# Memory manager endpoint
@app.get("/api/v1/memory")
async def get_memory():
    return memory_manager.get_memory_metrics()

# Monitoring system endpoint
@app.get("/api/v1/monitoring")
async def get_monitoring():
    return monitoring_system.get_metrics()

# Resource manager endpoint
@app.get("/api/v1/resources")
async def get_resources():
    return resource_manager.get_resource_metrics()

# Training manager endpoint
@app.get("/api/v1/training")
async def get_training():
    return training_manager.get_training_metrics()

# Integration manager endpoint
@app.get("/api/v1/integration")
async def get_integration():
    return integration_manager.get_integration_metrics()

# Start the system
async def start_system():
    # Start managers
    monitoring_system.start()
    resource_manager.start()
    communication_manager.start()
    integration_manager.start()
    
    # Start FastAPI server
    config = uvicorn.Config(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

# Main entry point
if __name__ == "__main__":
    try:
        asyncio.run(start_system())
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
    except Exception as e:
        logger.error(f"Error starting system: {str(e)}")
        raise 