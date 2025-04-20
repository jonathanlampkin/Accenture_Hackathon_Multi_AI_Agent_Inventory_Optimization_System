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

# Import local config
import src.config as config

# Import agent classes
from src.agents import InventoryAgents
from src.agent_implementations import DemandAgent, PricingAgent, CoordinationAgent
from src.coordinator import MultiAgentCoordinator
from src.qa_agent import QAAgent

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.utils.error_handler import EnhancedErrorHandler
from src.utils.task_manager import TaskManager
from src.utils.knowledge_manager import KnowledgeManager
from src.utils.memory_manager import MemoryManager
from src.utils.monitoring import MonitoringSystem
from src.utils.security import SecurityManager
from src.utils.resource_manager import ResourceManager
from src.utils.communication import CommunicationManager
from src.utils.training import TrainingManager
from src.utils.integration import IntegrationManager

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
        tuple: Initialized agents (inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent)
    """
    logger.info("Initializing agents...")
    
    # Initialize specialized agents
    inventory_agents = InventoryAgents()
    demand_agent = DemandAgent()
    pricing_agent = PricingAgent()
    qa_agent = QAAgent()
    
    # Initialize coordination agent with references to specialized agents
    coordination_agent = CoordinationAgent(
        inventory_agents=inventory_agents,
        demand_agent=demand_agent,
        pricing_agent=pricing_agent
    )
    
    logger.info("All agents initialized successfully")
    
    return inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent

def run_analysis(inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent):
    """
    Run analysis on all agents.
    
    Args:
        inventory_agents: InventoryAgents instance
        demand_agent: DemandAgent instance
        pricing_agent: PricingAgent instance
        qa_agent: QAAgent instance
        coordination_agent: CoordinationAgent instance
    """
    logger.info("Starting analysis on all agents...")
    
    # Get all inventory agents
    inventory_agents_list = inventory_agents.get_all_agents()
    
    # Run analysis on specialized agents
    logger.info("Running inventory analysis...")
    inventory_analysis = {
        "demand_analyst": inventory_agents_list[0].analyze() if len(inventory_agents_list) > 0 else None,
        "inventory_optimizer": inventory_agents_list[1].analyze() if len(inventory_agents_list) > 1 else None,
        "supply_chain_analyst": inventory_agents_list[2].analyze() if len(inventory_agents_list) > 2 else None,
        "risk_analyst": inventory_agents_list[3].analyze() if len(inventory_agents_list) > 3 else None
    }
    
    logger.info("Running demand analysis...")
    demand_analysis = demand_agent.analyze()
    
    logger.info("Running pricing analysis...")
    pricing_analysis = pricing_agent.analyze()
    
    # Run coordination analysis
    logger.info("Running coordination analysis...")
    coordination_analysis = coordination_agent.analyze()
    
    # Combined analysis results for QA
    combined_analysis = {
        'inventory': inventory_analysis,
        'demand': demand_analysis,
        'pricing': pricing_analysis
    }
    
    # Run QA analysis
    logger.info("Running QA analysis...")
    qa_analysis = qa_agent.analyze(combined_analysis)
    
    logger.info("All analyses completed successfully")
    
    return {
        'inventory': inventory_analysis,
        'demand': demand_analysis,
        'pricing': pricing_analysis,
        'coordination': coordination_analysis,
        'qa': qa_analysis
    }

def generate_recommendations(inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent):
    """
    Generate recommendations from all agents.
    
    Args:
        inventory_agents: InventoryAgents instance
        demand_agent: DemandAgent instance
        pricing_agent: PricingAgent instance
        qa_agent: QAAgent instance
        coordination_agent: CoordinationAgent instance
    """
    logger.info("Generating recommendations from all agents...")
    
    # Get all inventory agents
    inventory_agents_list = inventory_agents.get_all_agents()
    
    # Generate recommendations from specialized agents
    logger.info("Generating inventory recommendations...")
    inventory_recommendations = {
        "demand_analyst": inventory_agents_list[0].make_recommendation() if len(inventory_agents_list) > 0 else None,
        "inventory_optimizer": inventory_agents_list[1].make_recommendation() if len(inventory_agents_list) > 1 else None,
        "supply_chain_analyst": inventory_agents_list[2].make_recommendation() if len(inventory_agents_list) > 2 else None,
        "risk_analyst": inventory_agents_list[3].make_recommendation() if len(inventory_agents_list) > 3 else None
    }
    
    logger.info("Generating demand recommendations...")
    demand_recommendations = demand_agent.make_recommendation()
    
    logger.info("Generating pricing recommendations...")
    pricing_recommendations = pricing_agent.make_recommendation()
    
    # Combine recommendations for QA
    combined_recommendations = {
        'inventory': inventory_recommendations,
        'demand': demand_recommendations,
        'pricing': pricing_recommendations
    }
    
    # Generate QA recommendations
    logger.info("Generating QA recommendations...")
    qa_recommendations = qa_agent.make_recommendation()
    
    # Generate integrated recommendations
    logger.info("Generating integrated recommendations...")
    integrated_recommendations = coordination_agent.make_recommendation()
    
    logger.info("All recommendations generated successfully")
    
    return {
        'inventory': inventory_recommendations,
        'demand': demand_recommendations,
        'pricing': pricing_recommendations,
        'qa': qa_recommendations,
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
    
    # Print QA recommendations
    if 'qa' in recommendation_results and 'high_priority' in recommendation_results['qa']:
        print("\nQA RECOMMENDATIONS:")
        qa_high_priority = recommendation_results['qa']['high_priority']
        for i, rec in enumerate(qa_high_priority):
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

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Multi-Agent Inventory Optimization System')
    
    parser.add_argument('--optimize-for', choices=['cost', 'availability', 'balanced'], default='balanced',
                        help='Optimization target (cost, availability, or balanced)')
    
    parser.add_argument('--product-id', type=str, help='Specific product ID to focus on')
    
    parser.add_argument('--store-id', type=str, help='Specific store ID to focus on')
    
    parser.add_argument('--iterations', type=int, default=5,
                        help='Maximum number of optimization iterations')
    
    parser.add_argument('--output-dir', type=str,
                        help='Custom output directory')
    
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    
    parser.add_argument('--use-crewai', action='store_true', default=True,
                        help='Use the CrewAI-based system instead of traditional multi-agent system')
    
    parser.add_argument('--traditional', action='store_true',
                        help='Use the traditional multi-agent system instead of CrewAI')
    
    parser.add_argument('--analysis-only', action='store_true',
                        help='Run only the analysis phase without recommendations')
    
    return parser.parse_args()

def main():
    """
    Main entry point for the Multi-Agent Inventory Optimization System.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # If traditional flag is set, it overrides use-crewai
    if args.traditional:
        args.use_crewai = False
        
    try:
        # Initialize coordinator
        coordinator = MultiAgentCoordinator(
            optimization_target=args.optimize_for,
            product_id=args.product_id,
            store_id=args.store_id,
            max_iterations=args.iterations,
            output_dir=args.output_dir,
            use_gpu=args.use_gpu,
            use_crewai=args.use_crewai
        )
        
        # Run optimization
        if args.analysis_only:
            logger.info("Running analysis only...")
            
            if args.use_crewai:
                # Load data
                data = pd.read_csv(os.path.join('data', 'processed', 'inventory_data.csv'))
                
                # Run crewAI optimization with analysis only
                results = coordinator.crew.run_analysis(data)
            else:
                # Initialize traditional agents
                inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent = initialize_agents()
                
                # Run traditional analysis
                results = run_analysis(inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent)
                
            logger.info("Analysis completed successfully")
        else:
            logger.info("Running full optimization...")
            
            if args.use_crewai:
                results = coordinator.run_optimization()
            else:
                # Initialize traditional agents
                inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent = initialize_agents()
                
                # Run analysis
                analysis_results = run_analysis(inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent)
                
                # Generate recommendations
                recommendation_results = generate_recommendations(inventory_agents, demand_agent, pricing_agent, qa_agent, coordination_agent)
                
                # Save results
                save_results(analysis_results, recommendation_results)
                
                results = recommendation_results
                
            logger.info("Optimization completed successfully")
        
        # Print summary
        print_summary(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

# Load configuration
def load_config() -> Dict[str, Any]:
    """
    Load configuration or return default values if config file not found
    
    Returns:
        Dict: Configuration values
    """
    try:
        config_file = Path("config.json")
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        else:
            logger.warning("Configuration file not found. Using default values.")
            return {
                "monitoring": {
                    "interval": 60,
                    "metrics_dir": "metrics",
                    "alerts": []
                },
                "security": {
                    "secret_key": "default_secret_key_for_development",
                    "algorithm": "HS256"
                },
                "resources": {
                    "cpu_limit": 80,
                    "memory_limit": 80,
                    "disk_limit": 80,
                    "network_limit": 80
                },
                "training": {
                    "batch_size": 32,
                    "epochs": 10,
                    "learning_rate": 0.001,
                    "validation_split": 0.2
                },
                "api": {
                    "host": "127.0.0.1",
                    "port": 8000
                }
            }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

# Load configuration
config_data = load_config()

# Initialize managers
error_handler = EnhancedErrorHandler()
task_manager = TaskManager()
knowledge_manager = KnowledgeManager()
memory_manager = MemoryManager()
monitoring_system = MonitoringSystem(
    storage_dir=config_data["monitoring"]["metrics_dir"]
)
security_manager = SecurityManager(
    config_file="security_config.json"
)
resource_manager = ResourceManager(
    config_file="resource_config.json"
)
communication_manager = CommunicationManager()
training_manager = TrainingManager(
    config_file="training_config.json"
)
integration_manager = IntegrationManager()

# Main entry point
if __name__ == "__main__":
    sys.exit(main()) 