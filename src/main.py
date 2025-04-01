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

if __name__ == "__main__":
    main() 