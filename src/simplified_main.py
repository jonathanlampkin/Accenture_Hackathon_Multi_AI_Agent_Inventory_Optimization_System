"""
Simplified main script for the Multi-Agent Inventory Optimization System.

This script demonstrates the integration of the QA Agent in the inventory optimization process.
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Simplified_Main")

# Import necessary components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inventory_agent import InventoryAgent
from src.demand_agent import DemandAgent
from src.pricing_agent import PricingAgent
from src.qa_agent import QAAgent


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Simplified Inventory Optimization System')
    
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis only')
    
    parser.add_argument('--output-dir', type=str, default='output/simplified',
                        help='Output directory for results')
    
    return parser.parse_args()


def setup_environment(output_dir):
    """Set up the environment."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")


def run_analysis(output_dir):
    """Run analysis with all agents, including QA Agent."""
    logger.info("Running analysis with all agents...")
    
    # Initialize agents
    inventory_agent = InventoryAgent()
    demand_agent = DemandAgent()
    pricing_agent = PricingAgent()
    qa_agent = QAAgent()
    
    # Run analysis on individual agents
    inventory_results = inventory_agent.analyze()
    demand_results = demand_agent.analyze()
    pricing_results = pricing_agent.analyze()
    
    # Combine results for QA validation
    combined_results = {
        "inventory": inventory_results,
        "demand": demand_results,
        "pricing": pricing_results
    }
    
    # Run QA analysis
    qa_results = qa_agent.analyze(combined_results)
    
    # Combine all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "inventory_analysis": inventory_results,
        "demand_analysis": demand_results,
        "pricing_analysis": pricing_results,
        "qa_analysis": qa_results
    }
    
    # Save results
    output_file = os.path.join(output_dir, f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Analysis results saved to {output_file}")
    return all_results


def make_recommendations(output_dir):
    """Generate and validate recommendations."""
    logger.info("Generating recommendations...")
    
    # Initialize agents
    inventory_agent = InventoryAgent()
    demand_agent = DemandAgent()
    pricing_agent = PricingAgent()
    qa_agent = QAAgent()
    
    # Get recommendations from each agent
    inventory_recommendations = inventory_agent.make_recommendation()
    demand_recommendations = demand_agent.make_recommendation()
    pricing_recommendations = pricing_agent.make_recommendation()
    
    # Combine recommendations for QA validation
    combined_recommendations = {
        "inventory": inventory_recommendations,
        "demand": demand_recommendations,
        "pricing": pricing_recommendations
    }
    
    # Get QA recommendations (which validate other recommendations)
    qa_recommendations = qa_agent.make_recommendation()
    
    # Combine all recommendations
    all_recommendations = {
        "timestamp": datetime.now().isoformat(),
        "inventory_recommendations": inventory_recommendations,
        "demand_recommendations": demand_recommendations,
        "pricing_recommendations": pricing_recommendations,
        "qa_recommendations": qa_recommendations
    }
    
    # Save results
    output_file = os.path.join(output_dir, f"recommendation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(all_recommendations, f, indent=2, default=str)
    
    logger.info(f"Recommendation results saved to {output_file}")
    return all_recommendations


def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("INVENTORY OPTIMIZATION SUMMARY")
    print("="*80)
    
    if "qa_analysis" in results:
        print(f"\nQA ANALYSIS SUMMARY:")
        qa = results["qa_analysis"]
        print(f"  Validated recommendations: {qa.get('validated_count', 0)}")
        print(f"  Rejected recommendations: {qa.get('rejected_count', 0)}")
        print(f"  Consistency score: {qa.get('consistency_score', 'N/A')}")
        print(f"  Feasibility score: {qa.get('feasibility_score', 'N/A')}")
        print(f"  Alignment score: {qa.get('alignment_score', 'N/A')}")
    
    if "qa_recommendations" in results:
        print(f"\nQA RECOMMENDATIONS:")
        qa_recs = results["qa_recommendations"]
        if "high_priority" in qa_recs:
            print("  HIGH PRIORITY:")
            for rec in qa_recs["high_priority"]:
                print(f"    - {rec.get('action')}: {rec.get('impact')}")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up environment
    setup_environment(args.output_dir)
    
    try:
        # Run analysis
        results = run_analysis(args.output_dir)
        
        # Generate recommendations if not analysis only
        if not args.analyze:
            recommendation_results = make_recommendations(args.output_dir)
            results.update(recommendation_results)
        
        # Print summary
        print_summary(results)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 