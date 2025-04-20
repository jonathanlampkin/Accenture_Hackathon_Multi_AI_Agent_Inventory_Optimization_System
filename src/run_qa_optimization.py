"""
QA-enabled Inventory Optimization System Runner

This script runs the inventory optimization system with QA agent integration.
It provides a simpler alternative to main.py, focusing on the core functionality.
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
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('output', 'qa_optimizer.log'))
    ]
)

logger = logging.getLogger("QA_Optimizer")

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components
from src.inventory_agent import InventoryAgent
from src.demand_agent import DemandAgent
from src.pricing_agent import PricingAgent
from src.qa_agent import QAAgent
from src.crew_agents import InventoryCrew


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='QA-enabled Inventory Optimization System')
    
    parser.add_argument('--analysis-only', action='store_true',
                        help='Run analysis only, without generating recommendations')
    
    parser.add_argument('--use-crewai', action='store_true', default=False,
                        help='Use CrewAI-based system instead of traditional agent system')
    
    parser.add_argument('--output-dir', type=str, default='output/qa_optimizer',
                        help='Output directory for results')
    
    parser.add_argument('--optimize-for', choices=['cost', 'availability', 'balanced'], 
                        default='balanced',
                        help='Optimization target (cost, availability, or balanced)')
    
    parser.add_argument('--product-id', type=str,
                        help='Specific product ID to focus on')
    
    parser.add_argument('--store-id', type=str,
                        help='Specific store ID to focus on')
    
    return parser.parse_args()


def setup_environment(output_dir):
    """Set up the environment."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")


def run_traditional_analysis(output_dir, opt_target='balanced'):
    """
    Run analysis with all agents, including QA Agent, using the traditional system.
    
    Args:
        output_dir: Output directory for results
        opt_target: Optimization target ('cost', 'availability', 'balanced')
        
    Returns:
        Dict with analysis results
    """
    logger.info(f"Running traditional analysis with optimization target: {opt_target}...")
    
    # Initialize agents
    inventory_agent = InventoryAgent(use_gpu=False)
    demand_agent = DemandAgent(use_gpu=False)
    pricing_agent = PricingAgent(use_gpu=False)
    qa_agent = QAAgent(use_gpu=False)
    
    # Run analysis on individual agents
    inventory_results = inventory_agent.analyze()
    demand_results = demand_agent.analyze()
    pricing_results = pricing_agent.analyze()
    
    # Combine results for QA validation
    combined_results = {
        "inventory": inventory_results,
        "demand": demand_results,
        "pricing": pricing_results,
        "optimization_target": opt_target
    }
    
    # Run QA analysis
    qa_results = qa_agent.analyze(combined_results)
    
    # Combine all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "optimization_target": opt_target,
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


def run_crewai_analysis(output_dir, opt_target='balanced', product_id=None, store_id=None):
    """
    Run analysis using the CrewAI system with QA Agent.
    
    Args:
        output_dir: Output directory for results
        opt_target: Optimization target ('cost', 'availability', 'balanced')
        product_id: Optional specific product ID to focus on
        store_id: Optional specific store ID to focus on
        
    Returns:
        Dict with analysis results
    """
    logger.info(f"Running CrewAI analysis with optimization target: {opt_target}...")
    
    # Initialize CrewAI system
    crew = InventoryCrew(
        data_path=os.path.join('data', 'processed'),
        output_dir=output_dir
    )
    
    # Load data
    try:
        data = pd.read_csv(os.path.join('data', 'processed', 'inventory_data.csv'))
        
        # Filter data if requested
        if product_id:
            data = data[data['product_id'] == product_id]
            logger.info(f"Filtered data for product ID: {product_id}")
        if store_id:
            data = data[data['store_id'] == store_id]
            logger.info(f"Filtered data for store ID: {store_id}")
            
        # Save filtered data for reference
        filtered_data_path = os.path.join(output_dir, 'filtered_data.csv')
        data.to_csv(filtered_data_path, index=False)
        logger.info(f"Saved filtered data to {filtered_data_path}")
    except FileNotFoundError:
        logger.warning("Inventory data file not found. Using synthetic data for demonstration.")
        # Create synthetic data for testing
        data = pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'store_id': ['S001', 'S001', 'S002', 'S002', 'S003'],
            'inventory_level': [100, 75, 50, 200, 125],
            'reorder_point': [25, 20, 15, 50, 30],
            'lead_time_days': [5, 7, 3, 10, 5],
            'price': [9.99, 14.99, 19.99, 24.99, 29.99],
            'cost': [5.99, 7.99, 9.99, 14.99, 19.99],
            'sales_last_30_days': [45, 32, 28, 67, 41]
        })
    
    # Run CrewAI analysis
    results = crew.run_analysis(data)
    
    # Add optimization target info
    results['optimization_target'] = opt_target
    if product_id:
        results['product_filter'] = product_id
    if store_id:
        results['store_filter'] = store_id
        
    logger.info("CrewAI analysis completed successfully")
    return results


def generate_traditional_recommendations(output_dir, analysis_results=None, opt_target='balanced'):
    """
    Generate recommendations using all agents, including QA Agent, with the traditional system.
    
    Args:
        output_dir: Output directory for results
        analysis_results: Optional analysis results to use as input
        opt_target: Optimization target ('cost', 'availability', 'balanced')
        
    Returns:
        Dict with recommendation results
    """
    logger.info(f"Generating traditional recommendations with optimization target: {opt_target}...")
    
    # Initialize agents
    inventory_agent = InventoryAgent(use_gpu=False)
    demand_agent = DemandAgent(use_gpu=False)
    pricing_agent = PricingAgent(use_gpu=False)
    qa_agent = QAAgent(use_gpu=False)
    
    # Update agents with analysis results if provided
    if analysis_results:
        if "inventory_analysis" in analysis_results:
            inventory_agent.update({"analysis_results": analysis_results["inventory_analysis"]})
        if "demand_analysis" in analysis_results:
            demand_agent.update({"analysis_results": analysis_results["demand_analysis"]})
        if "pricing_analysis" in analysis_results:
            pricing_agent.update({"analysis_results": analysis_results["pricing_analysis"]})
    
    # Get recommendations from each agent
    inventory_recommendations = inventory_agent.make_recommendation()
    demand_recommendations = demand_agent.make_recommendation()
    pricing_recommendations = pricing_agent.make_recommendation()
    
    # Combine recommendations for QA validation
    combined_recommendations = {
        "inventory": inventory_recommendations,
        "demand": demand_recommendations,
        "pricing": pricing_recommendations,
        "optimization_target": opt_target
    }
    
    # Get QA recommendations (which validate other recommendations)
    qa_recommendations = qa_agent.make_recommendation()
    
    # Combine all recommendations
    all_recommendations = {
        "timestamp": datetime.now().isoformat(),
        "optimization_target": opt_target,
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


def run_crewai_optimization(output_dir, opt_target='balanced', product_id=None, store_id=None):
    """
    Run full optimization using the CrewAI system with QA Agent.
    
    Args:
        output_dir: Output directory for results
        opt_target: Optimization target ('cost', 'availability', 'balanced')
        product_id: Optional specific product ID to focus on
        store_id: Optional specific store ID to focus on
        
    Returns:
        Dict with optimization results
    """
    logger.info(f"Running CrewAI optimization with target: {opt_target}...")
    
    # Initialize CrewAI system
    crew = InventoryCrew(
        data_path=os.path.join('data', 'processed'),
        output_dir=output_dir
    )
    
    # Load data
    try:
        data = pd.read_csv(os.path.join('data', 'processed', 'inventory_data.csv'))
        
        # Filter data if requested
        if product_id:
            data = data[data['product_id'] == product_id]
            logger.info(f"Filtered data for product ID: {product_id}")
        if store_id:
            data = data[data['store_id'] == store_id]
            logger.info(f"Filtered data for store ID: {store_id}")
    except FileNotFoundError:
        logger.warning("Inventory data file not found. Using synthetic data for demonstration.")
        # Create synthetic data for testing
        data = pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'store_id': ['S001', 'S001', 'S002', 'S002', 'S003'],
            'inventory_level': [100, 75, 50, 200, 125],
            'reorder_point': [25, 20, 15, 50, 30],
            'lead_time_days': [5, 7, 3, 10, 5],
            'price': [9.99, 14.99, 19.99, 24.99, 29.99],
            'cost': [5.99, 7.99, 9.99, 14.99, 19.99],
            'sales_last_30_days': [45, 32, 28, 67, 41]
        })
    
    # Run CrewAI optimization
    results = crew.run_optimization(data)
    
    # Add optimization target info
    results['optimization_target'] = opt_target
    if product_id:
        results['product_filter'] = product_id
    if store_id:
        results['store_filter'] = store_id
        
    logger.info("CrewAI optimization completed successfully")
    return results


def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("QA-ENABLED INVENTORY OPTIMIZATION SUMMARY")
    print("="*80)
    
    # Print optimization target
    print(f"\nOptimization Target: {results.get('optimization_target', 'balanced').upper()}")
    
    # Print QA analysis summary if available
    if "qa_analysis" in results:
        print(f"\nQA ANALYSIS SUMMARY:")
        qa = results["qa_analysis"]
        print(f"  Validated recommendations: {qa.get('validated_count', 0)}")
        print(f"  Rejected recommendations: {qa.get('rejected_count', 0)}")
        print(f"  Consistency score: {qa.get('consistency_score', 'N/A')}")
        print(f"  Feasibility score: {qa.get('feasibility_score', 'N/A')}")
        print(f"  Alignment score: {qa.get('alignment_score', 'N/A')}")
    
    # Print QA recommendations if available
    if "qa_recommendations" in results:
        print(f"\nQA RECOMMENDATIONS:")
        qa_recs = results["qa_recommendations"]
        if "high_priority" in qa_recs:
            print("  HIGH PRIORITY:")
            for i, rec in enumerate(qa_recs["high_priority"]):
                print(f"    {i+1}. [{rec.get('source', 'qa').upper()}] {rec.get('action')}")
                print(f"       Impact: {rec.get('impact')}")
                print(f"       Confidence: {rec.get('confidence', 'N/A')} | Priority: {rec.get('priority_score', 'N/A')}")
        
        if "medium_priority" in qa_recs and qa_recs["medium_priority"]:
            print("\n  MEDIUM PRIORITY:")
            for i, rec in enumerate(qa_recs["medium_priority"]):
                print(f"    {i+1}. [{rec.get('source', 'qa').upper()}] {rec.get('action')}")
                print(f"       Impact: {rec.get('impact')}")
    
    # Print high-priority recommendations from other agents if available
    for agent_type in ["inventory_recommendations", "demand_recommendations", "pricing_recommendations"]:
        if agent_type in results and "high_priority" in results[agent_type]:
            agent_name = agent_type.split("_")[0].upper()
            print(f"\n{agent_name} RECOMMENDATIONS (HIGH PRIORITY):")
            for i, rec in enumerate(results[agent_type]["high_priority"][:2]):  # Show only top 2
                print(f"    {i+1}. [{rec.get('source', agent_name).upper()}] {rec.get('action')}")
                print(f"       Impact: {rec.get('impact')}")
    
    # Print summary footer
    print("\n" + "="*80)
    timestamp = results.get("timestamp", datetime.now().isoformat())
    print(f"Analysis timestamp: {timestamp}")
    print("="*80 + "\n")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up environment
    setup_environment(args.output_dir)
    
    try:
        if args.analysis_only:
            logger.info("Running analysis only...")
            
            if args.use_crewai:
                # Run CrewAI analysis
                results = run_crewai_analysis(
                    output_dir=args.output_dir,
                    opt_target=args.optimize_for,
                    product_id=args.product_id,
                    store_id=args.store_id
                )
            else:
                # Run traditional analysis
                results = run_traditional_analysis(
                    output_dir=args.output_dir,
                    opt_target=args.optimize_for
                )
        else:
            logger.info("Running full optimization...")
            
            if args.use_crewai:
                # Run CrewAI optimization
                results = run_crewai_optimization(
                    output_dir=args.output_dir,
                    opt_target=args.optimize_for,
                    product_id=args.product_id,
                    store_id=args.store_id
                )
            else:
                # Run traditional analysis first
                analysis_results = run_traditional_analysis(
                    output_dir=args.output_dir,
                    opt_target=args.optimize_for
                )
                
                # Then generate recommendations based on analysis
                recommendation_results = generate_traditional_recommendations(
                    output_dir=args.output_dir,
                    analysis_results=analysis_results,
                    opt_target=args.optimize_for
                )
                
                # Combine results
                results = {**analysis_results, **recommendation_results}
        
        # Print summary
        print_summary(results)
        
        logger.info("QA-enabled inventory optimization completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 