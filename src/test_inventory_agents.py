"""
Simple test script to validate that InventoryAgents works correctly.
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TestInventoryAgents")

# Import the InventoryAgents class
from agents import InventoryAgents

def main():
    """Test InventoryAgents functionality."""
    logger.info("Testing InventoryAgents initialization...")
    
    try:
        # Initialize InventoryAgents with default parameters
        inventory_agents = InventoryAgents(
            model_name="llama3", 
            ollama_base_url="http://localhost:11434",
            use_gpu=False
        )
        
        logger.info("InventoryAgents initialized successfully")
        
        # Test getting all agents
        logger.info("Testing get_all_agents()...")
        agents = inventory_agents.get_all_agents()
        logger.info(f"Retrieved {len(agents)} agents")
        
        # Test analyze method
        logger.info("Testing analyze()...")
        analysis_results = inventory_agents.analyze()
        logger.info(f"Analysis results: {analysis_results}")
        
        # Test make_recommendation method
        logger.info("Testing make_recommendation()...")
        recommendation_results = inventory_agents.make_recommendation()
        logger.info(f"Recommendation results: {recommendation_results}")
        
        # Test receive_message method
        logger.info("Testing receive_message()...")
        inventory_agents.receive_message({"type": "test", "content": "This is a test message"})
        
        # Test update method
        logger.info("Testing update()...")
        inventory_agents.update({
            "critical_products": ["test_product1", "test_product2"],
            "excess_inventory": ["test_product3"],
            "safety_stock_levels": {"test_product1": 100},
            "reorder_points": {"test_product1": 50}
        })
        
        logger.info(f"Updated state: {inventory_agents.state}")
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 