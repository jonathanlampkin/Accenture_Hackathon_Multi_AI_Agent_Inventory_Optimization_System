"""
Inventory optimization agent factory and crew setup.

This module provides classes for creating specialized agents and coordinating
their interactions for inventory optimization tasks.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import pandas as pd

# Attempt to import tools
try:
    from .tools import (
        forecast_demand,
        calculate_safety_stock,
        calculate_reorder_point,
        identify_anomalies,
        analyze_product_performance
    )
except ImportError:
    logging.warning("Could not import tools from the same package")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InventoryAgentFactory:
    """Factory class for creating specialized inventory optimization agents."""
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        model: Optional[Any] = None
    ):
        """
        Initialize the agent factory.
        
        Args:
            config_path: Path to agent configuration JSON file
            model: Language model to use for agents (defaults to OpenAI if None)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up the language model
        self.model = model or ChatOpenAI(
            model="gpt-4",
            temperature=0.2
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load agent configuration from a JSON file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with agent configurations
        """
        default_config = {
            "forecasting_agent": {
                "name": "Demand Forecaster",
                "role": "Inventory Demand Forecasting Specialist",
                "goal": "Analyze historical sales data to generate accurate demand forecasts for inventory planning",
                "backstory": "You are an experienced data scientist specializing in time series forecasting for retail and supply chain. You've helped numerous companies optimize their inventory levels through accurate demand predictions."
            },
            "optimization_agent": {
                "name": "Inventory Optimizer",
                "role": "Inventory Optimization Specialist",
                "goal": "Calculate optimal inventory levels, safety stocks, and reorder points to minimize costs while maintaining service levels",
                "backstory": "You are a supply chain optimization expert with years of experience in inventory management. You understand the delicate balance between inventory costs and service levels."
            },
            "anomaly_detection_agent": {
                "name": "Anomaly Detector",
                "role": "Data Anomaly Detection Specialist",
                "goal": "Identify unusual patterns, outliers, and anomalies in inventory and sales data",
                "backstory": "You are a data analytics expert who specializes in detecting unusual patterns and anomalies in complex datasets. Your insights help companies identify issues before they become problems."
            },
            "scenario_planning_agent": {
                "name": "Scenario Planner",
                "role": "Inventory Scenario Planning Specialist",
                "goal": "Develop scenarios and recommendations for inventory optimization under different conditions",
                "backstory": "You are a strategic planner who excels at modeling different scenarios to prepare organizations for various potential futures. Your scenario planning helps companies make robust inventory decisions."
            },
            "coordinator_agent": {
                "name": "Inventory Coordinator",
                "role": "Inventory Analysis Coordinator",
                "goal": "Coordinate analysis across specialists and synthesize findings into actionable recommendations",
                "backstory": "You are a senior supply chain consultant who excels at coordinating complex analyses and synthesizing insights from various specialists. You translate technical findings into clear, actionable recommendations."
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded agent configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}. Using default configuration.")
                return default_config
        else:
            logger.info("Using default agent configuration")
            return default_config
    
    def create_forecasting_agent(self) -> Agent:
        """
        Create a forecasting agent specialized in demand prediction.
        
        Returns:
            Agent configured for forecasting tasks
        """
        config = self.config.get("forecasting_agent", {})
        
        return Agent(
            name=config.get("name", "Demand Forecaster"),
            role=config.get("role", "Inventory Demand Forecasting Specialist"),
            goal=config.get("goal", "Generate accurate demand forecasts"),
            backstory=config.get("backstory", "You are an experienced forecasting specialist"),
            verbose=True,
            llm=self.model,
            tools=[forecast_demand]
        )
    
    def create_optimization_agent(self) -> Agent:
        """
        Create an optimization agent specialized in inventory level calculations.
        
        Returns:
            Agent configured for optimization tasks
        """
        config = self.config.get("optimization_agent", {})
        
        return Agent(
            name=config.get("name", "Inventory Optimizer"),
            role=config.get("role", "Inventory Optimization Specialist"),
            goal=config.get("goal", "Calculate optimal inventory parameters"),
            backstory=config.get("backstory", "You are an experienced optimization specialist"),
            verbose=True,
            llm=self.model,
            tools=[calculate_safety_stock, calculate_reorder_point]
        )
    
    def create_anomaly_detection_agent(self) -> Agent:
        """
        Create an anomaly detection agent specialized in identifying unusual patterns.
        
        Returns:
            Agent configured for anomaly detection tasks
        """
        config = self.config.get("anomaly_detection_agent", {})
        
        return Agent(
            name=config.get("name", "Anomaly Detector"),
            role=config.get("role", "Data Anomaly Detection Specialist"),
            goal=config.get("goal", "Identify unusual patterns and anomalies"),
            backstory=config.get("backstory", "You are an experienced data analytics expert"),
            verbose=True,
            llm=self.model,
            tools=[identify_anomalies]
        )
    
    def create_scenario_planning_agent(self) -> Agent:
        """
        Create a scenario planning agent specialized in modeling different futures.
        
        Returns:
            Agent configured for scenario planning tasks
        """
        config = self.config.get("scenario_planning_agent", {})
        
        return Agent(
            name=config.get("name", "Scenario Planner"),
            role=config.get("role", "Inventory Scenario Planning Specialist"),
            goal=config.get("goal", "Develop scenarios and recommendations"),
            backstory=config.get("backstory", "You are an experienced strategic planner"),
            verbose=True,
            llm=self.model,
            tools=[]  # No specific tools for now
        )
    
    def create_coordinator_agent(self) -> Agent:
        """
        Create a coordinator agent to synthesize insights from specialists.
        
        Returns:
            Agent configured for coordination tasks
        """
        config = self.config.get("coordinator_agent", {})
        
        return Agent(
            name=config.get("name", "Inventory Coordinator"),
            role=config.get("role", "Inventory Analysis Coordinator"),
            goal=config.get("goal", "Synthesize findings into actionable recommendations"),
            backstory=config.get("backstory", "You are a senior supply chain consultant"),
            verbose=True,
            llm=self.model,
            tools=[analyze_product_performance]
        )


class InventoryOptimizationCrew:
    """Crew to orchestrate inventory optimization agents."""
    
    def __init__(
        self,
        agent_factory: InventoryAgentFactory,
        data_path: str,
        output_dir: str = "./output",
        process: Process = Process.sequential
    ):
        """
        Initialize the inventory optimization crew.
        
        Args:
            agent_factory: Factory to create specialized agents
            data_path: Path to historical data file
            output_dir: Directory to save results
            process: Process type for task execution (sequential or hierarchical)
        """
        self.agent_factory = agent_factory
        self.data_path = data_path
        self.output_dir = output_dir
        self.process = process
        
        # Create agents
        self.forecasting_agent = agent_factory.create_forecasting_agent()
        self.optimization_agent = agent_factory.create_optimization_agent()
        self.anomaly_detection_agent = agent_factory.create_anomaly_detection_agent()
        self.scenario_planning_agent = agent_factory.create_scenario_planning_agent()
        self.coordinator_agent = agent_factory.create_coordinator_agent()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def create_product_analysis_tasks(self, product_id: str) -> List[Task]:
        """
        Create tasks for analyzing a specific product.
        
        Args:
            product_id: ID of the product to analyze
            
        Returns:
            List of tasks for product analysis
        """
        # Define tasks for the product
        forecast_task = Task(
            description=f"""
            Analyze historical data for product {product_id} and generate a detailed demand forecast.
            Focus on identifying patterns, seasonality, and trends.
            Use the forecast_demand tool to generate the forecast.
            Your output should include:
            1. Key metrics about the forecast
            2. Interpretation of the results
            3. Any notable patterns or concerns
            
            Input data path: {self.data_path}
            """,
            agent=self.forecasting_agent,
            expected_output="Detailed demand forecast and analysis for the product"
        )
        
        anomaly_task = Task(
            description=f"""
            Analyze historical data for product {product_id} to identify any anomalies or unusual patterns.
            Look for outliers, sudden changes, or inconsistencies in the data.
            Use the identify_anomalies tool to find anomalies.
            Your output should include:
            1. List of identified anomalies
            2. Interpretation of what might have caused them
            3. Recommendations for handling these anomalies
            
            Input data path: {self.data_path}
            """,
            agent=self.anomaly_detection_agent,
            expected_output="List of anomalies and their analysis",
            context=[forecast_task]
        )
        
        optimization_task = Task(
            description=f"""
            Based on the demand forecast and any identified anomalies for product {product_id}, 
            calculate optimal inventory parameters including safety stock and reorder point.
            Assume a service level of 95% unless otherwise specified.
            Use the calculate_safety_stock and calculate_reorder_point tools.
            Your output should include:
            1. Recommended safety stock level
            2. Recommended reorder point
            3. Explanation of how these values were calculated
            
            You need to extract lead time and demand variability information from the forecast and anomaly analysis.
            """,
            agent=self.optimization_agent,
            expected_output="Optimal inventory parameters with calculations",
            context=[forecast_task, anomaly_task]
        )
        
        scenario_task = Task(
            description=f"""
            Based on the forecast, anomalies, and optimization recommendations for product {product_id},
            develop 2-3 different inventory scenarios for future planning.
            Consider factors like:
            - What if demand increases/decreases by 20%?
            - What if lead times change?
            - What if seasonality patterns shift?
            
            Your output should include:
            1. Description of each scenario
            2. Recommended inventory strategy for each scenario
            3. Potential risks and opportunities in each scenario
            """,
            agent=self.scenario_planning_agent,
            expected_output="Multiple scenario analyses with recommendations",
            context=[forecast_task, anomaly_task, optimization_task]
        )
        
        coordinator_task = Task(
            description=f"""
            Synthesize all the analyses for product {product_id} into a comprehensive recommendation.
            Review the forecast, anomalies, optimization parameters, and scenarios.
            Your output should include:
            1. Executive summary of key findings
            2. Concrete recommendations for inventory management
            3. Implementation steps and monitoring suggestions
            4. Any additional insights not covered by the specialists
            
            Use the analyze_product_performance tool to add performance metrics to your analysis.
            Input data path: {self.data_path}
            """,
            agent=self.coordinator_agent,
            expected_output="Comprehensive analysis and recommendations",
            context=[forecast_task, anomaly_task, optimization_task, scenario_task]
        )
        
        return [forecast_task, anomaly_task, optimization_task, scenario_task, coordinator_task]
    
    def run_product_analysis(self, product_id: str) -> Dict:
        """
        Run analysis for a specific product.
        
        Args:
            product_id: ID of the product to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Create tasks for the product
        tasks = self.create_product_analysis_tasks(product_id)
        
        # Create a crew for this product
        crew = Crew(
            agents=[
                self.forecasting_agent,
                self.anomaly_detection_agent,
                self.optimization_agent,
                self.scenario_planning_agent,
                self.coordinator_agent
            ],
            tasks=tasks,
            verbose=2,
            process=self.process
        )
        
        # Run the crew
        result = crew.kickoff()
        
        # Format and return results
        return {
            "product_id": product_id,
            "forecast": tasks[0].output,
            "anomalies": tasks[1].output,
            "optimization": tasks[2].output,
            "scenarios": tasks[3].output,
            "recommendations": tasks[4].output,
            "summary": result
        }
    
    def run_multi_product_analysis(self, product_ids: List[str]) -> Dict:
        """
        Run analysis for multiple products.
        
        Args:
            product_ids: List of product IDs to analyze
            
        Returns:
            Dictionary with analysis results for all products
        """
        results = {}
        
        for product_id in product_ids:
            logger.info(f"Starting analysis for product {product_id}")
            try:
                product_results = self.run_product_analysis(product_id)
                results[product_id] = product_results
                logger.info(f"Completed analysis for product {product_id}")
            except Exception as e:
                logger.error(f"Error analyzing product {product_id}: {str(e)}")
                results[product_id] = {"error": str(e)}
        
        return results


def load_product_data(data_path: str) -> List[str]:
    """
    Load product IDs from data file.
    
    Args:
        data_path: Path to data file
        
    Returns:
        List of product IDs
    """
    try:
        # Determine file type based on extension
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            logger.warning(f"Unsupported file type: {data_path}")
            raise ValueError(f"Unsupported file type: {data_path}")
        
        # Extract unique product IDs
        if 'product_id' in df.columns or 'Product ID' in df.columns:
            id_col = 'product_id' if 'product_id' in df.columns else 'Product ID'
            product_ids = df[id_col].unique().tolist()
            
            # Convert to strings if they're not already
            product_ids = [str(pid) for pid in product_ids]
            
            return product_ids
        else:
            logger.warning("No product ID column found in data file")
            raise ValueError("No product ID column found in data file")
    except Exception as e:
        logger.error(f"Error loading product data: {str(e)}")
        raise


def save_analysis_results(results: Dict, output_path: str) -> None:
    """
    Save analysis results to a file.
    
    Args:
        results: Dictionary with analysis results
        output_path: Path to save results
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def run_inventory_optimization(
    product_data_path: str,
    output_dir: str = "./output",
    config_path: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Run the inventory optimization process.
    
    Args:
        product_data_path: Path to product data file
        output_dir: Directory to save results
        config_path: Path to agent configuration file
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary with optimization results
    """
    try:
        # Set up logging level
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load product data
        logger.info(f"Loading product data from {product_data_path}")
        product_ids = load_product_data(product_data_path)
        logger.info(f"Found {len(product_ids)} products to analyze")
        
        # Initialize agent factory
        factory = InventoryAgentFactory(config_path=config_path)
        
        # Create crew
        crew = InventoryOptimizationCrew(
            agent_factory=factory,
            data_path=product_data_path,
            output_dir=output_dir
        )
        
        # Run analysis
        logger.info("Starting inventory optimization")
        results = crew.run_multi_product_analysis(product_ids)
        
        # Save results
        output_path = os.path.join(output_dir, "inventory_analysis_results.json")
        save_analysis_results(results, output_path)
        
        return results
    
    except Exception as e:
        logger.error(f"Error running inventory optimization: {str(e)}")
        raise 