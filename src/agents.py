"""
Multi-agent framework for inventory optimization using CrewAI.

This module defines specialized agents that collaborate to optimize inventory management
processes, including forecasting, optimization, anomaly detection, and scenario planning.
"""

from crewai import Agent, Task, Crew, Process
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import os
import json

logger = logging.getLogger(__name__)

# Import tools that agents will use
try:
    from .tools import (
        calculate_reorder_point,
        calculate_safety_stock,
        analyze_product_performance,
        identify_anomalies,
        forecast_demand
    )
except ImportError:
    logger.warning("Could not import tools module. Agent functionality may be limited.")


class InventoryAgents:
    """Class to create and manage inventory optimization agents."""
    
    def __init__(self, model_name='gpt-4-turbo-preview', verbose=False):
        """
        Initialize the inventory agents.
        
        Args:
            model_name: Name of the language model to use
            verbose: Whether to enable verbose output
        """
        self.model_name = model_name
        self.verbose = verbose
        self.llm_config = {
            "temperature": 0.2,
            "model": model_name,
        }
        logger.info(f"Initialized InventoryAgents with model: {model_name}")
        
    def create_forecast_agent(self) -> Agent:
        """Create the demand forecasting agent."""
        return Agent(
            name="Demand Forecaster",
            role="Demand Forecasting Specialist",
            goal="Accurately predict future demand for products",
            backstory="An expert in time series analysis and forecasting with deep knowledge of demand patterns and seasonality effects.",
            verbose=self.verbose,
            llm=self.llm_config
        )
    
    def create_inventory_agent(self) -> Agent:
        """Create the inventory optimization agent."""
        return Agent(
            name="Inventory Optimizer",
            role="Inventory Optimization Specialist",
            goal="Determine optimal inventory levels and reorder points",
            backstory="An operations research expert specialized in inventory management with years of experience optimizing supply chains.",
            verbose=self.verbose,
            llm=self.llm_config
        )
    
    def create_anomaly_agent(self) -> Agent:
        """Create the anomaly detection agent."""
        return Agent(
            name="Anomaly Detector",
            role="Data Anomaly Detection Specialist",
            goal="Identify unusual patterns or outliers in inventory and demand data",
            backstory="A data scientist with expertise in statistical analysis and pattern recognition who excels at finding the needle in the haystack.",
            verbose=self.verbose,
            llm=self.llm_config
        )
    
    def create_coordinator_agent(self) -> Agent:
        """Create the coordinator agent."""
        return Agent(
            name="Operations Coordinator",
            role="Inventory Operations Coordinator",
            goal="Coordinate the inventory optimization process and synthesize insights",
            backstory="A seasoned supply chain manager who excels at integrating cross-functional insights and making balanced decisions.",
            verbose=self.verbose,
            llm=self.llm_config
        )


class InventoryAgentFactory:
    """Factory class for creating specialized inventory optimization agents."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent factory with configuration.
        
        Args:
            config_path: Path to agent configuration JSON file
            verbose: Whether to enable verbose logging
            llm_config: Configuration for the language model
        """
        self.verbose = verbose
        self.llm_config = llm_config or {
            "temperature": 0.2,
            "model": "gpt-4-turbo-preview",
        }
        
        # Load configuration
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load agent configuration from file or use defaults."""
        default_config = {
            "forecasting_agent": {
                "name": "Demand Forecaster",
                "role": "Demand Forecasting Specialist",
                "goal": "Accurately predict future demand for products",
                "backstory": "An expert in time series analysis and forecasting with deep knowledge of demand patterns and seasonality effects."
            },
            "optimization_agent": {
                "name": "Inventory Optimizer",
                "role": "Inventory Optimization Specialist",
                "goal": "Determine optimal inventory levels and reorder points",
                "backstory": "An operations research expert specialized in inventory management with years of experience optimizing supply chains."
            },
            "anomaly_agent": {
                "name": "Anomaly Detector",
                "role": "Data Anomaly Detection Specialist",
                "goal": "Identify unusual patterns or outliers in inventory and demand data",
                "backstory": "A data scientist with expertise in statistical analysis and pattern recognition who excels at finding the needle in the haystack."
            },
            "scenario_agent": {
                "name": "Scenario Planner",
                "role": "Scenario Planning Strategist",
                "goal": "Generate and evaluate alternative future scenarios for inventory planning",
                "backstory": "A strategic thinker with experience in risk management and contingency planning across various supply chain disruptions."
            },
            "coordinator_agent": {
                "name": "Operations Coordinator",
                "role": "Inventory Operations Coordinator",
                "goal": "Coordinate the inventory optimization process and synthesize insights",
                "backstory": "A seasoned supply chain manager who excels at integrating cross-functional insights and making balanced decisions."
            }
        }
        
        if not config_path:
            logger.info("No config path provided, using default configuration")
            return default_config
        
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"Config file {config_path} not found, using default configuration")
                return default_config
                
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Merge with defaults
            for agent_type, agent_config in user_config.items():
                if agent_type in default_config:
                    default_config[agent_type].update(agent_config)
                else:
                    default_config[agent_type] = agent_config
                    
            return default_config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return default_config

    def create_forecasting_agent(self) -> Agent:
        """
        Create a specialized agent for demand forecasting.
        
        Returns:
            CrewAI Agent configured for demand forecasting
        """
        config = self.config["forecasting_agent"]
        
        return Agent(
            name=config["name"],
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=self.verbose,
            llm=self.llm_config,
            tools=[forecast_demand]
        )
    
    def create_optimization_agent(self) -> Agent:
        """
        Create a specialized agent for inventory optimization.
        
        Returns:
            CrewAI Agent configured for inventory optimization
        """
        config = self.config["optimization_agent"]
        
        return Agent(
            name=config["name"],
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=self.verbose,
            llm=self.llm_config,
            tools=[calculate_reorder_point, calculate_safety_stock]
        )
    
    def create_anomaly_agent(self) -> Agent:
        """
        Create a specialized agent for anomaly detection.
        
        Returns:
            CrewAI Agent configured for anomaly detection
        """
        config = self.config["anomaly_agent"]
        
        return Agent(
            name=config["name"],
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=self.verbose,
            llm=self.llm_config,
            tools=[identify_anomalies]
        )
    
    def create_scenario_agent(self) -> Agent:
        """
        Create a specialized agent for scenario planning.
        
        Returns:
            CrewAI Agent configured for scenario planning
        """
        config = self.config["scenario_agent"]
        
        return Agent(
            name=config["name"],
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=self.verbose,
            llm=self.llm_config,
            tools=[]  # Scenario planning uses mostly reasoning
        )
    
    def create_coordinator_agent(self) -> Agent:
        """
        Create a coordinator agent to orchestrate the inventory optimization process.
        
        Returns:
            CrewAI Agent configured for coordination
        """
        config = self.config["coordinator_agent"]
        
        return Agent(
            name=config["name"],
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=self.verbose,
            llm=self.llm_config,
            tools=[analyze_product_performance]
        )


class InventoryOptimizationCrew:
    """
    A crew of specialized agents collaborating on inventory optimization.
    
    This class orchestrates multiple agents working together on different aspects
    of inventory optimization, from forecasting to anomaly detection.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
        process: Process = Process.sequential
    ):
        """
        Initialize the inventory optimization crew.
        
        Args:
            config_path: Path to agent configuration JSON file
            verbose: Whether to enable verbose logging
            llm_config: Configuration for the language model
            process: Process type for task execution (sequential or hierarchical)
        """
        self.agent_factory = InventoryAgentFactory(
            config_path=config_path,
            verbose=verbose,
            llm_config=llm_config
        )
        self.process = process
        
        # Create agents
        self.forecasting_agent = self.agent_factory.create_forecasting_agent()
        self.optimization_agent = self.agent_factory.create_optimization_agent()
        self.anomaly_agent = self.agent_factory.create_anomaly_agent()
        self.scenario_agent = self.agent_factory.create_scenario_agent()
        self.coordinator_agent = self.agent_factory.create_coordinator_agent()
    
    def create_product_analysis_tasks(self, product_data: Dict[str, Any]) -> List[Task]:
        """
        Create tasks for analyzing a specific product.
        
        Args:
            product_data: Data for the product to analyze
            
        Returns:
            List of tasks for product analysis
        """
        product_id = product_data.get("product_id", "unknown")
        product_name = product_data.get("product_name", f"Product {product_id}")
        
        forecasting_task = Task(
            description=f"""
            Analyze historical demand data for {product_name} (ID: {product_id}) and generate forecasts.
            
            The product has the following characteristics:
            - Category: {product_data.get('category', 'Unknown')}
            - Price: {product_data.get('price', 'Unknown')}
            - Lead Time: {product_data.get('lead_time', 'Unknown')} days
            - Is Perishable: {product_data.get('is_perishable', False)}
            
            Your forecast should:
            1. Account for any seasonality or trends
            2. Provide a point forecast with confidence intervals
            3. Generate forecasts for the next 4 weeks
            4. Consider the impact of any promotions or holidays
            """,
            agent=self.forecasting_agent
        )
        
        anomaly_task = Task(
            description=f"""
            Identify any anomalies in the historical demand and inventory data for {product_name} (ID: {product_id}).
            
            Focus on:
            1. Outliers in demand patterns
            2. Unexpected stockouts
            3. Irregular order patterns
            4. Seasonal anomalies
            
            For each anomaly detected, provide:
            - When it occurred
            - Likely causes
            - Impact on inventory
            - Recommendations for handling similar anomalies
            """,
            agent=self.anomaly_agent
        )
        
        optimization_task = Task(
            description=f"""
            Calculate optimal inventory levels for {product_name} (ID: {product_id}) based on:
            
            - Forecasted demand
            - Lead time of {product_data.get('lead_time', 'Unknown')} days
            - Service level target of {product_data.get('service_level', '95%')}
            - Holding cost rate of {product_data.get('holding_cost_rate', '20%')} per year
            - Stockout cost or margin impact
            
            Determine:
            1. Optimal reorder point
            2. Economic order quantity
            3. Safety stock levels
            4. Min/max inventory levels
            
            Factor in any anomalies detected and explain your reasoning.
            """,
            agent=self.optimization_agent,
            dependencies=[forecasting_task, anomaly_task]
        )
        
        scenario_task = Task(
            description=f"""
            Develop three distinct demand scenarios for {product_name} (ID: {product_id}):
            
            1. Base case (most likely scenario)
            2. High demand scenario (positive outlier)
            3. Low demand scenario (negative outlier)
            
            For each scenario:
            - Estimate demand levels
            - Calculate required inventory levels
            - Identify early warning indicators
            - Recommend inventory policies
            
            Use the forecast and any anomalies as input to your scenario development.
            """,
            agent=self.scenario_agent,
            dependencies=[forecasting_task, anomaly_task]
        )
        
        coordination_task = Task(
            description=f"""
            Synthesize all analyses for {product_name} (ID: {product_id}) and provide final recommendations:
            
            1. Summarize key findings from forecasting, optimization, anomaly detection, and scenario planning
            2. Identify conflicts or trade-offs in the recommendations
            3. Provide final recommendations for:
               - Reorder point
               - Safety stock level
               - Order quantity
               - Monitoring priorities
            4. Suggest a contingency plan for each major risk identified
            
            Your synthesis should balance service level targets with inventory costs.
            """,
            agent=self.coordinator_agent,
            dependencies=[optimization_task, scenario_task]
        )
        
        return [forecasting_task, anomaly_task, optimization_task, scenario_task, coordination_task]
    
    def run_product_analysis(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete analysis for a specific product.
        
        Args:
            product_data: Data for the product to analyze
            
        Returns:
            Dictionary with analysis results
        """
        tasks = self.create_product_analysis_tasks(product_data)
        
        crew = Crew(
            agents=[
                self.forecasting_agent,
                self.optimization_agent,
                self.anomaly_agent,
                self.scenario_agent,
                self.coordinator_agent
            ],
            tasks=tasks,
            verbose=self.agent_factory.verbose,
            process=self.process
        )
        
        result = crew.kickoff()
        
        # Process and structure the results
        return {
            "product_id": product_data.get("product_id", "unknown"),
            "product_name": product_data.get("product_name", ""),
            "analysis_result": result,
            "recommendations": self._extract_recommendations(result)
        }
    
    def _extract_recommendations(self, result: str) -> Dict[str, Any]:
        """
        Extract structured recommendations from the coordinator's output.
        
        Args:
            result: Raw result from the crew execution
            
        Returns:
            Dictionary with structured recommendations
        """
        # A simple extraction that will be enhanced in a real implementation
        recommendations = {
            "reorder_point": None,
            "safety_stock": None,
            "order_quantity": None,
            "risks": [],
            "monitoring_priorities": []
        }
        
        # In a real implementation, this would parse the text more intelligently
        # For now we return a placeholder
        return recommendations
    
    def run_multi_product_analysis(self, products_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Run analysis for multiple products.
        
        Args:
            products_data: List of product data dictionaries
            
        Returns:
            Dictionary mapping product_id to analysis results
        """
        results = {}
        
        for product_data in products_data:
            product_id = product_data.get("product_id", "unknown")
            logger.info(f"Starting analysis for product {product_id}")
            
            try:
                product_result = self.run_product_analysis(product_data)
                results[product_id] = product_result
            except Exception as e:
                logger.error(f"Error analyzing product {product_id}: {str(e)}")
                results[product_id] = {
                    "product_id": product_id,
                    "error": str(e),
                    "status": "failed"
                }
        
        return results


def load_product_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load product data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of product data dictionaries
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading product data: {str(e)}")
        return []


def save_analysis_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """
    Save analysis results to a file.
    
    Args:
        results: Dictionary mapping product_id to analysis results
        output_dir: Directory to save results
        
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "inventory_analysis_results.json")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        return ""


def run_inventory_optimization(
    product_data_path: str,
    output_dir: str = "./output",
    config_path: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Run the complete inventory optimization process.
    
    Args:
        product_data_path: Path to product data JSON file
        output_dir: Directory to save results
        config_path: Path to agent configuration JSON file
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary mapping product_id to analysis results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load product data
    products_data = load_product_data(product_data_path)
    
    if not products_data:
        logger.error(f"No valid product data found in {product_data_path}")
        return {}
    
    # Create and run crew
    crew = InventoryOptimizationCrew(
        config_path=config_path,
        verbose=verbose,
        process=Process.sequential
    )
    
    results = crew.run_multi_product_analysis(products_data)
    
    # Save results
    save_analysis_results(results, output_dir)
    
    return results 