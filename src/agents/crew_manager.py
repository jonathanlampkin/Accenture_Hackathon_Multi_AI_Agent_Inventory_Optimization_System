"""
Crew Manager for orchestrating multi-agent interactions.

This module sets up and manages the CrewAI crew of specialized
inventory agents, defines their tasks, and orchestrates their
collaborative workflow.
"""

from crewai import Crew, Task, Agent
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import pandas as pd
import json
import os

from .agent_definitions import (
    create_forecasting_agent,
    create_optimization_agent,
    create_scenario_planning_agent,
    create_anomaly_detection_agent,
    create_recommendation_agent,
    create_monitoring_agent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryCrewManager:
    """
    Manager class for coordinating inventory optimization agent crew.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the crew manager.
        
        Args:
            config: Configuration dictionary with parameters for agents and tasks
        """
        self.config = config
        self.agents = {}
        self.tasks = {}
        self.crew = None
        self.results = {}
        self._setup_agents()
        self._setup_tasks()
        self._setup_crew()
    
    def _setup_agents(self):
        """Create and configure all agents in the system."""
        logger.info("Initializing inventory optimization agents...")
        
        # Create agents
        self.agents["forecasting"] = create_forecasting_agent(self.config)
        self.agents["optimization"] = create_optimization_agent(self.config)
        self.agents["scenario_planning"] = create_scenario_planning_agent(self.config)
        self.agents["anomaly_detection"] = create_anomaly_detection_agent(self.config)
        self.agents["recommendation"] = create_recommendation_agent(self.config)
        self.agents["monitoring"] = create_monitoring_agent(self.config)
        
        logger.info(f"Initialized {len(self.agents)} agents successfully")
    
    def _setup_tasks(self):
        """Define tasks for each agent."""
        logger.info("Setting up agent tasks...")
        
        # 1. Demand Forecasting Task
        self.tasks["forecasting"] = Task(
            description="""
            Generate accurate demand forecasts for all products in the inventory.
            
            You should:
            1. Analyze historical demand data for patterns and seasonality
            2. Apply appropriate forecasting models for each product category
            3. Generate forecasts for the specified time horizon
            4. Calculate forecast accuracy metrics
            5. Identify products with high forecast uncertainty
            
            Return the forecasts as a DataFrame with columns for product_id,
            date, forecasted_demand, lower_bound, and upper_bound.
            """,
            agent=self.agents["forecasting"],
            expected_output="""
            JSON containing forecasts for all products with the following structure:
            {
                "product_id": [101, 102, ...],
                "forecast_values": [...],
                "lower_bounds": [...],
                "upper_bounds": [...],
                "dates": ["2023-01-01", ...],
                "metrics": {"rmse": 10.5, "mae": 8.2, "r2": 0.85}
            }
            """,
            context=[
                f"Data path: {self.config.get('data', {}).get('demand_data_path', 'data/demand_data.csv')}",
                f"Forecast horizon: {self.config.get('forecasting', {}).get('horizon', 30)} days",
                f"Confidence level: {self.config.get('forecasting', {}).get('confidence_level', 0.95)}",
            ]
        )
        
        # 2. Anomaly Detection Task
        self.tasks["anomaly_detection"] = Task(
            description="""
            Identify unusual patterns or anomalies in the historical demand data.
            
            You should:
            1. Apply statistical methods to detect outliers in demand
            2. Identify seasonal anomalies and demand spikes
            3. Flag potentially erroneous data points
            4. Assess the impact of anomalies on forecast accuracy
            
            Return a list of detected anomalies with their timestamps and significance.
            """,
            agent=self.agents["anomaly_detection"],
            expected_output="""
            JSON containing detected anomalies with the following structure:
            {
                "product_id": [101, 102, ...],
                "anomaly_dates": ["2023-01-15", ...],
                "anomaly_values": [150, 200, ...],
                "severity": ["high", "medium", ...],
                "potential_causes": ["promotion", "supply issue", ...]
            }
            """,
            context=[
                f"Detection threshold: {self.config.get('anomaly_detection', {}).get('threshold', 3.0)}",
                f"Detection method: {self.config.get('anomaly_detection', {}).get('method', 'zscore')}"
            ]
        )
        
        # 3. Inventory Optimization Task
        self.tasks["optimization"] = Task(
            description="""
            Determine optimal inventory parameters for each product.
            
            You should:
            1. Calculate optimal safety stock levels based on demand variability
            2. Determine reorder points and economic order quantities
            3. Set min/max inventory levels for each product
            4. Balance inventory costs with service level requirements
            5. Consider product characteristics (perishable, bulky, expensive)
            
            Use the demand forecasts as an input for your calculations.
            """,
            agent=self.agents["optimization"],
            expected_output="""
            JSON containing optimized inventory parameters for each product:
            {
                "product_id": [101, 102, ...],
                "safety_stock": [50, 75, ...],
                "reorder_point": [100, 150, ...],
                "economic_order_quantity": [200, 300, ...],
                "min_level": [50, 75, ...],
                "max_level": [250, 350, ...],
                "service_level": [0.95, 0.98, ...]
            }
            """,
            context=[
                f"Target service level: {self.config.get('optimization', {}).get('service_level_target', 0.95)}",
                f"Holding cost rate: {self.config.get('optimization', {}).get('holding_cost_rate', 0.25)}",
                f"Lead times should be taken from the demand data if available"
            ],
            async_execution=False,
            human_input=False
        )
        
        # 4. Scenario Planning Task
        self.tasks["scenario_planning"] = Task(
            description="""
            Evaluate inventory policies under different risk scenarios.
            
            You should:
            1. Define relevant risk scenarios (supply disruptions, demand spikes)
            2. Simulate inventory levels under each scenario
            3. Calculate key metrics for each scenario (stockouts, fill rate)
            4. Identify vulnerable products and potential mitigation strategies
            
            Use the optimized inventory parameters and demand forecasts as inputs.
            """,
            agent=self.agents["scenario_planning"],
            expected_output="""
            JSON containing scenario analysis results:
            {
                "scenarios": ["base", "high_demand", "supply_disruption", ...],
                "product_results": {
                    "101": {
                        "stockout_probability": [0.02, 0.15, 0.35, ...],
                        "average_inventory": [150, 130, 90, ...],
                        "fill_rate": [0.98, 0.91, 0.85, ...],
                        "cost_impact": [0, 500, 1200, ...]
                    },
                    "102": {...}
                },
                "risk_assessment": ["high", "medium", "low", ...]
            }
            """,
            context=[
                f"Scenario definitions: {json.dumps(self.config.get('scenario_planning', {}).get('scenarios', {}))}",
                "Base the scenarios on optimized inventory parameters",
                "Consider both demand and supply risks"
            ],
            async_execution=False,
            human_input=False
        )
        
        # 5. Recommendation Generation Task
        self.tasks["recommendation"] = Task(
            description="""
            Generate actionable inventory management recommendations.
            
            You should:
            1. Synthesize insights from forecasting, optimization, and scenario planning
            2. Prioritize recommendations based on business impact
            3. Provide specific, actionable guidance for each product or category
            4. Include expected benefits and implementation considerations
            5. Format recommendations for business stakeholders
            
            Your recommendations should balance service levels and inventory costs.
            """,
            agent=self.agents["recommendation"],
            expected_output="""
            JSON containing prioritized inventory recommendations:
            {
                "high_priority": [
                    {
                        "product_id": 101,
                        "recommendation": "Increase safety stock by 20%",
                        "rationale": "High demand variability and critical product",
                        "expected_benefit": "Reduce stockouts by 35%",
                        "cost_impact": "Increase holding cost by $1,200/year"
                    },
                    ...
                ],
                "medium_priority": [...],
                "low_priority": [...],
                "general_recommendations": [
                    "Implement cycle counting for high-value items",
                    ...
                ]
            }
            """,
            context=[
                "Prioritize recommendations by potential business impact",
                "Consider implementation complexity",
                "Focus on actionable insights"
            ],
            async_execution=False,
            human_input=False
        )
        
        # 6. Performance Monitoring Task
        self.tasks["monitoring"] = Task(
            description="""
            Track inventory performance metrics and KPIs.
            
            You should:
            1. Calculate key inventory KPIs (service level, inventory turnover)
            2. Compare actual performance against targets
            3. Identify performance trends and deviations
            4. Flag products requiring attention or policy adjustments
            
            Provide a summary dashboard of inventory performance.
            """,
            agent=self.agents["monitoring"],
            expected_output="""
            JSON containing inventory performance metrics:
            {
                "overall_kpis": {
                    "service_level": 0.96,
                    "fill_rate": 0.98,
                    "inventory_turnover": 12.5,
                    "days_of_supply": 15.2,
                    "stockout_rate": 0.02
                },
                "product_kpis": {
                    "101": {
                        "service_level": 0.97,
                        "fill_rate": 0.99,
                        "inventory_turnover": 14.2,
                        "status": "on_target"
                    },
                    "102": {...}
                },
                "alerts": [
                    {"product_id": 103, "issue": "declining_service_level", "value": 0.91},
                    ...
                ]
            }
            """,
            context=[
                f"KPI targets: {json.dumps(self.config.get('monitoring', {}).get('kpi_targets', {}))}",
                "Flag any KPIs that deviate more than 10% from targets"
            ],
            async_execution=False,
            human_input=False
        )
        
        logger.info(f"Initialized {len(self.tasks)} tasks successfully")
    
    def _setup_crew(self):
        """Create and configure the agent crew."""
        logger.info("Setting up inventory optimization crew...")
        
        # Create the crew with all agents
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            verbose=self.config.get("verbose", 2),
            process=self.config.get("process", "sequential")
        )
        
        logger.info("Crew setup completed")
    
    def run(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the inventory optimization crew to generate recommendations.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with results from each agent
        """
        logger.info("Starting inventory optimization crew execution...")
        
        try:
            # Run the crew
            results = self.crew.kickoff()
            
            # Parse and process results
            self.results = self._process_results(results)
            
            # Save results if requested
            if save_results:
                self._save_results()
            
            logger.info("Inventory optimization completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error running inventory optimization crew: {str(e)}")
            raise
    
    def _process_results(self, raw_results: List[str]) -> Dict[str, Any]:
        """
        Process the raw results from the crew execution.
        
        Args:
            raw_results: List of string results from each task
            
        Returns:
            Processed results as a dictionary
        """
        processed_results = {}
        
        # Attempt to parse each result as JSON
        for i, task_name in enumerate(self.tasks.keys()):
            if i < len(raw_results):
                try:
                    # Try to parse as JSON
                    result = json.loads(raw_results[i])
                    processed_results[task_name] = result
                except json.JSONDecodeError:
                    # If not valid JSON, store as string
                    processed_results[task_name] = raw_results[i]
        
        return processed_results
    
    def _save_results(self):
        """Save results to disk."""
        # Create output directory if it doesn't exist
        output_dir = self.config.get("output_dir", "./output/inventory_optimization")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall results
        results_path = os.path.join(output_dir, "optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save individual task results
        for task_name, result in self.results.items():
            task_path = os.path.join(output_dir, f"{task_name}_results.json")
            with open(task_path, "w") as f:
                json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get the final recommendations from the crew execution.
        
        Returns:
            Dictionary with recommendations
        """
        if "recommendation" in self.results:
            return self.results["recommendation"]
        return {"error": "No recommendations available. Run the crew first."}


def run_inventory_optimization(config_path: Optional[str] = None, config_dict: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run the inventory optimization process with the given configuration.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        config_dict: Configuration as a dictionary
        
    Returns:
        Dictionary with optimization results
    """
    # Load configuration
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        if config_path.endswith(".json"):
            with open(config_path, "r") as f:
                config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML format")
    else:
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
            }
        }
    
    # Create and run the crew manager
    manager = InventoryCrewManager(config)
    results = manager.run()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inventory optimization with multi-agent system")
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON or YAML)")
    args = parser.parse_args()
    
    results = run_inventory_optimization(config_path=args.config)
    print(f"Optimization completed. Results saved to {results.get('output_dir', './output/inventory_optimization')}") 