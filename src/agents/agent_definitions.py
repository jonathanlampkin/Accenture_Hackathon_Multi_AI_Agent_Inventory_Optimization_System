"""
Agent definitions for multi-agent inventory optimization system.

This module defines the agents that will collaborate in the inventory
optimization system, including their responsibilities, skills and tools.
"""

from crewai import Agent, Task, Crew
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from improved_forecasting import ImprovedForecaster
from inventory_utils import (
    optimize_inventory_parameters,
    simulate_inventory_policy,
    run_scenario_analysis,
    generate_inventory_recommendations
)

# Configure logging
logger = logging.getLogger(__name__)


def create_forecasting_agent(config: Dict[str, Any]) -> Agent:
    """
    Create a forecasting agent responsible for demand prediction.
    
    Args:
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured forecasting agent
    """
    return Agent(
        role="Demand Forecasting Specialist",
        goal="Generate accurate demand forecasts for all products",
        backstory="""
        You are an expert in time series forecasting with years of experience
        in retail and supply chain. You use advanced statistical models and
        machine learning techniques to predict future demand patterns with
        high accuracy. You understand seasonal patterns, trends, and the impact
        of promotions on demand.
        """,
        verbose=config.get("verbose", True),
        allow_delegation=config.get("allow_delegation", False),
        memory=config.get("memory", False),
        tools=[
            lambda x: forecasting_tool(x, config=config)
        ]
    )


def create_optimization_agent(config: Dict[str, Any]) -> Agent:
    """
    Create an inventory optimization agent.
    
    Args:
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured optimization agent
    """
    return Agent(
        role="Inventory Optimization Specialist",
        goal="Determine optimal inventory policies for all products",
        backstory="""
        You are an expert in inventory management with deep knowledge of 
        supply chain operations. You use mathematical models to determine 
        optimal inventory levels, safety stock, and reorder points. You 
        understand the tradeoffs between inventory costs and service levels,
        and can optimize these parameters for different types of products.
        """,
        verbose=config.get("verbose", True),
        allow_delegation=config.get("allow_delegation", False),
        memory=config.get("memory", False),
        tools=[
            lambda x: optimization_tool(x, config=config)
        ]
    )


def create_scenario_planning_agent(config: Dict[str, Any]) -> Agent:
    """
    Create a scenario planning agent for risk assessment.
    
    Args:
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured scenario planning agent
    """
    return Agent(
        role="Scenario Planning Specialist",
        goal="Assess inventory policies under various risk scenarios",
        backstory="""
        You are a strategic thinker specializing in risk assessment and
        scenario planning. You evaluate how different supply chain disruptions,
        demand surges, and other unexpected events might impact inventory
        levels and service performance. You help organizations prepare for
        uncertainties by testing inventory policies under various scenarios.
        """,
        verbose=config.get("verbose", True),
        allow_delegation=config.get("allow_delegation", False),
        memory=config.get("memory", False),
        tools=[
            lambda x: scenario_planning_tool(x, config=config)
        ]
    )


def create_anomaly_detection_agent(config: Dict[str, Any]) -> Agent:
    """
    Create an anomaly detection agent.
    
    Args:
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured anomaly detection agent
    """
    return Agent(
        role="Anomaly Detection Specialist",
        goal="Identify unusual patterns in demand and inventory data",
        backstory="""
        You are an expert in statistical anomaly detection with a focus on
        supply chain data. You can identify unusual patterns in demand, inventory
        levels, and other supply chain metrics that may indicate errors,
        fraud, or unexpected market changes. Your insights help companies
        respond quickly to outliers and maintain data quality.
        """,
        verbose=config.get("verbose", True),
        allow_delegation=config.get("allow_delegation", False),
        memory=config.get("memory", False),
        tools=[
            lambda x: anomaly_detection_tool(x, config=config)
        ]
    )


def create_recommendation_agent(config: Dict[str, Any]) -> Agent:
    """
    Create a recommendation agent that synthesizes insights.
    
    Args:
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured recommendation agent
    """
    return Agent(
        role="Inventory Strategy Advisor",
        goal="Generate actionable inventory management recommendations",
        backstory="""
        You are a seasoned inventory strategy consultant who synthesizes
        insights from forecasting, optimization, and scenario planning to
        create actionable recommendations. You understand business priorities
        and translate technical findings into clear, prioritized actions.
        You excel at explaining complex inventory concepts to business stakeholders.
        """,
        verbose=config.get("verbose", True),
        allow_delegation=config.get("allow_delegation", True),
        memory=config.get("memory", True),
        tools=[
            lambda x: generate_recommendations_tool(x, config=config)
        ]
    )


def create_monitoring_agent(config: Dict[str, Any]) -> Agent:
    """
    Create a monitoring agent for ongoing inventory evaluation.
    
    Args:
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured monitoring agent
    """
    return Agent(
        role="Inventory Performance Monitor",
        goal="Track KPIs and identify optimization opportunities",
        backstory="""
        You are an analytical expert who continuously monitors inventory
        performance metrics and KPIs. You can identify when inventory policies
        need adjustment due to changing conditions. You understand the seasonal
        patterns of different products and can detect when performance is
        deviating from expectations.
        """,
        verbose=config.get("verbose", True),
        allow_delegation=config.get("allow_delegation", False),
        memory=config.get("memory", True),
        tools=[
            lambda x: monitoring_tool(x, config=config)
        ]
    )


# Agent tool implementations
def forecasting_tool(query: str, config: Dict[str, Any]) -> str:
    """
    Tool for generating demand forecasts.
    
    Args:
        query: Parameters for forecasting in JSON or string format
        config: Configuration settings
        
    Returns:
        JSON string with forecasting results
    """
    try:
        # Parse the query if it's in string format
        if isinstance(query, str):
            import json
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # If not valid JSON, extract parameters from the query string
                params = {
                    "product_id": extract_param(query, "product_id"),
                    "horizon": extract_param(query, "horizon", default=30, convert=int),
                    "method": extract_param(query, "method", default="auto")
                }
        else:
            params = query
        
        # Get data path from config
        data_path = config.get("data", {}).get("demand_data_path", "data/demand_data.csv")
        
        # Initialize forecaster
        forecaster = ImprovedForecaster(
            data_path=data_path,
            output_dir=config.get("output_dir", "./output/forecasts"),
            test_proportion=config.get("forecasting", {}).get("test_proportion", 0.2)
        )
        
        # Generate forecast
        product_id = params.get("product_id")
        horizon = params.get("horizon", 30)
        method = params.get("method", "auto")
        
        if product_id:
            # Forecast for a specific product
            result = forecaster.forecast_product(
                product_id=product_id,
                forecast_horizon=horizon,
                method=method
            )
        else:
            # Forecast for all products
            result = forecaster.forecast_all_products(
                forecast_horizon=horizon,
                max_products=params.get("max_products", None)
            )
        
        # Convert result to JSON string
        import json
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error in forecasting tool: {str(e)}")
        return json.dumps({"error": str(e)})


def optimization_tool(query: str, config: Dict[str, Any]) -> str:
    """
    Tool for optimizing inventory parameters.
    
    Args:
        query: Parameters for optimization in JSON or string format
        config: Configuration settings
        
    Returns:
        JSON string with optimization results
    """
    try:
        # Parse the query
        import json
        if isinstance(query, str):
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # Extract parameters from the query string
                params = {
                    "product_id": extract_param(query, "product_id"),
                    "lead_time": extract_param(query, "lead_time", default=7, convert=float),
                    "service_level": extract_param(query, "service_level", default=0.95, convert=float)
                }
        else:
            params = query
        
        # Get data path from config
        data_path = config.get("data", {}).get("demand_data_path", "data/demand_data.csv")
        
        # Load demand data
        demand_data = pd.read_csv(data_path)
        
        # Get product-specific data
        product_id = params.get("product_id")
        if product_id:
            product_data = demand_data[demand_data["Product ID"] == product_id]
        else:
            # If no product ID specified, use all data
            product_data = demand_data
        
        # Extract demand time series
        demand_series = product_data["Sales Quantity"].values
        
        # Get optimization parameters
        lead_time = params.get("lead_time", 7)
        service_level = params.get("service_level", 0.95)
        unit_cost = params.get("unit_cost", 10.0)
        holding_cost_rate = params.get("holding_cost_rate", 0.25)
        ordering_cost = params.get("ordering_cost", 25.0)
        review_period = params.get("review_period", 7)
        
        # Optimize inventory parameters
        optimized_params = optimize_inventory_parameters(
            demand_history=pd.Series(demand_series),
            lead_time=lead_time,
            unit_cost=unit_cost,
            holding_cost_rate=holding_cost_rate,
            ordering_cost=ordering_cost,
            service_level_target=service_level,
            review_period=review_period
        )
        
        # Return optimized parameters
        return json.dumps(optimized_params)
        
    except Exception as e:
        logger.error(f"Error in optimization tool: {str(e)}")
        return json.dumps({"error": str(e)})


def scenario_planning_tool(query: str, config: Dict[str, Any]) -> str:
    """
    Tool for running scenario analyses.
    
    Args:
        query: Parameters for scenario planning in JSON or string format
        config: Configuration settings
        
    Returns:
        JSON string with scenario analysis results
    """
    try:
        # Parse the query
        import json
        if isinstance(query, str):
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # Extract parameters from the query string
                params = {
                    "product_id": extract_param(query, "product_id"),
                    "scenarios": extract_param(query, "scenarios", default=None)
                }
        else:
            params = query
        
        # Get data path from config
        data_path = config.get("data", {}).get("demand_data_path", "data/demand_data.csv")
        
        # Load demand data
        demand_data = pd.read_csv(data_path)
        
        # Get product-specific data
        product_id = params.get("product_id")
        if product_id:
            product_data = demand_data[demand_data["Product ID"] == product_id]
        else:
            # If no product ID specified, use first product
            product_id = demand_data["Product ID"].iloc[0]
            product_data = demand_data[demand_data["Product ID"] == product_id]
        
        # Extract demand time series
        demand_series = pd.Series(product_data["Sales Quantity"].values)
        
        # Get inventory parameters (either from params or compute them)
        inventory_params = params.get("inventory_params", None)
        if not inventory_params:
            # Compute inventory parameters
            lead_time = params.get("lead_time", 7)
            service_level = params.get("service_level", 0.95)
            
            inventory_params = optimize_inventory_parameters(
                demand_history=demand_series,
                lead_time=lead_time,
                unit_cost=params.get("unit_cost", 10.0),
                holding_cost_rate=params.get("holding_cost_rate", 0.25),
                ordering_cost=params.get("ordering_cost", 25.0),
                service_level_target=service_level
            )
        
        # Define scenarios
        scenarios = params.get("scenarios", None)
        if not scenarios:
            # Use default scenarios
            scenarios = {
                "base": {"demand_factor": 1.0, "lead_time_factor": 1.0},
                "high_demand": {"demand_factor": 1.5, "lead_time_factor": 1.0},
                "supply_disruption": {"demand_factor": 1.0, "lead_time_factor": 2.0},
                "worst_case": {"demand_factor": 1.3, "lead_time_factor": 1.7}
            }
        
        # Run scenario analysis
        lead_time = params.get("lead_time", 7)
        scenario_results = run_scenario_analysis(
            base_demand=demand_series,
            lead_time=lead_time,
            inventory_params=inventory_params,
            scenarios=scenarios
        )
        
        # Format results for JSON return
        formatted_results = {}
        for scenario, results in scenario_results.items():
            formatted_results[scenario] = {
                "metrics": results["metrics"]
            }
        
        # Return scenario results
        return json.dumps({
            "product_id": product_id,
            "scenarios": formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error in scenario planning tool: {str(e)}")
        return json.dumps({"error": str(e)})


def anomaly_detection_tool(query: str, config: Dict[str, Any]) -> str:
    """
    Tool for detecting anomalies in demand data.
    
    Args:
        query: Parameters for anomaly detection in JSON or string format
        config: Configuration settings
        
    Returns:
        JSON string with detected anomalies
    """
    try:
        # Parse the query
        import json
        if isinstance(query, str):
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # Extract parameters from the query string
                params = {
                    "product_id": extract_param(query, "product_id"),
                    "method": extract_param(query, "method", default="zscore")
                }
        else:
            params = query
        
        # Get data path from config
        data_path = config.get("data", {}).get("demand_data_path", "data/demand_data.csv")
        
        # Load demand data
        demand_data = pd.read_csv(data_path)
        
        # Get product-specific data
        product_id = params.get("product_id")
        if product_id:
            product_data = demand_data[demand_data["Product ID"] == product_id]
        else:
            # If no product ID specified, use all data
            product_data = demand_data
        
        # Extract demand time series
        if "Date" in product_data.columns:
            product_data = product_data.sort_values("Date")
        
        demand_series = product_data["Sales Quantity"].values
        
        # Get anomaly detection method
        method = params.get("method", "zscore").lower()
        threshold = params.get("threshold", 3.0)
        
        # Detect anomalies
        anomalies = detect_anomalies(
            demand_series, 
            method=method, 
            threshold=threshold
        )
        
        # Return detected anomalies
        if "Date" in product_data.columns:
            dates = product_data["Date"].values
            anomaly_dates = [dates[i] for i in anomalies["indices"]]
            anomalies["dates"] = anomaly_dates
        
        return json.dumps(anomalies)
        
    except Exception as e:
        logger.error(f"Error in anomaly detection tool: {str(e)}")
        return json.dumps({"error": str(e)})


def generate_recommendations_tool(query: str, config: Dict[str, Any]) -> str:
    """
    Tool for generating inventory recommendations.
    
    Args:
        query: Parameters for recommendation generation in JSON or string format
        config: Configuration settings
        
    Returns:
        JSON string with recommendations
    """
    try:
        # Parse the query
        import json
        if isinstance(query, str):
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # Extract parameters from the query string
                params = {
                    "product_id": extract_param(query, "product_id"),
                    "current_params": extract_param(query, "current_params", default=None),
                    "optimized_params": extract_param(query, "optimized_params", default=None),
                    "scenario_results": extract_param(query, "scenario_results", default=None)
                }
        else:
            params = query
        
        # Get required parameters
        product_id = params.get("product_id")
        current_params = params.get("current_params")
        optimized_params = params.get("optimized_params")
        scenario_results = params.get("scenario_results")
        
        # If parameters are missing, compute them
        if not product_id or not current_params or not optimized_params or not scenario_results:
            # Get data path from config
            data_path = config.get("data", {}).get("demand_data_path", "data/demand_data.csv")
            
            # Load demand data
            demand_data = pd.read_csv(data_path)
            
            # Get product ID if not provided
            if not product_id:
                product_id = demand_data["Product ID"].iloc[0]
            
            # Get product-specific data
            product_data = demand_data[demand_data["Product ID"] == product_id]
            demand_series = pd.Series(product_data["Sales Quantity"].values)
            
            # Compute current parameters (as baseline)
            if not current_params:
                current_params = {
                    "safety_stock": 50,
                    "reorder_point": 100,
                    "economic_order_quantity": 200,
                    "min_level": 100,
                    "max_level": 300
                }
            
            # Compute optimized parameters if not provided
            if not optimized_params:
                lead_time = params.get("lead_time", 7)
                service_level = params.get("service_level", 0.95)
                
                optimized_params = optimize_inventory_parameters(
                    demand_history=demand_series,
                    lead_time=lead_time,
                    unit_cost=params.get("unit_cost", 10.0),
                    holding_cost_rate=params.get("holding_cost_rate", 0.25),
                    ordering_cost=params.get("ordering_cost", 25.0),
                    service_level_target=service_level
                )
            
            # Run scenario analysis if not provided
            if not scenario_results:
                lead_time = params.get("lead_time", 7)
                scenarios = {
                    "base": {"demand_factor": 1.0, "lead_time_factor": 1.0},
                    "high_demand": {"demand_factor": 1.5, "lead_time_factor": 1.0},
                    "supply_disruption": {"demand_factor": 1.0, "lead_time_factor": 2.0},
                    "worst_case": {"demand_factor": 1.3, "lead_time_factor": 1.7}
                }
                
                scenario_results = run_scenario_analysis(
                    base_demand=demand_series,
                    lead_time=lead_time,
                    inventory_params=optimized_params,
                    scenarios=scenarios
                )
        
        # Generate recommendations
        recommendations = generate_inventory_recommendations(
            product_id=product_id,
            current_params=current_params,
            optimized_params=optimized_params,
            scenario_results=scenario_results
        )
        
        # Return recommendations
        return json.dumps(recommendations)
        
    except Exception as e:
        logger.error(f"Error in recommendation generation tool: {str(e)}")
        return json.dumps({"error": str(e)})


def monitoring_tool(query: str, config: Dict[str, Any]) -> str:
    """
    Tool for monitoring inventory performance.
    
    Args:
        query: Parameters for monitoring in JSON or string format
        config: Configuration settings
        
    Returns:
        JSON string with monitoring results
    """
    try:
        # Parse the query
        import json
        if isinstance(query, str):
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # Extract parameters from the query string
                params = {
                    "product_id": extract_param(query, "product_id"),
                    "time_period": extract_param(query, "time_period", default="last_30_days")
                }
        else:
            params = query
        
        # Get data paths from config
        inventory_data_path = config.get("data", {}).get("inventory_data_path", "data/inventory_data.csv")
        demand_data_path = config.get("data", {}).get("demand_data_path", "data/demand_data.csv")
        
        # Load data
        try:
            inventory_data = pd.read_csv(inventory_data_path)
            demand_data = pd.read_csv(demand_data_path)
        except FileNotFoundError:
            # Return sample data if files not found
            return json.dumps({
                "status": "simulated",
                "kpis": {
                    "service_level": 0.96,
                    "fill_rate": 0.98,
                    "inventory_turnover": 12.5,
                    "days_of_supply": 15.2,
                    "stockout_rate": 0.02
                }
            })
        
        # Get product-specific data
        product_id = params.get("product_id")
        if product_id:
            inventory_product_data = inventory_data[inventory_data["Product ID"] == product_id]
            demand_product_data = demand_data[demand_data["Product ID"] == product_id]
        else:
            # If no product ID specified, use all data
            inventory_product_data = inventory_data
            demand_product_data = demand_data
        
        # Calculate KPIs
        kpis = calculate_inventory_kpis(inventory_product_data, demand_product_data)
        
        # Return KPIs
        return json.dumps({
            "product_id": product_id,
            "kpis": kpis
        })
        
    except Exception as e:
        logger.error(f"Error in monitoring tool: {str(e)}")
        return json.dumps({"error": str(e)})


# Helper functions
def extract_param(query: str, param_name: str, default=None, convert=None):
    """Extract parameter from query string."""
    import re
    pattern = rf'{param_name}[=:][\s]*([^ ,\n]+)'
    match = re.search(pattern, query)
    if match:
        value = match.group(1)
        if convert:
            try:
                return convert(value)
            except ValueError:
                return default
        return value
    return default


def detect_anomalies(time_series, method="zscore", threshold=3.0):
    """
    Detect anomalies in a time series.
    
    Args:
        time_series: Array-like time series data
        method: Detection method ('zscore', 'iqr', or 'moving_avg')
        threshold: Threshold for anomaly detection
        
    Returns:
        Dictionary with anomaly indices and values
    """
    anomaly_indices = []
    
    if method == "zscore":
        # Z-score method
        mean = np.mean(time_series)
        std = np.std(time_series)
        z_scores = np.abs((time_series - mean) / std)
        anomaly_indices = np.where(z_scores > threshold)[0]
    
    elif method == "iqr":
        # IQR method
        q1 = np.percentile(time_series, 25)
        q3 = np.percentile(time_series, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        anomaly_indices = np.where((time_series < lower_bound) | (time_series > upper_bound))[0]
    
    elif method == "moving_avg":
        # Moving average method
        window = min(10, len(time_series) // 3)
        if window < 3:
            window = 3
        
        moving_avg = np.convolve(time_series, np.ones(window)/window, mode='same')
        deviation = np.abs(time_series - moving_avg)
        mean_deviation = np.mean(deviation)
        anomaly_indices = np.where(deviation > threshold * mean_deviation)[0]
    
    # Get anomaly values
    anomaly_values = [time_series[i] for i in anomaly_indices]
    
    return {
        "indices": anomaly_indices.tolist(),
        "values": anomaly_values,
        "count": len(anomaly_indices)
    }


def calculate_inventory_kpis(inventory_data, demand_data):
    """
    Calculate inventory KPIs from inventory and demand data.
    
    This is a placeholder that returns sample KPIs.
    In a real implementation, this would calculate actual KPIs
    from the provided data.
    
    Args:
        inventory_data: DataFrame with inventory data
        demand_data: DataFrame with demand data
        
    Returns:
        Dictionary of KPIs
    """
    # Sample KPIs for demonstration
    return {
        "service_level": 0.95,
        "fill_rate": 0.97,
        "inventory_turnover": 12.0,
        "days_of_supply": 15.0,
        "stockout_rate": 0.03,
        "average_inventory_value": 10000.0,
        "inventory_carrying_cost": 2500.0
    } 