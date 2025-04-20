"""
Specialized agents for the inventory optimization system.

This module contains the implementation of various specialized agents used
in the inventory optimization system, including forecasting, optimization,
anomaly detection, scenario planning, recommendation, and monitoring agents.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from crewai import Agent
from crewai.tools import BaseTool

from src.tools.forecasting_tools import (
    ForecastDemandTool,
    EvaluateModelPerformanceTool,
    IdentifySalesPatternsTool
)
from src.tools.optimization_tools import (
    CalculateEconomicOrderQuantityTool, 
    CalculateReorderPointTool,
    CalculateSafetyStockTool,
    DefineInventoryPolicyTool
)
from src.tools.anomaly_tools import (
    DetectSalesAnomaliesTool,
    DetectInventoryAnomaliesTool
)
from src.tools.scenario_tools import (
    SimulateSupplyDisruptionTool,
    SimulateDemandSpikeTool,
    SimulateSeasonalityTool
)
from src.tools.recommendation_tools import (
    GenerateInsightsTool,
    PrioritizeActionsTool,
    CreateActionPlanTool
)
from src.tools.monitoring_tools import (
    CalculateKPIsTool,
    ComparePerformanceTool
)


def create_forecasting_agent() -> Agent:
    """Create a specialized agent for demand forecasting.
    
    Returns:
        Agent: A CrewAI agent specialized in demand forecasting.
    """
    tools = [
        ForecastDemandTool(),
        EvaluateModelPerformanceTool(),
        IdentifySalesPatternsTool()
    ]
    
    return Agent(
        role="Demand Forecasting Specialist",
        goal="Generate accurate demand forecasts using time series analysis and machine learning",
        backstory="""You are an expert in time series forecasting and demand prediction.
        Your background includes extensive experience with SARIMA, Exponential Smoothing,
        and Prophet models. You excel at identifying seasonal patterns and trends in
        sales data and selecting the most appropriate forecasting models for different
        product categories.""",
        verbose=True,
        allow_delegation=True,
        tools=tools
    )


def create_optimization_agent() -> Agent:
    """Create a specialized agent for inventory optimization.
    
    Returns:
        Agent: A CrewAI agent specialized in inventory optimization.
    """
    tools = [
        CalculateEconomicOrderQuantityTool(),
        CalculateReorderPointTool(), 
        CalculateSafetyStockTool(),
        DefineInventoryPolicyTool()
    ]
    
    return Agent(
        role="Inventory Optimization Specialist",
        goal="Determine optimal inventory policies that minimize costs while meeting service level targets",
        backstory="""You are an inventory optimization expert with deep knowledge of
        supply chain management. You've helped numerous companies implement efficient
        inventory policies including min-max levels, economic order quantities, and
        optimal reorder points. You balance the trade-offs between holding costs, 
        ordering costs, and stockout costs to minimize total inventory costs.""",
        verbose=True,
        allow_delegation=True,
        tools=tools
    )


def create_anomaly_detection_agent() -> Agent:
    """Create a specialized agent for anomaly detection.
    
    Returns:
        Agent: A CrewAI agent specialized in anomaly detection.
    """
    tools = [
        DetectSalesAnomaliesTool(),
        DetectInventoryAnomaliesTool()
    ]
    
    return Agent(
        role="Anomaly Detection Specialist",
        goal="Identify unusual patterns and anomalies in demand and inventory data",
        backstory="""You are an expert in statistical analysis and anomaly detection.
        Your expertise includes identifying outliers, sudden shifts in patterns, and
        unusual behaviors in time series data. You use various statistical methods
        and machine learning algorithms to detect anomalies that could indicate issues
        or opportunities in inventory management.""",
        verbose=True,
        allow_delegation=False,
        tools=tools
    )


def create_scenario_planning_agent() -> Agent:
    """Create a specialized agent for scenario planning.
    
    Returns:
        Agent: A CrewAI agent specialized in scenario planning.
    """
    tools = [
        SimulateSupplyDisruptionTool(),
        SimulateDemandSpikeTool(),
        SimulateSeasonalityTool()
    ]
    
    return Agent(
        role="Scenario Planning Specialist",
        goal="Simulate various business scenarios and their impact on inventory management",
        backstory="""You are a forward-thinking strategist specialized in scenario planning
        and simulation. Your background includes modeling complex business environments
        and predicting the impact of various scenarios on supply chain operations.
        You help companies prepare for uncertainties by simulating demand spikes,
        supply disruptions, and other potential challenges.""",
        verbose=True,
        allow_delegation=True,
        tools=tools
    )


def create_recommendation_agent() -> Agent:
    """Create a specialized agent for generating recommendations.
    
    Returns:
        Agent: A CrewAI agent specialized in recommendation generation.
    """
    tools = [
        GenerateInsightsTool(),
        PrioritizeActionsTool(),
        CreateActionPlanTool()
    ]
    
    return Agent(
        role="Inventory Strategy Advisor",
        goal="Generate actionable recommendations to optimize inventory management",
        backstory="""You are a strategic advisor with extensive experience in inventory
        management and supply chain optimization. You excel at synthesizing complex data
        and analyses into clear, actionable recommendations. Your background includes
        consulting for major retail and manufacturing companies, helping them transform
        their inventory management practices.""",
        verbose=True,
        allow_delegation=True,
        tools=tools
    )


def create_monitoring_agent() -> Agent:
    """Create a specialized agent for performance monitoring.
    
    Returns:
        Agent: A CrewAI agent specialized in performance monitoring.
    """
    tools = [
        CalculateKPIsTool(),
        ComparePerformanceTool()
    ]
    
    return Agent(
        role="Performance Monitoring Specialist",
        goal="Track inventory KPIs and compare performance against targets and benchmarks",
        backstory="""You are an analytics expert focused on performance monitoring and
        continuous improvement. You specialize in developing and tracking key performance
        indicators for inventory management, including service levels, inventory turnover,
        and carrying costs. You excel at identifying improvement opportunities through
        data analysis and visualization.""",
        verbose=True,
        allow_delegation=False,
        tools=tools
    ) 