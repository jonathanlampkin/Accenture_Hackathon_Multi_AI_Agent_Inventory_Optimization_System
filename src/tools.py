"""
Inventory Optimization Tools

This module defines the tools available to the CrewAI agents for inventory analysis,
forecasting, and optimization tasks.
"""
import logging
import numpy as np
# Import Pydantic v1 BaseModel
from pydantic.v1 import BaseModel, Field, ValidationError
from typing import List
from crewai_tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)

# --- Input Schemas (Optional: Can be used for validation *inside* functions if needed) ---
# Define these if you want explicit validation beyond type hints within the function body.
class ReorderPointInputInternal(BaseModel):
    avg_daily_demand: float
    lead_time: int
    safety_stock: float

class SafetyStockInputInternal(BaseModel):
    z_score: float = 1.65
    demand_std: float
    lead_time: int

class DemandForecastInputInternal(BaseModel):
    historical_demand: List[float]
    period: int = 3

class StockoutRiskInputInternal(BaseModel):
    current_inventory: float
    daily_demand: float
    lead_time: int

# --- Tool Definitions (Standalone Functions with Direct Args) ---

@tool("calculate_reorder_point")
def calculate_reorder_point_tool(avg_daily_demand: float, lead_time: int, safety_stock: float) -> str:
    """Calculates the reorder point (ROP) for inventory management.
    ROP = (Average Daily Demand × Lead Time) + Safety Stock.
    Use this tool when you need to determine the inventory level at which a new order should be placed.
    Args:
        avg_daily_demand (float): Average daily demand for the product.
        lead_time (int): Lead time in days for receiving inventory.
        safety_stock (float): Calculated safety stock level.
    """
    try:
        # Optional internal validation:
        # ReorderPointInputInternal(avg_daily_demand=avg_daily_demand, lead_time=lead_time, safety_stock=safety_stock)
        
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        logger.info(f"Calculated reorder point: {reorder_point:.2f}")
        return f"Reorder Point calculated: {reorder_point:.2f} units"
    # except ValidationError as ve:
    #     logger.error(f"Input validation error in calculate_reorder_point: {ve}")
    #     return f"Error: Invalid input - {ve}"
    except Exception as e:
        logger.error(f"Error in calculate_reorder_point: {e}")
        return f"Error calculating reorder point: {str(e)}"

@tool("calculate_safety_stock")
def calculate_safety_stock_tool(demand_std: float, lead_time: int, z_score: float = 1.65) -> str:
    """Calculates the safety stock level needed to maintain a desired service level.
    Safety Stock = Z-score × Standard Deviation of Demand × √Lead Time.
    Use this tool to determine the buffer stock needed to account for demand variability.
    Args:
        demand_std (float): Standard deviation of daily demand.
        lead_time (int): Lead time in days.
        z_score (float): Z-score for service level (default: 1.65 for 95%).
    """
    try:
        # Optional internal validation:
        # SafetyStockInputInternal(demand_std=demand_std, lead_time=lead_time, z_score=z_score)
        
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        logger.info(f"Calculated safety stock: {safety_stock:.2f}")
        return f"Safety Stock calculated: {safety_stock:.2f} units"
    # except ValidationError as ve:
    #     logger.error(f"Input validation error in calculate_safety_stock: {ve}")
    #     return f"Error: Invalid input - {ve}"
    except Exception as e:
        logger.error(f"Error in calculate_safety_stock: {e}")
        return f"Error calculating safety stock: {str(e)}"

@tool("forecast_demand")
def forecast_demand_tool(historical_demand: List[float], period: int = 3) -> str:
    """Forecasts future demand using a simple moving average based on historical data.
    Use this tool to predict upcoming demand based on recent trends.
    Args:
        historical_demand (List[float]): List of historical demand figures.
        period (int): Number of periods for SMA (default: 3).
    """
    try:
        # Optional internal validation:
        # DemandForecastInputInternal(historical_demand=historical_demand, period=period)

        if not isinstance(historical_demand, list) or not all(isinstance(x, (int, float)) for x in historical_demand):
             raise ValueError("historical_demand must be a list of numbers.")
        if not isinstance(period, int) or period <= 0:
             raise ValueError("period must be a positive integer.")

        if len(historical_demand) < period:
            logger.warning(f"Not enough historical data ({len(historical_demand)}) for period {period}.")
            return f"Not enough historical data (need {period}, have {len(historical_demand)}) for forecasting."
        
        forecast = np.mean(historical_demand[-period:])
        logger.info(f"Forecasted demand: {forecast:.2f}")
        return f"Forecasted Demand (next period, {period}-period SMA): {forecast:.2f} units"
    # except ValidationError as ve:
    #     logger.error(f"Input validation error in forecast_demand: {ve}")
    #     return f"Error: Invalid input - {ve}"
    except Exception as e:
        logger.error(f"Error in forecast_demand: {e}")
        return f"Error forecasting demand: {str(e)}"

@tool("analyze_stockout_risk")
def analyze_stockout_risk_tool(current_inventory: float, daily_demand: float, lead_time: int) -> str:
    """Analyzes the risk of running out of stock based on current levels, demand, and lead time.
    Estimates days until potential stockout and classifies risk.
    Use this tool to evaluate the urgency of needing to reorder.
    Args:
        current_inventory (float): Current inventory level.
        daily_demand (float): Estimated average daily demand.
        lead_time (int): Lead time in days.
    """
    try:
        # Optional internal validation:
        # StockoutRiskInputInternal(current_inventory=current_inventory, daily_demand=daily_demand, lead_time=lead_time)

        if daily_demand < 0: # Allow zero demand
            raise ValueError("daily_demand cannot be negative.")

        if daily_demand == 0:
             days_until_stockout = float('inf')
             risk_level = "Very Low (Zero Demand)"
        else:
            days_until_stockout = current_inventory / daily_demand
            if days_until_stockout < lead_time:
                risk_level = "High"
            elif days_until_stockout < lead_time * 1.5:
                risk_level = "Medium"
            else:
                risk_level = "Low"
        
        logger.info(f"Stockout risk analysis: Level={risk_level}, Days_until_stockout={days_until_stockout:.2f}")
        return f"Stockout Risk: {risk_level} (Estimated days until stockout: {days_until_stockout:.2f})"
    # except ValidationError as ve:
    #     logger.error(f"Input validation error in analyze_stockout_risk: {ve}")
    #     return f"Error: Invalid input - {ve}"
    except Exception as e:
        logger.error(f"Error in analyze_stockout_risk: {e}")
        return f"Error analyzing stockout risk: {str(e)}"

# Instantiate the search tool
search_tool = DuckDuckGoSearchRun()

# --- Tool Groupings (Using standalone functions) ---

# Tools for Demand Analyst
demand_analyst_tools = [
    forecast_demand_tool,
    search_tool
]

# Tools for Inventory Optimizer
inventory_optimizer_tools = [
    calculate_reorder_point_tool,
    calculate_safety_stock_tool,
    analyze_stockout_risk_tool
]

# Tools for Supply Chain Analyst
supply_chain_analyst_tools = [
    analyze_stockout_risk_tool,
    calculate_safety_stock_tool,
    search_tool
]

# Tools for Risk Analyst
risk_analyst_tools = [
    analyze_stockout_risk_tool,
    calculate_safety_stock_tool,
    search_tool
]

# List of all unique tool instances (optional, might not be needed)
# all_tools_instances = [inventory_tools, search_tool]

# You can potentially wrap DuckDuckGoSearchRun in a @tool decorator as well if needed
# Example:
# @tool("internet_search")
# def internet_search(query: str) -> str:
#    """Performs an internet search using DuckDuckGo."""
#    return DuckDuckGoSearchRun().run(query) 