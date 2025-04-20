"""
Inventory Optimization Tools

This module defines the tools available to the CrewAI agents for inventory analysis,
forecasting, and optimization tasks.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
# Import Pydantic v1 BaseModel
from pydantic.v1 import BaseModel, Field, ValidationError
from crewai_tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchRun

# Import forecasting tools
try:
    from src.models.forecasting_integration import forecasting_tools
except ImportError:
    # Create placeholder if module doesn't exist
    forecasting_tools = []
    print("Warning: Forecasting integration module not found. Using empty tools list.")

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

# Tools for Demand Analyst - Add advanced forecasting tools
demand_analyst_tools = [
    forecast_demand_tool,  # Simple moving average (keep for backwards compatibility)
    search_tool,
    *forecasting_tools     # Add all advanced forecasting tools
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

def calculate_reorder_point(
    average_daily_demand: float,
    lead_time_days: float,
    safety_stock: float,
    product_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate the reorder point for a product.
    
    Args:
        average_daily_demand: Average daily demand for the product
        lead_time_days: Supplier lead time in days
        safety_stock: Safety stock level in units
        product_info: Optional dictionary with additional product information
        
    Returns:
        Dictionary with reorder point and calculation details
    """
    try:
        # Basic reorder point calculation: Lead time demand + safety stock
        lead_time_demand = average_daily_demand * lead_time_days
        reorder_point = lead_time_demand + safety_stock
        
        # Additional adjustments based on product info
        adjustment = 0
        
        if product_info:
            # Adjust for product perishability
            if product_info.get('is_perishable', False):
                # Perishable products may need lower reorder points
                adjustment -= lead_time_demand * 0.1
            
            # Adjust for product criticality
            if product_info.get('criticality', '').lower() == 'high':
                # Critical products may need higher reorder points
                adjustment += lead_time_demand * 0.2
        
        final_reorder_point = max(0, reorder_point + adjustment)
        
        return {
            'reorder_point': round(final_reorder_point, 2),
            'lead_time_demand': round(lead_time_demand, 2),
            'safety_stock': round(safety_stock, 2),
            'adjustment': round(adjustment, 2),
            'rationale': f"Reorder point calculated as Lead Time Demand ({lead_time_demand:.2f}) + "
                        f"Safety Stock ({safety_stock:.2f}) + Adjustment ({adjustment:.2f})"
        }
    
    except Exception as e:
        logger.error(f"Error calculating reorder point: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'reorder_point': 0,
            'lead_time_demand': 0,
            'safety_stock': 0
        }


def calculate_safety_stock(
    avg_demand: float, 
    std_demand: float,
    lead_time: float,
    std_lead_time: float = 0,
    service_level: float = 0.95,
    product_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate safety stock level for a product.
    
    Args:
        avg_demand: Average daily demand
        std_demand: Standard deviation of daily demand
        lead_time: Average lead time in days
        std_lead_time: Standard deviation of lead time in days
        service_level: Target service level (0-1)
        product_info: Optional dictionary with additional product information
        
    Returns:
        Dictionary with safety stock and calculation details
    """
    try:
        # Calculate safety factor based on service level
        # This is a simplified approach; in practice, use a normal distribution function
        service_factors = {
            0.5: 0,
            0.75: 0.67,
            0.8: 0.84,
            0.85: 1.04,
            0.9: 1.28,
            0.95: 1.64,
            0.98: 2.05,
            0.99: 2.33,
            0.995: 2.58
        }
        
        # Find closest service level in our table
        service_factor = service_factors.get(service_level, 1.64)  # Default to 95% if not found
        
        # Calculate safety stock using the formula:
        # Safety Stock = Z × sqrt(LT × σD² + D² × σLT²)
        # where Z is the service factor, LT is lead time, D is average demand,
        # σD is standard deviation of demand, and σLT is standard deviation of lead time
        
        # Simplified version if lead time variation is not available
        if std_lead_time <= 0:
            safety_stock = service_factor * std_demand * np.sqrt(lead_time)
        else:
            safety_stock = service_factor * np.sqrt(
                (lead_time * std_demand ** 2) + 
                (avg_demand ** 2 * std_lead_time ** 2)
            )
        
        # Adjustments based on product info
        adjustment = 0
        
        if product_info:
            # Adjust for perishability
            if product_info.get('is_perishable', False):
                # Reduce safety stock for perishable items
                adjustment -= safety_stock * 0.2
            
            # Adjust for product value
            if product_info.get('unit_cost', 0) > 100:
                # Reduce safety stock for high-value items
                adjustment -= safety_stock * 0.1
            
            # Adjust for criticality
            if product_info.get('criticality', '').lower() == 'high':
                # Increase safety stock for critical items
                adjustment += safety_stock * 0.3
        
        final_safety_stock = max(0, safety_stock + adjustment)
        
        return {
            'safety_stock': round(final_safety_stock, 2),
            'base_safety_stock': round(safety_stock, 2),
            'adjustment': round(adjustment, 2),
            'service_level': service_level,
            'service_factor': service_factor,
            'rationale': f"Safety stock calculated with service level {service_level*100:.1f}%, "
                        f"lead time {lead_time} days, and standard deviation of demand {std_demand:.2f} units."
        }
    
    except Exception as e:
        logger.error(f"Error calculating safety stock: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'safety_stock': 0,
            'service_level': service_level
        }


def analyze_product_performance(
    product_id: str,
    sales_data: List[Dict[str, Any]],
    inventory_data: List[Dict[str, Any]],
    period_days: int = 90
) -> Dict[str, Any]:
    """
    Analyze performance of a specific product.
    
    Args:
        product_id: Product ID to analyze
        sales_data: List of sales data dictionaries
        inventory_data: List of inventory data dictionaries
        period_days: Analysis period in days
        
    Returns:
        Dictionary with product performance metrics
    """
    try:
        # Convert to DataFrames for easier manipulation
        sales_df = pd.DataFrame(sales_data)
        inventory_df = pd.DataFrame(inventory_data)
        
        # Filter data for specific product
        product_sales = sales_df[sales_df['product_id'] == product_id]
        product_inventory = inventory_df[inventory_df['product_id'] == product_id]
        
        # Check if we have data
        if product_sales.empty or product_inventory.empty:
            return {
                'warning': f"Insufficient data for product {product_id}",
                'product_id': product_id,
                'metrics': {}
            }
        
        # Calculate metrics
        total_sales = product_sales['quantity'].sum()
        total_revenue = (product_sales['quantity'] * product_sales.get('price', 0)).sum()
        avg_price = total_revenue / total_sales if total_sales > 0 else 0
        
        # Calculate average inventory
        avg_inventory = product_inventory['quantity'].mean()
        
        # Calculate inventory turnover
        inventory_turnover = total_sales / avg_inventory if avg_inventory > 0 else 0
        
        # Calculate days of supply
        days_of_supply = (avg_inventory / (total_sales / period_days)) if total_sales > 0 else float('inf')
        
        # Calculate stockout frequency
        stockout_count = (product_inventory['quantity'] <= 0).sum()
        stockout_rate = stockout_count / len(product_inventory) if len(product_inventory) > 0 else 0
        
        # Calculate profit margin if cost data is available
        profit_margin = 0
        if 'cost' in product_sales.columns:
            total_cost = (product_sales['quantity'] * product_sales['cost']).sum()
            profit = total_revenue - total_cost
            profit_margin = profit / total_revenue if total_revenue > 0 else 0
        
        return {
            'product_id': product_id,
            'metrics': {
                'total_sales_quantity': int(total_sales),
                'total_revenue': round(total_revenue, 2),
                'average_price': round(avg_price, 2),
                'average_inventory': round(avg_inventory, 2),
                'inventory_turnover': round(inventory_turnover, 2),
                'days_of_supply': round(days_of_supply, 2),
                'stockout_rate': round(stockout_rate, 4),
                'profit_margin': round(profit_margin, 4) if profit_margin else None
            },
            'period_days': period_days,
            'recommendations': _generate_product_recommendations(
                inventory_turnover, days_of_supply, stockout_rate, profit_margin
            )
        }
    
    except Exception as e:
        logger.error(f"Error analyzing product performance: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'product_id': product_id,
            'metrics': {}
        }


def _generate_product_recommendations(
    inventory_turnover: float,
    days_of_supply: float,
    stockout_rate: float,
    profit_margin: float
) -> List[Dict[str, str]]:
    """
    Generate product recommendations based on performance metrics.
    
    Args:
        inventory_turnover: Inventory turnover ratio
        days_of_supply: Days of supply
        stockout_rate: Stockout rate
        profit_margin: Profit margin
        
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    # Check inventory turnover
    if inventory_turnover < 2:
        recommendations.append({
            'type': 'warning',
            'area': 'inventory',
            'message': f"Low inventory turnover ({inventory_turnover:.2f}). Consider reducing order quantities or promotional activities."
        })
    elif inventory_turnover > 12:
        recommendations.append({
            'type': 'info',
            'area': 'inventory',
            'message': f"High inventory turnover ({inventory_turnover:.2f}). May indicate need for increased safety stock or order frequency."
        })
    
    # Check days of supply
    if days_of_supply > 90:
        recommendations.append({
            'type': 'warning',
            'area': 'inventory',
            'message': f"High days of supply ({days_of_supply:.1f} days). Consider reducing inventory levels or promotional activities."
        })
    elif days_of_supply < 10:
        recommendations.append({
            'type': 'warning',
            'area': 'inventory',
            'message': f"Low days of supply ({days_of_supply:.1f} days). May indicate risk of stockout."
        })
    
    # Check stockout rate
    if stockout_rate > 0.05:
        recommendations.append({
            'type': 'alert',
            'area': 'service',
            'message': f"High stockout rate ({stockout_rate:.1%}). Increase safety stock or reorder point."
        })
    
    # Check profit margin
    if profit_margin is not None:
        if profit_margin < 0.15:
            recommendations.append({
                'type': 'warning',
                'area': 'pricing',
                'message': f"Low profit margin ({profit_margin:.1%}). Consider pricing adjustments or cost reduction."
            })
    
    return recommendations


def identify_anomalies(
    data: List[Dict[str, Any]],
    value_column: str = 'quantity',
    date_column: str = 'date',
    method: str = 'z-score',
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Identify anomalies in time series data.
    
    Args:
        data: List of data dictionaries
        value_column: Column name for the value to analyze
        date_column: Column name for the date
        method: Method to use for anomaly detection ('z-score', 'iqr', 'moving_avg')
        threshold: Threshold for anomaly detection
        
    Returns:
        Dictionary with identified anomalies
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure date column is datetime
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            # Sort by date
            df = df.sort_values(date_column)
        
        # Check if value column exists
        if value_column not in df.columns:
            return {
                'error': f"Column '{value_column}' not found in data",
                'anomalies': []
            }
        
        # Detect anomalies based on selected method
        anomalies = []
        
        if method == 'z-score':
            # Z-score method
            mean = df[value_column].mean()
            std = df[value_column].std()
            
            # If standard deviation is 0, return empty result
            if std == 0:
                return {
                    'warning': "Standard deviation is 0, cannot detect anomalies with z-score method",
                    'anomalies': [],
                    'stats': {'mean': mean, 'std': std}
                }
            
            # Calculate z-scores
            z_scores = (df[value_column] - mean) / std
            
            # Find anomalies
            anomaly_indices = np.abs(z_scores) > threshold
            anomaly_df = df[anomaly_indices].copy()
            
            if not anomaly_df.empty:
                anomaly_df['z_score'] = z_scores[anomaly_indices]
                anomaly_df['direction'] = np.where(z_scores[anomaly_indices] > 0, 'high', 'low')
                
                # Format anomalies
                for _, row in anomaly_df.iterrows():
                    anomaly = {
                        'date': row[date_column].strftime('%Y-%m-%d') if date_column in row else None,
                        'value': row[value_column],
                        'z_score': row['z_score'],
                        'direction': row['direction'],
                        'deviation_percent': abs((row[value_column] - mean) / mean * 100) if mean != 0 else float('inf')
                    }
                    # Add all other columns
                    for col in row.index:
                        if col not in [date_column, value_column, 'z_score', 'direction']:
                            anomaly[col] = row[col]
                    
                    anomalies.append(anomaly)
        
        elif method == 'iqr':
            # IQR method
            q1 = df[value_column].quantile(0.25)
            q3 = df[value_column].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Find anomalies
            anomaly_low = df[df[value_column] < lower_bound].copy()
            anomaly_high = df[df[value_column] > upper_bound].copy()
            
            # Format low anomalies
            for _, row in anomaly_low.iterrows():
                anomaly = {
                    'date': row[date_column].strftime('%Y-%m-%d') if date_column in row else None,
                    'value': row[value_column],
                    'direction': 'low',
                    'bound': lower_bound,
                    'deviation_percent': abs((row[value_column] - q1) / q1 * 100) if q1 != 0 else float('inf')
                }
                # Add all other columns
                for col in row.index:
                    if col not in [date_column, value_column]:
                        anomaly[col] = row[col]
                
                anomalies.append(anomaly)
            
            # Format high anomalies
            for _, row in anomaly_high.iterrows():
                anomaly = {
                    'date': row[date_column].strftime('%Y-%m-%d') if date_column in row else None,
                    'value': row[value_column],
                    'direction': 'high',
                    'bound': upper_bound,
                    'deviation_percent': abs((row[value_column] - q3) / q3 * 100) if q3 != 0 else float('inf')
                }
                # Add all other columns
                for col in row.index:
                    if col not in [date_column, value_column]:
                        anomaly[col] = row[col]
                
                anomalies.append(anomaly)
        
        elif method == 'moving_avg':
            # Moving average method
            window = min(7, len(df) // 3)  # Use smaller window for small datasets
            if window < 2:
                return {
                    'warning': "Insufficient data for moving average method",
                    'anomalies': []
                }
            
            # Calculate moving average
            df['moving_avg'] = df[value_column].rolling(window=window, center=True).mean()
            
            # Fill NaN values at the beginning and end
            df['moving_avg'] = df['moving_avg'].fillna(df[value_column])
            
            # Calculate deviation from moving average
            df['deviation'] = df[value_column] - df['moving_avg']
            df['deviation_pct'] = df['deviation'] / df['moving_avg'] * 100
            
            # Find anomalies
            anomaly_df = df[np.abs(df['deviation_pct']) > threshold * 100].copy()
            
            if not anomaly_df.empty:
                # Format anomalies
                for _, row in anomaly_df.iterrows():
                    anomaly = {
                        'date': row[date_column].strftime('%Y-%m-%d') if date_column in row else None,
                        'value': row[value_column],
                        'moving_avg': row['moving_avg'],
                        'deviation_percent': row['deviation_pct'],
                        'direction': 'high' if row['deviation'] > 0 else 'low'
                    }
                    # Add all other columns
                    for col in row.index:
                        if col not in [date_column, value_column, 'moving_avg', 'deviation', 'deviation_pct']:
                            anomaly[col] = row[col]
                    
                    anomalies.append(anomaly)
        
        else:
            return {
                'error': f"Unknown anomaly detection method: {method}",
                'anomalies': []
            }
        
        # Sort anomalies by date if available
        if anomalies and date_column in df.columns:
            anomalies = sorted(anomalies, key=lambda x: x.get('date', ''))
        
        # Sort by deviation percentage (descending) as a secondary sort
        anomalies = sorted(anomalies, key=lambda x: x.get('deviation_percent', 0), reverse=True)
        
        return {
            'anomalies': anomalies,
            'method': method,
            'threshold': threshold,
            'count': len(anomalies),
            'analyzed_records': len(df)
        }
    
    except Exception as e:
        logger.error(f"Error identifying anomalies: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'anomalies': []
        }


def forecast_demand(
    historical_data: List[Dict[str, Any]],
    forecast_periods: int = 30,
    product_id: Optional[str] = None,
    date_column: str = 'date',
    value_column: str = 'quantity',
    freq: str = 'D'
) -> Dict[str, Any]:
    """
    Forecast demand based on historical data.
    
    Args:
        historical_data: List of historical data dictionaries
        forecast_periods: Number of periods to forecast
        product_id: Optional product ID to filter data
        date_column: Column name for the date
        value_column: Column name for the value to forecast
        freq: Frequency of the time series ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        Dictionary with forecast results
    """
    try:
        # Import forecast libraries only when needed
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings('ignore')  # Suppress statsmodels warnings
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Filter by product if specified
        if product_id is not None:
            df = df[df['product_id'] == product_id]
        
        # Check if we have data
        if len(df) < 10:  # Need minimum data for forecasting
            return {
                'warning': f"Insufficient data for forecasting. Need at least 10 records, found {len(df)}.",
                'forecast': [],
                'product_id': product_id
            }
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Set date as index
        df = df.set_index(date_column)
        
        # Resample to ensure regular intervals
        ts = df[value_column].resample(freq).mean().fillna(method='ffill')
        
        # Split into train and test
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        
        # Try multiple models and use the best one
        
        # 1. Simple Moving Average
        def sma_forecast(train, test, periods):
            window = min(7, len(train) // 4)
            sma = train.rolling(window=window).mean()
            # Fill NaN values at the beginning
            sma = sma.fillna(train.mean())
            # Forecast is the last SMA value repeated
            forecast = pd.Series([sma.iloc[-1]] * (len(test) + periods),
                                index=test.index.union(pd.date_range(test.index[-1] + pd.Timedelta(days=1), 
                                                                   periods=periods, freq=freq)))
            return sma, forecast
        
        # 2. Exponential Smoothing
        def exp_smoothing_forecast(train, test, periods):
            model = ExponentialSmoothing(train, 
                                        trend='add', 
                                        seasonal='add', 
                                        seasonal_periods=7)
            fitted_model = model.fit()
            # Forecast for test period and future periods
            forecast_horizon = len(test) + periods
            forecast = fitted_model.forecast(forecast_horizon)
            return fitted_model.fittedvalues, forecast
        
        # 3. ARIMA
        def arima_forecast(train, test, periods):
            # Simple ARIMA model - can be improved with parameter tuning
            model = ARIMA(train, order=(1, 1, 1))
            fitted_model = model.fit()
            # Forecast for test period and future periods
            forecast_horizon = len(test) + periods
            forecast = fitted_model.forecast(forecast_horizon)
            return fitted_model.fittedvalues, forecast
        
        # Try all models and select best
        models = [
            ('SMA', sma_forecast),
            ('Exponential Smoothing', exp_smoothing_forecast),
            ('ARIMA', arima_forecast)
        ]
        
        best_rmse = float('inf')
        best_model_name = None
        best_train_pred = None
        best_forecast = None
        
        for model_name, model_func in models:
            try:
                # Fit model and generate forecast
                train_pred, forecast = model_func(train, test, forecast_periods)
                
                # Calculate RMSE on test data
                test_pred = forecast[:len(test)]
                rmse = np.sqrt(np.mean((test - test_pred) ** 2))
                
                # Update best model if this one is better
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = model_name
                    best_train_pred = train_pred
                    best_forecast = forecast
            except Exception as model_error:
                logger.warning(f"Error with {model_name} model: {str(model_error)}")
                continue
        
        # If no model worked, fall back to simple moving average
        if best_model_name is None:
            logger.warning("All forecasting models failed, falling back to simple average")
            avg_value = train.mean()
            # Create forecast for test + future periods
            forecast_idx = test.index.union(pd.date_range(test.index[-1] + pd.Timedelta(days=1), 
                                                        periods=forecast_periods, freq=freq))
            best_forecast = pd.Series([avg_value] * (len(test) + forecast_periods), index=forecast_idx)
            best_model_name = "Average"
            best_rmse = np.sqrt(np.mean((test - avg_value) ** 2))
            best_train_pred = pd.Series([avg_value] * len(train), index=train.index)
        
        # Prepare results
        forecast_results = []
        
        # Split forecast into test period and future period
        test_forecast = best_forecast[:len(test)]
        future_forecast = best_forecast[len(test):]
        
        # Add test period forecasts with actuals for comparison
        for date, actual, predicted in zip(test.index, test.values, test_forecast.values):
            forecast_results.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': float(predicted),
                'actual': float(actual),
                'period_type': 'test'
            })
        
        # Add future period forecasts
        for date, predicted in zip(future_forecast.index, future_forecast.values):
            forecast_results.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': float(predicted),
                'actual': None,
                'period_type': 'forecast'
            })
        
        # Calculate metrics
        mae = np.mean(np.abs(test - test_forecast))
        mape = np.mean(np.abs((test - test_forecast) / test)) * 100
        
        return {
            'product_id': product_id,
            'forecast': forecast_results,
            'metrics': {
                'rmse': float(best_rmse),
                'mae': float(mae),
                'mape': float(mape)
            },
            'model': best_model_name,
            'forecast_periods': forecast_periods,
            'train_size': len(train),
            'test_size': len(test)
        }
    
    except Exception as e:
        logger.error(f"Error forecasting demand: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'forecast': [],
            'product_id': product_id
        } 