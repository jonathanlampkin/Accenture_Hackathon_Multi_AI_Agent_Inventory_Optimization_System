"""
Inventory optimization utilities for calculating key inventory metrics.

This module provides functions for calculating safety stock, reorder points,
economic order quantities, and other inventory management metrics.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
import scipy.stats as stats
import logging

# Setup logging
logger = logging.getLogger(__name__)


def calculate_safety_stock(
    demand_std: float,
    lead_time: float,
    service_level: float,
    lead_time_std: Optional[float] = None,
    review_period: Optional[float] = 0
) -> float:
    """
    Calculate safety stock using the standard formula.
    
    Args:
        demand_std: Standard deviation of demand (per day/week)
        lead_time: Average lead time (in days/weeks)
        service_level: Desired service level (between 0 and 1)
        lead_time_std: Standard deviation of lead time (optional)
        review_period: Review period length (0 for continuous review)
        
    Returns:
        Safety stock quantity
    """
    # Calculate Z-score for the desired service level
    z_score = stats.norm.ppf(service_level)
    
    # Calculate safety stock
    if lead_time_std is not None:
        # Use formula that accounts for lead time variability
        safety_stock = z_score * math.sqrt(
            (lead_time * demand_std**2) + 
            ((demand_std**2) * (lead_time_std**2))
        )
    else:
        # Standard formula for fixed lead time
        safety_stock = z_score * demand_std * math.sqrt(lead_time + review_period)
    
    return max(0, safety_stock)


def calculate_reorder_point(
    avg_daily_demand: float,
    lead_time: float,
    safety_stock: float
) -> float:
    """
    Calculate reorder point.
    
    Args:
        avg_daily_demand: Average daily demand
        lead_time: Lead time in days
        safety_stock: Safety stock quantity
        
    Returns:
        Reorder point quantity
    """
    return (avg_daily_demand * lead_time) + safety_stock


def calculate_economic_order_quantity(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_rate: float,
    unit_cost: float
) -> float:
    """
    Calculate Economic Order Quantity (EOQ) using Wilson's formula.
    
    Args:
        annual_demand: Annual demand in units
        ordering_cost: Cost per order
        holding_cost_rate: Annual holding cost as a fraction of unit cost
        unit_cost: Cost per unit
        
    Returns:
        Economic order quantity
    """
    try:
        holding_cost = holding_cost_rate * unit_cost
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return max(1, round(eoq))
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating EOQ: {str(e)}")
        return 1


def calculate_min_max_levels(
    avg_daily_demand: float,
    lead_time: float,
    safety_stock: float,
    review_period: float,
    eoq: Optional[float] = None,
    min_max_factor: float = 1.5
) -> Tuple[float, float]:
    """
    Calculate minimum and maximum inventory levels.
    
    Args:
        avg_daily_demand: Average daily demand
        lead_time: Lead time in days
        safety_stock: Safety stock quantity
        review_period: Review period in days
        eoq: Economic order quantity (optional)
        min_max_factor: Factor for max level calculation
        
    Returns:
        Tuple of (min_level, max_level)
    """
    # Min level = reorder point
    min_level = calculate_reorder_point(avg_daily_demand, lead_time, safety_stock)
    
    # Max level calculation
    if eoq is not None:
        # If EOQ is provided, use it for max level calculation
        max_level = min_level + eoq
    else:
        # Otherwise use the min_max_factor
        max_level = min_level + (avg_daily_demand * review_period * min_max_factor)
    
    return min_level, max_level


def calculate_service_level_from_stockouts(
    demand_history: pd.Series,
    stockout_events: int
) -> float:
    """
    Calculate achieved service level from stockout history.
    
    Args:
        demand_history: Series of historical demand events
        stockout_events: Number of stockout events
        
    Returns:
        Achieved service level (0-1)
    """
    total_demand_events = len(demand_history)
    if total_demand_events == 0:
        return 1.0
    
    service_level = 1 - (stockout_events / total_demand_events)
    return max(0, min(1, service_level))


def calculate_fill_rate(
    demand_quantity: float,
    stockout_quantity: float
) -> float:
    """
    Calculate fill rate based on demand quantity and stockout quantity.
    
    Args:
        demand_quantity: Total demand quantity
        stockout_quantity: Total stockout quantity
        
    Returns:
        Fill rate (0-1)
    """
    if demand_quantity == 0:
        return 1.0
    
    fill_rate = 1 - (stockout_quantity / demand_quantity)
    return max(0, min(1, fill_rate))


def simulate_inventory_policy(
    demand_series: pd.Series,
    lead_time: float,
    initial_inventory: float,
    reorder_point: float,
    order_quantity: float,
    review_period: int = 1,
    max_inventory: Optional[float] = None
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Simulate inventory policy over time.
    
    Args:
        demand_series: Time series of demand
        lead_time: Lead time in days
        initial_inventory: Starting inventory level
        reorder_point: Reorder point quantity
        order_quantity: Order quantity
        review_period: Review period in days
        max_inventory: Maximum inventory level
        
    Returns:
        Dictionary containing simulation results
    """
    # Initialize simulation variables
    inventory = initial_inventory
    order_placed = False
    order_arrival_day = 0
    orders_outstanding = []
    
    # Results tracking
    results = {
        'inventory_level': [],
        'demand': [],
        'order_placed': [],
        'order_received': [],
        'stockout': []
    }
    
    # Run simulation
    for day, demand in enumerate(demand_series):
        # Check for order arrivals
        order_received = 0
        orders_to_remove = []
        
        for i, (arrival_day, qty) in enumerate(orders_outstanding):
            if day >= arrival_day:
                inventory += qty
                order_received += qty
                orders_to_remove.append(i)
        
        # Remove fulfilled orders
        for i in sorted(orders_to_remove, reverse=True):
            orders_outstanding.pop(i)
        
        # Process demand
        stockout = max(0, demand - inventory)
        inventory = max(0, inventory - demand)
        
        # Check for reordering (on review days)
        new_order = 0
        if day % review_period == 0 and inventory <= reorder_point and not order_placed:
            # Determine order quantity
            if max_inventory is not None:
                # Order-up-to policy
                new_order = max_inventory - inventory
            else:
                # Fixed order quantity
                new_order = order_quantity
            
            # Place order if quantity > 0
            if new_order > 0:
                arrival_day = day + lead_time
                orders_outstanding.append((arrival_day, new_order))
                order_placed = True
        
        # Reset order flag after ordering
        if order_placed:
            order_placed = False
        
        # Store results
        results['inventory_level'].append(inventory)
        results['demand'].append(demand)
        results['order_placed'].append(new_order)
        results['order_received'].append(order_received)
        results['stockout'].append(stockout)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    total_demand = results_df['demand'].sum()
    total_stockout = results_df['stockout'].sum()
    stockout_days = (results_df['stockout'] > 0).sum()
    total_days = len(results_df)
    
    metrics = {
        'fill_rate': calculate_fill_rate(total_demand, total_stockout),
        'service_level': 1 - (stockout_days / total_days),
        'average_inventory': results_df['inventory_level'].mean(),
        'max_inventory': results_df['inventory_level'].max(),
        'min_inventory': results_df['inventory_level'].min(),
        'stockout_days': stockout_days,
        'total_orders': (results_df['order_placed'] > 0).sum(),
        'total_order_quantity': results_df['order_placed'].sum()
    }
    
    return {
        'simulation': results_df,
        'metrics': metrics
    }


def optimize_inventory_parameters(
    demand_history: pd.Series,
    lead_time: float,
    unit_cost: float,
    holding_cost_rate: float = 0.25,
    ordering_cost: float = 25.0,
    service_level_target: float = 0.95,
    lead_time_std: Optional[float] = None,
    review_period: int = 7
) -> Dict[str, float]:
    """
    Optimize inventory parameters based on historical demand.
    
    Args:
        demand_history: Time series of historical demand
        lead_time: Lead time in days
        unit_cost: Cost per unit
        holding_cost_rate: Annual holding cost as a fraction of unit cost
        ordering_cost: Cost per order
        service_level_target: Target service level
        lead_time_std: Standard deviation of lead time (optional)
        review_period: Review period in days
        
    Returns:
        Dictionary of optimized inventory parameters
    """
    # Calculate demand statistics
    avg_daily_demand = demand_history.mean()
    demand_std = demand_history.std() or 0.1  # Avoid zero std
    annual_demand = avg_daily_demand * 365
    
    # Calculate EOQ
    eoq = calculate_economic_order_quantity(
        annual_demand=annual_demand,
        ordering_cost=ordering_cost,
        holding_cost_rate=holding_cost_rate,
        unit_cost=unit_cost
    )
    
    # Calculate safety stock
    ss = calculate_safety_stock(
        demand_std=demand_std,
        lead_time=lead_time,
        service_level=service_level_target,
        lead_time_std=lead_time_std,
        review_period=review_period
    )
    
    # Calculate reorder point
    rop = calculate_reorder_point(
        avg_daily_demand=avg_daily_demand,
        lead_time=lead_time,
        safety_stock=ss
    )
    
    # Calculate min/max levels
    min_level, max_level = calculate_min_max_levels(
        avg_daily_demand=avg_daily_demand,
        lead_time=lead_time,
        safety_stock=ss,
        review_period=review_period,
        eoq=eoq
    )
    
    # Return optimized parameters
    return {
        'economic_order_quantity': eoq,
        'safety_stock': ss,
        'reorder_point': rop,
        'min_level': min_level,
        'max_level': max_level,
        'avg_daily_demand': avg_daily_demand,
        'demand_std': demand_std
    }


def calculate_inventory_costs(
    inventory_levels: List[float],
    order_quantities: List[float],
    stockout_quantities: List[float],
    unit_cost: float,
    holding_cost_rate: float,
    ordering_cost: float,
    stockout_cost: float
) -> Dict[str, float]:
    """
    Calculate inventory-related costs.
    
    Args:
        inventory_levels: List of daily inventory levels
        order_quantities: List of daily order quantities
        stockout_quantities: List of daily stockout quantities
        unit_cost: Cost per unit
        holding_cost_rate: Annual holding cost as a fraction of unit cost
        ordering_cost: Cost per order
        stockout_cost: Cost per stockout unit
        
    Returns:
        Dictionary of cost components
    """
    # Calculate daily holding cost rate
    daily_holding_cost_rate = holding_cost_rate / 365
    
    # Calculate holding costs
    daily_holding_costs = [level * unit_cost * daily_holding_cost_rate 
                          for level in inventory_levels]
    total_holding_cost = sum(daily_holding_costs)
    
    # Calculate ordering costs
    order_events = sum(1 for qty in order_quantities if qty > 0)
    total_ordering_cost = order_events * ordering_cost
    
    # Calculate stockout costs
    total_stockout_cost = sum(stockout_quantities) * stockout_cost
    
    # Total costs
    total_cost = total_holding_cost + total_ordering_cost + total_stockout_cost
    
    return {
        'holding_cost': total_holding_cost,
        'ordering_cost': total_ordering_cost,
        'stockout_cost': total_stockout_cost,
        'total_cost': total_cost
    }


def run_scenario_analysis(
    base_demand: pd.Series,
    lead_time: float,
    inventory_params: Dict[str, float],
    scenarios: Dict[str, Dict[str, float]]
) -> Dict[str, Dict]:
    """
    Run scenario analysis for different demand and lead time scenarios.
    
    Args:
        base_demand: Base demand time series
        lead_time: Base lead time
        inventory_params: Base inventory parameters
        scenarios: Dictionary of scenarios with demand and lead time factors
        
    Returns:
        Dictionary of scenario results
    """
    results = {}
    
    for scenario_name, factors in scenarios.items():
        # Apply scenario factors
        scenario_demand = base_demand * factors.get('demand_factor', 1.0)
        scenario_lead_time = lead_time * factors.get('lead_time_factor', 1.0)
        
        # Run simulation
        sim_results = simulate_inventory_policy(
            demand_series=scenario_demand,
            lead_time=scenario_lead_time,
            initial_inventory=inventory_params['max_level'],
            reorder_point=inventory_params['reorder_point'],
            order_quantity=inventory_params['economic_order_quantity'],
            max_inventory=inventory_params['max_level']
        )
        
        results[scenario_name] = sim_results
    
    return results


def generate_inventory_recommendations(
    product_id: str,
    current_params: Dict[str, float],
    optimized_params: Dict[str, float],
    scenario_results: Dict[str, Dict]
) -> Dict:
    """
    Generate inventory recommendations based on optimization results.
    
    Args:
        product_id: Product identifier
        current_params: Current inventory parameters
        optimized_params: Optimized inventory parameters
        scenario_results: Results of scenario analysis
        
    Returns:
        Dictionary of recommendations
    """
    # Calculate improvement percentages
    improvements = {}
    for param in ['safety_stock', 'reorder_point', 'economic_order_quantity']:
        if param in current_params and current_params[param] > 0:
            pct_change = ((optimized_params[param] - current_params[param]) / 
                          current_params[param]) * 100
            improvements[param] = pct_change
    
    # Identify risk scenarios
    risk_scenarios = []
    base_metrics = scenario_results.get('base', {}).get('metrics', {})
    for scenario, results in scenario_results.items():
        if scenario == 'base':
            continue
        
        scenario_metrics = results.get('metrics', {})
        if (scenario_metrics.get('service_level', 0) < 
            base_metrics.get('service_level', 1) * 0.9):
            risk_scenarios.append({
                'scenario': scenario,
                'service_level': scenario_metrics.get('service_level', 0),
                'stockout_days': scenario_metrics.get('stockout_days', 0)
            })
    
    # Generate recommendations
    recommendations = {
        'product_id': product_id,
        'parameter_changes': {
            'safety_stock': {
                'current': current_params.get('safety_stock', 0),
                'recommended': optimized_params.get('safety_stock', 0),
                'change_pct': improvements.get('safety_stock', 0)
            },
            'reorder_point': {
                'current': current_params.get('reorder_point', 0),
                'recommended': optimized_params.get('reorder_point', 0),
                'change_pct': improvements.get('reorder_point', 0)
            },
            'min_level': {
                'current': current_params.get('min_level', 0),
                'recommended': optimized_params.get('min_level', 0),
                'change_pct': improvements.get('min_level', 0) if 'min_level' in current_params else None
            },
            'max_level': {
                'current': current_params.get('max_level', 0),
                'recommended': optimized_params.get('max_level', 0),
                'change_pct': improvements.get('max_level', 0) if 'max_level' in current_params else None
            },
            'order_quantity': {
                'current': current_params.get('economic_order_quantity', 0),
                'recommended': optimized_params.get('economic_order_quantity', 0),
                'change_pct': improvements.get('economic_order_quantity', 0)
            }
        },
        'risk_assessment': {
            'scenarios_analyzed': len(scenario_results),
            'risk_scenarios': risk_scenarios,
            'recommendation_confidence': calculate_recommendation_confidence(optimized_params, scenario_results)
        },
        'expected_outcomes': {
            'service_level': base_metrics.get('service_level', 0),
            'fill_rate': base_metrics.get('fill_rate', 0),
            'average_inventory': base_metrics.get('average_inventory', 0)
        }
    }
    
    return recommendations


def calculate_recommendation_confidence(
    optimized_params: Dict[str, float],
    scenario_results: Dict[str, Dict]
) -> float:
    """
    Calculate confidence level for inventory recommendations.
    
    Args:
        optimized_params: Optimized inventory parameters
        scenario_results: Results of scenario analysis
        
    Returns:
        Confidence score (0-1)
    """
    # Count scenarios where service level meets target
    target_service_level = 0.95
    scenarios_above_target = 0
    
    for scenario, results in scenario_results.items():
        metrics = results.get('metrics', {})
        if metrics.get('service_level', 0) >= target_service_level:
            scenarios_above_target += 1
    
    # Calculate confidence based on scenario performance
    if len(scenario_results) > 0:
        confidence = scenarios_above_target / len(scenario_results)
    else:
        confidence = 0.5  # Default if no scenarios
    
    return confidence 