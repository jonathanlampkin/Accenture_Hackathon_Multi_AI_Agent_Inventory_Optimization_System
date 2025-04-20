"""
Scenario Planning for Inventory Optimization

This module provides capabilities for what-if analysis and simulating different
inventory scenarios such as demand fluctuations, supply chain disruptions,
and price changes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import copy

logger = logging.getLogger(__name__)

class ScenarioPlanner:
    """
    Scenario planning for inventory optimization.
    
    Enables what-if analysis for different inventory and supply chain scenarios.
    """
    
    def __init__(self):
        """Initialize the scenario planner."""
        logger.info("Initialized ScenarioPlanner")
    
    def simulate_demand_scenario(self,
                                baseline_demand: pd.DataFrame,
                                scenario_adjustments: Dict[str, Any],
                                demand_col: str = 'demand',
                                date_col: str = 'date',
                                product_col: Optional[str] = None) -> pd.DataFrame:
        """
        Simulate a demand scenario based on baseline demand and adjustments.
        
        Args:
            baseline_demand: DataFrame with baseline demand data
            scenario_adjustments: Dictionary with scenario parameters:
                - 'increase_pct': Overall percentage increase (e.g., 0.1 for 10% increase)
                - 'decrease_pct': Overall percentage decrease (e.g., 0.1 for 10% decrease)
                - 'seasonal_factor': Seasonal adjustment factor (list or dict by date)
                - 'product_factors': Dict of product-specific factors (product_id -> factor)
                - 'date_range_factor': Dict with 'start', 'end', and 'factor' for date range adjustments
                - 'random_noise': Standard deviation of random noise to add
                - 'trend_factor': Monthly trend factor (e.g., 0.01 for 1% monthly increase)
            demand_col: Column name for demand values
            date_col: Column name for dates
            product_col: Column name for product IDs
            
        Returns:
            DataFrame with simulated demand data
        """
        # Make a copy of the baseline demand
        simulated_demand = baseline_demand.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(simulated_demand[date_col]):
            simulated_demand[date_col] = pd.to_datetime(simulated_demand[date_col])
        
        # Apply overall percentage increase/decrease
        if 'increase_pct' in scenario_adjustments:
            factor = 1 + scenario_adjustments['increase_pct']
            simulated_demand[demand_col] = simulated_demand[demand_col] * factor
            logger.info(f"Applied {scenario_adjustments['increase_pct']*100}% increase to all demand")
        
        if 'decrease_pct' in scenario_adjustments:
            factor = 1 - scenario_adjustments['decrease_pct']
            simulated_demand[demand_col] = simulated_demand[demand_col] * factor
            logger.info(f"Applied {scenario_adjustments['decrease_pct']*100}% decrease to all demand")
        
        # Apply seasonal factors
        if 'seasonal_factor' in scenario_adjustments:
            seasonal_factor = scenario_adjustments['seasonal_factor']
            
            if isinstance(seasonal_factor, dict):
                # Apply seasonal factors by date
                for date_str, factor in seasonal_factor.items():
                    date = pd.to_datetime(date_str)
                    simulated_demand.loc[simulated_demand[date_col] == date, demand_col] *= factor
            elif isinstance(seasonal_factor, list):
                # Apply seasonal factors cyclically
                for i, date_group in enumerate(simulated_demand.groupby(simulated_demand[date_col].dt.month)):
                    month, group = date_group
                    month_idx = month - 1  # 0-based index
                    factor = seasonal_factor[month_idx % len(seasonal_factor)]
                    simulated_demand.loc[group.index, demand_col] *= factor
            
            logger.info(f"Applied seasonal factors to demand")
        
        # Apply product-specific factors
        if 'product_factors' in scenario_adjustments and product_col:
            product_factors = scenario_adjustments['product_factors']
            
            for product_id, factor in product_factors.items():
                mask = simulated_demand[product_col] == product_id
                simulated_demand.loc[mask, demand_col] *= factor
            
            logger.info(f"Applied product-specific factors to {len(product_factors)} products")
        
        # Apply date range factors
        if 'date_range_factor' in scenario_adjustments:
            for range_adj in scenario_adjustments['date_range_factor']:
                start = pd.to_datetime(range_adj['start'])
                end = pd.to_datetime(range_adj['end'])
                factor = range_adj['factor']
                
                mask = (simulated_demand[date_col] >= start) & (simulated_demand[date_col] <= end)
                simulated_demand.loc[mask, demand_col] *= factor
            
            logger.info(f"Applied date range factors to demand")
        
        # Apply trend factor
        if 'trend_factor' in scenario_adjustments:
            trend_factor = scenario_adjustments['trend_factor']
            start_date = simulated_demand[date_col].min()
            
            # Calculate months since start for each date
            months_since_start = (simulated_demand[date_col].dt.year - start_date.year) * 12 + \
                                (simulated_demand[date_col].dt.month - start_date.month)
            
            # Apply trend factor based on number of months
            trend_adjustments = (1 + trend_factor) ** months_since_start
            simulated_demand[demand_col] *= trend_adjustments
            
            logger.info(f"Applied trend factor of {trend_factor*100}% per month to demand")
        
        # Add random noise
        if 'random_noise' in scenario_adjustments:
            noise_std = scenario_adjustments['random_noise']
            noise = np.random.normal(0, noise_std, size=len(simulated_demand))
            
            # Ensure demand doesn't go negative due to noise
            simulated_demand[demand_col] = np.maximum(0, simulated_demand[demand_col] * (1 + noise))
            
            logger.info(f"Added random noise with std={noise_std} to demand")
        
        # Round demand to integers if the original was integer
        if pd.api.types.is_integer_dtype(baseline_demand[demand_col]):
            simulated_demand[demand_col] = simulated_demand[demand_col].round().astype(int)
        
        return simulated_demand
    
    def simulate_supply_disruption(self,
                                 baseline_supply: pd.DataFrame,
                                 disruption_params: Dict[str, Any],
                                 supply_col: str = 'supply_quantity',
                                 date_col: str = 'date',
                                 supplier_col: Optional[str] = None) -> pd.DataFrame:
        """
        Simulate a supply disruption scenario.
        
        Args:
            baseline_supply: DataFrame with baseline supply data
            disruption_params: Dictionary with disruption parameters:
                - 'start_date': Start date of disruption
                - 'end_date': End date of disruption
                - 'impact_pct': Percentage reduction in supply (e.g., 0.5 for 50% reduction)
                - 'suppliers': List of affected suppliers (if None, affects all)
                - 'recovery_time': Days to fully recover after disruption ends
                - 'recovery_type': 'linear', 'exponential', or 'step'
            supply_col: Column name for supply quantity
            date_col: Column name for dates
            supplier_col: Column name for supplier IDs
            
        Returns:
            DataFrame with simulated supply data including disruption
        """
        # Make a copy of the baseline supply
        simulated_supply = baseline_supply.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(simulated_supply[date_col]):
            simulated_supply[date_col] = pd.to_datetime(simulated_supply[date_col])
        
        # Extract disruption parameters
        start_date = pd.to_datetime(disruption_params['start_date'])
        end_date = pd.to_datetime(disruption_params['end_date'])
        impact_pct = disruption_params['impact_pct']
        affected_suppliers = disruption_params.get('suppliers', None)
        recovery_time = disruption_params.get('recovery_time', 0)
        recovery_type = disruption_params.get('recovery_type', 'linear')
        
        # Create mask for dates within disruption period
        disruption_mask = (simulated_supply[date_col] >= start_date) & (simulated_supply[date_col] <= end_date)
        
        # Apply supplier filter if specified
        if affected_suppliers and supplier_col:
            supplier_mask = simulated_supply[supplier_col].isin(affected_suppliers)
            disruption_mask = disruption_mask & supplier_mask
        
        # Apply disruption impact
        disruption_factor = 1 - impact_pct
        simulated_supply.loc[disruption_mask, supply_col] *= disruption_factor
        
        # Apply recovery period if specified
        if recovery_time > 0:
            recovery_end_date = end_date + timedelta(days=recovery_time)
            recovery_mask = (simulated_supply[date_col] > end_date) & (simulated_supply[date_col] <= recovery_end_date)
            
            # Additional supplier filter for recovery period
            if affected_suppliers and supplier_col:
                recovery_mask = recovery_mask & supplier_mask
            
            # Calculate recovery factors based on recovery type
            if recovery_type == 'linear':
                # Calculate days into recovery period
                days_into_recovery = (simulated_supply.loc[recovery_mask, date_col] - end_date).dt.days
                max_days = (recovery_end_date - end_date).days
                
                # Linear recovery from disruption_factor to 1.0
                recovery_factors = disruption_factor + (1 - disruption_factor) * (days_into_recovery / max_days)
                simulated_supply.loc[recovery_mask, supply_col] *= recovery_factors.values
                
            elif recovery_type == 'exponential':
                # Calculate days into recovery period
                days_into_recovery = (simulated_supply.loc[recovery_mask, date_col] - end_date).dt.days
                max_days = (recovery_end_date - end_date).days
                
                # Exponential recovery (faster at the beginning)
                recovery_progress = days_into_recovery / max_days
                recovery_factors = disruption_factor + (1 - disruption_factor) * (1 - np.exp(-3 * recovery_progress))
                simulated_supply.loc[recovery_mask, supply_col] *= recovery_factors.values
                
            elif recovery_type == 'step':
                # Step recovery (equal increments)
                recovery_steps = 4  # Number of steps
                max_days = (recovery_end_date - end_date).days
                step_size = max_days / recovery_steps
                
                for step in range(recovery_steps):
                    step_start = end_date + timedelta(days=step * step_size)
                    step_end = end_date + timedelta(days=(step + 1) * step_size)
                    step_mask = (simulated_supply[date_col] > step_start) & (simulated_supply[date_col] <= step_end)
                    
                    if affected_suppliers and supplier_col:
                        step_mask = step_mask & supplier_mask
                    
                    step_factor = disruption_factor + (1 - disruption_factor) * ((step + 1) / recovery_steps)
                    simulated_supply.loc[step_mask, supply_col] *= step_factor
        
        logger.info(f"Applied supply disruption from {start_date} to {end_date} with {impact_pct*100}% reduction")
        if recovery_time > 0:
            logger.info(f"Applied {recovery_type} recovery over {recovery_time} days")
        
        return simulated_supply
    
    def simulate_price_change(self,
                            baseline_data: pd.DataFrame,
                            price_changes: Dict[str, Any],
                            demand_col: str = 'demand',
                            price_col: str = 'price',
                            date_col: str = 'date',
                            product_col: Optional[str] = None) -> pd.DataFrame:
        """
        Simulate the impact of price changes on demand.
        
        Args:
            baseline_data: DataFrame with baseline data including price and demand
            price_changes: Dictionary with price change parameters:
                - 'product_elasticities': Dict of product elasticities (product_id -> elasticity)
                - 'default_elasticity': Default price elasticity if not specified by product
                - 'price_changes': Dict of product price changes (product_id -> pct_change)
                - 'global_price_change': Overall percentage price change
                - 'date_effective': Date from which price changes take effect
            demand_col: Column name for demand values
            price_col: Column name for price values
            date_col: Column name for dates
            product_col: Column name for product IDs
            
        Returns:
            DataFrame with updated prices and demand based on elasticity
        """
        # Make a copy of the baseline data
        simulated_data = baseline_data.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(simulated_data[date_col]):
            simulated_data[date_col] = pd.to_datetime(simulated_data[date_col])
        
        # Get default elasticity
        default_elasticity = price_changes.get('default_elasticity', -1.0)  # Default is -1.0 (unit elasticity)
        
        # Get product-specific elasticities
        product_elasticities = price_changes.get('product_elasticities', {})
        
        # Get date from which price changes take effect
        date_effective = pd.to_datetime(price_changes.get('date_effective', simulated_data[date_col].min()))
        
        # Create mask for dates after effective date
        date_mask = simulated_data[date_col] >= date_effective
        
        # Apply global price change if specified
        if 'global_price_change' in price_changes:
            change_pct = price_changes['global_price_change']
            original_prices = simulated_data.loc[date_mask, price_col].copy()
            
            # Apply price change
            simulated_data.loc[date_mask, price_col] *= (1 + change_pct)
            
            # Calculate price ratio for elasticity calculation
            price_ratio = simulated_data.loc[date_mask, price_col] / original_prices
            
            # Apply elasticity to update demand
            if product_col:
                for product_id in simulated_data[product_col].unique():
                    # Get product-specific elasticity or default
                    elasticity = product_elasticities.get(product_id, default_elasticity)
                    
                    # Create mask for this product and after effective date
                    product_mask = (simulated_data[product_col] == product_id) & date_mask
                    
                    # Calculate demand adjustment
                    price_ratio_product = price_ratio[product_mask]
                    demand_factor = price_ratio_product ** elasticity
                    
                    # Apply demand adjustment
                    simulated_data.loc[product_mask, demand_col] *= demand_factor.values
            else:
                # Apply default elasticity to all products
                demand_factor = price_ratio ** default_elasticity
                simulated_data.loc[date_mask, demand_col] *= demand_factor.values
            
            logger.info(f"Applied global price change of {change_pct*100}% effective {date_effective}")
        
        # Apply product-specific price changes
        if 'price_changes' in price_changes and product_col:
            for product_id, change_pct in price_changes['price_changes'].items():
                # Create mask for this product and after effective date
                product_mask = (simulated_data[product_col] == product_id) & date_mask
                
                if not product_mask.any():
                    continue
                
                # Get original price
                original_price = simulated_data.loc[product_mask, price_col].iloc[0]
                
                # Apply price change
                simulated_data.loc[product_mask, price_col] *= (1 + change_pct)
                
                # Calculate price ratio
                new_price = simulated_data.loc[product_mask, price_col].iloc[0]
                price_ratio = new_price / original_price
                
                # Get product-specific elasticity or default
                elasticity = product_elasticities.get(product_id, default_elasticity)
                
                # Apply elasticity to update demand
                demand_factor = price_ratio ** elasticity
                simulated_data.loc[product_mask, demand_col] *= demand_factor
                
                logger.info(f"Applied price change of {change_pct*100}% to product {product_id} (elasticity={elasticity})")
        
        # Round demand to integers if the original was integer
        if pd.api.types.is_integer_dtype(baseline_data[demand_col]):
            simulated_data[demand_col] = simulated_data[demand_col].round().astype(int)
        
        return simulated_data
    
    def simulate_lead_time_changes(self,
                                 baseline_data: pd.DataFrame,
                                 lead_time_changes: Dict[str, Any],
                                 lead_time_col: str = 'lead_time',
                                 date_col: str = 'date',
                                 supplier_col: Optional[str] = None) -> pd.DataFrame:
        """
        Simulate changes in supplier lead times.
        
        Args:
            baseline_data: DataFrame with baseline data including lead times
            lead_time_changes: Dictionary with lead time change parameters:
                - 'global_increase_days': Days to add to all lead times
                - 'global_increase_pct': Percentage increase for all lead times
                - 'supplier_changes': Dict of supplier-specific changes (supplier_id -> days_change)
                - 'date_effective': Date from which lead time changes take effect
                - 'variability_increase': Increase in lead time variability (std deviation)
            lead_time_col: Column name for lead time values
            date_col: Column name for dates
            supplier_col: Column name for supplier IDs
            
        Returns:
            DataFrame with updated lead times
        """
        # Make a copy of the baseline data
        simulated_data = baseline_data.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(simulated_data[date_col]):
            simulated_data[date_col] = pd.to_datetime(simulated_data[date_col])
        
        # Get date from which lead time changes take effect
        date_effective = pd.to_datetime(lead_time_changes.get('date_effective', simulated_data[date_col].min()))
        
        # Create mask for dates after effective date
        date_mask = simulated_data[date_col] >= date_effective
        
        # Apply global lead time increase (days) if specified
        if 'global_increase_days' in lead_time_changes:
            days_increase = lead_time_changes['global_increase_days']
            simulated_data.loc[date_mask, lead_time_col] += days_increase
            logger.info(f"Applied global lead time increase of {days_increase} days effective {date_effective}")
        
        # Apply global lead time increase (percentage) if specified
        if 'global_increase_pct' in lead_time_changes:
            pct_increase = lead_time_changes['global_increase_pct']
            simulated_data.loc[date_mask, lead_time_col] *= (1 + pct_increase)
            logger.info(f"Applied global lead time increase of {pct_increase*100}% effective {date_effective}")
        
        # Apply supplier-specific lead time changes
        if 'supplier_changes' in lead_time_changes and supplier_col:
            for supplier_id, days_change in lead_time_changes['supplier_changes'].items():
                # Create mask for this supplier and after effective date
                supplier_mask = (simulated_data[supplier_col] == supplier_id) & date_mask
                
                # Apply lead time change
                simulated_data.loc[supplier_mask, lead_time_col] += days_change
                
                logger.info(f"Applied lead time change of {days_change} days to supplier {supplier_id}")
        
        # Apply lead time variability increase if specified
        if 'variability_increase' in lead_time_changes:
            var_increase = lead_time_changes['variability_increase']
            
            # Add random variability to lead times
            np.random.seed(42)  # For reproducibility
            variability = np.random.normal(0, var_increase, size=sum(date_mask))
            
            # Ensure lead times don't go negative due to variability
            simulated_data.loc[date_mask, lead_time_col] = np.maximum(
                1, simulated_data.loc[date_mask, lead_time_col] + variability)
            
            logger.info(f"Applied lead time variability increase of std={var_increase}")
        
        # Round lead times to integers
        simulated_data[lead_time_col] = simulated_data[lead_time_col].round().astype(int)
        
        return simulated_data
    
    def run_what_if_scenario(self,
                           baseline_data: pd.DataFrame,
                           scenario_params: Dict[str, Any],
                           inventory_model: Callable,
                           metrics_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run a comprehensive what-if scenario and evaluate its impact.
        
        Args:
            baseline_data: DataFrame with baseline data
            scenario_params: Dictionary with comprehensive scenario parameters
            inventory_model: Function to calculate inventory policies based on scenarios
            metrics_func: Optional function to calculate performance metrics
            
        Returns:
            Dictionary with scenario results
        """
        simulated_data = baseline_data.copy()
        scenario_description = scenario_params.get('description', 'Unnamed scenario')
        
        logger.info(f"Running what-if scenario: {scenario_description}")
        
        # Apply demand changes if specified
        if 'demand_scenario' in scenario_params:
            demand_params = scenario_params['demand_scenario']
            demand_col = demand_params.get('demand_col', 'demand')
            date_col = demand_params.get('date_col', 'date')
            product_col = demand_params.get('product_col', None)
            
            simulated_data = self.simulate_demand_scenario(
                simulated_data, 
                demand_params,
                demand_col=demand_col,
                date_col=date_col,
                product_col=product_col
            )
        
        # Apply supply disruptions if specified
        if 'supply_disruption' in scenario_params:
            supply_params = scenario_params['supply_disruption']
            supply_col = supply_params.get('supply_col', 'supply_quantity')
            date_col = supply_params.get('date_col', 'date')
            supplier_col = supply_params.get('supplier_col', None)
            
            simulated_data = self.simulate_supply_disruption(
                simulated_data,
                supply_params,
                supply_col=supply_col,
                date_col=date_col,
                supplier_col=supplier_col
            )
        
        # Apply price changes if specified
        if 'price_changes' in scenario_params:
            price_params = scenario_params['price_changes']
            demand_col = price_params.get('demand_col', 'demand')
            price_col = price_params.get('price_col', 'price')
            date_col = price_params.get('date_col', 'date')
            product_col = price_params.get('product_col', None)
            
            simulated_data = self.simulate_price_change(
                simulated_data,
                price_params,
                demand_col=demand_col,
                price_col=price_col,
                date_col=date_col,
                product_col=product_col
            )
        
        # Apply lead time changes if specified
        if 'lead_time_changes' in scenario_params:
            lead_time_params = scenario_params['lead_time_changes']
            lead_time_col = lead_time_params.get('lead_time_col', 'lead_time')
            date_col = lead_time_params.get('date_col', 'date')
            supplier_col = lead_time_params.get('supplier_col', None)
            
            simulated_data = self.simulate_lead_time_changes(
                simulated_data,
                lead_time_params,
                lead_time_col=lead_time_col,
                date_col=date_col,
                supplier_col=supplier_col
            )
        
        # Calculate inventory policies based on simulated data
        inventory_policies = inventory_model(simulated_data)
        
        # Calculate performance metrics if function provided
        performance_metrics = None
        if metrics_func:
            performance_metrics = metrics_func(simulated_data, inventory_policies)
        
        # Prepare results
        results = {
            'scenario_name': scenario_description,
            'simulated_data': simulated_data,
            'inventory_policies': inventory_policies,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now().isoformat(),
            'scenario_params': scenario_params
        }
        
        logger.info(f"Completed what-if scenario: {scenario_description}")
        
        return results
    
    def compare_scenarios(self,
                        scenario_results: List[Dict[str, Any]],
                        metrics_of_interest: List[str],
                        baseline_idx: int = 0) -> pd.DataFrame:
        """
        Compare multiple scenario results.
        
        Args:
            scenario_results: List of scenario result dictionaries
            metrics_of_interest: List of metric names to compare
            baseline_idx: Index of the baseline scenario for comparison
            
        Returns:
            DataFrame with scenario comparison
        """
        if not scenario_results:
            logger.warning("No scenarios to compare")
            return pd.DataFrame()
        
        # Extract scenario names
        scenario_names = [result.get('scenario_name', f"Scenario {i}") for i, result in enumerate(scenario_results)]
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Get baseline metrics for comparison
        baseline_metrics = scenario_results[baseline_idx].get('performance_metrics', {})
        
        for i, result in enumerate(scenario_results):
            metrics = result.get('performance_metrics', {})
            row = {'scenario': scenario_names[i]}
            
            for metric in metrics_of_interest:
                if metric in metrics:
                    row[metric] = metrics[metric]
                    
                    # Calculate difference from baseline
                    if i != baseline_idx and metric in baseline_metrics:
                        baseline_value = baseline_metrics[metric]
                        if baseline_value != 0:
                            pct_diff = ((metrics[metric] - baseline_value) / baseline_value) * 100
                            row[f"{metric}_pct_diff"] = pct_diff
                        else:
                            row[f"{metric}_pct_diff"] = np.nan
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info(f"Generated comparison of {len(scenario_results)} scenarios")
        
        return comparison_df 