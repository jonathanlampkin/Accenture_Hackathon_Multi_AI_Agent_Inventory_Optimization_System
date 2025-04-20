"""
Tools for scenario planning operations used by the scenario planning agent.

This module contains tools for simulating demand scenarios, evaluating
inventory policies under different scenarios, and cost-benefit analysis.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import seaborn as sns


class SimulateDemandScenariosTool(BaseTool):
    """Tool for simulating various demand scenarios."""
    
    name: str = "Simulate Demand Scenarios"
    description: str = """
    Simulate different demand scenarios based on historical data and specified parameters.
    
    Input should include:
    - demand_data_path: Path to the historical demand data CSV file
    - scenario_params: Dict with scenario parameters (e.g., growth_rate, volatility, seasonality)
    - num_simulations: Number of simulations to run for each scenario
    - horizon: Number of days to forecast
    - product_ids: Optional list of product IDs to simulate (None for all products)
    - output_path: Path to save the simulation results
    """
    
    class InputSchema(BaseModel):
        demand_data_path: str = Field(
            ..., 
            description="Path to the historical demand data CSV file"
        )
        scenario_params: Dict[str, Dict[str, Union[float, bool, str]]] = Field(
            ..., 
            description="Dict with scenario parameters (e.g., growth_rate, volatility, seasonality)"
        )
        num_simulations: int = Field(
            10, 
            description="Number of simulations to run for each scenario"
        )
        horizon: int = Field(
            90, 
            description="Number of days to forecast"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to simulate (None for all products)"
        )
        output_path: str = Field(
            "output/scenario_analysis/demand_scenarios.csv", 
            description="Path to save the simulation results"
        )
    
    def run(self, demand_data_path: str,
            scenario_params: Dict[str, Dict[str, Union[float, bool, str]]],
            num_simulations: int = 10,
            horizon: int = 90,
            product_ids: Optional[List[int]] = None,
            output_path: str = "output/scenario_analysis/demand_scenarios.csv") -> Dict[str, Any]:
        """
        Simulate different demand scenarios based on historical data and specified parameters.
        
        Args:
            demand_data_path: Path to the historical demand data CSV file
            scenario_params: Dict with scenario parameters (e.g., growth_rate, volatility, seasonality)
            num_simulations: Number of simulations to run for each scenario
            horizon: Number of days to forecast
            product_ids: Optional list of product IDs to simulate
            output_path: Path to save the simulation results
            
        Returns:
            Dict containing simulation results
        """
        try:
            # Load demand data
            demand_data = pd.read_csv(demand_data_path)
            
            # Convert date to datetime
            demand_data['Date'] = pd.to_datetime(demand_data['Date'])
            
            # Filter for specific products if provided
            if product_ids:
                demand_data = demand_data[demand_data['Product ID'].isin(product_ids)]
            
            # Group by product and date
            daily_demand = demand_data.groupby(['Product ID', 'Date'])['Sales Quantity'].sum().reset_index()
            
            # Get unique product IDs
            unique_products = daily_demand['Product ID'].unique()
            
            # Calculate daily demand statistics for each product
            product_stats = {}
            for product_id in unique_products:
                product_data = daily_demand[daily_demand['Product ID'] == product_id]
                product_stats[product_id] = {
                    'mean': product_data['Sales Quantity'].mean(),
                    'std': product_data['Sales Quantity'].std(),
                    'min': product_data['Sales Quantity'].min(),
                    'max': product_data['Sales Quantity'].max(),
                    'last_date': product_data['Date'].max()
                }
            
            # Generate future dates
            last_date = demand_data['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
            
            # Prepare results dataframe
            results = []
            
            # Simulate for each scenario and product
            for scenario_name, params in scenario_params.items():
                for product_id in unique_products:
                    stats = product_stats[product_id]
                    
                    # Extract parameters with defaults
                    growth_rate = params.get('growth_rate', 0.0)  # Daily growth rate
                    volatility = params.get('volatility', 1.0)    # Multiplier for standard deviation
                    seasonality = params.get('seasonality', False)
                    seasonality_factor = params.get('seasonality_factor', 0.2)
                    promotion_probability = params.get('promotion_probability', 0.0)
                    promotion_impact = params.get('promotion_impact', 1.5)
                    shock_probability = params.get('shock_probability', 0.0)
                    shock_impact = params.get('shock_impact', 0.5)
                    
                    # Run multiple simulations for this scenario and product
                    for sim_num in range(num_simulations):
                        for i, date in enumerate(future_dates):
                            # Base forecast starts with historical mean
                            base_forecast = stats['mean']
                            
                            # Apply growth trend
                            trend_factor = (1 + growth_rate) ** (i+1)
                            
                            # Apply seasonality if enabled
                            season_factor = 1.0
                            if seasonality:
                                # Simple sine wave seasonality
                                day_of_year = date.dayofyear
                                season_factor = 1 + seasonality_factor * np.sin(2 * np.pi * day_of_year / 365)
                            
                            # Apply promotion effect (random)
                            promo_factor = 1.0
                            is_promotion = np.random.random() < promotion_probability
                            if is_promotion:
                                promo_factor = promotion_impact
                            
                            # Apply shock effect (random)
                            shock_factor = 1.0
                            is_shock = np.random.random() < shock_probability
                            if is_shock:
                                shock_factor = shock_impact
                            
                            # Calculate mean forecast with all factors
                            mean_forecast = base_forecast * trend_factor * season_factor * promo_factor * shock_factor
                            
                            # Apply volatility to standard deviation
                            std_forecast = stats['std'] * volatility
                            
                            # Generate random demand from normal distribution
                            # Clip to ensure non-negative values
                            random_demand = np.random.normal(mean_forecast, std_forecast)
                            simulated_demand = max(0, round(random_demand))
                            
                            # Append to results
                            results.append({
                                'Scenario': scenario_name,
                                'Simulation': sim_num + 1,
                                'Product ID': product_id,
                                'Date': date,
                                'Simulated Demand': simulated_demand,
                                'Is Promotion': is_promotion,
                                'Is Shock': is_shock
                            })
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            
            # Create summary by scenario and product
            summary = results_df.groupby(['Scenario', 'Product ID'])['Simulated Demand'].agg(
                ['mean', 'std', 'min', 'max']
            ).reset_index()
            
            # Save summary
            summary_path = os.path.join(os.path.dirname(output_path), 'demand_scenarios_summary.csv')
            summary.to_csv(summary_path, index=False)
            
            # Create visualizations
            vis_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Plot average demand by scenario for each product
            for product_id in unique_products:
                product_data = results_df[results_df['Product ID'] == product_id]
                pivot_data = product_data.pivot_table(
                    index='Date', 
                    columns='Scenario', 
                    values='Simulated Demand', 
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(12, 6))
                pivot_data.plot(kind='line')
                plt.title(f'Average Simulated Demand for Product {product_id} by Scenario')
                plt.ylabel('Demand')
                plt.grid(True, alpha=0.3)
                
                fig_path = os.path.join(vis_dir, f'product_{product_id}_scenario_comparison.png')
                plt.savefig(fig_path)
                plt.close()
            
            # Prepare results dictionary
            return {
                "message": f"Simulated {len(scenario_params)} scenarios for {len(unique_products)} products over {horizon} days",
                "summary": summary.to_dict(orient='records'),
                "output_path": output_path,
                "summary_path": summary_path,
                "visualization_directory": vis_dir
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to simulate demand scenarios"
            }


class EvaluateInventoryPolicyTool(BaseTool):
    """Tool for evaluating inventory policies under different scenarios."""
    
    name: str = "Evaluate Inventory Policy"
    description: str = """
    Evaluate the performance of inventory policies under different demand scenarios.
    
    Input should include:
    - inventory_policy_path: Path to the inventory policy definitions
    - scenario_data_path: Path to the simulated demand scenarios
    - initial_inventory: Dict mapping product IDs to initial inventory levels
    - cost_data_path: Path to the cost data (holding cost, stockout cost, etc.)
    - product_ids: Optional list of product IDs to evaluate (None for all products)
    - output_path: Path to save the evaluation results
    """
    
    class InputSchema(BaseModel):
        inventory_policy_path: str = Field(
            ..., 
            description="Path to the inventory policy definitions"
        )
        scenario_data_path: str = Field(
            ..., 
            description="Path to the simulated demand scenarios"
        )
        initial_inventory: Dict[int, int] = Field(
            ..., 
            description="Dict mapping product IDs to initial inventory levels"
        )
        cost_data_path: str = Field(
            ..., 
            description="Path to the cost data (holding cost, stockout cost, etc.)"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to evaluate (None for all products)"
        )
        output_path: str = Field(
            "output/scenario_analysis/policy_evaluation.csv", 
            description="Path to save the evaluation results"
        )
    
    def run(self, inventory_policy_path: str,
            scenario_data_path: str,
            initial_inventory: Dict[int, int],
            cost_data_path: str,
            product_ids: Optional[List[int]] = None,
            output_path: str = "output/scenario_analysis/policy_evaluation.csv") -> Dict[str, Any]:
        """
        Evaluate the performance of inventory policies under different demand scenarios.
        
        Args:
            inventory_policy_path: Path to the inventory policy definitions
            scenario_data_path: Path to the simulated demand scenarios
            initial_inventory: Dict mapping product IDs to initial inventory levels
            cost_data_path: Path to the cost data
            product_ids: Optional list of product IDs to evaluate
            output_path: Path to save the evaluation results
            
        Returns:
            Dict containing evaluation results
        """
        try:
            # Load inventory policies
            inventory_policies = pd.read_csv(inventory_policy_path)
            
            # Load scenario data
            scenario_data = pd.read_csv(scenario_data_path)
            scenario_data['Date'] = pd.to_datetime(scenario_data['Date'])
            
            # Load cost data
            cost_data = pd.read_csv(cost_data_path)
            
            # Filter for specific products if provided
            if product_ids:
                inventory_policies = inventory_policies[inventory_policies['Product ID'].isin(product_ids)]
                scenario_data = scenario_data[scenario_data['Product ID'].isin(product_ids)]
                # Also filter initial inventory to only include specified products
                initial_inventory = {k: v for k, v in initial_inventory.items() if k in product_ids}
            
            # Get all unique scenarios
            scenarios = scenario_data['Scenario'].unique()
            
            # Get all unique simulation numbers
            simulations = scenario_data['Simulation'].unique()
            
            # Get unique product IDs
            products = inventory_policies['Product ID'].unique()
            
            # Prepare result container
            results = []
            
            # For each product, scenario, and simulation number
            for product_id in products:
                # Get product-specific policy
                policy = inventory_policies[inventory_policies['Product ID'] == product_id].iloc[0]
                
                # Get product-specific cost data
                product_costs = cost_data[cost_data['Product ID'] == product_id].iloc[0]
                
                for scenario in scenarios:
                    for sim_num in simulations:
                        # Get this specific simulation run
                        sim_data = scenario_data[
                            (scenario_data['Product ID'] == product_id) &
                            (scenario_data['Scenario'] == scenario) &
                            (scenario_data['Simulation'] == sim_num)
                        ].sort_values('Date')
                        
                        if sim_data.empty:
                            continue
                        
                        # Initialize tracking variables
                        inventory_level = initial_inventory.get(product_id, 0)
                        order_quantity = policy['EOQ']
                        reorder_point = policy['Reorder Point']
                        
                        total_demand = 0
                        total_sales = 0
                        total_stockouts = 0
                        stockout_days = 0
                        total_holding_cost = 0
                        total_ordering_cost = 0
                        total_stockout_cost = 0
                        orders_placed = 0
                        order_lead_time = product_costs['Reorder Lead Time']
                        
                        # Create a pending orders queue
                        pending_orders = []
                        
                        # Get costs
                        holding_cost_rate = product_costs['Holding Cost']  # per unit per day
                        ordering_cost = product_costs['Ordering Cost']     # per order
                        stockout_cost = product_costs['Stockout Cost']     # per unit
                        
                        # Daily inventory simulation
                        daily_records = []
                        
                        for _, row in sim_data.iterrows():
                            date = row['Date']
                            demand = row['Simulated Demand']
                            
                            # Check for order arrivals
                            arrived_quantity = 0
                            for i, (arrival_date, quantity) in enumerate(pending_orders):
                                if date >= arrival_date:
                                    arrived_quantity += quantity
                                    pending_orders.pop(i)
                            
                            inventory_level += arrived_quantity
                            
                            # Process demand
                            total_demand += demand
                            
                            if inventory_level >= demand:
                                sales = demand
                                stockouts = 0
                            else:
                                sales = inventory_level
                                stockouts = demand - inventory_level
                                stockout_days += 1
                            
                            total_sales += sales
                            total_stockouts += stockouts
                            
                            # Update inventory
                            inventory_level -= sales
                            
                            # Calculate daily holding cost
                            daily_holding_cost = inventory_level * holding_cost_rate / 365  # daily rate
                            total_holding_cost += daily_holding_cost
                            
                            # Calculate stockout cost
                            daily_stockout_cost = stockouts * stockout_cost
                            total_stockout_cost += daily_stockout_cost
                            
                            # Check if we need to place an order (continuous review policy)
                            if inventory_level <= reorder_point and not pending_orders:
                                # Place an order
                                pending_orders.append((date + pd.Timedelta(days=order_lead_time), order_quantity))
                                orders_placed += 1
                                total_ordering_cost += ordering_cost
                            
                            # Record daily status
                            daily_records.append({
                                'Date': date,
                                'Demand': demand,
                                'Sales': sales,
                                'Stockouts': stockouts,
                                'Inventory Level': inventory_level,
                                'Order Placed': 1 if (inventory_level <= reorder_point and not pending_orders) else 0,
                                'Order Arrived': arrived_quantity
                            })
                        
                        # Calculate service level
                        service_level = total_sales / total_demand if total_demand > 0 else 1.0
                        
                        # Calculate total cost
                        total_cost = total_holding_cost + total_ordering_cost + total_stockout_cost
                        
                        # Calculate average inventory
                        daily_inventory = [record['Inventory Level'] for record in daily_records]
                        avg_inventory = sum(daily_inventory) / len(daily_inventory) if daily_inventory else 0
                        
                        # Calculate inventory turns
                        inventory_turns = total_sales / avg_inventory if avg_inventory > 0 else 0
                        
                        # Append to results
                        results.append({
                            'Product ID': product_id,
                            'Scenario': scenario,
                            'Simulation': sim_num,
                            'Service Level': service_level,
                            'Total Demand': total_demand,
                            'Total Sales': total_sales,
                            'Total Stockouts': total_stockouts,
                            'Stockout Days': stockout_days,
                            'Orders Placed': orders_placed,
                            'Average Inventory': avg_inventory,
                            'Holding Cost': total_holding_cost,
                            'Ordering Cost': total_ordering_cost,
                            'Stockout Cost': total_stockout_cost,
                            'Total Cost': total_cost,
                            'Inventory Turns': inventory_turns
                        })
                        
                        # Save daily records for this simulation
                        daily_df = pd.DataFrame(daily_records)
                        daily_dir = os.path.join(os.path.dirname(output_path), 'daily_records')
                        os.makedirs(daily_dir, exist_ok=True)
                        daily_file = f"product_{product_id}_scenario_{scenario}_sim_{sim_num}_daily.csv"
                        daily_df.to_csv(os.path.join(daily_dir, daily_file), index=False)
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            
            # Create summary by scenario and product
            summary = results_df.groupby(['Scenario', 'Product ID']).agg({
                'Service Level': 'mean',
                'Total Demand': 'mean',
                'Total Sales': 'mean',
                'Total Stockouts': 'mean',
                'Stockout Days': 'mean',
                'Orders Placed': 'mean',
                'Average Inventory': 'mean',
                'Holding Cost': 'mean',
                'Ordering Cost': 'mean',
                'Stockout Cost': 'mean',
                'Total Cost': 'mean',
                'Inventory Turns': 'mean'
            }).reset_index()
            
            # Save summary
            summary_path = os.path.join(output_dir, 'policy_evaluation_summary.csv')
            summary.to_csv(summary_path, index=False)
            
            # Create visualizations
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Plot service level and costs by scenario for each product
            for product_id in products:
                product_summary = summary[summary['Product ID'] == product_id]
                
                # Service level comparison
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Scenario', y='Service Level', data=product_summary)
                plt.title(f'Average Service Level for Product {product_id} by Scenario')
                plt.ylabel('Service Level')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig_path = os.path.join(vis_dir, f'product_{product_id}_service_level.png')
                plt.savefig(fig_path)
                plt.close()
                
                # Cost comparison
                cost_data = product_summary.melt(
                    id_vars=['Scenario', 'Product ID'],
                    value_vars=['Holding Cost', 'Ordering Cost', 'Stockout Cost'],
                    var_name='Cost Type', value_name='Cost'
                )
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Scenario', y='Cost', hue='Cost Type', data=cost_data)
                plt.title(f'Cost Breakdown for Product {product_id} by Scenario')
                plt.ylabel('Cost')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig_path = os.path.join(vis_dir, f'product_{product_id}_cost_breakdown.png')
                plt.savefig(fig_path)
                plt.close()
            
            # Prepare results dictionary
            return {
                "message": f"Evaluated inventory policies for {len(products)} products across {len(scenarios)} scenarios",
                "summary": summary.to_dict(orient='records'),
                "output_path": output_path,
                "summary_path": summary_path,
                "visualization_directory": vis_dir
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to evaluate inventory policies"
            }


class PerformCostBenefitAnalysisTool(BaseTool):
    """Tool for performing cost-benefit analysis of inventory policies."""
    
    name: str = "Perform Cost-Benefit Analysis"
    description: str = """
    Perform a cost-benefit analysis of different inventory policies across scenarios.
    
    Input should include:
    - policy_evaluation_path: Path to the policy evaluation results
    - cost_data_path: Path to the cost data
    - revenue_data_path: Path to the revenue data
    - product_ids: Optional list of product IDs to analyze (None for all products)
    - output_path: Path to save the cost-benefit analysis results
    """
    
    class InputSchema(BaseModel):
        policy_evaluation_path: str = Field(
            ..., 
            description="Path to the policy evaluation results"
        )
        cost_data_path: str = Field(
            ..., 
            description="Path to the cost data"
        )
        revenue_data_path: str = Field(
            ..., 
            description="Path to the revenue data"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to analyze (None for all products)"
        )
        output_path: str = Field(
            "output/scenario_analysis/cost_benefit_analysis.csv", 
            description="Path to save the cost-benefit analysis results"
        )
    
    def run(self, policy_evaluation_path: str,
            cost_data_path: str,
            revenue_data_path: str,
            product_ids: Optional[List[int]] = None,
            output_path: str = "output/scenario_analysis/cost_benefit_analysis.csv") -> Dict[str, Any]:
        """
        Perform a cost-benefit analysis of different inventory policies across scenarios.
        
        Args:
            policy_evaluation_path: Path to the policy evaluation results
            cost_data_path: Path to the cost data
            revenue_data_path: Path to the revenue data
            product_ids: Optional list of product IDs to analyze
            output_path: Path to save the cost-benefit analysis results
            
        Returns:
            Dict containing cost-benefit analysis results
        """
        try:
            # Load policy evaluation results
            eval_data = pd.read_csv(policy_evaluation_path)
            
            # Load cost data
            cost_data = pd.read_csv(cost_data_path)
            
            # Load revenue data
            revenue_data = pd.read_csv(revenue_data_path)
            
            # Filter for specific products if provided
            if product_ids:
                eval_data = eval_data[eval_data['Product ID'].isin(product_ids)]
                cost_data = cost_data[cost_data['Product ID'].isin(product_ids)]
                revenue_data = revenue_data[revenue_data['Product ID'].isin(product_ids)]
            
            # Merge revenue data with evaluation data
            results = pd.merge(eval_data, revenue_data[['Product ID', 'Unit Price']], on='Product ID', how='left')
            
            # Calculate revenue
            results['Revenue'] = results['Total Sales'] * results['Unit Price']
            
            # Calculate profit
            results['Profit'] = results['Revenue'] - results['Total Cost']
            
            # Calculate ROI
            results['ROI'] = results['Profit'] / results['Total Cost']
            
            # Calculate profit margin
            results['Profit Margin'] = results['Profit'] / results['Revenue']
            
            # Calculate opportunity cost of stockouts
            results['Lost Revenue'] = results['Total Stockouts'] * results['Unit Price']
            
            # Calculate net profit
            results['Net Profit'] = results['Profit']
            
            # Save results
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            results.to_csv(output_path, index=False)
            
            # Create summary by scenario and product
            summary = results.groupby(['Scenario', 'Product ID']).agg({
                'Revenue': 'mean',
                'Total Cost': 'mean',
                'Profit': 'mean',
                'ROI': 'mean',
                'Profit Margin': 'mean',
                'Lost Revenue': 'mean',
                'Service Level': 'mean'
            }).reset_index()
            
            # Save summary
            summary_path = os.path.join(output_dir, 'cost_benefit_summary.csv')
            summary.to_csv(summary_path, index=False)
            
            # Create visualizations
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Plot profit and ROI by scenario for each product
            for product_id in summary['Product ID'].unique():
                product_summary = summary[summary['Product ID'] == product_id]
                
                # Profit comparison
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Scenario', y='Profit', data=product_summary)
                plt.title(f'Average Profit for Product {product_id} by Scenario')
                plt.ylabel('Profit')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig_path = os.path.join(vis_dir, f'product_{product_id}_profit.png')
                plt.savefig(fig_path)
                plt.close()
                
                # ROI comparison
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Scenario', y='ROI', data=product_summary)
                plt.title(f'Average ROI for Product {product_id} by Scenario')
                plt.ylabel('ROI')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig_path = os.path.join(vis_dir, f'product_{product_id}_roi.png')
                plt.savefig(fig_path)
                plt.close()
                
                # Service Level vs Profit Margin
                plt.figure(figsize=(10, 6))
                plt.scatter(product_summary['Service Level'], product_summary['Profit Margin'])
                for i, row in product_summary.iterrows():
                    plt.annotate(row['Scenario'], 
                                 (row['Service Level'], row['Profit Margin']),
                                 textcoords="offset points", 
                                 xytext=(0,10), 
                                 ha='center')
                plt.title(f'Service Level vs Profit Margin for Product {product_id}')
                plt.xlabel('Service Level')
                plt.ylabel('Profit Margin')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig_path = os.path.join(vis_dir, f'product_{product_id}_service_vs_profit.png')
                plt.savefig(fig_path)
                plt.close()
            
            # Prepare top-level comparison across all scenarios and products
            overall_summary = summary.groupby('Scenario').agg({
                'Revenue': 'sum',
                'Total Cost': 'sum',
                'Profit': 'sum',
                'ROI': 'mean',
                'Profit Margin': 'mean',
                'Lost Revenue': 'sum',
                'Service Level': 'mean'
            }).reset_index()
            
            # Save overall summary
            overall_path = os.path.join(output_dir, 'overall_cost_benefit_summary.csv')
            overall_summary.to_csv(overall_path, index=False)
            
            # Create overall visualization
            plt.figure(figsize=(12, 8))
            
            # Create a subplot for the overall profit by scenario
            plt.subplot(2, 2, 1)
            sns.barplot(x='Scenario', y='Profit', data=overall_summary)
            plt.title('Total Profit by Scenario')
            plt.ylabel('Profit')
            plt.xticks(rotation=45)
            
            # Create a subplot for the overall ROI by scenario
            plt.subplot(2, 2, 2)
            sns.barplot(x='Scenario', y='ROI', data=overall_summary)
            plt.title('Average ROI by Scenario')
            plt.ylabel('ROI')
            plt.xticks(rotation=45)
            
            # Create a subplot for the overall service level by scenario
            plt.subplot(2, 2, 3)
            sns.barplot(x='Scenario', y='Service Level', data=overall_summary)
            plt.title('Average Service Level by Scenario')
            plt.ylabel('Service Level')
            plt.xticks(rotation=45)
            
            # Create a subplot for the overall profit margin by scenario
            plt.subplot(2, 2, 4)
            sns.barplot(x='Scenario', y='Profit Margin', data=overall_summary)
            plt.title('Average Profit Margin by Scenario')
            plt.ylabel('Profit Margin')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            fig_path = os.path.join(vis_dir, 'overall_scenario_comparison.png')
            plt.savefig(fig_path)
            plt.close()
            
            # Prepare results dictionary
            return {
                "message": f"Performed cost-benefit analysis for {len(summary['Product ID'].unique())} products across {len(summary['Scenario'].unique())} scenarios",
                "summary": summary.to_dict(orient='records'),
                "overall_summary": overall_summary.to_dict(orient='records'),
                "output_path": output_path,
                "summary_path": summary_path,
                "overall_summary_path": overall_path,
                "visualization_directory": vis_dir
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to perform cost-benefit analysis"
            } 