"""
Inventory Management Agent for the Multi-Agent Inventory Optimization System.
This agent is responsible for analyzing inventory data and making recommendations
for optimal stock levels, reorder points, and safety stock to minimize costs
while avoiding stockouts.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.agents.base_agent import BaseAgent
from src.utils.data_loader import load_inventory_data, load_demand_data, get_product_data, get_store_data
from src.utils.visualizer import plot_inventory_levels, plot_stockout_risk, generate_inventory_dashboard

class InventoryAgent(BaseAgent):
    """
    Agent responsible for inventory management and optimization.
    """
    
    def __init__(self, optimization_weights=None, product_id=None, store_id=None):
        """
        Initialize the Inventory Management Agent.
        
        Args:
            optimization_weights: Optional dictionary with weights for optimization objectives
            product_id: Optional specific product ID to focus on
            store_id: Optional specific store ID to focus on
        """
        super().__init__("InventoryAgent")
        
        # Initialize agent-specific state
        self.state.update({
            'inventory_data': None,
            'demand_data': None,
            'inventory_metrics': {},
            'critical_products': [],
            'surplus_products': [],
            'optimal_reorder_points': {},
            'last_analysis_time': None,
            'optimization_weights': optimization_weights or {
                'inventory_carrying_cost': 0.4,
                'stockout_cost': 0.4,
                'order_cost': 0.2
            },
            'product_id': product_id,
            'store_id': store_id
        })
        
        # Load previous state if available
        self.load_state()
        
        # Load data if not already in state
        if self.state['inventory_data'] is None:
            self.load_data()
    
    def load_data(self):
        """
        Load and preprocess the inventory and demand data.
        """
        self.log("Loading inventory and demand data...")
        
        # Load inventory data
        inventory_df = load_inventory_data()
        self.state['inventory_data'] = inventory_df
        
        # Load demand data for correlation analysis
        demand_df = load_demand_data()
        self.state['demand_data'] = demand_df
        
        self.log(f"Loaded inventory data with {len(inventory_df)} records")
        self.log(f"Loaded demand data with {len(demand_df)} records")
        
        # Save state after loading data
        self.save_state()
    
    def analyze(self, data=None):
        """
        Analyze inventory data to identify critical issues and optimization opportunities.
        
        Args:
            data: Optional updated inventory data
            
        Returns:
            dict: Analysis results
        """
        self.log("Starting inventory analysis...")
        
        # Use provided data or data from state
        inventory_df = data if data is not None else self.state['inventory_data']
        demand_df = self.state['demand_data']
        
        if inventory_df is None:
            self.log("No inventory data available. Please load data first.", level='error')
            return None
        
        # Update last analysis time
        self.state['last_analysis_time'] = pd.Timestamp.now().isoformat()
        
        # Calculate key inventory metrics
        metrics = self._calculate_inventory_metrics(inventory_df)
        self.state['inventory_metrics'] = metrics
        
        # Identify products at risk of stockout
        critical_products = self._identify_critical_products(inventory_df)
        self.state['critical_products'] = critical_products
        
        # Identify products with excess inventory
        surplus_products = self._identify_surplus_products(inventory_df, demand_df)
        self.state['surplus_products'] = surplus_products
        
        # Calculate optimal reorder points
        optimal_reorder_points = self._calculate_optimal_reorder_points(inventory_df, demand_df)
        self.state['optimal_reorder_points'] = optimal_reorder_points
        
        # Generate inventory visualizations
        self._generate_inventory_visualizations(inventory_df, critical_products, surplus_products)
        
        # Create analysis summary
        analysis_results = {
            'timestamp': self.state['last_analysis_time'],
            'metrics': metrics,
            'critical_products_count': len(critical_products),
            'surplus_products_count': len(surplus_products),
            'reorder_recommendations_count': len(optimal_reorder_points),
            'top_critical_products': critical_products[:5] if critical_products else [],
            'top_surplus_products': surplus_products[:5] if surplus_products else []
        }
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        # Save state after analysis
        self.save_state()
        
        return analysis_results
    
    def _calculate_inventory_metrics(self, inventory_df):
        """
        Calculate overall inventory metrics.
        
        Args:
            inventory_df: DataFrame with inventory data
            
        Returns:
            dict: Inventory metrics
        """
        self.log("Calculating inventory metrics...")
        
        # Calculate total inventory value
        total_stock = inventory_df['Stock Levels'].sum()
        
        # Calculate average stock level per product
        avg_stock_per_product = inventory_df.groupby('Product ID')['Stock Levels'].mean()
        
        # Calculate average stock level per store
        avg_stock_per_store = inventory_df.groupby('Store ID')['Stock Levels'].mean()
        
        # Calculate average stockout frequency
        avg_stockout_frequency = inventory_df['Stockout Frequency'].mean()
        
        # Calculate average supplier lead time
        avg_lead_time = inventory_df['Supplier Lead Time (days)'].mean()
        
        # Calculate average order fulfillment time
        avg_fulfillment_time = inventory_df['Order Fulfillment Time (days)'].mean()
        
        # Calculate percentage of products below reorder point
        below_reorder_count = inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']].shape[0]
        below_reorder_percentage = (below_reorder_count / len(inventory_df)) * 100
        
        # Calculate overall warehouse capacity utilization
        capacity_utilization = (inventory_df['Stock Levels'].sum() / inventory_df['Warehouse Capacity'].sum()) * 100
        
        # Calculate days until expiry statistics
        expiring_soon_percentage = 0
        try:
            # Ensure 'Expiry Date' is in datetime format
            inventory_df['Expiry Date'] = pd.to_datetime(inventory_df['Expiry Date'], errors='coerce')
            
            # Only process if we have valid dates
            if not inventory_df['Expiry Date'].isna().all():
                current_date = datetime.now().date()
                
                # Filter out NaT values when calculating days until expiry
                valid_dates = inventory_df.dropna(subset=['Expiry Date'])
                if not valid_dates.empty:
                    # Calculate days until expiry for valid dates
                    valid_dates['Days Until Expiry'] = (valid_dates['Expiry Date'].dt.date - current_date).dt.days
                    
                    # Calculate percentage of products expiring within 30 days
                    expiring_soon = valid_dates[(valid_dates['Days Until Expiry'] >= 0) & 
                                                (valid_dates['Days Until Expiry'] <= 30)]
                    expiring_soon_percentage = (len(expiring_soon) / len(inventory_df)) * 100
        except Exception as e:
            self.log(f"Error calculating expiry metrics: {e}", level='warning')
            expiring_soon_percentage = 0
        
        metrics = {
            'total_stock': total_stock,
            'avg_stock_per_product': avg_stock_per_product.mean(),
            'avg_stock_per_store': avg_stock_per_store.mean(),
            'avg_stockout_frequency': avg_stockout_frequency,
            'avg_lead_time': avg_lead_time,
            'avg_fulfillment_time': avg_fulfillment_time,
            'below_reorder_percentage': below_reorder_percentage,
            'capacity_utilization': capacity_utilization,
            'expiring_soon_percentage': expiring_soon_percentage,
            'product_count': inventory_df['Product ID'].nunique(),
            'store_count': inventory_df['Store ID'].nunique()
        }
        
        return metrics
    
    def _identify_critical_products(self, inventory_df):
        """
        Identify products at risk of stockout.
        
        Args:
            inventory_df: DataFrame with inventory data
            
        Returns:
            list: List of dictionaries with critical product information
        """
        self.log("Identifying products at risk of stockout...")
        
        # Products with stock levels below reorder point
        critical_df = inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']]
        
        # Sort by stockout risk (combination of current stock level vs reorder point and stockout frequency)
        critical_df['Stockout Risk Score'] = (
            (critical_df['Reorder Point'] - critical_df['Stock Levels']) / critical_df['Reorder Point'] + 
            critical_df['Stockout Frequency']
        )
        
        critical_df = critical_df.sort_values('Stockout Risk Score', ascending=False)
        
        # Create list of critical products with relevant details
        critical_products = []
        for _, row in critical_df.iterrows():
            critical_products.append({
                'product_id': row['Product ID'],
                'store_id': row['Store ID'],
                'current_stock': row['Stock Levels'],
                'reorder_point': row['Reorder Point'],
                'below_reorder_percent': ((row['Reorder Point'] - row['Stock Levels']) / row['Reorder Point']) * 100,
                'stockout_frequency': row['Stockout Frequency'],
                'lead_time': row['Supplier Lead Time (days)'],
                'risk_score': row['Stockout Risk Score']
            })
        
        return critical_products
    
    def _identify_surplus_products(self, inventory_df, demand_df):
        """
        Identify products with excess inventory.
        
        Args:
            inventory_df: DataFrame with inventory data
            demand_df: DataFrame with demand data
            
        Returns:
            list: List of dictionaries with surplus product information
        """
        self.log("Identifying products with excess inventory...")
        
        # Group demand data by product and store to get average daily demand
        demand_agg = demand_df.groupby(['Product ID', 'Store ID'])['Sales Quantity'].mean().reset_index()
        demand_agg.rename(columns={'Sales Quantity': 'Avg_Daily_Demand'}, inplace=True)
        
        # Merge inventory and demand data
        merged_df = inventory_df.merge(demand_agg, on=['Product ID', 'Store ID'], how='left')
        merged_df['Avg_Daily_Demand'].fillna(0.1, inplace=True)  # Fill missing demand with small value
        
        # Calculate days of supply (stock levels / average daily demand)
        merged_df['Days_Of_Supply'] = merged_df['Stock Levels'] / merged_df['Avg_Daily_Demand']
        
        # Calculate maximum reasonable days of supply based on lead time and reorder point
        merged_df['Max_Reasonable_Supply'] = (merged_df['Supplier Lead Time (days)'] * 2) + 30  # Lead time + buffer
        
        # Identify products with excessive inventory
        surplus_df = merged_df[merged_df['Days_Of_Supply'] > merged_df['Max_Reasonable_Supply']]
        
        # Calculate excess percentage
        surplus_df['Excess_Percentage'] = ((surplus_df['Days_Of_Supply'] - surplus_df['Max_Reasonable_Supply']) / 
                                          surplus_df['Max_Reasonable_Supply']) * 100
        
        # Sort by excess percentage
        surplus_df = surplus_df.sort_values('Excess_Percentage', ascending=False)
        
        # Create list of surplus products with relevant details
        surplus_products = []
        for _, row in surplus_df.iterrows():
            surplus_products.append({
                'product_id': row['Product ID'],
                'store_id': row['Store ID'],
                'current_stock': row['Stock Levels'],
                'avg_daily_demand': row['Avg_Daily_Demand'],
                'days_of_supply': row['Days_Of_Supply'],
                'max_reasonable_supply': row['Max_Reasonable_Supply'],
                'excess_percentage': row['Excess_Percentage'],
                'excess_units': row['Stock Levels'] - (row['Avg_Daily_Demand'] * row['Max_Reasonable_Supply'])
            })
        
        return surplus_products
    
    def _calculate_optimal_reorder_points(self, inventory_df, demand_df):
        """
        Calculate optimal reorder points for products based on demand patterns and lead times.
        
        Args:
            inventory_df: DataFrame with inventory data
            demand_df: DataFrame with demand data
            
        Returns:
            dict: Dictionary with optimal reorder points by product and store
        """
        self.log("Calculating optimal reorder points...")
        
        # Group demand data by product and store to get average and standard deviation of daily demand
        demand_stats = demand_df.groupby(['Product ID', 'Store ID'])['Sales Quantity'].agg(['mean', 'std']).reset_index()
        demand_stats.columns = ['Product ID', 'Store ID', 'Avg_Daily_Demand', 'Std_Daily_Demand']
        
        # Handle missing standard deviations
        demand_stats['Std_Daily_Demand'].fillna(demand_stats['Avg_Daily_Demand'] * 0.2, inplace=True)
        
        # Merge with inventory data
        merged_df = inventory_df.merge(demand_stats, on=['Product ID', 'Store ID'], how='left')
        
        # Handle missing demand data
        merged_df['Avg_Daily_Demand'].fillna(0.1, inplace=True)
        merged_df['Std_Daily_Demand'].fillna(0.05, inplace=True)
        
        # Calculate optimal reorder points
        # Reorder Point = Lead Time Demand + Safety Stock
        # Lead Time Demand = Average Daily Demand * Lead Time
        # Safety Stock = Z-score * Standard Deviation of Demand * sqrt(Lead Time)
        # Z-score based on desired service level (use 1.96 for 97.5% service level)
        z_score = 1.96
        
        merged_df['Lead_Time_Demand'] = merged_df['Avg_Daily_Demand'] * merged_df['Supplier Lead Time (days)']
        merged_df['Safety_Stock'] = (z_score * merged_df['Std_Daily_Demand'] * 
                                    np.sqrt(merged_df['Supplier Lead Time (days)']))
        
        merged_df['Optimal_Reorder_Point'] = merged_df['Lead_Time_Demand'] + merged_df['Safety_Stock']
        
        # Round up reorder points to nearest integer
        merged_df['Optimal_Reorder_Point'] = np.ceil(merged_df['Optimal_Reorder_Point'])
        
        # Calculate difference from current reorder point
        merged_df['Reorder_Point_Difference'] = merged_df['Optimal_Reorder_Point'] - merged_df['Reorder Point']
        
        # Create dictionary of optimal reorder points
        optimal_reorder_points = {}
        for _, row in merged_df.iterrows():
            key = f"{row['Product ID']}_{row['Store ID']}"
            optimal_reorder_points[key] = {
                'product_id': row['Product ID'],
                'store_id': row['Store ID'],
                'current_reorder_point': row['Reorder Point'],
                'optimal_reorder_point': row['Optimal_Reorder_Point'],
                'reorder_point_difference': row['Reorder_Point_Difference'],
                'avg_daily_demand': row['Avg_Daily_Demand'],
                'lead_time': row['Supplier Lead Time (days)'],
                'lead_time_demand': row['Lead_Time_Demand'],
                'safety_stock': row['Safety_Stock'],
                'current_stock': row['Stock Levels']
            }
        
        return optimal_reorder_points
    
    def _generate_inventory_visualizations(self, inventory_df, critical_products, surplus_products):
        """
        Generate visualizations for inventory analysis.
        
        Args:
            inventory_df: DataFrame with inventory data
            critical_products: List of critical products
            surplus_products: List of surplus products
        """
        self.log("Generating inventory visualizations...")
        
        # Create directory for visualizations
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot overall inventory metrics
        plt.figure(figsize=(10, 6))
        metrics = self.state['inventory_metrics']
        metrics_to_plot = {
            'Avg Stock per Product': metrics['avg_stock_per_product'],
            'Avg Stock per Store': metrics['avg_stock_per_store'],
            'Avg Lead Time (days)': metrics['avg_lead_time'],
            'Avg Stockout Frequency': metrics['avg_stockout_frequency'],
            'Capacity Utilization (%)': metrics['capacity_utilization']
        }
        
        plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        plt.title('Key Inventory Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'key_metrics.png'))
        plt.close()
        
        # Plot critical products (top 10)
        if critical_products:
            critical_df = pd.DataFrame(critical_products[:10])
            plt.figure(figsize=(12, 6))
            sns.barplot(x='product_id', y='risk_score', data=critical_df)
            plt.title('Top 10 Products at Risk of Stockout')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'critical_products.png'))
            plt.close()
        
        # Plot surplus products (top 10)
        if surplus_products:
            surplus_df = pd.DataFrame(surplus_products[:10])
            plt.figure(figsize=(12, 6))
            sns.barplot(x='product_id', y='excess_percentage', data=surplus_df)
            plt.title('Top 10 Products with Excess Inventory')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'surplus_products.png'))
            plt.close()
        
        # Create visualization for inventory distribution across stores
        store_inventory = inventory_df.groupby('Store ID')['Stock Levels'].sum().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Store ID', y='Stock Levels', data=store_inventory)
        plt.title('Total Inventory by Store')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'inventory_by_store.png'))
        plt.close()
    
    def _save_analysis_results(self, results):
        """
        Save analysis results to a file.
        
        Args:
            results: Dictionary with analysis results
        """
        results_file = os.path.join(self.output_dir, 'analysis_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializable)
        self.log(f"Analysis results saved to {results_file}")
    
    def make_recommendation(self):
        """
        Generate recommendations based on inventory analysis.
        
        Returns:
            dict: Dictionary with recommendations
        """
        self.log("Generating inventory recommendations...")
        
        # Check if we have analysis results
        if self.state.get('last_analysis_time') is None:
            self.analyze()
        
        recommendations = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'restock_recommendations': [],
            'reduce_stock_recommendations': [],
            'reorder_point_adjustments': [],
            'general_recommendations': []
        }
        
        # Recommendations for products at risk of stockout
        critical_products = self.state.get('critical_products', [])
        for product in critical_products[:20]:  # Top 20 most critical
            reorder_key = f"{product['product_id']}_{product['store_id']}"
            optimal_reorder = self.state['optimal_reorder_points'].get(reorder_key, {})
            
            restock_rec = {
                'product_id': product['product_id'],
                'store_id': product['store_id'],
                'current_stock': product['current_stock'],
                'reorder_point': product['reorder_point'],
                'risk_score': product['risk_score'],
                'recommended_order_quantity': max(
                    optimal_reorder.get('optimal_reorder_point', product['reorder_point']) - product['current_stock'],
                    0
                ),
                'urgency': 'High' if product['risk_score'] > 1.5 else 'Medium'
            }
            recommendations['restock_recommendations'].append(restock_rec)
        
        # Recommendations for products with excess inventory
        surplus_products = self.state.get('surplus_products', [])
        for product in surplus_products[:20]:  # Top 20 with most excess
            reduce_rec = {
                'product_id': product['product_id'],
                'store_id': product['store_id'],
                'current_stock': product['current_stock'],
                'days_of_supply': product['days_of_supply'],
                'excess_percentage': product['excess_percentage'],
                'recommended_reduction': max(int(product['excess_units']), 0),
                'recommendation_type': 'Transfer' if product['excess_percentage'] > 100 else 'Promotion'
            }
            recommendations['reduce_stock_recommendations'].append(reduce_rec)
        
        # Recommendations for reorder point adjustments
        optimal_reorder_points = self.state.get('optimal_reorder_points', {})
        significant_adjustments = []
        
        for key, reorder_data in optimal_reorder_points.items():
            # Focus on significant differences (>20% change)
            current = reorder_data['current_reorder_point']
            optimal = reorder_data['optimal_reorder_point']
            if current == 0:
                percent_change = 100 if optimal > 0 else 0
            else:
                percent_change = abs((optimal - current) / current * 100)
            
            if percent_change > 20:
                reorder_data['percent_change'] = percent_change
                significant_adjustments.append(reorder_data)
        
        # Sort by percent change
        significant_adjustments.sort(key=lambda x: x['percent_change'], reverse=True)
        
        # Take top 20 most significant adjustments
        for adjustment in significant_adjustments[:20]:
            adjust_rec = {
                'product_id': adjustment['product_id'],
                'store_id': adjustment['store_id'],
                'current_reorder_point': adjustment['current_reorder_point'],
                'recommended_reorder_point': adjustment['optimal_reorder_point'],
                'percent_change': adjustment['percent_change'],
                'avg_daily_demand': adjustment['avg_daily_demand'],
                'lead_time': adjustment['lead_time']
            }
            recommendations['reorder_point_adjustments'].append(adjust_rec)
        
        # General recommendations based on overall metrics
        metrics = self.state.get('inventory_metrics', {})
        
        if metrics.get('capacity_utilization', 0) > 85:
            recommendations['general_recommendations'].append({
                'type': 'Capacity',
                'recommendation': 'Warehouse capacity utilization is high. Consider expanding storage or reducing overall inventory levels.',
                'priority': 'Medium'
            })
        
        if metrics.get('below_reorder_percentage', 0) > 15:
            recommendations['general_recommendations'].append({
                'type': 'Inventory Levels',
                'recommendation': 'High percentage of products below reorder point. Review supply chain and increase order frequencies.',
                'priority': 'High'
            })
        
        if metrics.get('expiring_soon_percentage', 0) > 10:
            recommendations['general_recommendations'].append({
                'type': 'Expiry Management',
                'recommendation': 'Significant inventory approaching expiry dates. Implement FIFO strictly and consider promotional activities.',
                'priority': 'High'
            })
        
        # Add recommendations to agent's recommendation history
        for rec_type, rec_list in recommendations.items():
            if isinstance(rec_list, list) and rec_list:
                summary = f"{len(rec_list)} {rec_type}"
                self.add_recommendation(summary, confidence=0.9, context=rec_type)
        
        # Save recommendations
        rec_file = os.path.join(self.output_dir, 'inventory_recommendations.json')
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=self._json_serializable)
        
        self.log(f"Recommendations saved to {rec_file}")
        return recommendations
    
    def update(self, feedback=None):
        """
        Update the agent's state based on feedback or new data.
        
        Args:
            feedback: Dictionary with feedback on recommendations
            
        Returns:
            bool: True if update was successful
        """
        self.log("Updating agent state based on feedback...")
        
        if feedback:
            # Process feedback on recommendations
            self.log(f"Received feedback: {feedback}")
            
            # If feedback includes accepted recommendations, track them
            if 'accepted_recommendations' in feedback:
                for rec in feedback['accepted_recommendations']:
                    self.log(f"Recommendation accepted: {rec}")
                    # Could update learning weights here for future recommendations
            
            # If feedback includes new inventory data, update the state
            if 'new_inventory_data' in feedback:
                self.state['inventory_data'] = feedback['new_inventory_data']
                self.log("Updated inventory data based on feedback")
                
                # Re-run analysis with new data
                self.analyze()
        
        # Save state after update
        self.save_state()
        return True
    
    def receive_message(self, sender, message, message_type='info'):
        """
        Process a message from another agent.
        
        Args:
            sender: The name of the agent sending the message
            message: The content of the message
            message_type: The type of message (info, request, response)
            
        Returns:
            Any: Optional response to the message
        """
        # Call parent method to log the message
        super().receive_message(sender, message, message_type)
        
        response = None
        
        # Process different types of messages
        if message_type == 'request':
            if 'get_critical_products' in message:
                # Return critical products data
                response = {
                    'critical_products': self.state.get('critical_products', [])
                }
                self.log(f"Sending critical products data to {sender}")
                
            elif 'get_surplus_products' in message:
                # Return surplus products data
                response = {
                    'surplus_products': self.state.get('surplus_products', [])
                }
                self.log(f"Sending surplus products data to {sender}")
                
            elif 'get_optimal_reorder_points' in message:
                # Return optimal reorder points
                response = {
                    'optimal_reorder_points': self.state.get('optimal_reorder_points', {})
                }
                self.log(f"Sending optimal reorder points to {sender}")
                
            elif 'analyze_product' in message and 'product_id' in message:
                # Analyze a specific product
                product_id = message['product_id']
                store_id = message.get('store_id')  # Optional
                
                # Get product data
                product_data = get_product_data(product_id, store_id)
                
                # Perform product-specific analysis
                product_analysis = self._analyze_product(product_data)
                
                response = {
                    'product_analysis': product_analysis
                }
                self.log(f"Sending product analysis for {product_id} to {sender}")
        
        return response
    
    def _analyze_product(self, product_data):
        """
        Perform detailed analysis on a specific product.
        
        Args:
            product_data: Dictionary with product data
            
        Returns:
            dict: Analysis results for the product
        """
        if not product_data or 'inventory' not in product_data:
            return {'error': 'Product data not available'}
        
        inventory_data = product_data['inventory']
        demand_data = product_data.get('demand', [])
        
        # Calculate average stock level
        avg_stock = sum(item['Stock Levels'] for item in inventory_data) / len(inventory_data) if inventory_data else 0
        
        # Calculate average stockout frequency
        avg_stockout = sum(item['Stockout Frequency'] for item in inventory_data) / len(inventory_data) if inventory_data else 0
        
        # Calculate average lead time
        avg_lead_time = sum(item['Supplier Lead Time (days)'] for item in inventory_data) / len(inventory_data) if inventory_data else 0
        
        # Calculate average daily demand
        avg_daily_demand = sum(item['Sales Quantity'] for item in demand_data) / len(demand_data) if demand_data else 0
        
        # Calculate days of supply
        days_of_supply = avg_stock / avg_daily_demand if avg_daily_demand > 0 else 0
        
        analysis = {
            'product_id': product_data.get('product_id'),
            'avg_stock_level': avg_stock,
            'avg_stockout_frequency': avg_stockout,
            'avg_lead_time': avg_lead_time,
            'avg_daily_demand': avg_daily_demand,
            'days_of_supply': days_of_supply,
            'stock_status': 'Low' if days_of_supply < avg_lead_time else 'Adequate' if days_of_supply < avg_lead_time * 3 else 'Excess',
            'risk_assessment': 'High' if avg_stockout > 0.3 else 'Medium' if avg_stockout > 0.1 else 'Low'
        }
        
        return analysis 