"""
Supply Chain Optimization Agent for the Multi-Agent Inventory Optimization System.
This agent is responsible for analyzing supplier performance, lead times, and
logistics to recommend improvements to the supply chain.
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
from src.utils.data_loader import load_inventory_data, load_demand_data

class SupplyChainAgent(BaseAgent):
    """
    Agent responsible for supply chain optimization.
    """
    
    def __init__(self, optimization_weights=None, product_id=None, store_id=None):
        """
        Initialize the Supply Chain Agent.
        
        Args:
            optimization_weights: Dictionary with weights for optimization goals
            product_id: Optional specific product ID to focus on
            store_id: Optional specific store ID to focus on
        """
        super().__init__("SupplyChainAgent")
        
        # Store optimization weights
        self.optimization_weights = optimization_weights or {
            'inventory_carrying_cost': 0.4,
            'stockout_cost': 0.4,
            'order_cost': 0.2
        }
        
        # Initialize state
        self.state.update({
            'optimization_weights': self.optimization_weights,
            'supplier_performance': {},
            'lead_time_analysis': {},
            'logistics_efficiency': {},
            'last_analysis_time': None,
            'product_id': product_id,
            'store_id': store_id
        })
        
        # Load previous state if available
        self.load_state()
        
        self.log(f"Supply Chain Agent initialized with optimization weights: {self.optimization_weights}")
    
    def analyze(self, data=None):
        """
        Analyze supply chain data to identify optimization opportunities.
        
        Args:
            data: Optional updated supply chain data
            
        Returns:
            dict: Analysis results
        """
        self.log("Starting supply chain analysis...")
        
        # Load data if not provided
        inventory_df = data if data is not None else load_inventory_data()
        demand_df = load_demand_data()
        
        if inventory_df is None:
            self.log("No inventory data available. Please load data first.", level='error')
            return None
        
        # Update last analysis time
        self.state['last_analysis_time'] = pd.Timestamp.now().isoformat()
        
        # Analyze supplier lead times
        lead_time_analysis = self._analyze_lead_times(inventory_df)
        self.state['lead_time_analysis'] = lead_time_analysis
        
        # Create analysis summary
        analysis_results = {
            'timestamp': self.state['last_analysis_time'],
            'lead_time_analysis': lead_time_analysis
        }
        
        # Save state after analysis
        self.save_state()
        
        return analysis_results
    
    def _analyze_lead_times(self, inventory_df):
        """
        Analyze supplier lead times.
        
        Args:
            inventory_df: DataFrame with inventory data
            
        Returns:
            dict: Lead time analysis results
        """
        # Calculate average lead time by store
        avg_lead_time_by_store = inventory_df.groupby('Store ID')['Supplier Lead Time (days)'].mean()
        
        # Calculate average lead time by product
        avg_lead_time_by_product = inventory_df.groupby('Product ID')['Supplier Lead Time (days)'].mean()
        
        # Identify long lead times (> 20 days)
        long_lead_times = inventory_df[inventory_df['Supplier Lead Time (days)'] > 20]
        
        # Calculate lead time variability
        lead_time_std = inventory_df.groupby(['Product ID', 'Store ID'])['Supplier Lead Time (days)'].std().reset_index()
        lead_time_std = lead_time_std.rename(columns={'Supplier Lead Time (days)': 'Lead Time Std'})
        
        # Create lead time analysis dictionary
        lead_time_analysis = {
            'avg_lead_time': inventory_df['Supplier Lead Time (days)'].mean(),
            'median_lead_time': inventory_df['Supplier Lead Time (days)'].median(),
            'max_lead_time': inventory_df['Supplier Lead Time (days)'].max(),
            'min_lead_time': inventory_df['Supplier Lead Time (days)'].min(),
            'long_lead_time_count': len(long_lead_times),
            'long_lead_time_percentage': (len(long_lead_times) / len(inventory_df)) * 100,
            'avg_lead_time_by_store': avg_lead_time_by_store.to_dict(),
            'avg_lead_time_by_product': {k: v for k, v in list(avg_lead_time_by_product.items())[:10]},  # First 10 products
            'high_variability_products': lead_time_std[lead_time_std['Lead Time Std'] > 5]['Product ID'].tolist()
        }
        
        return lead_time_analysis
    
    def make_recommendation(self):
        """
        Generate recommendations for supply chain optimization.
        
        Returns:
            dict: Dictionary with recommendations
        """
        self.log("Generating supply chain recommendations...")
        
        # Check if we have analysis results
        if self.state.get('last_analysis_time') is None:
            self.analyze()
        
        recommendations = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'lead_time_recommendations': [],
            'supplier_recommendations': [],
            'logistics_recommendations': []
        }
        
        # Generate lead time recommendations
        lead_time_analysis = self.state.get('lead_time_analysis', {})
        if lead_time_analysis:
            # Recommend improving lead times for products with high variability
            high_var_products = lead_time_analysis.get('high_variability_products', [])
            for product_id in high_var_products:
                recommendations['lead_time_recommendations'].append({
                    'product_id': product_id,
                    'recommendation': 'Reduce lead time variability',
                    'action': 'Work with suppliers to establish more consistent delivery schedules',
                    'priority': 'High',
                    'estimated_impact': 'Medium'
                })
            
            # Recommend alternative suppliers for products with long lead times
            if lead_time_analysis.get('long_lead_time_percentage', 0) > 10:
                recommendations['supplier_recommendations'].append({
                    'recommendation': 'Identify alternative suppliers',
                    'action': 'Source alternative suppliers for products with lead times > 20 days',
                    'priority': 'High',
                    'estimated_impact': 'High'
                })
        
        # Add recommendations to agent's recommendation history
        for rec_type, rec_list in recommendations.items():
            if isinstance(rec_list, list) and rec_list:
                summary = f"{len(rec_list)} {rec_type}"
                self.add_recommendation(summary, confidence=0.8, context=rec_type)
        
        # Save recommendations
        rec_file = os.path.join(self.output_dir, 'supply_chain_recommendations.json')
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=self._json_serializable)
        
        self.log(f"Recommendations saved to {rec_file}")
        return recommendations
    
    def update(self, feedback=None):
        """
        Update the agent's state based on feedback.
        
        Args:
            feedback: Dictionary with feedback on recommendations
            
        Returns:
            bool: True if update was successful
        """
        self.log("Updating agent state based on feedback...")
        
        if feedback:
            self.log(f"Received feedback: {feedback}")
            
            # If feedback includes accepted recommendations, track them
            if 'accepted_recommendations' in feedback:
                for rec in feedback['accepted_recommendations']:
                    self.log(f"Recommendation accepted: {rec}")
        
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
            if 'get_lead_time_analysis' in message:
                response = {
                    'lead_time_analysis': self.state.get('lead_time_analysis', {})
                }
                self.log(f"Sending lead time analysis to {sender}")
        
        return response
    
    def _json_serializable(self, obj):
        """Helper method to make objects JSON serializable"""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable") 