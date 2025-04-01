"""
Pricing Optimization Agent for the Multi-Agent Inventory Optimization System.
This agent is responsible for analyzing pricing data and making recommendations
for optimal pricing strategies to maximize revenue and manage inventory.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.agents.base_agent import BaseAgent
from src.utils.data_loader import load_pricing_data, load_demand_data, get_product_data

class PricingAgent(BaseAgent):
    """
    Agent responsible for pricing optimization and analysis.
    """
    
    def __init__(self, optimization_weights=None, product_id=None, store_id=None):
        """
        Initialize the Pricing Optimization Agent.
        
        Args:
            optimization_weights: Optional dictionary with weights for optimization objectives
            product_id: Optional specific product ID to focus on
            store_id: Optional specific store ID to focus on
        """
        super().__init__("PricingAgent")
        
        # Initialize agent-specific state
        self.state.update({
            'pricing_data': None,
            'demand_data': None,
            'price_elasticity': {},
            'competitor_analysis': {},
            'optimal_prices': {},
            'price_performance': {},
            'last_analysis_time': None,
            'optimization_weights': optimization_weights or {
                'profit_margin': 0.4,
                'sales_volume': 0.4,
                'market_share': 0.2
            },
            'product_id': product_id,
            'store_id': store_id
        })
        
        # Load previous state if available
        self.load_state()
        
        # Load data if not already in state
        if self.state['pricing_data'] is None:
            self.load_data()
    
    def load_data(self):
        """
        Load and preprocess the pricing and demand data.
        """
        self.log("Loading pricing and demand data...")
        
        # Load pricing data
        pricing_df = load_pricing_data()
        self.state['pricing_data'] = pricing_df
        
        # Load demand data for correlation analysis
        demand_df = load_demand_data()
        self.state['demand_data'] = demand_df
        
        self.log(f"Loaded pricing data with {len(pricing_df)} records")
        self.log(f"Loaded demand data with {len(demand_df)} records")
        
        # Save state after loading data
        self.save_state()
    
    def analyze(self, data=None):
        """
        Analyze pricing data to identify optimal pricing strategies.
        
        Args:
            data: Optional updated pricing data
            
        Returns:
            dict: Analysis results
        """
        self.log("Starting pricing analysis...")
        
        # Use provided data or data from state
        pricing_df = data if data is not None else self.state['pricing_data']
        demand_df = self.state['demand_data']
        
        if pricing_df is None:
            self.log("No pricing data available. Please load data first.", level='error')
            return None
        
        # Update last analysis time
        self.state['last_analysis_time'] = pd.Timestamp.now().isoformat()
        
        # Calculate price elasticity
        price_elasticity = self._analyze_price_elasticity(pricing_df, demand_df)
        self.state['price_elasticity'] = price_elasticity
        
        # Analyze competitor pricing
        competitor_analysis = self._analyze_competitor_pricing(pricing_df)
        self.state['competitor_analysis'] = competitor_analysis
        
        # Calculate optimal prices
        optimal_prices = self._calculate_optimal_prices(pricing_df, demand_df, price_elasticity)
        self.state['optimal_prices'] = optimal_prices
        
        # Analyze price performance
        price_performance = self._analyze_price_performance(pricing_df, demand_df)
        self.state['price_performance'] = price_performance
        
        # Create analysis summary
        analysis_results = {
            'timestamp': self.state['last_analysis_time'],
            'price_elasticity': {k: v for k, v in list(price_elasticity.items())[:10]},  # First 10 elasticity values
            'competitor_analysis': {k: v for k, v in list(competitor_analysis.items())[:10]},  # First 10 competitor analyses
            'optimal_prices': {k: v for k, v in list(optimal_prices.items())[:10]},  # First 10 optimal prices
            'price_performance': {k: v for k, v in list(price_performance.items())[:10]}  # First 10 price performances
        }
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        # Save state after analysis
        self.save_state()
        
        return analysis_results
    
    def _analyze_price_elasticity(self, pricing_df, demand_df):
        """
        Analyze price elasticity by comparing pricing and demand across different stores.
        
        Args:
            pricing_df: DataFrame with pricing data
            demand_df: DataFrame with demand data
            
        Returns:
            dict: Dictionary with price elasticity by product
        """
        self.log("Analyzing price elasticity...")
        
        # Dictionary to store price elasticity
        price_elasticity = {}
        
        # Make sure we have the right column names
        price_column = None
        for col in pricing_df.columns:
            if 'Price' in col and not 'Competitor' in col:
                price_column = col
                break
        
        sales_column = None
        for col in demand_df.columns:
            if 'Sales' in col or 'Quantity' in col:
                sales_column = col
                break
        
        if not price_column:
            self.log(f"WARNING: Price column not found in pricing data. Available columns: {pricing_df.columns.tolist()}")
            return price_elasticity
        
        if not sales_column:
            self.log(f"WARNING: Sales quantity column not found in demand data. Available columns: {demand_df.columns.tolist()}")
            return price_elasticity
        
        # Get unique product IDs
        product_ids = pricing_df['Product ID'].unique()
        
        for product_id in product_ids:
            # Convert product_id to standard Python int if it's numpy int
            product_id_key = int(product_id) if isinstance(product_id, (np.integer, int)) else product_id
            
            # Filter data for this product
            product_pricing_data = pricing_df[pricing_df['Product ID'] == product_id]
            product_demand_data = demand_df[demand_df['Product ID'] == product_id]
            
            # Merge the data
            product_data = product_pricing_data.merge(product_demand_data, on=['Product ID', 'Store ID'], how='inner')
            
            # Skip if not enough data
            if len(product_data) < 2:
                continue
            
            # Calculate average price and demand for each store
            try:
                store_avg = product_data.groupby('Store ID').agg({
                    price_column: 'mean',
                    sales_column: 'mean'
                }).reset_index()
                
                # Need at least 2 stores to calculate elasticity
                if len(store_avg) > 1:
                    # Calculate elasticity using log-log regression if possible
                    try:
                        # Calculate log values
                        store_avg['Log_Price'] = np.log(store_avg[price_column])
                        store_avg['Log_Demand'] = np.log(store_avg[sales_column])
                        
                        # Create linear regression model
                        model = LinearRegression()
                        X = store_avg['Log_Price'].values.reshape(-1, 1)
                        y = store_avg['Log_Demand'].values
                        
                        # Fit model
                        model.fit(X, y)
                        
                        # Elasticity is the coefficient (slope)
                        elasticity = model.coef_[0]
                    except Exception as e:
                        self.log(f"WARNING: Error calculating elasticity for product {product_id}: {str(e)}")
                        # Fall back to simple calculation if regression fails
                        elasticity = -1.0  # Default elasticity
                else:
                    elasticity = -1.0  # Default elasticity if not enough data
                
                # Determine elasticity type
                if abs(elasticity) > 1.5:
                    elasticity_type = "Highly Elastic"
                elif abs(elasticity) > 1:
                    elasticity_type = "Elastic"
                elif abs(elasticity) > 0.5:
                    elasticity_type = "Moderately Elastic"
                else:
                    elasticity_type = "Inelastic"
                
                # Store elasticity information
                price_elasticity[product_id_key] = {
                    'elasticity': float(elasticity),  # Convert to native float
                    'elasticity_type': elasticity_type,
                    'price_sensitivity': 'High' if abs(elasticity) > 1 else 'Medium' if abs(elasticity) > 0.5 else 'Low'
                }
            except Exception as e:
                self.log(f"WARNING: Price column missing for product {product_id}: {str(e)}")
        
        return price_elasticity
    
    def _analyze_competitor_pricing(self, pricing_df):
        """
        Analyze competitor pricing relative to our prices.
        
        Args:
            pricing_df: DataFrame with pricing data
            
        Returns:
            dict: Dictionary with competitor pricing analysis by product
        """
        self.log("Analyzing competitor pricing...")
        
        # Dictionary to store competitor analysis
        competitor_analysis = {}
        
        # Make sure we have the right column names
        price_column = None
        competitor_column = None
        
        for col in pricing_df.columns:
            if 'Price' in col and not 'Competitor' in col:
                price_column = col
            elif 'Competitor' in col and 'Price' in col:
                competitor_column = col
        
        if not price_column or not competitor_column:
            self.log(f"WARNING: Required columns not found for competitor analysis. Available columns: {pricing_df.columns.tolist()}")
            return competitor_analysis
        
        try:
            # Get unique product IDs
            product_ids = pricing_df['Product ID'].unique()
            
            for product_id in product_ids:
                # Convert product_id to standard Python int if it's numpy int
                product_id_key = int(product_id) if isinstance(product_id, (np.integer, int)) else product_id
                
                # Filter data for this product
                product_data = pricing_df[pricing_df['Product ID'] == product_id]
                
                # Calculate average price and competitor price for each store
                relative_price = []
                for _, row in product_data.iterrows():
                    try:
                        our_price = float(row[price_column])
                        competitor_prices = str(row[competitor_column])
                        
                        # Parse competitor prices (assuming comma-separated values)
                        # Handle the possibility of misformatted data
                        try:
                            # Try parsing as comma-separated
                            comp_prices = [float(p.strip()) for p in competitor_prices.split(',')]
                        except:
                            # Try as a single value
                            try:
                                comp_prices = [float(competitor_prices)]
                            except:
                                comp_prices = []
                        
                        avg_comp_price = sum(comp_prices) / len(comp_prices) if comp_prices else 0
                        if avg_comp_price > 0:
                            relative_price.append(our_price / avg_comp_price)
                    except Exception as e:
                        self.log(f"WARNING: Error processing competitor prices for product {product_id}, row {_}: {str(e)}")
                
                # Calculate average relative price across all stores
                avg_relative_price = sum(relative_price) / len(relative_price) if relative_price else 1.0
                
                # Determine pricing position
                if avg_relative_price > 1.05:
                    price_position = "Premium"
                elif avg_relative_price < 0.95:
                    price_position = "Discount"
                else:
                    price_position = "Competitive"
                
                # Store competitor analysis
                competitor_analysis[product_id_key] = {
                    'relative_price': float(avg_relative_price),
                    'price_position': price_position,
                    'premium_percentage': float((avg_relative_price - 1) * 100) if avg_relative_price > 1 else 0,
                    'discount_percentage': float((1 - avg_relative_price) * 100) if avg_relative_price < 1 else 0
                }
        except Exception as e:
            self.log(f"WARNING: Error analyzing competitor pricing: {str(e)}")
        
        return competitor_analysis
    
    def _calculate_optimal_prices(self, pricing_df, demand_df, price_elasticity):
        """
        Calculate optimal prices based on price elasticity and revenue optimization.
        
        Args:
            pricing_df: DataFrame with pricing data
            demand_df: DataFrame with demand data
            price_elasticity: Dictionary with price elasticity by product
            
        Returns:
            dict: Dictionary with optimal prices by product and store
        """
        self.log("Calculating optimal prices...")
        
        # Dictionary to store optimal prices
        optimal_prices = {}
        
        # Make sure we have the right column names
        price_column = None
        for col in pricing_df.columns:
            if 'Price' in col and not 'Competitor' in col:
                price_column = col
                break
        
        sales_column = None
        for col in demand_df.columns:
            if 'Sales' in col or 'Quantity' in col:
                sales_column = col
                break
        
        storage_column = None
        for col in pricing_df.columns:
            if 'Storage' in col or 'Cost' in col:
                storage_column = col
                break
        
        elasticity_column = None
        for col in pricing_df.columns:
            if 'Elasticity' in col:
                elasticity_column = col
                break
        
        if not price_column or not sales_column or not storage_column:
            self.log(f"WARNING: Required columns not found. Available pricing columns: {pricing_df.columns.tolist()}")
            self.log(f"WARNING: Available demand columns: {demand_df.columns.tolist()}")
            return optimal_prices
        
        # Merge pricing and demand data
        merged_df = pricing_df.merge(demand_df, on=['Product ID', 'Store ID'], how='inner')
        
        try:
            # Calculate average price, sales, and storage cost by product and store
            agg_dict = {
                price_column: 'mean',
                sales_column: 'mean',
                storage_column: 'mean'
            }
            
            if elasticity_column:
                agg_dict[elasticity_column] = 'mean'
            
            product_store_data = merged_df.groupby(['Product ID', 'Store ID']).agg(agg_dict).reset_index()
            
            # Calculate optimal prices for each product-store combination
            for _, row in product_store_data.iterrows():
                product_id = row['Product ID']
                store_id = row['Store ID']
                
                # Convert to standard Python types
                product_id_key = int(product_id) if isinstance(product_id, (np.integer, int)) else product_id
                store_id_key = int(store_id) if isinstance(store_id, (np.integer, int)) else store_id
                
                current_price = float(row[price_column])
                avg_sales = float(row[sales_column])
                storage_cost = float(row[storage_column])
                
                # Get elasticity from analysis or use default
                elasticity = price_elasticity.get(product_id_key, {}).get('elasticity', -1.0)
                
                # If product is elastic, optimal price is lower than current price
                # If product is inelastic, optimal price can be higher
                
                # For elastic products (|e| > 1), optimal markup is |e|/(|e|-1)
                # For inelastic products (|e| < 1), we'll use a different approach
                
                if abs(elasticity) > 1:
                    # Elastic demand: optimize based on elasticity
                    markup_factor = abs(elasticity) / (abs(elasticity) - 1)
                    optimal_price = storage_cost * markup_factor
                    
                    # Cap changes to avoid extreme prices
                    max_change = 0.15  # Maximum 15% change
                    if optimal_price < current_price * (1 - max_change):
                        optimal_price = current_price * (1 - max_change)
                    elif optimal_price > current_price * (1 + max_change):
                        optimal_price = current_price * (1 + max_change)
                else:
                    # Inelastic demand: consider small price increase
                    elasticity_factor = 1 - 0.5 * abs(elasticity)  # Factor between 0.5 and 1
                    optimal_price = current_price * (1 + 0.05 * elasticity_factor)  # Up to 5% increase
                
                # Calculate expected change in revenue
                expected_demand = avg_sales * ((optimal_price / current_price) ** elasticity)
                current_revenue = current_price * avg_sales
                expected_revenue = optimal_price * expected_demand
                revenue_change = (expected_revenue - current_revenue) / current_revenue * 100
                
                # Store optimal price
                key = f"{product_id_key}_{store_id_key}"
                optimal_prices[key] = {
                    'product_id': product_id_key,
                    'store_id': store_id_key,
                    'current_price': current_price,
                    'optimal_price': optimal_price,
                    'price_change_percentage': (optimal_price - current_price) / current_price * 100,
                    'expected_demand': float(expected_demand),
                    'expected_revenue': float(expected_revenue),
                    'revenue_change_percentage': float(revenue_change),
                    'elasticity': float(elasticity),
                    'storage_cost': float(storage_cost)
                }
        except Exception as e:
            self.log(f"WARNING: Error calculating optimal prices: {str(e)}")
        
        return optimal_prices
    
    def _analyze_price_performance(self, pricing_df, demand_df):
        """
        Analyze price performance and its impact on sales and revenue.
        
        Args:
            pricing_df: DataFrame with pricing data
            demand_df: DataFrame with demand data
            
        Returns:
            dict: Dictionary with price performance analysis by product
        """
        self.log("Analyzing price performance...")
        
        # Dictionary to store price performance
        price_performance = {}
        
        # Make sure we have the right column names
        price_column = None
        for col in pricing_df.columns:
            if 'Price' in col and not 'Competitor' in col:
                price_column = col
                break
        
        sales_column = None
        for col in demand_df.columns:
            if 'Sales' in col or 'Quantity' in col:
                sales_column = col
                break
        
        if not price_column or not sales_column:
            self.log(f"WARNING: Required columns not found for price performance analysis")
            return price_performance
        
        try:
            # Merge pricing and demand data
            merged_df = pricing_df.merge(demand_df, on=['Product ID', 'Store ID'], how='inner')
            
            # Get unique product IDs
            product_ids = merged_df['Product ID'].unique()
            
            for product_id in product_ids:
                # Convert product_id to standard Python int if it's numpy int
                product_id_key = int(product_id) if isinstance(product_id, (np.integer, int)) else product_id
                
                # Filter data for this product
                product_data = merged_df[merged_df['Product ID'] == product_id]
                
                # Skip if not enough data
                if len(product_data) < 2:
                    continue
                
                # Calculate average price, demand, and revenue by store
                product_data['Revenue'] = product_data[price_column] * product_data[sales_column]
                
                # Calculate price ranges
                min_price = float(product_data[price_column].min())
                max_price = float(product_data[price_column].max())
                avg_price = float(product_data[price_column].mean())
                
                # Find price that maximizes revenue
                store_avg = product_data.groupby('Store ID').agg({
                    price_column: 'mean',
                    sales_column: 'mean',
                    'Revenue': 'mean'
                }).reset_index()
                
                # Find most profitable price point
                max_revenue_idx = store_avg['Revenue'].idxmax()
                optimal_price = float(store_avg.loc[max_revenue_idx, price_column])
                
                # Calculate price performance metrics
                price_range_percentage = (max_price - min_price) / min_price * 100 if min_price > 0 else 0
                price_to_optimal_ratio = avg_price / optimal_price if optimal_price > 0 else 1.0
                
                # Store price performance metrics
                price_performance[product_id_key] = {
                    'min_price': min_price,
                    'max_price': max_price,
                    'avg_price': avg_price,
                    'optimal_price': optimal_price,
                    'price_range_percentage': float(price_range_percentage),
                    'price_to_optimal_ratio': float(price_to_optimal_ratio),
                    'price_adjustment_needed': float(optimal_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
                }
        except Exception as e:
            self.log(f"WARNING: Error analyzing price performance: {str(e)}")
        
        return price_performance
    
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
        Generate pricing recommendations based on analysis.
        
        Returns:
            dict: Dictionary with recommendations
        """
        self.log("Generating pricing recommendations...")
        
        # Check if we have analysis results
        if self.state.get('last_analysis_time') is None:
            self.analyze()
        
        recommendations = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'price_adjustment_recommendations': [],
            'competitive_pricing_recommendations': [],
            'clearance_pricing_recommendations': [],
            'premium_pricing_recommendations': []
        }
        
        # Get optimal prices from state
        optimal_prices = self.state.get('optimal_prices', {})
        
        # Get price elasticity data
        price_elasticity = self.state.get('price_elasticity', {})
        
        # Get competitor analysis
        competitor_analysis = self.state.get('competitor_analysis', {})
        
        # Price adjustment recommendations based on optimal prices
        for key, price_data in optimal_prices.items():
            # Only recommend changes above threshold
            change_threshold = 3.0  # 3% change threshold
            
            if abs(price_data['price_change_percentage']) >= change_threshold:
                # Create recommendation
                adjustment_rec = {
                    'product_id': price_data['product_id'],
                    'store_id': price_data['store_id'],
                    'current_price': price_data['current_price'],
                    'recommended_price': price_data['optimal_price'],
                    'price_change_percentage': price_data['price_change_percentage'],
                    'expected_revenue_impact': price_data['revenue_change_percentage'],
                    'confidence': 0.8 if abs(price_data['price_change_percentage']) > 10 else 0.9,
                    'priority': 'High' if abs(price_data['revenue_change_percentage']) > 10 else 'Medium'
                }
                
                recommendations['price_adjustment_recommendations'].append(adjustment_rec)
        
        # Sort price adjustment recommendations by expected revenue impact
        recommendations['price_adjustment_recommendations'].sort(
            key=lambda x: abs(x['expected_revenue_impact']), 
            reverse=True
        )
        
        # Limit to top 20 price adjustments
        recommendations['price_adjustment_recommendations'] = recommendations['price_adjustment_recommendations'][:20]
        
        # Competitive pricing recommendations
        for product_id, comp_data in competitor_analysis.items():
            elasticity_data = price_elasticity.get(product_id, {'price_sensitivity': 'Medium'})
            
            # For products priced significantly higher than competitors with high elasticity
            if comp_data['price_position'] == 'Premium' and comp_data['premium_percentage'] > 10 and elasticity_data['price_sensitivity'] == 'High':
                recommendations['competitive_pricing_recommendations'].append({
                    'product_id': product_id,
                    'price_position': comp_data['price_position'],
                    'premium_percentage': comp_data['premium_percentage'],
                    'elasticity': elasticity_data.get('elasticity', -1.0),
                    'recommendation': 'Reduce price to be more competitive',
                    'suggested_reduction': min(comp_data['premium_percentage'] * 0.5, 10),  # Reduce by half of premium or 10%, whichever is less
                    'priority': 'High'
                })
            
            # For products priced significantly lower than competitors with low elasticity
            elif comp_data['price_position'] == 'Discount' and comp_data['discount_percentage'] > 10 and elasticity_data['price_sensitivity'] == 'Low':
                recommendations['competitive_pricing_recommendations'].append({
                    'product_id': product_id,
                    'price_position': comp_data['price_position'],
                    'discount_percentage': comp_data['discount_percentage'],
                    'elasticity': elasticity_data.get('elasticity', -1.0),
                    'recommendation': 'Increase price to improve margins',
                    'suggested_increase': min(comp_data['discount_percentage'] * 0.5, 8),  # Increase by half of discount or 8%, whichever is less
                    'priority': 'Medium'
                })
        
        # Sort competitive pricing recommendations by priority
        recommendations['competitive_pricing_recommendations'].sort(
            key=lambda x: 0 if x['priority'] == 'High' else 1 if x['priority'] == 'Medium' else 2
        )
        
        # Generate premium pricing recommendations
        for product_id, elasticity_data in price_elasticity.items():
            if elasticity_data['price_sensitivity'] == 'Low' and elasticity_data['elasticity_type'] == 'Inelastic':
                # Check competitor analysis
                comp_data = competitor_analysis.get(product_id, {'price_position': 'Competitive'})
                
                if comp_data['price_position'] != 'Premium':
                    recommendations['premium_pricing_recommendations'].append({
                        'product_id': product_id,
                        'elasticity': elasticity_data.get('elasticity', -1.0),
                        'price_sensitivity': elasticity_data['price_sensitivity'],
                        'recommendation': 'Consider premium pricing strategy',
                        'suggested_increase': 5.0,  # 5% increase
                        'rationale': 'Low price sensitivity indicates potential for premium pricing'
                    })
        
        # Sort premium pricing recommendations by elasticity (less elastic first)
        recommendations['premium_pricing_recommendations'].sort(
            key=lambda x: abs(x['elasticity'])
        )
        
        # Limit to top 10 premium pricing recommendations
        recommendations['premium_pricing_recommendations'] = recommendations['premium_pricing_recommendations'][:10]
        
        # Add recommendations to agent's recommendation history
        for rec_type, rec_list in recommendations.items():
            if isinstance(rec_list, list) and rec_list:
                summary = f"{len(rec_list)} {rec_type}"
                self.add_recommendation(summary, confidence=0.85, context=rec_type)
        
        # Save recommendations
        rec_file = os.path.join(self.output_dir, 'pricing_recommendations.json')
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
            
            # If feedback includes new pricing data, update the state
            if 'new_pricing_data' in feedback:
                self.state['pricing_data'] = feedback['new_pricing_data']
                self.log("Updated pricing data based on feedback")
                
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
            if 'get_optimal_price' in message and 'product_id' in message:
                product_id = message['product_id']
                store_id = message.get('store_id')
                
                # Find optimal price for the product
                if store_id:
                    key = f"{product_id}_{store_id}"
                    optimal_price_data = self.state['optimal_prices'].get(key)
                else:
                    # Find optimal price across all stores
                    optimal_price_data = None
                    for key, data in self.state['optimal_prices'].items():
                        if data['product_id'] == product_id:
                            if optimal_price_data is None or data['expected_revenue'] > optimal_price_data['expected_revenue']:
                                optimal_price_data = data
                
                if optimal_price_data:
                    response = {
                        'optimal_price': optimal_price_data
                    }
                    self.log(f"Sending optimal price for product {product_id} to {sender}")
                else:
                    response = {
                        'error': f"No optimal price available for product {product_id}"
                    }
                    self.log(f"No optimal price found for product {product_id}", level='warning')
                    
            elif 'get_price_elasticity' in message and 'product_id' in message:
                product_id = message['product_id']
                
                # Get price elasticity for product
                elasticity_data = self.state['price_elasticity'].get(product_id)
                
                if elasticity_data:
                    response = {
                        'price_elasticity': elasticity_data
                    }
                    self.log(f"Sending price elasticity for product {product_id} to {sender}")
                else:
                    response = {
                        'error': f"No price elasticity available for product {product_id}"
                    }
                    self.log(f"No price elasticity found for product {product_id}", level='warning')
        
        return response