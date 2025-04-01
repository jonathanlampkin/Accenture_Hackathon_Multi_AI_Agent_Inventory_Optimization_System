"""
Demand Forecasting Agent for the Multi-Agent Inventory Optimization System.
This agent is responsible for analyzing historical demand data and generating
forecasts for future demand to support inventory planning.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.agents.base_agent import BaseAgent
from src.utils.data_loader import load_demand_data, load_pricing_data, get_product_data

class DemandAgent(BaseAgent):
    """
    Agent responsible for demand forecasting and analysis.
    """
    
    def __init__(self, product_id=None, store_id=None):
        """
        Initialize the Demand Forecasting Agent.
        
        Args:
            product_id: Optional specific product ID to focus on
            store_id: Optional specific store ID to focus on
        """
        super().__init__("DemandAgent")
        
        # Initialize agent-specific state
        self.state.update({
            'demand_data': None,
            'pricing_data': None,
            'forecast_horizon': config.FORECAST_HORIZON,
            'demand_patterns': {},
            'forecasts': {},
            'last_analysis_time': None,
            'seasonality_factors': {},
            'product_id': product_id,
            'store_id': store_id
        })
        
        # Load previous state if available
        self.load_state()
        
        # Load data if not already in state
        if self.state['demand_data'] is None:
            self.load_data()
    
    def load_data(self):
        """
        Load and preprocess the demand and pricing data.
        """
        self.log("Loading demand and pricing data...")
        
        # Load demand data
        demand_df = load_demand_data()
        self.state['demand_data'] = demand_df
        
        # Load pricing data for correlation analysis
        pricing_df = load_pricing_data()
        self.state['pricing_data'] = pricing_df
        
        self.log(f"Loaded demand data with {len(demand_df)} records")
        self.log(f"Loaded pricing data with {len(pricing_df)} records")
        
        # Save state after loading data
        self.save_state()
    
    def analyze(self, data=None):
        """
        Analyze demand data to identify patterns and trends.
        
        Args:
            data: Optional updated demand data
            
        Returns:
            dict: Analysis results
        """
        self.log("Starting demand analysis...")
        
        # Use provided data or data from state
        demand_df = data if data is not None else self.state['demand_data']
        pricing_df = self.state['pricing_data']
        
        if demand_df is None:
            self.log("No demand data available. Please load data first.", level='error')
            return None
        
        # Update last analysis time
        self.state['last_analysis_time'] = pd.Timestamp.now().isoformat()
        
        # Ensure date is in datetime format
        demand_df['Date'] = pd.to_datetime(demand_df['Date'])
        
        # Analyze demand patterns
        demand_patterns = self._analyze_demand_patterns(demand_df)
        self.state['demand_patterns'] = demand_patterns
        
        # Analyze seasonality factors
        seasonality = self._analyze_seasonality(demand_df)
        self.state['seasonality_factors'] = seasonality
        
        # Analyze price elasticity
        price_elasticity = self._analyze_price_elasticity(demand_df, pricing_df)
        self.state['price_elasticity'] = price_elasticity
        
        # Create analysis summary
        analysis_results = {
            'timestamp': self.state['last_analysis_time'],
            'demand_patterns': {k: v for k, v in list(demand_patterns.items())[:10]},  # First 10 patterns
            'seasonality_factors': {k: v for k, v in list(seasonality.items())[:10]},  # First 10 seasonality factors
            'price_elasticity': {k: v for k, v in list(price_elasticity.items())[:10]}  # First 10 elasticity values
        }
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        # Save state after analysis
        self.save_state()
        
        return analysis_results
    
    def _analyze_demand_patterns(self, demand_df):
        """
        Analyze demand patterns for each product.
        
        Args:
            demand_df: DataFrame or dict with demand data
            
        Returns:
            dict: Dictionary with demand patterns by product
        """
        self.log("Analyzing demand patterns...")
        
        # Ensure demand_df is a DataFrame
        if isinstance(demand_df, dict):
            demand_df = pd.DataFrame(demand_df)
            
        # Check if we have valid DataFrame
        if demand_df is None or len(demand_df) == 0:
            self.log("Warning: Empty demand data for pattern analysis", level='warning')
            return {}
            
        # Check if required columns exist
        required_columns = ['Product ID', 'Date', 'Sales Quantity']
        if not all(col in demand_df.columns for col in required_columns):
            self.log(f"Warning: Missing required columns in demand data: {required_columns}", level='warning')
            return {}
        
        try:
            # Group by product and date
            product_demand = demand_df.groupby(['Product ID', 'Date'])['Sales Quantity'].sum().reset_index()
            
            # Dictionary to store demand patterns
            demand_patterns = {}
            
            # Get unique product IDs
            product_ids = product_demand['Product ID'].unique()
            
            for product_id in product_ids:
                # Convert numpy types to Python native types
                product_id_key = int(product_id) if np.issubdtype(type(product_id), np.integer) else product_id
                
                # Filter data for this product
                product_data = product_demand[product_demand['Product ID'] == product_id]
                
                # Sort by date
                product_data = product_data.sort_values('Date')
                
                # Calculate rolling statistics
                rolling_mean = product_data['Sales Quantity'].rolling(window=7, min_periods=1).mean()
                rolling_std = product_data['Sales Quantity'].rolling(window=7, min_periods=1).std()
                
                # Identify demand pattern based on trend and volatility
                # Calculate trend as the difference between latest and earliest rolling mean
                trend = rolling_mean.iloc[-1] - rolling_mean.iloc[0] if len(rolling_mean) > 1 else 0
                
                # Calculate volatility as the average coefficient of variation
                volatility = rolling_std.mean() / rolling_mean.mean() if rolling_mean.mean() > 0 else 0
                
                # Determine demand pattern
                if trend > 0.1 * rolling_mean.mean():
                    trend_type = "Increasing"
                elif trend < -0.1 * rolling_mean.mean():
                    trend_type = "Decreasing"
                else:
                    trend_type = "Stable"
                    
                if volatility > 0.3:
                    volatility_type = "High"
                elif volatility > 0.1:
                    volatility_type = "Medium"
                else:
                    volatility_type = "Low"
                
                # Store demand pattern information
                demand_patterns[product_id_key] = {
                    'trend': trend_type,
                    'volatility': volatility_type,
                    'mean_demand': float(product_data['Sales Quantity'].mean()),
                    'max_demand': float(product_data['Sales Quantity'].max()),
                    'min_demand': float(product_data['Sales Quantity'].min()),
                    'last_demand': float(product_data['Sales Quantity'].iloc[-1]) if not product_data.empty else 0
                }
            
            return demand_patterns
        except Exception as e:
            self.log(f"Error analyzing demand patterns: {e}", level='error')
            return {}
    
    def _analyze_seasonality(self, demand_df):
        """
        Analyze seasonality factors in demand data.
        
        Args:
            demand_df: DataFrame with demand data
            
        Returns:
            dict: Dictionary with seasonality factors by product
        """
        self.log("Analyzing seasonality factors...")
        
        # Ensure demand_df is a DataFrame
        if isinstance(demand_df, dict):
            demand_df = pd.DataFrame(demand_df)
            
        # Ensure date is in datetime format
        demand_df['Date'] = pd.to_datetime(demand_df['Date'])
        
        # Add month column for seasonality analysis
        demand_df['Month'] = demand_df['Date'].dt.month
        
        # Dictionary to store seasonality factors
        seasonality_factors = {}
        
        # Get unique product IDs
        product_ids = demand_df['Product ID'].unique()
        
        for product_id in product_ids:
            # Convert numpy types to Python native types
            product_id_key = int(product_id) if np.issubdtype(type(product_id), np.integer) else product_id
            
            # Filter data for this product
            product_data = demand_df[demand_df['Product ID'] == product_id]
            
            # Calculate monthly average demand
            monthly_demand = product_data.groupby('Month')['Sales Quantity'].mean().reset_index()
            
            # Calculate overall average demand
            overall_avg = monthly_demand['Sales Quantity'].mean()
            
            # Calculate seasonality factor for each month (relative to overall average)
            if overall_avg > 0:
                monthly_demand['Seasonality Factor'] = monthly_demand['Sales Quantity'] / overall_avg
            else:
                monthly_demand['Seasonality Factor'] = 1.0  # Default if no demand
            
            # Extract seasonality factors as dictionary {month: factor}
            # Convert month values to Python native integers
            product_seasonality = {int(month): float(factor) for month, factor in 
                                  zip(monthly_demand['Month'], monthly_demand['Seasonality Factor'])}
            
            # Determine peak and low seasons
            if len(monthly_demand) > 0:
                peak_month = int(monthly_demand.loc[monthly_demand['Sales Quantity'].idxmax(), 'Month'])
                low_month = int(monthly_demand.loc[monthly_demand['Sales Quantity'].idxmin(), 'Month'])
            else:
                peak_month = 0
                low_month = 0
            
            # Store seasonality information
            seasonality_factors[product_id_key] = {
                'monthly_factors': product_seasonality,
                'peak_month': peak_month,
                'low_month': low_month,
                'seasonality_strength': float(monthly_demand['Seasonality Factor'].max() - monthly_demand['Seasonality Factor'].min()) if not monthly_demand.empty else 0
            }
        
        return seasonality_factors
    
    def _analyze_price_elasticity(self, demand_df, pricing_df):
        """
        Analyze price elasticity of demand for each product.
        
        Args:
            demand_df: DataFrame with demand data
            pricing_df: DataFrame with pricing data
            
        Returns:
            dict: Dictionary with price elasticity by product
        """
        self.log("Analyzing price elasticity...")
        
        # Dictionary to store price elasticity
        price_elasticity = {}
        
        # Ensure inputs are DataFrames
        if isinstance(demand_df, dict):
            demand_df = pd.DataFrame(demand_df)
        if isinstance(pricing_df, dict):
            pricing_df = pd.DataFrame(pricing_df)
        
        # Check if required columns exist
        if not all(col in demand_df.columns for col in ['Product ID', 'Store ID', 'Sales Quantity']):
            self.log("Missing required columns in demand data", level='warning')
            return price_elasticity
        
        if not all(col in pricing_df.columns for col in ['Product ID', 'Store ID', 'Price']):
            self.log("Missing required columns in pricing data", level='warning')
            return price_elasticity
        
        try:
            # Merge pricing and demand data
            merged_df = pricing_df.merge(demand_df, on=['Product ID', 'Store ID'], how='inner')
            
            # Get unique product IDs
            product_ids = merged_df['Product ID'].unique()
            
            for product_id in product_ids:
                # Convert numpy types to Python native types
                product_id_key = int(product_id) if np.issubdtype(type(product_id), np.integer) else product_id
                
                # Filter data for this product
                product_data = merged_df[merged_df['Product ID'] == product_id]
                
                # Check if 'Price' column exists in the merged dataframe
                if 'Price' not in product_data.columns:
                    self.log(f"Price column missing for product {product_id}", level='warning')
                    price_elasticity[product_id_key] = {
                        'elasticity': 0,
                        'elasticity_type': "Unknown"
                    }
                    continue
                
                # Calculate average price and demand for each store
                store_avg = product_data.groupby('Store ID').agg({
                    'Price': 'mean',
                    'Sales Quantity': 'mean'
                }).reset_index()
                
                # Need at least 2 stores to calculate elasticity
                if len(store_avg) > 1:
                    # Calculate elasticity using log-log regression if possible
                    try:
                        # Calculate log values
                        store_avg['Log_Price'] = np.log(store_avg['Price'])
                        store_avg['Log_Demand'] = np.log(store_avg['Sales Quantity'])
                        
                        # Simple linear regression
                        x = store_avg['Log_Price']
                        y = store_avg['Log_Demand']
                        
                        # Calculate elasticity using covariance and variance
                        elasticity = -np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
                    except Exception as e:
                        self.log(f"Error calculating elasticity for product {product_id}: {e}", level='warning')
                        elasticity = 0
                else:
                    elasticity = 0
                
                # Store elasticity information
                if elasticity != 0:
                    if abs(elasticity) > 1:
                        elasticity_type = "Elastic"
                    elif abs(elasticity) < 1:
                        elasticity_type = "Inelastic"
                    else:
                        elasticity_type = "Unit Elastic"
                else:
                    elasticity_type = "Unknown"
                
                price_elasticity[product_id_key] = {
                    'elasticity': float(elasticity),
                    'elasticity_type': elasticity_type
                }
        except Exception as e:
            self.log(f"Error in price elasticity analysis: {e}", level='error')
        
        return price_elasticity
    
    def make_forecast(self, product_id=None, store_id=None):
        """
        Generate demand forecasts for specific product/store or all products.
        
        Args:
            product_id: Optional specific product ID to forecast
            store_id: Optional specific store ID to forecast
            
        Returns:
            dict: Forecasts with predicted values
        """
        self.log(f"Generating demand forecasts for product_id={product_id}, store_id={store_id}")
        
        demand_df = self.state['demand_data']
        
        if demand_df is None:
            self.log("No demand data available. Please load data first.", level='error')
            return None
        
        # Ensure date is in datetime format
        demand_df['Date'] = pd.to_datetime(demand_df['Date'])
        
        # Dictionary to store forecasts
        forecasts = {}
        
        # Filter data based on parameters
        if product_id is not None:
            filtered_df = demand_df[demand_df['Product ID'] == product_id]
            if store_id is not None:
                filtered_df = filtered_df[filtered_df['Store ID'] == store_id]
                
            # Get product-store combinations
            combinations = [{'product_id': product_id, 'store_id': store_id}]
        else:
            # Get all product-store combinations
            combinations = [
                {'product_id': p, 'store_id': s} 
                for p, s in demand_df[['Product ID', 'Store ID']].drop_duplicates().values
            ]
            # Limit to top products if there are too many
            if len(combinations) > 100:
                combinations = combinations[:100]
            
        # Generate forecasts for each combination
        for combo in combinations:
            p_id = combo['product_id']
            s_id = combo['store_id']
            
            # Skip if store_id is None but product_id is specified
            if product_id is not None and store_id is None and s_id is not None:
                continue
                
            key = f"{p_id}_{s_id}" if s_id is not None else str(p_id)
            
            # Filter data for this combination
            if s_id is not None:
                combo_data = demand_df[(demand_df['Product ID'] == p_id) & (demand_df['Store ID'] == s_id)]
            else:
                combo_data = demand_df[demand_df['Product ID'] == p_id]
                # Aggregate across stores
                combo_data = combo_data.groupby('Date')['Sales Quantity'].sum().reset_index()
            
            # Sort by date
            combo_data = combo_data.sort_values('Date')
            
            # Check if we have enough data
            if len(combo_data) < 14:  # Need at least 2 weeks of data
                self.log(f"Not enough data for product {p_id}, store {s_id}. Skipping forecast.", level='warning')
                continue
            
            # Generate forecast
            try:
                forecast_values = self._generate_time_series_forecast(combo_data)
                
                # Store forecast
                forecasts[key] = {
                    'product_id': p_id,
                    'store_id': s_id,
                    'forecast_values': forecast_values,
                    'forecast_dates': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                                      for i in range(1, self.state['forecast_horizon'] + 1)],
                    'confidence': 0.8,  # Default confidence level
                    'method': 'ARIMA'
                }
                
                self.log(f"Generated forecast for product {p_id}, store {s_id}")
            except Exception as e:
                self.log(f"Error generating forecast for product {p_id}, store {s_id}: {e}", level='error')
        
        # Update state with new forecasts
        self.state['forecasts'].update(forecasts)
        self.save_state()
        
        return forecasts
    
    def _generate_time_series_forecast(self, data):
        """
        Generate time series forecast using ARIMA or Exponential Smoothing.
        
        Args:
            data: DataFrame with date and sales quantity
            
        Returns:
            list: Forecast values for forecast_horizon days
        """
        # Extract sales quantity series
        if 'Sales Quantity' in data.columns:
            sales = data['Sales Quantity'].values
        else:
            return [0] * self.state['forecast_horizon']  # Return zeros if no sales data
        
        # Check if we have enough data
        if len(sales) < 14:
            return [sales.mean()] * self.state['forecast_horizon']  # Return mean if not enough data
        
        # Try ARIMA model first
        try:
            # Simple ARIMA model
            model = ARIMA(sales, order=(1, 0, 0))
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=self.state['forecast_horizon'])
            
            # Ensure non-negative values
            forecast = np.maximum(forecast, 0)
            
            return forecast.tolist()
        except:
            # Fallback to Exponential Smoothing
            try:
                model = ExponentialSmoothing(sales)
                model_fit = model.fit()
                
                # Generate forecast
                forecast = model_fit.forecast(self.state['forecast_horizon'])
                
                # Ensure non-negative values
                forecast = np.maximum(forecast, 0)
                
                return forecast.tolist()
            except:
                # If all else fails, use simple moving average
                avg_sales = np.mean(sales[-7:])  # Average of last 7 days
                return [avg_sales] * self.state['forecast_horizon']
    
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
        Generate recommendations based on demand analysis and forecasts.
        
        Returns:
            dict: Dictionary with recommendations
        """
        self.log("Generating demand recommendations...")
        
        # Check if we have analysis results
        if self.state.get('last_analysis_time') is None:
            self.analyze()
        
        # Check if we have forecasts
        if not self.state.get('forecasts'):
            self.make_forecast()
        
        recommendations = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'trend_based_recommendations': [],
            'seasonality_recommendations': [],
            'peak_demand_products': [],
            'declining_demand_products': []
        }
        
        # Recommendations based on demand patterns
        demand_patterns = self.state.get('demand_patterns', {})
        for product_id, pattern in demand_patterns.items():
            if pattern['trend'] == 'Increasing':
                recommendations['trend_based_recommendations'].append({
                    'product_id': product_id,
                    'trend': pattern['trend'],
                    'recommendation': 'Increase stock levels to meet growing demand',
                    'confidence': 0.8 if pattern['volatility'] == 'Low' else 0.6
                })
                
                # Add to peak demand products
                recommendations['peak_demand_products'].append({
                    'product_id': product_id,
                    'mean_demand': pattern['mean_demand'],
                    'growth_potential': 'High'
                })
                
            elif pattern['trend'] == 'Decreasing':
                recommendations['trend_based_recommendations'].append({
                    'product_id': product_id,
                    'trend': pattern['trend'],
                    'recommendation': 'Reduce stock levels to avoid excess inventory',
                    'confidence': 0.8 if pattern['volatility'] == 'Low' else 0.6
                })
                
                # Add to declining demand products
                recommendations['declining_demand_products'].append({
                    'product_id': product_id,
                    'mean_demand': pattern['mean_demand'],
                    'decline_severity': 'High' if pattern['mean_demand'] - pattern['last_demand'] > 0.3 * pattern['mean_demand'] else 'Medium'
                })
        
        # Recommendations based on seasonality
        seasonality_factors = self.state.get('seasonality_factors', {})
        current_month = datetime.now().month
        
        for product_id, seasonality in seasonality_factors.items():
            # Check if we're approaching peak season (within 1 month)
            peak_month = seasonality['peak_month']
            months_to_peak = (peak_month - current_month) % 12
            
            if months_to_peak <= 1 and seasonality['seasonality_strength'] > 0.3:
                # Approaching peak season
                recommendations['seasonality_recommendations'].append({
                    'product_id': product_id,
                    'months_to_peak': months_to_peak,
                    'seasonality_strength': seasonality['seasonality_strength'],
                    'recommendation': 'Increase stock in preparation for seasonal peak',
                    'confidence': 0.9
                })
            elif months_to_peak >= 10 and seasonality['seasonality_strength'] > 0.3:
                # Currently in peak season
                recommendations['seasonality_recommendations'].append({
                    'product_id': product_id,
                    'peak_month': peak_month,
                    'seasonality_strength': seasonality['seasonality_strength'],
                    'recommendation': 'Maintain high stock levels during peak season',
                    'confidence': 0.9
                })
        
        # Add recommendations to agent's recommendation history
        for rec_type, rec_list in recommendations.items():
            if isinstance(rec_list, list) and rec_list:
                summary = f"{len(rec_list)} {rec_type}"
                self.add_recommendation(summary, confidence=0.85, context=rec_type)
        
        # Save recommendations
        rec_file = os.path.join(self.output_dir, 'demand_recommendations.json')
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        self.log(f"Recommendations saved to {rec_file}")
        return recommendations
    
    def update(self, feedback=None):
        """
        Update the agent's state based on feedback or new data.
        
        Args:
            feedback: Dictionary with feedback on forecasts or recommendations
            
        Returns:
            bool: True if update was successful
        """
        self.log("Updating agent state based on feedback...")
        
        if feedback:
            # Process feedback on forecasts or recommendations
            self.log(f"Received feedback: {feedback}")
            
            # If feedback includes new demand data, update the state
            if 'new_demand_data' in feedback:
                self.state['demand_data'] = feedback['new_demand_data']
                self.log("Updated demand data based on feedback")
                
                # Re-run analysis with new data
                self.analyze()
                
                # Re-generate forecasts with new data
                self.make_forecast()
        
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
            if 'get_forecast' in message:
                product_id = message.get('product_id')
                store_id = message.get('store_id')
                
                # Generate forecast if needed
                if product_id:
                    forecast = self.make_forecast(product_id, store_id)
                    
                    # Return forecast
                    response = {
                        'forecast': forecast
                    }
                    self.log(f"Sending forecast for product {product_id} to {sender}")
                    
            elif 'get_demand_pattern' in message and 'product_id' in message:
                product_id = message['product_id']
                
                # Get demand pattern for product
                pattern = self.state['demand_patterns'].get(product_id)
                
                if pattern:
                    response = {
                        'demand_pattern': pattern
                    }
                    self.log(f"Sending demand pattern for product {product_id} to {sender}")
                else:
                    response = {
                        'error': f"No demand pattern available for product {product_id}"
                    }
                    self.log(f"No demand pattern found for product {product_id}", level='warning')
        
        return response