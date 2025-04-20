"""
Tools for demand forecasting operations used by the forecasting agent.

This module contains tools for generating forecasts, evaluating model performance,
and identifying sales patterns.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Import from improved_forecasting.py
from improved_forecasting import ImprovedForecaster


class ForecastDemandTool(BaseTool):
    """Tool for generating demand forecasts for products."""
    
    name: str = "Forecast Demand"
    description: str = """
    Generate demand forecasts for specified products or all products.
    
    Input should include:
    - product_ids (optional): List of product IDs to forecast, or None for all products
    - horizon: Number of days to forecast into the future (default: 30)
    - data_path: Path to the demand data CSV file
    - output_dir: Directory to save forecast results and visualizations
    """
    
    class InputSchema(BaseModel):
        product_ids: Optional[List[int]] = Field(
            None, 
            description="List of product IDs to forecast, or None for all products"
        )
        horizon: int = Field(
            30, 
            description="Number of days to forecast into the future"
        )
        data_path: str = Field(
            ..., 
            description="Path to the demand data CSV file"
        )
        output_dir: str = Field(
            ..., 
            description="Directory to save forecast results and visualizations"
        )
    
    def run(self, product_ids: Optional[List[int]] = None, 
            horizon: int = 30, 
            data_path: str = "data/demand_data.csv",
            output_dir: str = "output/forecasts") -> Dict[str, Any]:
        """
        Generate demand forecasts for the specified products or all products.
        
        Args:
            product_ids: List of product IDs to forecast, or None for all products
            horizon: Number of days to forecast into the future
            data_path: Path to the demand data CSV file
            output_dir: Directory to save forecast results and visualizations
            
        Returns:
            Dict containing forecast results
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize the improved forecaster
            forecaster = ImprovedForecaster()
            
            # Load demand data
            forecaster.load_demand_data(data_path)
            
            # Generate forecasts
            if product_ids:
                # Forecast specific products
                results = {}
                for product_id in product_ids:
                    result = forecaster.forecast_product(product_id, horizon=horizon)
                    results[product_id] = result
                
                # Combine results
                combined_results = pd.DataFrame()
                for product_id, result in results.items():
                    if 'forecast' in result:
                        forecast_df = pd.DataFrame({
                            'Date': result['forecast_dates'],
                            'Product ID': product_id,
                            'Forecast': result['forecast'],
                            'Method': result['method']
                        })
                        combined_results = pd.concat([combined_results, forecast_df])
                
                # Save results
                combined_results.to_csv(f"{output_dir}/forecasts_selected.csv", index=False)
                
                return {
                    "message": f"Forecasts generated for {len(product_ids)} products",
                    "products": product_ids,
                    "results_path": f"{output_dir}/forecasts_selected.csv"
                }
            
            else:
                # Forecast all products
                results_df = forecaster.forecast_all_products(horizon=horizon)
                
                # Save results
                results_path = f"{output_dir}/forecasts_all.csv"
                results_df.to_csv(results_path, index=False)
                
                # Generate visualizations
                forecaster.visualize_forecasts(results_df, output_dir)
                
                # Generate summary report
                summary_path = forecaster.generate_summary_report(results_df, output_dir)
                
                return {
                    "message": f"Forecasts generated for all products",
                    "products_count": results_df['Product ID'].nunique(),
                    "results_path": results_path,
                    "summary_path": summary_path
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate forecasts"
            }


class EvaluateModelPerformanceTool(BaseTool):
    """Tool for evaluating forecast model performance."""
    
    name: str = "Evaluate Model Performance"
    description: str = """
    Evaluate the performance of different forecasting models on the demand data.
    
    Input should include:
    - data_path: Path to the demand data CSV file
    - test_proportion: Proportion of data to use for testing (default: 0.2)
    - output_dir: Directory to save evaluation results
    """
    
    class InputSchema(BaseModel):
        data_path: str = Field(
            ..., 
            description="Path to the demand data CSV file"
        )
        test_proportion: float = Field(
            0.2, 
            description="Proportion of data to use for testing"
        )
        output_dir: str = Field(
            ..., 
            description="Directory to save evaluation results"
        )
    
    def run(self, data_path: str = "data/demand_data.csv",
            test_proportion: float = 0.2,
            output_dir: str = "output/model_evaluation") -> Dict[str, Any]:
        """
        Evaluate the performance of different forecasting models.
        
        Args:
            data_path: Path to the demand data CSV file
            test_proportion: Proportion of data to use for testing
            output_dir: Directory to save evaluation results
            
        Returns:
            Dict containing evaluation results
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize the improved forecaster
            forecaster = ImprovedForecaster()
            
            # Load demand data
            forecaster.load_demand_data(data_path)
            
            # Get unique product IDs
            product_ids = forecaster.data['Product ID'].unique()
            
            # Evaluate each model on each product
            models = ['sarima', 'exponential_smoothing', 'prophet', 'random_forest']
            results = []
            
            for product_id in product_ids:
                for model in models:
                    # Set forecasting method and evaluate
                    forecaster.method = model
                    result = forecaster.forecast_product(
                        product_id, 
                        test_proportion=test_proportion
                    )
                    
                    # Extract metrics
                    if 'metrics' in result:
                        metrics = result['metrics']
                        results.append({
                            'Product ID': product_id,
                            'Model': model,
                            'RMSE': metrics.get('rmse', np.nan),
                            'MAE': metrics.get('mae', np.nan),
                            'R2': metrics.get('r2', np.nan)
                        })
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results
            results_path = f"{output_dir}/model_evaluation.csv"
            results_df.to_csv(results_path, index=False)
            
            # Create performance comparison plots
            forecaster._create_performance_comparison(results_df, output_dir)
            
            return {
                "message": "Model evaluation completed",
                "products_count": len(product_ids),
                "models_evaluated": models,
                "results_path": results_path
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to evaluate model performance"
            }


class IdentifySalesPatternsTool(BaseTool):
    """Tool for identifying patterns in sales data."""
    
    name: str = "Identify Sales Patterns"
    description: str = """
    Analyze sales data to identify patterns, trends, seasonality, and anomalies.
    
    Input should include:
    - data_path: Path to the demand data CSV file
    - output_dir: Directory to save pattern analysis results
    """
    
    class InputSchema(BaseModel):
        data_path: str = Field(
            ..., 
            description="Path to the demand data CSV file"
        )
        output_dir: str = Field(
            ..., 
            description="Directory to save pattern analysis results"
        )
    
    def run(self, data_path: str = "data/demand_data.csv",
            output_dir: str = "output/pattern_analysis") -> Dict[str, Any]:
        """
        Analyze sales data to identify patterns and trends.
        
        Args:
            data_path: Path to the demand data CSV file
            output_dir: Directory to save pattern analysis results
            
        Returns:
            Dict containing pattern analysis results
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Load demand data
            data = pd.read_csv(data_path)
            
            # Ensure date is in datetime format
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Prepare results container
            patterns = []
            
            # Get unique product IDs
            product_ids = data['Product ID'].unique()
            
            for product_id in product_ids:
                # Filter data for this product
                product_data = data[data['Product ID'] == product_id].copy()
                product_data = product_data.sort_values('Date')
                
                # Get product category
                category = product_data['Category'].iloc[0] if 'Category' in product_data.columns else 'Unknown'
                
                # Calculate basic statistics
                sales = product_data['Sales Quantity'].values
                mean_sales = np.mean(sales)
                std_sales = np.std(sales)
                cv = std_sales / mean_sales if mean_sales > 0 else 0
                
                # Check for promotion effect if data available
                promo_effect = 0
                if 'On Promotion' in product_data.columns:
                    promo_sales = product_data[product_data['On Promotion'] == 1]['Sales Quantity'].mean()
                    non_promo_sales = product_data[product_data['On Promotion'] == 0]['Sales Quantity'].mean()
                    promo_effect = (promo_sales / non_promo_sales - 1) * 100 if non_promo_sales > 0 else 0
                
                # Check for holiday effect if data available
                holiday_effect = 0
                if 'Holiday' in product_data.columns:
                    holiday_sales = product_data[product_data['Holiday'] == 1]['Sales Quantity'].mean()
                    non_holiday_sales = product_data[product_data['Holiday'] == 0]['Sales Quantity'].mean()
                    holiday_effect = (holiday_sales / non_holiday_sales - 1) * 100 if non_holiday_sales > 0 else 0
                
                # Identify trend (simple approach: compare first and last quarters)
                n = len(sales)
                if n >= 8:
                    first_quarter = np.mean(sales[:n//4])
                    last_quarter = np.mean(sales[-n//4:])
                    trend_pct = (last_quarter / first_quarter - 1) * 100 if first_quarter > 0 else 0
                else:
                    trend_pct = 0
                
                # Determine pattern type
                if cv < 0.1:
                    pattern_type = "Stable"
                elif cv < 0.3:
                    pattern_type = "Moderately variable"
                else:
                    pattern_type = "Highly variable"
                
                # Add to patterns list
                patterns.append({
                    'Product ID': product_id,
                    'Category': category,
                    'Mean Sales': mean_sales,
                    'Sales Std Dev': std_sales,
                    'Coefficient of Variation': cv,
                    'Pattern Type': pattern_type,
                    'Trend (%)': trend_pct,
                    'Promotion Effect (%)': promo_effect,
                    'Holiday Effect (%)': holiday_effect
                })
            
            # Convert to DataFrame
            patterns_df = pd.DataFrame(patterns)
            
            # Save results
            results_path = f"{output_dir}/sales_patterns.csv"
            patterns_df.to_csv(results_path, index=False)
            
            return {
                "message": "Sales pattern analysis completed",
                "products_analyzed": len(product_ids),
                "results_path": results_path,
                "summary": {
                    "stable_products": len(patterns_df[patterns_df['Pattern Type'] == 'Stable']),
                    "moderate_products": len(patterns_df[patterns_df['Pattern Type'] == 'Moderately variable']),
                    "variable_products": len(patterns_df[patterns_df['Pattern Type'] == 'Highly variable']),
                    "avg_promotion_effect": patterns_df['Promotion Effect (%)'].mean(),
                    "avg_holiday_effect": patterns_df['Holiday Effect (%)'].mean()
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to identify sales patterns"
            } 