#!/usr/bin/env python3
"""
Simplified API for Inventory Optimization System
This is a direct wrapper around ImprovedForecaster without the full FastAPI interface
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import json
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ImprovedForecaster
from improved_forecasting import ImprovedForecaster, run_improved_forecasting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api/simple_api.log')
    ]
)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Convert pandas/numpy objects to Python native types for JSON serialization"""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def generate_forecast(data_path: str, output_dir: str, forecast_horizon: int = 30, max_products: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate forecasts for all products in the data
    
    Args:
        data_path: Path to the input data file
        output_dir: Path to the output directory
        forecast_horizon: Number of days to forecast
        max_products: Maximum number of products to process
        
    Returns:
        Dictionary with forecast results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the forecasting directly using the run_improved_forecasting function
        logger.info(f"Running improved forecasting with output to {output_dir}")
        run_improved_forecasting(
            demand_data_path=data_path,
            output_dir=output_dir,
            test_proportion=0.2,
            forecast_horizon=forecast_horizon
        )
        
        # Gather the results
        results = {
            'status': 'success',
            'output_directory': output_dir,
            'visualizations': [],
            'summary': []
        }
        
        # Find all visualization files
        for file in os.listdir(output_dir):
            if file.endswith(".png"):
                results['visualizations'].append(os.path.join(output_dir, file))
        
        # Try to read summary file
        summary_path = os.path.join(output_dir, "forecast_summary.csv")
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
            results['summary'] = convert_to_serializable(summary_df)
            results['forecast_count'] = len(summary_df)
        
        return results
    
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

def forecast_product(data_path: str, product_id: int, output_dir: str, forecast_horizon: int = 30) -> Dict[str, Any]:
    """
    Generate forecast for a specific product
    
    Args:
        data_path: Path to the input data file
        product_id: ID of the product to forecast
        output_dir: Path to the output directory
        forecast_horizon: Number of days to forecast
        
    Returns:
        Dictionary with forecast results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Create forecaster
        forecaster = ImprovedForecaster(data)
        
        # Generate forecast for the specific product
        result = forecaster.forecast_product(
            product_id=product_id,
            forecast_horizon=forecast_horizon
        )
        
        if not result:
            return {'status': 'error', 'message': f"Product ID {product_id} not found"}
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical, test, and forecast data if available
        if 'historical_data' in result and len(result['historical_data']) > 0:
            ax.plot(result['historical_dates'], result['historical_data'], label='Historical', color='blue')
        
        if 'test_data' in result and len(result['test_data']) > 0:
            ax.plot(result['test_dates'], result['test_data'], label='Test', color='green')
        
        ax.plot(result['forecast_dates'], result['forecast'], label='Forecast', color='red')
        
        # Add confidence intervals if available
        if 'lower_bound' in result and 'upper_bound' in result:
            ax.fill_between(result['forecast_dates'], 
                          result['lower_bound'], 
                          result['upper_bound'], 
                          color='red', alpha=0.2, label='95% Confidence')
        
        # Add metrics if available
        metrics_text = ""
        if 'metrics' in result:
            for metric, value in result['metrics'].items():
                if value is not None:
                    metrics_text += f"{metric}: {value:.2f}\n"
        
        if metrics_text:
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax.set_title(f"Forecast for Product {product_id} using {result['method']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales Quantity')
        ax.legend()
        plt.tight_layout()
        
        # Save visualization
        viz_file = f"forecast_product_{product_id}.png"
        viz_path = os.path.join(output_dir, viz_file)
        plt.savefig(viz_path)
        plt.close()
        
        # Prepare response
        response = {
            'status': 'success',
            'product_id': product_id,
            'method': result['method'],
            'forecast': convert_to_serializable(result['forecast']),
            'dates': [str(d) for d in result['forecast_dates']],
            'visualization': viz_path
        }
        
        if 'metrics' in result:
            metrics = {k: convert_to_serializable(v) for k, v in result['metrics'].items()}
            response.update(metrics)
        
        return response
        
    except Exception as e:
        logger.error(f"Error forecasting product: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    # Example usage
    data_path = "data/demand_data.csv"
    output_dir = "api/results/simple_test"
    
    if len(sys.argv) > 1 and sys.argv[1] == "product":
        # Generate forecast for a specific product
        product_id = int(sys.argv[2]) if len(sys.argv) > 2 else 101
        result = forecast_product(data_path, product_id, output_dir)
        print(json.dumps(convert_to_serializable(result), indent=2))
    else:
        # Generate forecasts for all products
        result = generate_forecast(data_path, output_dir)
        print(json.dumps(convert_to_serializable(result), indent=2)) 