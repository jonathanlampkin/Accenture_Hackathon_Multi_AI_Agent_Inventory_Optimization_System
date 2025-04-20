"""
Advanced Forecasting Models Demo

This script demonstrates the usage of various advanced forecasting models
and their comparison for inventory optimization.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ForecastingDemo")

# Import forecasting models
from models.forecasting_models import (
    SARIMAModel, ProphetModel, XGBoostForecastModel, 
    LightGBMForecastModel, NeuralProphetModel
)
from models.model_comparison import ModelComparison

def run_demo(data_path, output_dir=None, models=None, test_size=0.2):
    """
    Run the forecasting models demonstration
    
    Args:
        data_path: Path to CSV data file (required)
        output_dir: Directory to save outputs (optional)
        models: List of model types to include (optional)
        test_size: Proportion of data to use for testing
    """
    # Validate data path
    if not data_path or not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        print(f"Error: Data file not found: {data_path}")
        print("Please provide a valid path to a CSV file with historical data.")
        sys.exit(1)
    
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'output/forecasting_demo_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to {output_dir}")
    
    # Load data
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Determine columns
    date_col = None
    target_col = None
    
    # Try to find date column
    date_candidates = ['date', 'ds', 'timestamp', 'time', 'Date', 'DATE']
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.error("Could not identify a date column in the data")
        print("Error: Could not identify a date column in the data.")
        print(f"Available columns: {', '.join(df.columns)}")
        print("Please ensure your data has a date column with one of these names: date, ds, timestamp, time, Date, DATE")
        sys.exit(1)
    
    # Try to find target column
    target_candidates = ['demand', 'sales', 'value', 'y', 'target', 'quantity', 'Demand', 'Sales']
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Use the first numeric column that's not the date
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != date_col:
                target_col = col
                break
    
    if target_col is None:
        logger.error("Could not identify a target column in the data")
        print("Error: Could not identify a target column in the data.")
        print(f"Available columns: {', '.join(df.columns)}")
        print("Please ensure your data has a numeric target column.")
        sys.exit(1)
    
    logger.info(f"Using date column: '{date_col}', target column: '{target_col}'")
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[target_col])
    plt.title(f'Historical {target_col.capitalize()} Data')
    plt.xlabel('Date')
    plt.ylabel(target_col.capitalize())
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'historical_data.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Historical data plot saved")
    
    # Initialize model comparison
    model_comparison = ModelComparison(output_dir=output_dir)
    
    # Add specified models or all models
    all_model_types = ['sarima', 'prophet', 'xgboost', 'lightgbm', 'neuralprophet']
    models_to_use = [m.lower() for m in models] if models else all_model_types
    
    # Validate models
    for model in list(models_to_use):  # Create a copy to iterate over
        if model not in all_model_types:
            logger.warning(f"Unknown model type: {model}. Ignoring.")
            models_to_use.remove(model)
    
    logger.info(f"Using these models: {', '.join(models_to_use)}")
    
    # Add models
    if 'sarima' in models_to_use:
        model_comparison.add_model(SARIMAModel())
    if 'prophet' in models_to_use:
        model_comparison.add_model(ProphetModel())
    if 'xgboost' in models_to_use:
        model_comparison.add_model(XGBoostForecastModel())
    if 'lightgbm' in models_to_use:
        model_comparison.add_model(LightGBMForecastModel())
    if 'neuralprophet' in models_to_use:
        model_comparison.add_model(NeuralProphetModel())
    
    # Run comparison
    logger.info("Running model comparison...")
    
    results = model_comparison.run_full_comparison(
        data=df,
        date_col=date_col,
        target_col=target_col,
        test_size=test_size
    )
    
    logger.info(f"Model comparison completed. Results saved to {output_dir}")
    
    # Print summary of results
    evaluation_df = results['evaluation']
    logger.info("\nEvaluation Results:")
    logger.info(evaluation_df.to_string())
    
    # Print best model for each metric
    logger.info("\nBest Models by Metric:")
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in evaluation_df.columns:
            best_model = evaluation_df.loc[evaluation_df[metric].idxmin()]
            logger.info(f"{metric}: {best_model['model']} ({best_model[metric]:.4f})")
    
    if 'R²' in evaluation_df.columns:
        best_model = evaluation_df.loc[evaluation_df['R²'].idxmax()]
        logger.info(f"R²: {best_model['model']} ({best_model['R²']:.4f})")
    
    logger.info(f"\nDemo completed successfully. All outputs saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Advanced Forecasting Models Demo')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file (required)')
    parser.add_argument('--output', type=str, help='Directory to save outputs (optional)')
    parser.add_argument('--models', nargs='+', help='List of models to include (optional)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    run_demo(
        data_path=args.data,
        output_dir=args.output,
        models=args.models,
        test_size=args.test_size
    ) 