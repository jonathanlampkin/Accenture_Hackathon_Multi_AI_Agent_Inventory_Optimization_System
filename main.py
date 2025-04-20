#!/usr/bin/env python3
"""
Multi-Agent Inventory Optimization System - Main Entry Point

Refactored to use CrewAI with Ollama integration for improved performance and structure.
"""

import argparse
import os
import logging
import time
import sys
import signal
import pandas as pd
import threading
import numpy as np
from datetime import datetime
import sqlite3
import traceback
import multiprocessing

# Assuming project structure allows this import path
from src.config import LOG_DIR, DATA_DIR
from src.agents import InventoryAgents
from src.tasks import InventoryTasks
# Correct the import for data loader - Import both functions
from src.utils.data_loader import load_inventory_data, load_demand_data
# Import logger utils
from src.utils.db_logger import initialize_db, log_result

# Set up logging
# --- Start Logging Setup ---
# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"inventory_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure root logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(thread)d - %(filename)s-%(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler(sys.stdout)
                    ])

# Configure specific loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)  # Reduce Ollama logging
logging.getLogger("crewai").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def configure_logging():
    """Configure and return the logger for use in other modules."""
    return logger

def load_data(file_name):
    """Load data from the specified CSV file.
    
    This is a wrapper function to provide compatibility with modules that expect
    a general load_data function rather than specific loaders.
    
    Args:
        file_name (str): The name of the CSV file to load
        
    Returns:
        pandas.DataFrame: The loaded data or None if loading fails
    """
    logger.info(f"Loading data from {file_name}")
    try:
        # Import directly from src.config to avoid name conflicts
        from src import config
        
        # Determine data type based on filename
        if 'inventory' in file_name.lower():
            # Update config.INVENTORY_DATA temporarily
            original_path = config.INVENTORY_DATA
            config.INVENTORY_DATA = os.path.join(DATA_DIR, file_name)
            data = load_inventory_data()
            config.INVENTORY_DATA = original_path  # Restore original path
            return data
        elif 'demand' in file_name.lower():
            # Update config.DEMAND_DATA temporarily
            original_path = config.DEMAND_DATA
            config.DEMAND_DATA = os.path.join(DATA_DIR, file_name)
            data = load_demand_data()
            config.DEMAND_DATA = original_path  # Restore original path
            return data
        else:
            # Generic CSV loading
            file_path = os.path.join(DATA_DIR, file_name)
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} records from {file_name}")
            return df
    except Exception as e:
        logger.error(f"Error loading data from {file_name}: {e}")
        logger.error(traceback.format_exc())
        return None

# Global timeout handler
def timeout_handler(signum, frame):
    logger.error("Script timed out!")
    raise TimeoutError("Script execution timed out")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the Multi-Agent Inventory Optimization System.')
    parser.add_argument('--model-name', type=str, default='llama3', help='Name of the Ollama model to use')
    parser.add_argument('--ollama-base-url', type=str, default='http://localhost:11434', help='Base URL for Ollama')
    default_data_file = os.path.join(DATA_DIR, "inventory_monitoring.csv") 
    parser.add_argument('--data-file', type=str, default=default_data_file, help='Path to inventory data')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for outputs')
    parser.add_argument('--timeout', type=int, default=180, help='Maximum execution time in seconds (default: 180)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return vars(args)

def evaluate_demand_forecasting(demand_data, product_id, output_dir, test_size=6):
    """
    Perform statistical forecasting on the demand data for a specific product.
    Uses a simple time series model with exponential smoothing.
    
    Args:
        demand_data (pd.DataFrame): The demand data
        product_id (str): The product ID to forecast for, or 'all' for all products
        output_dir (str): Directory to save results
        test_size (int): Number of periods to use for testing
        
    Returns:
        dict: Dictionary with forecasting performance metrics
    """
    try:
        logger.info(f"Evaluating demand forecasting for product {product_id}")
        
        # Determine which column name is used for demand
        demand_column = 'Sales Quantity'
        if demand_column not in demand_data.columns:
            logger.error(f"Required column '{demand_column}' not found in demand data")
            return None
        
        # Handle 'all' products case
        if product_id == 'all':
            # Get the list of unique product IDs
            product_ids = demand_data['Product ID'].unique()
            if len(product_ids) == 0:
                logger.warning("No product IDs found in demand data")
                return None
                
            logger.info(f"Forecasting for all {len(product_ids)} products")
            
            # Initialize metrics for all products
            all_rmse = []
            all_mape = []
            all_mae = []
            
            # Process each product
            for pid in product_ids:
                product_metrics = evaluate_demand_forecasting(demand_data, pid, output_dir, test_size)
                if product_metrics:
                    all_rmse.append(product_metrics['rmse'])
                    all_mape.append(product_metrics['mape'])
                    all_mae.append(product_metrics['mae'])
            
            # Calculate aggregate metrics
            if all_rmse:
                return {
                    'product_id': 'all',
                    'model_type': 'ExponentialSmoothing',
                    'rmse': np.mean(all_rmse),
                    'mape': np.mean(all_mape),
                    'mae': np.mean(all_mae),
                    'products_forecasted': len(all_rmse)
                }
            else:
                logger.warning("No successful forecasts for any product")
                return None
        
        # Extract the demand series for the specified product
        product_demand = demand_data[demand_data['Product ID'] == product_id][['Date', demand_column]].copy()
        
        if product_demand.empty:
            logger.warning(f"No demand data found for product {product_id}")
            return None
            
        # Convert to time series format
        product_demand['Date'] = pd.to_datetime(product_demand['Date'])
        product_demand.set_index('Date', inplace=True)
        product_demand.sort_index(inplace=True)
        
        # Check if we have enough data
        if len(product_demand) <= test_size:
            logger.warning(f"Insufficient data points for product {product_id}. Need more than {test_size} periods.")
            return None
            
        # Split into train and test sets
        train = product_demand[:-test_size]
        test = product_demand[-test_size:]
        
        # Simple exponential smoothing model
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Try different smoothing settings
        best_rmse = float('inf')
        best_forecast = None
        best_params = None
        
        # Grid search for optimal parameters
        for trend in [None, 'add', 'mul']:
            for seasonal in [None, 'add', 'mul']:
                try:
                    # Skip invalid combinations
                    if seasonal and len(train) < 2:
                        continue
                        
                    model = ExponentialSmoothing(
                        train[demand_column],
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=12 if seasonal else None
                    )
                    
                    model_fit = model.fit()
                    forecast = model_fit.forecast(test_size)
                    
                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((test[demand_column].values - forecast.values) ** 2))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_forecast = forecast
                        best_params = {'trend': trend, 'seasonal': seasonal}
                        
                except Exception as e:
                    logger.warning(f"Error fitting model with trend={trend}, seasonal={seasonal}: {e}")
                    continue
        
        if best_forecast is None:
            logger.warning(f"Could not find a suitable forecasting model for product {product_id}")
            return None
            
        # Calculate performance metrics
        mape = np.mean(np.abs((test[demand_column].values - best_forecast.values) / test[demand_column].values)) * 100
        mae = np.mean(np.abs(test[demand_column].values - best_forecast.values))
        
        # Prepare the results
        results = {
            'product_id': product_id,
            'model_type': 'ExponentialSmoothing',
            'parameters': best_params,
            'rmse': best_rmse,
            'mape': mape,
            'mae': mae,
            'forecast_values': best_forecast.tolist()
        }
        
        logger.info(f"Forecasting results for product {product_id}: RMSE={best_rmse:.2f}, MAPE={mape:.2f}%")
        
        # Save the forecast to a file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(output_dir, f"forecast_{product_id}.txt")
            
            with open(result_file, 'w') as f:
                f.write(f"Product ID: {product_id}\n")
                f.write(f"Model: ExponentialSmoothing\n")
                f.write(f"Parameters: {best_params}\n")
                f.write(f"RMSE: {best_rmse:.2f}\n")
                f.write(f"MAPE: {mape:.2f}%\n")
                f.write(f"MAE: {mae:.2f}\n\n")
                f.write("Forecast Values:\n")
                
                for date, value in zip(test.index, best_forecast):
                    f.write(f"{date.strftime('%Y-%m-%d')}: {value:.2f}\n")
                    
                f.write("\nActual Values:\n")
                for date, value in zip(test.index, test[demand_column]):
                    f.write(f"{date.strftime('%Y-%m-%d')}: {value}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in demand forecasting for product {product_id}: {e}")
        logger.error(traceback.format_exc())
        return None

def evaluate_advanced_forecasting(demand_data, product_id, output_dir, test_size=6):
    """
    Perform advanced forecasting on the demand data with ensemble methods and feature engineering.
    
    Args:
        demand_data (pd.DataFrame): The demand data
        product_id (str): The product ID to forecast for, or 'all' for all products
        output_dir (str): Directory to save results
        test_size (int): Number of periods to use for testing
        
    Returns:
        dict: Dictionary with forecasting performance metrics
    """
    try:
        logger.info(f"Running advanced forecasting for product {product_id}")
        
        # Determine which column name is used for demand
        demand_column = 'Sales Quantity'
        if demand_column not in demand_data.columns and 'Demand' in demand_data.columns:
            demand_column = 'Demand'
            logger.info(f"Using '{demand_column}' as demand column")
        
        # Handle 'all' products case
        if product_id == 'all':
            # Get the list of unique product IDs
            product_ids = demand_data['Product ID'].unique()
            if len(product_ids) == 0:
                logger.warning("No product IDs found in demand data")
                return None
                
            logger.info(f"Advanced forecasting for all {len(product_ids)} products")
            
            # Initialize metrics for all products
            all_rmse = []
            all_mape = []
            all_mae = []
            
            # Process each product
            for pid in product_ids:
                product_metrics = evaluate_advanced_forecasting(demand_data, pid, output_dir, test_size)
                if product_metrics:
                    all_rmse.append(product_metrics['rmse'])
                    all_mape.append(product_metrics['mape'])
                    all_mae.append(product_metrics['mae'])
            
            # Calculate aggregate metrics
            if all_rmse:
                return {
                    'product_id': 'all',
                    'model_type': 'Ensemble',
                    'rmse': np.mean(all_rmse),
                    'mape': np.mean(all_mape),
                    'mae': np.mean(all_mae),
                    'products_forecasted': len(all_rmse)
                }
            else:
                logger.warning("No successful forecasts for any product")
                return None
        
        # Extract the demand series for the specified product
        product_demand = demand_data[demand_data['Product ID'] == product_id][['Date', demand_column]].copy()
        
        if product_demand.empty:
            logger.warning(f"No demand data found for product {product_id}")
            return None
            
        # Convert to time series format
        product_demand['Date'] = pd.to_datetime(product_demand['Date'])
        product_demand.set_index('Date', inplace=True)
        product_demand.sort_index(inplace=True)
        
        # Check if we have enough data
        if len(product_demand) <= test_size:
            logger.warning(f"Insufficient data points for product {product_id}. Need more than {test_size} periods.")
            return None
            
        # Feature engineering
        # Add calendar features
        product_demand['dayofweek'] = product_demand.index.dayofweek
        product_demand['month'] = product_demand.index.month
        product_demand['quarter'] = product_demand.index.quarter
        
        # Add lag features
        for lag in [1, 2, 3, 6, 12]:
            if lag < len(product_demand):
                product_demand[f'lag_{lag}'] = product_demand[demand_column].shift(lag)
        
        # Add rolling statistics
        for window in [3, 6, 12]:
            if window < len(product_demand):
                product_demand[f'rolling_mean_{window}'] = product_demand[demand_column].rolling(window=window).mean()
                product_demand[f'rolling_std_{window}'] = product_demand[demand_column].rolling(window=window).std()
        
        # Replace NaN values that result from the lag and rolling features
        product_demand = product_demand.fillna(method='bfill')
        product_demand = product_demand.fillna(method='ffill')
        
        # Split into train and test sets
        train = product_demand[:-test_size]
        test = product_demand[-test_size:]
        
        # Create ensemble of forecasts
        forecasts = []
        forecast_weights = []
        
        # 1. Statistical Models
        
        # Simple exponential smoothing model
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Try different smoothing settings
        best_es_rmse = float('inf')
        best_es_forecast = None
        
        for trend in [None, 'add', 'mul']:
            for seasonal in [None, 'add', 'mul']:
                try:
                    if seasonal and len(train) < 2:
                        continue
                        
                    model = ExponentialSmoothing(
                        train[demand_column],
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=12 if seasonal else None
                    )
                    
                    model_fit = model.fit()
                    forecast = model_fit.forecast(test_size)
                    
                    rmse = np.sqrt(np.mean((test[demand_column].values - forecast.values) ** 2))
                    
                    if rmse < best_es_rmse:
                        best_es_rmse = rmse
                        best_es_forecast = forecast
                except Exception as e:
                    logger.warning(f"Error fitting ES model with trend={trend}, seasonal={seasonal}: {e}")
                    continue
        
        if best_es_forecast is not None:
            forecasts.append(best_es_forecast)
            # Weight inversely proportional to RMSE
            forecast_weights.append(1.0 / max(best_es_rmse, 0.001))
        
        # 2. ARIMA model
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Try different ARIMA parameters
            best_arima_rmse = float('inf')
            best_arima_forecast = None
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(train[demand_column], order=(p, d, q))
                            model_fit = model.fit()
                            forecast = model_fit.forecast(steps=test_size)
                            
                            rmse = np.sqrt(np.mean((test[demand_column].values - forecast.values) ** 2))
                            
                            if rmse < best_arima_rmse:
                                best_arima_rmse = rmse
                                best_arima_forecast = forecast
                        except Exception as e:
                            continue
            
            if best_arima_forecast is not None:
                forecasts.append(best_arima_forecast)
                forecast_weights.append(1.0 / max(best_arima_rmse, 0.001))
                
        except Exception as e:
            logger.warning(f"Error with ARIMA models: {e}")
        
        # 3. Machine Learning Models
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for ML (exclude the target variable)
            feature_cols = [col for col in train.columns if col != demand_column]
            
            if feature_cols:  # Only proceed if we have features
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train[feature_cols])
                X_test = scaler.transform(test[feature_cols])
                
                # Random Forest model
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, train[demand_column])
                
                rf_forecast = rf.predict(X_test)
                rf_rmse = np.sqrt(np.mean((test[demand_column].values - rf_forecast) ** 2))
                
                forecasts.append(pd.Series(rf_forecast, index=test.index))
                forecast_weights.append(1.0 / max(rf_rmse, 0.001))
        except Exception as e:
            logger.warning(f"Error with ML models: {e}")
        
        # 4. Naive seasonal model (as baseline)
        try:
            if len(train) >= 12:  # Need at least one year of data for seasonality
                # Use same month from previous year
                seasonal_naive = []
                for test_date in test.index:
                    # Find same month/day from previous year
                    if test_date.replace(year=test_date.year-1) in train.index:
                        seasonal_naive.append(train.loc[test_date.replace(year=test_date.year-1), demand_column])
                    else:
                        # Fallback to previous period
                        seasonal_naive.append(train[demand_column].iloc[-1])
                
                naive_forecast = pd.Series(seasonal_naive, index=test.index)
                naive_rmse = np.sqrt(np.mean((test[demand_column].values - naive_forecast.values) ** 2))
                
                forecasts.append(naive_forecast)
                forecast_weights.append(1.0 / max(naive_rmse, 0.001))
        except Exception as e:
            logger.warning(f"Error with naive seasonal model: {e}")
        
        # Combine forecasts if we have multiple models
        if len(forecasts) > 0:
            # Normalize weights
            total_weight = sum(forecast_weights)
            if total_weight > 0:
                forecast_weights = [w / total_weight for w in forecast_weights]
            else:
                forecast_weights = [1.0 / len(forecasts)] * len(forecasts)
            
            # Weighted ensemble forecast
            ensemble_forecast = sum(f * w for f, w in zip(forecasts, forecast_weights))
            
            # Calculate performance metrics
            rmse = np.sqrt(np.mean((test[demand_column].values - ensemble_forecast.values) ** 2))
            mape = np.mean(np.abs((test[demand_column].values - ensemble_forecast.values) / test[demand_column].values)) * 100
            mae = np.mean(np.abs(test[demand_column].values - ensemble_forecast.values))
            
            # Prepare the results
            results = {
                'product_id': product_id,
                'model_type': 'Ensemble',
                'rmse': rmse,
                'mape': mape,
                'mae': mae,
                'forecast_values': ensemble_forecast.tolist(),
                'model_weights': {f"model_{i}": w for i, w in enumerate(forecast_weights)}
            }
            
            logger.info(f"Advanced forecasting results for product {product_id}: RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            
            # Save the forecast to a file
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                result_file = os.path.join(output_dir, f"advanced_forecast_{product_id}.txt")
                
                with open(result_file, 'w') as f:
                    f.write(f"Product ID: {product_id}\n")
                    f.write(f"Model: Ensemble\n")
                    f.write(f"Model Weights: {dict(zip([f'Model_{i}' for i in range(len(forecast_weights))], forecast_weights))}\n")
                    f.write(f"RMSE: {rmse:.2f}\n")
                    f.write(f"MAPE: {mape:.2f}%\n")
                    f.write(f"MAE: {mae:.2f}\n\n")
                    f.write("Forecast Values:\n")
                    
                    for date, value in zip(test.index, ensemble_forecast):
                        f.write(f"{date.strftime('%Y-%m-%d')}: {value:.2f}\n")
                        
                    f.write("\nActual Values:\n")
                    for date, value in zip(test.index, test[demand_column]):
                        f.write(f"{date.strftime('%Y-%m-%d')}: {value}\n")
                
                # Create visualization of the forecast
                try:
                    import matplotlib.pyplot as plt
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(train.index, train[demand_column], label='Training Data')
                    plt.plot(test.index, test[demand_column], label='Actual')
                    plt.plot(test.index, ensemble_forecast, label='Ensemble Forecast')
                    
                    # Plot individual forecasts if available
                    colors = ['red', 'green', 'orange', 'purple']
                    for i, forecast in enumerate(forecasts):
                        if i < len(colors):
                            plt.plot(test.index, forecast, '--', color=colors[i], alpha=0.5, 
                                    label=f'Model {i} (weight={forecast_weights[i]:.2f})')
                    
                    plt.title(f'Advanced Demand Forecast for Product {product_id}')
                    plt.xlabel('Date')
                    plt.ylabel(demand_column)
                    plt.legend()
                    plt.grid(True)
                    
                    viz_file = os.path.join(output_dir, f"forecast_viz_{product_id}.png")
                    plt.savefig(viz_file)
                    plt.close()
                    
                    logger.info(f"Forecast visualization saved to {viz_file}")
                except Exception as e:
                    logger.warning(f"Error creating visualization: {e}")
            
            return results
        else:
            logger.warning(f"No successful forecasts for product {product_id}")
            return None
        
    except Exception as e:
        logger.error(f"Error in advanced forecasting for product {product_id}: {e}")
        logger.error(traceback.format_exc())
        return None

def run_crew_process(config, inventory_data, demand_data, result_queue):
    """Run the CrewAI process in a separate process"""
    try:
        logger.info(f"Starting crew process with {len(inventory_data)} inventory records and {len(demand_data)} demand records")
        
        # Initialize agents
        agent_provider = InventoryAgents(
            model_name=config['model_name'], 
            ollama_base_url=config['ollama_base_url']
        )
        
        agents_dict = {
            "demand_analyst": agent_provider.create_demand_analyst(),
            "inventory_optimizer": agent_provider.create_inventory_optimizer(),
            "supply_chain_analyst": agent_provider.create_supply_chain_analyst(),
            "risk_analyst": agent_provider.create_risk_analyst()
        }
        
        # Initialize tasks
        task_provider = InventoryTasks(inventory_data=inventory_data, demand_data=demand_data)
        tasks_list = task_provider.get_all_tasks(agents_dict)
        
        # Create and run crew
        from crewai import Crew
        crew = Crew(
            agents=list(agents_dict.values()),
            tasks=tasks_list,
            verbose=2
        )
        
        # Set a timeout for crew execution
        result = None
        
        def run_with_timeout():
            nonlocal result
            try:
                result = crew.kickoff()
            except Exception as e:
                logger.error(f"Error in crew execution: {e}")
                result = f"Error: {str(e)}"
        
        # Run with timeout
        crew_thread = threading.Thread(target=run_with_timeout)
        crew_thread.daemon = True
        crew_thread.start()
        crew_thread.join(120)  # 2 minute timeout
        
        if crew_thread.is_alive():
            logger.warning("Crew execution timed out")
            result = "Execution timed out - using statistical forecasting instead"
        
        # Put result in queue
        result_queue.put(result)
        
    except Exception as e:
        logger.error(f"Error in crew process: {e}")
        logger.error(traceback.format_exc())
        result_queue.put(f"Error: {str(e)}")

def main():
    """Main execution flow for the Inventory Optimization System."""
    start_time = time.time()
    config = None
    
    try:
        # Parse arguments
        config = parse_args()
        logger.info(f"Starting inventory optimization with model={config['model_name']}")
        
        # Initialize database
        initialize_db()
        
        # Load data
        logger.info("Loading inventory and demand data...")
        # Load inventory data from the specified file
        inventory_data = load_data('inventory_data.csv')
        # Load demand data from a separate file
        demand_data = load_data('demand_data.csv')
        
        if inventory_data is None or inventory_data.empty or demand_data is None or demand_data.empty:
            logger.error("Failed to load required data")
            return "Error: Missing required data"
        
        logger.info(f"Successfully loaded {len(inventory_data)} inventory records and {len(demand_data)} demand records")
        
        # Create results directory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(config['output_dir'], timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # Try to run the CrewAI process with a timeout
        result_queue = multiprocessing.Queue()
        crew_process = multiprocessing.Process(
            target=run_crew_process,
            args=(config, inventory_data, demand_data, result_queue)
        )
        
        logger.info("Starting CrewAI process...")
        crew_process.start()
        
        # Wait for the maximum allowed time
        max_wait = config.get('timeout', 180)
        crew_process.join(max_wait)
        
        # Get result or terminate if taking too long
        crew_result = None
        if crew_process.is_alive():
            logger.warning(f"Crew process did not complete in {max_wait} seconds, terminating...")
            crew_process.terminate()
            crew_process.join(5)  # Give it 5 more seconds to clean up
            crew_result = f"Execution timed out after {max_wait} seconds. Using advanced forecasting instead."
        else:
            # Try to get the result from the queue
            try:
                if not result_queue.empty():
                    crew_result = result_queue.get(block=False)
                    logger.info("Successfully retrieved crew result from queue")
                else:
                    logger.warning("Crew process completed but no result in queue")
                    crew_result = "No result from CrewAI process. Using advanced forecasting instead."
            except Exception as e:
                logger.error(f"Error retrieving result from queue: {e}")
                crew_result = f"Error retrieving result: {str(e)}"
        
        # Now perform forecasting using both methods - traditional and advanced
        logger.info("Performing statistical and advanced forecasting evaluation...")
        
        # Create subdirectories for different analysis types
        traditional_dir = os.path.join(run_dir, "traditional")
        advanced_dir = os.path.join(run_dir, "advanced")
        cv_dir = os.path.join(run_dir, "cross_validation")
        inventory_dir = os.path.join(run_dir, "inventory_optimization")
        
        os.makedirs(traditional_dir, exist_ok=True)
        os.makedirs(advanced_dir, exist_ok=True)
        os.makedirs(cv_dir, exist_ok=True)
        os.makedirs(inventory_dir, exist_ok=True)
        
        # Run traditional forecasting
        logger.info("Running traditional forecasting...")
        trad_metrics = evaluate_demand_forecasting(demand_data, 'all', traditional_dir)
        logger.info("Traditional forecasting completed")
        
        # Run advanced ensemble forecasting
        logger.info("Running advanced ensemble forecasting...")
        adv_metrics = evaluate_advanced_forecasting(demand_data, 'all', advanced_dir)
        logger.info("Advanced forecasting completed")
        
        # Select the best performing forecast for inventory optimization
        best_forecast = None
        if adv_metrics and trad_metrics:
            if adv_metrics.get('rmse', float('inf')) < trad_metrics.get('rmse', float('inf')):
                best_forecast = adv_metrics
                logger.info("Using advanced forecasting results for inventory optimization")
            else:
                best_forecast = trad_metrics
                logger.info("Using traditional forecasting results for inventory optimization")
        elif adv_metrics:
            best_forecast = adv_metrics
        elif trad_metrics:
            best_forecast = trad_metrics
        
        # Run inventory optimization analysis
        logger.info("Running inventory optimization analysis...")
        inventory_recommendations = None
        if best_forecast:
            inventory_recommendations = generate_inventory_recommendations(
                inventory_data, 
                demand_data, 
                best_forecast, 
                inventory_dir
            )
            logger.info("Inventory optimization completed")
        
        # Run cross-validation to evaluate model stability
        logger.info("Running time series cross-validation...")
        cv_metrics = run_time_series_cross_validation(demand_data, cv_dir)
        logger.info("Cross-validation completed")
        
        # Combine all results for a comprehensive analysis
        final_result = "=== INVENTORY OPTIMIZATION SYSTEM ANALYSIS ===\n\n"
        final_result += f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        final_result += f"Analysis ID: {timestamp}\n\n"
        
        if crew_result:
            final_result += "=== Multi-Agent Analysis ===\n"
            final_result += str(crew_result) + "\n\n"
        
        final_result += "=== Forecasting Performance ===\n\n"
        
        # Traditional forecasting results
        final_result += "Traditional Statistical Forecasting:\n"
        if trad_metrics:
            for metric, value in trad_metrics.items():
                if isinstance(value, (int, float)):
                    final_result += f"  {metric}: {value:.4f}\n"
                else:
                    final_result += f"  {metric}: {value}\n"
        else:
            final_result += "  No performance metrics available.\n"
            
        # Advanced forecasting results    
        final_result += "\nAdvanced Ensemble Forecasting:\n"
        if adv_metrics:
            for metric, value in adv_metrics.items():
                if isinstance(value, (int, float)):
                    final_result += f"  {metric}: {value:.4f}\n"
                elif metric == 'model_weights' and isinstance(value, dict):
                    final_result += "  Model Weights:\n"
                    for model, weight in value.items():
                        final_result += f"    {model}: {weight:.4f}\n"
                else:
                    final_result += f"  {metric}: {value}\n"
        else:
            final_result += "  No performance metrics available.\n"
            
        # Cross-validation results    
        final_result += "\nTime Series Cross-Validation:\n"
        if cv_metrics:
            for metric, value in cv_metrics.items():
                if isinstance(value, (int, float)):
                    final_result += f"  {metric}: {value:.4f}\n"
                else:
                    final_result += f"  {metric}: {value}\n"
        else:
            final_result += "  No cross-validation metrics available.\n"
        
        # Inventory optimization summary
        final_result += "\n=== Inventory Optimization Recommendations ===\n\n"
        
        if inventory_recommendations:
            # Check if we have ABC analysis results
            if 'abc_analysis' in inventory_recommendations:
                final_result += "ABC Inventory Classification Summary:\n"
                
                for cls in ['A', 'B', 'C']:
                    if cls in inventory_recommendations.get('class_metrics', {}):
                        metrics = inventory_recommendations['class_metrics'][cls]
                        final_result += f"\nClass {cls}:\n"
                        final_result += f"  Items: {metrics['count']} ({metrics['percent_of_items']:.1f}% of products)\n"
                        final_result += f"  Value: {metrics['percent_of_value']:.1f}% of total inventory value\n"
                
                # Extract key strategies
                final_result += "\nInventory Management Strategy:\n"
                for cls in ['A_items', 'B_items', 'C_items']:
                    if cls in inventory_recommendations.get('abc_analysis', {}):
                        strategy = inventory_recommendations['abc_analysis'][cls]
                        class_letter = cls[0]
                        final_result += f"\n  Class {class_letter} Strategy:\n"
                        for item in strategy.get('strategy', [])[:2]:  # Show just top two strategies
                            final_result += f"    - {item}\n"
            else:
                # We have product-specific recommendations
                final_result += f"Product ID: {inventory_recommendations.get('product_id', 'Unknown')}\n"
                final_result += f"Current Inventory: {inventory_recommendations.get('current_inventory', 0):.0f} units\n"
                final_result += f"Days of Supply: {inventory_recommendations.get('days_of_supply', 0):.1f} days\n"
                
                if inventory_recommendations.get('need_to_order', False):
                    final_result += f"Recommendation: Order {inventory_recommendations.get('recommended_order', 0):.0f} units\n"
                
                if inventory_recommendations.get('stockout_risk', False):
                    final_result += "Warning: Potential stockout risk detected\n"
                
                final_result += "\nRecommended Actions:\n"
                for action in inventory_recommendations.get('action_plan', []):
                    final_result += f"  - {action}\n"
        else:
            final_result += "No inventory recommendations available.\n"
        
        # Create method comparison visualization
        try:
            create_method_comparison_chart(trad_metrics, adv_metrics, cv_metrics, run_dir)
            final_result += "\nMethod comparison chart saved to results directory.\n"
        except Exception as e:
            logger.error(f"Error creating method comparison chart: {e}")
        
        # Save comprehensive results
        results_file = os.path.join(run_dir, "comprehensive_results.txt")
        with open(results_file, "w") as f:
            f.write(final_result)
        
        logger.info(f"Comprehensive results saved to {results_file}")
        
        # Create an executive summary
        try:
            executive_summary = create_executive_summary(
                trad_metrics, 
                adv_metrics, 
                cv_metrics, 
                inventory_recommendations,
                run_dir
            )
            logger.info("Executive summary created")
        except Exception as e:
            logger.error(f"Error creating executive summary: {e}")
        
        # Also save summary to global tracking file
        try:
            summary_file = os.path.join(config['output_dir'], "summary.txt")
            with open(summary_file, "a") as f:
                f.write(f"{timestamp} - Execution time: {time.time() - start_time:.2f}s\n")
                
                # Get the best model between traditional and advanced
                best_model = "Traditional"
                best_rmse = float('inf')
                
                if trad_metrics and 'rmse' in trad_metrics:
                    best_rmse = trad_metrics['rmse']
                
                if adv_metrics and 'rmse' in adv_metrics and adv_metrics['rmse'] < best_rmse:
                    best_model = "Advanced Ensemble"
                    best_rmse = adv_metrics['rmse']
                
                f.write(f"  Best Model: {best_model}\n")
                f.write(f"  Best RMSE: {best_rmse:.4f}\n")
                
                if cv_metrics and 'avg_rmse' in cv_metrics:
                    f.write(f"  CV RMSE: {cv_metrics['avg_rmse']:.4f}\n")
                
                if inventory_recommendations:
                    if 'abc_analysis' in inventory_recommendations:
                        # For ABC analysis
                        f.write("  Inventory: ABC analysis performed\n")
                    else:
                        # For product-specific recommendations
                        need_to_order = inventory_recommendations.get('need_to_order', False)
                        f.write(f"  Inventory Action: {'Order' if need_to_order else 'No action needed'}\n")
                
                f.write("\n")
        except Exception as e:
            logger.error(f"Error writing summary: {e}")
        
        return final_result
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"
    finally:
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")

def create_executive_summary(trad_metrics, adv_metrics, cv_metrics, inventory_recommendations, output_dir):
    """
    Create an executive summary of the entire analysis.
    
    Args:
        trad_metrics (dict): Traditional forecasting metrics
        adv_metrics (dict): Advanced forecasting metrics
        cv_metrics (dict): Cross-validation metrics
        inventory_recommendations (dict): Inventory optimization recommendations
        output_dir (str): Directory to save the summary
        
    Returns:
        str: Path to the created summary file
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        # Determine best forecasting method
        best_model = "Traditional"
        if adv_metrics and trad_metrics:
            if adv_metrics.get('rmse', float('inf')) < trad_metrics.get('rmse', float('inf')):
                best_model = "Advanced Ensemble"
        
        # Get key metrics
        best_rmse = min(
            adv_metrics.get('rmse', float('inf')),
            trad_metrics.get('rmse', float('inf'))
        )
        
        best_mape = min(
            adv_metrics.get('mape', float('inf')),
            trad_metrics.get('mape', float('inf'))
        )
        
        # Create a dashboard-style executive summary
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # Title
        fig.suptitle('Inventory Optimization System - Executive Summary', fontsize=16, y=0.98)
        fig.text(0.5, 0.94, f"Generated on {datetime.now().strftime('%Y-%m-%d')}", 
                 ha='center', fontsize=12)
        
        # 1. Key Performance Indicators
        ax_kpi = fig.add_subplot(gs[0, :])
        ax_kpi.axis('off')
        ax_kpi.text(0.1, 0.8, f"Best Forecasting Method: {best_model}", fontsize=14)
        ax_kpi.text(0.1, 0.6, f"Forecast Error (RMSE): {best_rmse:.2f}", fontsize=14)
        ax_kpi.text(0.1, 0.4, f"Forecast Error (MAPE): {best_mape:.2f}%", fontsize=14)
        
        if cv_metrics:
            stability = cv_metrics.get('stability_score', 0)
            ax_kpi.text(0.6, 0.6, f"Model Stability: {stability:.2f}/1.00", fontsize=14)
        
        # 2. Forecasting Method Comparison
        ax_methods = fig.add_subplot(gs[1, 0:2])
        metrics = ['rmse', 'mape', 'mae']
        methods = []
        results = []
        
        if trad_metrics:
            methods.append('Traditional')
            results.append([trad_metrics.get(m, 0) for m in metrics])
            
        if adv_metrics:
            methods.append('Advanced')
            results.append([adv_metrics.get(m, 0) for m in metrics])
            
        if cv_metrics:
            methods.append('Cross-Val')
            results.append([cv_metrics.get(f'avg_{m}', 0) for m in metrics])
        
        if methods:
            results = np.array(results)
            x = np.arange(len(metrics))
            width = 0.2
            multiplier = 0
            
            for i, method in enumerate(methods):
                offset = width * multiplier
                rects = ax_methods.bar(x + offset, results[i], width, label=method)
                ax_methods.bar_label(rects, fmt='%.2f')
                multiplier += 1
                
            ax_methods.set_ylabel('Error Value')
            ax_methods.set_title('Forecasting Method Comparison')
            ax_methods.set_xticks(x + width, metrics)
            ax_methods.legend(loc='best')
            ax_methods.set_ylim(0, np.max(results) * 1.2)
        
        # 3. Inventory Management Summary
        ax_inv = fig.add_subplot(gs[1, 2])
        ax_inv.axis('off')
        
        if inventory_recommendations:
            if 'abc_analysis' in inventory_recommendations:
                # ABC analysis summary
                ax_inv.text(0.5, 0.9, "Inventory Classification", fontsize=14, ha='center')
                
                class_metrics = inventory_recommendations.get('class_metrics', {})
                if class_metrics:
                    y_pos = 0.75
                    for cls in ['A', 'B', 'C']:
                        if cls in class_metrics:
                            metrics = class_metrics[cls]
                            ax_inv.text(0.1, y_pos, f"Class {cls}: {metrics['count']} items", fontsize=12)
                            ax_inv.text(0.1, y_pos-0.1, f"   {metrics['percent_of_value']:.1f}% of value", fontsize=12)
                            y_pos -= 0.25
            else:
                # Product-specific summary
                ax_inv.text(0.5, 0.9, "Inventory Recommendations", fontsize=14, ha='center')
                if inventory_recommendations.get('need_to_order', False):
                    ax_inv.text(0.1, 0.7, "Action: Place Order", fontsize=12)
                    ax_inv.text(0.1, 0.6, f"Amount: {inventory_recommendations.get('recommended_order', 0):.0f} units", fontsize=12)
                else:
                    ax_inv.text(0.1, 0.7, "Action: No Order Needed", fontsize=12)
                
                dos = inventory_recommendations.get('days_of_supply', 0)
                ax_inv.text(0.1, 0.5, f"Days of Supply: {dos:.1f}", fontsize=12)
                
                if inventory_recommendations.get('stockout_risk', False):
                    ax_inv.text(0.1, 0.4, "Warning: Stockout Risk", fontsize=12, color='red')
        else:
            ax_inv.text(0.1, 0.5, "No inventory recommendations available", fontsize=12)
        
        # 4. Key Actions
        ax_actions = fig.add_subplot(gs[2, :])
        ax_actions.axis('off')
        ax_actions.text(0.5, 0.9, "Recommended Actions", fontsize=14, ha='center')
        
        actions = []
        
        # Add actions based on forecasting
        if best_model == "Advanced Ensemble":
            actions.append("Implement ensemble forecasting for improved accuracy")
        
        if cv_metrics and cv_metrics.get('stability_score', 0) < 0.7:
            actions.append("Monitor forecast stability - high variability detected")
        
        # Add actions based on inventory
        if inventory_recommendations:
            if 'abc_analysis' in inventory_recommendations:
                actions.append("Implement differentiated inventory management by ABC class")
                
                # Get top Class A products
                product_class = inventory_recommendations.get('product_classifications', [])
                class_a_products = [p for p in product_class if p.get('class') == 'A']
                if class_a_products:
                    top_product = max(class_a_products, key=lambda x: x.get('annual_value', 0))
                    actions.append(f"Focus on Product {top_product.get('product_id')} (highest value Class A item)")
            else:
                if inventory_recommendations.get('need_to_order', False):
                    actions.append(f"Order {inventory_recommendations.get('recommended_order', 0):.0f} units of "
                                 f"Product {inventory_recommendations.get('product_id')}")
                
                if inventory_recommendations.get('stockout_risk', False):
                    actions.append("Urgent: Address potential stockout risk")
        
        # Adding custom actions based on analysis
        if best_rmse > 100:
            actions.append("Investigate data quality issues - forecast error is high")
        
        # Display actions
        y_pos = 0.8
        for i, action in enumerate(actions[:5]):  # Show top 5 actions
            ax_actions.text(0.1, y_pos, f"{i+1}. {action}", fontsize=12)
            y_pos -= 0.12
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust for title
        
        # Save the executive summary
        exec_summary_file = os.path.join(output_dir, "executive_summary.png")
        plt.savefig(exec_summary_file, dpi=120)
        plt.close()
        
        logger.info(f"Executive summary saved to {exec_summary_file}")
        
        # Create text version of executive summary
        text_summary = "EXECUTIVE SUMMARY - INVENTORY OPTIMIZATION SYSTEM\n"
        text_summary += "===================================================\n\n"
        text_summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        text_summary += "KEY PERFORMANCE INDICATORS:\n"
        text_summary += f"- Best Forecasting Method: {best_model}\n"
        text_summary += f"- Forecast Error (RMSE): {best_rmse:.2f}\n"
        text_summary += f"- Forecast Error (MAPE): {best_mape:.2f}%\n"
        
        if cv_metrics:
            text_summary += f"- Model Stability: {cv_metrics.get('stability_score', 0):.2f}/1.00\n"
            
        text_summary += "\nRECOMMENDED ACTIONS:\n"
        for i, action in enumerate(actions):
            text_summary += f"{i+1}. {action}\n"
            
        text_summary += "\nSee comprehensive analysis for full details.\n"
        
        text_summary_file = os.path.join(output_dir, "executive_summary.txt")
        with open(text_summary_file, 'w') as f:
            f.write(text_summary)
            
        return exec_summary_file
        
    except Exception as e:
        logger.error(f"Error creating executive summary: {e}")
        logger.error(traceback.format_exc())
        return None

def run_time_series_cross_validation(demand_data, output_dir, n_splits=3, test_size=6):
    """
    Perform time series cross-validation to evaluate model stability.
    
    Args:
        demand_data (pd.DataFrame): The demand data
        output_dir (str): Directory to save results
        n_splits (int): Number of time series CV splits
        test_size (int): Size of each test set in periods
        
    Returns:
        dict: Cross-validation metrics
    """
    logger.info(f"Running time series cross-validation with {n_splits} splits")
    
    # Track metrics across folds
    all_rmse = []
    all_mape = []
    all_mae = []
    
    # Get unique products
    product_ids = demand_data['Product ID'].unique()
    
    # Limit to a few products to save time
    if len(product_ids) > 5:
        # Use 5 random products for cross-validation
        import random
        random.seed(42)  # For reproducibility
        cv_products = random.sample(list(product_ids), 5)
    else:
        cv_products = product_ids
        
    logger.info(f"Cross-validation on {len(cv_products)} products")
    
    # Determine which column name is used for demand
    demand_column = 'Sales Quantity'
    if demand_column not in demand_data.columns and 'Demand' in demand_data.columns:
        demand_column = 'Demand'
        
    for product_id in cv_products:
        # Extract the demand series for this product
        product_demand = demand_data[demand_data['Product ID'] == product_id][['Date', demand_column]].copy()
        
        if product_demand.empty or len(product_demand) <= n_splits * test_size:
            # Not enough data for cross-validation
            continue
            
        # Convert to time series format
        product_demand['Date'] = pd.to_datetime(product_demand['Date'])
        product_demand.set_index('Date', inplace=True)
        product_demand.sort_index(inplace=True)
        
        # Create time series splits
        total_periods = len(product_demand)
        fold_metrics = []
        
        for i in range(n_splits):
            # Calculate split points working backwards from the end
            test_end = total_periods - i * test_size
            test_start = test_end - test_size
            
            if test_start <= 0:  # Not enough data for this split
                break
                
            # Split into train and test
            train = product_demand.iloc[:test_start]
            test = product_demand.iloc[test_start:test_end]
            
            if len(train) < 2*test_size:  # Need minimum amount of training data
                continue
                
            # Run exponential smoothing on this fold
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
                # Simple model with optimal parameters
                model = ExponentialSmoothing(
                    train[demand_column],
                    trend='add',  # Add trend component
                    seasonal=None  # No seasonal component for simplicity in CV
                )
                
                model_fit = model.fit()
                forecast = model_fit.forecast(test_size)
                
                # Calculate metrics for this fold
                fold_rmse = np.sqrt(np.mean((test[demand_column].values - forecast.values) ** 2))
                fold_mape = np.mean(np.abs((test[demand_column].values - forecast.values) / test[demand_column].values)) * 100
                fold_mae = np.mean(np.abs(test[demand_column].values - forecast.values))
                
                fold_metrics.append({
                    'fold': i+1,
                    'product_id': product_id,
                    'rmse': fold_rmse,
                    'mape': fold_mape,
                    'mae': fold_mae
                })
                
                logger.info(f"CV Fold {i+1} for product {product_id}: RMSE={fold_rmse:.2f}")
                
                # Create visualization for this fold
                try:
                    import matplotlib.pyplot as plt
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(train.index[-12:], train[demand_column].iloc[-12:], label='Training')
                    plt.plot(test.index, test[demand_column], label='Actual')
                    plt.plot(test.index, forecast, label='Forecast', linestyle='--')
                    plt.title(f'CV Fold {i+1} for Product {product_id}')
                    plt.legend()
                    
                    fold_viz_file = os.path.join(output_dir, f"cv_fold{i+1}_product{product_id}.png")
                    plt.savefig(fold_viz_file)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Error creating CV visualization: {e}")
                
            except Exception as e:
                logger.warning(f"Error in CV fold {i+1} for product {product_id}: {e}")
        
        # Collect metrics for this product across folds
        if fold_metrics:
            product_rmse = [fold['rmse'] for fold in fold_metrics]
            product_mape = [fold['mape'] for fold in fold_metrics]
            product_mae = [fold['mae'] for fold in fold_metrics]
            
            all_rmse.extend(product_rmse)
            all_mape.extend(product_mape)
            all_mae.extend(product_mae)
            
            # Calculate stability metrics (standard deviation across folds)
            rmse_std = np.std(product_rmse)
            mape_std = np.std(product_mape)
            
            logger.info(f"CV results for product {product_id}: " +
                       f"Avg RMSE={np.mean(product_rmse):.2f}, RMSE Std={rmse_std:.2f}")
    
    # Aggregate metrics across all products and folds
    if all_rmse:
        avg_rmse = np.mean(all_rmse)
        avg_mape = np.mean(all_mape)
        avg_mae = np.mean(all_mae)
        
        # Calculate stability metrics
        rmse_std = np.std(all_rmse)
        
        metrics = {
            'avg_rmse': avg_rmse,
            'avg_mape': avg_mape,
            'avg_mae': avg_mae,
            'rmse_std': rmse_std,
            'n_folds': len(all_rmse),
            'stability_score': 1.0 / (1.0 + rmse_std/avg_rmse)  # Higher = more stable
        }
        
        # Save CV results to a file
        cv_results_file = os.path.join(output_dir, "cv_results.txt")
        with open(cv_results_file, 'w') as f:
            f.write("Time Series Cross-Validation Results\n")
            f.write("=====================================\n\n")
            f.write(f"Number of products: {len(cv_products)}\n")
            f.write(f"Number of splits: {n_splits}\n")
            f.write(f"Test size: {test_size}\n\n")
            f.write(f"Average RMSE: {avg_rmse:.4f}\n")
            f.write(f"Average MAPE: {avg_mape:.4f}%\n")
            f.write(f"Average MAE: {avg_mae:.4f}\n")
            f.write(f"RMSE Std Dev: {rmse_std:.4f}\n")
            f.write(f"Stability Score: {metrics['stability_score']:.4f}\n")
        
        logger.info(f"CV results saved to {cv_results_file}")
        return metrics
    else:
        logger.warning("No successful CV folds")
        return None

def create_method_comparison_chart(trad_metrics, adv_metrics, cv_metrics, output_dir):
    """Create a comparison chart of forecasting methods"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract metrics for comparison
        metrics = ['rmse', 'mape', 'mae']
        methods = []
        results = []
        
        if trad_metrics:
            methods.append('Traditional')
            results.append([trad_metrics.get(m, 0) for m in metrics])
            
        if adv_metrics:
            methods.append('Advanced')
            results.append([adv_metrics.get(m, 0) for m in metrics])
            
        if cv_metrics:
            methods.append('Cross-Val')
            results.append([cv_metrics.get(f'avg_{m}', 0) for m in metrics])
            
        if not methods:
            logger.warning("No metrics available for comparison chart")
            return
            
        results = np.array(results)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.2
        multiplier = 0
        
        for i, method in enumerate(methods):
            offset = width * multiplier
            rects = ax.bar(x + offset, results[i], width, label=method)
            ax.bar_label(rects, fmt='%.2f')
            multiplier += 1
            
        ax.set_ylabel('Error Value')
        ax.set_title('Forecasting Method Comparison')
        ax.set_xticks(x + width, metrics)
        ax.legend(loc='best')
        ax.set_ylim(0, np.max(results) * 1.2)
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, "method_comparison.png")
        plt.savefig(chart_file)
        plt.close()
        
        logger.info(f"Method comparison chart saved to {chart_file}")
    except Exception as e:
        logger.error(f"Error creating comparison chart: {e}")

def generate_inventory_recommendations(inventory_data, demand_data, forecast_results, output_dir):
    """
    Generate inventory optimization recommendations based on forecasting results.
    
    Args:
        inventory_data (pd.DataFrame): Current inventory data
        demand_data (pd.DataFrame): Historical demand data
        forecast_results (dict): Results from the forecasting process
        output_dir (str): Directory to save results
        
    Returns:
        dict: Inventory optimization recommendations
    """
    try:
        logger.info("Generating inventory optimization recommendations")
        
        # Ensure we have the necessary data
        if inventory_data is None or demand_data is None or forecast_results is None:
            logger.error("Missing required data for generating recommendations")
            return None
            
        # Extract forecast values if available
        forecast_values = forecast_results.get('forecast_values', None)
        product_id = forecast_results.get('product_id', 'all')
        
        if product_id == 'all' or not forecast_values:
            # If we have aggregate results, analyze the overall inventory situation
            return generate_overall_inventory_strategy(inventory_data, demand_data, output_dir)
            
        # Get current inventory levels for this product
        current_inventory = inventory_data[inventory_data['Product ID'] == product_id]
        
        if current_inventory.empty:
            logger.warning(f"No inventory data found for product {product_id}")
            return None
            
        # Calculate inventory metrics
        try:
            # Determine which column name is used for inventory and demand
            inventory_column = 'Current Stock'
            if inventory_column not in current_inventory.columns:
                inventory_column = current_inventory.columns[current_inventory.columns.str.contains('inventory|stock', case=False)][0]
                
            demand_column = 'Sales Quantity'
            if demand_column not in demand_data.columns and 'Demand' in demand_data.columns:
                demand_column = 'Demand'
                
            # Get latest inventory level
            latest_inventory = current_inventory[inventory_column].iloc[0]
            
            # Get average demand for this product
            product_demand = demand_data[demand_data['Product ID'] == product_id][demand_column]
            avg_demand = product_demand.mean() if not product_demand.empty else 0
            
            # Calculate days of supply
            days_of_supply = 0 if avg_demand == 0 else latest_inventory / avg_demand
            
            # Calculate variability in demand
            demand_std = product_demand.std() if not product_demand.empty else 0
            coefficient_of_variation = 0 if avg_demand == 0 else demand_std / avg_demand
            
            # Extract lead time information if available
            lead_time_column = next((col for col in current_inventory.columns if 'lead' in col.lower()), None)
            lead_time = current_inventory[lead_time_column].iloc[0] if lead_time_column else 14  # Default to 14 days
            
            # Calculate safety stock based on service level and demand variability
            service_level_z = 1.645  # 95% service level
            safety_stock = service_level_z * demand_std * np.sqrt(lead_time)
            
            # Calculate reorder point
            reorder_point = (avg_demand * lead_time) + safety_stock
            
            # Calculate economic order quantity (EOQ)
            annual_demand = avg_demand * 365  # Assuming daily demand
            holding_cost_percent = 0.25  # Assuming 25% of item value as annual holding cost
            item_cost_column = next((col for col in current_inventory.columns if 'cost' in col.lower() or 'price' in col.lower()), None)
            item_cost = current_inventory[item_cost_column].iloc[0] if item_cost_column else 100  # Default value
            ordering_cost = 50  # Assumed fixed cost per order
            
            # EOQ formula
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost_percent * item_cost))
            
            # Determine if we need to order and how much
            need_to_order = latest_inventory <= reorder_point
            order_amount = max(0, eoq) if need_to_order else 0
            
            # Calculate projected stock levels using forecast
            projected_stock = []
            stock_level = latest_inventory
            
            for i, forecast in enumerate(forecast_values):
                # Simulate ordering if needed
                if i == 0 and need_to_order:
                    stock_level += order_amount
                
                # Subtract forecasted demand (with a floor at 0)
                stock_level = max(0, stock_level - forecast)
                projected_stock.append(stock_level)
                
            # Identify potential stockout periods
            stockout_risk = [i for i, level in enumerate(projected_stock) if level <= safety_stock]
            high_risk = len(stockout_risk) > 0
            
            # Prepare recommendations
            recommendations = {
                'product_id': product_id,
                'current_inventory': latest_inventory,
                'avg_daily_demand': avg_demand,
                'demand_variability': coefficient_of_variation,
                'days_of_supply': days_of_supply,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'economic_order_quantity': eoq,
                'need_to_order': need_to_order,
                'recommended_order': order_amount,
                'stockout_risk': high_risk,
                'potential_stockout_periods': stockout_risk
            }
            
            # Generate action plan
            action_plan = []
            
            if need_to_order:
                action_plan.append(f"Place order for {order_amount:.0f} units immediately")
            elif latest_inventory <= (reorder_point * 1.2):
                action_plan.append(f"Monitor closely - approaching reorder point")
                
            if high_risk:
                action_plan.append(f"WARNING: Potential stockout risk in periods {stockout_risk}")
                action_plan.append(f"Consider expediting orders or increasing order quantity")
                
            if days_of_supply > 90:
                action_plan.append(f"Excess inventory: Consider reducing order quantities or promotions")
                
            recommendations['action_plan'] = action_plan
            
            # Save recommendations to file
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                rec_file = os.path.join(output_dir, f"inventory_rec_{product_id}.txt")
                
                with open(rec_file, 'w') as f:
                    f.write(f"Inventory Optimization Recommendations - Product {product_id}\n")
                    f.write("=========================================================\n\n")
                    f.write(f"Current Inventory: {latest_inventory:.0f} units\n")
                    f.write(f"Average Daily Demand: {avg_demand:.2f} units\n")
                    f.write(f"Days of Supply: {days_of_supply:.1f} days\n")
                    f.write(f"Demand Variability (CV): {coefficient_of_variation:.2f}\n")
                    f.write(f"Lead Time: {lead_time} days\n\n")
                    
                    f.write("Calculated Parameters:\n")
                    f.write(f"Safety Stock: {safety_stock:.0f} units\n")
                    f.write(f"Reorder Point: {reorder_point:.0f} units\n")
                    f.write(f"Economic Order Quantity: {eoq:.0f} units\n\n")
                    
                    f.write("Recommended Actions:\n")
                    for action in action_plan:
                        f.write(f"- {action}\n")
                    
                    f.write("\nProjected Inventory Levels:\n")
                    for i, level in enumerate(projected_stock):
                        f.write(f"Period {i+1}: {level:.0f} units\n")
                
                # Create visualization of projected inventory
                try:
                    import matplotlib.pyplot as plt
                    
                    periods = range(1, len(projected_stock) + 1)
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(periods, projected_stock, marker='o', label='Projected Inventory')
                    plt.axhline(y=safety_stock, color='r', linestyle='--', label='Safety Stock')
                    plt.axhline(y=reorder_point, color='g', linestyle='--', label='Reorder Point')
                    
                    plt.title(f'Projected Inventory Levels - Product {product_id}')
                    plt.xlabel('Forecast Period')
                    plt.ylabel('Inventory Units')
                    plt.grid(True)
                    plt.legend()
                    
                    viz_file = os.path.join(output_dir, f"inventory_projection_{product_id}.png")
                    plt.savefig(viz_file)
                    plt.close()
                    
                    logger.info(f"Inventory projection saved to {viz_file}")
                except Exception as e:
                    logger.warning(f"Error creating inventory visualization: {e}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error calculating inventory metrics for product {product_id}: {e}")
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Error generating inventory recommendations: {e}")
        logger.error(traceback.format_exc())
        return None

def generate_overall_inventory_strategy(inventory_data, demand_data, output_dir):
    """Generate overall inventory strategy based on ABC analysis"""
    try:
        logger.info("Generating overall inventory strategy with ABC analysis")
        
        # Determine column names
        inventory_column = 'Current Stock'
        if inventory_column not in inventory_data.columns:
            inventory_column = inventory_data.columns[inventory_data.columns.str.contains('inventory|stock', case=False)][0]
            
        value_column = 'Unit Cost'
        if value_column not in inventory_data.columns:
            value_column = inventory_data.columns[inventory_data.columns.str.contains('cost|price|value', case=False)][0]
        
        demand_column = 'Sales Quantity'
        if demand_column not in demand_data.columns and 'Demand' in demand_data.columns:
            demand_column = 'Demand'
            
        product_id_column = 'Product ID'
        
        # Calculate the total value of each product
        abc_data = []
        
        for product_id in inventory_data[product_id_column].unique():
            product_inventory = inventory_data[inventory_data[product_id_column] == product_id]
            product_demand = demand_data[demand_data[product_id_column] == product_id]
            
            if product_inventory.empty or product_demand.empty:
                continue
                
            # Get inventory level and unit cost
            inventory_level = product_inventory[inventory_column].iloc[0]
            unit_cost = product_inventory[value_column].iloc[0] if value_column in product_inventory.columns else 100
            
            # Calculate average demand
            avg_demand = product_demand[demand_column].mean()
            
            # Calculate annual usage value
            annual_usage = avg_demand * 365  # Assuming daily demand
            annual_value = annual_usage * unit_cost
            
            abc_data.append({
                'product_id': product_id,
                'inventory_level': inventory_level,
                'unit_cost': unit_cost,
                'avg_demand': avg_demand,
                'annual_value': annual_value
            })
        
        if not abc_data:
            logger.warning("No valid data for ABC analysis")
            return None
            
        # Convert to DataFrame and sort by annual value
        abc_df = pd.DataFrame(abc_data)
        abc_df = abc_df.sort_values('annual_value', ascending=False)
        
        # Calculate cumulative percentage
        abc_df['cum_value'] = abc_df['annual_value'].cumsum()
        abc_df['value_percent'] = abc_df['cum_value'] / abc_df['annual_value'].sum() * 100
        
        # Assign ABC classes
        abc_df['class'] = 'C'
        abc_df.loc[abc_df['value_percent'] <= 80, 'class'] = 'A'
        abc_df.loc[(abc_df['value_percent'] > 80) & (abc_df['value_percent'] <= 95), 'class'] = 'B'
        
        # Count items in each class
        class_counts = abc_df['class'].value_counts().to_dict()
        
        # Calculate metrics by class
        class_metrics = {}
        for cls in ['A', 'B', 'C']:
            class_data = abc_df[abc_df['class'] == cls]
            class_metrics[cls] = {
                'count': len(class_data),
                'percent_of_items': len(class_data) / len(abc_df) * 100,
                'percent_of_value': class_data['annual_value'].sum() / abc_df['annual_value'].sum() * 100,
                'avg_unit_cost': class_data['unit_cost'].mean(),
                'total_value': class_data['annual_value'].sum()
            }
        
        # Generate inventory strategy recommendations
        inventory_strategy = {
            'A_items': {
                'item_count': class_metrics['A']['count'],
                'value_percent': class_metrics['A']['percent_of_value'],
                'strategy': [
                    "Tight control with frequent reviews",
                    "Low safety stock, frequent ordering",
                    "Use precise forecasting methods",
                    "High service level (98-99%)",
                    "Consider JIT approach where possible"
                ]
            },
            'B_items': {
                'item_count': class_metrics['B']['count'],
                'value_percent': class_metrics['B']['percent_of_value'],
                'strategy': [
                    "Moderate control, periodic reviews",
                    "Medium safety stock",
                    "Standard forecasting approaches",
                    "Medium service level (95%)",
                    "Balance between service and inventory cost"
                ]
            },
            'C_items': {
                'item_count': class_metrics['C']['count'],
                'value_percent': class_metrics['C']['percent_of_value'],
                'strategy': [
                    "Loose control, bulk ordering",
                    "Higher safety stock to prevent stockouts",
                    "Simpler forecasting (moving average)",
                    "Lower service level (90%)",
                    "Minimize ordering and handling costs"
                ]
            }
        }
        
        # Save results to file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            abc_file = os.path.join(output_dir, "abc_analysis.txt")
            
            with open(abc_file, 'w') as f:
                f.write("ABC Inventory Analysis Results\n")
                f.write("=============================\n\n")
                
                f.write("Summary:\n")
                f.write(f"Total products analyzed: {len(abc_df)}\n")
                f.write(f"Total annual inventory value: ${abc_df['annual_value'].sum():,.2f}\n\n")
                
                f.write("Class A (High Value):\n")
                f.write(f"  Items: {class_metrics['A']['count']} ({class_metrics['A']['percent_of_items']:.1f}% of total)\n")
                f.write(f"  Value: ${class_metrics['A']['total_value']:,.2f} ({class_metrics['A']['percent_of_value']:.1f}% of total)\n")
                f.write(f"  Avg Unit Cost: ${class_metrics['A']['avg_unit_cost']:,.2f}\n")
                f.write("  Recommended Strategy:\n")
                for strat in inventory_strategy['A_items']['strategy']:
                    f.write(f"    - {strat}\n")
                    
                f.write("\nClass B (Medium Value):\n")
                f.write(f"  Items: {class_metrics['B']['count']} ({class_metrics['B']['percent_of_items']:.1f}% of total)\n")
                f.write(f"  Value: ${class_metrics['B']['total_value']:,.2f} ({class_metrics['B']['percent_of_value']:.1f}% of total)\n")
                f.write(f"  Avg Unit Cost: ${class_metrics['B']['avg_unit_cost']:,.2f}\n")
                f.write("  Recommended Strategy:\n")
                for strat in inventory_strategy['B_items']['strategy']:
                    f.write(f"    - {strat}\n")
                    
                f.write("\nClass C (Low Value):\n")
                f.write(f"  Items: {class_metrics['C']['count']} ({class_metrics['C']['percent_of_items']:.1f}% of total)\n")
                f.write(f"  Value: ${class_metrics['C']['total_value']:,.2f} ({class_metrics['C']['percent_of_value']:.1f}% of total)\n")
                f.write(f"  Avg Unit Cost: ${class_metrics['C']['avg_unit_cost']:,.2f}\n")
                f.write("  Recommended Strategy:\n")
                for strat in inventory_strategy['C_items']['strategy']:
                    f.write(f"    - {strat}\n")
                
                f.write("\n\nDetailed Product Classification:\n")
                f.write("-------------------------------------------------\n")
                f.write("Product ID | Class | Annual Value ($) | % of Total\n")
                f.write("-------------------------------------------------\n")
                
                # Write details for each product
                total_value = abc_df['annual_value'].sum()
                for _, row in abc_df.iterrows():
                    f.write(f"{row['product_id']} | {row['class']} | ${row['annual_value']:,.2f} | {row['annual_value']/total_value*100:.1f}%\n")
            
            # Create Pareto/ABC chart
            try:
                import matplotlib.pyplot as plt
                
                # Prepare data for plotting
                items = range(1, len(abc_df) + 1)
                values = abc_df['value_percent'].values
                
                # Create figure with two y-axes
                fig, ax1 = plt.subplots(figsize=(12, 8))
                
                # Plot cumulative percentage curve
                ax1.plot(items, values, 'b-', marker='o', markersize=5)
                ax1.set_xlabel('Number of Products')
                ax1.set_ylabel('Cumulative Percentage of Value', color='b')
                ax1.tick_params('y', colors='b')
                
                # Add reference lines for ABC classification
                ax1.axhline(y=80, color='r', linestyle='--', label='A-B Threshold (80%)')
                ax1.axhline(y=95, color='g', linestyle='--', label='B-C Threshold (95%)')
                
                # Add the number of items in each class
                a_items = class_metrics['A']['count']
                b_items = class_metrics['B']['count']
                
                ax1.axvline(x=a_items, color='r', linestyle=':')
                ax1.axvline(x=a_items+b_items, color='g', linestyle=':')
                
                # Add class labels
                mid_a = a_items / 2
                mid_b = a_items + b_items / 2
                mid_c = a_items + b_items + class_metrics['C']['count'] / 2
                
                ax1.text(mid_a, 40, 'Class A', fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
                ax1.text(mid_b, 40, 'Class B', fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
                ax1.text(mid_c, 40, 'Class C', fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
                
                plt.title('ABC Analysis - Pareto Chart', fontsize=16)
                plt.grid(True)
                plt.legend()
                
                # Save the plot
                abc_chart = os.path.join(output_dir, "abc_pareto_chart.png")
                plt.savefig(abc_chart)
                plt.close()
                
                logger.info(f"ABC analysis chart saved to {abc_chart}")
            except Exception as e:
                logger.warning(f"Error creating ABC chart: {e}")
        
        # Return the overall strategy
        return {
            'abc_analysis': inventory_strategy,
            'class_metrics': class_metrics,
            'product_classifications': abc_df[['product_id', 'class', 'annual_value']].to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error in ABC analysis: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Set a global timeout to ensure the script eventually finishes
    def global_timeout_handler(signum, frame):
        logger.error("Global timeout reached, terminating...")
        sys.exit(1)
    
    signal.signal(signal.SIGALRM, global_timeout_handler)
    signal.alarm(300)  # 5-minute hard timeout
    
    try:
        multiprocessing.set_start_method('spawn', force=True)
        result = main()
        print("\n=== EXECUTION COMPLETED ===")
        if result:
            print("\nRESULTS SUMMARY:")
            print(result if len(result) < 1500 else result[:1500] + "...(truncated)")
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        signal.alarm(0)  # Disable alarm
        sys.exit(0) 