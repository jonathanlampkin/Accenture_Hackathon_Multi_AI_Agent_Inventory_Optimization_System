"""
Forecasting Models Integration with AI Agents

This module integrates the statistical forecasting models with the AI agent system,
providing tools and interfaces for agents to work with the models.
"""

from crewai_tools import BaseTool, tool
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import internal modules
from .forecasting_models import (
    ForecastModel, SARIMAModel, ProphetModel, 
    XGBoostForecastModel, LightGBMForecastModel, NeuralProphetModel
)
from .model_comparison import ModelComparison

logger = logging.getLogger(__name__)

# Global model registry to maintain state between tool calls
MODEL_REGISTRY = {}
TRAINED_MODELS = {}
COMPARISON_RESULTS = {}

@tool("train_forecast_model")
def train_forecast_model_tool(model_type: str, data_path: str, date_column: str, 
                             target_column: str, model_params: Optional[Dict] = None) -> str:
    """
    Trains a statistical forecasting model with the specified parameters.
    
    Args:
        model_type: The type of model to train ('sarima', 'prophet', 'xgboost', 'lightgbm', 'neuralprophet')
        data_path: Path to the CSV file containing the training data
        date_column: Name of the date/timestamp column in the CSV
        target_column: Name of the target column (e.g., sales, demand) in the CSV
        model_params: Dictionary of model-specific parameters (optional)
    
    Returns:
        A message indicating the result of the training process
    """
    try:
        # Load and validate data
        try:
            df = pd.read_csv(data_path)
            
            # Ensure date column is parsed correctly
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Sort by date
            df = df.sort_values(by=date_column)
            
            # Check for missing values in target column
            if df[target_column].isnull().any():
                logger.warning(f"Missing values detected in {target_column}. Filling with forward fill method.")
                df[target_column] = df[target_column].ffill()
            
            logger.info(f"Loaded data from {data_path}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return f"Error loading data: {str(e)}"
        
        # Initialize model parameters
        params = model_params or {}
        
        # Initialize the appropriate model
        model_type = model_type.lower()
        
        if model_type == 'sarima':
            order = params.get('order', (1, 1, 1))
            seasonal_order = params.get('seasonal_order', (1, 1, 1, 7))
            model = SARIMAModel(order=order, seasonal_order=seasonal_order)
        
        elif model_type == 'prophet':
            yearly_seasonality = params.get('yearly_seasonality', True)
            weekly_seasonality = params.get('weekly_seasonality', True)
            daily_seasonality = params.get('daily_seasonality', False)
            changepoint_prior_scale = params.get('changepoint_prior_scale', 0.05)
            model = ProphetModel(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                changepoint_prior_scale=changepoint_prior_scale
            )
        
        elif model_type == 'xgboost':
            max_lag = params.get('max_lag', 7)
            n_estimators = params.get('n_estimators', 100)
            learning_rate = params.get('learning_rate', 0.1)
            max_depth = params.get('max_depth', 5)
            model = XGBoostForecastModel(
                max_lag=max_lag,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth
            )
        
        elif model_type == 'lightgbm':
            max_lag = params.get('max_lag', 7)
            n_estimators = params.get('n_estimators', 100)
            learning_rate = params.get('learning_rate', 0.1)
            num_leaves = params.get('num_leaves', 31)
            model = LightGBMForecastModel(
                max_lag=max_lag,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves
            )
        
        elif model_type == 'neuralprophet':
            n_changepoints = params.get('n_changepoints', 10)
            n_forecasts = params.get('n_forecasts', 1)
            yearly_seasonality = params.get('yearly_seasonality', True)
            weekly_seasonality = params.get('weekly_seasonality', True)
            model = NeuralProphetModel(
                n_changepoints=n_changepoints,
                n_forecasts=n_forecasts,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality
            )
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return f"Error: Unknown model type '{model_type}'. Supported types are: sarima, prophet, xgboost, lightgbm, neuralprophet"
        
        # Train the model
        logger.info(f"Training {model_type} model...")
        model.fit(df, target_column, date_column)
        
        # Store the model in the registry
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        TRAINED_MODELS[model_id] = {
            'model': model,
            'data_path': data_path,
            'date_column': date_column,
            'target_column': target_column,
            'trained_on': datetime.now().isoformat()
        }
        
        logger.info(f"Model trained successfully: {model_id}")
        return f"Model trained successfully. Model ID: {model_id}"
        
    except Exception as e:
        logger.error(f"Error training {model_type} model: {str(e)}")
        return f"Error training model: {str(e)}"

@tool("generate_forecast")
def generate_forecast_tool(model_id: str, horizon: int = 30, 
                          output_path: Optional[str] = None) -> str:
    """
    Generates forecasts using a trained model.
    
    Args:
        model_id: ID of the trained model to use
        horizon: Number of periods to forecast
        output_path: Path to save the forecast results (optional)
    
    Returns:
        A summary of the forecast results
    """
    try:
        # Retrieve the model from the registry
        if model_id not in TRAINED_MODELS:
            logger.error(f"Model not found: {model_id}")
            return f"Error: Model with ID '{model_id}' not found"
        
        model_info = TRAINED_MODELS[model_id]
        model = model_info['model']
        
        # Generate forecast
        logger.info(f"Generating forecast with {model.name} model (horizon={horizon})...")
        forecast_df = model.predict(horizon=horizon)
        
        # Save forecast to CSV if output path is provided
        if output_path:
            forecast_df.to_csv(output_path, index=False)
            logger.info(f"Forecast saved to {output_path}")
        
        # Generate summary statistics
        forecast_summary = {
            'model_type': model.name,
            'horizon': horizon,
            'forecast_start_date': forecast_df['ds'].min().strftime('%Y-%m-%d'),
            'forecast_end_date': forecast_df['ds'].max().strftime('%Y-%m-%d'),
            'average_forecast': forecast_df['yhat'].mean(),
            'min_forecast': forecast_df['yhat'].min(),
            'max_forecast': forecast_df['yhat'].max()
        }
        
        # Store the forecast in the model info for later access
        model_info['latest_forecast'] = {
            'dataframe': forecast_df,
            'summary': forecast_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate a summary message with the first few and last few forecasted values
        sample_size = min(5, horizon)
        first_rows = forecast_df.head(sample_size)
        last_rows = forecast_df.tail(sample_size) if horizon > sample_size * 2 else pd.DataFrame()
        
        message = [
            f"Forecast generated successfully with {model.name} model:",
            f"- Horizon: {horizon} periods",
            f"- Forecast period: {forecast_summary['forecast_start_date']} to {forecast_summary['forecast_end_date']}",
            f"- Average forecast: {forecast_summary['average_forecast']:.2f}",
            f"- Range: {forecast_summary['min_forecast']:.2f} to {forecast_summary['max_forecast']:.2f}",
            "\nFirst few forecasted values:"
        ]
        
        for _, row in first_rows.iterrows():
            message.append(f"- {row['ds'].strftime('%Y-%m-%d')}: {row['yhat']:.2f}")
        
        if not last_rows.empty:
            message.append("\nLast few forecasted values:")
            for _, row in last_rows.iterrows():
                message.append(f"- {row['ds'].strftime('%Y-%m-%d')}: {row['yhat']:.2f}")
        
        if output_path:
            message.append(f"\nFull forecast saved to: {output_path}")
        
        return "\n".join(message)
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return f"Error generating forecast: {str(e)}"

@tool("compare_forecast_models")
def compare_forecast_models_tool(data_path: str, date_column: str, target_column: str,
                               test_size: float = 0.2, split_date: Optional[str] = None,
                               models_to_compare: Optional[List[str]] = None,
                               output_dir: Optional[str] = None) -> str:
    """
    Compares multiple forecasting models on the same dataset.
    
    Args:
        data_path: Path to the CSV file containing the data
        date_column: Name of the date/timestamp column in the CSV
        target_column: Name of the target column (e.g., sales, demand) in the CSV
        test_size: Proportion of data to use for testing (default: 0.2)
        split_date: Date to split the data (format: 'YYYY-MM-DD')
        models_to_compare: List of model types to compare (default: all available models)
        output_dir: Directory to save comparison results (optional)
    
    Returns:
        A summary of the comparison results
    """
    try:
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'output/model_comparison_{timestamp}'
        
        # Load and validate data
        try:
            df = pd.read_csv(data_path)
            
            # Ensure date column is parsed correctly
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Sort by date
            df = df.sort_values(by=date_column)
            
            # Check for missing values in target column
            if df[target_column].isnull().any():
                logger.warning(f"Missing values detected in {target_column}. Filling with forward fill method.")
                df[target_column] = df[target_column].ffill()
            
            logger.info(f"Loaded data from {data_path}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return f"Error loading data: {str(e)}"
        
        # Initialize model comparison
        model_comparison = ModelComparison(output_dir=output_dir)
        
        # Add models to compare
        all_models = ['sarima', 'prophet', 'xgboost', 'lightgbm', 'neuralprophet']
        models_to_use = models_to_compare or all_models
        
        # Validate model types
        for model_type in models_to_use:
            if model_type.lower() not in all_models:
                logger.warning(f"Unknown model type: {model_type}")
                return f"Error: Unknown model type '{model_type}'. Supported types are: {', '.join(all_models)}"
        
        # Add models based on specified types
        for model_type in models_to_use:
            model_type = model_type.lower()
            
            if model_type == 'sarima':
                model_comparison.add_model(SARIMAModel())
            elif model_type == 'prophet':
                model_comparison.add_model(ProphetModel())
            elif model_type == 'xgboost':
                model_comparison.add_model(XGBoostForecastModel())
            elif model_type == 'lightgbm':
                model_comparison.add_model(LightGBMForecastModel())
            elif model_type == 'neuralprophet':
                model_comparison.add_model(NeuralProphetModel())
        
        # Run comparison
        logger.info(f"Running comparison for models: {', '.join(models_to_use)}")
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = model_comparison.run_full_comparison(
            data=df,
            date_col=date_column,
            target_col=target_column,
            test_size=test_size,
            split_date=split_date
        )
        
        # Store comparison results
        COMPARISON_RESULTS[comparison_id] = {
            'comparison': model_comparison,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate summary message
        evaluation_df = results['evaluation']
        
        # Find best model for each metric
        best_models = {}
        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in evaluation_df.columns:
                best_model = evaluation_df.loc[evaluation_df[metric].idxmin()]
                best_models[metric] = {
                    'model': best_model['model'],
                    'value': best_model[metric]
                }
        
        # For R², higher is better
        if 'R²' in evaluation_df.columns:
            best_model = evaluation_df.loc[evaluation_df['R²'].idxmax()]
            best_models['R²'] = {
                'model': best_model['model'],
                'value': best_model['R²']
            }
        
        message = [
            f"Model comparison completed successfully. Comparison ID: {comparison_id}",
            f"Models compared: {', '.join(models_to_use)}",
            f"Results saved to: {output_dir}",
            "\nEvaluation results:"
        ]
        
        # Add table header
        message.append("\nModel      | MAE      | RMSE     | MAPE (%) | R²")
        message.append("----------|----------|----------|----------|----------")
        
        # Add each model's results
        for _, row in evaluation_df.iterrows():
            model_name = row['model']
            mae = f"{row['MAE']:.4f}" if not pd.isna(row['MAE']) else "N/A"
            rmse = f"{row['RMSE']:.4f}" if not pd.isna(row['RMSE']) else "N/A"
            mape = f"{row['MAPE']:.2f}" if not pd.isna(row['MAPE']) else "N/A"
            r2 = f"{row['R²']:.4f}" if not pd.isna(row['R²']) else "N/A"
            
            # Pad model name
            model_name_padded = model_name.ljust(10)
            
            message.append(f"{model_name_padded}| {mae.ljust(8)} | {rmse.ljust(8)} | {mape.ljust(8)} | {r2.ljust(8)}")
        
        message.append("\nBest models by metric:")
        for metric, info in best_models.items():
            message.append(f"- {metric}: {info['model']} ({info['value']:.4f})")
        
        message.append(f"\nVisualization plots saved to: {output_dir}")
        
        return "\n".join(message)
        
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return f"Error comparing models: {str(e)}"

@tool("visualize_forecast")
def visualize_forecast_tool(model_id: str, include_history: bool = True,
                          output_path: Optional[str] = None) -> str:
    """
    Visualizes the forecast from a trained model.
    
    Args:
        model_id: ID of the trained model to use
        include_history: Whether to include historical data in the plot
        output_path: Path to save the visualization (optional)
    
    Returns:
        A message indicating the result of the visualization process
    """
    try:
        # Retrieve the model from the registry
        if model_id not in TRAINED_MODELS:
            logger.error(f"Model not found: {model_id}")
            return f"Error: Model with ID '{model_id}' not found"
        
        model_info = TRAINED_MODELS[model_id]
        model = model_info['model']
        
        # Check if the model has a latest forecast
        if 'latest_forecast' not in model_info:
            # Generate a new forecast
            forecast_df = model.predict()
            logger.info(f"Generated new forecast for visualization")
        else:
            # Use the existing forecast
            forecast_df = model_info['latest_forecast']['dataframe']
            logger.info(f"Using existing forecast for visualization")
        
        # Load historical data if including history
        historical_data = None
        if include_history:
            try:
                # Load data from the original path
                df = pd.read_csv(model_info['data_path'])
                
                # Ensure date column is parsed correctly
                date_col = model_info['date_column']
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Sort by date
                historical_data = df.sort_values(by=date_col)
                
                logger.info(f"Loaded historical data for visualization: {len(historical_data)} rows")
            except Exception as e:
                logger.warning(f"Could not load historical data: {str(e)}")
                # Continue without historical data
                historical_data = None
        
        # Set up output path
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'output/{model.name}_forecast_{timestamp}.png'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data if available
        if historical_data is not None:
            ax.plot(historical_data[model_info['date_column']], 
                   historical_data[model_info['target_column']], 
                   label='Historical', color='blue', marker='o', markersize=3)
        
        # Plot forecast
        ax.plot(forecast_df['ds'], forecast_df['yhat'], 
               label='Forecast', color='red', linestyle='--')
        
        # Plot confidence intervals if available
        if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
            ax.fill_between(forecast_df['ds'], 
                           forecast_df['yhat_lower'], 
                           forecast_df['yhat_upper'], 
                           color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Customize plot
        ax.set_title(f'{model.name} Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel(model_info['target_column'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format date labels
        fig.autofmt_xdate()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Forecast visualization saved to {output_path}")
        
        return f"Forecast visualization created successfully and saved to {output_path}"
        
    except Exception as e:
        logger.error(f"Error visualizing forecast: {str(e)}")
        return f"Error visualizing forecast: {str(e)}"

@tool("get_best_forecast_model")
def get_best_forecast_model_tool(comparison_id: str, metric: str = 'RMSE') -> str:
    """
    Returns information about the best performing forecast model from a comparison.
    
    Args:
        comparison_id: ID of the model comparison to use
        metric: Metric to use for ranking (default: 'RMSE')
    
    Returns:
        Information about the best performing model
    """
    try:
        # Retrieve the comparison from the registry
        if comparison_id not in COMPARISON_RESULTS:
            logger.error(f"Comparison not found: {comparison_id}")
            return f"Error: Comparison with ID '{comparison_id}' not found"
        
        comparison_info = COMPARISON_RESULTS[comparison_id]
        evaluation_df = comparison_info['results']['evaluation']
        
        # Find best model based on metric
        if metric not in evaluation_df.columns:
            logger.error(f"Metric not found: {metric}")
            return f"Error: Metric '{metric}' not found in comparison results"
        
        # For R², higher is better
        if metric == 'R²':
            best_idx = evaluation_df[metric].idxmax()
        else:
            # For other metrics (MAE, RMSE, MAPE), lower is better
            best_idx = evaluation_df[metric].idxmin()
        
        best_model = evaluation_df.loc[best_idx]
        
        # Generate summary message
        message = [
            f"Best forecasting model based on {metric}:",
            f"- Model: {best_model['model']}",
            f"- {metric}: {best_model[metric]:.4f}",
            "\nAll evaluation metrics for this model:"
        ]
        
        for col in evaluation_df.columns:
            if col != 'model':
                message.append(f"- {col}: {best_model[col]:.4f}")
        
        # Add recommendations
        message.append("\nRecommendations:")
        message.append(f"1. Use the {best_model['model']} model for production forecasting.")
        message.append(f"2. When training the model, use the following model-specific parameters:")
        
        # Model-specific recommendations
        if best_model['model'] == 'SARIMA':
            message.append("   - Tune the order parameters (p, d, q) and seasonal_order parameters (P, D, Q, s)")
            message.append("   - Consider different seasonal periods (weekly, monthly, quarterly)")
        elif best_model['model'] == 'Prophet':
            message.append("   - Adjust changepoint_prior_scale to control flexibility of the trend")
            message.append("   - Enable/disable yearly, weekly, and daily seasonality components")
            message.append("   - Add holiday effects if relevant")
        elif best_model['model'] == 'XGBoost' or best_model['model'] == 'LightGBM':
            message.append("   - Tune the number of lag features to capture temporal patterns")
            message.append("   - Adjust learning_rate, n_estimators, and tree-specific parameters")
            message.append("   - Consider feature engineering for day-of-week, month, holiday effects")
        elif best_model['model'] == 'NeuralProphet':
            message.append("   - Adjust n_changepoints to control trend flexibility")
            message.append("   - Enable/disable yearly and weekly seasonality components")
            message.append("   - Consider adding custom events or seasonality")
        
        message.append(f"\n3. Regularly retrain the model as new data becomes available.")
        message.append(f"4. Monitor forecast accuracy over time and adjust parameters as needed.")
        
        return "\n".join(message)
        
    except Exception as e:
        logger.error(f"Error getting best forecast model: {str(e)}")
        return f"Error getting best forecast model: {str(e)}"

# Collect all tools in a list for easy access
forecasting_tools = [
    train_forecast_model_tool,
    generate_forecast_tool,
    compare_forecast_models_tool,
    visualize_forecast_tool,
    get_best_forecast_model_tool
] 