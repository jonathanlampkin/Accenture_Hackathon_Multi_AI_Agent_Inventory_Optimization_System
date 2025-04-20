"""
Experiment tracking utilities using MLflow.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Default MLflow tracking URI
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

def setup_mlflow(experiment_name: str = "inventory-optimization") -> str:
    """Set up MLflow and create or get experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        str: Experiment ID
    """
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
            
        logger.info(f"Using MLflow experiment '{experiment_name}' (ID: {experiment_id})")
        return experiment_id
    except Exception as e:
        logger.error(f"Failed to set up MLflow: {e}")
        logger.warning("MLflow tracking disabled. Running in local mode.")
        return None
        
def log_model_training(
    model: Union[BaseEstimator, Pipeline],
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    features: List[str],
    artifact_path: Optional[str] = None,
    model_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Log model training to MLflow.
    
    Args:
        model: Trained model
        model_name: Name of the model
        params: Model parameters
        metrics: Model performance metrics
        features: List of feature names
        artifact_path: Path to save artifacts
        model_info: Additional model information
        
    Returns:
        str: Run ID
    """
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log model name and type
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_type", type(model).__name__)
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log feature names
            mlflow.log_param("features", ", ".join(features))
            
            # Log additional model info if provided
            if model_info:
                for key, value in model_info.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)
                        
            # Log model
            if artifact_path is None:
                artifact_path = model_name
                
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=f"{model_name}",
            )
            
            logger.info(f"Logged model training to MLflow (Run ID: {run_id})")
            return run_id
    except Exception as e:
        logger.error(f"Failed to log model training to MLflow: {e}")
        return None
        
def log_forecast(
    product_id: str,
    model_type: str,
    forecast_values: List[float],
    actual_values: Optional[List[float]] = None,
    upper_bound: Optional[List[float]] = None,
    lower_bound: Optional[List[float]] = None,
    metrics: Optional[Dict[str, float]] = None,
    forecast_dates: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Log forecast to MLflow.
    
    Args:
        product_id: Product ID
        model_type: Model type
        forecast_values: Forecast values
        actual_values: Actual values (if available for comparison)
        upper_bound: Upper confidence bound
        lower_bound: Lower confidence bound
        metrics: Forecast performance metrics
        forecast_dates: Dates for the forecast
        params: Model parameters used for forecasting
        
    Returns:
        str: Run ID
    """
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log tags
            mlflow.set_tag("product_id", product_id)
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("content_type", "forecast")
            
            # Log parameters if provided
            if params:
                mlflow.log_params(params)
                
            # Log metrics if provided
            if metrics:
                mlflow.log_metrics(metrics)
                
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({"forecast": forecast_values})
            
            if forecast_dates:
                forecast_df["date"] = forecast_dates
                
            if actual_values:
                forecast_df["actual"] = actual_values
                
            if upper_bound:
                forecast_df["upper_bound"] = upper_bound
                
            if lower_bound:
                forecast_df["lower_bound"] = lower_bound
                
            # Save forecast data as artifact
            forecast_path = f"forecasts/{product_id}_{model_type}_forecast.csv"
            forecast_df.to_csv(forecast_path, index=False)
            mlflow.log_artifact(forecast_path)
            
            # Clean up
            try:
                os.remove(forecast_path)
            except:
                pass
                
            logger.info(f"Logged forecast to MLflow (Run ID: {run_id})")
            return run_id
    except Exception as e:
        logger.error(f"Failed to log forecast to MLflow: {e}")
        return None
        
def load_model(model_name: str, stage: str = "Production") -> Tuple[Any, Dict[str, Any]]:
    """Load a model from MLflow model registry.
    
    Args:
        model_name: Name of the model
        stage: Model stage (e.g., "Production", "Staging")
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Loaded model and model info
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        model_details = client.get_latest_versions(model_name, stages=[stage])[0]
        run_id = model_details.run_id
        
        # Get run info
        run = client.get_run(run_id)
        model_info = {
            "run_id": run_id,
            "model_name": model_name,
            "stage": stage,
            "params": run.data.params,
            "metrics": run.data.metrics,
        }
        
        logger.info(f"Loaded model {model_name} (stage: {stage}) from MLflow")
        return model, model_info
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        return None, {}
        
def compare_models(
    product_id: str,
    model_names: List[str],
    metric: str = "rmse",
) -> Dict[str, Any]:
    """Compare models for a specific product based on a metric.
    
    Args:
        product_id: Product ID
        model_names: List of model names to compare
        metric: Metric to use for comparison
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        results = {}
        best_model = None
        best_value = float("inf") if metric in ["rmse", "mae"] else float("-inf")
        
        for model_name in model_names:
            # Get latest version
            try:
                versions = client.get_latest_versions(model_name, stages=["Production", "Staging"])
                if not versions:
                    continue
                    
                version = versions[0]
                run = client.get_run(version.run_id)
                
                # Check if this run has the product_id tag
                if run.data.tags.get("product_id") != product_id:
                    continue
                    
                # Get metric value
                metric_value = run.data.metrics.get(metric)
                if metric_value is None:
                    continue
                    
                results[model_name] = {
                    "run_id": version.run_id,
                    "version": version.version,
                    "stage": version.current_stage,
                    "metric": metric,
                    "value": metric_value,
                }
                
                # Check if this is the best model
                is_better = (metric_value < best_value if metric in ["rmse", "mae"] else metric_value > best_value)
                if is_better:
                    best_model = model_name
                    best_value = metric_value
            except Exception as e:
                logger.error(f"Error getting info for model {model_name}: {e}")
                
        return {
            "product_id": product_id,
            "metric": metric,
            "models": results,
            "best_model": best_model,
            "best_value": best_value if best_model else None,
        }
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return {
            "product_id": product_id,
            "metric": metric,
            "models": {},
            "best_model": None,
            "best_value": None,
            "error": str(e),
        } 