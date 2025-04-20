"""
Celery tasks for forecasting.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from src.models.database import get_db_context
from src.models.forecast import Forecast, ForecastJob, ForecastModel, ForecastStatus, ModelType
from src.models.inventory import Product
from src.tasks.celery_app import celery_app, with_logging
from src.utils.experiment_tracking import log_forecast, log_model_training, setup_mlflow
from src.utils.metrics import record_forecast_metrics

logger = logging.getLogger(__name__)

@celery_app.task(name="src.tasks.forecasting.forecast_product")
@with_logging
def forecast_product(product_id: int, horizon: int = 30, model_type: Optional[str] = None) -> Dict[str, Any]:
    """Generate forecast for a specific product.
    
    Args:
        product_id: Product ID
        horizon: Forecast horizon in days
        model_type: Model type to use (if None, use best model)
        
    Returns:
        Dict: Forecast results
    """
    start_time = time.time()
    
    with get_db_context() as db:
        # Get product
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise ValueError(f"Product with ID {product_id} not found")
            
        # Load historical data
        # In a real implementation, this would load data from a database
        # For this example, we'll simulate data
        historical_data = _load_historical_data(product_id)
        
        # Select model type
        if model_type is None:
            model_type = _select_best_model_type(product_id, historical_data, db)
        
        # Generate forecast
        forecast_result = _generate_forecast(
            product_id=product_id,
            product_name=product.name,
            data=historical_data,
            horizon=horizon,
            model_type=model_type,
        )
        
        # Save forecast to database
        forecast = Forecast(
            product_id=product_id,
            forecast_dates=forecast_result["dates"],
            forecast_values=forecast_result["values"],
            lower_bounds=forecast_result.get("lower_bounds"),
            upper_bounds=forecast_result.get("upper_bounds"),
            metrics=forecast_result.get("metrics"),
            horizon=horizon,
            status=ForecastStatus.COMPLETED,
        )
        db.add(forecast)
        db.commit()
        
        # Log forecast to MLflow
        if forecast_result.get("metrics"):
            log_forecast(
                product_id=str(product_id),
                model_type=model_type,
                forecast_values=forecast_result["values"],
                upper_bound=forecast_result.get("upper_bounds"),
                lower_bound=forecast_result.get("lower_bounds"),
                metrics=forecast_result.get("metrics"),
                forecast_dates=forecast_result["dates"],
            )
            
            # Record metrics for Prometheus
            record_forecast_metrics(
                product_id=str(product_id),
                model_type=model_type,
                rmse=forecast_result["metrics"].get("rmse", 0.0),
                mae=forecast_result["metrics"].get("mae", 0.0),
                r2=forecast_result["metrics"].get("r2", 0.0),
                processing_time=time.time() - start_time,
            )
        
        return {
            "product_id": product_id,
            "product_name": product.name,
            "model_type": model_type,
            "horizon": horizon,
            "forecast_dates": forecast_result["dates"],
            "forecast_values": forecast_result["values"],
            "lower_bounds": forecast_result.get("lower_bounds"),
            "upper_bounds": forecast_result.get("upper_bounds"),
            "metrics": forecast_result.get("metrics"),
            "processing_time": time.time() - start_time,
        }

@celery_app.task(name="src.tasks.forecasting.forecast_all_products")
@with_logging
def forecast_all_products(horizon: int = 30) -> Dict[str, Any]:
    """Generate forecasts for all products.
    
    Args:
        horizon: Forecast horizon in days
        
    Returns:
        Dict: Summary of forecasting results
    """
    start_time = time.time()
    
    with get_db_context() as db:
        # Get all active products
        products = db.query(Product).filter(Product.is_active == True).all()
        
        if not products:
            return {
                "status": "error",
                "message": "No active products found",
                "processing_time": time.time() - start_time,
            }
            
        # Create forecast job
        job = ForecastJob(
            job_id=forecast_all_products.request.id,
            status=ForecastStatus.RUNNING,
            parameters={"horizon": horizon},
        )
        db.add(job)
        db.commit()
        
        # Start tasks for each product
        results = []
        for product in products:
            # Queue forecast task for this product
            task_result = forecast_product.delay(product.id, horizon)
            results.append({
                "product_id": product.id,
                "task_id": task_result.id,
            })
            
        return {
            "status": "success",
            "message": f"Forecasting started for {len(products)} products",
            "product_count": len(products),
            "tasks": results,
            "processing_time": time.time() - start_time,
        }

@celery_app.task(name="src.tasks.forecasting.update_all_forecasts")
@with_logging
def update_all_forecasts(horizon: int = 30) -> Dict[str, Any]:
    """Update forecasts for all products.
    
    This is typically run on a schedule (e.g., daily).
    
    Args:
        horizon: Forecast horizon in days
        
    Returns:
        Dict: Summary of forecasting results
    """
    return forecast_all_products(horizon)

@celery_app.task(name="src.tasks.forecasting.retrain_model")
@with_logging
def retrain_model(product_id: int, model_type: str) -> Dict[str, Any]:
    """Retrain a forecasting model for a specific product.
    
    Args:
        product_id: Product ID
        model_type: Model type to retrain
        
    Returns:
        Dict: Model training results
    """
    start_time = time.time()
    
    with get_db_context() as db:
        # Get product
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise ValueError(f"Product with ID {product_id} not found")
            
        # Load historical data
        historical_data = _load_historical_data(product_id)
        
        # Train model
        model_result = _train_model(
            product_id=product_id,
            product_name=product.name,
            data=historical_data,
            model_type=model_type,
        )
        
        # Save model to database
        model = db.query(ForecastModel).filter(
            ForecastModel.product_id == product_id,
            ForecastModel.model_type == model_type,
        ).first()
        
        if model:
            # Update existing model
            model.parameters = model_result["parameters"]
            model.metrics = model_result["metrics"]
            model.mlflow_run_id = model_result.get("mlflow_run_id")
            model.mlflow_model_uri = model_result.get("mlflow_model_uri")
            model.updated_at = datetime.utcnow()
        else:
            # Create new model
            model = ForecastModel(
                name=f"{product.name}_{model_type}",
                product_id=product_id,
                model_type=model_type,
                parameters=model_result["parameters"],
                metrics=model_result["metrics"],
                mlflow_run_id=model_result.get("mlflow_run_id"),
                mlflow_model_uri=model_result.get("mlflow_model_uri"),
            )
            db.add(model)
            
        db.commit()
        
        return {
            "product_id": product_id,
            "product_name": product.name,
            "model_type": model_type,
            "metrics": model_result["metrics"],
            "parameters": model_result["parameters"],
            "mlflow_run_id": model_result.get("mlflow_run_id"),
            "processing_time": time.time() - start_time,
        }

@celery_app.task(name="src.tasks.forecasting.retrain_all_models")
@with_logging
def retrain_all_models() -> Dict[str, Any]:
    """Retrain all forecasting models.
    
    This is typically run on a schedule (e.g., weekly).
    
    Returns:
        Dict: Summary of model training results
    """
    start_time = time.time()
    
    with get_db_context() as db:
        # Get all active products
        products = db.query(Product).filter(Product.is_active == True).all()
        
        if not products:
            return {
                "status": "error",
                "message": "No active products found",
                "processing_time": time.time() - start_time,
            }
            
        # Create job
        job = ForecastJob(
            job_id=retrain_all_models.request.id,
            status=ForecastStatus.RUNNING,
            parameters={},
        )
        db.add(job)
        db.commit()
        
        # Start tasks for each product
        results = []
        for product in products:
            # Determine which models to retrain
            model_types = [model_type.value for model_type in ModelType]
            
            for model_type in model_types:
                # Queue retrain task for this product and model type
                task_result = retrain_model.delay(product.id, model_type)
                results.append({
                    "product_id": product.id,
                    "model_type": model_type,
                    "task_id": task_result.id,
                })
                
        return {
            "status": "success",
            "message": f"Model retraining started for {len(products)} products",
            "product_count": len(products),
            "task_count": len(results),
            "tasks": results,
            "processing_time": time.time() - start_time,
        }

def _load_historical_data(product_id: int) -> pd.DataFrame:
    """Load historical data for a product.
    
    In a real implementation, this would load data from a database.
    For this example, we'll simulate data.
    
    Args:
        product_id: Product ID
        
    Returns:
        pd.DataFrame: Historical data
    """
    # Simulate historical data
    np.random.seed(product_id)  # For reproducibility
    
    # Create dates for past 2 years
    dates = [datetime.now() - timedelta(days=i) for i in range(365 * 2, 0, -1)]
    dates.sort()
    
    # Generate random demand with trend and seasonality
    base_demand = 100 + product_id % 10 * 10
    trend = np.linspace(0, 20, len(dates))
    seasonality = 10 * np.sin(np.arange(len(dates)) * (2 * np.pi / 365))
    weekly = 5 * np.sin(np.arange(len(dates)) * (2 * np.pi / 7))
    noise = np.random.normal(0, 5, len(dates))
    
    demand = base_demand + trend + seasonality + weekly + noise
    demand = np.maximum(demand, 0)  # Ensure non-negative
    
    # Create DataFrame
    data = pd.DataFrame({
        "date": dates,
        "demand": demand.astype(int),
    })
    
    return data

def _select_best_model_type(product_id: int, data: pd.DataFrame, db: Session) -> str:
    """Select the best model type for a product.
    
    Args:
        product_id: Product ID
        data: Historical data
        db: Database session
        
    Returns:
        str: Best model type
    """
    # Check if we have previously trained models
    models = db.query(ForecastModel).filter(
        ForecastModel.product_id == product_id,
        ForecastModel.is_active == True,
    ).all()
    
    if models:
        # Find model with best RMSE
        best_model = min(models, key=lambda m: m.metrics.get("rmse", float("inf")))
        return best_model.model_type.value
        
    # If no existing models, select based on data characteristics
    # This is a simplistic example; in practice, you'd use more sophisticated logic
    
    # Check if we have enough data
    if len(data) < 30:
        return "exponential_smoothing"  # Simple model for limited data
        
    # Check for strong weekly pattern
    data["dayofweek"] = data["date"].dt.dayofweek
    weekly_variance = data.groupby("dayofweek")["demand"].mean().var()
    
    if weekly_variance > 100:
        return "prophet"  # Good for multiple seasonalities
        
    # Check for strong trend
    first_half = data["demand"].iloc[:len(data)//2].mean()
    second_half = data["demand"].iloc[len(data)//2:].mean()
    trend_strength = abs(second_half - first_half) / data["demand"].mean()
    
    if trend_strength > 0.2:
        return "sarima"  # Good for strong trends and seasonality
        
    # Default to ensemble
    return "ensemble"

def _generate_forecast(
    product_id: int,
    product_name: str,
    data: pd.DataFrame,
    horizon: int = 30,
    model_type: str = "ensemble",
) -> Dict[str, Any]:
    """Generate forecast for a product.
    
    This is a simplified implementation; in practice, you'd use real forecasting models.
    
    Args:
        product_id: Product ID
        product_name: Product name
        data: Historical data
        horizon: Forecast horizon in days
        model_type: Model type
        
    Returns:
        Dict: Forecast results
    """
    # Get the last date in the data
    last_date = data["date"].max()
    
    # Generate forecast dates
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    forecast_dates_str = [d.strftime("%Y-%m-%d") for d in forecast_dates]
    
    # Generate forecast values based on model type
    # This is a simplistic simulation; in practice, you'd use real forecasting models
    np.random.seed(product_id + hash(model_type) % 1000)
    
    # Use the last 30 days as a baseline
    recent_demand = data["demand"].iloc[-30:].values
    recent_mean = recent_demand.mean()
    recent_std = recent_demand.std()
    
    if model_type == "sarima":
        # Simulate SARIMA forecast
        trend = np.linspace(0, 5, horizon)
        seasonality = 10 * np.sin(np.arange(horizon) * (2 * np.pi / 30))
        forecast = recent_mean + trend + seasonality + np.random.normal(0, recent_std * 0.5, horizon)
    elif model_type == "exponential_smoothing":
        # Simulate exponential smoothing forecast
        forecast = recent_mean + np.random.normal(0, recent_std * 0.3, horizon)
    elif model_type == "prophet":
        # Simulate Prophet forecast
        trend = np.linspace(0, 8, horizon)
        weekly = 8 * np.sin(np.arange(horizon) * (2 * np.pi / 7))
        forecast = recent_mean + trend + weekly + np.random.normal(0, recent_std * 0.4, horizon)
    elif model_type == "random_forest" or model_type == "gradient_boosting":
        # Simulate machine learning forecast
        forecast = recent_mean + np.random.normal(0, recent_std * 0.2, horizon)
    elif model_type == "lstm":
        # Simulate LSTM forecast
        trend = np.linspace(0, 10, horizon)
        forecast = recent_mean + trend + np.random.normal(0, recent_std * 0.3, horizon)
    else:
        # Default ensemble forecast
        trend = np.linspace(0, 3, horizon)
        seasonality = 5 * np.sin(np.arange(horizon) * (2 * np.pi / 30))
        weekly = 3 * np.sin(np.arange(horizon) * (2 * np.pi / 7))
        forecast = recent_mean + trend + seasonality + weekly + np.random.normal(0, recent_std * 0.1, horizon)
    
    # Ensure non-negative values
    forecast = np.maximum(forecast, 0).astype(int)
    
    # Generate confidence intervals
    lower_bounds = np.maximum(forecast - recent_std, 0).astype(int)
    upper_bounds = (forecast + recent_std).astype(int)
    
    # Simulate forecast metrics
    metrics = {
        "rmse": float(np.random.uniform(5, 15)),
        "mae": float(np.random.uniform(3, 10)),
        "r2": float(np.random.uniform(0.7, 0.95)),
    }
    
    return {
        "dates": forecast_dates_str,
        "values": forecast.tolist(),
        "lower_bounds": lower_bounds.tolist(),
        "upper_bounds": upper_bounds.tolist(),
        "metrics": metrics,
    }

def _train_model(
    product_id: int,
    product_name: str,
    data: pd.DataFrame,
    model_type: str,
) -> Dict[str, Any]:
    """Train a forecasting model.
    
    This is a simplified implementation; in practice, you'd train real forecasting models.
    
    Args:
        product_id: Product ID
        product_name: Product name
        data: Historical data
        model_type: Model type
        
    Returns:
        Dict: Model training results
    """
    # Set up MLflow experiment
    experiment_id = setup_mlflow("inventory-forecasting")
    
    # Simulate model training
    np.random.seed(product_id + hash(model_type) % 1000)
    
    # Generate simulated model parameters based on model type
    if model_type == "sarima":
        parameters = {
            "p": np.random.randint(1, 4),
            "d": np.random.randint(0, 2),
            "q": np.random.randint(1, 4),
            "P": np.random.randint(1, 3),
            "D": np.random.randint(0, 2),
            "Q": np.random.randint(1, 3),
            "s": 7,  # Weekly seasonality
        }
    elif model_type == "exponential_smoothing":
        parameters = {
            "trend": np.random.choice(["add", "mul", None]),
            "seasonal": np.random.choice(["add", "mul", None]),
            "seasonal_periods": np.random.choice([7, 30]),
        }
    elif model_type == "prophet":
        parameters = {
            "changepoint_prior_scale": np.random.uniform(0.01, 0.5),
            "seasonality_prior_scale": np.random.uniform(0.01, 10.0),
            "holidays_prior_scale": np.random.uniform(0.01, 10.0),
            "seasonality_mode": np.random.choice(["additive", "multiplicative"]),
        }
    elif model_type in ["random_forest", "gradient_boosting"]:
        parameters = {
            "n_estimators": np.random.randint(50, 200),
            "max_depth": np.random.randint(3, 10),
            "min_samples_split": np.random.randint(2, 6),
            "min_samples_leaf": np.random.randint(1, 4),
        }
    elif model_type == "lstm":
        parameters = {
            "units": np.random.randint(32, 128),
            "layers": np.random.randint(1, 4),
            "dropout": np.random.uniform(0.1, 0.5),
            "epochs": np.random.randint(50, 200),
            "batch_size": np.random.choice([16, 32, 64]),
        }
    else:
        # Ensemble parameters
        parameters = {
            "models": ["sarima", "exponential_smoothing", "prophet"],
            "weights": [0.4, 0.3, 0.3],
        }
    
    # Simulate model metrics
    metrics = {
        "rmse": float(np.random.uniform(5, 15)),
        "mae": float(np.random.uniform(3, 10)),
        "r2": float(np.random.uniform(0.7, 0.95)),
        "aic": float(np.random.uniform(100, 500)),
    }
    
    # Log to MLflow if experiment is set up
    if experiment_id:
        run_id = log_model_training(
            model=None,  # In a real implementation, this would be the trained model
            model_name=f"{product_name}_{model_type}",
            params=parameters,
            metrics=metrics,
            features=["date", "demand"],
            model_info={
                "product_id": product_id,
                "model_type": model_type,
            },
        )
    else:
        run_id = None
    
    return {
        "parameters": parameters,
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "mlflow_model_uri": f"runs:/{run_id}/model" if run_id else None,
    } 