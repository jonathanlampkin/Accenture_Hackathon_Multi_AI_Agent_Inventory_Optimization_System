"""
Forecasting API Service

This module provides a RESTful API for the enhanced forecasting system.
It allows for:
- Generating forecasts for specific products
- Batch forecasting for multiple products
- Model selection and hyperparameter tuning
- Model performance evaluation
- Forecast visualization
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
import tempfile
import uuid
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Add the app directory to the path to import the forecasting module
sys.path.append("/app")
from improved_forecasting import ImprovedForecaster, run_improved_forecasting
from enhanced_forecasting import run_enhanced_forecasting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.environ.get("LOG_DIR", "/app/logs"), "forecasting_api.log"))
    ]
)
logger = logging.getLogger("forecasting_api")

# Initialize the FastAPI app
app = FastAPI(
    title="Inventory Optimization Forecasting API",
    description="API for advanced time series forecasting to support inventory optimization",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup MLflow tracking
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service:80")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("forecasting-api")

# Cached forecaster instance
forecaster_instance = None

# Pydantic models for request/response
class ForecastRequest(BaseModel):
    data_path: str
    forecast_horizon: int = 30
    test_proportion: float = 0.2
    product_ids: Optional[List[int]] = None
    
class ForecastResponse(BaseModel):
    request_id: str
    status: str
    message: str
    output_directory: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# In-memory storage for job status
forecast_jobs = {}

def get_forecaster():
    """Get or create a forecaster instance"""
    global forecaster_instance
    if forecaster_instance is None:
        try:
            data_path = os.path.join(os.environ.get("DATA_DIR", "/app/data"), "demand_data.csv")
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                data['Date'] = pd.to_datetime(data['Date'])
                
                output_dir = os.environ.get("OUTPUT_DIR", "/app/output")
                os.makedirs(output_dir, exist_ok=True)
                
                forecaster_instance = ImprovedForecaster(
                    data=data,
                    target_col='Sales Quantity',
                    date_col='Date',
                    product_col='Product ID',
                    output_dir=output_dir
                )
                logger.info(f"Forecaster initialized with {len(data)} records")
            else:
                logger.error(f"Data file not found at {data_path}")
                raise FileNotFoundError(f"Data file not found at {data_path}")
        except Exception as e:
            logger.error(f"Error initializing forecaster: {str(e)}")
            raise
    return forecaster_instance

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/products")
async def get_products(forecaster: ImprovedForecaster = Depends(get_forecaster)):
    """Get list of available products"""
    try:
        products = forecaster.data[forecaster.product_col].unique().tolist()
        return {"products": products, "count": len(products)}
    except Exception as e:
        logger.error(f"Error getting products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_forecast_task(request_id: str, data_path: str, output_dir: str, 
                     test_proportion: float, forecast_horizon: int, product_ids: Optional[List[int]] = None):
    """Background task to run forecasting"""
    try:
        logger.info(f"Starting forecast job {request_id}")
        forecast_jobs[request_id]["status"] = "running"
        
        # Load the data
        demand_data = pd.read_csv(data_path)
        
        # Filter by product_ids if specified
        if product_ids:
            demand_data = demand_data[demand_data['Product ID'].isin(product_ids)]
            if demand_data.empty:
                raise ValueError(f"No data found for the specified product IDs: {product_ids}")
        
        # Initialize forecaster
        forecaster = ImprovedForecaster(output_dir=output_dir)
        
        # Run forecasting
        results = forecaster.forecast_all_products(
            demand_data, 
            forecast_horizon=forecast_horizon,
            test_proportion=test_proportion
        )
        
        # Generate visualizations
        forecaster.visualize_forecasts(results)
        
        # Generate summary report
        forecaster.generate_summary_report(results)
        
        # Update job status
        forecast_jobs[request_id]["status"] = "completed"
        forecast_jobs[request_id]["message"] = "Forecasting completed successfully"
        forecast_jobs[request_id]["results"] = {
            "forecast_count": len(results),
            "output_files": os.listdir(output_dir)
        }
        logger.info(f"Completed forecast job {request_id}")
        
    except Exception as e:
        logger.error(f"Error in forecast job {request_id}: {str(e)}", exc_info=True)
        forecast_jobs[request_id]["status"] = "failed"
        forecast_jobs[request_id]["message"] = f"Error: {str(e)}"

@app.post("/forecasts", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """Create a new forecast"""
    try:
        # Validate input
        if not os.path.exists(request.data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # Create unique ID and output directory
        request_id = str(uuid.uuid4())
        output_dir = os.path.join("output", f"forecast_{request_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize job status
        forecast_jobs[request_id] = {
            "status": "queued",
            "message": "Job queued for processing",
            "output_directory": output_dir,
            "request": request.dict()
        }
        
        # Start background task
        background_tasks.add_task(
            run_forecast_task,
            request_id=request_id,
            data_path=request.data_path,
            output_dir=output_dir,
            test_proportion=request.test_proportion,
            forecast_horizon=request.forecast_horizon,
            product_ids=request.product_ids
        )
        
        return {
            "request_id": request_id,
            "status": "queued",
            "message": "Forecast job queued for processing",
            "output_directory": output_dir
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecasts/{request_id}", response_model=Dict[str, Any])
async def get_forecast(request_id: str):
    """Get status and results of a forecast job"""
    if request_id not in forecast_jobs:
        raise HTTPException(status_code=404, detail=f"Forecast job not found: {request_id}")
    
    return forecast_jobs[request_id]

@app.get("/forecasts/{request_id}/download")
async def download_forecast(request_id: str, file_name: str = Query(..., description="Name of the file to download")):
    """Download a forecast file"""
    if request_id not in forecast_jobs:
        raise HTTPException(status_code=404, detail=f"Forecast job not found: {request_id}")
    
    job = forecast_jobs[request_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Forecast job not completed: {job['status']}")
    
    output_dir = job["output_directory"]
    file_path = os.path.join(output_dir, file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_name}")
    
    return FileResponse(file_path, filename=file_name)

@app.get("/forecasts", response_model=Dict[str, Any])
async def list_forecasts():
    """List all forecast jobs"""
    return {
        "jobs": list(forecast_jobs.keys()),
        "count": len(forecast_jobs),
        "details": forecast_jobs
    }

@app.delete("/forecasts/{request_id}")
async def delete_forecast(request_id: str):
    """Delete a forecast job and its files"""
    if request_id not in forecast_jobs:
        raise HTTPException(status_code=404, detail=f"Forecast job not found: {request_id}")
    
    job = forecast_jobs[request_id]
    output_dir = job["output_directory"]
    
    # Delete output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Remove job from in-memory storage
    forecast_jobs.pop(request_id)
    
    return {"status": "deleted", "message": f"Forecast job {request_id} deleted successfully"}

@app.get("/forecast/visualization/{product_id}")
async def get_forecast_visualization(
    product_id: int, 
    horizon: int = Query(default=30, ge=1, le=365),
    method: Optional[str] = Query(default=None),
    include_history: bool = Query(default=True),
    include_test: bool = Query(default=True),
    include_intervals: bool = Query(default=True),
):
    """Get visualization for a product forecast"""
    try:
        # Generate forecast for the specified product
        result = get_forecaster().forecast_product(
            product_id=product_id,
            horizon=horizon,
            method=method
        )
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot historical data if available and requested
        if include_history and "historical_data" in result and "historical_dates" in result:
            plt.plot(
                result["historical_dates"], 
                result["historical_data"], 
                'b-', 
                label='Historical Data'
            )
        
        # Plot test data if available and requested
        if include_test and "test_data" in result and "test_dates" in result:
            plt.plot(
                result["test_dates"], 
                result["test_data"], 
                'g-', 
                label='Test Data'
            )
        
        # Plot forecast
        plt.plot(
            result["forecast_dates"], 
            result["forecast"], 
            'r-', 
            label=f'Forecast ({result["method"]})'
        )
        
        # Plot confidence intervals if available and requested
        if include_intervals and "confidence_intervals" in result:
            plt.fill_between(
                result["forecast_dates"],
                result["confidence_intervals"]["lower"],
                result["confidence_intervals"]["upper"],
                color='r',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        # Add metrics to title if available
        title = f"Forecast for Product {product_id}"
        if "metrics" in result:
            metrics_str = ", ".join([f"{k}: {v:.2f}" for k, v in result["metrics"].items()])
            title += f" ({metrics_str})"
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.legend()
        plt.grid(True)
        
        # Save to bytesIO
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close()
        
        # Return as file response
        return FileResponse(
            img_bytes, 
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=forecast_{product_id}.png"}
        )
            
    except Exception as e:
        logger.error(f"Error generating forecast visualization for product {product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get list of available forecasting models"""
    models = {
        "sarima": "Seasonal ARIMA model for time series with seasonality",
        "exponential_smoothing": "Exponential smoothing for trend and seasonality",
        "prophet": "Facebook Prophet for complex seasonality patterns",
        "random_forest": "Random Forest regression for feature-rich forecasting",
        "gradient_boosting": "Gradient Boosting for high accuracy forecasting",
        "ensemble": "Ensemble approach combining multiple models"
    }
    return {"models": models}

@app.post("/models/tune/{product_id}")
async def tune_model(
    product_id: int,
    model_type: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    forecaster: ImprovedForecaster = Depends(get_forecaster)
):
    """Tune hyperparameters for a specific model and product"""
    try:
        # Validate model type
        valid_models = ["sarima", "exponential_smoothing", "prophet", "random_forest", "gradient_boosting"]
        if model_type not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of: {', '.join(valid_models)}")
        
        # Start background task for tuning
        def tune_model_task():
            try:
                with mlflow.start_run(run_name=f"tune_{model_type}_{product_id}"):
                    mlflow.log_params({
                        "product_id": product_id,
                        "model_type": model_type
                    })
                    
                    # Run hyperparameter optimization
                    result = forecaster.optimize_hyperparameters(
                        product_id=str(product_id),
                        model_type=model_type
                    )
                    
                    # Log best parameters
                    if result and 'best_params' in result:
                        mlflow.log_params(result['best_params'])
                    
                    # Log best metrics
                    if result and 'best_metrics' in result:
                        mlflow.log_metrics(result['best_metrics'])
                    
                    logger.info(f"Model tuning completed for {model_type} on product {product_id}")
            except Exception as e:
                logger.error(f"Error in model tuning: {str(e)}")
        
        # Start the background task
        background_tasks.add_task(tune_model_task)
        
        return {
            "status": "tuning",
            "message": f"Started hyperparameter tuning for {model_type} on product {product_id}",
            "results_path": f"{os.environ.get('OUTPUT_DIR', '/app/output')}/tuning/{model_type}_{product_id}"
        }
    
    except Exception as e:
        logger.error(f"Error starting model tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/compare")
async def compare_model_performance(
    forecaster: ImprovedForecaster = Depends(get_forecaster)
):
    """Compare performance of different forecasting models"""
    try:
        # Get forecast summary
        output_dir = os.environ.get("OUTPUT_DIR", "/app/output")
        summary_path = f"{output_dir}/forecast_summary.csv"
        
        if not os.path.exists(summary_path):
            # Generate summary first
            summary_df = forecaster.generate_summary_report()
        else:
            summary_df = pd.read_csv(summary_path)
        
        # Group by model type and calculate average metrics
        if 'Best Method' in summary_df.columns:
            performance_by_model = summary_df.groupby('Best Method').agg({
                'RMSE': 'mean',
                'MAE': 'mean',
                'R²': 'mean'
            }).reset_index()
            
            return {
                "model_performance": performance_by_model.to_dict(orient='records'),
                "best_overall": performance_by_model.loc[performance_by_model['RMSE'].idxmin(), 'Best Method']
            }
        else:
            raise HTTPException(status_code=404, detail="Performance metrics not available")
    
    except Exception as e:
        logger.error(f"Error comparing model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/validate")
async def validate_data(
    data: Dict[str, Any]
):
    """Validate input data for forecasting"""
    try:
        required_fields = ['date', 'product_id', 'sales_quantity']
        
        # Check for required fields
        for field in required_fields:
            if field not in data:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Missing required field: {field}"}
                )
        
        # Basic validation checks
        if not isinstance(data['sales_quantity'], (int, float)) or data['sales_quantity'] < 0:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "sales_quantity must be a non-negative number"}
            )
        
        # Validate date format
        try:
            datetime.fromisoformat(data['date'].replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "date must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"}
            )
        
        return {"status": "valid", "message": "Data validation passed"}
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anomaly/detect")
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalies in product data"""
    try:
        # Get product data
        product_data = get_forecaster().data[get_forecaster().data[get_forecaster().product_col] == request.product_id].copy()
        
        if product_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for product {request.product_id}")
        
        # Run anomaly detection
        cleaned_df = get_forecaster().detect_and_handle_anomalies(product_data, "Sales Quantity", request.sensitivity)
        
        # Identify which rows were flagged as anomalies
        anomaly_mask = product_data.index.difference(cleaned_df.index)
        anomalies = product_data.loc[anomaly_mask]
        
        # Format output
        anomaly_points = []
        for idx, row in anomalies.iterrows():
            anomaly_points.append({
                "date": row[get_forecaster().date_col].strftime('%Y-%m-%d'),
                "value": float(row["Sales Quantity"]),
                "expected_range": [
                    float(cleaned_df.loc[idx, "lower_bound"]),
                    float(cleaned_df.loc[idx, "upper_bound"])
                ],
                "deviation_percent": float(
                    abs(row["Sales Quantity"] - cleaned_df.loc[idx, "rolling_mean"]) / cleaned_df.loc[idx, "rolling_mean"] * 100
                    if cleaned_df.loc[idx, "rolling_mean"] > 0 else 0
                )
            })
        
        return {
            "product_id": request.product_id,
            "total_points": len(product_data),
            "anomalies_detected": len(anomaly_points),
            "anomaly_percentage": len(anomaly_points) / len(product_data) * 100 if len(product_data) > 0 else 0,
            "anomalies": anomaly_points
        }
    
    except Exception as e:
        logger.error(f"Error detecting anomalies for product {request.product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload custom data for forecasting"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save uploaded file temporarily
        temp_file_path = f"./output/api_forecasts/temp_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Reinitialize forecaster with new data
        forecaster = ImprovedForecaster()
        forecaster.load_data(temp_file_path)
        
        # Get data summary
        data_summary = {
            "num_products": len(forecaster.data["Product ID"].unique()),
            "date_range": [
                forecaster.data["Date"].min().strftime("%Y-%m-%d"),
                forecaster.data["Date"].max().strftime("%Y-%m-%d")
            ],
            "total_records": len(forecaster.data),
        }
        
        logger.info(f"Successfully loaded custom data from {file.filename}")
        return {
            "status": "success", 
            "message": "Data uploaded and loaded successfully",
            "data_summary": data_summary
        }
            
    except Exception as e:
        logger.error(f"Error loading custom data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data loading failed: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/model/tune")
async def tune_model(request: ModelTuningRequest):
    """Tune hyperparameters for a specific model and product"""
    try:
        # Validate method
        valid_methods = ["sarima", "prophet", "exponential_smoothing", "random_forest", "gradient_boosting"]
        if request.method not in valid_methods:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method '{request.method}'. Valid methods are: {', '.join(valid_methods)}"
            )
        
        # Filter data for the specified product
        product_data = get_forecaster().data[get_forecaster().data["Product ID"] == request.product_id]
        
        if len(product_data) == 0:
            raise HTTPException(status_code=404, detail=f"Product {request.product_id} not found in data")
        
        # Define cross-validation parameters
        cv_results = []
        horizon = 30
        step = 30
        windows = 3
        
        # Get the number of data points
        n_points = len(product_data)
        
        if n_points < (windows + 1) * step:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough data points for product {request.product_id} to perform cross-validation tuning"
            )
        
        # Perform cross-validation
        for i in range(windows):
            train_end = n_points - (windows - i) * step
            test_end = train_end + step
            
            if train_end <= 0:
                continue
                
            # Prepare training and test sets
            train_data = product_data.iloc[:train_end]
            test_data = product_data.iloc[train_end:min(test_end, n_points)]
            
            # Tune model with the specified parameters
            logger.info(f"Tuning {request.method} for product {request.product_id}, window {i+1}/{windows}")
            
            # Fit model with custom parameters
            if request.method == "sarima":
                fitted_model = get_forecaster()._fit_sarima(
                    train_data["Sales Quantity"].values,
                    order=request.hyperparameters.get("order", (1, 1, 1)),
                    seasonal_order=request.hyperparameters.get("seasonal_order", (1, 1, 1, 12))
                )
                
                # Generate forecast
                forecast = get_forecaster()._forecast_sarima(
                    fitted_model, 
                    len(test_data)
                )
                
            elif request.method == "prophet":
                fitted_model = get_forecaster()._fit_prophet(
                    pd.DataFrame({
                        'ds': train_data["Date"],
                        'y': train_data["Sales Quantity"]
                    }),
                    changepoint_prior_scale=request.hyperparameters.get("changepoint_prior_scale", 0.05),
                    seasonality_prior_scale=request.hyperparameters.get("seasonality_prior_scale", 10),
                    holidays_prior_scale=request.hyperparameters.get("holidays_prior_scale", 10),
                    seasonality_mode=request.hyperparameters.get("seasonality_mode", "additive")
                )
                
                # Generate forecast
                future = fitted_model.make_future_dataframe(periods=len(test_data))
                forecast_df = fitted_model.predict(future)
                forecast = forecast_df.iloc[-len(test_data):]['yhat'].values
                
            elif request.method == "exponential_smoothing":
                fitted_model = get_forecaster()._fit_exponential_smoothing(
                    train_data["Sales Quantity"].values,
                    trend=request.hyperparameters.get("trend", None),
                    damped_trend=request.hyperparameters.get("damped_trend", False),
                    seasonal=request.hyperparameters.get("seasonal", None),
                    seasonal_periods=request.hyperparameters.get("seasonal_periods", 12)
                )
                
                # Generate forecast
                forecast = get_forecaster()._forecast_exponential_smoothing(
                    fitted_model, 
                    len(test_data)
                )
                
            elif request.method == "random_forest":
                # Prepare features
                feature_data = get_forecaster()._prepare_ml_features(
                    train_data, 
                    "Sales Quantity",
                    lag_features=request.hyperparameters.get("lag_features", [1, 7, 14]),
                    window_sizes=request.hyperparameters.get("window_sizes", [7, 30])
                )
                
                # Train model
                fitted_model = get_forecaster()._fit_random_forest(
                    feature_data.drop(["Sales Quantity"], axis=1), 
                    feature_data["Sales Quantity"],
                    n_estimators=request.hyperparameters.get("n_estimators", 100),
                    max_depth=request.hyperparameters.get("max_depth", None),
                    min_samples_split=request.hyperparameters.get("min_samples_split", 2)
                )
                
                # Generate forecast (one-step-ahead forecasting for simplicity)
                forecast = []
                current_data = train_data.copy()
                
                for _ in range(len(test_data)):
                    # Prepare features for the next forecast
                    feature_data = get_forecaster()._prepare_ml_features(
                        current_data, 
                        "Sales Quantity",
                        lag_features=request.hyperparameters.get("lag_features", [1, 7, 14]),
                        window_sizes=request.hyperparameters.get("window_sizes", [7, 30])
                    )
                    
                    # Generate next forecast
                    next_forecast = fitted_model.predict(feature_data.drop(["Sales Quantity"], axis=1).iloc[-1:])
                    forecast.append(next_forecast[0])
                    
                    # Update current data
                    new_row = current_data.iloc[-1:].copy()
                    new_row["Date"] = new_row["Date"] + pd.Timedelta(days=1)
                    new_row["Sales Quantity"] = next_forecast[0]
                    current_data = pd.concat([current_data, new_row])
                
            elif request.method == "gradient_boosting":
                # Prepare features
                feature_data = get_forecaster()._prepare_ml_features(
                    train_data, 
                    "Sales Quantity",
                    lag_features=request.hyperparameters.get("lag_features", [1, 7, 14]),
                    window_sizes=request.hyperparameters.get("window_sizes", [7, 30])
                )
                
                # Train model
                fitted_model = get_forecaster()._fit_gradient_boosting(
                    feature_data.drop(["Sales Quantity"], axis=1), 
                    feature_data["Sales Quantity"],
                    n_estimators=request.hyperparameters.get("n_estimators", 100),
                    learning_rate=request.hyperparameters.get("learning_rate", 0.1),
                    max_depth=request.hyperparameters.get("max_depth", 3)
                )
                
                # Generate forecast (one-step-ahead forecasting for simplicity)
                forecast = []
                current_data = train_data.copy()
                
                for _ in range(len(test_data)):
                    # Prepare features for the next forecast
                    feature_data = get_forecaster()._prepare_ml_features(
                        current_data, 
                        "Sales Quantity",
                        lag_features=request.hyperparameters.get("lag_features", [1, 7, 14]),
                        window_sizes=request.hyperparameters.get("window_sizes", [7, 30])
                    )
                    
                    # Generate next forecast
                    next_forecast = fitted_model.predict(feature_data.drop(["Sales Quantity"], axis=1).iloc[-1:])
                    forecast.append(next_forecast[0])
                    
                    # Update current data
                    new_row = current_data.iloc[-1:].copy()
                    new_row["Date"] = new_row["Date"] + pd.Timedelta(days=1)
                    new_row["Sales Quantity"] = next_forecast[0]
                    current_data = pd.concat([current_data, new_row])
            
            # Calculate metrics
            actual = test_data["Sales Quantity"].values
            rmse = np.sqrt(np.mean((actual - forecast)**2))
            mae = np.mean(np.abs(actual - forecast))
            
            # Calculate R^2
            mean_actual = np.mean(actual)
            ss_tot = np.sum((actual - mean_actual)**2)
            ss_res = np.sum((actual - forecast)**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Add results
            cv_results.append({
                "window": i + 1,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            })
        
        # Calculate average metrics
        avg_metrics = {
            "rmse": np.mean([r["rmse"] for r in cv_results]),
            "mae": np.mean([r["mae"] for r in cv_results]),
            "r2": np.mean([r["r2"] for r in cv_results])
        }
        
        return {
            "product_id": request.product_id,
            "method": request.method,
            "hyperparameters": request.hyperparameters,
            "cross_validation_results": cv_results,
            "average_metrics": avg_metrics,
            "windows_evaluated": len(cv_results)
        }
            
    except Exception as e:
        logger.error(f"Error tuning model for product {request.product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model tuning failed: {str(e)}")

@app.get("/models/available")
async def get_available_models():
    """Get list of available forecasting models"""
    models = {
        "sarima": "Seasonal ARIMA model for time series with seasonality",
        "exponential_smoothing": "Exponential smoothing for trend and seasonality",
        "prophet": "Facebook Prophet for complex seasonality patterns",
        "random_forest": "Random Forest regression for feature-rich forecasting",
        "gradient_boosting": "Gradient Boosting for high accuracy forecasting",
        "ensemble": "Ensemble approach combining multiple models"
    }
    return {"models": models}

if __name__ == "__main__":
    # When run directly, start the API server
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("forecasting_api:app", host="0.0.0.0", port=port, reload=False) 