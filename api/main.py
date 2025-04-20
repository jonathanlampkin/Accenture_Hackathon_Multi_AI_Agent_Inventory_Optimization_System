"""
Inventory Optimization API

This module contains the FastAPI application for inventory forecasting and optimization.
It provides endpoints for uploading files, generating forecasts, and generating reports.
"""

import os
import logging
import traceback
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import reporting tools
try:
    from src.tools.reporting_tools import (
        GenerateInventoryStatusReportTool,
        GenerateForecastReportTool,
        GeneratePolicyEvaluationReportTool,
        GenerateSupplyChainPerformanceReportTool,
        GenerateDashboardTool
    )
except ImportError as e:
    logging.warning(f"Error importing reporting tools: {e}")
    # Create dummy versions of the tools that return a simple response
    from src.tools.reporting_tools import ReportingTool
    GenerateInventoryStatusReportTool = ReportingTool
    GenerateForecastReportTool = ReportingTool
    GeneratePolicyEvaluationReportTool = ReportingTool
    GenerateSupplyChainPerformanceReportTool = ReportingTool
    GenerateDashboardTool = ReportingTool

# Import forecaster
try:
    from improved_forecasting import ImprovedForecaster
except ImportError as e:
    logging.warning(f"Error importing improved forecaster: {e}")
    # Create a simple forecaster class
    class ImprovedForecaster:
        def __init__(self, data=None):
            self.data = data
            self.results = {}
            
        def forecast(self, data=None, product_id=None, horizon=30, method="auto"):
            if data is None:
                data = self.data
            forecast_values = [10] * horizon
            return {
                'product_id': product_id,
                'horizon': horizon,
                'method': method,
                'forecast': forecast_values,
                'timestamp': datetime.now().isoformat()
            }
            
        def forecast_all_products(self, forecast_horizon=30, max_products=None):
            import pandas as pd
            df = pd.DataFrame({
                'Product ID': [101, 102, 103],
                'Date': [datetime.now().strftime('%Y-%m-%d')] * 3,
                'Forecast': [[10] * forecast_horizon, [20] * forecast_horizon, [30] * forecast_horizon],
                'Method': ['Average'] * 3
            })
            return df
            
        def visualize_forecasts(self):
            # Do nothing
            pass

# Import rate limiter
try:
    from src.utils.rate_limiter import add_rate_limiting
except ImportError as e:
    logging.warning(f"Error importing rate limiter: {e}")
    # Create dummy rate limiter
    def add_rate_limiting(app, resources=None):
        pass

# Import metrics
try:
    from src.utils.metrics import add_metrics
except ImportError as e:
    logging.warning(f"Error importing metrics: {e}")
    # Create dummy metrics
    def add_metrics(app):
        pass

# Import JWT
try:
    from src.auth.jwt import oauth2_scheme, verify_token
except ImportError as e:
    logging.warning(f"Error importing JWT: {e}")
    # Create dummy JWT functions
    from fastapi.security import OAuth2PasswordBearer
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    def verify_token(token: str):
        return {"sub": "user"}

# Import API documentation
try:
    from api.api_docs import setup_api_docs
except ImportError as e:
    logging.warning(f"Error importing API docs: {e}")
    # Create dummy API docs
    def setup_api_docs(app):
        pass

# Import our new modules
from api.forecasting import ImprovedForecaster
from api.reporting import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api/api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Inventory Optimization System",
    description="API for inventory optimization with forecasting and machine learning",
    version="1.0.0",
)

# Setup API documentation
setup_api_docs(app)

# Add metrics middleware
add_metrics(app)

# Apply rate limiting to the API
add_rate_limiting(
    app,
    resources={
        # General API rate limits (100 requests per minute)
        "/": (100, 60),
        
        # File upload operations (10 per minute)
        "/upload": (10, 60),
        
        # Forecast operations (20 per minute)
        "/forecast": (20, 60),
        "/forecast-product": (30, 60),
        
        # Report generation operations (5 per minute)
        "/reports/inventory-status": (5, 60),
        "/reports/forecast": (5, 60),
        "/reports/policy-evaluation": (5, 60),
        "/reports/supply-chain": (5, 60),
        "/reports/dashboard": (5, 60),
    }
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Templates
templates = Jinja2Templates(directory="api/templates")

# Create data directories if they don't exist
os.makedirs("api/uploads", exist_ok=True)
os.makedirs("api/results", exist_ok=True)

class ForecastRequest(BaseModel):
    product_id: int
    horizon: int = 30
    method: Optional[str] = None

class ForecastResponse(BaseModel):
    product_id: int
    method: str
    forecast: List[float]
    dates: List[str]
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=JSONResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form(...)
):
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("api/uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = f"api/uploads/{file_type}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Validate the file based on its type
        try:
            df = pd.read_csv(file_path)
            validation_passed = True
            error_message = ""
            
            # Define required columns based on file type
            required_columns = []
            if file_type == "demand":
                required_columns = ['Date', 'Product ID', 'Sales Quantity']
            elif file_type == "inventory":
                required_columns = ['Product ID', 'Location', 'Quantity', 'Last Update']
            elif file_type == "product":
                required_columns = ['Product ID', 'Name', 'Category', 'Price', 'Weight']
            elif file_type == "location":
                required_columns = ['Location ID', 'Name', 'Region', 'Type']
            else:
                validation_passed = False
                error_message = f"Unknown file type: {file_type}"
            
            # Check for required columns
            for col in required_columns:
                if col not in df.columns:
                    validation_passed = False
                    error_message = f"Missing required column: {col}"
                    break
            
            if validation_passed:
                return {
                    "filename": f"{file_type}_{file.filename}",
                    "status": "success", 
                    "file_type": file_type,
                    "rows": len(df),
                    "columns": list(df.columns)
                }
            else:
                # Remove the invalid file
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(status_code=400, detail=error_message)
                
        except Exception as e:
            # Remove the invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/files", response_class=JSONResponse)
async def list_files():
    try:
        files = []
        for filename in os.listdir("api/uploads"):
            if filename.endswith('.csv'):
                file_path = os.path.join("api/uploads", filename)
                file_stats = os.stat(file_path)
                # Get basic info about the file
                try:
                    df = pd.read_csv(file_path)
                    info = {
                        "filename": filename,
                        "size_bytes": file_stats.st_size,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "upload_time": file_stats.st_mtime
                    }
                    files.append(info)
                except:
                    files.append({
                        "filename": filename,
                        "size_bytes": file_stats.st_size,
                        "error": "Could not read file"
                    })
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_class=JSONResponse)
async def generate_forecast(
    filename: str = Form(...),
    horizon: int = Form(30),
    test_proportion: float = Form(0.2)
):
    """Generate forecast from uploaded file."""
    try:
        # Load data
        file_path = os.path.join("api/uploads", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Load data
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data with {len(data)} rows and {data.columns.tolist()} columns")
        
        # Generate forecast using our new forecaster
        result = ImprovedForecaster().forecast(data, horizon=horizon, test_proportion=test_proportion)
        
        # Save results
        result_filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join("api/results", result_filename)
        with open(result_path, "w") as f:
            json.dump(result, f, default=str)
        
        return {
            "filename": result_filename,
            "forecast_count": result["product_count"],
            "horizon": horizon,
            "metrics": result["metrics"]
        }
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    for root, dirs, files in os.walk("api/results"):
        for file in files:
            if file == filename:
                return FileResponse(os.path.join(root, file))
    
    raise HTTPException(status_code=404, detail=f"File not found: {filename}")

@app.post("/forecast-product", response_class=JSONResponse)
async def forecast_product(
    filename: str = Form(...),
    product_id: int = Form(...),
    horizon: int = Form(30)
):
    """Generate forecast for a specific product."""
    try:
        # Load data
        file_path = os.path.join("api/uploads", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Generate forecast using our new forecaster
        result = ImprovedForecaster().forecast_product(data, product_id=product_id, horizon=horizon)
        
        # Save results
        result_filename = f"forecast_product_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join("api/results", result_filename)
        with open(result_path, "w") as f:
            json.dump(result, f, default=str)
        
        return {
            "filename": result_filename,
            "product_id": product_id,
            "horizon": horizon,
            "metrics": result["metrics"]
        }
    except Exception as e:
        logger.error(f"Error forecasting product: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# New endpoints for reporting tools

class ReportRequest(BaseModel):
    data_paths: Dict[str, str]
    product_ids: Optional[List[int]] = None
    output_format: str = "html"
    output_path: Optional[str] = None

@app.post("/reports/inventory-status", response_class=JSONResponse)
async def generate_inventory_status_report(request: ReportRequest):
    """Generate inventory status report."""
    try:
        # Use our new report generator
        result = ReportGenerator().generate_inventory_status_report(
            data_paths=request.data_paths,
            output_format=request.output_format
        )
        
        return {
            "report_path": result["report_path"],
            "report_type": result["report_type"],
            "summary": result["summary"]
        }
    except ValueError as e:
        # Specific error handling for missing paths
        if "Missing required data path" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        logger.error(f"Error generating inventory status report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating inventory status report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/forecast", response_class=JSONResponse)
async def generate_forecast_report(request: ReportRequest):
    try:
        # Validate required paths
        required_paths = ['forecast_data_path', 'historical_data_path']
        for path in required_paths:
            if path not in request.data_paths:
                raise HTTPException(status_code=400, detail=f"Missing required data path: {path}")
            
            if not os.path.exists(request.data_paths[path]):
                raise HTTPException(status_code=404, detail=f"File not found: {request.data_paths[path]}")
        
        # Create output path if not provided
        if not request.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("api", "results", f"forecast_report_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"forecast_report.{request.output_format.lower()}"
            request.output_path = os.path.join(output_dir, output_file)
        
        # Create tool and generate report
        forecast_report_tool = GenerateForecastReportTool()
        result = forecast_report_tool.run(
            forecast_data_path=request.data_paths['forecast_data_path'],
            historical_data_path=request.data_paths['historical_data_path'],
            product_ids=request.product_ids,
            output_format=request.output_format,
            output_path=request.output_path
        )
        
        # Return the result with relative paths for visualizations
        if 'visualizations' in result:
            result['visualizations'] = [
                f"/reports/view?path={os.path.join(os.path.dirname(request.output_path), vis)}"
                for vis in result['visualizations']
            ]
        
        # Add link to the report
        result['report_url'] = f"/reports/view?path={request.output_path}"
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating forecast report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/policy-evaluation", response_class=JSONResponse)
async def generate_policy_evaluation_report(request: ReportRequest):
    try:
        # Validate required paths
        required_paths = ['policy_data_path', 'scenario_results_path', 'inventory_data_path']
        for path in required_paths:
            if path not in request.data_paths:
                raise HTTPException(status_code=400, detail=f"Missing required data path: {path}")
            
            if not os.path.exists(request.data_paths[path]):
                raise HTTPException(status_code=404, detail=f"File not found: {request.data_paths[path]}")
        
        # Create output path if not provided
        if not request.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("api", "results", f"policy_evaluation_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"policy_evaluation.{request.output_format.lower()}"
            request.output_path = os.path.join(output_dir, output_file)
        
        # Create tool and generate report
        policy_report_tool = GeneratePolicyEvaluationReportTool()
        result = policy_report_tool.run(
            policy_data_path=request.data_paths['policy_data_path'],
            scenario_results_path=request.data_paths['scenario_results_path'],
            inventory_data_path=request.data_paths['inventory_data_path'],
            product_ids=request.product_ids,
            output_format=request.output_format,
            output_path=request.output_path
        )
        
        # Return the result with relative paths for visualizations
        if 'visualizations' in result:
            result['visualizations'] = [
                f"/reports/view?path={os.path.join(os.path.dirname(request.output_path), vis)}"
                for vis in result['visualizations']
            ]
        
        # Add link to the report
        result['report_url'] = f"/reports/view?path={request.output_path}"
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating policy evaluation report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/supply-chain", response_class=JSONResponse)
async def generate_supply_chain_report(request: ReportRequest):
    try:
        # Validate required paths
        required_paths = ['inventory_data_path', 'order_data_path', 'lead_time_data_path']
        for path in required_paths:
            if path not in request.data_paths:
                raise HTTPException(status_code=400, detail=f"Missing required data path: {path}")
            
            if not os.path.exists(request.data_paths[path]):
                raise HTTPException(status_code=404, detail=f"File not found: {request.data_paths[path]}")
        
        # Create output path if not provided
        if not request.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("api", "results", f"supply_chain_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"supply_chain.{request.output_format.lower()}"
            request.output_path = os.path.join(output_dir, output_file)
        
        # Create tool and generate report
        supply_chain_report_tool = GenerateSupplyChainPerformanceReportTool()
        result = supply_chain_report_tool.run(
            inventory_data_path=request.data_paths['inventory_data_path'],
            order_data_path=request.data_paths['order_data_path'],
            lead_time_data_path=request.data_paths['lead_time_data_path'],
            supplier_data_path=request.data_paths.get('supplier_data_path'),
            product_ids=request.product_ids,
            output_format=request.output_format,
            output_path=request.output_path
        )
        
        # Return the result with relative paths for visualizations
        if 'visualizations' in result:
            result['visualizations'] = [
                f"/reports/view?path={os.path.join(os.path.dirname(request.output_path), vis)}"
                for vis in result['visualizations'] if vis is not None
            ]
        
        # Add link to the report
        result['report_url'] = f"/reports/view?path={request.output_path}"
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating supply chain report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/dashboard", response_class=JSONResponse)
async def generate_dashboard(request: ReportRequest):
    try:
        # Validate required paths
        required_paths = [
            'forecast_data_path', 'inventory_data_path', 'policy_data_path', 
            'scenario_results_path', 'order_data_path', 'lead_time_data_path'
        ]
        for path in required_paths:
            if path not in request.data_paths:
                raise HTTPException(status_code=400, detail=f"Missing required data path: {path}")
            
            if not os.path.exists(request.data_paths[path]):
                raise HTTPException(status_code=404, detail=f"File not found: {request.data_paths[path]}")
        
        # Create output path if not provided
        if not request.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("api", "results", f"dashboard_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            request.output_path = os.path.join(output_dir, "dashboard.html")
        
        # Create tool and generate dashboard
        dashboard_tool = GenerateDashboardTool()
        result = dashboard_tool.run(
            forecast_data_path=request.data_paths['forecast_data_path'],
            inventory_data_path=request.data_paths['inventory_data_path'],
            policy_data_path=request.data_paths['policy_data_path'],
            scenario_results_path=request.data_paths['scenario_results_path'],
            order_data_path=request.data_paths['order_data_path'],
            lead_time_data_path=request.data_paths['lead_time_data_path'],
            supplier_data_path=request.data_paths.get('supplier_data_path'),
            product_ids=request.product_ids,
            output_path=request.output_path
        )
        
        # Add link to the dashboard
        result['dashboard_url'] = f"/reports/view?path={request.output_path}"
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/view", response_class=FileResponse)
async def view_report(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    
    return FileResponse(path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 