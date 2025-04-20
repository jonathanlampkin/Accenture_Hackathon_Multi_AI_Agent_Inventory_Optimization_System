"""
Inventory Optimization API

This module provides API endpoints for the Multi-AI Agent Inventory Optimization System.
It includes endpoints for forecasting, optimization, and reporting functions.
"""

import os
import sys
import logging
import traceback
from typing import List, Optional, Dict
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Data handling and visualization
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reporting tools
from src.tools.reporting_tools import (
    GenerateInventoryStatusReportTool,
    GenerateForecastReportTool,
    GeneratePolicyEvaluationReportTool,
    GenerateSupplyChainPerformanceReportTool,
    GenerateDashboardTool
)

# Import forecaster
from improved_forecasting import ImprovedForecaster

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
app = FastAPI(title="Inventory Optimization API", version="1.0.0")

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
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = f"api/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Validate the file
        try:
            df = pd.read_csv(file_path)
            required_columns = ['Date', 'Product ID', 'Sales Quantity']
            for col in required_columns:
                if col not in df.columns:
                    raise HTTPException(status_code=400, 
                                       detail=f"Missing required column: {col}")
            
            return {"filename": file.filename, "status": "success", 
                   "rows": len(df), "columns": list(df.columns)}
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
    try:
        input_file = os.path.join("api/uploads", filename)
        if not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Create output directory name based on filename
        output_name = os.path.splitext(filename)[0]
        output_dir = os.path.join("api/results", output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = pd.read_csv(input_file)
        logger.info(f"Loaded data with {len(data)} rows and {data.columns.tolist()} columns")
        
        # Instantiate forecaster with data
        forecaster = ImprovedForecaster(data)
        
        # Process data and generate forecasts
        results = forecaster.forecast_all_products(forecast_horizon=horizon, max_products=None)
        
        # Store results in the forecaster and visualize
        forecaster.results = {
            row['Product ID']: {
                'forecast': row['Forecast'],
                'dates': row['Date'],
                'method': row['Method']
            }
            for _, row in results.iterrows()
        }
        
        # Generate visualizations
        forecaster.visualize_forecasts()
        
        # Generate summary report (we may need to create this manually)
        summary_df = results.groupby('Product ID').agg({
            'Forecast': ['mean', 'max', 'min'],
            'Method': 'first'
        }).reset_index()
        
        # Flatten the column names
        summary_df.columns = ['Product ID', 'Mean Forecast', 'Max Forecast', 'Min Forecast', 'Method']
        
        # Save the summary
        summary_path = os.path.join(output_dir, "forecast_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Get the summary for the response
        if os.path.exists(summary_path):
            summary = summary_df.to_dict(orient="records")
        else:
            summary = []
        
        # Create a list of visualization files that were generated
        visualizations = []
        for file in os.listdir(output_dir):
            if file.endswith(".png"):
                visualizations.append(f"/results/{output_name}/{file}")
        
        return {
            "status": "success", 
            "forecast_count": len(results) if results is not None else 0,
            "summary": summary,
            "visualizations": visualizations,
            "output_directory": output_dir
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
    try:
        input_file = os.path.join("api/uploads", filename)
        if not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Create output directory
        output_name = os.path.splitext(filename)[0]
        output_dir = os.path.join("api/results", output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data and create forecaster
        data = pd.read_csv(input_file)
        forecaster = ImprovedForecaster(data)
        
        # Generate forecast for the specific product
        result = forecaster.forecast_product(
            product_id=product_id,
            forecast_horizon=horizon
        )
        
        if result is None or not result:
            raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found")
        
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
            "product_id": product_id,
            "method": result['method'],
            "forecast": result['forecast'].tolist() if hasattr(result['forecast'], 'tolist') else result['forecast'],
            "dates": [str(d) for d in result['forecast_dates']],
            "visualization": f"/results/{output_name}/{viz_file}"
        }
        
        if 'metrics' in result:
            response.update(result['metrics'])
        
        return response
        
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
    try:
        # Validate required paths
        required_paths = ['inventory_data_path', 'sales_data_path']
        for path in required_paths:
            if path not in request.data_paths:
                raise HTTPException(status_code=400, detail=f"Missing required data path: {path}")
            
            if not os.path.exists(request.data_paths[path]):
                raise HTTPException(status_code=404, detail=f"File not found: {request.data_paths[path]}")
        
        # Create output path if not provided
        if not request.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("api", "results", f"inventory_status_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"inventory_status.{request.output_format.lower()}"
            request.output_path = os.path.join(output_dir, output_file)
        
        # Create tool and generate report
        inventory_report_tool = GenerateInventoryStatusReportTool()
        result = inventory_report_tool.run(
            inventory_data_path=request.data_paths['inventory_data_path'],
            sales_data_path=request.data_paths['sales_data_path'],
            policy_data_path=request.data_paths.get('policy_data_path'),
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