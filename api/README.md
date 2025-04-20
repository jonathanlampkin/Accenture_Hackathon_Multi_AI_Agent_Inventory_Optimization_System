# Inventory Optimization API

This directory contains a FastAPI-based API for the Inventory Optimization System. The API provides endpoints for uploading data, generating forecasts, and accessing results.

## Quick Start

Start the API server:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper script from the root directory:

```bash
python start_api.py
```

## API Documentation

API documentation is automatically generated and available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## File Structure

- `main.py`: The main API implementation with all endpoints
- `templates/`: HTML templates for the web UI
- `static/`: Static files (CSS, JS, images)
- `uploads/`: Directory for uploaded CSV files
- `results/`: Directory for forecast results and visualizations

## Endpoints

### Core Endpoints

- `GET /health`: Health check endpoint
- `GET /`: Web UI for the system
- `GET /files`: List available data files
- `POST /upload`: Upload a new data file
- `POST /forecast`: Generate forecasts for all products
- `POST /forecast-product`: Generate forecast for a specific product
- `GET /results/{filename}`: Retrieve generated files (visualizations, etc.)

### Web UI

The web UI provides a user-friendly interface for:
1. Uploading data files
2. Generating forecasts
3. Viewing results and visualizations

## Client Usage

You can use the API programmatically with the provided `api_client.py` script in the root directory:

```bash
python api_client.py
```

This will:
1. Connect to the API
2. Check available files
3. Generate forecasts for products
4. Save visualizations locally

## Data Format

The API expects CSV files with at least the following columns:
- `Date`: Date of the demand record
- `Product ID`: Identifier for the product
- `Sales Quantity`: The demand/sales quantity

Additional columns like `Category`, `Price`, etc. will improve forecast quality.

## Example API Calls

### Upload a file
```bash
curl -X POST -F "file=@data.csv" http://localhost:8000/upload
```

### Generate forecasts
```bash
curl -X POST -F "filename=data.csv" -F "horizon=30" -F "test_proportion=0.2" http://localhost:8000/forecast
```

### Forecast for a specific product
```bash
curl -X POST -F "filename=data.csv" -F "product_id=101" -F "horizon=30" http://localhost:8000/forecast-product
```

## Stopping the API

To stop all running API instances, use the helper script:

```bash
python stop_api.py
``` 