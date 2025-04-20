# Inventory Optimization System

This repository contains an AI-powered inventory optimization system with specialized forecasting capabilities, including ensemble methods and product-specific model selection.

## Features

- **Ensemble Forecasting**: Combines multiple models (SARIMA, Exponential Smoothing, Prophet, Random Forest, Gradient Boosting) for improved accuracy
- **Product-specific Model Selection**: Automatically assigns the most suitable forecasting model to each product
- **API Interface**: Provides a RESTful API for easy integration with other systems
- **Web UI**: Includes a user-friendly web interface for uploading data and visualizing forecasts
- **Performance Metrics**: Calculates RMSE, MAE, and R² metrics to evaluate forecast quality

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`)

### Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the API Server

Start the API server with:

```bash
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000, and the documentation can be accessed at http://localhost:8000/docs.

### Uploading Data

The system requires historical demand data in CSV format with the following columns:
- `Date`: Date of the demand record
- `Product ID`: Identifier for the product
- `Sales Quantity`: The demand/sales quantity
- Additional columns like `Category`, `Price`, etc. can improve forecast accuracy

You can upload data through:
1. The web UI at http://localhost:8000
2. The API endpoint `/upload`
3. Placing a CSV file in the `api/uploads` directory

### Generating Forecasts

#### Using the Web UI

1. Open http://localhost:8000 in your browser
2. Navigate to the "Generate Forecasts" tab
3. Select your data file and set parameters
4. Click "Generate Forecasts"

#### Using the API Client

Run the API client script:

```bash
python api_client.py
```

This will:
1. Connect to the API
2. List available data files
3. Generate forecasts for products 101-105
4. Display results and save visualizations in a timestamped directory

#### Using the API Directly

Send POST requests to:
- `/forecast-product` for forecasting a specific product
- `/forecast` for forecasting all products in a file

Example:
```bash
curl -X POST -F "filename=demand_data.csv" -F "product_id=101" -F "horizon=30" http://localhost:8000/forecast-product
```

## API Endpoints

- `GET /health`: Check API health status
- `GET /`: Access the web UI
- `GET /files`: List available data files
- `POST /upload`: Upload a new data file
- `POST /forecast`: Generate forecasts for all products
- `POST /forecast-product`: Generate forecast for a specific product
- `GET /results/{filename}`: Retrieve generated files (visualizations, etc.)

## Understanding the Forecasts

- The system selects the best forecasting method for each product from:
  - **SARIMA**: For products with strong seasonal patterns
  - **Exponential Smoothing**: For products with trends and simpler seasonal patterns
  - **Prophet**: For products with multiple seasonal patterns and holidays
  - **Random Forest/Gradient Boosting**: For products with important categorical features
  - **Ensemble**: Combines multiple models for more robust forecasts

- Each forecast includes:
  - Predicted demand values for the forecast horizon
  - Confidence intervals (where available)
  - Performance metrics (RMSE, MAE, R²)
  - Visualizations showing historical data, test data, and forecasts

## Customization

- Modify `improved_forecasting.py` to adjust forecasting parameters
- Add new models or ensemble techniques
- Update `main.py` to add new API endpoints

## License

This project is licensed under the MIT License - see the LICENSE file for details.