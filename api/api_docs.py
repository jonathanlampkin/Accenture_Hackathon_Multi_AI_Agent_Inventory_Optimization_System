"""
API Documentation Module

This module configures API documentation for the Inventory Optimization API.
"""

def setup_api_docs(app):
    """
    Set up API documentation for FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    # Configure OpenAPI settings
    app.title = "Inventory Optimization API"
    app.description = """
    # Inventory Optimization API
    
    This API provides endpoints for inventory forecasting and optimization.
    
    ## Features
    
    * **File Management**: Upload inventory data files and list available files
    * **Forecasting**: Generate demand forecasts for products
    * **Reporting**: Generate various inventory management reports
    * **Health Checks**: Monitor API health
    
    ## Authentication
    
    Some endpoints require authentication using JWT tokens.
    """
    app.version = "1.0.0"
    app.openapi_tags = [
        {
            "name": "Health",
            "description": "Health check endpoints",
        },
        {
            "name": "Files",
            "description": "File management endpoints",
        },
        {
            "name": "Forecasting",
            "description": "Demand forecasting endpoints",
        },
        {
            "name": "Reports",
            "description": "Report generation endpoints",
        }
    ]
    
    # Configure Swagger UI settings
    app.swagger_ui_parameters = {
        "defaultModelsExpandDepth": -1,  # Hide schemas section by default
        "deepLinking": True,             # Allow deep linking to operations
        "displayRequestDuration": True,  # Show request duration
        "filter": True,                  # Enable filtering operations
    }
    
    return app 