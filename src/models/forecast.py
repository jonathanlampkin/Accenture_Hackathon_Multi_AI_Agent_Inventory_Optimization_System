"""
Forecast-related database models.
"""
import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from src.models.database import Base

class ModelType(enum.Enum):
    """Forecasting model type enumeration."""
    SARIMA = "sarima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"

class ForecastStatus(enum.Enum):
    """Forecast status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ForecastModel(Base):
    """Model for storing trained forecasting models."""
    __tablename__ = "forecast_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    model_type = Column(Enum(ModelType), nullable=False)
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    mlflow_run_id = Column(String(255), nullable=True)
    mlflow_model_uri = Column(String(255), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    forecasts = relationship("Forecast", back_populates="model")
    
    def __repr__(self) -> str:
        return f"<ForecastModel {self.model_type.value}: {self.name}>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_forecast_models_model_type', 'model_type'),  # For filtering by model type
        Index('ix_forecast_models_is_active', 'is_active'),  # For filtering active models
        Index('ix_forecast_models_product_id', 'product_id'),  # For finding models for specific products
        Index('ix_forecast_models_mlflow_run_id', 'mlflow_run_id'),  # For finding models by MLflow run ID
    )

class Forecast(Base):
    """Forecast model for storing forecast results."""
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("forecast_models.id"), nullable=True)
    forecast_dates = Column(JSON, nullable=False)  # List of date strings
    forecast_values = Column(JSON, nullable=False)  # List of forecasted values
    lower_bounds = Column(JSON, nullable=True)  # List of lower confidence bounds
    upper_bounds = Column(JSON, nullable=True)  # List of upper confidence bounds
    metrics = Column(JSON, nullable=True)  # Dictionary of forecast metrics
    horizon = Column(Integer, nullable=False)  # Number of periods forecasted
    status = Column(Enum(ForecastStatus), nullable=False, default=ForecastStatus.PENDING)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="forecasts")
    model = relationship("ForecastModel", back_populates="forecasts")
    
    def __repr__(self) -> str:
        return f"<Forecast for Product {self.product_id} - {self.horizon} periods>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_forecasts_product_id', 'product_id'),  # For finding forecasts for specific products
        Index('ix_forecasts_status', 'status'),  # For filtering by status
        Index('ix_forecasts_created_at', 'created_at'),  # For time-based filtering
        Index('ix_forecasts_model_id', 'model_id'),  # For finding forecasts by model
        # Composite index for finding latest forecasts for each product
        Index('ix_forecasts_product_id_created_at', 'product_id', 'created_at'),
    )

class ForecastJob(Base):
    """Forecast job model for tracking forecast generation jobs."""
    __tablename__ = "forecast_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)  # Null for all products
    job_id = Column(String(255), nullable=False)  # Celery task ID
    status = Column(Enum(ForecastStatus), nullable=False, default=ForecastStatus.PENDING)
    parameters = Column(JSON, nullable=True)  # Job parameters
    result = Column(JSON, nullable=True)  # Job result
    error_message = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        if self.product_id:
            return f"<ForecastJob for Product {self.product_id} - {self.status.value}>"
        else:
            return f"<ForecastJob for All Products - {self.status.value}>"
    
    # Additional indexes for common queries
    __table_args__ = (
        Index('ix_forecast_jobs_job_id', 'job_id'),  # For finding jobs by Celery task ID
        Index('ix_forecast_jobs_status', 'status'),  # For filtering by status
        Index('ix_forecast_jobs_product_id', 'product_id'),  # For finding jobs for specific products
        Index('ix_forecast_jobs_created_by', 'created_by'),  # For finding jobs created by specific users
        Index('ix_forecast_jobs_created_at', 'created_at'),  # For time-based filtering
    ) 