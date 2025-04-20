#!/usr/bin/env python
"""
Model maintenance script.

This script performs model maintenance operations including retraining models
and cleaning up old model artifacts to maintain optimal prediction performance 
and storage efficiency.

Example usage:
    python scripts/model_maintenance.py --retrain-all
    python scripts/model_maintenance.py --retrain-high-volume
    python scripts/model_maintenance.py --cleanup-old-models --days 30
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import mlflow
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
log_dir = project_root / "logs" / "models"
os.makedirs(log_dir, exist_ok=True)

log_file = log_dir / f"model_maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("model_maintenance")

# Import after sys.path update
from src.models.database import get_db_session
from src.models.forecast import ForecastModel


class ModelMaintainer:
    """Utility for performing model maintenance operations."""

    def __init__(
        self,
        db_url: str,
        mlflow_tracking_uri: Optional[str] = None,
        model_registry_dir: Optional[str] = None,
    ):
        """Initialize ModelMaintainer.
        
        Args:
            db_url: Database connection URL
            mlflow_tracking_uri: MLflow tracking URI
            model_registry_dir: Directory for model registry
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Set up MLflow tracking
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            default_tracking_uri = "sqlite:///mlflow.db"
            logger.info(f"Using default MLflow tracking URI: {default_tracking_uri}")
            mlflow.set_tracking_uri(default_tracking_uri)
        
        self.model_registry_dir = model_registry_dir or os.path.join(project_root, "models")
        os.makedirs(self.model_registry_dir, exist_ok=True)
    
    def get_high_volume_products(self, threshold: int = 100) -> List[int]:
        """Get high-volume products based on transaction count.
        
        Args:
            threshold: Minimum number of transactions to be considered high-volume
            
        Returns:
            List of product IDs
        """
        with self.SessionLocal() as session:
            result = session.execute(text("""
                SELECT 
                    product_id, 
                    COUNT(*) as transaction_count
                FROM 
                    inventory_transactions
                WHERE 
                    created_at >= NOW() - INTERVAL '30 days'
                GROUP BY 
                    product_id
                HAVING 
                    COUNT(*) >= :threshold
                ORDER BY 
                    transaction_count DESC
            """), {"threshold": threshold})
            
            return [row[0] for row in result]
    
    def get_active_model_runs(self, days: int = 90) -> List[str]:
        """Get active model runs from the past N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of model run IDs
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        runs = mlflow.search_runs(
            filter_string=f"attributes.start_time >= {int(cutoff_date.timestamp() * 1000)}",
            order_by=["attributes.start_time DESC"]
        )
        
        return runs["run_id"].tolist()
    
    def get_registered_models(self) -> Dict[str, Dict]:
        """Get all registered models in MLflow.
        
        Returns:
            Dictionary of model name to model info
        """
        models = mlflow.search_registered_models()
        return {model.name: model for model in models}
    
    def get_model_info_from_db(self) -> Dict[int, Dict]:
        """Get model information from database.
        
        Returns:
            Dictionary of model ID to model info
        """
        with self.SessionLocal() as session:
            models = session.query(ForecastModel).all()
            return {model.id: {
                "name": model.name,
                "type": model.model_type,
                "product_id": model.product_id,
                "location_id": model.location_id,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "is_active": model.is_active,
                "metadata": model.metadata,
                "mlflow_run_id": model.mlflow_run_id,
            } for model in models}
    
    def retrain_model(self, model_id: int = None, product_id: int = None, location_id: int = None) -> str:
        """Retrain a specific model.
        
        Args:
            model_id: ID of the model to retrain
            product_id: Product ID to retrain model for
            location_id: Location ID to retrain model for
            
        Returns:
            New MLflow run ID
        """
        # Get model info
        model_info = None
        model_name = None
        
        if model_id is not None:
            with self.SessionLocal() as session:
                model = session.query(ForecastModel).filter(ForecastModel.id == model_id).first()
                if not model:
                    raise ValueError(f"Model with ID {model_id} not found")
                
                model_info = {
                    "name": model.name,
                    "type": model.model_type,
                    "product_id": model.product_id,
                    "location_id": model.location_id,
                    "metadata": model.metadata,
                    "mlflow_run_id": model.mlflow_run_id,
                }
                model_name = model.name
        elif product_id is not None and location_id is not None:
            with self.SessionLocal() as session:
                model = session.query(ForecastModel).filter(
                    ForecastModel.product_id == product_id,
                    ForecastModel.location_id == location_id,
                    ForecastModel.is_active == True
                ).first()
                
                if model:
                    model_info = {
                        "name": model.name,
                        "type": model.model_type,
                        "product_id": model.product_id,
                        "location_id": model.location_id,
                        "metadata": model.metadata,
                        "mlflow_run_id": model.mlflow_run_id,
                    }
                    model_name = model.name
                else:
                    # Create new model name for a new model
                    model_name = f"inventory_forecast_p{product_id}_l{location_id}"
                    model_info = {
                        "name": model_name,
                        "type": "prophet",  # Default model type
                        "product_id": product_id,
                        "location_id": location_id,
                        "metadata": {},
                        "mlflow_run_id": None,
                    }
        else:
            raise ValueError("Either model_id or both product_id and location_id must be provided")
        
        # Get training data
        train_data = self._get_training_data(
            product_id=model_info["product_id"], 
            location_id=model_info["location_id"]
        )
        
        if train_data.empty:
            logger.warning(f"No training data available for product {model_info['product_id']} "
                          f"at location {model_info['location_id']}")
            return None
        
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params({
                "product_id": model_info["product_id"],
                "location_id": model_info["location_id"],
                "model_type": model_info["type"],
                "training_rows": len(train_data),
                "training_start_date": train_data["date"].min(),
                "training_end_date": train_data["date"].max(),
            })
            
            # Train the model based on type
            model_type = model_info["type"]
            model = None
            
            try:
                if model_type == "prophet":
                    from prophet import Prophet
                    
                    # Prepare data for Prophet
                    prophet_df = train_data.rename(columns={"date": "ds", "quantity": "y"})
                    
                    # Train model
                    model = Prophet(
                        daily_seasonality=True,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                    )
                    model.fit(prophet_df)
                    
                    # Save model
                    model_path = os.path.join(self.model_registry_dir, f"{model_name}_{run_id}")
                    os.makedirs(model_path, exist_ok=True)
                    
                    with open(os.path.join(model_path, "model.json"), "w") as f:
                        f.write(model.to_json())
                    
                    # Log metrics
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    
                    # Calculate error metrics on training data
                    train_forecast = forecast[forecast["ds"].isin(prophet_df["ds"])]
                    train_actuals = prophet_df.set_index("ds")["y"]
                    train_preds = train_forecast.set_index("ds")["yhat"]
                    
                    rmse = ((train_actuals - train_preds) ** 2).mean() ** 0.5
                    mape = (abs(train_actuals - train_preds) / train_actuals).mean() * 100
                    
                    mlflow.log_metrics({
                        "rmse": rmse,
                        "mape": mape,
                    })
                    
                    # Log model
                    mlflow.prophet.log_model(model, "model")
                    
                elif model_type == "arima":
                    from statsmodels.tsa.arima.model import ARIMA
                    
                    # Prepare data for ARIMA
                    arima_df = train_data.set_index("date")["quantity"]
                    
                    # Train model
                    model = ARIMA(arima_df, order=(5, 1, 0))
                    model_fit = model.fit()
                    
                    # Log metrics
                    rmse = ((model_fit.fittedvalues - arima_df) ** 2).mean() ** 0.5
                    mape = (abs(model_fit.fittedvalues - arima_df) / arima_df).mean() * 100
                    
                    mlflow.log_metrics({
                        "rmse": rmse,
                        "mape": mape,
                    })
                    
                    # Save model
                    model_path = os.path.join(self.model_registry_dir, f"{model_name}_{run_id}")
                    os.makedirs(model_path, exist_ok=True)
                    model_fit.save(os.path.join(model_path, "model.pkl"))
                    
                    # Log model
                    mlflow.statsmodels.log_model(model_fit, "model")
                
                else:
                    logger.error(f"Unsupported model type: {model_type}")
                    return None
                
                # Update model in database
                self._update_model_in_db(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=model_type,
                    product_id=model_info["product_id"],
                    location_id=model_info["location_id"],
                    mlflow_run_id=run_id,
                    metadata=model_info["metadata"]
                )
                
                logger.info(f"Successfully retrained model {model_name} (run_id: {run_id})")
                return run_id
            
            except Exception as e:
                logger.error(f"Error training model {model_name}: {str(e)}")
                mlflow.log_param("error", str(e))
                return None
    
    def _get_training_data(self, product_id: int, location_id: int) -> pd.DataFrame:
        """Get training data for a product at a location.
        
        Args:
            product_id: Product ID
            location_id: Location ID
            
        Returns:
            DataFrame with training data
        """
        with self.SessionLocal() as session:
            # Query inventory transaction data
            query = text("""
                SELECT 
                    DATE(created_at) as date,
                    SUM(CASE WHEN transaction_type = 'outbound' THEN quantity ELSE 0 END) as quantity
                FROM 
                    inventory_transactions
                WHERE 
                    product_id = :product_id
                    AND location_id = :location_id
                    AND transaction_type = 'outbound'
                    AND created_at >= NOW() - INTERVAL '365 days'
                GROUP BY 
                    DATE(created_at)
                ORDER BY 
                    date
            """)
            
            result = session.execute(query, {"product_id": product_id, "location_id": location_id})
            df = pd.DataFrame(result.fetchall(), columns=["date", "quantity"])
            
            # Fill in missing dates with zeros
            if not df.empty:
                date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
                full_df = pd.DataFrame({"date": date_range})
                df = pd.merge(full_df, df, on="date", how="left")
                df["quantity"] = df["quantity"].fillna(0)
            
            return df
    
    def _update_model_in_db(
        self,
        model_id: Optional[int],
        model_name: str,
        model_type: str,
        product_id: int,
        location_id: int,
        mlflow_run_id: str,
        metadata: Dict
    ) -> int:
        """Update model record in database.
        
        Args:
            model_id: ID of the model to update (None for new model)
            model_name: Model name
            model_type: Model type
            product_id: Product ID
            location_id: Location ID
            mlflow_run_id: MLflow run ID
            metadata: Model metadata
            
        Returns:
            Model ID
        """
        with self.SessionLocal() as session:
            if model_id is not None:
                # Update existing model
                model = session.query(ForecastModel).filter(ForecastModel.id == model_id).first()
                if not model:
                    raise ValueError(f"Model with ID {model_id} not found")
                
                model.updated_at = datetime.now()
                model.mlflow_run_id = mlflow_run_id
                model.is_active = True
                
                # Update metadata
                current_metadata = model.metadata or {}
                current_metadata.update({
                    "last_retrained": datetime.now().isoformat(),
                    "retraining_run_id": mlflow_run_id,
                })
                model.metadata = current_metadata
                
                session.commit()
                return model.id
            else:
                # Check if model exists for product/location
                existing_model = session.query(ForecastModel).filter(
                    ForecastModel.product_id == product_id,
                    ForecastModel.location_id == location_id,
                    ForecastModel.is_active == True
                ).first()
                
                if existing_model:
                    # Update existing model
                    existing_model.updated_at = datetime.now()
                    existing_model.mlflow_run_id = mlflow_run_id
                    
                    # Update metadata
                    current_metadata = existing_model.metadata or {}
                    current_metadata.update({
                        "last_retrained": datetime.now().isoformat(),
                        "retraining_run_id": mlflow_run_id,
                    })
                    existing_model.metadata = current_metadata
                    
                    session.commit()
                    return existing_model.id
                else:
                    # Create new model
                    new_model = ForecastModel(
                        name=model_name,
                        model_type=model_type,
                        product_id=product_id,
                        location_id=location_id,
                        mlflow_run_id=mlflow_run_id,
                        is_active=True,
                        metadata={
                            "created_at": datetime.now().isoformat(),
                            "last_retrained": datetime.now().isoformat(),
                            "retraining_run_id": mlflow_run_id,
                        }
                    )
                    
                    session.add(new_model)
                    session.commit()
                    session.refresh(new_model)
                    return new_model.id
    
    def cleanup_old_models(self, days: int = 30, dry_run: bool = False) -> Dict[str, List[str]]:
        """Clean up old model files and runs.
        
        Args:
            days: Remove models older than this many days
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with deleted run IDs and model files
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_runs = []
        deleted_files = []
        
        # Get active models from database
        with self.SessionLocal() as session:
            active_models = session.query(ForecastModel).filter(
                ForecastModel.is_active == True
            ).all()
            active_run_ids = [m.mlflow_run_id for m in active_models if m.mlflow_run_id]
        
        # Find old runs
        all_runs = mlflow.search_runs(
            filter_string=f"attributes.start_time < {int(cutoff_date.timestamp() * 1000)}"
        )
        
        for _, run in all_runs.iterrows():
            run_id = run["run_id"]
            
            # Skip active runs
            if run_id in active_run_ids:
                continue
            
            # Delete MLflow run
            if not dry_run:
                try:
                    mlflow.delete_run(run_id)
                    deleted_runs.append(run_id)
                    logger.info(f"Deleted MLflow run: {run_id}")
                except Exception as e:
                    logger.error(f"Error deleting MLflow run {run_id}: {str(e)}")
            else:
                deleted_runs.append(run_id)
                logger.info(f"Would delete MLflow run: {run_id}")
        
        # Clean up model files
        for root, dirs, files in os.walk(self.model_registry_dir):
            for dir_name in dirs:
                if "_" in dir_name:
                    # Parse run ID from directory name
                    parts = dir_name.split("_")
                    if len(parts) > 1:
                        run_id = parts[-1]
                        
                        # Skip active runs
                        if run_id in active_run_ids:
                            continue
                        
                        # Get directory modification time
                        dir_path = os.path.join(root, dir_name)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
                        
                        if mod_time < cutoff_date:
                            if not dry_run:
                                try:
                                    import shutil
                                    shutil.rmtree(dir_path)
                                    deleted_files.append(dir_path)
                                    logger.info(f"Deleted model directory: {dir_path}")
                                except Exception as e:
                                    logger.error(f"Error deleting directory {dir_path}: {str(e)}")
                            else:
                                deleted_files.append(dir_path)
                                logger.info(f"Would delete model directory: {dir_path}")
        
        return {
            "deleted_runs": deleted_runs,
            "deleted_files": deleted_files
        }
    
    def retrain_all_models(self) -> Dict[int, str]:
        """Retrain all active models.
        
        Returns:
            Dictionary mapping model ID to new run ID
        """
        results = {}
        
        with self.SessionLocal() as session:
            active_models = session.query(ForecastModel).filter(
                ForecastModel.is_active == True
            ).all()
            
            logger.info(f"Retraining {len(active_models)} active models")
            
            for model in active_models:
                logger.info(f"Retraining model {model.id}: {model.name}")
                try:
                    run_id = self.retrain_model(model_id=model.id)
                    if run_id:
                        results[model.id] = run_id
                except Exception as e:
                    logger.error(f"Error retraining model {model.id}: {str(e)}")
        
        return results
    
    def retrain_high_volume_models(self, threshold: int = 100) -> Dict[int, str]:
        """Retrain models for high-volume products.
        
        Args:
            threshold: Minimum number of transactions to be considered high-volume
            
        Returns:
            Dictionary mapping model ID to new run ID
        """
        results = {}
        
        # Get high-volume products
        high_volume_products = self.get_high_volume_products(threshold=threshold)
        logger.info(f"Found {len(high_volume_products)} high-volume products")
        
        with self.SessionLocal() as session:
            for product_id in high_volume_products:
                # Get active models for this product
                models = session.query(ForecastModel).filter(
                    ForecastModel.product_id == product_id,
                    ForecastModel.is_active == True
                ).all()
                
                if models:
                    # Retrain existing models
                    for model in models:
                        logger.info(f"Retraining high-volume model {model.id}: {model.name}")
                        try:
                            run_id = self.retrain_model(model_id=model.id)
                            if run_id:
                                results[model.id] = run_id
                        except Exception as e:
                            logger.error(f"Error retraining model {model.id}: {str(e)}")
                else:
                    # No existing models, check inventory locations for this product
                    locations_query = text("""
                        SELECT DISTINCT location_id 
                        FROM inventories 
                        WHERE product_id = :product_id
                    """)
                    
                    locations = session.execute(locations_query, {"product_id": product_id})
                    location_ids = [row[0] for row in locations]
                    
                    # Create models for each location
                    for location_id in location_ids:
                        logger.info(f"Creating new model for high-volume product {product_id} at location {location_id}")
                        try:
                            run_id = self.retrain_model(
                                product_id=product_id,
                                location_id=location_id
                            )
                            if run_id:
                                # Get the newly created model ID
                                model = session.query(ForecastModel).filter(
                                    ForecastModel.product_id == product_id,
                                    ForecastModel.location_id == location_id,
                                    ForecastModel.mlflow_run_id == run_id
                                ).first()
                                
                                if model:
                                    results[model.id] = run_id
                        except Exception as e:
                            logger.error(f"Error creating model for product {product_id}, location {location_id}: {str(e)}")
        
        return results


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Perform model maintenance operations."
    )
    
    # Database connection arguments
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inventory"),
        help="Database connection URL",
    )
    
    # MLflow arguments
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI",
    )
    
    parser.add_argument(
        "--model-registry-dir",
        type=str,
        default=os.environ.get("MODEL_REGISTRY_DIR"),
        help="Directory for model registry",
    )
    
    # Operation arguments
    parser.add_argument(
        "--retrain-all",
        action="store_true",
        help="Retrain all active models",
    )
    
    parser.add_argument(
        "--retrain-high-volume",
        action="store_true",
        help="Retrain models for high-volume products",
    )
    
    parser.add_argument(
        "--retrain-model",
        type=int,
        help="Retrain a specific model by ID",
    )
    
    parser.add_argument(
        "--cleanup-old-models",
        action="store_true",
        help="Clean up old model files and runs",
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Threshold for high-volume products",
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days threshold for cleanup operations",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (no actual changes)",
    )
    
    return parser.parse_args()


def main():
    """Run the script."""
    args = parse_args()
    
    # Initialize model maintainer
    maintainer = ModelMaintainer(
        db_url=args.db_url,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        model_registry_dir=args.model_registry_dir,
    )
    
    # Record start time
    start_time = time.time()
    
    # Retrain all models if requested
    if args.retrain_all:
        logger.info("Retraining all active models")
        results = maintainer.retrain_all_models()
        logger.info(f"Retrained {len(results)} models")
    
    # Retrain high-volume models if requested
    if args.retrain_high_volume:
        logger.info(f"Retraining models for high-volume products (threshold: {args.threshold})")
        results = maintainer.retrain_high_volume_models(threshold=args.threshold)
        logger.info(f"Retrained {len(results)} high-volume models")
    
    # Retrain specific model if requested
    if args.retrain_model is not None:
        logger.info(f"Retraining model {args.retrain_model}")
        run_id = maintainer.retrain_model(model_id=args.retrain_model)
        if run_id:
            logger.info(f"Successfully retrained model {args.retrain_model} (run_id: {run_id})")
        else:
            logger.error(f"Failed to retrain model {args.retrain_model}")
    
    # Clean up old models if requested
    if args.cleanup_old_models:
        logger.info(f"Cleaning up models older than {args.days} days (dry run: {args.dry_run})")
        results = maintainer.cleanup_old_models(days=args.days, dry_run=args.dry_run)
        
        logger.info(f"Deleted {len(results['deleted_runs'])} MLflow runs")
        logger.info(f"Deleted {len(results['deleted_files'])} model directories")
    
    # Record end time and total duration
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"All operations completed in {total_duration:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 