from typing import Dict, List, Any, Optional, Callable
import json
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import time
import uuid
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    REINFORCEMENT = "reinforcement"

@dataclass
class TrainingData:
    data_id: str
    features: np.ndarray
    labels: np.ndarray
    metadata: Dict[str, Any] = None

@dataclass
class Model:
    model_id: str
    model_type: ModelType
    model: Any
    version: str
    created_at: datetime
    metadata: Dict[str, Any] = None

class TrainingManager:
    def __init__(self, config_file: str = "training_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        self.training_queue = Queue()
        self.training_data: Dict[str, TrainingData] = {}
        self.models: Dict[str, Model] = {}
        self.training_history: Dict[str, List[Dict[str, Any]]] = {}
        self.running = False
        
        # Initialize logging
        self.logger = logging.getLogger("training_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path("logs") / f"training_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}
    
    def start(self) -> None:
        """Start the training manager"""
        if self.running:
            return
            
        self.running = True
        
        # Start training processing
        self.process_thread = threading.Thread(target=self._process_training)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self.logger.info("Training manager started")
    
    def stop(self) -> None:
        """Stop the training manager"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for thread to finish
        self.process_thread.join()
        
        self.logger.info("Training manager stopped")
    
    def _process_training(self) -> None:
        """Process training tasks"""
        while self.running:
            try:
                training_task = self.training_queue.get(timeout=1)
                self._handle_training(training_task)
            except Exception:
                continue
    
    def _handle_training(self, training_task: Dict[str, Any]) -> None:
        """Handle a training task"""
        try:
            model_id = training_task["model_id"]
            data_id = training_task["data_id"]
            model_type = training_task["model_type"]
            hyperparameters = training_task["hyperparameters"]
            
            # Get training data
            if data_id not in self.training_data:
                raise ValueError(f"Training data not found: {data_id}")
                
            data = self.training_data[data_id]
            
            # Update training status
            self._update_training_status(model_id, TrainingStatus.RUNNING)
            
            # Train model
            model = self._train_model(model_type, data, hyperparameters)
            
            # Evaluate model
            metrics = self._evaluate_model(model, data)
            
            # Save model
            self._save_model(model_id, model, model_type, metrics)
            
            # Update training status
            self._update_training_status(model_id, TrainingStatus.COMPLETED)
            
            self.logger.info(f"Completed training: {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            self._update_training_status(model_id, TrainingStatus.FAILED)
            self._handle_training_error(model_id, str(e))
    
    def _train_model(
        self,
        model_type: ModelType,
        data: TrainingData,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """Train a model"""
        # This is a placeholder for actual model training
        # In a real implementation, this would use a machine learning library
        # like scikit-learn, TensorFlow, or PyTorch
        
        if model_type == ModelType.CLASSIFICATION:
            # Example: Train a classification model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**hyperparameters)
            model.fit(data.features, data.labels)
            
        elif model_type == ModelType.REGRESSION:
            # Example: Train a regression model
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**hyperparameters)
            model.fit(data.features, data.labels)
            
        elif model_type == ModelType.CLUSTERING:
            # Example: Train a clustering model
            from sklearn.cluster import KMeans
            model = KMeans(**hyperparameters)
            model.fit(data.features)
            
        elif model_type == ModelType.REINFORCEMENT:
            # Example: Train a reinforcement learning model
            # This would require a more complex implementation
            raise NotImplementedError("Reinforcement learning not implemented")
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model
    
    def _evaluate_model(self, model: Any, data: TrainingData) -> Dict[str, float]:
        """Evaluate a model"""
        # This is a placeholder for actual model evaluation
        # In a real implementation, this would use appropriate metrics
        # based on the model type
        
        predictions = model.predict(data.features)
        
        if isinstance(model, (RandomForestClassifier, KMeans)):
            metrics = {
                "accuracy": accuracy_score(data.labels, predictions),
                "precision": precision_score(data.labels, predictions, average='weighted'),
                "recall": recall_score(data.labels, predictions, average='weighted'),
                "f1": f1_score(data.labels, predictions, average='weighted')
            }
        elif isinstance(model, RandomForestRegressor):
            metrics = {
                "mse": np.mean((data.labels - predictions) ** 2),
                "mae": np.mean(np.abs(data.labels - predictions)),
                "r2": model.score(data.features, data.labels)
            }
        else:
            metrics = {}
            
        return metrics
    
    def _save_model(
        self,
        model_id: str,
        model: Any,
        model_type: ModelType,
        metrics: Dict[str, float]
    ) -> None:
        """Save a model"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.models[model_id] = Model(
            model_id=model_id,
            model_type=model_type,
            model=model,
            version=version,
            created_at=datetime.now(),
            metadata={"metrics": metrics}
        )
        
        # Save model to file
        model_file = Path("models") / f"{model_id}_{version}.pkl"
        model_file.parent.mkdir(exist_ok=True)
        
        # This would require a proper serialization method
        # For example, using joblib or pickle
        # import joblib
        # joblib.dump(model, model_file)
        
        self.logger.info(f"Saved model: {model_id}")
    
    def _update_training_status(
        self,
        model_id: str,
        status: TrainingStatus
    ) -> None:
        """Update training status"""
        if model_id not in self.training_history:
            self.training_history[model_id] = []
            
        self.training_history[model_id].append({
            "status": status.value,
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info(f"Updated training status: {model_id} - {status.value}")
    
    def _handle_training_error(self, model_id: str, error: str) -> None:
        """Handle a training error"""
        error_data = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        if model_id not in self.training_history:
            self.training_history[model_id] = []
            
        self.training_history[model_id].append(error_data)
        
        self.logger.error(f"Training error: {model_id} - {error}")
    
    def add_training_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add training data"""
        data_id = str(uuid.uuid4())
        
        self.training_data[data_id] = TrainingData(
            data_id=data_id,
            features=features,
            labels=labels,
            metadata=metadata
        )
        
        self.logger.info(f"Added training data: {data_id}")
        return data_id
    
    def start_training(
        self,
        model_type: ModelType,
        data_id: str,
        hyperparameters: Dict[str, Any]
    ) -> str:
        """Start a training task"""
        model_id = str(uuid.uuid4())
        
        training_task = {
            "model_id": model_id,
            "data_id": data_id,
            "model_type": model_type,
            "hyperparameters": hyperparameters
        }
        
        # Add to training queue
        self.training_queue.put(training_task)
        
        # Update training status
        self._update_training_status(model_id, TrainingStatus.PENDING)
        
        self.logger.info(f"Started training: {model_id}")
        return model_id
    
    def get_training_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get training status"""
        if model_id not in self.training_history:
            return None
            
        history = self.training_history[model_id]
        if not history:
            return None
            
        return {
            "model_id": model_id,
            "status": history[-1]["status"],
            "timestamp": history[-1]["timestamp"]
        }
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a model"""
        return self.models.get(model_id)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        metrics = {
            "total_models": len(self.models),
            "model_types": {
                model_type.value: 0
                for model_type in ModelType
            },
            "status_counts": {
                status.value: 0
                for status in TrainingStatus
            }
        }
        
        for model in self.models.values():
            metrics["model_types"][model.model_type.value] += 1
            
        for history in self.training_history.values():
            if history:
                metrics["status_counts"][history[-1]["status"]] += 1
                
        return metrics
    
    def save_training_report(self) -> None:
        """Save training report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_training_metrics(),
            "models": {
                model_id: {
                    "model_type": model.model_type.value,
                    "version": model.version,
                    "created_at": model.created_at.isoformat(),
                    "metadata": model.metadata
                }
                for model_id, model in self.models.items()
            },
            "training_history": self.training_history
        }
        
        report_file = Path("reports") / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 