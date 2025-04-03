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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class IntegrationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class IntegrationType(Enum):
    API = "api"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    CUSTOM = "custom"

@dataclass
class IntegrationConfig:
    config_id: str
    integration_type: IntegrationType
    config: Dict[str, Any]
    version: str
    created_at: datetime
    metadata: Dict[str, Any] = None

class IntegrationManager:
    def __init__(self, config_file: str = "integration_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        self.integration_queue = Queue()
        self.integration_configs: Dict[str, IntegrationConfig] = {}
        self.integration_status: Dict[str, IntegrationStatus] = {}
        self.integration_history: Dict[str, List[Dict[str, Any]]] = {}
        self.running = False
        
        # Initialize logging
        self.logger = logging.getLogger("integration_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path("logs") / f"integration_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
        
        # Initialize HTTP session
        self.session = self._create_session()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load integration configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def start(self) -> None:
        """Start the integration manager"""
        if self.running:
            return
            
        self.running = True
        
        # Start integration processing
        self.process_thread = threading.Thread(target=self._process_integrations)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self.logger.info("Integration manager started")
    
    def stop(self) -> None:
        """Stop the integration manager"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for thread to finish
        self.process_thread.join()
        
        self.logger.info("Integration manager stopped")
    
    def _process_integrations(self) -> None:
        """Process integration tasks"""
        while self.running:
            try:
                integration_task = self.integration_queue.get(timeout=1)
                self._handle_integration(integration_task)
            except Exception:
                continue
    
    def _handle_integration(self, integration_task: Dict[str, Any]) -> None:
        """Handle an integration task"""
        try:
            integration_id = integration_task["integration_id"]
            config_id = integration_task["config_id"]
            action = integration_task["action"]
            data = integration_task.get("data")
            
            # Get integration config
            if config_id not in self.integration_configs:
                raise ValueError(f"Integration config not found: {config_id}")
                
            config = self.integration_configs[config_id]
            
            # Update integration status
            self._update_integration_status(integration_id, IntegrationStatus.RUNNING)
            
            # Execute integration
            result = self._execute_integration(config, action, data)
            
            # Update integration status
            self._update_integration_status(integration_id, IntegrationStatus.COMPLETED)
            
            self.logger.info(f"Completed integration: {integration_id}")
            
        except Exception as e:
            self.logger.error(f"Error in integration: {str(e)}")
            self._update_integration_status(integration_id, IntegrationStatus.FAILED)
            self._handle_integration_error(integration_id, str(e))
    
    def _execute_integration(
        self,
        config: IntegrationConfig,
        action: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute an integration"""
        if config.integration_type == IntegrationType.API:
            return self._execute_api_integration(config, action, data)
        elif config.integration_type == IntegrationType.DATABASE:
            return self._execute_database_integration(config, action, data)
        elif config.integration_type == IntegrationType.MESSAGE_QUEUE:
            return self._execute_message_queue_integration(config, action, data)
        elif config.integration_type == IntegrationType.FILE_SYSTEM:
            return self._execute_file_system_integration(config, action, data)
        else:
            raise ValueError(f"Unknown integration type: {config.integration_type}")
    
    def _execute_api_integration(
        self,
        config: IntegrationConfig,
        action: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute API integration"""
        url = config.config.get("url")
        method = config.config.get("method", "GET")
        headers = config.config.get("headers", {})
        params = config.config.get("params", {})
        
        if method == "GET":
            response = self.session.get(url, headers=headers, params=params)
        elif method == "POST":
            response = self.session.post(url, headers=headers, json=data)
        elif method == "PUT":
            response = self.session.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = self.session.delete(url, headers=headers)
        else:
            raise ValueError(f"Unknown HTTP method: {method}")
            
        response.raise_for_status()
        return response.json()
    
    def _execute_database_integration(
        self,
        config: IntegrationConfig,
        action: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute database integration"""
        # This would require a database driver
        # For example, using SQLAlchemy or psycopg2
        raise NotImplementedError("Database integration not implemented")
    
    def _execute_message_queue_integration(
        self,
        config: IntegrationConfig,
        action: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute message queue integration"""
        # This would require a message queue driver
        # For example, using RabbitMQ or Kafka
        raise NotImplementedError("Message queue integration not implemented")
    
    def _execute_file_system_integration(
        self,
        config: IntegrationConfig,
        action: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute file system integration"""
        path = config.config.get("path")
        
        if action == "read":
            with open(path, "r") as f:
                return f.read()
        elif action == "write":
            with open(path, "w") as f:
                f.write(str(data))
        else:
            raise ValueError(f"Unknown file system action: {action}")
    
    def _update_integration_status(
        self,
        integration_id: str,
        status: IntegrationStatus
    ) -> None:
        """Update integration status"""
        if integration_id not in self.integration_history:
            self.integration_history[integration_id] = []
            
        self.integration_history[integration_id].append({
            "status": status.value,
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info(f"Updated integration status: {integration_id} - {status.value}")
    
    def _handle_integration_error(self, integration_id: str, error: str) -> None:
        """Handle an integration error"""
        error_data = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        if integration_id not in self.integration_history:
            self.integration_history[integration_id] = []
            
        self.integration_history[integration_id].append(error_data)
        
        self.logger.error(f"Integration error: {integration_id} - {error}")
    
    def add_integration_config(
        self,
        integration_type: IntegrationType,
        config: Dict[str, Any],
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add integration config"""
        config_id = str(uuid.uuid4())
        
        self.integration_configs[config_id] = IntegrationConfig(
            config_id=config_id,
            integration_type=integration_type,
            config=config,
            version=version,
            created_at=datetime.now(),
            metadata=metadata
        )
        
        self.logger.info(f"Added integration config: {config_id}")
        return config_id
    
    def start_integration(
        self,
        config_id: str,
        action: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start an integration task"""
        integration_id = str(uuid.uuid4())
        
        integration_task = {
            "integration_id": integration_id,
            "config_id": config_id,
            "action": action,
            "data": data
        }
        
        # Add to integration queue
        self.integration_queue.put(integration_task)
        
        # Update integration status
        self._update_integration_status(integration_id, IntegrationStatus.PENDING)
        
        self.logger.info(f"Started integration: {integration_id}")
        return integration_id
    
    def get_integration_status(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Get integration status"""
        if integration_id not in self.integration_history:
            return None
            
        history = self.integration_history[integration_id]
        if not history:
            return None
            
        return {
            "integration_id": integration_id,
            "status": history[-1]["status"],
            "timestamp": history[-1]["timestamp"]
        }
    
    def get_integration_config(self, config_id: str) -> Optional[IntegrationConfig]:
        """Get integration config"""
        return self.integration_configs.get(config_id)
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        metrics = {
            "total_configs": len(self.integration_configs),
            "integration_types": {
                integration_type.value: 0
                for integration_type in IntegrationType
            },
            "status_counts": {
                status.value: 0
                for status in IntegrationStatus
            }
        }
        
        for config in self.integration_configs.values():
            metrics["integration_types"][config.integration_type.value] += 1
            
        for history in self.integration_history.values():
            if history:
                metrics["status_counts"][history[-1]["status"]] += 1
                
        return metrics
    
    def save_integration_report(self) -> None:
        """Save integration report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_integration_metrics(),
            "configs": {
                config_id: {
                    "integration_type": config.integration_type.value,
                    "version": config.version,
                    "created_at": config.created_at.isoformat(),
                    "metadata": config.metadata
                }
                for config_id, config in self.integration_configs.items()
            },
            "integration_history": self.integration_history
        }
        
        report_file = Path("reports") / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 