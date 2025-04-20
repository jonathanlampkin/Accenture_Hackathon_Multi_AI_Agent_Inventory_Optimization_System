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
import psutil

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    CUSTOM = "custom"

class ResourceStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    RESERVED = "reserved"
    ERROR = "error"

@dataclass
class Resource:
    resource_id: str
    resource_type: ResourceType
    capacity: float
    status: ResourceStatus
    owner: Optional[str] = None
    metadata: Dict[str, Any] = None

class ResourceManager:
    def __init__(self, config_file: str = "resource_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        self.resources: Dict[str, Resource] = {}
        self.resource_queue = Queue()
        self.cleanup_queue = Queue()
        self.running = False
        
        # Initialize logging
        self.logger = logging.getLogger("resource_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path("logs") / f"resource_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
        
        # Initialize system resources
        self._initialize_system_resources()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load resource configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}
    
    def _initialize_system_resources(self) -> None:
        """Initialize system resources"""
        # CPU resources
        cpu_count = psutil.cpu_count()
        for i in range(cpu_count):
            self.add_resource(
                f"cpu_{i}",
                ResourceType.CPU,
                100.0,  # Percentage
                ResourceStatus.AVAILABLE,
                metadata={"core": i}
            )
        
        # Memory resources
        memory = psutil.virtual_memory()
        self.add_resource(
            "memory",
            ResourceType.MEMORY,
            memory.total,
            ResourceStatus.AVAILABLE,
            metadata={"unit": "bytes"}
        )
        
        # Disk resources
        disk = psutil.disk_usage('/')
        self.add_resource(
            "disk",
            ResourceType.DISK,
            disk.total,
            ResourceStatus.AVAILABLE,
            metadata={"unit": "bytes"}
        )
        
        # Network resources
        self.add_resource(
            "network",
            ResourceType.NETWORK,
            100.0,  # Percentage
            ResourceStatus.AVAILABLE,
            metadata={"unit": "percentage"}
        )
    
    def start(self) -> None:
        """Start the resource manager"""
        if self.running:
            return
            
        self.running = True
        
        # Start resource monitoring
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._process_cleanup)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        self.logger.info("Resource manager started")
    
    def stop(self) -> None:
        """Stop the resource manager"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads to finish
        self.monitor_thread.join()
        self.cleanup_thread.join()
        
        self.logger.info("Resource manager stopped")
    
    def _monitor_resources(self) -> None:
        """Monitor system resources"""
        while self.running:
            try:
                # Monitor CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                for i in range(psutil.cpu_count()):
                    resource_id = f"cpu_{i}"
                    if resource_id in self.resources:
                        self.resources[resource_id].capacity = 100.0 - cpu_percent
                
                # Monitor memory usage
                memory = psutil.virtual_memory()
                if "memory" in self.resources:
                    self.resources["memory"].capacity = memory.available
                
                # Monitor disk usage
                disk = psutil.disk_usage('/')
                if "disk" in self.resources:
                    self.resources["disk"].capacity = disk.free
                
                # Monitor network usage
                net_io = psutil.net_io_counters()
                if "network" in self.resources:
                    # Calculate network usage percentage
                    network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
                    self.resources["network"].capacity = max(0, 100.0 - (network_usage / 1000))  # Assuming 1GB as max
                
                # Sleep for monitoring interval
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {str(e)}")
                time.sleep(5)
    
    def _process_cleanup(self) -> None:
        """Process resource cleanup"""
        while self.running:
            try:
                resource_id = self.cleanup_queue.get(timeout=1)
                self._cleanup_resource(resource_id)
            except Exception:
                continue
    
    def _cleanup_resource(self, resource_id: str) -> None:
        """Cleanup a resource"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            
            # Reset resource status
            resource.status = ResourceStatus.AVAILABLE
            resource.owner = None
            
            self.logger.info(f"Cleaned up resource: {resource_id}")
    
    def add_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        capacity: float,
        status: ResourceStatus = ResourceStatus.AVAILABLE,
        owner: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a resource"""
        self.resources[resource_id] = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            capacity=capacity,
            status=status,
            owner=owner,
            metadata=metadata or {}
        )
        
        self.logger.info(f"Added resource: {resource_id}")
    
    def remove_resource(self, resource_id: str) -> None:
        """Remove a resource"""
        if resource_id in self.resources:
            del self.resources[resource_id]
            self.logger.info(f"Removed resource: {resource_id}")
    
    def allocate_resource(
        self,
        resource_id: str,
        owner: str,
        amount: float
    ) -> bool:
        """Allocate a resource"""
        if resource_id not in self.resources:
            return False
            
        resource = self.resources[resource_id]
        
        if resource.status != ResourceStatus.AVAILABLE:
            return False
            
        if amount > resource.capacity:
            return False
            
        resource.status = ResourceStatus.IN_USE
        resource.owner = owner
        resource.capacity -= amount
        
        self.logger.info(f"Allocated resource: {resource_id} to {owner}")
        return True
    
    def release_resource(self, resource_id: str, owner: str) -> bool:
        """Release a resource"""
        if resource_id not in self.resources:
            return False
            
        resource = self.resources[resource_id]
        
        if resource.owner != owner:
            return False
            
        resource.status = ResourceStatus.AVAILABLE
        resource.owner = None
        
        # Add to cleanup queue
        self.cleanup_queue.put(resource_id)
        
        self.logger.info(f"Released resource: {resource_id} from {owner}")
        return True
    
    def reserve_resource(
        self,
        resource_id: str,
        owner: str,
        timeout: int = 300
    ) -> bool:
        """Reserve a resource"""
        if resource_id not in self.resources:
            return False
            
        resource = self.resources[resource_id]
        
        if resource.status != ResourceStatus.AVAILABLE:
            return False
            
        resource.status = ResourceStatus.RESERVED
        resource.owner = owner
        
        # Schedule timeout
        asyncio.create_task(self._release_reserved_resource(resource_id, timeout))
        
        self.logger.info(f"Reserved resource: {resource_id} for {owner}")
        return True
    
    async def _release_reserved_resource(self, resource_id: str, timeout: int) -> None:
        """Release a reserved resource after timeout"""
        await asyncio.sleep(timeout)
        
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            if resource.status == ResourceStatus.RESERVED:
                self.release_resource(resource_id, resource.owner)
    
    def get_resource_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get resource status"""
        if resource_id not in self.resources:
            return None
            
        resource = self.resources[resource_id]
        return {
            "resource_id": resource.resource_id,
            "resource_type": resource.resource_type.value,
            "capacity": resource.capacity,
            "status": resource.status.value,
            "owner": resource.owner,
            "metadata": resource.metadata
        }
    
    def get_available_resources(
        self,
        resource_type: Optional[ResourceType] = None
    ) -> List[Dict[str, Any]]:
        """Get available resources"""
        available = []
        
        for resource in self.resources.values():
            if resource.status == ResourceStatus.AVAILABLE:
                if resource_type is None or resource.resource_type == resource_type:
                    available.append({
                        "resource_id": resource.resource_id,
                        "resource_type": resource.resource_type.value,
                        "capacity": resource.capacity,
                        "metadata": resource.metadata
                    })
                    
        return available
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get resource metrics"""
        metrics = {
            "total_resources": len(self.resources),
            "resource_types": {},
            "status_counts": {
                status.value: 0
                for status in ResourceStatus
            }
        }
        
        for resource in self.resources.values():
            # Count by resource type
            if resource.resource_type.value not in metrics["resource_types"]:
                metrics["resource_types"][resource.resource_type.value] = 0
            metrics["resource_types"][resource.resource_type.value] += 1
            
            # Count by status
            metrics["status_counts"][resource.status.value] += 1
            
        return metrics
    
    def save_resource_report(self) -> None:
        """Save resource report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_resource_metrics(),
            "resources": {
                resource_id: {
                    "resource_type": resource.resource_type.value,
                    "capacity": resource.capacity,
                    "status": resource.status.value,
                    "owner": resource.owner,
                    "metadata": resource.metadata
                }
                for resource_id, resource in self.resources.items()
            }
        }
        
        report_file = Path("reports") / f"resource_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 