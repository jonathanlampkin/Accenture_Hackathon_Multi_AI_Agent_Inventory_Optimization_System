from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging
from datetime import datetime
import asyncio
import psutil
import time
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    name: str
    type: MetricType
    value: Any
    labels: Dict[str, str]
    timestamp: datetime

class MonitoringSystem:
    def __init__(self, storage_dir: str = "monitoring"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: Dict[str, List[Dict[str, Any]]] = {}
        self.alert_queue = Queue()
        self.running = False
        
        # Initialize logging
        self.logger = logging.getLogger("monitoring_system")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.storage_dir / f"monitoring_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
        
    def start(self) -> None:
        """Start the monitoring system"""
        if self.running:
            return
            
        self.running = True
        
        # Start metric collection
        self.metric_thread = threading.Thread(target=self._collect_metrics)
        self.metric_thread.daemon = True
        self.metric_thread.start()
        
        # Start alert processing
        self.alert_thread = threading.Thread(target=self._process_alerts)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        self.logger.info("Monitoring system started")
    
    def stop(self) -> None:
        """Stop the monitoring system"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads to finish
        self.metric_thread.join()
        self.alert_thread.join()
        
        self.logger.info("Monitoring system stopped")
    
    def _collect_metrics(self) -> None:
        """Collect system metrics"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Save metrics
                self._save_metrics()
                
                # Sleep for collection interval
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                time.sleep(60)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        # CPU metrics
        self.record_metric(
            "cpu_usage",
            MetricType.GAUGE,
            psutil.cpu_percent(),
            {"type": "system"}
        )
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric(
            "memory_usage",
            MetricType.GAUGE,
            memory.percent,
            {"type": "system"}
        )
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.record_metric(
            "disk_usage",
            MetricType.GAUGE,
            disk.percent,
            {"type": "system"}
        )
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.record_metric(
            "network_bytes_sent",
            MetricType.COUNTER,
            net_io.bytes_sent,
            {"type": "system"}
        )
        self.record_metric(
            "network_bytes_recv",
            MetricType.COUNTER,
            net_io.bytes_recv,
            {"type": "system"}
        )
    
    def _collect_application_metrics(self) -> None:
        """Collect application-level metrics"""
        # Task metrics
        self.record_metric(
            "active_tasks",
            MetricType.GAUGE,
            len(self.metrics.get("task_metrics", [])),
            {"type": "application"}
        )
        
        # Memory metrics
        self.record_metric(
            "active_memories",
            MetricType.GAUGE,
            len(self.metrics.get("memory_metrics", [])),
            {"type": "application"}
        )
        
        # Knowledge metrics
        self.record_metric(
            "active_knowledge",
            MetricType.GAUGE,
            len(self.metrics.get("knowledge_metrics", [])),
            {"type": "application"}
        )
    
    def record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: Any,
        labels: Dict[str, str]
    ) -> None:
        """Record a metric"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels,
            timestamp=datetime.now()
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append(metric)
        
        # Check alerts
        self._check_alerts(metric)
    
    def _check_alerts(self, metric: Metric) -> None:
        """Check if metric triggers any alerts"""
        if metric.name not in self.alerts:
            return
            
        for alert in self.alerts[metric.name]:
            if self._evaluate_alert(metric, alert):
                self.alert_queue.put({
                    "metric": metric,
                    "alert": alert,
                    "timestamp": datetime.now()
                })
    
    def _evaluate_alert(self, metric: Metric, alert: Dict[str, Any]) -> bool:
        """Evaluate if metric triggers alert"""
        threshold = alert["threshold"]
        operator = alert["operator"]
        
        if operator == ">":
            return metric.value > threshold
        elif operator == ">=":
            return metric.value >= threshold
        elif operator == "<":
            return metric.value < threshold
        elif operator == "<=":
            return metric.value <= threshold
        elif operator == "==":
            return metric.value == threshold
        elif operator == "!=":
            return metric.value != threshold
            
        return False
    
    def _process_alerts(self) -> None:
        """Process alerts from queue"""
        while self.running:
            try:
                alert_data = self.alert_queue.get(timeout=1)
                self._handle_alert(alert_data)
            except Exception:
                continue
    
    def _handle_alert(self, alert_data: Dict[str, Any]) -> None:
        """Handle an alert"""
        metric = alert_data["metric"]
        alert = alert_data["alert"]
        timestamp = alert_data["timestamp"]
        
        # Log alert
        self.logger.warning(
            f"Alert triggered: {alert['name']} - "
            f"Metric: {metric.name} = {metric.value} "
            f"{alert['operator']} {alert['threshold']}"
        )
        
        # Save alert
        alert_file = self.storage_dir / f"alerts_{timestamp.strftime('%Y%m%d')}.json"
        alert_data = {
            "timestamp": timestamp.isoformat(),
            "metric": {
                "name": metric.name,
                "value": metric.value,
                "labels": metric.labels
            },
            "alert": alert
        }
        
        with open(alert_file, 'a') as f:
            json.dump(alert_data, f)
            f.write('\n')
    
    def _save_metrics(self) -> None:
        """Save metrics to storage"""
        timestamp = datetime.now()
        metrics_file = self.storage_dir / f"metrics_{timestamp.strftime('%Y%m%d')}.json"
        
        metrics_data = {
            "timestamp": timestamp.isoformat(),
            "metrics": {
                name: [
                    {
                        "value": metric.value,
                        "labels": metric.labels,
                        "timestamp": metric.timestamp.isoformat()
                    }
                    for metric in metrics
                ]
                for name, metrics in self.metrics.items()
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def add_alert(
        self,
        metric_name: str,
        alert_name: str,
        threshold: float,
        operator: str,
        description: str
    ) -> None:
        """Add an alert"""
        if metric_name not in self.alerts:
            self.alerts[metric_name] = []
            
        self.alerts[metric_name].append({
            "name": alert_name,
            "threshold": threshold,
            "operator": operator,
            "description": description
        })
        
        self.logger.info(f"Added alert {alert_name} for metric {metric_name}")
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[Metric]]:
        """Get metrics matching criteria"""
        if metric_name:
            metrics = {metric_name: self.metrics.get(metric_name, [])}
        else:
            metrics = self.metrics.copy()
            
        if start_time or end_time:
            filtered_metrics = {}
            for name, metric_list in metrics.items():
                filtered_metrics[name] = [
                    metric for metric in metric_list
                    if (not start_time or metric.timestamp >= start_time)
                    and (not end_time or metric.timestamp <= end_time)
                ]
            metrics = filtered_metrics
            
        return metrics
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get metric summary"""
        summary = {
            "total_metrics": len(self.metrics),
            "metric_types": {},
            "alerts": {
                name: len(alerts)
                for name, alerts in self.alerts.items()
            }
        }
        
        for name, metrics in self.metrics.items():
            if metrics:
                summary["metric_types"][name] = {
                    "count": len(metrics),
                    "latest_value": metrics[-1].value,
                    "latest_timestamp": metrics[-1].timestamp.isoformat()
                }
                
        return summary
    
    def save_monitoring_report(self) -> None:
        """Save monitoring report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_metric_summary(),
            "metrics": {
                name: [
                    {
                        "value": metric.value,
                        "labels": metric.labels,
                        "timestamp": metric.timestamp.isoformat()
                    }
                    for metric in metrics
                ]
                for name, metrics in self.metrics.items()
            },
            "alerts": self.alerts
        }
        
        report_file = self.storage_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 