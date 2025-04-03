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

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

class MessagePriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Message:
    message_id: str
    message_type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    priority: MessagePriority
    timestamp: datetime
    metadata: Dict[str, Any] = None

class CommunicationManager:
    def __init__(self, config_file: str = "communication_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        self.message_queue = Queue()
        self.retry_queue = Queue()
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.message_history: Dict[str, List[Message]] = {}
        self.running = False
        
        # Initialize logging
        self.logger = logging.getLogger("communication_manager")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path("logs") / f"communication_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(file_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load communication configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}
    
    def start(self) -> None:
        """Start the communication manager"""
        if self.running:
            return
            
        self.running = True
        
        # Start message processing
        self.process_thread = threading.Thread(target=self._process_messages)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start retry processing
        self.retry_thread = threading.Thread(target=self._process_retries)
        self.retry_thread.daemon = True
        self.retry_thread.start()
        
        self.logger.info("Communication manager started")
    
    def stop(self) -> None:
        """Stop the communication manager"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads to finish
        self.process_thread.join()
        self.retry_thread.join()
        
        self.logger.info("Communication manager stopped")
    
    def _process_messages(self) -> None:
        """Process messages from queue"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                self._handle_message(message)
            except Exception:
                continue
    
    def _process_retries(self) -> None:
        """Process retry queue"""
        while self.running:
            try:
                retry_data = self.retry_queue.get(timeout=1)
                self._handle_retry(retry_data)
            except Exception:
                continue
    
    def _handle_message(self, message: Message) -> None:
        """Handle a message"""
        try:
            # Store message in history
            if message.receiver not in self.message_history:
                self.message_history[message.receiver] = []
            self.message_history[message.receiver].append(message)
            
            # Call message handlers
            if message.message_type.value in self.message_handlers:
                for handler in self.message_handlers[message.message_type.value]:
                    try:
                        handler(message)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {str(e)}")
            
            self.logger.info(f"Processed message: {message.message_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self._handle_error(message, str(e))
    
    def _handle_retry(self, retry_data: Dict[str, Any]) -> None:
        """Handle a retry"""
        message = retry_data["message"]
        retry_count = retry_data["retry_count"]
        max_retries = retry_data["max_retries"]
        
        if retry_count >= max_retries:
            self.logger.error(f"Max retries reached for message: {message.message_id}")
            return
            
        # Wait before retry
        time.sleep(retry_data["retry_delay"])
        
        # Retry message
        self.send_message(message)
        
        self.logger.info(f"Retried message: {message.message_id}")
    
    def _handle_error(self, message: Message, error: str) -> None:
        """Handle an error"""
        error_message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            sender="system",
            receiver=message.sender,
            content={"error": error, "original_message": message.message_id},
            priority=MessagePriority.HIGH,
            timestamp=datetime.now()
        )
        
        self.send_message(error_message)
    
    def send_message(
        self,
        message_type: MessageType,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> str:
        """Send a message"""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            content=content,
            priority=priority,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Add to message queue
        self.message_queue.put(message)
        
        # Add to retry queue if needed
        if retry_count < max_retries:
            self.retry_queue.put({
                "message": message,
                "retry_count": retry_count + 1,
                "max_retries": max_retries,
                "retry_delay": retry_delay
            })
        
        self.logger.info(f"Sent message: {message.message_id}")
        return message.message_id
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], None]
    ) -> None:
        """Register a message handler"""
        if message_type.value not in self.message_handlers:
            self.message_handlers[message_type.value] = []
            
        self.message_handlers[message_type.value].append(handler)
        self.logger.info(f"Registered handler for message type: {message_type.value}")
    
    def unregister_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], None]
    ) -> None:
        """Unregister a message handler"""
        if message_type.value in self.message_handlers:
            self.message_handlers[message_type.value].remove(handler)
            self.logger.info(f"Unregistered handler for message type: {message_type.value}")
    
    def get_message_history(
        self,
        receiver: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        message_type: Optional[MessageType] = None
    ) -> List[Message]:
        """Get message history"""
        if receiver not in self.message_history:
            return []
            
        messages = self.message_history[receiver]
        
        if start_time:
            messages = [
                message for message in messages
                if message.timestamp >= start_time
            ]
            
        if end_time:
            messages = [
                message for message in messages
                if message.timestamp <= end_time
            ]
            
        if message_type:
            messages = [
                message for message in messages
                if message.message_type == message_type
            ]
            
        return messages
    
    def get_message_metrics(self) -> Dict[str, Any]:
        """Get message metrics"""
        metrics = {
            "total_messages": sum(len(messages) for messages in self.message_history.values()),
            "message_types": {
                message_type.value: 0
                for message_type in MessageType
            },
            "priorities": {
                priority.value: 0
                for priority in MessagePriority
            }
        }
        
        for messages in self.message_history.values():
            for message in messages:
                metrics["message_types"][message.message_type.value] += 1
                metrics["priorities"][message.priority.value] += 1
                
        return metrics
    
    def save_communication_report(self) -> None:
        """Save communication report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_message_metrics(),
            "message_history": {
                receiver: [
                    {
                        "message_id": message.message_id,
                        "message_type": message.message_type.value,
                        "sender": message.sender,
                        "content": message.content,
                        "priority": message.priority.value,
                        "timestamp": message.timestamp.isoformat(),
                        "metadata": message.metadata
                    }
                    for message in messages
                ]
                for receiver, messages in self.message_history.items()
            }
        }
        
        report_file = Path("reports") / f"communication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2) 