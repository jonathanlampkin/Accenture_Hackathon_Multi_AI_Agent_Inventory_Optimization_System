from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime
import json
from pathlib import Path
import traceback
from functools import wraps

class EnhancedErrorHandler:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger("crewai_error_handler")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"crewai_errors_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics
        self.error_counts: Dict[str, int] = {}
        self.error_contexts: Dict[str, list] = {}
        self.retry_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}
        
    def handle_error(self, error: Exception, context: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """Handle errors with context and retry logic"""
        error_type = type(error).__name__
        error_id = f"{error_type}_{int(time.time())}"
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.retry_counts[error_id] = retry_count
        self.last_error_time[error_id] = time.time()
        
        # Create error context
        error_context = {
            "error_id": error_id,
            "error_type": error_type,
            "error_message": str(error),
            "retry_count": retry_count,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        # Store error context
        self.error_contexts[error_id] = error_context
        
        # Log error
        self.logger.error(f"Error {error_id}: {str(error)}", extra=error_context)
        
        # Determine retry strategy
        should_retry = self._should_retry(error, retry_count)
        backoff_time = self._calculate_backoff(retry_count)
        
        return {
            "error_id": error_id,
            "should_retry": should_retry,
            "backoff_time": backoff_time,
            "error_context": error_context
        }
    
    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if error should be retried"""
        max_retries = 3
        retryable_errors = ["ConnectionError", "TimeoutError", "RateLimitError"]
        
        if retry_count >= max_retries:
            return False
            
        if type(error).__name__ in retryable_errors:
            return True
            
        return False
    
    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate exponential backoff time"""
        base_delay = 1.0
        max_delay = 30.0
        return min(base_delay * (2 ** retry_count), max_delay)
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error handling metrics"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts,
            "retry_counts": self.retry_counts,
            "error_types": list(self.error_counts.keys()),
            "last_errors": {
                error_id: self.error_contexts[error_id]
                for error_id in list(self.error_contexts.keys())[-5:]
            }
        }
    
    def save_error_report(self) -> None:
        """Save error report to file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_error_metrics(),
            "error_contexts": self.error_contexts
        }
        
        report_file = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

def error_handler_decorator(error_handler: EnhancedErrorHandler):
    """Decorator for error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "retry_count": retry_count
                    }
                    result = error_handler.handle_error(e, context, retry_count)
                    
                    if not result["should_retry"]:
                        raise
                        
                    time.sleep(result["backoff_time"])
                    retry_count += 1
        return wrapper
    return decorator 