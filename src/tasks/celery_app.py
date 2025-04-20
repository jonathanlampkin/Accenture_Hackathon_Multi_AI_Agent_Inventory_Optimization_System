"""
Celery application for distributed task processing.
"""
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from celery import Celery
from celery.schedules import crontab
from celery.signals import task_failure, task_postrun, task_prerun, worker_init, worker_ready

# Type variable for generic return type
T = TypeVar("T")

# Create Celery app
celery_app = Celery(
    "inventory_optimization",
    broker=os.environ.get("RABBITMQ_URL", "pyamqp://guest:guest@localhost:5672//"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
)

# Load Celery config from environment
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
    result_expires=3600 * 24 * 7,  # 7 days
)

# Configure periodic tasks
celery_app.conf.beat_schedule = {
    "daily_forecast_update": {
        "task": "src.tasks.forecasting.update_all_forecasts",
        "schedule": crontab(hour=1, minute=0),  # Run at 1:00 AM UTC
        "args": (),
    },
    "daily_inventory_optimization": {
        "task": "src.tasks.inventory.optimize_inventory_levels",
        "schedule": crontab(hour=2, minute=0),  # Run at 2:00 AM UTC
        "args": (),
    },
    "weekly_model_retraining": {
        "task": "src.tasks.forecasting.retrain_all_models",
        "schedule": crontab(day_of_week=0, hour=3, minute=0),  # Run Sundays at 3:00 AM UTC
        "args": (),
    },
}

# Set task routes for different queues
celery_app.conf.task_routes = {
    "src.tasks.forecasting.*": {"queue": "forecasting"},
    "src.tasks.inventory.*": {"queue": "inventory"},
    "src.tasks.reports.*": {"queue": "reports"},
    "src.tasks.data.*": {"queue": "data"},
}

# Celery task signals
@task_prerun.connect
def task_prerun_handler(task_id: str, task: Celery.Task, args: List, kwargs: Dict) -> None:
    """Handle task pre-run signal.
    
    Args:
        task_id: Task ID
        task: Task object
        args: Task arguments
        kwargs: Task keyword arguments
    """
    from src.utils.metrics import TASK_STARTED
    from src.models.forecast import ForecastJob
    from src.models.database import get_db_context
    
    # Update metrics
    TASK_STARTED.labels(task_name=task.name).inc()
    
    # Update forecast job status if applicable
    if task.name.startswith("src.tasks.forecasting."):
        try:
            with get_db_context() as db:
                job = db.query(ForecastJob).filter(ForecastJob.job_id == task_id).first()
                if job:
                    job.status = "running"
                    db.commit()
        except Exception as e:
            print(f"Error updating forecast job status: {e}")
            
@task_postrun.connect
def task_postrun_handler(task_id: str, task: Celery.Task, retval: Any, state: str, args: List, kwargs: Dict) -> None:
    """Handle task post-run signal.
    
    Args:
        task_id: Task ID
        task: Task object
        retval: Task return value
        state: Task state
        args: Task arguments
        kwargs: Task keyword arguments
    """
    from src.utils.metrics import TASK_COMPLETED, TASK_FAILED
    from src.models.forecast import ForecastJob
    from src.models.database import get_db_context
    
    # Update metrics
    if state == "SUCCESS":
        TASK_COMPLETED.labels(task_name=task.name).inc()
    else:
        TASK_FAILED.labels(task_name=task.name).inc()
        
    # Update forecast job status if applicable
    if task.name.startswith("src.tasks.forecasting."):
        try:
            with get_db_context() as db:
                job = db.query(ForecastJob).filter(ForecastJob.job_id == task_id).first()
                if job:
                    job.status = "completed" if state == "SUCCESS" else "failed"
                    if state == "SUCCESS" and isinstance(retval, dict):
                        job.result = retval
                    elif state != "SUCCESS":
                        job.error_message = str(retval) if retval else f"Task failed with state: {state}"
                    db.commit()
        except Exception as e:
            print(f"Error updating forecast job status: {e}")
            
@task_failure.connect
def task_failure_handler(task_id: str, exception: Exception, traceback: Any, einfo: Any, args: List, kwargs: Dict) -> None:
    """Handle task failure signal.
    
    Args:
        task_id: Task ID
        exception: Exception
        traceback: Traceback
        einfo: Error info
        args: Task arguments
        kwargs: Task keyword arguments
    """
    from src.utils.metrics import TASK_FAILED
    from src.models.forecast import ForecastJob
    from src.models.database import get_db_context
    
    # Update metrics
    TASK_FAILED.labels(task_name=task_id).inc()
    
    # Update forecast job status if applicable
    try:
        with get_db_context() as db:
            job = db.query(ForecastJob).filter(ForecastJob.job_id == task_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(exception)
                db.commit()
    except Exception as e:
        print(f"Error updating forecast job status: {e}")
        
@worker_init.connect
def worker_init_handler(**kwargs: Any) -> None:
    """Handle worker init signal."""
    print("Inventory Optimization worker initialized")
    
@worker_ready.connect
def worker_ready_handler(**kwargs: Any) -> None:
    """Handle worker ready signal."""
    print("Inventory Optimization worker ready")
    
def with_logging(f: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add logging to Celery tasks.
    
    Args:
        f: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        task_name = f.__name__
        print(f"Starting task {task_name}")
        try:
            result = f(*args, **kwargs)
            print(f"Task {task_name} completed successfully")
            return result
        except Exception as e:
            print(f"Task {task_name} failed: {e}")
            raise
    return wrapper 