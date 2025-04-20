#!/usr/bin/env python3
"""
Unified System Runner for Inventory Optimization System

This script provides a single entry point to run the entire Inventory Optimization System,
including API server, background workers, and required services.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("system.log")
    ]
)
logger = logging.getLogger(__name__)

# Process registry to keep track of started processes
processes = {}

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "api/uploads",
        "api/results",
        "api/static",
        "api/templates",
        "logs",
        "output",
        "output/multi_agent",
        "output/forecasts",
        "output/reports",
        "mlflow",
        "expectations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def start_api_server(host, port, reload, workers):
    """Start the FastAPI server."""
    logger.info(f"Starting API server on {host}:{port} (reload={reload}, workers={workers})")
    
    cmd = [
        sys.executable,
        "run_api.py",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    if reload:
        cmd.append("--reload")
        
    env = os.environ.copy()
    
    # Ensure correct Python path
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = os.getcwd()
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    
    processes["api"] = proc
    logger.info(f"API server started with PID {proc.pid}")
    return proc

def start_celery_worker(queues, concurrency):
    """Start Celery worker."""
    logger.info(f"Starting Celery worker for queues: {queues}, concurrency: {concurrency}")
    
    cmd = [
        "celery",
        "-A", "src.tasks.celery_app",
        "worker",
        "--loglevel=info",
        f"--concurrency={concurrency}",
        f"--queues={queues}"
    ]
    
    env = os.environ.copy()
    
    # Ensure correct Python path
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = os.getcwd()
    
    # Set Celery-specific environment variables
    env["RABBITMQ_URL"] = os.environ.get("RABBITMQ_URL", "pyamqp://guest:guest@localhost:5672//")
    env["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    
    processes["celery_worker"] = proc
    logger.info(f"Celery worker started with PID {proc.pid}")
    return proc

def start_celery_beat():
    """Start Celery beat scheduler."""
    logger.info("Starting Celery beat scheduler")
    
    cmd = [
        "celery",
        "-A", "src.tasks.celery_app",
        "beat",
        "--loglevel=info"
    ]
    
    env = os.environ.copy()
    
    # Ensure correct Python path
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = os.getcwd()
    
    # Set Celery-specific environment variables
    env["RABBITMQ_URL"] = os.environ.get("RABBITMQ_URL", "pyamqp://guest:guest@localhost:5672//")
    env["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    
    processes["celery_beat"] = proc
    logger.info(f"Celery beat scheduler started with PID {proc.pid}")
    return proc

def start_mlflow_server(port):
    """Start MLflow tracking server."""
    logger.info(f"Starting MLflow tracking server on port {port}")
    
    os.makedirs("mlflow", exist_ok=True)
    
    cmd = [
        "mlflow",
        "server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--backend-store-uri", "sqlite:///mlflow/mlflow.db",
        "--default-artifact-root", os.path.join(os.getcwd(), "mlflow/artifacts")
    ]
    
    env = os.environ.copy()
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    
    processes["mlflow"] = proc
    logger.info(f"MLflow server started with PID {proc.pid}")
    return proc

def stop_all_processes():
    """Stop all started processes."""
    logger.info("Stopping all processes...")
    
    for name, proc in processes.items():
        try:
            logger.info(f"Stopping {name} process (PID: {proc.pid})...")
            proc.terminate()
            # Give it some time to terminate gracefully
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"{name} process did not terminate gracefully, killing it...")
            proc.kill()
        except Exception as e:
            logger.error(f"Error stopping {name} process: {e}")
    
    logger.info("All processes stopped")

def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {sig}, shutting down...")
    stop_all_processes()
    sys.exit(0)

def monitor_processes():
    """Monitor all started processes and print their output."""
    while True:
        for name, proc in list(processes.items()):
            # Check if process is still running
            if proc.poll() is not None:
                logger.error(f"{name} process exited with code {proc.returncode}")
                # Remove from registry
                del processes[name]
            
            # Read and print output
            output = proc.stdout.readline()
            if output:
                print(f"[{name}] {output.strip()}")
        
        # If no processes left, exit
        if not processes:
            logger.error("All processes have exited, shutting down")
            sys.exit(1)
            
        # Sleep to avoid high CPU usage
        time.sleep(0.1)

def set_environment_variables(config):
    """Set environment variables based on configuration."""
    # Database connection
    os.environ["DATABASE_URL"] = config.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inventory")
    
    # Redis connection
    os.environ["REDIS_HOST"] = config.get("REDIS_HOST", "localhost")
    os.environ["REDIS_PORT"] = config.get("REDIS_PORT", "6379")
    os.environ["REDIS_URL"] = config.get("REDIS_URL", "redis://localhost:6379/0")
    
    # RabbitMQ connection
    os.environ["RABBITMQ_HOST"] = config.get("RABBITMQ_HOST", "localhost")
    os.environ["RABBITMQ_PORT"] = config.get("RABBITMQ_PORT", "5672")
    os.environ["RABBITMQ_URL"] = config.get("RABBITMQ_URL", "pyamqp://guest:guest@localhost:5672//")
    
    # MLflow connection
    os.environ["MLFLOW_TRACKING_URI"] = config.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Output directories
    os.environ["OUTPUT_DIR"] = config.get("OUTPUT_DIR", "output")
    os.environ["EXPECTATIONS_DIR"] = config.get("EXPECTATIONS_DIR", "expectations")
    
    # Create configuration module for src.config.OUTPUT_DIR
    Path("src/config.py").write_text(
        f"""\"\"\"Configuration module for the inventory optimization system.\"\"\"
import os

# Output directories
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
"""
    )
    
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the Inventory Optimization System")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of API server worker processes")
    parser.add_argument("--mlflow-port", type=int, default=5000, help="Port for MLflow tracking server")
    parser.add_argument("--celery-workers", type=int, default=2, help="Number of Celery worker processes")
    parser.add_argument("--skip-celery", action="store_true", help="Skip starting Celery workers")
    parser.add_argument("--skip-mlflow", action="store_true", help="Skip starting MLflow server")
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create necessary directories
    ensure_directories()
    
    # Set environment variables
    config = {
        "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/inventory",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_URL": "redis://localhost:6379/0",
        "RABBITMQ_HOST": "localhost",
        "RABBITMQ_PORT": "5672",
        "RABBITMQ_URL": "pyamqp://guest:guest@localhost:5672//",
        "MLFLOW_TRACKING_URI": f"http://localhost:{args.mlflow_port}",
        "OUTPUT_DIR": "output",
        "EXPECTATIONS_DIR": "expectations"
    }
    set_environment_variables(config)
    
    logger.info("Starting Inventory Optimization System")
    logger.info("--------------------------------------")
    logger.info(f"API Host: {args.host}")
    logger.info(f"API Port: {args.port}")
    logger.info(f"Development mode (auto-reload): {args.reload}")
    logger.info(f"API Workers: {args.workers}")
    logger.info(f"MLflow Port: {args.mlflow_port}")
    logger.info(f"Celery Workers: {args.celery_workers}")
    logger.info("--------------------------------------")
    
    # Start MLflow server if not skipped
    if not args.skip_mlflow:
        start_mlflow_server(args.mlflow_port)
    
    # Start Celery workers if not skipped
    if not args.skip_celery:
        start_celery_worker("forecasting,inventory,reports,data", args.celery_workers)
        start_celery_beat()
    
    # Start API server
    start_api_server(args.host, args.port, args.reload, args.workers)
    
    try:
        logger.info("All components started, monitoring processes...")
        logger.info(f"API will be available at http://{args.host}:{args.port}")
        logger.info(f"API documentation at http://{args.host}:{args.port}/docs")
        if not args.skip_mlflow:
            logger.info(f"MLflow interface at http://localhost:{args.mlflow_port}")
        logger.info("--------------------------------------")
        logger.info("Press Ctrl+C to stop the system")
        
        # Monitor processes and print their output
        monitor_processes()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    finally:
        stop_all_processes()
    
    logger.info("System stopped")

if __name__ == "__main__":
    main() 