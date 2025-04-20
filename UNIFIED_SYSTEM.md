# Unified Inventory Optimization System

This document describes how to run the complete Inventory Optimization System using a single command. The unified system runner orchestrates all components of the system, including the API server, background workers, and other required services.

## Overview

The Inventory Optimization System consists of several components:

1. **FastAPI Server** - The primary API that serves forecasts, inventory optimization, and reports
2. **Celery Workers** - Background task processing for computationally intensive operations
3. **MLflow Server** - Experiment tracking for machine learning models
4. **Redis** - Cache and message broker backend
5. **RabbitMQ** - Message broker for Celery tasks

The unified system runner (`run_system.py`) manages all these components, ensuring they are properly configured and can communicate with each other.

## Prerequisites

Before running the system, ensure you have:

1. Python 3.9+ with required packages installed (`pip install -r requirements.txt`)
2. Redis server running locally (or accessible via network)
3. RabbitMQ server running locally (or accessible via network)

## Running the System

To start the entire system with default configuration:

```bash
./run_system.py
```

This will start:
- FastAPI server on port 8000
- MLflow server on port 5000
- Celery workers for task processing
- Celery beat for scheduled tasks

## Configuration Options

You can customize the system using the following command-line arguments:

```bash
./run_system.py --help
```

### Common Options

- `--host HOST` - Hostname to bind the API server to (default: 0.0.0.0)
- `--port PORT` - Port to bind the API server to (default: 8000)
- `--reload` - Enable auto-reload for development
- `--workers WORKERS` - Number of API server worker processes (default: 1)
- `--mlflow-port PORT` - Port for MLflow tracking server (default: 5000)
- `--celery-workers N` - Number of Celery worker processes (default: 2)
- `--skip-celery` - Skip starting Celery workers
- `--skip-mlflow` - Skip starting MLflow server

### Environment Variables

You can configure the system using environment variables:

- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_HOST` - Redis server hostname
- `REDIS_PORT` - Redis server port
- `RABBITMQ_HOST` - RabbitMQ server hostname
- `RABBITMQ_PORT` - RabbitMQ server port

## Example Usage

### Production Setup

```bash
./run_system.py --workers 4 --celery-workers 4
```

### Development Setup

```bash
./run_system.py --reload --port 8080
```

### Running just the API server

```bash
./run_system.py --skip-celery --skip-mlflow
```

## Accessing the System

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

## Stopping the System

Press `Ctrl+C` to gracefully stop all components of the system. The unified runner will properly terminate all processes.

## Troubleshooting

If you encounter issues:

1. Check the logs in `system.log`
2. Ensure Redis and RabbitMQ are running
3. Make sure the required ports are not already in use

### Common Issues

- **Port conflicts**: If ports are already in use, specify different ports using `--port` and `--mlflow-port`
- **Redis connection issues**: Ensure Redis is running locally or update the REDIS_HOST environment variable
- **RabbitMQ connection issues**: Ensure RabbitMQ is running locally or update the RABBITMQ_HOST environment variable

## System Components and Features

The unified system includes all the following enhanced technologies:

1. **Redis Cache** - Improves API response times and reduces database load
2. **RabbitMQ Messaging** - Ensures reliable task delivery and communication
3. **Prometheus Metrics** - Monitors system performance and health
4. **Great Expectations Data Validation** - Ensures data quality throughout the system
5. **Celery Distributed Tasks** - Enables scalable processing of computationally intensive operations
6. **JWT Authentication** - Secures the API endpoints
7. **MLflow Experiment Tracking** - Tracks model performance and artifacts
8. **FastAPI** - Provides a modern, high-performance API server 