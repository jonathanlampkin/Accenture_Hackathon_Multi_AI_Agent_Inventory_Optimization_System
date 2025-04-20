# Inventory Optimization System Tech Stack

This document outlines the technology stack used in this Inventory Optimization System, explaining each component and its purpose.

## Core Technologies

### Containerization & Orchestration

- **Docker**: Containerization platform for consistent development and deployment environments
- **Docker Compose**: Multi-container orchestration for local development and testing

### Data Processing & Storage

- **PostgreSQL**: Relational database for structured data storage
- **Redis**: In-memory data store for caching and pub/sub messaging
- **Pandas/NumPy**: Python libraries for data manipulation and numerical processing

### API & Web Framework

- **FastAPI**: Modern, high-performance web framework with automatic OpenAPI documentation
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for serving the FastAPI application

### Machine Learning & Forecasting

- **Scikit-learn**: Machine learning library for classical forecasting models
- **TensorFlow**: Deep learning library for advanced forecasting models
- **Statsmodels**: Statistical models including ARIMA, SARIMA, and exponential smoothing
- **MLflow**: Platform for the complete machine learning lifecycle

### Workflow Orchestration

- **Apache Airflow**: Workflow orchestration platform for scheduling and monitoring data pipelines
- **Celery**: Distributed task queue for asynchronous processing

### Messaging & Event Processing

- **RabbitMQ**: Message broker for reliable message delivery and event-driven architecture
- **Pika**: Python client for RabbitMQ

### Monitoring & Observability

- **Prometheus**: Time series database for metrics collection and alerting
- **Grafana**: Visualization and dashboarding platform for monitoring data
- **Prometheus Client Library**: Python client for exposing application metrics

## System Architecture

The system is organized into the following components:

### Data Layer
- **PostgreSQL**: Primary data store for inventory, product, and historical demand data
- **Redis**: Caching layer for API responses and frequently accessed data

### Application Layer
- **FastAPI**: REST API for client interactions
- **Celery Workers**: Background processing for long-running tasks
- **RabbitMQ**: Message broker for inter-service communication

### Machine Learning Layer
- **Forecasting Services**: Time series forecasting models for demand prediction
- **Optimization Engine**: Inventory optimization algorithms
- **MLflow**: Experiment tracking, model registry, and deployment

### Orchestration Layer
- **Airflow**: Workflow scheduling and orchestration
- **DAGs**: Directed Acyclic Graphs defining workflow processes

### Monitoring Layer
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards for system monitoring

## Benefits of This Tech Stack

1. **Scalability**: Containerized architecture allows for easy horizontal scaling
2. **Flexibility**: Modular design enables swapping components as needed
3. **Observability**: Comprehensive monitoring and alerting
4. **Reproducibility**: Experiment tracking and model versioning
5. **Reliability**: Message queues and task distribution for fault tolerance
6. **Performance**: Caching and optimized data processing

## Getting Started

To run the complete system with all components, use Docker Compose:

```bash
docker-compose up -d
```

This will start all services in the background. You can access:

- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Airflow UI: http://localhost:8080
- MLflow UI: http://localhost:5000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- RabbitMQ Management: http://localhost:15672

## Development Setup

For development, you may want to run only specific services:

```bash
# Run only the database and API
docker-compose up -d db api

# Run monitoring stack
docker-compose up -d prometheus grafana

# Run message broker
docker-compose up -d rabbitmq
```

## Additional Documentation

For more detailed information on each component, refer to:

- [API Documentation](./docs/API.md)
- [Machine Learning Pipeline](./docs/ML_PIPELINE.md)
- [Deployment Guide](./docs/DEPLOYMENT.md) 