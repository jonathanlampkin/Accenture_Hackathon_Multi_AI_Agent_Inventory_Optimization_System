# Multi-Agent Inventory Optimization System

An intelligent system for optimizing inventory management using machine learning, forecasting, and multi-agent collaboration.

## Overview

This system helps businesses optimize their inventory levels by:

1. **Forecasting demand** using advanced time series models
2. **Calculating optimal inventory levels** based on demand variability and lead times
3. **Generating purchase recommendations** to maintain optimal stock levels
4. **Monitoring inventory health** through real-time dashboards and alerts

## Key Features

- **Machine Learning-Powered Forecasting**: Multiple models (SARIMA, Prophet, ensemble approaches) for accurate demand prediction
- **Inventory Optimization**: Safety stock calculation, reorder points, and economic order quantities
- **Distributed Processing**: Celery-based task system for scalability and reliability
- **API-First Design**: FastAPI with Swagger/OpenAPI documentation
- **Monitoring and Metrics**: Prometheus and Grafana for comprehensive observability
- **Secure Authentication**: JWT-based API security

## Tech Stack

- **Python** with FastAPI for API development
- **PostgreSQL** for data persistence
- **Redis** for caching and Celery broker
- **RabbitMQ** for message queuing
- **Celery** for distributed task processing
- **Docker** and Docker Compose for containerization
- **MLflow** for experiment tracking
- **Prometheus** and **Grafana** for monitoring
- **Great Expectations** for data validation

For a comprehensive overview of the technology stack, see [TECH_STACK.md](TECH_STACK.md).

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- PostgreSQL 13+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/inventory-optimization-system.git
cd inventory-optimization-system
```

2. Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure the environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Unified System

The easiest way to run the entire system is with our unified system runner:

```bash
./run_system.py
```

This single command will start:
- FastAPI server
- Celery workers for background tasks
- MLflow for experiment tracking
- Establishes connections to Redis and RabbitMQ

For detailed options and configuration, see [UNIFIED_SYSTEM.md](UNIFIED_SYSTEM.md).

### Alternative: Using Docker Compose

If you prefer to use Docker, you can start all services with:

```bash
docker-compose up -d
```

### Database Initialization

Initialize the database:
```bash
python -m src.models.database init
alembic upgrade head
```

### Load Sample Data
```bash
python scripts/load_sample_data.py
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

## Usage

### API Access

The API is accessible at `http://localhost:8000` with documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Monitoring

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- MLflow: `http://localhost:5000`

### Example Workflows

1. **Generate a demand forecast**:
```bash
curl -X POST "http://localhost:8000/api/forecasts/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"product_id": 101, "horizon": 30}'
```

2. **Optimize inventory levels**:
```bash
curl -X POST "http://localhost:8000/api/inventory/optimize" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"product_ids": [101, 102, 103], "service_level": 0.95}'
```

## Development

### Testing

Run tests with:
```bash
pytest tests/
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- Flake8 for linting
- mypy for type checking

### CI/CD

The GitHub Actions workflow in `.github/workflows/ci-cd.yml` handles:
- Testing
- Linting
- Docker image building
- Deployment

## Project Structure

```
.
├── api/               # API routes and endpoints
├── dashboards/        # Grafana dashboard definitions
├── data/              # Sample data files
├── dags/              # Airflow DAGs
├── docs/              # Documentation
├── migrations/        # Alembic migrations
├── prometheus/        # Prometheus configuration
├── scripts/           # Utility scripts
├── src/               # Source code
│   ├── auth/          # Authentication
│   ├── models/        # Database models
│   ├── tasks/         # Celery tasks
│   └── utils/         # Utilities
└── tests/             # Tests
```

## Final Steps

For final setup and deployment steps, see [FINAL_STEPS.md](FINAL_STEPS.md).

## License

[MIT License](LICENSE)

## Acknowledgements

- This project was built for the Accenture Hackathon
- Uses open-source libraries and frameworks