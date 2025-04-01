# Multi-Agent Inventory Optimization System

A sophisticated AI system for optimizing retail inventory management, built for the Accenture Hackathon. This system employs multiple specialized AI agents working collaboratively to optimize inventory levels, forecast demand, and recommend optimal pricing strategies.

## Features

- **Multi-Agent Architecture**: Specialized agents for inventory management, demand forecasting, pricing optimization, and supply chain operations working cooperatively
- **Adaptive Inventory Management**: Dynamic reorder point calculation based on lead times, stockout risk, and demand patterns
- **Advanced Demand Forecasting**: Utilizes historical sales data, seasonality factors, and external influences to predict future demand
- **Intelligent Pricing Optimization**: Sets optimal prices considering competitor pricing, elasticity, and inventory costs
- **Supply Chain Intelligence**: Improves supplier selection and lead time management
- **Customizable Optimization Goals**: Configure the system to prioritize cost reduction, product availability, or a balanced approach

## System Architecture

The system consists of the following components:

- **Inventory Management Agent**: Optimizes stock levels and reorder timing
- **Demand Forecasting Agent**: Predicts future sales patterns
- **Pricing Optimization Agent**: Recommends optimal pricing strategies
- **Supply Chain Optimization Agent**: Enhances supplier relationships and lead times
- **Coordination Agent**: Integrates insights from all agents to provide balanced recommendations

## Quick Start

1. Ensure you have Python 3.8+ installed
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your data files in the project root:
   - `inventory_monitoring.csv`
   - `demand_forecasting.csv`
   - `pricing_optimization.csv`
4. Run the system:
   ```
   python main.py
   ```

## Docker Deployment

You can run the system using Docker with GPU acceleration support:

### Using Docker Compose

1. Make sure you have Docker and Docker Compose installed. For GPU support, install NVIDIA Container Toolkit:
   ```
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. Run with CPU only:
   ```
   docker-compose up inventory-optimization-cpu
   ```

3. Run with GPU acceleration (requires NVIDIA GPU and drivers):
   ```
   docker-compose up inventory-optimization-gpu
   ```

### Using Docker Directly

1. Build the Docker image:
   ```
   docker build -t inventory-optimization .
   ```

2. Run with CPU only:
   ```
   docker run -v $(pwd)/output:/app/output -e USE_GPU=0 inventory-optimization
   ```

3. Run with GPU support:
   ```
   docker run --gpus all -v $(pwd)/output:/app/output -e USE_GPU=1 inventory-optimization
   ```

## Command Line Options

```
python main.py [options]

Options:
  --simple-mode            Run in simple mode without advanced ML models
  --optimize-for {cost,availability,balanced}
                           Optimization target (default: balanced)
  --product-id PRODUCT_ID  Focus on a specific product ID
  --store-id STORE_ID      Focus on a specific store ID
  --iterations ITERATIONS  Number of optimization iterations (default: 5)
  --output-dir OUTPUT_DIR  Custom output directory
```

## Data Requirements

The system requires three CSV files with specific formats:

1. **inventory_monitoring.csv**: Current inventory status
   - Columns: Product ID, Store ID, Stock Levels, Supplier Lead Time (days), Stockout Frequency, Reorder Point, Expiry Date, Warehouse Capacity, Order Fulfillment Time (days)

2. **demand_forecasting.csv**: Historical sales data
   - Columns: Product ID, Date, Store ID, Sales Quantity, Price, Promotions, Seasonality Factors, External Factors, Demand Trend, Customer Segments

3. **pricing_optimization.csv**: Pricing and cost data
   - Columns: Product ID, Store ID, Price, Competitor Prices, Discounts, Sales Volume, Customer Reviews, Return Rate (%), Storage Cost, Elasticity Index

## Advanced Usage

### Running with Limited Dependencies

For environments with limited package availability:

```
python main.py --simple-mode
```

This will use simpler statistical models instead of advanced ML models, requiring fewer dependencies.

### Focusing on Specific Products or Stores

```
python main.py --product-id P1000 --store-id S005
```

### Optimizing for Different Goals

```
# Prioritize cost reduction
python main.py --optimize-for cost

# Prioritize product availability
python main.py --optimize-for availability

# Balanced approach (default)
python main.py --optimize-for balanced
```

## Output

The system generates:

1. Optimization recommendations
2. Visualizations of inventory levels, sales trends, and pricing elasticity
3. Detailed logs of agent interactions and decision processes
4. CSV exports of optimized inventory plans

All output is saved to the `output` directory (customizable via `--output-dir`).

## Installation

For detailed installation instructions, especially for environments with limited package availability, see [INSTALLATION.md](INSTALLATION.md).

## Project Structure

```
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── INSTALLATION.md          # Detailed installation guide
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── base_agent.py    # Base agent class
│   │   ├── inventory_agent.py
│   │   ├── demand_agent.py
│   │   ├── pricing_agent.py
│   │   ├── supply_chain_agent.py
│   │   └── coordination_agent.py
│   ├── models/              # Prediction models
│   │   ├── forecasting_models.py
│   │   ├── optimization_models.py
│   │   └── risk_models.py
│   ├── utils/               # Utility functions
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── visualizer.py    # Visualization utilities
│   ├── coordinator.py       # Multi-agent coordination
│   └── config.py            # System configuration
└── data/                    # Data files (not included in repo)
    ├── inventory_monitoring.csv
    ├── demand_forecasting.csv
    └── pricing_optimization.csv
```

## License

This project is proprietary and confidential. Unauthorized copying, redistribution, or use is prohibited.

Created for the Accenture Hackathon, 2023.