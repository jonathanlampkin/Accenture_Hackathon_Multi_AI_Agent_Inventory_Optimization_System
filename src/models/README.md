# Advanced Forecasting Models for Inventory Optimization

This directory contains implementations of state-of-the-art time series forecasting models for inventory demand prediction and optimization.

## Included Models

1. **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
   - Powerful statistical model for time series with trend and seasonality
   - Handles multiple seasonal patterns

2. **Prophet**
   - Facebook's time series forecasting library
   - Decomposable model that handles yearly, weekly, and daily seasonality
   - Robust to missing data and outliers

3. **XGBoost**
   - Gradient boosting machine learning model adapted for time series
   - Automatically handles feature importance
   - High performance for many forecasting tasks

4. **LightGBM**
   - Gradient boosting framework optimized for efficiency
   - Often faster than XGBoost with comparable accuracy
   - Well-suited for large datasets

5. **Neural Prophet**
   - Neural network-based extension of Prophet
   - Combines deep learning with decomposable time series modeling
   - Handles complex patterns and multiple seasonalities

## How to Use

### Basic Usage

```python
from models.forecasting_models import ProphetModel

# Initialize a model
model = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)

# Fit the model
model.fit(data=your_dataframe, target_col='sales', date_col='date')

# Generate forecasts
forecast_df = model.predict(horizon=30)

# Evaluate the model
metrics = model.evaluate(test_data=your_test_data, target_col='sales')
```

### Model Comparison

```python
from models.model_comparison import ModelComparison

# Initialize comparison
comparison = ModelComparison(output_dir='output/comparison')

# Add models to compare
comparison.add_all_models()
# Or add specific models
# comparison.add_model(SARIMAModel())
# comparison.add_model(ProphetModel())

# Run full comparison
results = comparison.run_full_comparison(
    data=your_dataframe,
    date_col='date',
    target_col='sales',
    test_size=0.2
)

# Access results
evaluation_df = results['evaluation']
forecasts = results['forecasts']
```

### Integration with AI Agents

Our forecasting models are integrated with the AI agent system through specialized tools:

```python
from models.forecasting_integration import forecasting_tools

# Access the forecasting tools
train_tool = forecasting_tools[0]  # train_forecast_model_tool
generate_tool = forecasting_tools[1]  # generate_forecast_tool
compare_tool = forecasting_tools[2]  # compare_forecast_models_tool
visualize_tool = forecasting_tools[3]  # visualize_forecast_tool
best_model_tool = forecasting_tools[4]  # get_best_forecast_model_tool
```

## Running the Demo

We provide a demo script that showcases all the forecasting models and their comparison:

```bash
python src/demo_forecasting.py --data path/to/your/data.csv --output output/myresults
```

If no data is provided, the demo will generate synthetic data with trend, seasonality, and noise.

## Dependencies

The forecasting models require these packages:
- statsmodels (for SARIMA)
- prophet
- neuralprophet
- xgboost
- lightgbm
- scikit-learn
- matplotlib, seaborn (for visualization)

These are included in the project's main `requirements.txt` file.

## Model Selection Guide

- **SARIMA**: Best for data with clear, consistent seasonality patterns
- **Prophet**: Good general-purpose model, especially for data with multiple seasonality patterns and holidays
- **XGBoost/LightGBM**: Best for datasets with many external features or when seasonality is irregular
- **Neural Prophet**: Good for complex datasets where standard methods underperform

Always run a model comparison to determine which model works best for your specific data. 