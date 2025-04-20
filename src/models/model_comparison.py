"""
Model Comparison Module for Time Series Forecasting

This module provides functionality to compare different forecasting models
based on various metrics and visualize the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import logging
import json
from datetime import datetime

# Import forecasting models
from .forecasting_models import (
    ForecastModel, SARIMAModel, ProphetModel, 
    XGBoostForecastModel, LightGBMForecastModel, NeuralProphetModel
)

logger = logging.getLogger(__name__)

class ModelComparison:
    """Class for comparing different forecasting models"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the model comparison
        
        Args:
            output_dir: Directory to save results and plots (default: None)
        """
        self.models = []
        self.results = {}
        self.train_data = None
        self.test_data = None
        self.date_col = None
        self.target_col = None
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(f'output/model_comparison_{timestamp}')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_model(self, model: ForecastModel) -> None:
        """
        Add a forecasting model to the comparison
        
        Args:
            model: A forecasting model instance
        """
        self.models.append(model)
        logger.info(f"Added {model.name} model to comparison")
    
    def add_all_models(self, **kwargs) -> None:
        """
        Add all available forecasting models to the comparison
        
        Args:
            **kwargs: Parameters for model initialization
        """
        # Add SARIMA model
        sarima_order = kwargs.get('sarima_order', (1, 1, 1))
        sarima_seasonal_order = kwargs.get('sarima_seasonal_order', (1, 1, 1, 7))
        self.add_model(SARIMAModel(order=sarima_order, seasonal_order=sarima_seasonal_order))
        
        # Add Prophet model
        prophet_yearly = kwargs.get('prophet_yearly_seasonality', True)
        prophet_weekly = kwargs.get('prophet_weekly_seasonality', True)
        prophet_daily = kwargs.get('prophet_daily_seasonality', False)
        prophet_changepoint = kwargs.get('prophet_changepoint_prior_scale', 0.05)
        self.add_model(ProphetModel(
            yearly_seasonality=prophet_yearly,
            weekly_seasonality=prophet_weekly,
            daily_seasonality=prophet_daily,
            changepoint_prior_scale=prophet_changepoint
        ))
        
        # Add XGBoost model
        xgb_max_lag = kwargs.get('xgb_max_lag', 7)
        xgb_n_estimators = kwargs.get('xgb_n_estimators', 100)
        xgb_learning_rate = kwargs.get('xgb_learning_rate', 0.1)
        xgb_max_depth = kwargs.get('xgb_max_depth', 5)
        self.add_model(XGBoostForecastModel(
            max_lag=xgb_max_lag,
            n_estimators=xgb_n_estimators,
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth
        ))
        
        # Add LightGBM model
        lgb_max_lag = kwargs.get('lgb_max_lag', 7)
        lgb_n_estimators = kwargs.get('lgb_n_estimators', 100)
        lgb_learning_rate = kwargs.get('lgb_learning_rate', 0.1)
        lgb_num_leaves = kwargs.get('lgb_num_leaves', 31)
        self.add_model(LightGBMForecastModel(
            max_lag=lgb_max_lag,
            n_estimators=lgb_n_estimators,
            learning_rate=lgb_learning_rate,
            num_leaves=lgb_num_leaves
        ))
        
        # Add Neural Prophet model
        np_changepoints = kwargs.get('np_changepoints', 10)
        np_forecasts = kwargs.get('np_forecasts', 1)
        np_yearly = kwargs.get('np_yearly_seasonality', True)
        np_weekly = kwargs.get('np_weekly_seasonality', True)
        self.add_model(NeuralProphetModel(
            n_changepoints=np_changepoints,
            n_forecasts=np_forecasts,
            yearly_seasonality=np_yearly,
            weekly_seasonality=np_weekly
        ))
        
        logger.info(f"Added all available models to comparison ({len(self.models)} models)")
    
    def split_data(self, data: pd.DataFrame, date_col: str, target_col: str, 
                  test_size: float = 0.2, split_date: Optional[str] = None) -> None:
        """
        Split data into training and testing sets
        
        Args:
            data: DataFrame with time series data
            date_col: Column name for date
            target_col: Column name for target variable
            test_size: Proportion of data to use for testing (default: 0.2)
            split_date: Date to split the data (format: 'YYYY-MM-DD')
        """
        # Store column names
        self.date_col = date_col
        self.target_col = target_col
        
        # Ensure data is sorted by date
        data_sorted = data.sort_values(by=date_col).copy()
        
        if split_date:
            # Split by date
            train_data = data_sorted[data_sorted[date_col] < split_date]
            test_data = data_sorted[data_sorted[date_col] >= split_date]
        else:
            # Split by proportion
            n = len(data_sorted)
            train_size = int(n * (1 - test_size))
            train_data = data_sorted.iloc[:train_size]
            test_data = data_sorted.iloc[train_size:]
        
        self.train_data = train_data
        self.test_data = test_data
        
        logger.info(f"Data split into training ({len(train_data)} rows) and testing ({len(test_data)} rows)")
    
    def train_models(self) -> None:
        """Train all models on the training data"""
        if self.train_data is None or self.date_col is None or self.target_col is None:
            raise ValueError("Data must be split before training models")
        
        for model in self.models:
            try:
                logger.info(f"Training {model.name} model...")
                model.fit(self.train_data, self.target_col, self.date_col)
                logger.info(f"{model.name} model training completed")
            except Exception as e:
                logger.error(f"Error training {model.name} model: {str(e)}")
                # Remove failed model from the list
                self.models.remove(model)
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all models on the test data
        
        Returns:
            DataFrame with evaluation metrics for each model
        """
        if self.test_data is None or self.date_col is None or self.target_col is None:
            raise ValueError("Data must be split before evaluating models")
        
        evaluation_results = []
        
        for model in self.models:
            try:
                logger.info(f"Evaluating {model.name} model...")
                metrics = model.evaluate(self.test_data, self.target_col)
                evaluation_results.append(metrics)
                logger.info(f"{model.name} model evaluation completed")
            except Exception as e:
                logger.error(f"Error evaluating {model.name} model: {str(e)}")
                # Add failed evaluation with NaN values
                evaluation_results.append({
                    'model': model.name,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'R²': np.nan,
                    'MAPE': np.nan
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(evaluation_results)
        
        # Store results
        self.results['evaluation'] = results_df
        
        # Save to CSV
        results_path = self.output_dir / 'model_evaluation_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results_df
    
    def generate_forecasts(self, horizon: int = None) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for all models
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            Dictionary mapping model names to forecast DataFrames
        """
        forecasts = {}
        
        for model in self.models:
            try:
                logger.info(f"Generating {model.name} forecast...")
                forecast_df = model.predict(horizon=horizon)
                forecasts[model.name] = forecast_df
                logger.info(f"{model.name} forecast generation completed")
            except Exception as e:
                logger.error(f"Error generating {model.name} forecast: {str(e)}")
        
        # Store forecasts
        self.results['forecasts'] = forecasts
        
        return forecasts
    
    def plot_metric_comparison(self, metric: str = 'RMSE') -> plt.Figure:
        """
        Plot model comparison based on a specific metric
        
        Args:
            metric: Metric to compare (default: 'RMSE')
            
        Returns:
            Matplotlib figure
        """
        if 'evaluation' not in self.results:
            raise ValueError("Models must be evaluated before plotting metrics")
        
        # Get evaluation results
        results_df = self.results['evaluation']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by the metric
        sorted_df = results_df.sort_values(by=metric)
        
        # Plot barplot
        sns.barplot(x='model', y=metric, data=sorted_df, ax=ax)
        
        # Customize plot
        ax.set_title(f'Model Comparison - {metric}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / f'model_comparison_{metric}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metric comparison plot saved to {fig_path}")
        
        return fig
    
    def plot_forecast_comparison(self, include_history: bool = True) -> plt.Figure:
        """
        Plot forecast comparison for all models
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            Matplotlib figure
        """
        if 'forecasts' not in self.results:
            raise ValueError("Forecasts must be generated before plotting")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot historical data if requested
        if include_history and self.test_data is not None:
            historical = self.test_data.sort_values(by=self.date_col)
            ax.plot(historical[self.date_col], historical[self.target_col], 
                    label='Actual', color='black', marker='o', markersize=3)
        
        # Plot forecasts for each model with different colors
        colors = plt.cm.tab10.colors
        forecasts = self.results['forecasts']
        
        for i, (model_name, forecast_df) in enumerate(forecasts.items()):
            color_idx = i % len(colors)
            ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                    label=model_name, color=colors[color_idx], linestyle='--')
        
        # Customize plot
        ax.set_title('Forecast Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel(self.target_col)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format date labels
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'forecast_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Forecast comparison plot saved to {fig_path}")
        
        return fig
    
    def plot_all_metrics(self) -> List[plt.Figure]:
        """
        Plot comparison for all metrics
        
        Returns:
            List of Matplotlib figures
        """
        metrics = ['MAE', 'RMSE', 'MAPE', 'R²']
        figures = []
        
        for metric in metrics:
            try:
                fig = self.plot_metric_comparison(metric=metric)
                figures.append(fig)
            except Exception as e:
                logger.error(f"Error plotting {metric} comparison: {str(e)}")
        
        return figures
    
    def save_results(self) -> None:
        """Save all results to the output directory"""
        # Save evaluation results (already saved in evaluate_models)
        
        # Save forecasts
        if 'forecasts' in self.results:
            forecasts_dir = self.output_dir / 'forecasts'
            forecasts_dir.mkdir(exist_ok=True)
            
            for model_name, forecast_df in self.results['forecasts'].items():
                forecast_path = forecasts_dir / f'{model_name}_forecast.csv'
                forecast_df.to_csv(forecast_path, index=False)
            
            logger.info(f"Forecasts saved to {forecasts_dir}")
        
        # Save summary
        try:
            summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_models': len(self.models),
                'model_names': [model.name for model in self.models],
                'train_size': len(self.train_data) if self.train_data is not None else None,
                'test_size': len(self.test_data) if self.test_data is not None else None
            }
            
            # Add best model for each metric
            if 'evaluation' in self.results:
                evaluation = self.results['evaluation']
                
                for metric in ['MAE', 'RMSE', 'MAPE', 'R²']:
                    if metric in evaluation.columns:
                        # For R², higher is better, for others lower is better
                        ascending = metric != 'R²'
                        best_model = evaluation.sort_values(by=metric, ascending=ascending).iloc[0]
                        summary[f'best_model_{metric}'] = {
                            'model': best_model['model'],
                            'value': float(best_model[metric])
                        }
            
            summary_path = self.output_dir / 'comparison_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info(f"Summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
    
    def run_full_comparison(self, data: pd.DataFrame, date_col: str, target_col: str,
                          test_size: float = 0.2, split_date: Optional[str] = None,
                          horizon: int = 30, **model_params) -> Dict[str, Any]:
        """
        Run a full model comparison workflow
        
        Args:
            data: DataFrame with time series data
            date_col: Column name for date
            target_col: Column name for target variable
            test_size: Proportion of data to use for testing
            split_date: Date to split the data (format: 'YYYY-MM-DD')
            horizon: Number of periods to forecast
            **model_params: Parameters for model initialization
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting full model comparison workflow")
        
        # 1. Add all models
        self.add_all_models(**model_params)
        
        # 2. Split data
        self.split_data(data, date_col, target_col, test_size, split_date)
        
        # 3. Train models
        self.train_models()
        
        # 4. Evaluate models
        evaluation_df = self.evaluate_models()
        
        # 5. Generate forecasts
        forecasts = self.generate_forecasts(horizon=horizon)
        
        # 6. Plot comparisons
        self.plot_all_metrics()
        self.plot_forecast_comparison()
        
        # 7. Save results
        self.save_results()
        
        logger.info("Model comparison workflow completed successfully")
        
        return {
            'evaluation': evaluation_df,
            'forecasts': forecasts,
            'output_dir': str(self.output_dir)
        } 