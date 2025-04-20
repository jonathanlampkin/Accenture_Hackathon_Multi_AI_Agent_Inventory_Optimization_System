"""
Statistical Forecasting Models for Inventory Optimization

This module contains implementations of various state-of-the-art time series forecasting models:
- SARIMA: Seasonal AutoRegressive Integrated Moving Average
- Prophet: Facebook's time series forecasting tool
- XGBoost: Gradient boosting for time series
- LightGBM: Light Gradient Boosting Machine
- Neural Prophet: Neural network based time series model
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logger = logging.getLogger(__name__)

class ForecastModel(ABC):
    """Base abstract class for all forecasting models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.forecast_horizon = 7  # Default forecast horizon (days)
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_col: str, date_col: str) -> None:
        """Fit the model to historical data"""
        pass
    
    @abstractmethod
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """Generate forecasts for the specified horizon"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        pass
    
    def plot_forecast(self, historical_data: pd.DataFrame, forecast_data: pd.DataFrame, 
                      date_col: str, target_col: str, title: Optional[str] = None) -> plt.Figure:
        """Plot historical data and forecasts"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_data[date_col], historical_data[target_col], 
                label='Historical', color='blue', marker='o', markersize=3)
        
        # Plot forecast
        ax.plot(forecast_data['ds'], forecast_data['yhat'], 
                label='Forecast', color='red', linestyle='--')
        
        # Plot confidence intervals if available
        if 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
            ax.fill_between(forecast_data['ds'], 
                           forecast_data['yhat_lower'], 
                           forecast_data['yhat_upper'], 
                           color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Set plot title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{self.name} Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format date labels
        fig.autofmt_xdate()
        
        return fig


class SARIMAModel(ForecastModel):
    """Seasonal Auto-Regressive Integrated Moving Average model"""
    
    def __init__(self, seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
                order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize SARIMA model
        
        Args:
            seasonal_order: Tuple of (P, D, Q, s) for seasonal component
            order: Tuple of (p, d, q) for non-seasonal component
        """
        super().__init__(name="SARIMA")
        self.seasonal_order = seasonal_order
        self.order = order
        self.date_column = None
        self.target_column = None
        self.data = None
        self.results = None
    
    def fit(self, data: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit SARIMA model to historical data
        
        Args:
            data: DataFrame with historical data
            target_col: Column name for target variable (e.g., 'sales')
            date_col: Column name for date
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Store column names for later use
            self.date_column = date_col
            self.target_column = target_col
            
            # Ensure data is sorted by date
            self.data = data.sort_values(by=date_col).copy()
            
            # Check for missing values
            if self.data[target_col].isnull().any():
                logger.warning(f"Missing values detected in {target_col}. Filling with forward fill method.")
                self.data[target_col] = self.data[target_col].ffill()
            
            # Fit SARIMA model
            model = SARIMAX(
                self.data[target_col],
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            logger.info(f"Fitting SARIMA model with order={self.order}, seasonal_order={self.seasonal_order}")
            self.results = model.fit(disp=False)
            self.is_fitted = True
            logger.info("SARIMA model fitting completed successfully")
            
        except ImportError:
            logger.error("statsmodels package is required for SARIMA model")
            raise
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            raise
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon
        
        Args:
            horizon: Number of periods to forecast (default: self.forecast_horizon)
            
        Returns:
            DataFrame with forecasted values and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon is None:
            horizon = self.forecast_horizon
        
        try:
            # Generate forecast
            forecast = self.results.get_forecast(steps=horizon)
            
            # Extract prediction and confidence intervals
            pred_mean = forecast.predicted_mean
            pred_ci = forecast.conf_int()
            
            # Create date range for forecast period
            last_date = self.data[self.date_column].max()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': pred_mean.values,
                'yhat_lower': pred_ci.iloc[:, 0].values,
                'yhat_upper': pred_ci.iloc[:, 1].values
            })
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating SARIMA forecast: {str(e)}")
            raise
    
    def evaluate(self, test_data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with test data
            target_col: Column name for target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        try:
            # Create copy of test data and ensure it's sorted
            test = test_data.sort_values(by=self.date_column).copy()
            
            # Generate predictions for test period
            horizon = len(test)
            forecast_df = self.predict(horizon=horizon)
            
            # Calculate metrics
            y_true = test[target_col].values
            y_pred = forecast_df['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
            
            return {
                'model': self.name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
        except Exception as e:
            logger.error(f"Error evaluating SARIMA model: {str(e)}")
            raise


class ProphetModel(ForecastModel):
    """Facebook Prophet forecasting model"""
    
    def __init__(self, yearly_seasonality: bool = True, 
                weekly_seasonality: bool = True, 
                daily_seasonality: bool = False,
                changepoint_prior_scale: float = 0.05):
        """
        Initialize Prophet model
        
        Args:
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            changepoint_prior_scale: Controls flexibility of trend
        """
        super().__init__(name="Prophet")
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.date_column = None
        self.target_column = None
    
    def fit(self, data: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit Prophet model to historical data
        
        Args:
            data: DataFrame with historical data
            target_col: Column name for target variable (e.g., 'sales')
            date_col: Column name for date
        """
        try:
            from prophet import Prophet
            
            # Store column names for later use
            self.date_column = date_col
            self.target_column = target_col
            
            # Create copy of data and ensure it's sorted
            df = data.sort_values(by=date_col).copy()
            
            # Check for missing values
            if df[target_col].isnull().any():
                logger.warning(f"Missing values detected in {target_col}. Filling with forward fill method.")
                df[target_col] = df[target_col].ffill()
            
            # Prophet requires columns named 'ds' and 'y'
            prophet_df = df[[date_col, target_col]].rename(
                columns={date_col: 'ds', target_col: 'y'}
            )
            
            # Initialize and fit Prophet model
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale
            )
            
            logger.info("Fitting Prophet model")
            self.model.fit(prophet_df)
            self.is_fitted = True
            logger.info("Prophet model fitting completed successfully")
            
        except ImportError:
            logger.error("prophet package is required for Prophet model")
            raise
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            raise
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon
        
        Args:
            horizon: Number of periods to forecast (default: self.forecast_horizon)
            
        Returns:
            DataFrame with forecasted values and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon is None:
            horizon = self.forecast_horizon
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=horizon)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Return only the forecast period
            return forecast.iloc[-horizon:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            logger.error(f"Error generating Prophet forecast: {str(e)}")
            raise
    
    def evaluate(self, test_data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with test data
            target_col: Column name for target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        try:
            # Create copy of test data and ensure it's sorted
            test = test_data.sort_values(by=self.date_column).copy()
            
            # Generate predictions for test period
            horizon = len(test)
            forecast_df = self.predict(horizon=horizon)
            
            # Prepare test data in Prophet format
            test_prophet = test[[self.date_column, target_col]].rename(
                columns={self.date_column: 'ds', target_col: 'y'}
            )
            
            # Merge predictions with actual values
            evaluation_df = pd.merge(
                test_prophet, forecast_df[['ds', 'yhat']], on='ds', how='left'
            )
            
            # Calculate metrics
            y_true = evaluation_df['y'].values
            y_pred = evaluation_df['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
            
            return {
                'model': self.name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
        except Exception as e:
            logger.error(f"Error evaluating Prophet model: {str(e)}")
            raise


class XGBoostForecastModel(ForecastModel):
    """XGBoost for time series forecasting"""
    
    def __init__(self, max_lag: int = 7, n_estimators: int = 100, 
                learning_rate: float = 0.1, max_depth: int = 5):
        """
        Initialize XGBoost forecasting model
        
        Args:
            max_lag: Number of lagged features to create
            n_estimators: Number of boosting rounds
            learning_rate: Boosting learning rate
            max_depth: Maximum tree depth
        """
        super().__init__(name="XGBoost")
        self.max_lag = max_lag
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.date_column = None
        self.target_column = None
        self.last_values = None
    
    def _create_features(self, data: pd.DataFrame, target_col: str, for_prediction: bool = False) -> pd.DataFrame:
        """Create lag features for XGBoost model"""
        df = data.copy()
        
        # Create lag features
        for lag in range(1, self.max_lag + 1):
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Create date-based features
        df['dayofweek'] = pd.to_datetime(df[self.date_column]).dt.dayofweek
        df['month'] = pd.to_datetime(df[self.date_column]).dt.month
        df['day'] = pd.to_datetime(df[self.date_column]).dt.day
        
        # Drop rows with NaN values (due to lag creation)
        if not for_prediction:
            df = df.dropna()
        
        return df
    
    def fit(self, data: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit XGBoost model to historical data
        
        Args:
            data: DataFrame with historical data
            target_col: Column name for target variable (e.g., 'sales')
            date_col: Column name for date
        """
        try:
            import xgboost as xgb
            
            # Store column names for later use
            self.date_column = date_col
            self.target_column = target_col
            
            # Create copy of data and ensure it's sorted
            df = data.sort_values(by=date_col).copy()
            
            # Check for missing values in target
            if df[target_col].isnull().any():
                logger.warning(f"Missing values detected in {target_col}. Filling with forward fill method.")
                df[target_col] = df[target_col].ffill()
            
            # Create features
            featured_df = self._create_features(df, target_col)
            
            # Store last values for prediction
            self.last_values = df.iloc[-self.max_lag:][target_col].values
            
            # Split features and target
            X = featured_df.drop([target_col, date_col], axis=1)
            y = featured_df[target_col]
            
            # Initialize and fit XGBoost model
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                objective='reg:squarederror',
                n_jobs=-1
            )
            
            logger.info(f"Fitting XGBoost model with {self.n_estimators} estimators")
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("XGBoost model fitting completed successfully")
            
            # Save feature names
            self.feature_names = X.columns.tolist()
            
        except ImportError:
            logger.error("xgboost package is required for XGBoost model")
            raise
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {str(e)}")
            raise
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon
        
        Args:
            horizon: Number of periods to forecast (default: self.forecast_horizon)
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon is None:
            horizon = self.forecast_horizon
        
        try:
            import xgboost as xgb
            
            # Generate forecast dates
            last_date = pd.to_datetime(self.date_column)
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            # Initialize containers for predictions
            predictions = []
            prediction_dates = []
            lower_bounds = []
            upper_bounds = []
            
            # Initial historical values
            historical_values = self.last_values.copy()
            
            # Generate predictions one step at a time
            for i in range(horizon):
                # Create a dataframe for next prediction
                next_date = forecast_dates[i]
                next_df = pd.DataFrame({
                    self.date_column: [next_date],
                    self.target_column: [0]  # Placeholder
                })
                
                # Create features for prediction
                for lag in range(1, self.max_lag + 1):
                    if lag <= len(historical_values):
                        next_df[f'lag_{lag}'] = historical_values[-lag]
                    else:
                        next_df[f'lag_{lag}'] = 0
                
                # Add date features
                next_df['dayofweek'] = next_date.dayofweek
                next_df['month'] = next_date.month
                next_df['day'] = next_date.day
                
                # Make prediction
                X_pred = next_df[self.feature_names]
                pred = self.model.predict(X_pred)[0]
                
                # Add prediction to historical values for next iteration
                historical_values = np.append(historical_values[1:], pred)
                
                # Store prediction
                predictions.append(pred)
                prediction_dates.append(next_date)
                
                # Generate simple confidence intervals (±10% as an example)
                lower_bounds.append(pred * 0.9)
                upper_bounds.append(pred * 1.1)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': prediction_dates,
                'yhat': predictions,
                'yhat_lower': lower_bounds,
                'yhat_upper': upper_bounds
            })
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating XGBoost forecast: {str(e)}")
            raise
    
    def evaluate(self, test_data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with test data
            target_col: Column name for target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        try:
            # Create copy of test data and ensure it's sorted
            test = test_data.sort_values(by=self.date_column).copy()
            
            # Generate predictions for test period
            horizon = len(test)
            forecast_df = self.predict(horizon=horizon)
            
            # Calculate metrics
            y_true = test[target_col].values
            y_pred = forecast_df['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
            
            return {
                'model': self.name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
        except Exception as e:
            logger.error(f"Error evaluating XGBoost model: {str(e)}")
            raise


class LightGBMForecastModel(ForecastModel):
    """LightGBM for time series forecasting"""
    
    def __init__(self, max_lag: int = 7, n_estimators: int = 100, 
                learning_rate: float = 0.1, num_leaves: int = 31):
        """
        Initialize LightGBM forecasting model
        
        Args:
            max_lag: Number of lagged features to create
            n_estimators: Number of boosting rounds
            learning_rate: Boosting learning rate
            num_leaves: Maximum number of leaves
        """
        super().__init__(name="LightGBM")
        self.max_lag = max_lag
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.date_column = None
        self.target_column = None
        self.last_values = None
    
    def _create_features(self, data: pd.DataFrame, target_col: str, for_prediction: bool = False) -> pd.DataFrame:
        """Create lag features for LightGBM model"""
        df = data.copy()
        
        # Create lag features
        for lag in range(1, self.max_lag + 1):
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Create date-based features
        df['dayofweek'] = pd.to_datetime(df[self.date_column]).dt.dayofweek
        df['month'] = pd.to_datetime(df[self.date_column]).dt.month
        df['day'] = pd.to_datetime(df[self.date_column]).dt.day
        
        # Drop rows with NaN values (due to lag creation)
        if not for_prediction:
            df = df.dropna()
        
        return df
    
    def fit(self, data: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit LightGBM model to historical data
        
        Args:
            data: DataFrame with historical data
            target_col: Column name for target variable (e.g., 'sales')
            date_col: Column name for date
        """
        try:
            import lightgbm as lgb
            
            # Store column names for later use
            self.date_column = date_col
            self.target_column = target_col
            
            # Create copy of data and ensure it's sorted
            df = data.sort_values(by=date_col).copy()
            
            # Check for missing values in target
            if df[target_col].isnull().any():
                logger.warning(f"Missing values detected in {target_col}. Filling with forward fill method.")
                df[target_col] = df[target_col].ffill()
            
            # Create features
            featured_df = self._create_features(df, target_col)
            
            # Store last values for prediction
            self.last_values = df.iloc[-self.max_lag:][target_col].values
            
            # Split features and target
            X = featured_df.drop([target_col, date_col], axis=1)
            y = featured_df[target_col]
            
            # Initialize and fit LightGBM model
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                objective='regression',
                n_jobs=-1
            )
            
            logger.info(f"Fitting LightGBM model with {self.n_estimators} estimators")
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("LightGBM model fitting completed successfully")
            
            # Save feature names
            self.feature_names = X.columns.tolist()
            
        except ImportError:
            logger.error("lightgbm package is required for LightGBM model")
            raise
        except Exception as e:
            logger.error(f"Error fitting LightGBM model: {str(e)}")
            raise
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon
        
        Args:
            horizon: Number of periods to forecast (default: self.forecast_horizon)
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon is None:
            horizon = self.forecast_horizon
        
        try:
            # Generate forecast dates
            last_date = pd.to_datetime(self.date_column)
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            # Initialize containers for predictions
            predictions = []
            prediction_dates = []
            lower_bounds = []
            upper_bounds = []
            
            # Initial historical values
            historical_values = self.last_values.copy()
            
            # Generate predictions one step at a time
            for i in range(horizon):
                # Create a dataframe for next prediction
                next_date = forecast_dates[i]
                next_df = pd.DataFrame({
                    self.date_column: [next_date],
                    self.target_column: [0]  # Placeholder
                })
                
                # Create features for prediction
                for lag in range(1, self.max_lag + 1):
                    if lag <= len(historical_values):
                        next_df[f'lag_{lag}'] = historical_values[-lag]
                    else:
                        next_df[f'lag_{lag}'] = 0
                
                # Add date features
                next_df['dayofweek'] = next_date.dayofweek
                next_df['month'] = next_date.month
                next_df['day'] = next_date.day
                
                # Make prediction
                X_pred = next_df[self.feature_names]
                pred = self.model.predict(X_pred)[0]
                
                # Add prediction to historical values for next iteration
                historical_values = np.append(historical_values[1:], pred)
                
                # Store prediction
                predictions.append(pred)
                prediction_dates.append(next_date)
                
                # Generate simple confidence intervals (±10% as an example)
                lower_bounds.append(pred * 0.9)
                upper_bounds.append(pred * 1.1)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': prediction_dates,
                'yhat': predictions,
                'yhat_lower': lower_bounds,
                'yhat_upper': upper_bounds
            })
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating LightGBM forecast: {str(e)}")
            raise
    
    def evaluate(self, test_data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with test data
            target_col: Column name for target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        try:
            # Create copy of test data and ensure it's sorted
            test = test_data.sort_values(by=self.date_column).copy()
            
            # Generate predictions for test period
            horizon = len(test)
            forecast_df = self.predict(horizon=horizon)
            
            # Calculate metrics
            y_true = test[target_col].values
            y_pred = forecast_df['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
            
            return {
                'model': self.name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
        except Exception as e:
            logger.error(f"Error evaluating LightGBM model: {str(e)}")
            raise


class NeuralProphetModel(ForecastModel):
    """Neural Prophet forecasting model"""
    
    def __init__(self, n_changepoints: int = 10, 
                n_forecasts: int = 1, 
                yearly_seasonality: bool = True, 
                weekly_seasonality: bool = True):
        """
        Initialize Neural Prophet model
        
        Args:
            n_changepoints: Number of changepoints for trend
            n_forecasts: Number of steps to forecast
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
        """
        super().__init__(name="NeuralProphet")
        self.n_changepoints = n_changepoints
        self.n_forecasts = n_forecasts
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.date_column = None
        self.target_column = None
    
    def fit(self, data: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit Neural Prophet model to historical data
        
        Args:
            data: DataFrame with historical data
            target_col: Column name for target variable (e.g., 'sales')
            date_col: Column name for date
        """
        try:
            from neuralprophet import NeuralProphet
            
            # Store column names for later use
            self.date_column = date_col
            self.target_column = target_col
            
            # Create copy of data and ensure it's sorted
            df = data.sort_values(by=date_col).copy()
            
            # Check for missing values
            if df[target_col].isnull().any():
                logger.warning(f"Missing values detected in {target_col}. Filling with forward fill method.")
                df[target_col] = df[target_col].ffill()
            
            # Neural Prophet requires columns named 'ds' and 'y'
            prophet_df = df[[date_col, target_col]].rename(
                columns={date_col: 'ds', target_col: 'y'}
            )
            
            # Initialize and fit Neural Prophet model
            self.model = NeuralProphet(
                n_changepoints=self.n_changepoints,
                n_forecasts=self.n_forecasts,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality
            )
            
            logger.info("Fitting Neural Prophet model")
            self.model.fit(prophet_df, freq='D')
            self.is_fitted = True
            logger.info("Neural Prophet model fitting completed successfully")
            
        except ImportError:
            logger.error("neuralprophet package is required for Neural Prophet model")
            raise
        except Exception as e:
            logger.error(f"Error fitting Neural Prophet model: {str(e)}")
            raise
    
    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon
        
        Args:
            horizon: Number of periods to forecast (default: self.forecast_horizon)
            
        Returns:
            DataFrame with forecasted values and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon is None:
            horizon = self.forecast_horizon
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(df=pd.DataFrame(), periods=horizon)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Rename columns to match our standard
            forecast_df = forecast.rename(columns={
                'ds': 'ds',
                'yhat1': 'yhat',
                'yhat1_lower': 'yhat_lower',
                'yhat1_upper': 'yhat_upper'
            })
            
            # If confidence intervals are not available, create them
            if 'yhat1_lower' not in forecast.columns or 'yhat1_upper' not in forecast.columns:
                forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.9
                forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.1
            
            # Return only necessary columns
            return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            logger.error(f"Error generating Neural Prophet forecast: {str(e)}")
            raise
    
    def evaluate(self, test_data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with test data
            target_col: Column name for target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        try:
            # Create copy of test data and ensure it's sorted
            test = test_data.sort_values(by=self.date_column).copy()
            
            # Generate predictions for test period
            horizon = len(test)
            forecast_df = self.predict(horizon=horizon)
            
            # Prepare test data in Neural Prophet format
            test_prophet = test[[self.date_column, target_col]].rename(
                columns={self.date_column: 'ds', target_col: 'y'}
            )
            
            # Merge predictions with actual values
            evaluation_df = pd.merge(
                test_prophet, forecast_df[['ds', 'yhat']], on='ds', how='left'
            )
            
            # Calculate metrics
            y_true = evaluation_df['y'].values
            y_pred = evaluation_df['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
            
            return {
                'model': self.name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
        except Exception as e:
            logger.error(f"Error evaluating Neural Prophet model: {str(e)}")
            raise 