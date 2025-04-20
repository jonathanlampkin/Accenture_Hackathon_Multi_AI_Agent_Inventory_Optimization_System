"""
Anomaly Detection for Inventory Management

This module provides methods for detecting anomalies in:
- Demand patterns
- Inventory levels
- Supply chain metrics
- Stockout frequency
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Advanced anomaly detection for inventory management.
    
    Detects unusual patterns in demand, inventory levels, and
    supply chain metrics.
    """
    
    def __init__(self, contamination: float = 0.05):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (default: 0.05)
        """
        self.contamination = contamination
        self.methods = {
            'statistical': self._statistical_detection,
            'isolation_forest': self._isolation_forest_detection,
            'local_outlier_factor': self._local_outlier_factor_detection,
            'time_series': self._time_series_detection,
            'pca': self._pca_detection
        }
        
        logger.info(f"Initialized AnomalyDetector with contamination: {contamination}")
    
    def _statistical_detection(self, data: np.ndarray, threshold: float = 3.0) -> List[bool]:
        """
        Detect anomalies using statistical methods (z-score, IQR).
        
        Args:
            data: Input data array
            threshold: Z-score threshold for anomalies (default: 3.0)
            
        Returns:
            Boolean array where True indicates an anomaly
        """
        if len(data) == 0:
            return []
        
        # Z-score method
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:  # Handle zero standard deviation
            return [False] * len(data)
        
        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold
    
    def _isolation_forest_detection(self, data: np.ndarray) -> List[bool]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: Input data array
            
        Returns:
            Boolean array where True indicates an anomaly
        """
        if len(data) < 10:  # Not enough data
            logger.warning("Not enough data for Isolation Forest, falling back to statistical method")
            return self._statistical_detection(data)
        
        # Reshape for sklearn
        X = data.reshape(-1, 1) if len(data.shape) == 1 else data
        
        # Fit Isolation Forest
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        clf.fit(X)
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        predictions = clf.predict(X)
        
        return predictions == -1  # True for anomalies
    
    def _local_outlier_factor_detection(self, data: np.ndarray) -> List[bool]:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            data: Input data array
            
        Returns:
            Boolean array where True indicates an anomaly
        """
        if len(data) < 10:  # Not enough data
            logger.warning("Not enough data for LOF, falling back to statistical method")
            return self._statistical_detection(data)
        
        # Reshape for sklearn
        X = data.reshape(-1, 1) if len(data.shape) == 1 else data
        
        # Fit LOF
        clf = LocalOutlierFactor(n_neighbors=min(20, len(data) // 2), contamination=self.contamination)
        predictions = clf.fit_predict(X)
        
        return predictions == -1  # True for anomalies
    
    def _time_series_detection(self, 
                             data: np.ndarray, 
                             seasonality: Optional[int] = None,
                             threshold: float = 3.0) -> List[bool]:
        """
        Detect anomalies in time series data using forecasting models.
        
        Args:
            data: Input time series data
            seasonality: Seasonality period (if known)
            threshold: Threshold for residual-based anomalies
            
        Returns:
            Boolean array where True indicates an anomaly
        """
        if len(data) < 10:  # Not enough data
            logger.warning("Not enough data for time series analysis, falling back to statistical method")
            return self._statistical_detection(data)
        
        try:
            # Choose model based on data characteristics
            if seasonality and len(data) >= max(4 * seasonality, 20):  # Enough data for seasonal model
                # SARIMA model
                model = SARIMAX(
                    data, 
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, seasonality)
                )
                result = model.fit(disp=False)
            else:
                # Exponential Smoothing
                model = ExponentialSmoothing(
                    data,
                    trend='add',
                    seasonal='add' if seasonality else None,
                    seasonal_periods=seasonality
                )
                result = model.fit()
            
            # Get fitted values
            fitted = result.fittedvalues
            
            # Calculate residuals
            residuals = data[result.k_exog:] - fitted
            
            # Calculate standardized residuals
            std_residuals = np.abs(residuals - np.mean(residuals)) / np.std(residuals)
            
            # Identify anomalies
            anomalies = std_residuals > threshold
            
            # Prepend False for any initial values that don't have fitted values
            if result.k_exog > 0:
                anomalies = np.concatenate([np.array([False] * result.k_exog), anomalies])
            
            return anomalies.tolist()
            
        except Exception as e:
            logger.warning(f"Error in time series detection: {e}, falling back to statistical method")
            return self._statistical_detection(data)
    
    def _pca_detection(self, data: np.ndarray, variance_retained: float = 0.8) -> List[bool]:
        """
        Detect anomalies using PCA reconstruction error.
        
        Args:
            data: Input multivariate data (n_samples, n_features)
            variance_retained: Amount of variance to retain
            
        Returns:
            Boolean array where True indicates an anomaly
        """
        if len(data.shape) < 2 or data.shape[1] < 2:
            logger.warning("PCA requires multivariate data, falling back to isolation forest")
            return self._isolation_forest_detection(data)
        
        # Standardize data
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        standardized_data = (data - mean) / std
        
        # Determine number of components to retain
        pca = PCA()
        pca.fit(standardized_data)
        
        var_cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(var_cumsum >= variance_retained) + 1
        
        # Create PCA with selected number of components
        pca = PCA(n_components=n_components)
        pca.fit(standardized_data)
        
        # Transform data to PCA space and back to original space
        transformed_data = pca.transform(standardized_data)
        reconstructed_data = pca.inverse_transform(transformed_data)
        
        # Calculate reconstruction error
        reconstruction_error = np.sum((standardized_data - reconstructed_data) ** 2, axis=1)
        
        # Determine threshold for anomalies
        threshold = np.mean(reconstruction_error) + threshold * np.std(reconstruction_error)
        
        return reconstruction_error > threshold
    
    def detect_anomalies(self, 
                       data: Union[np.ndarray, pd.Series, pd.DataFrame],
                       method: str = 'isolation_forest',
                       **kwargs) -> Tuple[List[bool], Dict[str, Any]]:
        """
        Detect anomalies in the provided data.
        
        Args:
            data: Input data (array, Series, or DataFrame)
            method: Detection method ('statistical', 'isolation_forest', 'local_outlier_factor', 'time_series', 'pca')
            **kwargs: Additional arguments for the specific detection method
            
        Returns:
            Tuple of (anomaly_flags, metadata)
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            values = data.values
        elif isinstance(data, pd.DataFrame):
            values = data.values
        else:
            values = np.array(data)
        
        # Validate method
        if method not in self.methods:
            logger.warning(f"Unknown method '{method}', falling back to isolation_forest")
            method = 'isolation_forest'
        
        # Call appropriate detection method
        detection_func = self.methods[method]
        anomalies = detection_func(values, **kwargs)
        
        # Calculate anomaly statistics
        anomaly_count = sum(anomalies)
        anomaly_percentage = (anomaly_count / len(values)) * 100 if len(values) > 0 else 0
        
        metadata = {
            'method': method,
            'anomaly_count': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'total_points': len(values),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_percentage:.2f}%) using {method}")
        
        return anomalies, metadata
    
    def detect_demand_anomalies(self, 
                              demand_data: pd.DataFrame,
                              demand_col: str = 'demand',
                              date_col: str = 'date',
                              product_col: Optional[str] = None) -> pd.DataFrame:
        """
        Detect anomalies in demand data.
        
        Args:
            demand_data: DataFrame with demand data
            demand_col: Column name for demand values
            date_col: Column name for dates
            product_col: Column name for product IDs (if None, treats all data as one product)
            
        Returns:
            DataFrame with anomaly flags
        """
        result_data = demand_data.copy()
        result_data['is_anomaly'] = False
        
        # Detect seasonality if sufficient data
        seasonality = None
        if len(demand_data) >= 14:
            # Check for weekly seasonality
            if len(set(demand_data[date_col].dt.dayofweek)) > 1:
                seasonality = 7
                
            # Check for monthly seasonality
            if len(set(demand_data[date_col].dt.month)) > 1:
                seasonality = 30
        
        # Process by product if product_col is specified
        if product_col:
            for product, group in demand_data.groupby(product_col):
                if len(group) < 5:  # Skip products with too few observations
                    continue
                
                anomalies, _ = self.detect_anomalies(
                    group[demand_col],
                    method='time_series' if len(group) >= 10 else 'statistical',
                    seasonality=seasonality
                )
                
                # Update the result DataFrame
                result_data.loc[group.index, 'is_anomaly'] = anomalies
        else:
            # Process all data together
            anomalies, _ = self.detect_anomalies(
                demand_data[demand_col],
                method='time_series' if len(demand_data) >= 10 else 'statistical',
                seasonality=seasonality
            )
            
            result_data['is_anomaly'] = anomalies
        
        return result_data
    
    def detect_inventory_level_anomalies(self,
                                       inventory_data: pd.DataFrame,
                                       level_col: str = 'inventory_level',
                                       product_col: Optional[str] = None,
                                       min_level_col: Optional[str] = None,
                                       max_level_col: Optional[str] = None) -> pd.DataFrame:
        """
        Detect anomalies in inventory levels.
        
        Args:
            inventory_data: DataFrame with inventory level data
            level_col: Column name for inventory level values
            product_col: Column name for product IDs (if None, treats all data as one product)
            min_level_col: Column name for min level values (if provided, will flag below-min as anomalies)
            max_level_col: Column name for max level values (if provided, will flag above-max as anomalies)
            
        Returns:
            DataFrame with anomaly flags and reasons
        """
        result_data = inventory_data.copy()
        result_data['is_anomaly'] = False
        result_data['anomaly_reason'] = ""
        
        # Process by product if product_col is specified
        if product_col:
            for product, group in inventory_data.groupby(product_col):
                if len(group) < 5:  # Skip products with too few observations
                    continue
                
                # Statistical anomaly detection
                anomalies, _ = self.detect_anomalies(
                    group[level_col],
                    method='isolation_forest' if len(group) >= 10 else 'statistical'
                )
                
                # Update the result DataFrame
                result_data.loc[group.index, 'is_anomaly'] = anomalies
                result_data.loc[group.index[anomalies], 'anomaly_reason'] = 'statistical'
        else:
            # Process all data together
            anomalies, _ = self.detect_anomalies(
                inventory_data[level_col],
                method='isolation_forest' if len(inventory_data) >= 10 else 'statistical'
            )
            
            result_data['is_anomaly'] = anomalies
            result_data.loc[result_data['is_anomaly'], 'anomaly_reason'] = 'statistical'
        
        # Check for min/max level violations if columns are provided
        if min_level_col:
            min_violations = inventory_data[level_col] < inventory_data[min_level_col]
            result_data.loc[min_violations, 'is_anomaly'] = True
            result_data.loc[min_violations, 'anomaly_reason'] = 'below_min'
        
        if max_level_col:
            max_violations = inventory_data[level_col] > inventory_data[max_level_col]
            result_data.loc[max_violations, 'is_anomaly'] = True
            result_data.loc[max_violations, 'anomaly_reason'] = 'above_max'
        
        return result_data
    
    def detect_supply_chain_anomalies(self,
                                     data: pd.DataFrame,
                                     features: List[str],
                                     date_col: str = 'date') -> pd.DataFrame:
        """
        Detect anomalies in supply chain metrics using multivariate analysis.
        
        Args:
            data: DataFrame with supply chain metrics
            features: List of feature columns to use for anomaly detection
            date_col: Column name for dates
            
        Returns:
            DataFrame with anomaly flags
        """
        result_data = data.copy()
        result_data['is_anomaly'] = False
        
        if len(data) < 5:
            logger.warning("Not enough data for supply chain anomaly detection")
            return result_data
        
        if len(features) < 2:
            # Single feature - use univariate method
            logger.info("Only one feature provided, using univariate method")
            anomalies, _ = self.detect_anomalies(
                data[features[0]],
                method='isolation_forest' if len(data) >= 10 else 'statistical'
            )
        else:
            # Multiple features - use multivariate method
            feature_data = data[features].values
            
            # Handle missing values
            feature_data = np.nan_to_num(feature_data)
            
            # Use PCA or Isolation Forest depending on data dimensions
            if feature_data.shape[1] >= 3 and feature_data.shape[0] >= 20:
                anomalies, _ = self.detect_anomalies(
                    feature_data,
                    method='pca'
                )
            else:
                anomalies, _ = self.detect_anomalies(
                    feature_data,
                    method='isolation_forest'
                )
        
        result_data['is_anomaly'] = anomalies
        
        return result_data

def detect_stockout_patterns(inventory_data: pd.DataFrame,
                           product_col: str,
                           date_col: str,
                           level_col: str,
                           threshold: float = 0) -> Dict[str, List[str]]:
    """
    Detect products with frequent stockouts.
    
    Args:
        inventory_data: DataFrame with inventory data
        product_col: Column name for product IDs
        date_col: Column name for dates
        level_col: Column name for inventory level
        threshold: Threshold to consider as stockout
        
    Returns:
        Dictionary with lists of products by stockout risk category
    """
    # Consider as stockout when inventory level is at or below threshold
    stockouts = inventory_data[inventory_data[level_col] <= threshold]
    
    # Count stockouts by product
    stockout_counts = stockouts.groupby(product_col).size()
    total_dates = inventory_data[date_col].nunique()
    
    # Calculate stockout frequency
    stockout_frequency = stockout_counts / total_dates
    
    # Categorize products by stockout frequency
    high_risk = stockout_frequency[stockout_frequency >= 0.1].index.tolist()  # 10%+ days with stockouts
    medium_risk = stockout_frequency[(stockout_frequency >= 0.05) & (stockout_frequency < 0.1)].index.tolist()
    low_risk = stockout_frequency[(stockout_frequency > 0) & (stockout_frequency < 0.05)].index.tolist()
    no_stockouts = set(inventory_data[product_col].unique()) - set(stockout_counts.index)
    
    return {
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'no_stockouts': list(no_stockouts)
    } 