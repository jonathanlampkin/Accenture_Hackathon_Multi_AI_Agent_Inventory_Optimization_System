import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger("InventorySystem")

def check_data_files() -> bool:
    """Check if required data files exist"""
    required_files = [
        'data/inventory_data.csv',
        'data/supplier_data.csv',
        'data/market_data.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        return False
    
    return True

def get_data_summary() -> Dict[str, Dict[str, Any]]:
    """Generate summary statistics for all data files"""
    summary = {}
    
    try:
        # Inventory data summary
        inventory_data = pd.read_csv('data/inventory_data.csv')
        summary['inventory'] = {
            'rows': len(inventory_data),
            'columns': len(inventory_data.columns),
            'products': inventory_data['product_id'].nunique(),
            'stores': inventory_data['store_id'].nunique(),
            'date_range': f"{inventory_data['date'].min()} to {inventory_data['date'].max()}"
        }
        
        # Supplier data summary
        supplier_data = pd.read_csv('data/supplier_data.csv')
        summary['supplier'] = {
            'rows': len(supplier_data),
            'columns': len(supplier_data.columns),
            'suppliers': supplier_data['supplier_id'].nunique(),
            'avg_lead_time': supplier_data['lead_time'].mean(),
            'avg_reliability': supplier_data['reliability'].mean()
        }
        
        # Market data summary
        market_data = pd.read_csv('data/market_data.csv')
        summary['market'] = {
            'rows': len(market_data),
            'columns': len(market_data.columns),
            'competitors': market_data['competitor_id'].nunique(),
            'avg_price': market_data['price'].mean(),
            'avg_demand': market_data['demand'].mean()
        }
        
    except Exception as e:
        logger.error(f"Error generating data summary: {e}")
        return {}
    
    return summary

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the inventory data"""
    try:
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Add derived features
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        
        # Calculate rolling statistics
        data['rolling_avg_sales'] = data.groupby('product_id')['sales'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        data['rolling_std_sales'] = data.groupby('product_id')['sales'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
        
        return data
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return data

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save optimization results to files"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save text results
        with open(f"{output_dir}/optimization_results.txt", "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        # Save visualizations if they exist
        if 'visualizations' in results:
            for name, fig in results['visualizations'].items():
                fig.savefig(f"{output_dir}/{name}.png")
        
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}") 