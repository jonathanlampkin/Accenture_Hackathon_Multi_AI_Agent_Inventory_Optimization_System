"""
Data loading utilities for the Multi-Agent Inventory Optimization System
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import logging

# Set up logger
logger = logging.getLogger("InventorySystem.DataLoader")

# Add the src directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_inventory_data():
    """
    Load and preprocess inventory monitoring data
    
    Returns:
        DataFrame: Preprocessed inventory data or None if loading fails
    """
    try:
        if not os.path.exists(config.INVENTORY_DATA):
            logger.error(f"Inventory data file not found at: {config.INVENTORY_DATA}")
            logger.error("Please make sure inventory_monitoring.csv is available.")
            return None
            
        df = pd.read_csv(config.INVENTORY_DATA)
        
        # Convert expiry date to datetime
        df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce')
        
        # Calculate days until expiry
        df['Days Until Expiry'] = (df['Expiry Date'] - pd.Timestamp.now()).dt.days
        
        # Replace NaN values with median values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64] and df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        logger.info(f"Loaded inventory data with {df.shape[0]} records and {df.shape[1]} features")
        return df
    
    except Exception as e:
        logger.error(f"Error loading inventory data: {e}")
        return None

def load_demand_data():
    """
    Load and preprocess demand forecasting data
    
    Returns:
        DataFrame: Preprocessed demand data or None if loading fails
    """
    try:
        if not os.path.exists(config.DEMAND_DATA):
            logger.error(f"Demand data file not found at: {config.DEMAND_DATA}")
            logger.error("Please make sure demand_forecasting.csv is available.")
            return None
            
        df = pd.read_csv(config.DEMAND_DATA)
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Sort by date
        df.sort_values('Date', inplace=True)
        
        # Replace NaN values with median values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64] and df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        logger.info(f"Loaded demand data with {df.shape[0]} records and {df.shape[1]} features")
        return df
    
    except Exception as e:
        logger.error(f"Error loading demand data: {e}")
        return None

def load_pricing_data():
    """
    Load and preprocess pricing optimization data
    
    Returns:
        DataFrame: Preprocessed pricing data or None if loading fails
    """
    try:
        if not os.path.exists(config.PRICING_DATA):
            logger.error(f"Pricing data file not found at: {config.PRICING_DATA}")
            logger.error("Please make sure pricing_optimization.csv is available.")
            return None
            
        df = pd.read_csv(config.PRICING_DATA)
        
        # Replace NaN values with median values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64] and df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        logger.info(f"Loaded pricing data with {df.shape[0]} records and {df.shape[1]} features")
        return df
    
    except Exception as e:
        logger.error(f"Error loading pricing data: {e}")
        return None

def get_product_data(product_id, store_id=None):
    """
    Get all data for a specific product (and optionally store)
    
    Args:
        product_id: ID of the product
        store_id: Optional ID of the store
    
    Returns:
        Dictionary with inventory, demand, and pricing data for the product
    """
    inventory_df = load_inventory_data()
    demand_df = load_demand_data()
    pricing_df = load_pricing_data()
    
    # Check if any of the data loading failed
    if any(df is None for df in [inventory_df, demand_df, pricing_df]):
        logger.error("Cannot retrieve product data due to missing input data")
        return None
    
    result = {}
    
    # Filter inventory data
    if store_id is not None:
        inv_filter = (inventory_df['Product ID'] == product_id) & (inventory_df['Store ID'] == store_id)
    else:
        inv_filter = (inventory_df['Product ID'] == product_id)
    result['inventory'] = inventory_df[inv_filter]
    
    # Filter demand data
    if store_id is not None:
        demand_filter = (demand_df['Product ID'] == product_id) & (demand_df['Store ID'] == store_id)
    else:
        demand_filter = (demand_df['Product ID'] == product_id)
    result['demand'] = demand_df[demand_filter]
    
    # Filter pricing data
    if store_id is not None:
        price_filter = (pricing_df['Product ID'] == product_id) & (pricing_df['Store ID'] == store_id)
    else:
        price_filter = (pricing_df['Product ID'] == product_id)
    result['pricing'] = pricing_df[price_filter]
    
    # Check if any data was found
    if all(len(df) == 0 for df in result.values()):
        logger.warning(f"No data found for Product ID: {product_id}" + 
                     (f", Store ID: {store_id}" if store_id else ""))
    
    return result

def get_store_data(store_id):
    """
    Get all data for a specific store
    
    Args:
        store_id: ID of the store
    
    Returns:
        Dictionary with inventory, demand, and pricing data for the store
    """
    inventory_df = load_inventory_data()
    demand_df = load_demand_data()
    pricing_df = load_pricing_data()
    
    # Check if any of the data loading failed
    if any(df is None for df in [inventory_df, demand_df, pricing_df]):
        logger.error("Cannot retrieve store data due to missing input data")
        return None
    
    result = {}
    
    # Filter by store ID
    result['inventory'] = inventory_df[inventory_df['Store ID'] == store_id]
    result['demand'] = demand_df[demand_df['Store ID'] == store_id]
    result['pricing'] = pricing_df[pricing_df['Store ID'] == store_id]
    
    # Check if any data was found
    if all(len(df) == 0 for df in result.values()):
        logger.warning(f"No data found for Store ID: {store_id}")
    
    return result

def get_data_summary():
    """
    Generate a summary of the available data
    
    Returns:
        dict: Summary statistics for each dataset
    """
    summary = {}
    
    # Load datasets
    inventory_df = load_inventory_data()
    demand_df = load_demand_data()
    pricing_df = load_pricing_data()
    
    if inventory_df is not None:
        summary['inventory'] = {
            'records': len(inventory_df),
            'products': inventory_df['Product ID'].nunique(),
            'stores': inventory_df['Store ID'].nunique(),
            'avg_stock_level': inventory_df['Stock Levels'].mean(),
            'avg_lead_time': inventory_df['Supplier Lead Time (days)'].mean(),
            'avg_stockout_freq': inventory_df['Stockout Frequency'].mean()
        }
    
    if demand_df is not None:
        summary['demand'] = {
            'records': len(demand_df),
            'products': demand_df['Product ID'].nunique(),
            'stores': demand_df['Store ID'].nunique(),
            'date_range': (demand_df['Date'].min(), demand_df['Date'].max()),
            'avg_sales': demand_df['Sales Quantity'].mean()
        }
    
    if pricing_df is not None:
        summary['pricing'] = {
            'records': len(pricing_df),
            'products': pricing_df['Product ID'].nunique(),
            'stores': pricing_df['Store ID'].nunique(),
            'avg_price': pricing_df['Price'].mean(),
            'avg_storage_cost': pricing_df['Storage Cost'].mean()
        }
    
    return summary


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the data loading functions
    inventory_df = load_inventory_data()
    demand_df = load_demand_data()
    pricing_df = load_pricing_data()
    
    if inventory_df is not None and demand_df is not None and pricing_df is not None:
        print("\nInventory Data Sample:")
        print(inventory_df.head())
        
        print("\nDemand Data Sample:")
        print(demand_df.head())
        
        print("\nPricing Data Sample:")
        print(pricing_df.head())
        
        # Test product data extraction
        product_id = inventory_df['Product ID'].iloc[0]
        store_id = inventory_df['Store ID'].iloc[0]
        
        print(f"\nGetting data for Product ID {product_id}, Store ID {store_id}")
        product_data = get_product_data(product_id, store_id)
        
        if product_data:
            for key, value in product_data.items():
                print(f"\n{key.capitalize()} Data:")
                print(value.head())
        
        # Print data summary
        print("\nData Summary:")
        summary = get_data_summary()
        for dataset, stats in summary.items():
            print(f"\n{dataset.capitalize()} Dataset Summary:")
            for key, value in stats.items():
                print(f"  {key}: {value}") 