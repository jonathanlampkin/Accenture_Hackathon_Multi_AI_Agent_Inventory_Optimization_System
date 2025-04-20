"""
Visualization utilities for the Multi-Agent Inventory Optimization System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def save_figure(fig, filename):
    """
    Save a matplotlib figure to the output directory
    """
    output_path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    return output_path

def plot_inventory_levels(inventory_df, product_id=None, store_id=None, top_n=10):
    """
    Plot inventory levels for products
    
    Args:
        inventory_df: DataFrame with inventory data
        product_id: Optional specific product ID to plot
        store_id: Optional specific store ID to plot
        top_n: Number of top products to display if product_id is None
    """
    plt.figure(figsize=(12, 8))
    
    if product_id is not None:
        # Filter data for the specific product
        if store_id is not None:
            data = inventory_df[(inventory_df['Product ID'] == product_id) & 
                               (inventory_df['Store ID'] == store_id)]
            title = f"Inventory Levels for Product {product_id} at Store {store_id}"
        else:
            data = inventory_df[inventory_df['Product ID'] == product_id]
            title = f"Inventory Levels for Product {product_id} Across Stores"
            
        # Plot inventory levels by store
        sns.barplot(x='Store ID', y='Stock Levels', data=data)
        plt.title(title)
        
    elif store_id is not None:
        # Filter data for the specific store
        data = inventory_df[inventory_df['Store ID'] == store_id]
        # Get top N products by stock level
        top_products = data.nlargest(top_n, 'Stock Levels')
        
        # Plot inventory levels for top products
        sns.barplot(x='Product ID', y='Stock Levels', data=top_products)
        plt.title(f"Top {top_n} Products by Inventory Level at Store {store_id}")
        plt.xticks(rotation=45)
        
    else:
        # Group by product and sum stock levels
        product_totals = inventory_df.groupby('Product ID')['Stock Levels'].sum().reset_index()
        top_products = product_totals.nlargest(top_n, 'Stock Levels')
        
        # Plot inventory levels for top products
        sns.barplot(x='Product ID', y='Stock Levels', data=top_products)
        plt.title(f"Top {top_n} Products by Total Inventory Level")
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_stockout_risk(inventory_df, product_id=None, store_id=None, top_n=10):
    """
    Plot stockout risk (frequency) for products
    
    Args:
        inventory_df: DataFrame with inventory data
        product_id: Optional specific product ID to plot
        store_id: Optional specific store ID to plot
        top_n: Number of top products to display if product_id is None
    """
    plt.figure(figsize=(12, 8))
    
    if product_id is not None:
        # Filter data for the specific product
        if store_id is not None:
            data = inventory_df[(inventory_df['Product ID'] == product_id) & 
                               (inventory_df['Store ID'] == store_id)]
            title = f"Stockout Risk for Product {product_id} at Store {store_id}"
        else:
            data = inventory_df[inventory_df['Product ID'] == product_id]
            title = f"Stockout Risk for Product {product_id} Across Stores"
            
        # Plot stockout frequency by store
        sns.barplot(x='Store ID', y='Stockout Frequency', data=data)
        plt.title(title)
        
    elif store_id is not None:
        # Filter data for the specific store
        data = inventory_df[inventory_df['Store ID'] == store_id]
        # Get top N products by stockout frequency
        top_products = data.nlargest(top_n, 'Stockout Frequency')
        
        # Plot stockout frequency for top products
        sns.barplot(x='Product ID', y='Stockout Frequency', data=top_products)
        plt.title(f"Top {top_n} Products by Stockout Risk at Store {store_id}")
        plt.xticks(rotation=45)
        
    else:
        # Group by product and calculate average stockout frequency
        product_avg = inventory_df.groupby('Product ID')['Stockout Frequency'].mean().reset_index()
        top_products = product_avg.nlargest(top_n, 'Stockout Frequency')
        
        # Plot average stockout frequency for top products
        sns.barplot(x='Product ID', y='Stockout Frequency', data=top_products)
        plt.title(f"Top {top_n} Products by Average Stockout Risk")
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_sales_trend(demand_df, product_id, store_id=None, window=7):
    """
    Plot sales trend for a specific product
    
    Args:
        demand_df: DataFrame with demand data
        product_id: Specific product ID to plot
        store_id: Optional specific store ID to plot
        window: Window size for rolling average
    """
    plt.figure(figsize=(14, 8))
    
    # Filter data for the specific product
    if store_id is not None:
        data = demand_df[(demand_df['Product ID'] == product_id) & 
                       (demand_df['Store ID'] == store_id)]
        title = f"Sales Trend for Product {product_id} at Store {store_id}"
    else:
        data = demand_df[demand_df['Product ID'] == product_id]
        # Group by date and sum sales quantities
        data = data.groupby('Date')['Sales Quantity'].sum().reset_index()
        title = f"Overall Sales Trend for Product {product_id}"
    
    # Sort by date
    data = data.sort_values('Date')
    
    # Plot sales trend
    plt.plot(data['Date'], data['Sales Quantity'], marker='o', linestyle='-', alpha=0.5, label='Daily Sales')
    
    # Add rolling average
    rolling_avg = data['Sales Quantity'].rolling(window=window).mean()
    plt.plot(data['Date'], rolling_avg, linewidth=2, label=f'{window}-Day Rolling Average')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_price_elasticity(pricing_df, product_id=None, store_id=None, top_n=10):
    """
    Plot price elasticity for products
    
    Args:
        pricing_df: DataFrame with pricing data
        product_id: Optional specific product ID to plot
        store_id: Optional specific store ID to plot
        top_n: Number of top products to display if product_id is None
    """
    plt.figure(figsize=(12, 8))
    
    if product_id is not None:
        # Filter data for the specific product
        if store_id is not None:
            data = pricing_df[(pricing_df['Product ID'] == product_id) & 
                             (pricing_df['Store ID'] == store_id)]
            title = f"Price Elasticity for Product {product_id} at Store {store_id}"
        else:
            data = pricing_df[pricing_df['Product ID'] == product_id]
            title = f"Price Elasticity for Product {product_id} Across Stores"
            
        # Plot elasticity by store
        sns.barplot(x='Store ID', y='Elasticity Index', data=data)
        plt.title(title)
        
    elif store_id is not None:
        # Filter data for the specific store
        data = pricing_df[pricing_df['Store ID'] == store_id]
        # Get top N products by elasticity
        top_products = data.nlargest(top_n, 'Elasticity Index')
        
        # Plot elasticity for top products
        sns.barplot(x='Product ID', y='Elasticity Index', data=top_products)
        plt.title(f"Top {top_n} Products by Price Elasticity at Store {store_id}")
        plt.xticks(rotation=45)
        
    else:
        # Group by product and calculate average elasticity
        product_avg = pricing_df.groupby('Product ID')['Elasticity Index'].mean().reset_index()
        top_products = product_avg.nlargest(top_n, 'Elasticity Index')
        
        # Plot average elasticity for top products
        sns.barplot(x='Product ID', y='Elasticity Index', data=top_products)
        plt.title(f"Top {top_n} Products by Average Price Elasticity")
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_price_vs_sales(pricing_df, demand_df, product_id, store_id=None):
    """
    Plot price vs. sales relationship for a specific product
    
    Args:
        pricing_df: DataFrame with pricing data
        demand_df: DataFrame with demand data
        product_id: Specific product ID to plot
        store_id: Optional specific store ID to plot
    """
    plt.figure(figsize=(10, 8))
    
    # Filter pricing data for the specific product
    if store_id is not None:
        pricing_data = pricing_df[(pricing_df['Product ID'] == product_id) & 
                                 (pricing_df['Store ID'] == store_id)]
        demand_data = demand_df[(demand_df['Product ID'] == product_id) & 
                               (demand_df['Store ID'] == store_id)]
        title = f"Price vs. Sales for Product {product_id} at Store {store_id}"
    else:
        # Group data by store
        pricing_data = pricing_df[pricing_df['Product ID'] == product_id]
        demand_data = demand_df[demand_df['Product ID'] == product_id]
        title = f"Price vs. Sales for Product {product_id} by Store"
    
    # Merge pricing and demand data
    if store_id is None:
        # Group demand data by store and calculate average sales
        demand_by_store = demand_data.groupby('Store ID')['Sales Quantity'].mean().reset_index()
        # Merge with pricing data
        merged_data = pricing_data.merge(demand_by_store, on='Store ID')
    else:
        # Just use the single store data point
        merged_data = pd.DataFrame({
            'Price': pricing_data['Price'].values,
            'Sales Quantity': [demand_data['Sales Quantity'].mean()]
        })
    
    # Plot price vs. sales
    if store_id is None:
        # Scatter plot with stores
        plt.scatter(merged_data['Price'], merged_data['Sales Quantity'], s=80, alpha=0.7)
        
        # Add store IDs as labels
        for i, row in merged_data.iterrows():
            plt.annotate(f"Store {row['Store ID']}", 
                         (row['Price'], row['Sales Quantity']),
                         xytext=(5, 5), textcoords='offset points')
    else:
        # Simple plot for a single store
        plt.scatter(merged_data['Price'], merged_data['Sales Quantity'], s=100, color='red')
    
    # Add regression line
    if len(merged_data) > 1:  # Only add regression if we have multiple data points
        x = merged_data['Price']
        y = merged_data['Sales Quantity']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        # Calculate and display correlation
        corr = np.corrcoef(x, y)[0, 1]
        plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.title(title)
    plt.xlabel('Price')
    plt.ylabel('Average Sales Quantity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def generate_inventory_dashboard(inventory_df, demand_df, pricing_df, product_id=None, store_id=None):
    """
    Generate a comprehensive dashboard of visualizations
    
    Args:
        inventory_df: DataFrame with inventory data
        demand_df: DataFrame with demand data
        pricing_df: DataFrame with pricing data
        product_id: Optional specific product ID to focus on
        store_id: Optional specific store ID to focus on
    
    Returns:
        List of saved figure paths
    """
    saved_figures = []
    
    # Plot inventory levels
    fig1 = plot_inventory_levels(inventory_df, product_id, store_id)
    filename1 = f"inventory_levels_{product_id or 'all'}_{store_id or 'all'}.png"
    saved_figures.append(save_figure(fig1, filename1))
    plt.close(fig1)
    
    # Plot stockout risk
    fig2 = plot_stockout_risk(inventory_df, product_id, store_id)
    filename2 = f"stockout_risk_{product_id or 'all'}_{store_id or 'all'}.png"
    saved_figures.append(save_figure(fig2, filename2))
    plt.close(fig2)
    
    # If a specific product is selected, show sales trend and price elasticity
    if product_id is not None:
        fig3 = plot_sales_trend(demand_df, product_id, store_id)
        filename3 = f"sales_trend_{product_id}_{store_id or 'all'}.png"
        saved_figures.append(save_figure(fig3, filename3))
        plt.close(fig3)
        
        fig4 = plot_price_vs_sales(pricing_df, demand_df, product_id, store_id)
        filename4 = f"price_vs_sales_{product_id}_{store_id or 'all'}.png"
        saved_figures.append(save_figure(fig4, filename4))
        plt.close(fig4)
    
    # Plot price elasticity
    fig5 = plot_price_elasticity(pricing_df, product_id, store_id)
    filename5 = f"price_elasticity_{product_id or 'all'}_{store_id or 'all'}.png"
    saved_figures.append(save_figure(fig5, filename5))
    plt.close(fig5)
    
    return saved_figures 