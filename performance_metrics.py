#!/usr/bin/env python3
"""
Performance Metrics Dashboard for Inventory Optimization System
Calculates and displays key performance indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("Set2")

# Load data
def load_data():
    """Load the datasets and results"""
    demand_data = pd.read_csv("data/demand_forecasting.csv")
    inventory_data = pd.read_csv("data/inventory_monitoring.csv")
    pricing_data = pd.read_csv("data/pricing_optimization.csv")
    
    # Load results if available
    results = {}
    results_dir = "optimization_results"
    
    if os.path.exists(f"{results_dir}/optimal_stock_levels.csv"):
        results["optimal_levels"] = pd.read_csv(f"{results_dir}/optimal_stock_levels.csv")
        
    if os.path.exists(f"{results_dir}/pricing_strategy.csv"):
        results["pricing_strategy"] = pd.read_csv(f"{results_dir}/pricing_strategy.csv")
    
    return demand_data, inventory_data, pricing_data, results

# Create output directory for performance metrics
METRICS_DIR = "performance_metrics"
if not os.path.exists(METRICS_DIR):
    os.makedirs(METRICS_DIR)

# Calculate service level metrics
def calculate_service_levels(inventory_data):
    """Calculate service level metrics"""
    
    # Calculate fill rate (percentage of demand that can be satisfied from stock)
    # Fill rate = 1 - (stockout frequency / order frequency)
    # We'll approximate order frequency based on stock levels
    
    # Normalize stockout frequency to a percentage
    max_stockout = inventory_data['Stockout Frequency'].max()
    inventory_data['Stockout Percentage'] = inventory_data['Stockout Frequency'] / max_stockout * 100
    
    # Calculate service level (1 - stockout probability)
    inventory_data['Service Level'] = 100 - inventory_data['Stockout Percentage']
    
    # Calculate average service level
    avg_service_level = inventory_data['Service Level'].mean()
    
    # Calculate service level by store
    store_service_levels = inventory_data.groupby('Store ID')['Service Level'].mean().reset_index()
    
    return {
        'avg_service_level': avg_service_level,
        'store_service_levels': store_service_levels,
        'inventory_data': inventory_data
    }

# Calculate inventory turnover
def calculate_inventory_turnover(demand_data, inventory_data):
    """Calculate inventory turnover metrics"""
    
    # Create a map of average demand by product
    product_demand = demand_data.groupby('Product ID')['Sales Quantity'].mean().to_dict()
    
    # Add average demand to inventory data
    inventory_data['Avg Demand'] = inventory_data['Product ID'].map(product_demand)
    
    # Fill NaN values with overall average
    overall_avg_demand = demand_data['Sales Quantity'].mean()
    inventory_data['Avg Demand'].fillna(overall_avg_demand, inplace=True)
    
    # Calculate turnover (annual demand / average inventory)
    # Assuming our demand data represents monthly figures, multiply by 12 for annual
    inventory_data['Annual Demand'] = inventory_data['Avg Demand'] * 12
    inventory_data['Inventory Turnover'] = inventory_data['Annual Demand'] / inventory_data['Stock Levels']
    
    # Calculate average turnover
    avg_turnover = inventory_data['Inventory Turnover'].mean()
    
    # Calculate turnover by store
    store_turnover = inventory_data.groupby('Store ID')['Inventory Turnover'].mean().reset_index()
    
    return {
        'avg_turnover': avg_turnover,
        'store_turnover': store_turnover,
        'inventory_data': inventory_data
    }

# Calculate stockout risk metrics
def calculate_stockout_metrics(inventory_data):
    """Calculate stockout risk metrics"""
    
    # If Stockout Risk Score hasn't been calculated yet
    if 'Stockout Risk Score' not in inventory_data.columns:
        inventory_data['Stockout Risk Score'] = (
            inventory_data['Stockout Frequency'] / 
            (inventory_data['Stock Levels'] / inventory_data['Reorder Point'])
        )
        
        # Handle infinite values
        inventory_data['Stockout Risk Score'] = inventory_data['Stockout Risk Score'].replace([np.inf, -np.inf], np.nan)
        inventory_data['Stockout Risk Score'] = inventory_data['Stockout Risk Score'].fillna(0)
    
    # Get top risky products
    high_risk_products = inventory_data.sort_values('Stockout Risk Score', ascending=False).head(10)
    
    # Calculate average risk score
    avg_risk = inventory_data['Stockout Risk Score'].mean()
    
    # Calculate percentage of products with high risk (above a threshold)
    high_risk_threshold = inventory_data['Stockout Risk Score'].quantile(0.9)  # Top 10% as high risk
    high_risk_pct = (inventory_data['Stockout Risk Score'] > high_risk_threshold).mean() * 100
    
    return {
        'avg_risk': avg_risk,
        'high_risk_pct': high_risk_pct,
        'high_risk_products': high_risk_products,
        'inventory_data': inventory_data
    }

# Calculate forecasting performance
def simulate_forecast_performance(demand_data):
    """Simulate forecasting performance metrics"""
    # We'll simulate forecast performance by creating a simple forecast and comparing to actual
    
    # Group data by product to get time series
    products = demand_data['Product ID'].unique()
    np.random.seed(42)  # For reproducibility
    sample_products = np.random.choice(products, size=min(10, len(products)), replace=False)
    
    forecast_errors = []
    product_metrics = []
    
    for product_id in sample_products:
        product_data = demand_data[demand_data['Product ID'] == product_id]
        
        if len(product_data) < 5:  # Skip products with too little data
            continue
            
        # Sort by date if available, otherwise just use the data as is
        if 'Date' in product_data.columns:
            product_data = product_data.sort_values('Date')
        
        # Use 80% for training, 20% for testing
        train_size = int(len(product_data) * 0.8)
        train_data = product_data.iloc[:train_size]
        test_data = product_data.iloc[train_size:]
        
        # Simple forecast: use mean of training data as forecast
        forecast = train_data['Sales Quantity'].mean()
        
        # Calculate errors
        actual = test_data['Sales Quantity'].values
        forecasts = np.full_like(actual, forecast)
        
        # Calculate MAE
        mae = np.mean(np.abs(actual - forecasts))
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual - forecasts) / actual)) * 100 if np.all(actual != 0) else np.nan
        
        # Calculate accuracy (percent within ±20% of actual)
        accuracy = np.mean(np.abs((actual - forecasts) / actual) <= 0.2) * 100 if np.all(actual != 0) else np.nan
        
        # Store results
        product_metrics.append({
            'Product ID': product_id,
            'MAE': mae,
            'MAPE': mape,
            'Accuracy': accuracy
        })
        
        # Collect all errors for aggregate metrics
        for a, f in zip(actual, forecasts):
            if a != 0:  # Avoid division by zero
                forecast_errors.append(abs((a - f) / a))
    
    # Calculate overall metrics
    forecast_errors = np.array([e for e in forecast_errors if not np.isnan(e)])
    overall_accuracy = np.mean(forecast_errors <= 0.2) * 100
    overall_mape = np.mean(forecast_errors) * 100
    
    product_metrics_df = pd.DataFrame(product_metrics)
    
    return {
        'overall_accuracy': overall_accuracy,
        'overall_mape': overall_mape,
        'product_metrics': product_metrics_df
    }

# Calculate pricing optimization metrics
def calculate_pricing_metrics(pricing_data, results):
    """Calculate pricing optimization metrics"""
    
    # If we have the pricing strategy results
    if 'pricing_strategy' in results:
        pricing_data = results['pricing_strategy']
    
    # Calculate metrics based on the pricing strategy
    if 'Price_Strategy' in pricing_data.columns:
        strategy_counts = pricing_data['Price_Strategy'].value_counts()
        
        # Calculate potential revenue impact
        # For increase: assume 5% price increase with 80% retention
        # For decrease: assume 10% price decrease with 15% volume increase
        
        pricing_data['Revenue_Current'] = pricing_data['Price'] * pricing_data['Sales Volume']
        
        # Create new columns with copies of the data
        pricing_data['New_Price'] = pricing_data['Price'].copy()
        pricing_data['New_Volume'] = pricing_data['Sales Volume'].copy()
        
        # Process each row to avoid numpy array issues
        for idx, row in pricing_data.iterrows():
            if row['Price_Strategy'] == 'Increase':
                pricing_data.at[idx, 'New_Price'] = row['Price'] * 1.05
                pricing_data.at[idx, 'New_Volume'] = row['Sales Volume'] * 0.95
            elif row['Price_Strategy'] == 'Decrease':
                pricing_data.at[idx, 'New_Price'] = row['Price'] * 0.9
                pricing_data.at[idx, 'New_Volume'] = row['Sales Volume'] * 1.15
        
        # Calculate new revenue
        pricing_data['Revenue_New'] = pricing_data['New_Price'] * pricing_data['New_Volume']
        
        # Calculate revenue impact
        pricing_data['Revenue_Impact'] = pricing_data['Revenue_New'] - pricing_data['Revenue_Current']
        
        total_current_revenue = pricing_data['Revenue_Current'].sum()
        total_new_revenue = pricing_data['Revenue_New'].sum()
        revenue_improvement = (total_new_revenue - total_current_revenue) / total_current_revenue * 100
        
        # Calculate average impacts by strategy
        strategy_impact = pricing_data.groupby('Price_Strategy')['Revenue_Impact'].sum()
        
        return {
            'strategy_counts': strategy_counts,
            'revenue_improvement': revenue_improvement,
            'strategy_impact': strategy_impact,
            'pricing_data': pricing_data
        }
    
    # If we don't have pricing strategy results, calculate simple metrics
    else:
        # Calculate price position vs competitors
        pricing_data['Price_Position'] = pricing_data['Price'] / pricing_data['Competitor Prices']
        
        # Calculate average price position
        avg_price_position = pricing_data['Price_Position'].mean()
        
        # Calculate price vs elasticity correlation
        price_elasticity_corr = pricing_data[['Price', 'Elasticity Index']].corr().iloc[0, 1]
        
        return {
            'avg_price_position': avg_price_position,
            'price_elasticity_corr': price_elasticity_corr,
            'pricing_data': pricing_data
        }

# Calculate inventory optimization metrics
def calculate_inventory_optimization_metrics(inventory_data, results):
    """Calculate inventory optimization metrics"""
    
    if 'optimal_levels' in results and not results['optimal_levels'].empty:
        optimal_data = results['optimal_levels']
        
        # Calculate current vs optimal safety stock
        if 'Optimal Safety Stock' in optimal_data.columns and not optimal_data['Optimal Safety Stock'].isna().all():
            # Calculate the average change in safety stock
            optimal_data['Safety_Stock_Change'] = (
                (optimal_data['Optimal Safety Stock'] - optimal_data['Current Reorder Point']) / 
                optimal_data['Current Reorder Point'] * 100
            )
            
            # Calculate the average absolute change
            avg_safety_stock_change = optimal_data['Safety_Stock_Change'].abs().mean()
            
            # Calculate the percentage of products needing adjustment
            pct_needing_adjustment = (
                (optimal_data['Safety_Stock_Change'].abs() > 20).mean() * 100
            )
            
            # Calculate potential inventory cost savings
            # Assume carrying cost is 25% of inventory value per year
            # and average product value is $50
            product_value = 50
            carrying_cost_rate = 0.25
            
            # Calculate excess inventory (current - optimal)
            optimal_data['Excess_Inventory'] = (
                optimal_data['Current Stock'] - 
                optimal_data['Optimal Reorder Point']
            )
            
            # Only count positive excess (don't count understocked items as negative excess)
            optimal_data['Excess_Inventory'] = optimal_data['Excess_Inventory'].clip(lower=0)
            
            # Calculate excess inventory cost
            total_excess_inventory = optimal_data['Excess_Inventory'].sum()
            excess_inventory_cost = total_excess_inventory * product_value * carrying_cost_rate
            
            return {
                'avg_safety_stock_change': avg_safety_stock_change,
                'pct_needing_adjustment': pct_needing_adjustment,
                'excess_inventory_cost': excess_inventory_cost,
                'optimal_data': optimal_data
            }
    
    # If we don't have optimal level results, calculate basic metrics
    # Calculate average days of supply
    inventory_data['Days_of_Supply'] = np.nan
    
    # Create a map of average demand by product
    if 'Avg Demand' not in inventory_data.columns:
        demand_data = pd.read_csv("data/demand_forecasting.csv")
        product_demand = demand_data.groupby('Product ID')['Sales Quantity'].mean().to_dict()
        
        # Add average demand to inventory data
        inventory_data['Avg Demand'] = inventory_data['Product ID'].map(product_demand)
        
        # Fill NaN values with overall average
        overall_avg_demand = demand_data['Sales Quantity'].mean()
        inventory_data['Avg Demand'].fillna(overall_avg_demand, inplace=True)
    
    # Calculate days of supply
    mask = inventory_data['Avg Demand'] > 0
    inventory_data.loc[mask, 'Days_of_Supply'] = (
        inventory_data.loc[mask, 'Stock Levels'] / inventory_data.loc[mask, 'Avg Demand']
    )
    
    # Handle infinite values
    inventory_data['Days_of_Supply'] = inventory_data['Days_of_Supply'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate average days of supply
    avg_days_supply = inventory_data['Days_of_Supply'].mean()
    
    # Calculate potential excess inventory
    # Assume optimal days of supply is 30 days
    optimal_days = 30
    inventory_data['Excess_Days'] = inventory_data['Days_of_Supply'] - optimal_days
    inventory_data['Excess_Days'] = inventory_data['Excess_Days'].clip(lower=0)
    
    # Calculate excess inventory units
    inventory_data['Excess_Units'] = inventory_data['Excess_Days'] * inventory_data['Avg Demand']
    
    # Calculate excess inventory cost (assume $50 per unit and 25% carrying cost)
    unit_cost = 50
    carrying_cost_rate = 0.25
    inventory_data['Excess_Cost'] = inventory_data['Excess_Units'] * unit_cost * carrying_cost_rate
    
    total_excess_cost = inventory_data['Excess_Cost'].sum()
    
    return {
        'avg_days_supply': avg_days_supply,
        'total_excess_cost': total_excess_cost,
        'inventory_data': inventory_data
    }

# Generate performance dashboard
def generate_performance_dashboard(metrics):
    """Generate performance dashboard visualizations"""
    
    # Create KPI summary figure
    plt.figure(figsize=(15, 10))
    
    # Service Level KPI
    plt.subplot(2, 3, 1)
    service_level = metrics.get('service_level', {}).get('avg_service_level', 0)
    plt.text(0.5, 0.5, f"{service_level:.1f}%", fontsize=36, ha='center')
    plt.text(0.5, 0.3, "Service Level", fontsize=16, ha='center')
    plt.axis('off')
    
    # Inventory Turnover KPI
    plt.subplot(2, 3, 2)
    turnover = metrics.get('inventory_turnover', {}).get('avg_turnover', 0)
    plt.text(0.5, 0.5, f"{turnover:.1f}x", fontsize=36, ha='center')
    plt.text(0.5, 0.3, "Inventory Turnover", fontsize=16, ha='center')
    plt.axis('off')
    
    # Forecast Accuracy KPI
    plt.subplot(2, 3, 3)
    forecast_accuracy = metrics.get('forecast_performance', {}).get('overall_accuracy', 0)
    plt.text(0.5, 0.5, f"{forecast_accuracy:.1f}%", fontsize=36, ha='center')
    plt.text(0.5, 0.3, "Forecast Accuracy", fontsize=16, ha='center')
    plt.axis('off')
    
    # Revenue Improvement KPI
    plt.subplot(2, 3, 4)
    revenue_improvement = metrics.get('pricing_metrics', {}).get('revenue_improvement', 0)
    plt.text(0.5, 0.5, f"{revenue_improvement:.1f}%", fontsize=36, ha='center')
    plt.text(0.5, 0.3, "Revenue Improvement", fontsize=16, ha='center')
    plt.axis('off')
    
    # High Stockout Risk KPI
    plt.subplot(2, 3, 5)
    high_risk_pct = metrics.get('stockout_metrics', {}).get('high_risk_pct', 0)
    plt.text(0.5, 0.5, f"{high_risk_pct:.1f}%", fontsize=36, ha='center')
    plt.text(0.5, 0.3, "Products at High Risk", fontsize=16, ha='center')
    plt.axis('off')
    
    # Excess Inventory Cost KPI
    plt.subplot(2, 3, 6)
    excess_cost = metrics.get('inventory_optimization', {}).get('total_excess_cost', 0)
    plt.text(0.5, 0.5, f"${excess_cost/1000:.1f}K", fontsize=36, ha='center')
    plt.text(0.5, 0.3, "Annual Excess Inventory Cost", fontsize=16, ha='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{METRICS_DIR}/kpi_dashboard.png")
    
    # Create service level by store chart
    if 'service_level' in metrics and 'store_service_levels' in metrics['service_level']:
        plt.figure(figsize=(12, 6))
        store_data = metrics['service_level']['store_service_levels']
        
        # Sort by service level
        store_data = store_data.sort_values('Service Level')
        
        # Plot top and bottom 10 stores
        bottom_10 = store_data.head(10)
        top_10 = store_data.tail(10)
        
        plt.subplot(1, 2, 1)
        plt.barh(bottom_10['Store ID'].astype(str), bottom_10['Service Level'], color='coral')
        plt.title('10 Stores with Lowest Service Levels')
        plt.xlabel('Service Level (%)')
        plt.ylabel('Store ID')
        plt.xlim(0, 100)
        
        plt.subplot(1, 2, 2)
        plt.barh(top_10['Store ID'].astype(str), top_10['Service Level'], color='forestgreen')
        plt.title('10 Stores with Highest Service Levels')
        plt.xlabel('Service Level (%)')
        plt.ylabel('Store ID')
        plt.xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f"{METRICS_DIR}/service_level_by_store.png")
    
    # Create pricing strategy impact chart
    if 'pricing_metrics' in metrics and 'strategy_impact' in metrics['pricing_metrics']:
        plt.figure(figsize=(10, 6))
        
        strategy_impact = metrics['pricing_metrics']['strategy_impact']
        
        # Convert to dataframe if it's a series
        if isinstance(strategy_impact, pd.Series):
            # Convert Series to list of strategies and values
            strategies = strategy_impact.index.tolist()
            impact_values = strategy_impact.values.tolist()
            
            # Set colors based on strategy
            colors = []
            for strategy in strategies:
                if strategy == 'Decrease':
                    colors.append('red')
                elif strategy == 'Increase':
                    colors.append('green')
                else:
                    colors.append('gray')
            
            # Create the bar chart
            plt.bar(strategies, impact_values, color=colors)
            plt.title('Revenue Impact by Pricing Strategy')
            plt.xlabel('Pricing Strategy')
            plt.ylabel('Revenue Impact ($)')
            plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{METRICS_DIR}/pricing_strategy_impact.png")
    
    # Create forecast performance chart
    if 'forecast_performance' in metrics and 'product_metrics' in metrics['forecast_performance']:
        plt.figure(figsize=(10, 6))
        
        product_metrics = metrics['forecast_performance']['product_metrics']
        
        if not product_metrics.empty:
            # Sort by accuracy
            product_metrics = product_metrics.sort_values('Accuracy')
            
            plt.bar(product_metrics['Product ID'].astype(str), product_metrics['Accuracy'], color='skyblue')
            plt.title('Forecast Accuracy by Product')
            plt.xlabel('Product ID')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{METRICS_DIR}/forecast_accuracy.png")
    
    # Create stockout risk chart
    if 'stockout_metrics' in metrics and 'high_risk_products' in metrics['stockout_metrics']:
        plt.figure(figsize=(12, 6))
        
        high_risk = metrics['stockout_metrics']['high_risk_products']
        
        plt.bar(high_risk['Product ID'].astype(str), high_risk['Stockout Risk Score'], color='orangered')
        plt.title('Products with Highest Stockout Risk')
        plt.xlabel('Product ID')
        plt.ylabel('Stockout Risk Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{METRICS_DIR}/stockout_risk.png")

# Main function
def main():
    """Run the performance metrics analysis"""
    print("Generating Inventory Optimization Performance Metrics...")
    
    # Load data
    demand_data, inventory_data, pricing_data, results = load_data()
    
    # Calculate metrics
    metrics = {}
    
    # Service level metrics
    metrics['service_level'] = calculate_service_levels(inventory_data.copy())
    print("✅ Calculated service level metrics")
    
    # Inventory turnover metrics
    metrics['inventory_turnover'] = calculate_inventory_turnover(demand_data.copy(), inventory_data.copy())
    print("✅ Calculated inventory turnover metrics")
    
    # Stockout risk metrics
    metrics['stockout_metrics'] = calculate_stockout_metrics(inventory_data.copy())
    print("✅ Calculated stockout risk metrics")
    
    # Forecast performance metrics
    metrics['forecast_performance'] = simulate_forecast_performance(demand_data.copy())
    print("✅ Calculated forecast performance metrics")
    
    # Pricing optimization metrics
    metrics['pricing_metrics'] = calculate_pricing_metrics(pricing_data.copy(), results)
    print("✅ Calculated pricing optimization metrics")
    
    # Inventory optimization metrics
    metrics['inventory_optimization'] = calculate_inventory_optimization_metrics(inventory_data.copy(), results)
    print("✅ Calculated inventory optimization metrics")
    
    # Generate dashboard
    generate_performance_dashboard(metrics)
    print("✅ Generated performance dashboard")
    
    # Save metrics summary to file
    write_metrics_summary(metrics)
    print("✅ Saved metrics summary to file")
    
    print(f"\nPerformance metrics analysis complete. Results saved to '{METRICS_DIR}' directory.")

def write_metrics_summary(metrics):
    """Write metrics summary to text file"""
    with open(f"{METRICS_DIR}/performance_summary.txt", "w") as f:
        f.write("===== INVENTORY OPTIMIZATION PERFORMANCE METRICS =====\n\n")
        
        # Service Level Metrics
        f.write("===== SERVICE LEVEL METRICS =====\n")
        service_level = metrics.get('service_level', {})
        f.write(f"Average Service Level: {service_level.get('avg_service_level', 0):.2f}%\n")
        
        # Inventory Turnover Metrics
        f.write("\n===== INVENTORY TURNOVER METRICS =====\n")
        inventory_turnover = metrics.get('inventory_turnover', {})
        f.write(f"Average Inventory Turnover: {inventory_turnover.get('avg_turnover', 0):.2f}x\n")
        
        # Stockout Risk Metrics
        f.write("\n===== STOCKOUT RISK METRICS =====\n")
        stockout_metrics = metrics.get('stockout_metrics', {})
        f.write(f"Average Stockout Risk Score: {stockout_metrics.get('avg_risk', 0):.2f}\n")
        f.write(f"Percentage of Products at High Risk: {stockout_metrics.get('high_risk_pct', 0):.2f}%\n")
        
        # Top 5 high risk products
        high_risk = stockout_metrics.get('high_risk_products', pd.DataFrame())
        if not high_risk.empty:
            f.write("\nTop 5 High Risk Products:\n")
            for i, (_, row) in enumerate(high_risk.iloc[:5].iterrows()):
                f.write(f"{i+1}. Product {row['Product ID']} at Store {row['Store ID']}: Risk Score {row['Stockout Risk Score']:.2f}\n")
        
        # Forecast Performance Metrics
        f.write("\n===== FORECAST PERFORMANCE METRICS =====\n")
        forecast_performance = metrics.get('forecast_performance', {})
        f.write(f"Overall Forecast Accuracy: {forecast_performance.get('overall_accuracy', 0):.2f}%\n")
        f.write(f"Overall Mean Absolute Percentage Error: {forecast_performance.get('overall_mape', 0):.2f}%\n")
        
        # Pricing Optimization Metrics
        f.write("\n===== PRICING OPTIMIZATION METRICS =====\n")
        pricing_metrics = metrics.get('pricing_metrics', {})
        
        if 'revenue_improvement' in pricing_metrics:
            f.write(f"Potential Revenue Improvement: {pricing_metrics.get('revenue_improvement', 0):.2f}%\n")
            
            # Strategy counts
            strategy_counts = pricing_metrics.get('strategy_counts', {})
            if not strategy_counts.empty:
                f.write("\nPricing Strategy Distribution:\n")
                total = strategy_counts.sum()
                for strategy, count in strategy_counts.items():
                    f.write(f"{strategy}: {count} products ({count/total*100:.1f}%)\n")
        else:
            f.write(f"Average Price Position vs Competitors: {pricing_metrics.get('avg_price_position', 0):.2f}\n")
            f.write(f"Price-Elasticity Correlation: {pricing_metrics.get('price_elasticity_corr', 0):.2f}\n")
        
        # Inventory Optimization Metrics
        f.write("\n===== INVENTORY OPTIMIZATION METRICS =====\n")
        inventory_optimization = metrics.get('inventory_optimization', {})
        
        if 'avg_safety_stock_change' in inventory_optimization:
            f.write(f"Average Safety Stock Change: {inventory_optimization.get('avg_safety_stock_change', 0):.2f}%\n")
            f.write(f"Percentage of Products Needing Adjustment: {inventory_optimization.get('pct_needing_adjustment', 0):.2f}%\n")
            f.write(f"Potential Annual Savings from Excess Inventory: ${inventory_optimization.get('excess_inventory_cost', 0)/1000:.2f}K\n")
        else:
            f.write(f"Average Days of Supply: {inventory_optimization.get('avg_days_supply', 0):.2f} days\n")
            f.write(f"Potential Annual Savings from Excess Inventory: ${inventory_optimization.get('total_excess_cost', 0)/1000:.2f}K\n")
        
        # Overall Performance Score
        f.write("\n===== OVERALL PERFORMANCE SCORE =====\n")
        # Calculate a simple weighted score based on key metrics
        service_score = service_level.get('avg_service_level', 0) / 100
        turnover_score = min(inventory_turnover.get('avg_turnover', 0) / 10, 1)  # Cap at 1.0
        forecast_score = forecast_performance.get('overall_accuracy', 0) / 100
        risk_score = 1 - (stockout_metrics.get('high_risk_pct', 0) / 100)
        
        # Weight the scores
        weighted_score = (
            service_score * 0.3 +
            turnover_score * 0.2 +
            forecast_score * 0.3 +
            risk_score * 0.2
        ) * 100
        
        f.write(f"Overall Performance Score: {weighted_score:.1f}/100\n")
        
        # Performance rating
        if weighted_score >= 90:
            rating = "Excellent"
        elif weighted_score >= 80:
            rating = "Very Good"
        elif weighted_score >= 70:
            rating = "Good"
        elif weighted_score >= 60:
            rating = "Satisfactory"
        else:
            rating = "Needs Improvement"
            
        f.write(f"Performance Rating: {rating}\n")
        
        # Key improvement areas
        f.write("\n===== KEY IMPROVEMENT AREAS =====\n")
        areas = []
        
        if service_level.get('avg_service_level', 0) < 95:
            areas.append("Service levels below target of 95%")
            
        if stockout_metrics.get('high_risk_pct', 0) > 10:
            areas.append("High percentage of products at stockout risk")
            
        if forecast_performance.get('overall_accuracy', 0) < 80:
            areas.append("Forecast accuracy below target of 80%")
            
        if inventory_turnover.get('avg_turnover', 0) < 6:
            areas.append("Inventory turnover below industry benchmark of 6x")
            
        if inventory_optimization.get('total_excess_cost', 0) > 50000:
            areas.append("Significant excess inventory cost")
            
        if not areas:
            areas.append("No critical improvement areas identified")
            
        for i, area in enumerate(areas):
            f.write(f"{i+1}. {area}\n")

if __name__ == "__main__":
    main() 