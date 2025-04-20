#!/usr/bin/env python3
"""
Inventory Optimization Analysis Script
Analyzes demand, inventory, and pricing data to generate optimization insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for reports and visualizations
RESULTS_DIR = "optimization_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

class InventoryOptimizer:
    """Analyzes inventory data and generates optimization insights"""
    
    def __init__(self):
        self.demand_data = None
        self.inventory_data = None
        self.pricing_data = None
        
    def load_data(self, demand_path, inventory_path, pricing_path):
        """Load the datasets"""
        logger.info("Loading datasets...")
        self.demand_data = pd.read_csv(demand_path)
        self.inventory_data = pd.read_csv(inventory_path)
        self.pricing_data = pd.read_csv(pricing_path)
        
        logger.info(f"Loaded demand data: {self.demand_data.shape[0]} rows")
        logger.info(f"Loaded inventory data: {self.inventory_data.shape[0]} rows")
        logger.info(f"Loaded pricing data: {self.pricing_data.shape[0]} rows")
        
        return self
    
    def print_data_summary(self):
        """Print summary statistics of the datasets"""
        logger.info("Generating data summary...")
        
        # Demand data summary
        print("\n===== DEMAND FORECASTING DATA SUMMARY =====")
        print(f"Total records: {self.demand_data.shape[0]}")
        print(f"Unique products: {self.demand_data['Product ID'].nunique()}")
        print(f"Unique stores: {self.demand_data['Store ID'].nunique()}")
        print("\nSales quantity statistics:")
        print(self.demand_data['Sales Quantity'].describe())
        
        # Inventory data summary
        print("\n===== INVENTORY MONITORING DATA SUMMARY =====")
        print(f"Total records: {self.inventory_data.shape[0]}")
        print(f"Unique products: {self.inventory_data['Product ID'].nunique()}")
        print(f"Unique stores: {self.inventory_data['Store ID'].nunique()}")
        print("\nStock levels statistics:")
        print(self.inventory_data['Stock Levels'].describe())
        print("\nLead time statistics:")
        print(self.inventory_data['Supplier Lead Time (days)'].describe())
        
        # Pricing data summary
        print("\n===== PRICING OPTIMIZATION DATA SUMMARY =====")
        print(f"Total records: {self.pricing_data.shape[0]}")
        print(f"Unique products: {self.pricing_data['Product ID'].nunique()}")
        print(f"Unique stores: {self.pricing_data['Store ID'].nunique()}")
        print("\nPrice statistics:")
        print(self.pricing_data['Price'].describe())
        print("\nElasticity statistics:")
        print(self.pricing_data['Elasticity Index'].describe())
    
    def analyze_top_products(self, top_n=10):
        """Identify top products by sales quantity"""
        logger.info(f"Identifying top {top_n} products by sales...")
        
        # Group by Product ID and calculate total sales
        product_sales = self.demand_data.groupby('Product ID')['Sales Quantity'].sum().reset_index()
        product_sales = product_sales.sort_values('Sales Quantity', ascending=False).head(top_n)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(product_sales['Product ID'].astype(str), product_sales['Sales Quantity'])
        plt.title(f'Top {top_n} Products by Sales Quantity')
        plt.xlabel('Product ID')
        plt.ylabel('Total Sales Quantity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/top_products.png")
        
        print(f"\n===== TOP {top_n} PRODUCTS BY SALES =====")
        print(product_sales)
        
        return product_sales
    
    def analyze_stockout_risk(self):
        """Identify products with high stockout risk"""
        logger.info("Analyzing stockout risk...")
        
        # Calculate stockout risk score based on stockout frequency and inventory levels
        self.inventory_data['Stockout Risk Score'] = (
            self.inventory_data['Stockout Frequency'] / 
            (self.inventory_data['Stock Levels'] / self.inventory_data['Reorder Point'])
        )
        
        # Handle infinite values
        self.inventory_data['Stockout Risk Score'] = self.inventory_data['Stockout Risk Score'].replace([np.inf, -np.inf], np.nan)
        self.inventory_data['Stockout Risk Score'] = self.inventory_data['Stockout Risk Score'].fillna(0)
        
        # Sort by risk score
        high_risk_products = self.inventory_data.sort_values('Stockout Risk Score', ascending=False).head(10)
        
        print("\n===== PRODUCTS WITH HIGH STOCKOUT RISK =====")
        print(high_risk_products[['Product ID', 'Store ID', 'Stock Levels', 'Stockout Frequency', 'Reorder Point', 'Stockout Risk Score']])
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(high_risk_products['Product ID'].astype(str), high_risk_products['Stockout Risk Score'])
        plt.title('Products with High Stockout Risk')
        plt.xlabel('Product ID')
        plt.ylabel('Stockout Risk Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/stockout_risk.png")
        
        return high_risk_products
    
    def calculate_optimal_stock_levels(self):
        """Calculate optimal min/max stock levels and reorder points"""
        logger.info("Calculating optimal stock levels...")
        
        # Merge inventory and demand data
        merged_data = self.inventory_data.merge(
            self.demand_data, 
            on=['Product ID', 'Store ID'],
            how='inner',
            suffixes=('_inv', '_dem')
        )
        
        if merged_data.empty:
            logger.warning("No matching products/stores between inventory and demand data")
            return None
        
        # Calculate optimal levels
        results = []
        for (product_id, store_id), group in merged_data.groupby(['Product ID', 'Store ID']):
            # Extract key metrics
            lead_time = group['Supplier Lead Time (days)'].mean()
            avg_sales = group['Sales Quantity'].mean()
            std_sales = group['Sales Quantity'].std()
            current_reorder = group['Reorder Point'].mean()
            current_stock = group['Stock Levels'].mean()
            
            # Calculate optimal levels (simplified calculations)
            # For a more comprehensive calculation, we would need more historical data
            service_factor = 1.645  # ~95% service level
            lead_time_demand = avg_sales * lead_time
            safety_stock = service_factor * std_sales * np.sqrt(lead_time)
            optimal_reorder_point = lead_time_demand + safety_stock
            
            # Calculate min and max levels
            min_stock = safety_stock
            max_stock = lead_time_demand * 2 + safety_stock  # Simplified EOQ calculation
            
            # Compare with current levels
            reorder_point_diff = ((optimal_reorder_point - current_reorder) / current_reorder * 100) if current_reorder > 0 else 0
            
            results.append({
                'Product ID': product_id,
                'Store ID': store_id,
                'Current Stock': current_stock,
                'Current Reorder Point': current_reorder,
                'Optimal Safety Stock': safety_stock,
                'Optimal Reorder Point': optimal_reorder_point,
                'Min Stock Level': min_stock,
                'Max Stock Level': max_stock,
                'Reorder Point Change (%)': reorder_point_diff
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n===== OPTIMAL STOCK LEVEL CALCULATIONS =====")
        print(results_df.head(10))
        
        # Save full results
        results_df.to_csv(f"{RESULTS_DIR}/optimal_stock_levels.csv", index=False)
        
        # Create visualization for reorder point comparison
        plt.figure(figsize=(12, 6))
        sample_df = results_df.head(10)
        
        x = range(len(sample_df))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], sample_df['Current Reorder Point'], width, label='Current Reorder Point')
        plt.bar([i + width/2 for i in x], sample_df['Optimal Reorder Point'], width, label='Optimal Reorder Point')
        
        plt.xlabel('Product-Store Combination')
        plt.ylabel('Reorder Point')
        plt.title('Current vs. Optimal Reorder Points')
        plt.xticks(x, [f"{p}-{s}" for p, s in zip(sample_df['Product ID'], sample_df['Store ID'])], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/reorder_point_comparison.png")
        
        return results_df
    
    def forecast_demand(self, forecast_horizon=30, sample_products=5):
        """Generate demand forecasts for sample products"""
        logger.info(f"Generating demand forecasts for {sample_products} products...")
        
        # Get products with most data points
        product_counts = self.demand_data['Product ID'].value_counts().head(sample_products)
        top_products = product_counts.index.tolist()
        
        forecast_results = {}
        
        for product_id in top_products:
            product_data = self.demand_data[self.demand_data['Product ID'] == product_id].copy()
            
            if len(product_data) < 10:  # Skip if not enough data
                continue
                
            # Feature engineering (basic)
            product_data['Promotion_Flag'] = (product_data['Promotions'] == 'Yes').astype(int)
            product_data['Seasonality_Flag'] = (product_data['Seasonality Factors'] != 'None').astype(int)
            product_data['External_Flag'] = (product_data['External Factors'] != 'None').astype(int)
            
            # Prepare features and target
            features = ['Price', 'Promotion_Flag', 'Seasonality_Flag', 'External_Flag']
            X = product_data[features]
            y = product_data['Sales Quantity']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_model_name = None
            best_score = float('inf')
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_model_name = name
            
            # Generate forecast based on latest features
            latest_features = X.iloc[-1:].copy()
            forecast = []
            
            for _ in range(forecast_horizon):
                # We're keeping features constant for simplicity
                # In a real system, you might vary price, promotions, etc.
                pred = best_model.predict(latest_features)[0]
                forecast.append(max(0, pred))  # Ensure non-negative
            
            # Store forecast results
            forecast_results[product_id] = {
                'model': best_model_name,
                'forecast': forecast,
                'metrics': {
                    'MSE': best_score,
                    'MAE': mean_absolute_error(y_test, best_model.predict(X_test)),
                    'R2': r2_score(y_test, best_model.predict(X_test))
                }
            }
            
            # Plot actual vs forecast
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.values, label='Actual')
            plt.plot(best_model.predict(X_test), label='Predicted')
            plt.title(f'Demand Forecast for Product {product_id} using {best_model_name}')
            plt.xlabel('Test Sample')
            plt.ylabel('Sales Quantity')
            plt.legend()
            plt.savefig(f"{RESULTS_DIR}/forecast_product_{product_id}.png")
        
        # Print forecast summary
        print("\n===== DEMAND FORECAST SUMMARY =====")
        for product_id, result in forecast_results.items():
            print(f"\nProduct {product_id} - Model: {result['model']}")
            print(f"Metrics: MSE={result['metrics']['MSE']:.2f}, MAE={result['metrics']['MAE']:.2f}, R2={result['metrics']['R2']:.2f}")
            print(f"30-day forecast average: {np.mean(result['forecast']):.2f}")
            print(f"30-day forecast min: {np.min(result['forecast']):.2f}")
            print(f"30-day forecast max: {np.max(result['forecast']):.2f}")
        
        # Save forecast data
        forecast_df = pd.DataFrame({
            'Product ID': [pid for pid in forecast_results.keys() for _ in range(forecast_horizon)],
            'Day': [i+1 for pid in forecast_results.keys() for i in range(forecast_horizon)],
            'Forecast': [val for pid in forecast_results.keys() for val in forecast_results[pid]['forecast']]
        })
        forecast_df.to_csv(f"{RESULTS_DIR}/demand_forecasts.csv", index=False)
        
        return forecast_results
    
    def analyze_pricing_strategy(self):
        """Analyze pricing strategies based on elasticity and competition"""
        logger.info("Analyzing pricing strategies...")
        
        # Calculate price-to-competitor ratio
        self.pricing_data['Price_Ratio'] = self.pricing_data['Price'] / self.pricing_data['Competitor Prices']
        
        # Categorize products
        self.pricing_data['Price_Strategy'] = 'Maintain'
        
        # Price sensitive products (high elasticity, price higher than competitors)
        self.pricing_data.loc[(self.pricing_data['Elasticity Index'] > 1.5) & 
                            (self.pricing_data['Price_Ratio'] > 1.05), 'Price_Strategy'] = 'Decrease'
        
        # Premium products (low elasticity, good reviews, low return rate)
        self.pricing_data.loc[(self.pricing_data['Elasticity Index'] < 1.0) & 
                            (self.pricing_data['Customer Reviews'] >= 4) &
                            (self.pricing_data['Return Rate (%)'] < 10), 'Price_Strategy'] = 'Increase'
        
        # Competitive advantage (price lower than competitors, good sales)
        self.pricing_data.loc[(self.pricing_data['Price_Ratio'] < 0.9) & 
                            (self.pricing_data['Sales Volume'] > self.pricing_data['Sales Volume'].median()), 
                            'Price_Strategy'] = 'Increase'
        
        # Summarize pricing strategies
        strategy_counts = self.pricing_data['Price_Strategy'].value_counts()
        
        print("\n===== PRICING STRATEGY ANALYSIS =====")
        print(f"Decrease Price: {strategy_counts.get('Decrease', 0)} products")
        print(f"Increase Price: {strategy_counts.get('Increase', 0)} products")
        print(f"Maintain Price: {strategy_counts.get('Maintain', 0)} products")
        
        # Sample of products for each strategy
        for strategy in ['Decrease', 'Increase', 'Maintain']:
            sample = self.pricing_data[self.pricing_data['Price_Strategy'] == strategy].head(5)
            if not sample.empty:
                print(f"\nSample products for {strategy} price strategy:")
                print(sample[['Product ID', 'Store ID', 'Price', 'Competitor Prices', 
                            'Elasticity Index', 'Customer Reviews', 'Sales Volume']])
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.pie(strategy_counts, labels=strategy_counts.index, autopct='%1.1f%%')
        plt.title('Pricing Strategy Distribution')
        plt.savefig(f"{RESULTS_DIR}/pricing_strategy.png")
        
        # Save pricing strategy data
        self.pricing_data[['Product ID', 'Store ID', 'Price', 'Competitor Prices', 
                         'Elasticity Index', 'Sales Volume', 'Price_Ratio', 
                         'Price_Strategy']].to_csv(f"{RESULTS_DIR}/pricing_strategy.csv", index=False)
        
        return self.pricing_data
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        logger.info("Generating comprehensive optimization report...")
        
        # Summary of key metrics
        report_file = f"{RESULTS_DIR}/optimization_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("===== INVENTORY OPTIMIZATION SYSTEM REPORT =====\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("===== DATA SUMMARY =====\n")
            f.write(f"Total products analyzed: {self.demand_data['Product ID'].nunique()}\n")
            f.write(f"Total stores analyzed: {self.demand_data['Store ID'].nunique()}\n")
            f.write(f"Total demand records: {self.demand_data.shape[0]}\n")
            f.write(f"Total inventory records: {self.inventory_data.shape[0]}\n")
            f.write(f"Total pricing records: {self.pricing_data.shape[0]}\n\n")
            
            # Key findings
            f.write("===== KEY FINDINGS =====\n")
            
            # Top products
            top_products = self.demand_data.groupby('Product ID')['Sales Quantity'].sum().nlargest(5)
            f.write("\nTop 5 Products by Sales:\n")
            for product, sales in top_products.items():
                f.write(f"Product {product}: {sales} units\n")
            
            # Stockout risk
            if 'Stockout Risk Score' in self.inventory_data.columns:
                high_risk = self.inventory_data.nlargest(5, 'Stockout Risk Score')
                f.write("\nTop 5 Products with Highest Stockout Risk:\n")
                for _, row in high_risk.iterrows():
                    f.write(f"Product {row['Product ID']} at Store {row['Store ID']}: Risk Score {row['Stockout Risk Score']:.2f}\n")
            
            # Pricing opportunities
            if 'Price_Strategy' in self.pricing_data.columns:
                increase_candidates = self.pricing_data[self.pricing_data['Price_Strategy'] == 'Increase'].nlargest(5, 'Sales Volume')
                f.write("\nTop 5 Products for Price Increase:\n")
                for _, row in increase_candidates.iterrows():
                    f.write(f"Product {row['Product ID']} at Store {row['Store ID']}: Current ${row['Price']:.2f}, Competitor ${row['Competitor Prices']:.2f}\n")
            
            # Reorder point recommendations
            if os.path.exists(f"{RESULTS_DIR}/optimal_stock_levels.csv"):
                reorder_recs = pd.read_csv(f"{RESULTS_DIR}/optimal_stock_levels.csv")
                significant_changes = reorder_recs[abs(reorder_recs['Reorder Point Change (%)']) > 20].nlargest(5, 'Current Stock')
                
                if not significant_changes.empty:
                    f.write("\nTop Reorder Point Adjustment Recommendations:\n")
                    for _, row in significant_changes.iterrows():
                        f.write(f"Product {row['Product ID']} at Store {row['Store ID']}: Change from {row['Current Reorder Point']:.0f} to {row['Optimal Reorder Point']:.0f} units ({row['Reorder Point Change (%)']:.1f}%)\n")
            
            # Overall recommendations
            f.write("\n===== OVERALL RECOMMENDATIONS =====\n")
            f.write("1. Implement the suggested reorder point adjustments to reduce stockout risk\n")
            f.write("2. Apply pricing strategy recommendations to optimize revenue\n")
            f.write("3. Focus inventory management attention on high-risk products\n")
            f.write("4. Utilize demand forecasts for the next 30 days to plan purchasing\n")
            f.write("5. Monitor pricing elasticity regularly to adjust pricing strategies\n")
        
        print(f"\nOptimization report generated: {report_file}")
        print(f"All results saved to directory: {RESULTS_DIR}")
        
        return report_file


def main():
    """Main function to run the optimization analysis"""
    # File paths
    demand_path = "data/demand_forecasting.csv"
    inventory_path = "data/inventory_monitoring.csv"
    pricing_path = "data/pricing_optimization.csv"
    
    # Create optimizer
    optimizer = InventoryOptimizer()
    
    # Load data
    optimizer.load_data(demand_path, inventory_path, pricing_path)
    
    # Print summary statistics
    optimizer.print_data_summary()
    
    # Run analyses
    optimizer.analyze_top_products()
    optimizer.analyze_stockout_risk()
    optimizer.calculate_optimal_stock_levels()
    optimizer.forecast_demand()
    optimizer.analyze_pricing_strategy()
    
    # Generate final report
    optimizer.generate_optimization_report()
    
    print("\nAnalysis complete. Check the optimization_results directory for detailed results.")


if __name__ == "__main__":
    main() 