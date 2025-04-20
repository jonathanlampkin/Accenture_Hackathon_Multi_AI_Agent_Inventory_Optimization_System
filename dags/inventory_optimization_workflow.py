"""
Inventory Optimization Workflow

This DAG orchestrates the complete inventory optimization process including:
1. Data loading and preprocessing
2. Demand forecasting
3. Inventory optimization
4. Order generation
5. Performance monitoring
"""

from datetime import datetime, timedelta
import os
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

import pandas as pd
import requests
import logging

import mlflow
import mlflow.sklearn
import mlflow.pytorch

# Default arguments for DAG
default_args = {
    'owner': 'inventory_system',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'inventory_optimization_workflow',
    default_args=default_args,
    description='End-to-end inventory optimization workflow',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['inventory', 'optimization', 'forecasting'],
)

# Helper functions for tasks
def load_data(**kwargs):
    """Load demand, inventory, and pricing data"""
    try:
        data_dir = os.environ.get('DATA_DIR', '/app/data')
        
        # Load demand data
        demand_data = pd.read_csv(f"{data_dir}/demand_data.csv")
        demand_data['Date'] = pd.to_datetime(demand_data['Date'])
        
        # Load inventory data if available
        try:
            inventory_data = pd.read_csv(f"{data_dir}/inventory_data.csv")
            kwargs['ti'].xcom_push(key='inventory_data_loaded', value=True)
        except FileNotFoundError:
            logging.warning("Inventory data file not found. Proceeding with only demand data.")
            kwargs['ti'].xcom_push(key='inventory_data_loaded', value=False)
        
        # Basic validation
        if demand_data.empty:
            raise ValueError("Demand data is empty")
            
        # Store data info
        data_info = {
            'demand_rows': len(demand_data),
            'demand_columns': demand_data.columns.tolist(),
            'start_date': demand_data['Date'].min().strftime('%Y-%m-%d'),
            'end_date': demand_data['Date'].max().strftime('%Y-%m-%d'),
            'product_count': demand_data['Product ID'].nunique()
        }
        
        # Pass data to next task via XCom
        kwargs['ti'].xcom_push(key='data_info', value=data_info)
        
        # Save processed data for subsequent tasks
        processed_dir = f"{data_dir}/processed"
        os.makedirs(processed_dir, exist_ok=True)
        demand_data.to_csv(f"{processed_dir}/processed_demand.csv", index=False)
        
        return f"Successfully loaded data with {data_info['demand_rows']} demand records"
    
    except Exception as e:
        logging.error(f"Error in load_data: {str(e)}")
        raise

def prepare_for_forecasting(**kwargs):
    """Prepare data for forecasting by transforming and feature engineering"""
    try:
        data_dir = os.environ.get('DATA_DIR', '/app/data')
        processed_dir = f"{data_dir}/processed"
        
        # Load processed demand data
        demand_data = pd.read_csv(f"{processed_dir}/processed_demand.csv")
        demand_data['Date'] = pd.to_datetime(demand_data['Date'])
        
        # Get unique products
        products = demand_data['Product ID'].unique()
        
        # Start MLflow run for tracking preprocessing
        mlflow.start_run(run_name="data_preparation")
        
        # Log data statistics
        mlflow.log_param("num_products", len(products))
        mlflow.log_param("date_range", f"{demand_data['Date'].min()} to {demand_data['Date'].max()}")
        
        # Create product-specific datasets
        for product_id in products:
            product_data = demand_data[demand_data['Product ID'] == product_id].copy()
            
            # Feature engineering (example)
            product_data['Day_of_Week'] = product_data['Date'].dt.dayofweek
            product_data['Month'] = product_data['Date'].dt.month
            product_data['Year'] = product_data['Date'].dt.year
            
            # Add lag features
            for lag in [1, 7, 14, 28]:
                product_data[f'Sales_Lag_{lag}'] = product_data['Sales Quantity'].shift(lag)
            
            # Add moving averages
            for window in [7, 14, 30]:
                product_data[f'Sales_MA_{window}'] = product_data['Sales Quantity'].rolling(window=window).mean()
            
            # Fill NA values from feature creation
            product_data = product_data.fillna(method='bfill').fillna(method='ffill')
            
            # Save product-specific data
            product_data.to_csv(f"{processed_dir}/product_{product_id}_prepared.csv", index=False)
            
            # Log feature statistics
            mlflow.log_metric(f"product_{product_id}_records", len(product_data))
        
        mlflow.end_run()
        
        # Pass product list to next task
        kwargs['ti'].xcom_push(key='products', value=products.tolist())
        
        return f"Successfully prepared data for {len(products)} products"
    
    except Exception as e:
        logging.error(f"Error in prepare_for_forecasting: {str(e)}")
        raise

def run_forecasting(**kwargs):
    """Generate forecasts for all products"""
    try:
        data_dir = os.environ.get('DATA_DIR', '/app/data')
        processed_dir = f"{data_dir}/processed"
        output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
        forecast_dir = f"{output_dir}/forecasts"
        os.makedirs(forecast_dir, exist_ok=True)
        
        # Get list of products from XCom
        ti = kwargs['ti']
        products = ti.xcom_pull(task_ids='prepare_for_forecasting', key='products')
        
        # Configure forecast parameters
        horizon = 30  # 30-day forecast
        confidence_level = 0.95
        
        all_forecasts = []
        all_metrics = []
        
        for product_id in products:
            # Load product data
            product_data = pd.read_csv(f"{processed_dir}/product_{product_id}_prepared.csv")
            product_data['Date'] = pd.to_datetime(product_data['Date'])
            
            # Start MLflow run for this product
            with mlflow.start_run(run_name=f"forecast_product_{product_id}"):
                # Split into train/test
                train_size = int(len(product_data) * 0.8)
                train_data = product_data.iloc[:train_size]
                test_data = product_data.iloc[train_size:]
                
                # Log training dataset info
                mlflow.log_param("product_id", product_id)
                mlflow.log_param("train_size", len(train_data))
                mlflow.log_param("test_size", len(test_data))
                
                # Get category and other product info
                if 'Category' in product_data.columns:
                    category = product_data['Category'].iloc[0]
                    mlflow.log_param("category", category)
                
                # Choose model based on product characteristics
                # This would call the forecasting service in a real deployment
                
                # For demo purposes, assume we get forecasts from the service
                forecast_result = {
                    'product_id': product_id,
                    'forecast_dates': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, horizon+1)],
                    'forecast_values': [float(100 + i*2) for i in range(horizon)],  # Dummy values
                    'upper_ci': [float(120 + i*2) for i in range(horizon)],
                    'lower_ci': [float(80 + i*2) for i in range(horizon)],
                    'method': 'ensemble',
                    'metrics': {
                        'rmse': 25.5,
                        'mae': 18.2,
                        'r2': 0.82
                    }
                }
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'rmse': forecast_result['metrics']['rmse'],
                    'mae': forecast_result['metrics']['mae'],
                    'r2': forecast_result['metrics']['r2']
                })
                
                # Save individual forecast
                with open(f"{forecast_dir}/forecast_{product_id}.json", 'w') as f:
                    json.dump(forecast_result, f, indent=2)
                
                # Append to combined results
                all_forecasts.append(forecast_result)
                all_metrics.append({
                    'product_id': product_id,
                    'rmse': forecast_result['metrics']['rmse'],
                    'mae': forecast_result['metrics']['mae'],
                    'r2': forecast_result['metrics']['r2']
                })
        
        # Save combined forecasts
        with open(f"{forecast_dir}/all_forecasts.json", 'w') as f:
            json.dump(all_forecasts, f, indent=2)
            
        # Create metrics DataFrame and save
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"{forecast_dir}/forecast_metrics.csv", index=False)
        
        # Pass forecast info to next tasks
        kwargs['ti'].xcom_push(key='forecast_metrics', value=metrics_df.to_dict(orient='records'))
        
        return f"Generated forecasts for {len(products)} products"
    
    except Exception as e:
        logging.error(f"Error in run_forecasting: {str(e)}")
        raise

def optimize_inventory_levels(**kwargs):
    """Optimize inventory levels based on forecasts"""
    try:
        output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
        forecast_dir = f"{output_dir}/forecasts"
        inventory_dir = f"{output_dir}/inventory"
        os.makedirs(inventory_dir, exist_ok=True)
        
        # Load forecasts
        with open(f"{forecast_dir}/all_forecasts.json", 'r') as f:
            forecasts = json.load(f)
        
        # Configure optimization parameters
        service_level = 0.95  # 95% service level
        lead_times = {  # Sample lead times in days
            101: 3,
            102: 5,
            103: 2,
            104: 4,
            105: 7
        }
        
        holding_cost_rate = 0.25  # 25% annual holding cost
        
        inventory_recommendations = []
        
        for forecast in forecasts:
            product_id = forecast['product_id']
            
            # Get mean forecast and confidence intervals
            mean_forecast = forecast['forecast_values']
            upper_ci = forecast['upper_ci']
            lower_ci = forecast['lower_ci']
            
            # Get lead time (default to 5 if not specified)
            lead_time = lead_times.get(product_id, 5)
            
            # Calculate safety stock based on forecast uncertainty
            forecast_std = [(u - l) / 3.92 for u, l in zip(upper_ci, lower_ci)]  # Approximate std dev from 95% CI
            avg_std = sum(forecast_std) / len(forecast_std)
            safety_stock = 1.645 * avg_std * (lead_time ** 0.5)  # Based on normal distribution for 95% SL
            
            # Calculate reorder point
            avg_daily_demand = sum(mean_forecast) / len(mean_forecast)
            reorder_point = avg_daily_demand * lead_time + safety_stock
            
            # Calculate economic order quantity (simplified)
            annual_demand = avg_daily_demand * 365
            order_cost = 100  # Placeholder fixed order cost
            unit_cost = 10  # Placeholder unit cost
            eoq = (2 * annual_demand * order_cost / (holding_cost_rate * unit_cost)) ** 0.5
            
            recommendation = {
                'product_id': product_id,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'economic_order_quantity': eoq,
                'max_inventory_level': reorder_point + eoq,
                'lead_time': lead_time,
                'service_level': service_level
            }
            
            inventory_recommendations.append(recommendation)
        
        # Save recommendations
        with open(f"{inventory_dir}/inventory_recommendations.json", 'w') as f:
            json.dump(inventory_recommendations, f, indent=2)
        
        # Pass recommendations to next tasks
        kwargs['ti'].xcom_push(key='inventory_recommendations', value=inventory_recommendations)
        
        return f"Generated inventory recommendations for {len(inventory_recommendations)} products"
    
    except Exception as e:
        logging.error(f"Error in optimize_inventory_levels: {str(e)}")
        raise

def generate_purchase_orders(**kwargs):
    """Generate purchase orders based on inventory recommendations"""
    try:
        output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
        inventory_dir = f"{output_dir}/inventory"
        orders_dir = f"{output_dir}/orders"
        os.makedirs(orders_dir, exist_ok=True)
        
        # Get inventory recommendations from XCom
        ti = kwargs['ti']
        inventory_recommendations = ti.xcom_pull(task_ids='optimize_inventory_levels', key='inventory_recommendations')
        
        # Sample current inventory levels (in a real system, this would come from a database)
        current_inventory = {
            101: 120,
            102: 80,
            103: 200,
            104: 150,
            105: 90
        }
        
        purchase_orders = []
        
        for recommendation in inventory_recommendations:
            product_id = recommendation['product_id']
            
            # Get current inventory level (default to 0 if not specified)
            current_level = current_inventory.get(int(product_id), 0)
            
            # Check if below reorder point
            if current_level < recommendation['reorder_point']:
                # Calculate order quantity
                order_quantity = recommendation['economic_order_quantity']
                
                # Adjust to not exceed max inventory level
                max_level = recommendation['max_inventory_level']
                if current_level + order_quantity > max_level:
                    order_quantity = max_level - current_level
                
                # Round to whole number
                order_quantity = max(0, round(order_quantity))
                
                # Only create order if quantity > 0
                if order_quantity > 0:
                    order = {
                        'order_id': f"PO-{datetime.now().strftime('%Y%m%d')}-{product_id}",
                        'product_id': product_id,
                        'order_quantity': order_quantity,
                        'current_inventory': current_level,
                        'reorder_point': recommendation['reorder_point'],
                        'expected_lead_time': recommendation['lead_time'],
                        'order_date': datetime.now().strftime('%Y-%m-%d'),
                        'expected_delivery': (datetime.now() + timedelta(days=recommendation['lead_time'])).strftime('%Y-%m-%d')
                    }
                    
                    purchase_orders.append(order)
        
        # Save purchase orders
        with open(f"{orders_dir}/purchase_orders.json", 'w') as f:
            json.dump(purchase_orders, f, indent=2)
        
        return f"Generated {len(purchase_orders)} purchase orders"
    
    except Exception as e:
        logging.error(f"Error in generate_purchase_orders: {str(e)}")
        raise

def generate_performance_report(**kwargs):
    """Generate performance report for the optimization process"""
    try:
        output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
        forecast_dir = f"{output_dir}/forecasts"
        inventory_dir = f"{output_dir}/inventory"
        orders_dir = f"{output_dir}/orders"
        reports_dir = f"{output_dir}/reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Get forecast metrics from XCom
        ti = kwargs['ti']
        forecast_metrics = ti.xcom_pull(task_ids='run_forecasting', key='forecast_metrics')
        
        # Load inventory recommendations
        with open(f"{inventory_dir}/inventory_recommendations.json", 'r') as f:
            inventory_recommendations = json.load(f)
        
        # Load purchase orders if they exist
        try:
            with open(f"{orders_dir}/purchase_orders.json", 'r') as f:
                purchase_orders = json.load(f)
        except FileNotFoundError:
            purchase_orders = []
        
        # Generate summary report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'products_analyzed': len(forecast_metrics),
                'products_with_orders': len(purchase_orders),
                'average_forecast_rmse': sum(m['rmse'] for m in forecast_metrics) / len(forecast_metrics) if forecast_metrics else 0,
                'average_safety_stock': sum(r['safety_stock'] for r in inventory_recommendations) / len(inventory_recommendations) if inventory_recommendations else 0,
                'total_ordered_quantity': sum(o['order_quantity'] for o in purchase_orders) if purchase_orders else 0
            },
            'forecast_performance': forecast_metrics,
            'order_summary': [{'product_id': o['product_id'], 'quantity': o['order_quantity']} for o in purchase_orders]
        }
        
        # Save report
        with open(f"{reports_dir}/optimization_report_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return "Generated performance report"
    
    except Exception as e:
        logging.error(f"Error in generate_performance_report: {str(e)}")
        raise

# DAG tasks
start = EmptyOperator(
    task_id='start',
    dag=dag,
)

with TaskGroup(group_id='data_preparation') as data_preparation:
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        dag=dag,
    )
    
    prepare_forecasting_task = PythonOperator(
        task_id='prepare_for_forecasting',
        python_callable=prepare_for_forecasting,
        dag=dag,
    )
    
    load_data_task >> prepare_forecasting_task

with TaskGroup(group_id='forecasting') as forecasting:
    run_forecasting_task = PythonOperator(
        task_id='run_forecasting',
        python_callable=run_forecasting,
        dag=dag,
    )

with TaskGroup(group_id='inventory_optimization') as inventory_optimization:
    optimize_inventory_task = PythonOperator(
        task_id='optimize_inventory_levels',
        python_callable=optimize_inventory_levels,
        dag=dag,
    )
    
    generate_orders_task = PythonOperator(
        task_id='generate_purchase_orders',
        python_callable=generate_purchase_orders,
        dag=dag,
    )
    
    optimize_inventory_task >> generate_orders_task

generate_report_task = PythonOperator(
    task_id='generate_performance_report',
    python_callable=generate_performance_report,
    dag=dag,
)

end = EmptyOperator(
    task_id='end',
    dag=dag,
)

# Define task dependencies
start >> data_preparation >> forecasting >> inventory_optimization >> generate_report_task >> end 