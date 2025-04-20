"""
Reporting module for the Inventory Optimization API.

This module provides reporting functionality for the API.
"""
import logging
import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Report generator for inventory optimization."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ReportGenerator")
        self.output_dir = os.path.join("api", "results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_inventory_status_report(self, data_paths: Dict[str, str], output_format: str = "html") -> Dict[str, Any]:
        """
        Generate inventory status report.
        
        Args:
            data_paths: Dictionary with data paths, must include 'inventory' and 'demand'
            output_format: Output format, either 'html', 'json', or 'csv'
            
        Returns:
            Dict with report results
        """
        try:
            self.logger.info("Generating inventory status report")
            
            # Check required paths
            required_paths = ['inventory', 'demand']
            for path in required_paths:
                if path not in data_paths:
                    self.logger.error(f"Missing required data path: {path}")
                    raise ValueError(f"Missing required data path: {path}")
            
            # Load data
            inventory_data = self._load_data(data_paths['inventory'])
            demand_data = self._load_data(data_paths['demand'])
            
            # Generate report
            report = self._generate_inventory_status(inventory_data, demand_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inventory_status_report_{timestamp}.{output_format}"
            output_path = os.path.join(self.output_dir, filename)
            
            if output_format == "html":
                self._save_html_report(report, output_path)
            elif output_format == "json":
                self._save_json_report(report, output_path)
            elif output_format == "csv":
                self._save_csv_report(report, output_path)
            else:
                self.logger.error(f"Unsupported output format: {output_format}")
                raise ValueError(f"Unsupported output format: {output_format}")
            
            return {
                'report_path': filename,
                'report_type': 'inventory_status',
                'timestamp': timestamp,
                'summary': report['summary']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating inventory status report: {str(e)}")
            raise
    
    def generate_forecast_report(self, data_paths: Dict[str, str], output_format: str = "html") -> Dict[str, Any]:
        """
        Generate forecast report.
        
        Args:
            data_paths: Dictionary with data paths, must include 'forecast' and 'demand'
            output_format: Output format, either 'html', 'json', or 'csv'
            
        Returns:
            Dict with report results
        """
        try:
            self.logger.info("Generating forecast report")
            
            # Check required paths
            required_paths = ['forecast', 'demand']
            for path in required_paths:
                if path not in data_paths:
                    self.logger.error(f"Missing required data path: {path}")
                    raise ValueError(f"Missing required data path: {path}")
            
            # Load data
            forecast_data = self._load_data(data_paths['forecast'])
            demand_data = self._load_data(data_paths['demand'])
            
            # Generate report
            report = self._generate_forecast_report(forecast_data, demand_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_report_{timestamp}.{output_format}"
            output_path = os.path.join(self.output_dir, filename)
            
            if output_format == "html":
                self._save_html_report(report, output_path)
            elif output_format == "json":
                self._save_json_report(report, output_path)
            elif output_format == "csv":
                self._save_csv_report(report, output_path)
            else:
                self.logger.error(f"Unsupported output format: {output_format}")
                raise ValueError(f"Unsupported output format: {output_format}")
            
            return {
                'report_path': filename,
                'report_type': 'forecast',
                'timestamp': timestamp,
                'summary': report['summary']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating forecast report: {str(e)}")
            raise
    
    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            path: Path to data file
            
        Returns:
            DataFrame with data
        """
        file_path = os.path.join("api", "uploads", path)
        
        if not os.path.exists(file_path):
            self.logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            return pd.read_excel(file_path)
        elif path.endswith(".json"):
            return pd.read_json(file_path)
        else:
            self.logger.error(f"Unsupported file format: {path}")
            raise ValueError(f"Unsupported file format: {path}")
    
    def _generate_inventory_status(self, inventory_data: pd.DataFrame, demand_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate inventory status report.
        
        Args:
            inventory_data: DataFrame with inventory data
            demand_data: DataFrame with demand data
            
        Returns:
            Dict with report data
        """
        # Mock report data
        products = inventory_data['Product ID'].unique() if 'Product ID' in inventory_data.columns else []
        
        # Create mock inventory status
        inventory_status = []
        for product_id in products:
            # Get product inventory
            product_inventory = inventory_data[inventory_data['Product ID'] == product_id]
            
            # Get product demand
            product_demand = demand_data[demand_data['Product ID'] == product_id]
            
            # Calculate inventory metrics
            current_stock = product_inventory['Quantity'].sum() if 'Quantity' in product_inventory.columns else np.random.randint(50, 500)
            avg_daily_demand = product_demand['Sales Quantity'].mean() if 'Sales Quantity' in product_demand.columns else np.random.randint(5, 50)
            days_of_supply = current_stock / avg_daily_demand if avg_daily_demand > 0 else 0
            reorder_point = avg_daily_demand * 7  # 7 days of supply
            safety_stock = avg_daily_demand * 3   # 3 days of safety stock
            
            # Add to inventory status
            inventory_status.append({
                'Product ID': product_id,
                'Current Stock': current_stock,
                'Average Daily Demand': avg_daily_demand,
                'Days of Supply': days_of_supply,
                'Reorder Point': reorder_point,
                'Safety Stock': safety_stock,
                'Status': 'OK' if current_stock > reorder_point else 'Reorder' if current_stock > safety_stock else 'Low'
            })
        
        # Create summary
        total_products = len(inventory_status)
        ok_count = sum(1 for item in inventory_status if item['Status'] == 'OK')
        reorder_count = sum(1 for item in inventory_status if item['Status'] == 'Reorder')
        low_count = sum(1 for item in inventory_status if item['Status'] == 'Low')
        
        summary = {
            'total_products': total_products,
            'ok_count': ok_count,
            'reorder_count': reorder_count,
            'low_count': low_count,
            'ok_percent': ok_count / total_products * 100 if total_products > 0 else 0,
            'reorder_percent': reorder_count / total_products * 100 if total_products > 0 else 0,
            'low_percent': low_count / total_products * 100 if total_products > 0 else 0
        }
        
        return {
            'inventory_status': inventory_status,
            'summary': summary,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_forecast_report(self, forecast_data: pd.DataFrame, demand_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate forecast report.
        
        Args:
            forecast_data: DataFrame with forecast data
            demand_data: DataFrame with demand data
            
        Returns:
            Dict with report data
        """
        # Mock report data
        products = forecast_data['Product ID'].unique() if 'Product ID' in forecast_data.columns else []
        
        # Create mock forecast report
        forecast_report = []
        for product_id in products:
            # Get product forecast
            product_forecast = forecast_data[forecast_data['Product ID'] == product_id]
            
            # Get product demand
            product_demand = demand_data[demand_data['Product ID'] == product_id]
            
            # Calculate forecast metrics
            total_forecast = product_forecast['Forecast'].sum() if 'Forecast' in product_forecast.columns else np.random.randint(500, 5000)
            avg_forecast = product_forecast['Forecast'].mean() if 'Forecast' in product_forecast.columns else np.random.randint(50, 500)
            
            # Add to forecast report
            forecast_report.append({
                'Product ID': product_id,
                'Total Forecast': total_forecast,
                'Average Forecast': avg_forecast,
                'Forecast Horizon': len(product_forecast),
                'Forecast Start': product_forecast['Date'].min() if 'Date' in product_forecast.columns else datetime.now().strftime("%Y-%m-%d"),
                'Forecast End': product_forecast['Date'].max() if 'Date' in product_forecast.columns else (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            })
        
        # Create summary
        total_products = len(forecast_report)
        total_forecast = sum(item['Total Forecast'] for item in forecast_report)
        avg_forecast = sum(item['Average Forecast'] for item in forecast_report) / total_products if total_products > 0 else 0
        
        summary = {
            'total_products': total_products,
            'total_forecast': total_forecast,
            'avg_forecast': avg_forecast
        }
        
        return {
            'forecast_report': forecast_report,
            'summary': summary,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _save_html_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report as HTML.
        
        Args:
            report: Report data
            output_path: Output path
        """
        # Create simple HTML report
        html = "<html><head><title>Inventory Report</title>"
        html += "<style>body { font-family: Arial, sans-serif; margin: 20px; }"
        html += "table { border-collapse: collapse; width: 100%; }"
        html += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
        html += "th { background-color: #f2f2f2; }"
        html += "tr:nth-child(even) { background-color: #f9f9f9; }"
        html += "h1, h2 { color: #333; }</style></head><body>"
        
        # Add timestamp
        html += f"<h1>Report - {report['timestamp']}</h1>"
        
        # Add summary
        html += "<h2>Summary</h2>"
        html += "<table><tr><th>Metric</th><th>Value</th></tr>"
        for key, value in report['summary'].items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html += "</table>"
        
        # Add details if available
        if 'inventory_status' in report:
            html += "<h2>Inventory Status</h2>"
            html += "<table><tr>"
            for key in report['inventory_status'][0].keys():
                html += f"<th>{key}</th>"
            html += "</tr>"
            
            for item in report['inventory_status']:
                html += "<tr>"
                for key, value in item.items():
                    html += f"<td>{value}</td>"
                html += "</tr>"
            html += "</table>"
        
        if 'forecast_report' in report:
            html += "<h2>Forecast Report</h2>"
            html += "<table><tr>"
            for key in report['forecast_report'][0].keys():
                html += f"<th>{key}</th>"
            html += "</tr>"
            
            for item in report['forecast_report']:
                html += "<tr>"
                for key, value in item.items():
                    html += f"<td>{value}</td>"
                html += "</tr>"
            html += "</table>"
        
        html += "</body></html>"
        
        # Save HTML
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _save_json_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report as JSON.
        
        Args:
            report: Report data
            output_path: Output path
        """
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _save_csv_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report as CSV.
        
        Args:
            report: Report data
            output_path: Output path
        """
        # Determine which report type
        if 'inventory_status' in report:
            df = pd.DataFrame(report['inventory_status'])
        elif 'forecast_report' in report:
            df = pd.DataFrame(report['forecast_report'])
        else:
            raise ValueError("Unknown report type")
        
        # Save to CSV
        df.to_csv(output_path, index=False) 