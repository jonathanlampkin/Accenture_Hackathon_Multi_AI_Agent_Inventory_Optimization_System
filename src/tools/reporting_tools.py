"""
Tools for generating reports and visualizations used by the reporting agent.

This module contains tools for creating inventory status reports, forecast reports,
policy evaluation reports, supply chain reports, and interactive dashboards.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import traceback

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import markdown

# Tool framework
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import jinja2


class GenerateInventoryStatusReportTool(BaseTool):
    """Tool for generating inventory status reports."""
    
    name: str = "Generate Inventory Status Report"
    description: str = """
    Generate a comprehensive inventory status report with current levels, trends, and alerts.
    
    Input should include:
    - inventory_data_path: Path to the current inventory data CSV file
    - sales_data_path: Path to the sales data CSV file
    - policy_data_path: Optional path to the inventory policy data for comparisons
    - product_ids: Optional list of product IDs to include in the report (None for all products)
    - output_format: Output format (markdown, html, csv)
    - output_path: Path to save the report
    """
    
    class InputSchema(BaseModel):
        inventory_data_path: str = Field(
            ..., 
            description="Path to the current inventory data CSV file"
        )
        sales_data_path: str = Field(
            ..., 
            description="Path to the sales data CSV file"
        )
        policy_data_path: Optional[str] = Field(
            None, 
            description="Optional path to the inventory policy data for comparisons"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to include in the report (None for all products)"
        )
        output_format: str = Field(
            "html", 
            description="Output format (markdown, html, csv)"
        )
        output_path: str = Field(
            "output/reports/inventory_status.html", 
            description="Path to save the report"
        )
    
    def run(self, inventory_data_path: str,
            sales_data_path: str,
            policy_data_path: Optional[str] = None,
            product_ids: Optional[List[int]] = None,
            output_format: str = "html",
            output_path: str = "output/reports/inventory_status.html") -> Dict[str, Any]:
        """
        Generate a comprehensive inventory status report with current levels, trends, and alerts.
        
        Args:
            inventory_data_path: Path to the current inventory data CSV file
            sales_data_path: Path to the sales data CSV file
            policy_data_path: Optional path to the inventory policy data for comparisons
            product_ids: Optional list of product IDs to include in the report
            output_format: Output format (markdown, html, csv)
            output_path: Path to save the report
            
        Returns:
            Dict containing report details and path to the generated report
        """
        try:
            # Load inventory data
            inventory_data = pd.read_csv(inventory_data_path)
            
            # Load sales data
            sales_data = pd.read_csv(sales_data_path)
            sales_data['Date'] = pd.to_datetime(sales_data['Date'])
            
            # Load policy data if provided
            policy_data = None
            if policy_data_path:
                policy_data = pd.read_csv(policy_data_path)
            
            # Filter for specific products if provided
            if product_ids:
                inventory_data = inventory_data[inventory_data['Product ID'].isin(product_ids)]
                sales_data = sales_data[sales_data['Product ID'].isin(product_ids)]
                if policy_data is not None:
                    policy_data = policy_data[policy_data['Product ID'].isin(product_ids)]
            
            # Calculate recent sales velocity (last 30 days)
            last_30_days = sales_data[sales_data['Date'] >= (sales_data['Date'].max() - pd.Timedelta(days=30))]
            sales_velocity = last_30_days.groupby('Product ID')['Sales Quantity'].mean().reset_index()
            sales_velocity.rename(columns={'Sales Quantity': 'Daily Sales Velocity'}, inplace=True)
            
            # Merge with inventory data
            report_data = pd.merge(inventory_data, sales_velocity, on='Product ID', how='left')
            
            # Calculate days of supply
            report_data['Days of Supply'] = report_data['Current Quantity'] / report_data['Daily Sales Velocity']
            report_data['Days of Supply'] = report_data['Days of Supply'].round(1)
            
            # Define inventory status
            def get_status(row):
                if row['Days of Supply'] <= 7:
                    return 'Critical'
                elif row['Days of Supply'] <= 14:
                    return 'Low'
                elif row['Days of Supply'] <= 30:
                    return 'Adequate'
                else:
                    return 'Excess'
            
            report_data['Status'] = report_data.apply(get_status, axis=1)
            
            # Add policy comparison if available
            if policy_data is not None:
                # Get relevant policy columns
                policy_cols = ['Product ID', 'Reorder Point', 'Safety Stock', 'Maximum Inventory Level']
                policy_subset = policy_data[policy_cols]
                
                # Merge with report data
                report_data = pd.merge(report_data, policy_subset, on='Product ID', how='left')
                
                # Calculate inventory position relative to policy
                report_data['Below Reorder Point'] = report_data['Current Quantity'] < report_data['Reorder Point']
                report_data['Below Safety Stock'] = report_data['Current Quantity'] < report_data['Safety Stock']
                report_data['Above Maximum'] = report_data['Current Quantity'] > report_data['Maximum Inventory Level']
                
                # Generate recommendations
                def get_recommendation(row):
                    if row['Below Safety Stock']:
                        return 'Urgent reorder needed'
                    elif row['Below Reorder Point']:
                        return 'Reorder now'
                    elif row['Above Maximum']:
                        return 'Consider reducing inventory'
                    else:
                        return 'Inventory within optimal range'
                
                report_data['Recommendation'] = report_data.apply(get_recommendation, axis=1)
            
            # Create visualizations directory
            vis_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate inventory status visualization
            status_counts = report_data['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            plt.figure(figsize=(10, 6))
            colors = {'Critical': 'red', 'Low': 'orange', 'Adequate': 'green', 'Excess': 'blue'}
            sns.barplot(x='Status', y='Count', data=status_counts, palette=colors)
            plt.title('Inventory Status Distribution')
            plt.ylabel('Number of Products')
            plt.tight_layout()
            status_vis_path = os.path.join(vis_dir, 'inventory_status_distribution.png')
            plt.savefig(status_vis_path)
            plt.close()
            
            # Generate days of supply visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Product ID', y='Days of Supply', data=report_data, 
                       hue='Status', palette=colors, dodge=False)
            plt.title('Days of Supply by Product')
            plt.ylabel('Days of Supply')
            plt.axhline(y=14, color='orange', linestyle='--', label='Low Threshold')
            plt.axhline(y=7, color='red', linestyle='--', label='Critical Threshold')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            days_vis_path = os.path.join(vis_dir, 'days_of_supply.png')
            plt.savefig(days_vis_path)
            plt.close()
            
            # Prepare and save the report
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if output_format.lower() == 'csv':
                # Save as CSV
                report_data.to_csv(output_path, index=False)
                report_content = "CSV report generated"
            else:
                # Generate markdown/HTML report
                report_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                
                # Create markdown report
                md_content = f"""
# Inventory Status Report
**Generated on:** {report_date}

## Summary
- **Total Products:** {len(report_data)}
- **Critical Status:** {len(report_data[report_data['Status'] == 'Critical'])}
- **Low Status:** {len(report_data[report_data['Status'] == 'Low'])}
- **Adequate Status:** {len(report_data[report_data['Status'] == 'Adequate'])}
- **Excess Status:** {len(report_data[report_data['Status'] == 'Excess'])}

## Inventory Status Distribution
![Inventory Status Distribution](./visualizations/inventory_status_distribution.png)

## Days of Supply by Product
![Days of Supply by Product](./visualizations/days_of_supply.png)

## Detailed Inventory Status

| Product ID | Current Quantity | Daily Sales Velocity | Days of Supply | Status |
|------------|------------------|----------------------|----------------|--------|
"""
                
                # Add each product's data to the markdown table
                for _, row in report_data.sort_values('Days of Supply').iterrows():
                    md_content += f"| {row['Product ID']} | {row['Current Quantity']} | {row['Daily Sales Velocity']:.1f} | {row['Days of Supply']} | {row['Status']} |\n"
                
                # Add recommendations if policy data was provided
                if policy_data is not None:
                    md_content += """
## Inventory Recommendations

| Product ID | Current Quantity | Reorder Point | Safety Stock | Maximum Level | Recommendation |
|------------|------------------|---------------|--------------|---------------|----------------|
"""
                    for _, row in report_data.sort_values('Product ID').iterrows():
                        md_content += f"| {row['Product ID']} | {row['Current Quantity']} | {row['Reorder Point']} | {row['Safety Stock']} | {row['Maximum Inventory Level']} | {row['Recommendation']} |\n"
                
                # Save as markdown or convert to HTML
                if output_format.lower() == 'markdown':
                    with open(output_path, 'w') as f:
                        f.write(md_content)
                    report_content = md_content
                else:  # HTML
                    html_content = markdown.markdown(md_content, extensions=['tables'])
                    
                    # Add HTML styling
                    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Inventory Status Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .critical {{ color: red; font-weight: bold; }}
        .low {{ color: orange; font-weight: bold; }}
        .adequate {{ color: green; }}
        .excess {{ color: blue; }}
        h1, h2 {{ color: #333366; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
                    with open(output_path, 'w') as f:
                        f.write(styled_html)
                    report_content = styled_html
            
            # Prepare results dictionary
            return {
                "message": f"Inventory status report generated for {len(report_data)} products",
                "report_data": report_data.to_dict(orient='records'),
                "output_path": output_path,
                "visualization_directory": vis_dir,
                "report_format": output_format,
                "report_content": report_content
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate inventory status report"
            }


class GenerateForecastReportTool(BaseTool):
    """Tool for generating forecast reports."""
    
    name: str = "Generate Forecast Report"
    description: str = """
    Generate a report on demand forecasts including trends, accuracy, and future predictions.
    
    Input should include:
    - forecast_data_path: Path to the forecast data CSV file
    - historical_data_path: Path to the historical data for comparison
    - product_ids: Optional list of product IDs to include in the report (None for all products)
    - output_format: Output format (markdown, html, csv)
    - output_path: Path to save the report
    """
    
    class InputSchema(BaseModel):
        forecast_data_path: str = Field(
            ..., 
            description="Path to the forecast data CSV file"
        )
        historical_data_path: str = Field(
            ..., 
            description="Path to the historical data for comparison"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to include in the report (None for all products)"
        )
        output_format: str = Field(
            "html", 
            description="Output format (markdown, html, csv)"
        )
        output_path: str = Field(
            "output/reports/forecast_report.html", 
            description="Path to save the report"
        )
    
    def run(self, forecast_data_path: str,
            historical_data_path: str,
            product_ids: Optional[List[int]] = None,
            output_format: str = "html",
            output_path: str = "output/reports/forecast_report.html") -> Dict[str, Any]:
        """
        Generate a report on demand forecasts including trends, accuracy, and future predictions.
        
        Args:
            forecast_data_path: Path to the forecast data CSV file
            historical_data_path: Path to the historical data for comparison
            product_ids: Optional list of product IDs to include in the report
            output_format: Output format (markdown, html, csv)
            output_path: Path to save the report
            
        Returns:
            Dict containing report details and path to the generated report
        """
        try:
            # Load forecast data
            forecast_data = pd.read_csv(forecast_data_path)
            
            # Load historical data
            historical_data = pd.read_csv(historical_data_path)
            
            # Convert dates to datetime
            if 'Date' in forecast_data.columns:
                forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
            if 'Date' in historical_data.columns:
                historical_data['Date'] = pd.to_datetime(historical_data['Date'])
            
            # Filter for specific products if provided
            if product_ids:
                forecast_data = forecast_data[forecast_data['Product ID'].isin(product_ids)]
                historical_data = historical_data[historical_data['Product ID'].isin(product_ids)]
            
            # Create visualizations directory
            vis_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate forecast vs historical visualizations for each product
            product_forecast_figs = {}
            for product_id in forecast_data['Product ID'].unique():
                # Get product-specific data
                product_forecast = forecast_data[forecast_data['Product ID'] == product_id]
                product_historical = historical_data[historical_data['Product ID'] == product_id]
                
                # Skip if no data
                if product_forecast.empty or product_historical.empty:
                    continue
                
                # Create the visualization
                plt.figure(figsize=(12, 6))
                
                # Plot historical data
                sns.lineplot(x='Date', y='Sales Quantity', data=product_historical.sort_values('Date'), 
                           label='Historical', color='blue', marker='o')
                
                # Plot forecast data
                if 'Forecast' in product_forecast.columns:
                    sns.lineplot(x='Date', y='Forecast', data=product_forecast.sort_values('Date'), 
                               label='Forecast', color='red', marker='x')
                
                plt.title(f'Forecast vs Historical Sales for Product {product_id}')
                plt.xlabel('Date')
                plt.ylabel('Quantity')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the figure
                fig_path = os.path.join(vis_dir, f'forecast_product_{product_id}.png')
                plt.savefig(fig_path)
                plt.close()
                
                product_forecast_figs[product_id] = fig_path
            
            # Generate forecast accuracy metrics if available
            accuracy_metrics = None
            if all(col in forecast_data.columns for col in ['RMSE', 'MAE', 'R2']):
                accuracy_metrics = forecast_data[['Product ID', 'RMSE', 'MAE', 'R2']].drop_duplicates()
                
                # Create accuracy metrics visualization
                plt.figure(figsize=(12, 8))
                
                # Create subplots for RMSE, MAE, and R2
                plt.subplot(3, 1, 1)
                sns.barplot(x='Product ID', y='RMSE', data=accuracy_metrics)
                plt.title('RMSE by Product')
                plt.ylabel('RMSE')
                
                plt.subplot(3, 1, 2)
                sns.barplot(x='Product ID', y='MAE', data=accuracy_metrics)
                plt.title('MAE by Product')
                plt.ylabel('MAE')
                
                plt.subplot(3, 1, 3)
                sns.barplot(x='Product ID', y='R2', data=accuracy_metrics)
                plt.title('R² by Product')
                plt.ylabel('R²')
                
                plt.tight_layout()
                metrics_fig_path = os.path.join(vis_dir, 'forecast_accuracy_metrics.png')
                plt.savefig(metrics_fig_path)
                plt.close()
            
            # Prepare and save the report
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if output_format.lower() == 'csv':
                # Save as CSV
                forecast_data.to_csv(output_path, index=False)
                report_content = "CSV report generated"
            else:
                # Generate markdown/HTML report
                report_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                
                # Create markdown report
                md_content = f"""
# Demand Forecast Report
**Generated on:** {report_date}

## Summary
- **Total Products:** {len(forecast_data['Product ID'].unique())}
- **Forecast Period:** {forecast_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in forecast_data.columns else 'N/A'} to {forecast_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in forecast_data.columns else 'N/A'}

## Forecast Visualizations
"""
                
                # Add each product's forecast visualization
                for product_id, fig_path in product_forecast_figs.items():
                    relative_path = os.path.join('./visualizations', os.path.basename(fig_path))
                    md_content += f"""
### Product {product_id}
![Forecast for Product {product_id}]({relative_path})
"""
                
                # Add accuracy metrics if available
                if accuracy_metrics is not None:
                    md_content += """
## Forecast Accuracy Metrics
![Forecast Accuracy Metrics](./visualizations/forecast_accuracy_metrics.png)

### Detailed Metrics

| Product ID | RMSE | MAE | R² |
|------------|------|-----|-----|
"""
                    for _, row in accuracy_metrics.iterrows():
                        md_content += f"| {row['Product ID']} | {row['RMSE']:.2f} | {row['MAE']:.2f} | {row['R2']:.3f} |\n"
                
                # Add forecast details table
                md_content += """
## Detailed Forecast Data

| Product ID | Date | Forecast | Lower Bound | Upper Bound |
|------------|------|----------|-------------|-------------|
"""
                forecast_table_data = forecast_data.copy()
                if 'Date' in forecast_table_data.columns:
                    forecast_table_data['Date'] = forecast_table_data['Date'].dt.strftime('%Y-%m-%d')
                
                # Only include specific columns if they exist
                display_cols = ['Product ID', 'Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                existing_cols = [col for col in display_cols if col in forecast_table_data.columns]
                
                for _, row in forecast_table_data[existing_cols].head(20).iterrows():
                    row_str = "| "
                    for col in existing_cols:
                        if col in row:
                            val = row[col]
                            # Format float values
                            if isinstance(val, float):
                                val = f"{val:.2f}"
                            row_str += f"{val} | "
                        else:
                            row_str += " | "
                    md_content += row_str + "\n"
                
                if len(forecast_table_data) > 20:
                    md_content += "| ... | ... | ... | ... | ... |\n"
                    md_content += "*Note: Table truncated for brevity. Full data available in CSV format.*\n"
                
                # Save as markdown or convert to HTML
                if output_format.lower() == 'markdown':
                    with open(output_path, 'w') as f:
                        f.write(md_content)
                    report_content = md_content
                else:  # HTML
                    html_content = markdown.markdown(md_content, extensions=['tables'])
                    
                    # Add HTML styling
                    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Demand Forecast Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        h1, h2, h3 {{ color: #333366; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
                    with open(output_path, 'w') as f:
                        f.write(styled_html)
                    report_content = styled_html
            
            # Prepare results dictionary
            return {
                "message": f"Forecast report generated for {len(forecast_data['Product ID'].unique())} products",
                "output_path": output_path,
                "visualization_directory": vis_dir,
                "report_format": output_format,
                "report_content": report_content
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate forecast report"
            } 


class GeneratePolicyEvaluationReportTool(BaseTool):
    """Tool for generating comprehensive policy evaluation reports."""
    
    name: str = "Generate Policy Evaluation Report"
    description: str = """
    Generate a comprehensive policy evaluation report that combines optimization results and scenario testing.
    
    Input should include:
    - policy_data_path: Path to the inventory policy data
    - scenario_results_path: Path to the scenario simulation results
    - inventory_data_path: Path to the current inventory data for comparison
    - product_ids: Optional list of product IDs to include in the report (None for all products)
    - output_format: Output format (markdown, html, csv)
    - output_path: Path to save the report
    """
    
    class InputSchema(BaseModel):
        policy_data_path: str = Field(
            ..., 
            description="Path to the inventory policy data"
        )
        scenario_results_path: str = Field(
            ..., 
            description="Path to the scenario simulation results"
        )
        inventory_data_path: str = Field(
            ..., 
            description="Path to the current inventory data for comparison"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to include in the report (None for all products)"
        )
        output_format: str = Field(
            "html", 
            description="Output format (markdown, html, csv)"
        )
        output_path: str = Field(
            "output/reports/policy_evaluation.html", 
            description="Path to save the report"
        )
    
    def run(self, policy_data_path: str,
            scenario_results_path: str,
            inventory_data_path: str,
            product_ids: Optional[List[int]] = None,
            output_format: str = "html",
            output_path: str = "output/reports/policy_evaluation.html") -> Dict[str, Any]:
        """
        Generate a comprehensive policy evaluation report.
        
        Args:
            policy_data_path: Path to the inventory policy data
            scenario_results_path: Path to the scenario simulation results
            inventory_data_path: Path to the current inventory data for comparison
            product_ids: Optional list of product IDs to include in the report
            output_format: Output format (markdown, html, csv)
            output_path: Path to save the report
            
        Returns:
            Dict containing report details and path to the generated report
        """
        try:
            # Load policy data
            policy_data = pd.read_csv(policy_data_path)
            
            # Load scenario results
            scenario_results = pd.read_csv(scenario_results_path)
            
            # Load current inventory data
            inventory_data = pd.read_csv(inventory_data_path)
            
            # Filter for specific products if provided
            if product_ids:
                policy_data = policy_data[policy_data['Product ID'].isin(product_ids)]
                scenario_results = scenario_results[scenario_results['Product ID'].isin(product_ids)]
                inventory_data = inventory_data[inventory_data['Product ID'].isin(product_ids)]
            
            # Create visualizations directory
            vis_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Get unique scenarios
            scenarios = scenario_results['Scenario'].unique()
            
            # Calculate policy performance metrics
            performance_data = []
            
            for product_id in policy_data['Product ID'].unique():
                product_policy = policy_data[policy_data['Product ID'] == product_id]
                product_scenarios = scenario_results[scenario_results['Product ID'] == product_id]
                product_inventory = inventory_data[inventory_data['Product ID'] == product_id]
                
                if product_policy.empty or product_scenarios.empty or product_inventory.empty:
                    continue
                
                for _, policy_row in product_policy.iterrows():
                    policy_metrics = {
                        'Product ID': product_id,
                        'EOQ': policy_row.get('Economic Order Quantity', 0),
                        'ROP': policy_row.get('Reorder Point', 0),
                        'Safety Stock': policy_row.get('Safety Stock', 0),
                        'Current Stock': product_inventory['Current Quantity'].values[0],
                        'Policy Type': policy_row.get('Policy Type', 'Standard')
                    }
                    
                    # Calculate scenario performance
                    for scenario in scenarios:
                        scenario_data = product_scenarios[product_scenarios['Scenario'] == scenario]
                        if not scenario_data.empty:
                            service_level = scenario_data['Service Level'].mean()
                            total_cost = scenario_data['Total Cost'].mean()
                            stock_outs = scenario_data['Stockouts'].mean()
                            
                            policy_metrics[f'{scenario} Service Level'] = service_level
                            policy_metrics[f'{scenario} Total Cost'] = total_cost
                            policy_metrics[f'{scenario} Stockouts'] = stock_outs
                    
                    performance_data.append(policy_metrics)
            
            performance_df = pd.DataFrame(performance_data)
            
            # Generate policy comparison visualization
            plt.figure(figsize=(12, 8))
            
            # Create a subplot for each scenario
            num_scenarios = len(scenarios)
            fig, axes = plt.subplots(num_scenarios, 1, figsize=(12, 5 * num_scenarios))
            
            if num_scenarios == 1:
                axes = [axes]
            
            for i, scenario in enumerate(scenarios):
                service_level_col = f'{scenario} Service Level'
                cost_col = f'{scenario} Total Cost'
                
                if service_level_col in performance_df.columns and cost_col in performance_df.columns:
                    ax = axes[i]
                    performance_df.plot.scatter(
                        x=cost_col, 
                        y=service_level_col, 
                        ax=ax,
                        s=50,
                        alpha=0.7
                    )
                    
                    # Add product labels
                    for idx, row in performance_df.iterrows():
                        ax.annotate(
                            f"Product {row['Product ID']}", 
                            (row[cost_col], row[service_level_col]),
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
                    
                    ax.set_title(f'Service Level vs. Cost - {scenario} Scenario')
                    ax.set_xlabel('Total Cost')
                    ax.set_ylabel('Service Level (%)')
                    ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            performance_vis_path = os.path.join(vis_dir, 'policy_performance_comparison.png')
            plt.savefig(performance_vis_path)
            plt.close()
            
            # Generate stockout risk visualization
            plt.figure(figsize=(10, 6))
            
            stockout_data = []
            for scenario in scenarios:
                stockout_col = f'{scenario} Stockouts'
                if stockout_col in performance_df.columns:
                    for _, row in performance_df.iterrows():
                        stockout_data.append({
                            'Product ID': row['Product ID'],
                            'Scenario': scenario,
                            'Stockouts': row[stockout_col]
                        })
            
            stockout_df = pd.DataFrame(stockout_data)
            
            if not stockout_df.empty:
                sns.barplot(x='Product ID', y='Stockouts', hue='Scenario', data=stockout_df)
                plt.title('Stockout Risk by Product and Scenario')
                plt.xticks(rotation=45)
                plt.tight_layout()
                stockout_vis_path = os.path.join(vis_dir, 'stockout_risk_comparison.png')
                plt.savefig(stockout_vis_path)
                plt.close()
            
            # Prepare and save the report
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if output_format.lower() == 'csv':
                # Save as CSV
                performance_df.to_csv(output_path, index=False)
                report_content = "CSV report generated"
            else:
                # Generate markdown/HTML report
                report_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                
                # Create markdown report
                md_content = f"""
# Inventory Policy Evaluation Report
**Generated on:** {report_date}

## Summary
- **Total Products Evaluated:** {len(performance_df)}
- **Scenarios Tested:** {', '.join(scenarios)}

## Policy Performance Comparison
![Policy Performance Comparison](./visualizations/policy_performance_comparison.png)

## Stockout Risk Comparison
![Stockout Risk Comparison](./visualizations/stockout_risk_comparison.png)

## Detailed Policy Evaluation

| Product ID | EOQ | ROP | Safety Stock | Current Stock | Policy Type |
|------------|-----|-----|--------------|---------------|-------------|
"""
                
                # Add each product's policy data to the markdown table
                for _, row in performance_df.iterrows():
                    md_content += f"| {row['Product ID']} | {row['EOQ']} | {row['ROP']} | {row['Safety Stock']} | {row['Current Stock']} | {row['Policy Type']} |\n"
                
                # Add scenario performance data
                for scenario in scenarios:
                    service_level_col = f'{scenario} Service Level'
                    cost_col = f'{scenario} Total Cost'
                    stockout_col = f'{scenario} Stockouts'
                    
                    if all(col in performance_df.columns for col in [service_level_col, cost_col, stockout_col]):
                        md_content += f"""
## {scenario} Scenario Performance

| Product ID | Service Level | Total Cost | Stockouts |
|------------|---------------|------------|-----------|
"""
                        for _, row in performance_df.iterrows():
                            md_content += f"| {row['Product ID']} | {row[service_level_col]:.2f}% | ${row[cost_col]:.2f} | {row[stockout_col]:.2f} |\n"
                
                # Add recommendations section
                md_content += """
## Recommendations

"""
                
                # Generate recommendations based on performance
                if not performance_df.empty:
                    # Best overall policy
                    for scenario in scenarios:
                        service_level_col = f'{scenario} Service Level'
                        cost_col = f'{scenario} Total Cost'
                        
                        if service_level_col in performance_df.columns and cost_col in performance_df.columns:
                            # Normalize service level and cost for scoring
                            performance_df[f'{scenario} Normalized SL'] = performance_df[service_level_col] / performance_df[service_level_col].max()
                            performance_df[f'{scenario} Normalized Cost'] = performance_df[cost_col].min() / performance_df[cost_col]
                            
                            # Calculate simple score (higher is better)
                            performance_df[f'{scenario} Score'] = performance_df[f'{scenario} Normalized SL'] + performance_df[f'{scenario} Normalized Cost']
                            
                            # Get best policy for this scenario
                            best_idx = performance_df[f'{scenario} Score'].idxmax()
                            best_policy = performance_df.loc[best_idx]
                            
                            md_content += f"### {scenario} Scenario Best Policy\n"
                            md_content += f"- **Product ID {best_policy['Product ID']}** with policy type **{best_policy['Policy Type']}**\n"
                            md_content += f"- EOQ: {best_policy['EOQ']}, ROP: {best_policy['ROP']}, Safety Stock: {best_policy['Safety Stock']}\n"
                            md_content += f"- Service Level: {best_policy[service_level_col]:.2f}%, Total Cost: ${best_policy[cost_col]:.2f}\n\n"
                
                # Save as markdown or convert to HTML
                if output_format.lower() == 'markdown':
                    with open(output_path, 'w') as f:
                        f.write(md_content)
                    report_content = md_content
                else:
                    # Convert markdown to HTML
                    html_content = markdown.markdown(md_content)
                    
                    # Add HTML header, styles, and make images responsive
                    html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Policy Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
                    
                    with open(output_path, 'w') as f:
                        f.write(html_report)
                    report_content = html_report
            
            return {
                'status': 'success',
                'report_path': output_path,
                'products_analyzed': len(performance_df),
                'scenarios_analyzed': len(scenarios),
                'visualizations': [
                    os.path.join('visualizations', 'policy_performance_comparison.png'),
                    os.path.join('visualizations', 'stockout_risk_comparison.png')
                ]
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }


class GenerateSupplyChainPerformanceReportTool(BaseTool):
    """Tool for generating supply chain performance reports."""
    
    name: str = "Generate Supply Chain Performance Report"
    description: str = """
    Generate a supply chain performance report with key metrics, trends, and recommendations.
    
    Input should include:
    - inventory_data_path: Path to the current inventory data CSV file
    - order_data_path: Path to the order history data
    - lead_time_data_path: Path to lead time data
    - supplier_data_path: Optional path to supplier performance data
    - product_ids: Optional list of product IDs to include in the report (None for all products)
    - output_format: Output format (markdown, html, csv)
    - output_path: Path to save the report
    """
    
    class InputSchema(BaseModel):
        inventory_data_path: str = Field(
            ..., 
            description="Path to the current inventory data CSV file"
        )
        order_data_path: str = Field(
            ..., 
            description="Path to the order history data"
        )
        lead_time_data_path: str = Field(
            ..., 
            description="Path to lead time data"
        )
        supplier_data_path: Optional[str] = Field(
            None, 
            description="Optional path to supplier performance data"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to include in the report (None for all products)"
        )
        output_format: str = Field(
            "html", 
            description="Output format (markdown, html, csv)"
        )
        output_path: str = Field(
            "output/reports/supply_chain_performance.html", 
            description="Path to save the report"
        )
    
    def run(self, inventory_data_path: str,
            order_data_path: str,
            lead_time_data_path: str,
            supplier_data_path: Optional[str] = None,
            product_ids: Optional[List[int]] = None,
            output_format: str = "html",
            output_path: str = "output/reports/supply_chain_performance.html") -> Dict[str, Any]:
        """
        Generate a supply chain performance report with key metrics, trends, and recommendations.
        
        Args:
            inventory_data_path: Path to the current inventory data CSV file
            order_data_path: Path to the order history data
            lead_time_data_path: Path to lead time data
            supplier_data_path: Optional path to supplier performance data
            product_ids: Optional list of product IDs to include in the report
            output_format: Output format (markdown, html, csv)
            output_path: Path to save the report
            
        Returns:
            Dict containing report details and path to the generated report
        """
        try:
            # Load inventory data
            inventory_data = pd.read_csv(inventory_data_path)
            
            # Load order data
            order_data = pd.read_csv(order_data_path)
            order_data['Order Date'] = pd.to_datetime(order_data['Order Date'])
            
            # Load lead time data
            lead_time_data = pd.read_csv(lead_time_data_path)
            
            # Load supplier data if provided
            supplier_data = None
            if supplier_data_path:
                supplier_data = pd.read_csv(supplier_data_path)
            
            # Filter for specific products if provided
            if product_ids:
                inventory_data = inventory_data[inventory_data['Product ID'].isin(product_ids)]
                order_data = order_data[order_data['Product ID'].isin(product_ids)]
                lead_time_data = lead_time_data[lead_time_data['Product ID'].isin(product_ids)]
                if supplier_data is not None:
                    supplier_data = supplier_data[supplier_data['Product ID'].isin(product_ids)]
            
            # Create visualizations directory
            vis_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Calculate key metrics
            
            # 1. Order Fill Rate
            order_data['Filled'] = order_data['Ordered Quantity'] <= order_data['Fulfilled Quantity']
            fill_rate = order_data.groupby('Product ID')['Filled'].mean().reset_index()
            fill_rate['Fill Rate'] = fill_rate['Filled'] * 100
            
            # 2. Average Lead Time
            avg_lead_time = lead_time_data.groupby('Product ID')['Lead Time Days'].mean().reset_index()
            avg_lead_time['Lead Time Variability'] = lead_time_data.groupby('Product ID')['Lead Time Days'].std().values
            
            # 3. Supplier On-Time Delivery
            supplier_otd = None
            if supplier_data is not None and 'On-Time Delivery' in supplier_data.columns:
                supplier_otd = supplier_data.groupby(['Supplier', 'Product ID'])['On-Time Delivery'].mean().reset_index()
                supplier_otd['On-Time Delivery %'] = supplier_otd['On-Time Delivery'] * 100
            
            # Merge metrics into a performance dataframe
            performance_data = pd.merge(fill_rate, avg_lead_time, on='Product ID', how='outer')
            
            # Generate Lead Time Visualization
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Product ID', y='Lead Time Days', data=lead_time_data)
            plt.title('Lead Time Distribution by Product')
            plt.ylabel('Lead Time (Days)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            lead_time_vis_path = os.path.join(vis_dir, 'lead_time_distribution.png')
            plt.savefig(lead_time_vis_path)
            plt.close()
            
            # Generate Fill Rate Visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Product ID', y='Fill Rate', data=fill_rate)
            plt.title('Order Fill Rate by Product')
            plt.ylabel('Fill Rate (%)')
            plt.axhline(y=95, color='green', linestyle='--', label='Target')
            plt.legend()
            plt.tight_layout()
            fill_rate_vis_path = os.path.join(vis_dir, 'fill_rate_by_product.png')
            plt.savefig(fill_rate_vis_path)
            plt.close()
            
            # Generate Supplier Performance Visualization if available
            supplier_vis_path = None
            if supplier_otd is not None:
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Supplier', y='On-Time Delivery %', hue='Product ID', data=supplier_otd)
                plt.title('Supplier On-Time Delivery Performance')
                plt.ylabel('On-Time Delivery (%)')
                plt.axhline(y=90, color='green', linestyle='--', label='Target')
                plt.legend(title='Product ID')
                plt.xticks(rotation=45)
                plt.tight_layout()
                supplier_vis_path = os.path.join(vis_dir, 'supplier_performance.png')
                plt.savefig(supplier_vis_path)
                plt.close()
            
            # Prepare and save the report
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if output_format.lower() == 'csv':
                # Save as CSV
                performance_data.to_csv(output_path, index=False)
                report_content = "CSV report generated"
            else:
                # Generate markdown/HTML report
                report_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                
                # Create markdown report
                md_content = f"""
# Supply Chain Performance Report
**Generated on:** {report_date}

## Summary
- **Total Products Analyzed:** {len(performance_data)}
- **Average Fill Rate:** {performance_data['Fill Rate'].mean():.2f}%
- **Average Lead Time:** {performance_data['Lead Time Days'].mean():.2f} days
- **Average Lead Time Variability:** {performance_data['Lead Time Variability'].mean():.2f} days

## Lead Time Distribution by Product
![Lead Time Distribution](./visualizations/lead_time_distribution.png)

## Order Fill Rate by Product
![Fill Rate by Product](./visualizations/fill_rate_by_product.png)

"""
                
                # Add supplier performance section if available
                if supplier_vis_path:
                    md_content += """
## Supplier On-Time Delivery Performance
![Supplier Performance](./visualizations/supplier_performance.png)

"""
                
                # Add detailed metrics table
                md_content += """
## Detailed Supply Chain Metrics

| Product ID | Fill Rate (%) | Avg. Lead Time (days) | Lead Time Variability |
|------------|---------------|------------------------|------------------------|
"""
                
                # Add each product's metrics to the markdown table
                for _, row in performance_data.iterrows():
                    md_content += f"| {row['Product ID']} | {row['Fill Rate']:.2f} | {row['Lead Time Days']:.2f} | {row['Lead Time Variability']:.2f} |\n"
                
                # Add supplier performance table if available
                if supplier_otd is not None:
                    md_content += """
## Supplier Performance

| Supplier | Product ID | On-Time Delivery (%) |
|----------|------------|----------------------|
"""
                    for _, row in supplier_otd.iterrows():
                        md_content += f"| {row['Supplier']} | {row['Product ID']} | {row['On-Time Delivery %']:.2f} |\n"
                
                # Add recommendations section
                md_content += """
## Recommendations

"""
                
                # Generate recommendations based on metrics
                for _, row in performance_data.iterrows():
                    md_content += f"### Product ID: {row['Product ID']}\n"
                    
                    if row['Fill Rate'] < 95:
                        md_content += f"- **Improve Fill Rate:** Current rate is {row['Fill Rate']:.2f}%, which is below the target of 95%. Consider increasing safety stock levels.\n"
                    
                    if row['Lead Time Variability'] > 5:
                        md_content += f"- **Address Lead Time Variability:** Variability is high at {row['Lead Time Variability']:.2f} days. Work with suppliers to stabilize lead times or increase safety stock to account for variability.\n"
                    
                    if supplier_otd is not None:
                        product_suppliers = supplier_otd[supplier_otd['Product ID'] == row['Product ID']]
                        poor_suppliers = product_suppliers[product_suppliers['On-Time Delivery %'] < 90]
                        
                        if not poor_suppliers.empty:
                            md_content += "- **Address Supplier Performance Issues:** The following suppliers have below-target on-time delivery performance:\n"
                            for _, supp_row in poor_suppliers.iterrows():
                                md_content += f"  - {supp_row['Supplier']}: {supp_row['On-Time Delivery %']:.2f}%\n"
                    
                    md_content += "\n"
                
                # Save as markdown or convert to HTML
                if output_format.lower() == 'markdown':
                    with open(output_path, 'w') as f:
                        f.write(md_content)
                    report_content = md_content
                else:
                    # Convert markdown to HTML
                    html_content = markdown.markdown(md_content)
                    
                    # Add HTML header, styles, and make images responsive
                    html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supply Chain Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
                    
                    with open(output_path, 'w') as f:
                        f.write(html_report)
                    report_content = html_report
            
            return {
                'status': 'success',
                'report_path': output_path,
                'products_analyzed': len(performance_data),
                'visualizations': [
                    os.path.join('visualizations', 'lead_time_distribution.png'),
                    os.path.join('visualizations', 'fill_rate_by_product.png'),
                    os.path.join('visualizations', 'supplier_performance.png') if supplier_vis_path else None
                ]
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            } 


class GenerateDashboardTool(BaseTool):
    """Tool for generating an interactive dashboard combining all aspects of inventory optimization."""
    
    name: str = "Generate Interactive Dashboard"
    description: str = """
    Generate an interactive HTML dashboard that brings together data from forecasting, optimization, scenario planning, 
    and reporting to provide a unified view of the inventory optimization system.
    
    Input should include:
    - forecast_data_path: Path to forecast data
    - inventory_data_path: Path to current inventory data
    - policy_data_path: Path to inventory policy data
    - scenario_results_path: Path to scenario simulation results
    - order_data_path: Path to order history data
    - lead_time_data_path: Path to lead time data
    - supplier_data_path: Optional path to supplier performance data
    - product_ids: Optional list of product IDs to include in the dashboard (None for all products)
    - output_path: Path to save the dashboard HTML
    """
    
    class InputSchema(BaseModel):
        forecast_data_path: str = Field(
            ..., 
            description="Path to forecast data"
        )
        inventory_data_path: str = Field(
            ..., 
            description="Path to current inventory data"
        )
        policy_data_path: str = Field(
            ..., 
            description="Path to inventory policy data"
        )
        scenario_results_path: str = Field(
            ..., 
            description="Path to scenario simulation results"
        )
        order_data_path: str = Field(
            ..., 
            description="Path to order history data"
        )
        lead_time_data_path: str = Field(
            ..., 
            description="Path to lead time data"
        )
        supplier_data_path: Optional[str] = Field(
            None, 
            description="Optional path to supplier performance data"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to include in the dashboard (None for all products)"
        )
        output_path: str = Field(
            "output/dashboard/inventory_dashboard.html", 
            description="Path to save the dashboard HTML"
        )
    
    def run(self, forecast_data_path: str,
            inventory_data_path: str,
            policy_data_path: str,
            scenario_results_path: str,
            order_data_path: str,
            lead_time_data_path: str,
            supplier_data_path: Optional[str] = None,
            product_ids: Optional[List[int]] = None,
            output_path: str = "output/dashboard/inventory_dashboard.html") -> Dict[str, Any]:
        """
        Generate an interactive HTML dashboard combining all aspects of inventory optimization.
        
        Args:
            forecast_data_path: Path to forecast data
            inventory_data_path: Path to current inventory data
            policy_data_path: Path to inventory policy data
            scenario_results_path: Path to scenario simulation results
            order_data_path: Path to order history data
            lead_time_data_path: Path to lead time data
            supplier_data_path: Optional path to supplier performance data
            product_ids: Optional list of product IDs to include in the dashboard
            output_path: Path to save the dashboard HTML
            
        Returns:
            Dict containing dashboard details and path to the generated dashboard
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from datetime import datetime
            
            # Load all data
            forecast_data = pd.read_csv(forecast_data_path)
            inventory_data = pd.read_csv(inventory_data_path)
            policy_data = pd.read_csv(policy_data_path)
            scenario_results = pd.read_csv(scenario_results_path)
            order_data = pd.read_csv(order_data_path)
            lead_time_data = pd.read_csv(lead_time_data_path)
            
            supplier_data = None
            if supplier_data_path:
                supplier_data = pd.read_csv(supplier_data_path)
            
            # Filter for specific products if provided
            if product_ids:
                forecast_data = forecast_data[forecast_data['Product ID'].isin(product_ids)]
                inventory_data = inventory_data[inventory_data['Product ID'].isin(product_ids)]
                policy_data = policy_data[policy_data['Product ID'].isin(product_ids)]
                scenario_results = scenario_results[scenario_results['Product ID'].isin(product_ids)]
                order_data = order_data[order_data['Product ID'].isin(product_ids)]
                lead_time_data = lead_time_data[lead_time_data['Product ID'].isin(product_ids)]
                if supplier_data is not None:
                    supplier_data = supplier_data[supplier_data['Product ID'].isin(product_ids)]
            
            # Ensure date columns are datetime
            if 'Date' in forecast_data.columns:
                forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
            if 'Order Date' in order_data.columns:
                order_data['Order Date'] = pd.to_datetime(order_data['Order Date'])
            
            # Create output directory
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Create plotly figures
            
            # 1. Inventory Status Overview
            fig_inventory = px.bar(
                inventory_data, 
                x='Product ID', 
                y='Current Quantity',
                title='Current Inventory Levels by Product',
                labels={'Current Quantity': 'Quantity', 'Product ID': 'Product'}
            )
            
            # 2. Forecasting Chart
            fig_forecast = go.Figure()
            
            # Group by product and date if multiple products
            if 'Product ID' in forecast_data.columns and 'Date' in forecast_data.columns and 'Forecast' in forecast_data.columns:
                for product_id in forecast_data['Product ID'].unique():
                    product_forecast = forecast_data[forecast_data['Product ID'] == product_id]
                    fig_forecast.add_trace(go.Scatter(
                        x=product_forecast['Date'],
                        y=product_forecast['Forecast'],
                        mode='lines+markers',
                        name=f'Product {product_id}'
                    ))
            
            fig_forecast.update_layout(
                title='Demand Forecast by Product',
                xaxis_title='Date',
                yaxis_title='Forecasted Demand'
            )
            
            # 3. Policy Metrics
            # Calculate days of supply from inventory and forecast if forecast has appropriate columns
            if 'Product ID' in forecast_data.columns and 'Forecast' in forecast_data.columns:
                avg_daily_demand = forecast_data.groupby('Product ID')['Forecast'].mean().reset_index()
                avg_daily_demand.rename(columns={'Forecast': 'Avg Daily Demand'}, inplace=True)
                
                # Merge with inventory
                inventory_with_demand = pd.merge(inventory_data, avg_daily_demand, on='Product ID', how='left')
                inventory_with_demand['Days of Supply'] = inventory_with_demand['Current Quantity'] / inventory_with_demand['Avg Daily Demand']
                inventory_with_demand['Days of Supply'] = inventory_with_demand['Days of Supply'].fillna(0).round(1)
                
                # Create days of supply figure
                fig_days_supply = px.bar(
                    inventory_with_demand,
                    x='Product ID',
                    y='Days of Supply',
                    title='Days of Supply by Product',
                    labels={'Days of Supply': 'Days', 'Product ID': 'Product'}
                )
                
                # Add thresholds
                fig_days_supply.add_hline(y=7, line_width=2, line_dash="dash", line_color="red", annotation_text="Critical (7 days)")
                fig_days_supply.add_hline(y=14, line_width=2, line_dash="dash", line_color="orange", annotation_text="Low (14 days)")
                fig_days_supply.add_hline(y=30, line_width=2, line_dash="dash", line_color="green", annotation_text="Target (30 days)")
            else:
                # Fallback if we don't have forecast data in the right format
                fig_days_supply = go.Figure()
                fig_days_supply.update_layout(
                    title='Days of Supply by Product (Data Not Available)',
                    xaxis_title='Product ID',
                    yaxis_title='Days of Supply'
                )
            
            # 4. Scenario Comparison
            if 'Scenario' in scenario_results.columns and 'Service Level' in scenario_results.columns and 'Total Cost' in scenario_results.columns:
                # Create bubble chart for scenario comparison
                fig_scenarios = px.scatter(
                    scenario_results,
                    x='Total Cost',
                    y='Service Level',
                    size='Stockouts',  # Size of bubble by stockouts (larger = more stockouts)
                    color='Scenario',
                    hover_name='Product ID',
                    size_max=50,
                    title='Scenario Comparison: Service Level vs Cost'
                )
                
                fig_scenarios.update_layout(
                    xaxis_title='Total Cost ($)',
                    yaxis_title='Service Level (%)'
                )
            else:
                # Fallback if we don't have scenario data in the right format
                fig_scenarios = go.Figure()
                fig_scenarios.update_layout(
                    title='Scenario Comparison (Data Not Available)',
                    xaxis_title='Total Cost',
                    yaxis_title='Service Level'
                )
            
            # 5. Lead Time Analysis
            fig_lead_time = px.box(
                lead_time_data, 
                x='Product ID', 
                y='Lead Time Days',
                title='Lead Time Distribution by Product',
                labels={'Lead Time Days': 'Days', 'Product ID': 'Product'}
            )
            
            # 6. Order Fill Rate
            if 'Ordered Quantity' in order_data.columns and 'Fulfilled Quantity' in order_data.columns:
                order_data['Fill Rate'] = (order_data['Fulfilled Quantity'] / order_data['Ordered Quantity'] * 100).clip(upper=100)
                avg_fill_rate = order_data.groupby('Product ID')['Fill Rate'].mean().reset_index()
                
                fig_fill_rate = px.bar(
                    avg_fill_rate, 
                    x='Product ID', 
                    y='Fill Rate',
                    title='Average Order Fill Rate by Product',
                    labels={'Fill Rate': 'Fill Rate (%)', 'Product ID': 'Product'}
                )
                
                fig_fill_rate.add_hline(y=95, line_width=2, line_dash="dash", line_color="green", annotation_text="Target (95%)")
            else:
                # Fallback if we don't have order data in the right format
                fig_fill_rate = go.Figure()
                fig_fill_rate.update_layout(
                    title='Average Order Fill Rate by Product (Data Not Available)',
                    xaxis_title='Product ID',
                    yaxis_title='Fill Rate (%)'
                )
            
            # 7. Supplier Performance if available
            if supplier_data is not None and 'Supplier' in supplier_data.columns and 'On-Time Delivery' in supplier_data.columns:
                supplier_otd = supplier_data.groupby(['Supplier', 'Product ID'])['On-Time Delivery'].mean().reset_index()
                supplier_otd['On-Time Delivery %'] = supplier_otd['On-Time Delivery'] * 100
                
                fig_supplier = px.bar(
                    supplier_otd, 
                    x='Supplier', 
                    y='On-Time Delivery %',
                    color='Product ID',
                    title='Supplier On-Time Delivery Performance',
                    labels={'On-Time Delivery %': 'On-Time (%)', 'Supplier': 'Supplier'}
                )
                
                fig_supplier.add_hline(y=90, line_width=2, line_dash="dash", line_color="green", annotation_text="Target (90%)")
            else:
                # Create empty supplier performance chart if data not available
                fig_supplier = go.Figure()
                fig_supplier.update_layout(
                    title='Supplier Performance (Data Not Available)',
                    xaxis_title='Supplier',
                    yaxis_title='On-Time Delivery (%)'
                )
            
            # 8. Inventory Policy Compliance
            if all(col in policy_data.columns for col in ['Product ID', 'Reorder Point', 'Safety Stock']):
                # Merge policy data with inventory data
                policy_compliance = pd.merge(inventory_data, policy_data, on='Product ID', how='left')
                
                # Calculate compliance metrics
                policy_compliance['Below Reorder Point'] = policy_compliance['Current Quantity'] < policy_compliance['Reorder Point']
                policy_compliance['Below Safety Stock'] = policy_compliance['Current Quantity'] < policy_compliance['Safety Stock']
                
                # Create compliance status
                def get_status(row):
                    if row['Below Safety Stock']:
                        return 'Below Safety Stock'
                    elif row['Below Reorder Point']:
                        return 'Below Reorder Point'
                    else:
                        return 'Compliant'
                
                policy_compliance['Compliance Status'] = policy_compliance.apply(get_status, axis=1)
                
                # Create compliance chart
                fig_compliance = px.bar(
                    policy_compliance,
                    x='Product ID',
                    y='Current Quantity',
                    color='Compliance Status',
                    title='Inventory Policy Compliance',
                    labels={'Current Quantity': 'Current Inventory', 'Product ID': 'Product'}
                )
                
                # Add reference lines for each product's ROP and safety stock
                for i, row in policy_compliance.iterrows():
                    fig_compliance.add_shape(
                        type="line",
                        x0=i-0.4, x1=i+0.4,
                        y0=row['Reorder Point'], y1=row['Reorder Point'],
                        line=dict(color="orange", width=2, dash="dash"),
                    )
                    fig_compliance.add_shape(
                        type="line",
                        x0=i-0.4, x1=i+0.4,
                        y0=row['Safety Stock'], y1=row['Safety Stock'],
                        line=dict(color="red", width=2, dash="dash"),
                    )
            else:
                # Create empty compliance chart if data not available
                fig_compliance = go.Figure()
                fig_compliance.update_layout(
                    title='Inventory Policy Compliance (Data Not Available)',
                    xaxis_title='Product ID',
                    yaxis_title='Current Inventory'
                )
            
            # Create the HTML dashboard using template
            dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Optimization Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .dashboard-row {{
            display: flex;
            flex-wrap: wrap;
            margin: -10px;
        }}
        .dashboard-cell {{
            flex: 1 1 calc(50% - 20px);
            margin: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .full-width {{
            flex: 1 1 calc(100% - 20px);
        }}
        .dashboard-cell-header {{
            background-color: #eef2f7;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .dashboard-cell-content {{
            padding: 15px;
            height: 400px;
        }}
        .summary-stats {{
            display: flex;
            flex-wrap: wrap;
        }}
        .stat-box {{
            flex: 1 1 calc(25% - 20px);
            margin: 10px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .stat-box p {{
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
            color: #3498db;
        }}
        .critical {{
            color: #e74c3c;
        }}
        .warning {{
            color: #f39c12;
        }}
        .good {{
            color: #2ecc71;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Inventory Optimization Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <!-- Summary Stats -->
        <div class="summary-stats">
            <div class="stat-box">
                <h3>Products</h3>
                <p>{len(inventory_data['Product ID'].unique())}</p>
            </div>
            <div class="stat-box">
                <h3>Avg Fill Rate</h3>
                <p class="{('critical' if 'Fill Rate' in locals() and avg_fill_rate['Fill Rate'].mean() < 90 else 'warning' if 'Fill Rate' in locals() and avg_fill_rate['Fill Rate'].mean() < 95 else 'good')}">
                    {avg_fill_rate['Fill Rate'].mean():.1f}% if 'Fill Rate' in locals() else '--'
                </p>
            </div>
            <div class="stat-box">
                <h3>Avg Lead Time</h3>
                <p>{lead_time_data['Lead Time Days'].mean():.1f} days</p>
            </div>
            <div class="stat-box">
                <h3>Critical Stock</h3>
                <p class="critical">{len(inventory_with_demand[inventory_with_demand['Days of Supply'] <= 7]) if 'inventory_with_demand' in locals() else 0}</p>
            </div>
        </div>
        
        <!-- First Row -->
        <div class="dashboard-row">
            <div class="dashboard-cell">
                <div class="dashboard-cell-header">Current Inventory Levels</div>
                <div class="dashboard-cell-content" id="inventory-chart"></div>
            </div>
            <div class="dashboard-cell">
                <div class="dashboard-cell-header">Days of Supply</div>
                <div class="dashboard-cell-content" id="days-supply-chart"></div>
            </div>
        </div>
        
        <!-- Second Row -->
        <div class="dashboard-row">
            <div class="dashboard-cell full-width">
                <div class="dashboard-cell-header">Demand Forecast</div>
                <div class="dashboard-cell-content" id="forecast-chart"></div>
            </div>
        </div>
        
        <!-- Third Row -->
        <div class="dashboard-row">
            <div class="dashboard-cell">
                <div class="dashboard-cell-header">Inventory Policy Compliance</div>
                <div class="dashboard-cell-content" id="compliance-chart"></div>
            </div>
            <div class="dashboard-cell">
                <div class="dashboard-cell-header">Order Fill Rate</div>
                <div class="dashboard-cell-content" id="fill-rate-chart"></div>
            </div>
        </div>
        
        <!-- Fourth Row -->
        <div class="dashboard-row">
            <div class="dashboard-cell">
                <div class="dashboard-cell-header">Lead Time Distribution</div>
                <div class="dashboard-cell-content" id="lead-time-chart"></div>
            </div>
            <div class="dashboard-cell">
                <div class="dashboard-cell-header">Scenario Comparison</div>
                <div class="dashboard-cell-content" id="scenarios-chart"></div>
            </div>
        </div>
        
        <!-- Fifth Row -->
        <div class="dashboard-row">
            <div class="dashboard-cell full-width">
                <div class="dashboard-cell-header">Supplier Performance</div>
                <div class="dashboard-cell-content" id="supplier-chart"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>Multi-Agent Inventory Optimization System &copy; {datetime.now().year}</p>
        </div>
    </div>
    
    <script>
        // Load charts
        var inventoryChart = {fig_inventory.to_json()};
        Plotly.newPlot('inventory-chart', inventoryChart.data, inventoryChart.layout);
        
        var daysSupplyChart = {fig_days_supply.to_json()};
        Plotly.newPlot('days-supply-chart', daysSupplyChart.data, daysSupplyChart.layout);
        
        var forecastChart = {fig_forecast.to_json()};
        Plotly.newPlot('forecast-chart', forecastChart.data, forecastChart.layout);
        
        var complianceChart = {fig_compliance.to_json()};
        Plotly.newPlot('compliance-chart', complianceChart.data, complianceChart.layout);
        
        var fillRateChart = {fig_fill_rate.to_json()};
        Plotly.newPlot('fill-rate-chart', fillRateChart.data, fillRateChart.layout);
        
        var leadTimeChart = {fig_lead_time.to_json()};
        Plotly.newPlot('lead-time-chart', leadTimeChart.data, leadTimeChart.layout);
        
        var scenariosChart = {fig_scenarios.to_json()};
        Plotly.newPlot('scenarios-chart', scenariosChart.data, scenariosChart.layout);
        
        var supplierChart = {fig_supplier.to_json()};
        Plotly.newPlot('supplier-chart', supplierChart.data, supplierChart.layout);
        
        // Make charts responsive
        window.onresize = function() {{
            Plotly.relayout('inventory-chart', {{
                'width': document.getElementById('inventory-chart').offsetWidth,
                'height': document.getElementById('inventory-chart').offsetHeight
            }});
            Plotly.relayout('days-supply-chart', {{
                'width': document.getElementById('days-supply-chart').offsetWidth,
                'height': document.getElementById('days-supply-chart').offsetHeight
            }});
            Plotly.relayout('forecast-chart', {{
                'width': document.getElementById('forecast-chart').offsetWidth,
                'height': document.getElementById('forecast-chart').offsetHeight
            }});
            Plotly.relayout('compliance-chart', {{
                'width': document.getElementById('compliance-chart').offsetWidth,
                'height': document.getElementById('compliance-chart').offsetHeight
            }});
            Plotly.relayout('fill-rate-chart', {{
                'width': document.getElementById('fill-rate-chart').offsetWidth,
                'height': document.getElementById('fill-rate-chart').offsetHeight
            }});
            Plotly.relayout('lead-time-chart', {{
                'width': document.getElementById('lead-time-chart').offsetWidth,
                'height': document.getElementById('lead-time-chart').offsetHeight
            }});
            Plotly.relayout('scenarios-chart', {{
                'width': document.getElementById('scenarios-chart').offsetWidth,
                'height': document.getElementById('scenarios-chart').offsetHeight
            }});
            Plotly.relayout('supplier-chart', {{
                'width': document.getElementById('supplier-chart').offsetWidth,
                'height': document.getElementById('supplier-chart').offsetHeight
            }});
        }};
    </script>
</body>
</html>
"""
            
            # Write dashboard to file
            with open(output_path, 'w') as f:
                f.write(dashboard_html)
            
            return {
                'status': 'success',
                'dashboard_path': output_path,
                'products_analyzed': len(inventory_data['Product ID'].unique()),
                'charts_included': [
                    'Current Inventory Levels',
                    'Days of Supply',
                    'Demand Forecast',
                    'Inventory Policy Compliance',
                    'Order Fill Rate',
                    'Lead Time Distribution',
                    'Scenario Comparison',
                    'Supplier Performance'
                ]
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            } 