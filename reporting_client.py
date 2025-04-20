#!/usr/bin/env python3
"""
Reporting Client Example

This script demonstrates how to use the reporting endpoints of the
Inventory Optimization API. It provides both command-line and interactive
options for generating various reports and dashboards.
"""

import os
import sys
import requests
import json
import argparse
import webbrowser
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Constants
API_BASE_URL = "http://localhost:8000"
DATA_DIR = "data"


def ensure_data_directory() -> None:
    """Create data directory and check for example data files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # List of expected example data files
    example_files = [
        "inventory_data.csv",
        "sales_data.csv",
        "forecast_data.csv", 
        "historical_data.csv",
        "policy_data.csv",
        "scenario_results.csv",
        "order_data.csv",
        "lead_time_data.csv",
        "supplier_data.csv"
    ]
    
    # Check which files exist
    missing_files = [file for file in example_files 
                    if not os.path.exists(os.path.join(DATA_DIR, file))]
    
    if missing_files:
        print("Warning: The following example data files are missing:")
        for file in missing_files:
            print(f"  - {DATA_DIR}/{file}")
        print("\nYou'll need to provide valid file paths when using the client.\n")


def check_api_connection() -> bool:
    """Check if the API is running and accessible.
    
    Returns:
        bool: True if API is running, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to API at {API_BASE_URL}")
        print("Please start the API server with: python start_api.py")
        return False
    except Exception as e:
        print(f"Error checking API connection: {str(e)}")
        return False


def generate_inventory_status_report(inventory_data_path: str, 
                                   sales_data_path: str, 
                                   policy_data_path: Optional[str] = None, 
                                   product_ids: Optional[List[int]] = None, 
                                   output_format: str = "html") -> Optional[Dict[str, Any]]:
    """
    Generate an inventory status report.
    
    Args:
        inventory_data_path: Path to inventory data CSV
        sales_data_path: Path to sales data CSV
        policy_data_path: Optional path to policy data CSV
        product_ids: Optional list of product IDs to include
        output_format: Output format (html, markdown, csv)
    
    Returns:
        Response dictionary from the API or None if error occurred
    """
    # Create request payload
    payload = {
        "data_paths": {
            "inventory_data_path": inventory_data_path,
            "sales_data_path": sales_data_path
        },
        "output_format": output_format
    }
    
    # Add optional parameters
    if policy_data_path:
        payload["data_paths"]["policy_data_path"] = policy_data_path
    
    if product_ids:
        payload["product_ids"] = product_ids
    
    # Make API request
    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/inventory-status",
            json=payload,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            print(f"Inventory status report generated successfully")
            
            if "report_url" in result:
                print(f"Report URL: {API_BASE_URL}{result['report_url']}")
                # Open report in browser
                webbrowser.open(f"{API_BASE_URL}{result['report_url']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return None


def generate_forecast_report(forecast_data_path: str, 
                           historical_data_path: str, 
                           product_ids: Optional[List[int]] = None, 
                           output_format: str = "html") -> Optional[Dict[str, Any]]:
    """
    Generate a forecast report.
    
    Args:
        forecast_data_path: Path to forecast data CSV
        historical_data_path: Path to historical data CSV
        product_ids: Optional list of product IDs to include
        output_format: Output format (html, markdown, csv)
    
    Returns:
        Response dictionary from the API or None if error occurred
    """
    # Create request payload
    payload = {
        "data_paths": {
            "forecast_data_path": forecast_data_path,
            "historical_data_path": historical_data_path
        },
        "output_format": output_format
    }
    
    # Add optional parameters
    if product_ids:
        payload["product_ids"] = product_ids
    
    # Make API request
    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/forecast",
            json=payload,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            print(f"Forecast report generated successfully")
            
            if "report_url" in result:
                print(f"Report URL: {API_BASE_URL}{result['report_url']}")
                # Open report in browser
                webbrowser.open(f"{API_BASE_URL}{result['report_url']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return None


def generate_policy_evaluation_report(policy_data_path: str,
                                    scenario_results_path: str,
                                    inventory_data_path: str,
                                    product_ids: Optional[List[int]] = None,
                                    output_format: str = "html") -> Optional[Dict[str, Any]]:
    """
    Generate a policy evaluation report.
    
    Args:
        policy_data_path: Path to policy data CSV
        scenario_results_path: Path to scenario results CSV
        inventory_data_path: Path to inventory data CSV
        product_ids: Optional list of product IDs to include
        output_format: Output format (html, markdown, csv)
    
    Returns:
        Response dictionary from the API or None if error occurred
    """
    # Create request payload
    payload = {
        "data_paths": {
            "policy_data_path": policy_data_path,
            "scenario_results_path": scenario_results_path,
            "inventory_data_path": inventory_data_path
        },
        "output_format": output_format
    }
    
    # Add optional parameters
    if product_ids:
        payload["product_ids"] = product_ids
    
    # Make API request
    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/policy-evaluation",
            json=payload,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            print(f"Policy evaluation report generated successfully")
            
            if "report_url" in result:
                print(f"Report URL: {API_BASE_URL}{result['report_url']}")
                # Open report in browser
                webbrowser.open(f"{API_BASE_URL}{result['report_url']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return None


def generate_supply_chain_report(inventory_data_path: str,
                               order_data_path: str,
                               lead_time_data_path: str,
                               supplier_data_path: Optional[str] = None,
                               product_ids: Optional[List[int]] = None,
                               output_format: str = "html") -> Optional[Dict[str, Any]]:
    """
    Generate a supply chain performance report.
    
    Args:
        inventory_data_path: Path to inventory data CSV
        order_data_path: Path to order data CSV
        lead_time_data_path: Path to lead time data CSV
        supplier_data_path: Optional path to supplier data CSV
        product_ids: Optional list of product IDs to include
        output_format: Output format (html, markdown, csv)
    
    Returns:
        Response dictionary from the API or None if error occurred
    """
    # Create request payload
    payload = {
        "data_paths": {
            "inventory_data_path": inventory_data_path,
            "order_data_path": order_data_path,
            "lead_time_data_path": lead_time_data_path
        },
        "output_format": output_format
    }
    
    # Add optional parameters
    if supplier_data_path:
        payload["data_paths"]["supplier_data_path"] = supplier_data_path
    
    if product_ids:
        payload["product_ids"] = product_ids
    
    # Make API request
    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/supply-chain",
            json=payload,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            print(f"Supply chain report generated successfully")
            
            if "report_url" in result:
                print(f"Report URL: {API_BASE_URL}{result['report_url']}")
                # Open report in browser
                webbrowser.open(f"{API_BASE_URL}{result['report_url']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return None


def generate_dashboard(forecast_data_path: str,
                     inventory_data_path: str,
                     policy_data_path: str,
                     scenario_results_path: str,
                     order_data_path: str,
                     lead_time_data_path: str,
                     supplier_data_path: Optional[str] = None,
                     product_ids: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """
    Generate an interactive dashboard.
    
    Args:
        forecast_data_path: Path to forecast data CSV
        inventory_data_path: Path to inventory data CSV
        policy_data_path: Path to policy data CSV
        scenario_results_path: Path to scenario results CSV
        order_data_path: Path to order data CSV
        lead_time_data_path: Path to lead time data CSV
        supplier_data_path: Optional path to supplier data CSV
        product_ids: Optional list of product IDs to include
    
    Returns:
        Response dictionary from the API or None if error occurred
    """
    # Create request payload
    payload = {
        "data_paths": {
            "forecast_data_path": forecast_data_path,
            "inventory_data_path": inventory_data_path,
            "policy_data_path": policy_data_path,
            "scenario_results_path": scenario_results_path,
            "order_data_path": order_data_path,
            "lead_time_data_path": lead_time_data_path
        }
    }
    
    # Add optional parameters
    if supplier_data_path:
        payload["data_paths"]["supplier_data_path"] = supplier_data_path
    
    if product_ids:
        payload["product_ids"] = product_ids
    
    # Make API request
    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/dashboard",
            json=payload,
            timeout=60  # Give more time for dashboard generation
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            print(f"Dashboard generated successfully")
            
            if "dashboard_url" in result:
                print(f"Dashboard URL: {API_BASE_URL}{result['dashboard_url']}")
                # Open dashboard in browser
                webbrowser.open(f"{API_BASE_URL}{result['dashboard_url']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Inventory Optimization Reporting Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main arguments
    parser.add_argument("--report-type", "-t", choices=[
        "inventory", "forecast", "policy", "supply-chain", "dashboard"
    ], help="Type of report to generate")
    
    # Data paths
    parser.add_argument("--inventory-data", help="Path to inventory data CSV")
    parser.add_argument("--sales-data", help="Path to sales data CSV")
    parser.add_argument("--forecast-data", help="Path to forecast data CSV")
    parser.add_argument("--historical-data", help="Path to historical data CSV")
    parser.add_argument("--policy-data", help="Path to policy data CSV")
    parser.add_argument("--scenario-results", help="Path to scenario results CSV")
    parser.add_argument("--order-data", help="Path to order data CSV")
    parser.add_argument("--lead-time-data", help="Path to lead time data CSV")
    parser.add_argument("--supplier-data", help="Path to supplier data CSV")
    
    # Other options
    parser.add_argument("--product-ids", help="Comma-separated list of product IDs")
    parser.add_argument("--format", default="html", choices=["html", "markdown", "csv"], 
                      help="Output format (default: html)")
    
    return parser.parse_args()


def show_interactive_menu() -> str:
    """
    Show an interactive menu for selecting report type.
    
    Returns:
        str: Selected report type
    """
    print("\n" + "=" * 40)
    print(" Inventory Optimization Reporting Tool ")
    print("=" * 40 + "\n")
    
    print("Please select a report type:")
    print("1. Inventory Status Report")
    print("2. Forecast Report")
    print("3. Policy Evaluation Report")
    print("4. Supply Chain Performance Report")
    print("5. Interactive Dashboard")
    
    choice = input("\nEnter choice (1-5): ")
    
    choices = {
        "1": "inventory",
        "2": "forecast",
        "3": "policy",
        "4": "supply-chain",
        "5": "dashboard"
    }
    
    return choices.get(choice, "inventory")


def get_input(prompt: str, optional: bool = False) -> Optional[str]:
    """
    Get user input with validation.
    
    Args:
        prompt: Prompt text to display
        optional: Whether the input is optional
    
    Returns:
        Input value or None if optional and no input given
    """
    while True:
        value = input(prompt)
        
        if not value and optional:
            return None
        
        if not value:
            print("This field is required. Please try again.")
            continue
        
        if not os.path.exists(value):
            print(f"Error: File '{value}' not found. Please enter a valid file path.")
            continue
        
        return value


def main() -> None:
    """Main function."""
    # Check if the API is running
    if not check_api_connection():
        return
    
    # Ensure data directory exists
    ensure_data_directory()
    
    # Parse arguments
    args = parse_args()
    
    # Convert product_ids string to list if provided
    product_ids = None
    if args.product_ids:
        try:
            product_ids = [int(pid.strip()) for pid in args.product_ids.split(",")]
        except ValueError:
            print("Error: Product IDs must be integers. Example: --product-ids 101,102,103")
            return
    
    # No report type specified, show interactive menu
    if not args.report_type:
        report_type = show_interactive_menu()
    else:
        report_type = args.report_type
    
    # Generate the requested report
    try:
        if report_type == "inventory":
            inventory_data_path = args.inventory_data or get_input("Enter path to inventory data CSV: ")
            sales_data_path = args.sales_data or get_input("Enter path to sales data CSV: ")
            policy_data_path = args.policy_data or get_input("Enter path to policy data CSV (optional): ", optional=True)
            
            generate_inventory_status_report(
                inventory_data_path=inventory_data_path,
                sales_data_path=sales_data_path,
                policy_data_path=policy_data_path,
                product_ids=product_ids,
                output_format=args.format
            )
        
        elif report_type == "forecast":
            forecast_data_path = args.forecast_data or get_input("Enter path to forecast data CSV: ")
            historical_data_path = args.historical_data or get_input("Enter path to historical data CSV: ")
            
            generate_forecast_report(
                forecast_data_path=forecast_data_path,
                historical_data_path=historical_data_path,
                product_ids=product_ids,
                output_format=args.format
            )
        
        elif report_type == "policy":
            policy_data_path = args.policy_data or get_input("Enter path to policy data CSV: ")
            scenario_results_path = args.scenario_results or get_input("Enter path to scenario results CSV: ")
            inventory_data_path = args.inventory_data or get_input("Enter path to inventory data CSV: ")
            
            generate_policy_evaluation_report(
                policy_data_path=policy_data_path,
                scenario_results_path=scenario_results_path,
                inventory_data_path=inventory_data_path,
                product_ids=product_ids,
                output_format=args.format
            )
        
        elif report_type == "supply-chain":
            inventory_data_path = args.inventory_data or get_input("Enter path to inventory data CSV: ")
            order_data_path = args.order_data or get_input("Enter path to order data CSV: ")
            lead_time_data_path = args.lead_time_data or get_input("Enter path to lead time data CSV: ")
            supplier_data_path = args.supplier_data or get_input("Enter path to supplier data CSV (optional): ", optional=True)
            
            generate_supply_chain_report(
                inventory_data_path=inventory_data_path,
                order_data_path=order_data_path,
                lead_time_data_path=lead_time_data_path,
                supplier_data_path=supplier_data_path,
                product_ids=product_ids,
                output_format=args.format
            )
        
        elif report_type == "dashboard":
            forecast_data_path = args.forecast_data or get_input("Enter path to forecast data CSV: ")
            inventory_data_path = args.inventory_data or get_input("Enter path to inventory data CSV: ")
            policy_data_path = args.policy_data or get_input("Enter path to policy data CSV: ")
            scenario_results_path = args.scenario_results or get_input("Enter path to scenario results CSV: ")
            order_data_path = args.order_data or get_input("Enter path to order data CSV: ")
            lead_time_data_path = args.lead_time_data or get_input("Enter path to lead time data CSV: ")
            supplier_data_path = args.supplier_data or get_input("Enter path to supplier data CSV (optional): ", optional=True)
            
            generate_dashboard(
                forecast_data_path=forecast_data_path,
                inventory_data_path=inventory_data_path,
                policy_data_path=policy_data_path,
                scenario_results_path=scenario_results_path,
                order_data_path=order_data_path,
                lead_time_data_path=lead_time_data_path,
                supplier_data_path=supplier_data_path,
                product_ids=product_ids
            )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main() 