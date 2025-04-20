"""
Tools for inventory optimization operations used by the optimization agent.

This module contains tools for calculating economic order quantity, reorder point,
safety stock, and defining inventory policies.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import math


class CalculateEconomicOrderQuantityTool(BaseTool):
    """Tool for calculating economic order quantity (EOQ)."""
    
    name: str = "Calculate Economic Order Quantity"
    description: str = """
    Calculate the economic order quantity (EOQ) for specified products.
    
    Input should include:
    - demand_data_path: Path to the demand data CSV file
    - cost_data_path: Path to the cost data CSV file containing ordering cost and holding cost
    - product_ids: Optional list of product IDs to calculate EOQ for (None for all products)
    - output_path: Optional path to save the EOQ calculations
    """
    
    class InputSchema(BaseModel):
        demand_data_path: str = Field(
            ..., 
            description="Path to the demand data CSV file"
        )
        cost_data_path: str = Field(
            ..., 
            description="Path to the cost data CSV file containing ordering cost and holding cost"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to calculate EOQ for (None for all products)"
        )
        output_path: Optional[str] = Field(
            None, 
            description="Optional path to save the EOQ calculations"
        )
    
    def run(self, demand_data_path: str,
            cost_data_path: str,
            product_ids: Optional[List[int]] = None,
            output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the economic order quantity (EOQ) for specified products.
        
        Args:
            demand_data_path: Path to the demand data CSV file
            cost_data_path: Path to the cost data CSV file
            product_ids: Optional list of product IDs to calculate EOQ for
            output_path: Optional path to save the EOQ calculations
            
        Returns:
            Dict containing EOQ calculations
        """
        try:
            # Load demand data
            demand_data = pd.read_csv(demand_data_path)
            
            # Load cost data
            cost_data = pd.read_csv(cost_data_path)
            
            # Calculate annual demand by product
            annual_demand = demand_data.groupby('Product ID')['Sales Quantity'].sum().reset_index()
            annual_demand.rename(columns={'Sales Quantity': 'Annual Demand'}, inplace=True)
            
            # Merge with cost data
            merged_data = pd.merge(annual_demand, cost_data, on='Product ID', how='inner')
            
            # Filter for specific products if provided
            if product_ids:
                merged_data = merged_data[merged_data['Product ID'].isin(product_ids)]
            
            # Calculate EOQ
            # EOQ = sqrt(2 * D * S / H)
            # where D = annual demand, S = ordering cost, H = holding cost
            merged_data['EOQ'] = np.sqrt(
                (2 * merged_data['Annual Demand'] * merged_data['Ordering Cost']) / 
                merged_data['Holding Cost']
            )
            
            # Round to nearest integer
            merged_data['EOQ'] = merged_data['EOQ'].round().astype(int)
            
            # Calculate number of orders per year
            merged_data['Orders Per Year'] = merged_data['Annual Demand'] / merged_data['EOQ']
            merged_data['Orders Per Year'] = merged_data['Orders Per Year'].round(2)
            
            # Calculate order cycle time in days
            merged_data['Order Cycle (days)'] = 365 / merged_data['Orders Per Year']
            merged_data['Order Cycle (days)'] = merged_data['Order Cycle (days)'].round(1)
            
            # Calculate total annual ordering cost
            merged_data['Annual Ordering Cost'] = merged_data['Orders Per Year'] * merged_data['Ordering Cost']
            
            # Calculate average inventory
            merged_data['Average Inventory'] = merged_data['EOQ'] / 2
            
            # Calculate annual holding cost
            merged_data['Annual Holding Cost'] = merged_data['Average Inventory'] * merged_data['Holding Cost']
            
            # Calculate total inventory cost
            merged_data['Total Inventory Cost'] = merged_data['Annual Ordering Cost'] + merged_data['Annual Holding Cost']
            
            # Save results if output path provided
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                merged_data.to_csv(output_path, index=False)
            
            # Prepare results dictionary
            result = {
                "message": f"EOQ calculated for {len(merged_data)} products",
                "results": merged_data.to_dict(orient='records')
            }
            
            if output_path:
                result["output_path"] = output_path
                
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to calculate economic order quantity"
            }


class CalculateReorderPointTool(BaseTool):
    """Tool for calculating reorder point (ROP)."""
    
    name: str = "Calculate Reorder Point"
    description: str = """
    Calculate the reorder point (ROP) for specified products.
    
    Input should include:
    - demand_data_path: Path to the demand data CSV file
    - lead_time_data_path: Path to the lead time data CSV file
    - service_level: Desired service level (default: 0.95)
    - product_ids: Optional list of product IDs to calculate ROP for (None for all products)
    - output_path: Optional path to save the ROP calculations
    """
    
    class InputSchema(BaseModel):
        demand_data_path: str = Field(
            ..., 
            description="Path to the demand data CSV file"
        )
        lead_time_data_path: str = Field(
            ..., 
            description="Path to the lead time data CSV file"
        )
        service_level: float = Field(
            0.95, 
            description="Desired service level (between 0 and 1)"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to calculate ROP for (None for all products)"
        )
        output_path: Optional[str] = Field(
            None, 
            description="Optional path to save the ROP calculations"
        )
    
    def run(self, demand_data_path: str,
            lead_time_data_path: str,
            service_level: float = 0.95,
            product_ids: Optional[List[int]] = None,
            output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the reorder point (ROP) for specified products.
        
        Args:
            demand_data_path: Path to the demand data CSV file
            lead_time_data_path: Path to the lead time data CSV file
            service_level: Desired service level (default: 0.95)
            product_ids: Optional list of product IDs to calculate ROP for
            output_path: Optional path to save the ROP calculations
            
        Returns:
            Dict containing ROP calculations
        """
        try:
            # Load demand data
            demand_data = pd.read_csv(demand_data_path)
            
            # Load lead time data
            lead_time_data = pd.read_csv(lead_time_data_path)
            
            # Calculate daily demand mean and std by product
            demand_data['Date'] = pd.to_datetime(demand_data['Date'])
            daily_demand = demand_data.groupby(['Product ID', 'Date'])['Sales Quantity'].sum().reset_index()
            
            demand_stats = daily_demand.groupby('Product ID').agg(
                mean_daily_demand=('Sales Quantity', 'mean'),
                std_daily_demand=('Sales Quantity', 'std')
            ).reset_index()
            
            # Merge with lead time data
            merged_data = pd.merge(demand_stats, lead_time_data, on='Product ID', how='inner')
            
            # Filter for specific products if provided
            if product_ids:
                merged_data = merged_data[merged_data['Product ID'].isin(product_ids)]
            
            # Calculate safety factor based on service level
            # Using the inverse of the standard normal cumulative distribution
            from scipy.stats import norm
            safety_factor = norm.ppf(service_level)
            
            # Calculate lead time demand
            merged_data['Lead Time Demand'] = merged_data['mean_daily_demand'] * merged_data['Reorder Lead Time']
            
            # Calculate lead time demand standard deviation
            merged_data['Lead Time Demand Std'] = merged_data['std_daily_demand'] * np.sqrt(merged_data['Reorder Lead Time'])
            
            # Calculate safety stock
            merged_data['Safety Stock'] = safety_factor * merged_data['Lead Time Demand Std']
            
            # Calculate reorder point
            merged_data['Reorder Point'] = merged_data['Lead Time Demand'] + merged_data['Safety Stock']
            
            # Round to nearest integer
            merged_data['Safety Stock'] = merged_data['Safety Stock'].round().astype(int)
            merged_data['Reorder Point'] = merged_data['Reorder Point'].round().astype(int)
            
            # Save results if output path provided
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                merged_data.to_csv(output_path, index=False)
            
            # Prepare results dictionary
            result = {
                "message": f"Reorder point calculated for {len(merged_data)} products with {service_level:.0%} service level",
                "results": merged_data.to_dict(orient='records')
            }
            
            if output_path:
                result["output_path"] = output_path
                
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to calculate reorder point"
            }


class CalculateSafetyStockTool(BaseTool):
    """Tool for calculating safety stock levels."""
    
    name: str = "Calculate Safety Stock"
    description: str = """
    Calculate the safety stock levels for specified products.
    
    Input should include:
    - demand_data_path: Path to the demand data CSV file
    - lead_time_data_path: Path to the lead time data CSV file
    - service_level: Desired service level (default: 0.95)
    - product_ids: Optional list of product IDs to calculate safety stock for (None for all products)
    - output_path: Optional path to save the safety stock calculations
    """
    
    class InputSchema(BaseModel):
        demand_data_path: str = Field(
            ..., 
            description="Path to the demand data CSV file"
        )
        lead_time_data_path: str = Field(
            ..., 
            description="Path to the lead time data CSV file"
        )
        service_level: float = Field(
            0.95, 
            description="Desired service level (between 0 and 1)"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to calculate safety stock for (None for all products)"
        )
        output_path: Optional[str] = Field(
            None, 
            description="Optional path to save the safety stock calculations"
        )
    
    def run(self, demand_data_path: str,
            lead_time_data_path: str,
            service_level: float = 0.95,
            product_ids: Optional[List[int]] = None,
            output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the safety stock levels for specified products.
        
        Args:
            demand_data_path: Path to the demand data CSV file
            lead_time_data_path: Path to the lead time data CSV file
            service_level: Desired service level (default: 0.95)
            product_ids: Optional list of product IDs to calculate safety stock for
            output_path: Optional path to save the safety stock calculations
            
        Returns:
            Dict containing safety stock calculations
        """
        try:
            # Load demand data
            demand_data = pd.read_csv(demand_data_path)
            
            # Load lead time data
            lead_time_data = pd.read_csv(lead_time_data_path)
            
            # Calculate daily demand mean and std by product
            demand_data['Date'] = pd.to_datetime(demand_data['Date'])
            daily_demand = demand_data.groupby(['Product ID', 'Date'])['Sales Quantity'].sum().reset_index()
            
            demand_stats = daily_demand.groupby('Product ID').agg(
                mean_daily_demand=('Sales Quantity', 'mean'),
                std_daily_demand=('Sales Quantity', 'std')
            ).reset_index()
            
            # Merge with lead time data
            merged_data = pd.merge(demand_stats, lead_time_data, on='Product ID', how='inner')
            
            # Also calculate lead time statistics if available
            if 'Lead Time Std' in lead_time_data.columns:
                lead_time_std = True
            else:
                lead_time_std = False
                merged_data['Lead Time Std'] = 0
            
            # Filter for specific products if provided
            if product_ids:
                merged_data = merged_data[merged_data['Product ID'].isin(product_ids)]
            
            # Calculate safety factor based on service level
            # Using the inverse of the standard normal cumulative distribution
            from scipy.stats import norm
            safety_factor = norm.ppf(service_level)
            
            # Calculate safety stock
            # If we have lead time variability:
            # SS = Z * sqrt(L * σ_d^2 + d^2 * σ_L^2)
            # where Z = safety factor, L = lead time, d = average daily demand
            # σ_d = standard deviation of daily demand, σ_L = standard deviation of lead time
            
            if lead_time_std:
                merged_data['Safety Stock'] = safety_factor * np.sqrt(
                    merged_data['Reorder Lead Time'] * merged_data['std_daily_demand']**2 +
                    merged_data['mean_daily_demand']**2 * merged_data['Lead Time Std']**2
                )
            else:
                # If no lead time variability, use simpler formula
                # SS = Z * σ_d * sqrt(L)
                merged_data['Safety Stock'] = safety_factor * merged_data['std_daily_demand'] * np.sqrt(merged_data['Reorder Lead Time'])
            
            # Round to nearest integer
            merged_data['Safety Stock'] = merged_data['Safety Stock'].round().astype(int)
            
            # Add service level to output
            merged_data['Service Level'] = service_level
            
            # Calculate expected stockouts per year
            days_per_year = 365
            merged_data['Expected Stockouts Per Year'] = (1 - service_level) * (days_per_year / merged_data['Reorder Lead Time'])
            merged_data['Expected Stockouts Per Year'] = merged_data['Expected Stockouts Per Year'].round(2)
            
            # Save results if output path provided
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                merged_data.to_csv(output_path, index=False)
            
            # Prepare results dictionary
            result = {
                "message": f"Safety stock calculated for {len(merged_data)} products with {service_level:.0%} service level",
                "results": merged_data.to_dict(orient='records')
            }
            
            if output_path:
                result["output_path"] = output_path
                
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to calculate safety stock"
            }


class DefineInventoryPolicyTool(BaseTool):
    """Tool for defining inventory policies based on optimization results."""
    
    name: str = "Define Inventory Policy"
    description: str = """
    Define inventory policies by combining EOQ, ROP, and safety stock calculations.
    
    Input should include:
    - eoq_data_path: Path to the EOQ calculation results
    - rop_data_path: Path to the ROP calculation results
    - safety_stock_data_path: Path to the safety stock calculation results
    - product_ids: Optional list of product IDs to define policies for (None for all products)
    - output_path: Path to save the inventory policy definitions
    """
    
    class InputSchema(BaseModel):
        eoq_data_path: str = Field(
            ..., 
            description="Path to the EOQ calculation results"
        )
        rop_data_path: str = Field(
            ..., 
            description="Path to the ROP calculation results"
        )
        safety_stock_data_path: str = Field(
            ..., 
            description="Path to the safety stock calculation results"
        )
        product_ids: Optional[List[int]] = Field(
            None, 
            description="Optional list of product IDs to define policies for (None for all products)"
        )
        output_path: str = Field(
            ..., 
            description="Path to save the inventory policy definitions"
        )
    
    def run(self, eoq_data_path: str,
            rop_data_path: str,
            safety_stock_data_path: str,
            product_ids: Optional[List[int]] = None,
            output_path: str = "output/inventory_policies.csv") -> Dict[str, Any]:
        """
        Define inventory policies by combining EOQ, ROP, and safety stock calculations.
        
        Args:
            eoq_data_path: Path to the EOQ calculation results
            rop_data_path: Path to the ROP calculation results
            safety_stock_data_path: Path to the safety stock calculation results
            product_ids: Optional list of product IDs to define policies for
            output_path: Path to save the inventory policy definitions
            
        Returns:
            Dict containing inventory policy definitions
        """
        try:
            # Load EOQ data
            eoq_data = pd.read_csv(eoq_data_path)
            
            # Load ROP data
            rop_data = pd.read_csv(rop_data_path)
            
            # Load safety stock data
            safety_stock_data = pd.read_csv(safety_stock_data_path)
            
            # Merge data
            merged_data = pd.merge(eoq_data[['Product ID', 'EOQ', 'Order Cycle (days)']], 
                                   rop_data[['Product ID', 'Reorder Point']], 
                                   on='Product ID', how='inner')
            
            merged_data = pd.merge(merged_data, 
                                   safety_stock_data[['Product ID', 'Safety Stock', 'Service Level']], 
                                   on='Product ID', how='inner')
            
            # Filter for specific products if provided
            if product_ids:
                merged_data = merged_data[merged_data['Product ID'].isin(product_ids)]
            
            # Add additional policy parameters
            # Calculate maximum inventory level
            merged_data['Maximum Inventory Level'] = merged_data['Reorder Point'] + merged_data['EOQ']
            
            # Determine policy type
            merged_data['Policy Type'] = 'Continuous Review (s,Q)'
            
            # Calculate inventory turns
            if 'Annual Demand' in eoq_data.columns:
                annual_demand = eoq_data[['Product ID', 'Annual Demand']]
                merged_data = pd.merge(merged_data, annual_demand, on='Product ID', how='left')
                merged_data['Average Inventory'] = merged_data['Safety Stock'] + (merged_data['EOQ'] / 2)
                merged_data['Inventory Turns'] = merged_data['Annual Demand'] / merged_data['Average Inventory']
                merged_data['Inventory Turns'] = merged_data['Inventory Turns'].round(2)
            
            # Save results
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            merged_data.to_csv(output_path, index=False)
            
            # Prepare results dictionary
            return {
                "message": f"Inventory policies defined for {len(merged_data)} products",
                "results": merged_data.to_dict(orient='records'),
                "output_path": output_path
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to define inventory policies"
            } 