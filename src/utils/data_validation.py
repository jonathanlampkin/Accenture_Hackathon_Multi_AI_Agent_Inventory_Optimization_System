"""
Data validation utilities using Great Expectations.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

logger = logging.getLogger(__name__)

# Directory for storing expectation suites
EXPECTATIONS_DIR = os.environ.get("EXPECTATIONS_DIR", "expectations")
os.makedirs(EXPECTATIONS_DIR, exist_ok=True)

class DataValidator:
    """Data validator using Great Expectations."""
    
    @staticmethod
    def validate_inventory_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate inventory data.
        
        Args:
            df: Inventory data
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Validation result and report
        """
        # Create suite
        suite = DataValidator._create_inventory_suite()
        
        # Validate data
        return DataValidator._validate_data(df, suite)
        
    @staticmethod
    def validate_demand_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate demand data.
        
        Args:
            df: Demand data
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Validation result and report
        """
        # Create suite
        suite = DataValidator._create_demand_suite()
        
        # Validate data
        return DataValidator._validate_data(df, suite)
        
    @staticmethod
    def validate_product_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate product data.
        
        Args:
            df: Product data
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Validation result and report
        """
        # Create suite
        suite = DataValidator._create_product_suite()
        
        # Validate data
        return DataValidator._validate_data(df, suite)
        
    @staticmethod
    def validate_location_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate location data.
        
        Args:
            df: Location data
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Validation result and report
        """
        # Create suite
        suite = DataValidator._create_location_suite()
        
        # Validate data
        return DataValidator._validate_data(df, suite)
        
    @staticmethod
    def _validate_data(df: pd.DataFrame, suite: ExpectationSuite) -> Tuple[bool, Dict[str, Any]]:
        """Validate data against an expectation suite.
        
        Args:
            df: Data to validate
            suite: Expectation suite
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Validation result and report
        """
        # Convert to Great Expectations dataset
        ge_df = ge.from_pandas(df)
        
        # Apply suite
        ge_df.expectation_suite = suite
        
        # Validate
        result = ge_df.validate()
        
        # Check if validation passed
        success = result.success
        
        # Create report
        report = {
            "success": success,
            "statistics": {
                "evaluated_expectations": result.statistics["evaluated_expectations"],
                "successful_expectations": result.statistics["successful_expectations"],
                "unsuccessful_expectations": result.statistics["unsuccessful_expectations"],
            },
            "results": [],
        }
        
        # Add detailed results
        for res in result.results:
            if not res.success:
                report["results"].append({
                    "expectation_type": res.expectation_config.expectation_type,
                    "kwargs": res.expectation_config.kwargs,
                    "success": res.success,
                    "exception_info": res.exception_info if hasattr(res, "exception_info") else None,
                })
                
        return success, report
        
    @staticmethod
    def _create_inventory_suite() -> ExpectationSuite:
        """Create expectation suite for inventory data.
        
        Returns:
            ExpectationSuite: Expectation suite
        """
        # Load suite if it exists
        suite_path = os.path.join(EXPECTATIONS_DIR, "inventory_suite.json")
        if os.path.exists(suite_path):
            with open(suite_path, "r") as f:
                suite_dict = json.load(f)
                return ExpectationSuite.from_dict(suite_dict)
                
        # Create new suite
        suite = ExpectationSuite(expectation_suite_name="inventory_suite")
        
        # Create dataset with empty data to build expectations
        ge_df = ge.dataset.PandasDataset(
            pd.DataFrame(columns=["product_id", "location_id", "quantity", "reserved_quantity"]),
            expectation_suite=suite,
        )
        
        # Define expectations
        ge_df.expect_table_columns_to_match_ordered_list([
            "product_id", "location_id", "quantity", "reserved_quantity"
        ])
        ge_df.expect_column_values_to_not_be_null("product_id")
        ge_df.expect_column_values_to_not_be_null("location_id")
        ge_df.expect_column_values_to_not_be_null("quantity")
        ge_df.expect_column_values_to_be_between("quantity", min_value=0, max_value=None)
        ge_df.expect_column_values_to_be_between("reserved_quantity", min_value=0, max_value=None)
        
        # Save suite
        with open(suite_path, "w") as f:
            json.dump(suite.to_dict(), f)
            
        return suite
        
    @staticmethod
    def _create_demand_suite() -> ExpectationSuite:
        """Create expectation suite for demand data.
        
        Returns:
            ExpectationSuite: Expectation suite
        """
        # Load suite if it exists
        suite_path = os.path.join(EXPECTATIONS_DIR, "demand_suite.json")
        if os.path.exists(suite_path):
            with open(suite_path, "r") as f:
                suite_dict = json.load(f)
                return ExpectationSuite.from_dict(suite_dict)
                
        # Create new suite
        suite = ExpectationSuite(expectation_suite_name="demand_suite")
        
        # Create dataset with empty data to build expectations
        ge_df = ge.dataset.PandasDataset(
            pd.DataFrame(columns=["Date", "Product ID", "Sales Quantity"]),
            expectation_suite=suite,
        )
        
        # Define expectations
        ge_df.expect_table_columns_to_match_ordered_list([
            "Date", "Product ID", "Sales Quantity"
        ])
        ge_df.expect_column_values_to_not_be_null("Date")
        ge_df.expect_column_values_to_not_be_null("Product ID")
        ge_df.expect_column_values_to_not_be_null("Sales Quantity")
        ge_df.expect_column_values_to_be_between("Sales Quantity", min_value=0, max_value=None)
        ge_df.expect_column_values_to_be_of_type("Date", "datetime64")
        
        # Save suite
        with open(suite_path, "w") as f:
            json.dump(suite.to_dict(), f)
            
        return suite
        
    @staticmethod
    def _create_product_suite() -> ExpectationSuite:
        """Create expectation suite for product data.
        
        Returns:
            ExpectationSuite: Expectation suite
        """
        # Load suite if it exists
        suite_path = os.path.join(EXPECTATIONS_DIR, "product_suite.json")
        if os.path.exists(suite_path):
            with open(suite_path, "r") as f:
                suite_dict = json.load(f)
                return ExpectationSuite.from_dict(suite_dict)
                
        # Create new suite
        suite = ExpectationSuite(expectation_suite_name="product_suite")
        
        # Create dataset with empty data to build expectations
        ge_df = ge.dataset.PandasDataset(
            pd.DataFrame(columns=["id", "sku", "name", "price", "cost"]),
            expectation_suite=suite,
        )
        
        # Define expectations
        ge_df.expect_table_columns_to_match_ordered_list([
            "id", "sku", "name", "price", "cost"
        ])
        ge_df.expect_column_values_to_not_be_null("id")
        ge_df.expect_column_values_to_not_be_null("sku")
        ge_df.expect_column_values_to_not_be_null("name")
        ge_df.expect_column_values_to_be_between("price", min_value=0, max_value=None)
        ge_df.expect_column_values_to_be_between("cost", min_value=0, max_value=None)
        
        # Save suite
        with open(suite_path, "w") as f:
            json.dump(suite.to_dict(), f)
            
        return suite
        
    @staticmethod
    def _create_location_suite() -> ExpectationSuite:
        """Create expectation suite for location data.
        
        Returns:
            ExpectationSuite: Expectation suite
        """
        # Load suite if it exists
        suite_path = os.path.join(EXPECTATIONS_DIR, "location_suite.json")
        if os.path.exists(suite_path):
            with open(suite_path, "r") as f:
                suite_dict = json.load(f)
                return ExpectationSuite.from_dict(suite_dict)
                
        # Create new suite
        suite = ExpectationSuite(expectation_suite_name="location_suite")
        
        # Create dataset with empty data to build expectations
        ge_df = ge.dataset.PandasDataset(
            pd.DataFrame(columns=["id", "code", "name", "is_warehouse", "is_store"]),
            expectation_suite=suite,
        )
        
        # Define expectations
        ge_df.expect_table_columns_to_match_ordered_list([
            "id", "code", "name", "is_warehouse", "is_store"
        ])
        ge_df.expect_column_values_to_not_be_null("id")
        ge_df.expect_column_values_to_not_be_null("code")
        ge_df.expect_column_values_to_not_be_null("name")
        ge_df.expect_column_values_to_be_in_set("is_warehouse", [True, False])
        ge_df.expect_column_values_to_be_in_set("is_store", [True, False])
        
        # Save suite
        with open(suite_path, "w") as f:
            json.dump(suite.to_dict(), f)
            
        return suite 