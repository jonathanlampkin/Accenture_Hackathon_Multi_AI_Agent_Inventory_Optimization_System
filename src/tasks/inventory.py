"""
Celery tasks for inventory optimization.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from src.models.database import get_db_context
from src.models.forecast import Forecast
from src.models.inventory import Product, Inventory, Location, InventoryTransaction
from src.tasks.celery_app import celery_app, with_logging
from src.utils.metrics import record_inventory_metrics

logger = logging.getLogger(__name__)

@celery_app.task(name="src.tasks.inventory.optimize_inventory_levels")
@with_logging
def optimize_inventory_levels(product_ids: Optional[List[int]] = None, service_level: float = 0.95) -> Dict[str, Any]:
    """Optimize inventory levels for products.
    
    Args:
        product_ids: List of product IDs to optimize (if None, all active products)
        service_level: Target service level (default: 0.95)
        
    Returns:
        Dict: Summary of optimization results
    """
    start_time = time.time()
    
    with get_db_context() as db:
        # Get products to optimize
        products_query = db.query(Product)
        if product_ids:
            products_query = products_query.filter(Product.id.in_(product_ids))
        else:
            products_query = products_query.filter(Product.is_active == True)
            
        products = products_query.all()
        
        if not products:
            return {
                "status": "error",
                "message": "No products found for optimization",
                "processing_time": time.time() - start_time,
            }
        
        # Get latest forecasts for each product
        optimization_results = []
        
        for product in products:
            # Get the latest forecast for this product
            forecast = db.query(Forecast).filter(
                Forecast.product_id == product.id
            ).order_by(Forecast.created_at.desc()).first()
            
            if not forecast:
                logger.warning(f"No forecast found for product {product.id}")
                continue
            
            # Get product lead time (default to 5 if not specified)
            lead_time = product.lead_time or 5
            
            # Calculate safety stock based on forecast uncertainty
            forecast_values = forecast.forecast_values
            if forecast.lower_bounds and forecast.upper_bounds:
                # Calculate standard deviation from confidence intervals
                forecast_std = [(u - l) / 3.92 for u, l in zip(forecast.upper_bounds, forecast.lower_bounds)]
                avg_std = sum(forecast_std) / len(forecast_std)
            else:
                # If no confidence intervals, estimate std as 20% of mean
                avg_std = 0.2 * (sum(forecast_values) / len(forecast_values))
            
            # Calculate safety stock based on service level and lead time
            z_score = _get_z_score(service_level)
            safety_stock = z_score * avg_std * (lead_time ** 0.5)
            
            # Calculate reorder point
            avg_daily_demand = sum(forecast_values) / len(forecast_values)
            reorder_point = avg_daily_demand * lead_time + safety_stock
            
            # Calculate economic order quantity
            annual_demand = avg_daily_demand * 365
            order_cost = product.order_cost or 100  # Default order cost
            unit_cost = product.unit_cost or 10  # Default unit cost
            holding_cost_rate = product.holding_cost_rate or 0.25  # Default holding cost rate
            
            eoq = (2 * annual_demand * order_cost / (holding_cost_rate * unit_cost)) ** 0.5
            
            # Calculate min/max levels
            min_level = safety_stock
            max_level = reorder_point + eoq
            
            # Create optimization result
            result = {
                "product_id": product.id,
                "product_name": product.name,
                "safety_stock": round(safety_stock, 2),
                "reorder_point": round(reorder_point, 2),
                "economic_order_quantity": round(eoq, 2),
                "min_level": round(min_level, 2),
                "max_level": round(max_level, 2),
                "lead_time": lead_time,
                "service_level": service_level,
                "average_daily_demand": round(avg_daily_demand, 2),
            }
            
            # Add to results
            optimization_results.append(result)
            
            # Record metrics
            record_inventory_metrics(
                product_id=str(product.id),
                safety_stock=safety_stock,
                reorder_point=reorder_point,
                eoq=eoq,
                service_level=service_level
            )
        
        return {
            "status": "success",
            "message": f"Inventory levels optimized for {len(optimization_results)} products",
            "product_count": len(optimization_results),
            "results": optimization_results,
            "processing_time": time.time() - start_time,
        }

@celery_app.task(name="src.tasks.inventory.generate_purchase_orders")
@with_logging
def generate_purchase_orders(optimization_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Generate purchase orders based on optimization results.
    
    Args:
        optimization_results: Results from optimize_inventory_levels (if None, run optimization)
        
    Returns:
        Dict: Summary of purchase orders generated
    """
    start_time = time.time()
    
    # If no optimization results provided, run optimization first
    if not optimization_results:
        logger.info("No optimization results provided, running optimization first")
        opt_result = optimize_inventory_levels()
        if opt_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Optimization failed: {opt_result['message']}",
                "processing_time": time.time() - start_time,
            }
        optimization_results = opt_result["results"]
    
    with get_db_context() as db:
        purchase_orders = []
        
        for result in optimization_results:
            product_id = result["product_id"]
            
            # Get current inventory level
            current_inventory = db.query(Inventory).filter(
                Inventory.product_id == product_id
            ).all()
            
            # Sum up inventory across all locations
            total_inventory = sum(inv.quantity for inv in current_inventory) if current_inventory else 0
            
            # Check if below reorder point
            if total_inventory < result["reorder_point"]:
                # Calculate order quantity
                order_quantity = result["economic_order_quantity"]
                
                # Adjust to not exceed max inventory level
                max_level = result["max_level"]
                if total_inventory + order_quantity > max_level:
                    order_quantity = max_level - total_inventory
                
                # Round to whole number
                order_quantity = max(0, round(order_quantity))
                
                # Only create order if quantity > 0
                if order_quantity > 0:
                    # Get product details
                    product = db.query(Product).filter(Product.id == product_id).first()
                    
                    # Create purchase order
                    order = {
                        "order_id": f"PO-{datetime.now().strftime('%Y%m%d')}-{product_id}",
                        "product_id": product_id,
                        "product_name": product.name if product else f"Product {product_id}",
                        "order_quantity": order_quantity,
                        "current_inventory": total_inventory,
                        "reorder_point": result["reorder_point"],
                        "lead_time": result["lead_time"],
                        "expected_delivery": (datetime.now() + timedelta(days=result["lead_time"])).strftime("%Y-%m-%d"),
                        "order_date": datetime.now().strftime("%Y-%m-%d"),
                    }
                    
                    purchase_orders.append(order)
        
        return {
            "status": "success",
            "message": f"Generated {len(purchase_orders)} purchase orders",
            "order_count": len(purchase_orders),
            "orders": purchase_orders,
            "processing_time": time.time() - start_time,
        }

@celery_app.task(name="src.tasks.inventory.update_inventory_levels")
@with_logging
def update_inventory_levels(location_id: Optional[int] = None) -> Dict[str, Any]:
    """Update inventory metrics and records.
    
    Args:
        location_id: Location ID to update (if None, update all locations)
        
    Returns:
        Dict: Summary of inventory update
    """
    start_time = time.time()
    
    with get_db_context() as db:
        # Get locations to update
        locations_query = db.query(Location)
        if location_id:
            locations_query = locations_query.filter(Location.id == location_id)
            
        locations = locations_query.all()
        
        if not locations:
            return {
                "status": "error",
                "message": "No locations found for updating inventory",
                "processing_time": time.time() - start_time,
            }
        
        # Update inventory for each location
        updates = []
        
        for location in locations:
            # Get inventory at this location
            inventory_items = db.query(Inventory).filter(
                Inventory.location_id == location.id
            ).all()
            
            for item in inventory_items:
                # Get product details
                product = db.query(Product).filter(Product.id == item.product_id).first()
                if not product:
                    continue
                
                # Get optimization results for this product
                optimization_result = _get_optimization_for_product(db, product.id)
                
                if not optimization_result:
                    continue
                
                # Check if inventory is below reorder point
                status = "OK"
                if item.quantity < optimization_result["reorder_point"]:
                    status = "BELOW_REORDER_POINT"
                if item.quantity < optimization_result["safety_stock"]:
                    status = "BELOW_SAFETY_STOCK"
                if item.quantity <= 0:
                    status = "STOCKOUT"
                
                # Update inventory status
                item.status = status
                
                # Get recent transactions
                recent_transactions = db.query(InventoryTransaction).filter(
                    InventoryTransaction.inventory_id == item.id,
                    InventoryTransaction.created_at >= datetime.now() - timedelta(days=30)
                ).all()
                
                # Calculate velocity metrics
                if recent_transactions:
                    # Calculate average daily demand
                    total_outbound = sum(t.quantity for t in recent_transactions if t.transaction_type == "outbound")
                    avg_daily_demand = total_outbound / 30
                    
                    # Update metrics
                    item.avg_daily_demand = avg_daily_demand
                    
                    # Calculate days of supply
                    if avg_daily_demand > 0:
                        item.days_of_supply = item.quantity / avg_daily_demand
                    else:
                        item.days_of_supply = 999  # Very high if no demand
                
                # Add to updates
                updates.append({
                    "product_id": item.product_id,
                    "product_name": product.name,
                    "location_id": location.id,
                    "location_name": location.name,
                    "quantity": item.quantity,
                    "status": status,
                    "avg_daily_demand": item.avg_daily_demand,
                    "days_of_supply": item.days_of_supply,
                })
            
        # Commit updates
        db.commit()
        
        return {
            "status": "success",
            "message": f"Updated inventory metrics for {len(updates)} items",
            "item_count": len(updates),
            "updates": updates,
            "processing_time": time.time() - start_time,
        }

def _get_z_score(service_level: float) -> float:
    """Get z-score for a given service level.
    
    Args:
        service_level: Desired service level (0-1)
        
    Returns:
        float: Z-score for normal distribution
    """
    # Common z-scores for service levels
    z_table = {
        0.50: 0.00,
        0.75: 0.67,
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.96: 1.75,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
        0.995: 2.58,
        0.999: 3.08,
    }
    
    # Find closest service level in the table
    service_levels = list(z_table.keys())
    closest_level = min(service_levels, key=lambda x: abs(x - service_level))
    
    return z_table[closest_level]

def _get_optimization_for_product(db: Session, product_id: int) -> Optional[Dict[str, Any]]:
    """Get latest optimization result for a product.
    
    This is a placeholder function. In a real implementation, you would retrieve
    the latest optimization result from a database.
    
    Args:
        db: Database session
        product_id: Product ID
        
    Returns:
        Optional[Dict]: Optimization result or None if not found
    """
    # Get the latest forecast for this product
    forecast = db.query(Forecast).filter(
        Forecast.product_id == product_id
    ).order_by(Forecast.created_at.desc()).first()
    
    if not forecast:
        return None
    
    # Get product details
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        return None
    
    # Get product lead time (default to 5 if not specified)
    lead_time = product.lead_time or 5
    service_level = 0.95
    
    # Calculate safety stock based on forecast uncertainty
    forecast_values = forecast.forecast_values
    if forecast.lower_bounds and forecast.upper_bounds:
        # Calculate standard deviation from confidence intervals
        forecast_std = [(u - l) / 3.92 for u, l in zip(forecast.upper_bounds, forecast.lower_bounds)]
        avg_std = sum(forecast_std) / len(forecast_std)
    else:
        # If no confidence intervals, estimate std as 20% of mean
        avg_std = 0.2 * (sum(forecast_values) / len(forecast_values))
    
    # Calculate safety stock based on service level and lead time
    z_score = _get_z_score(service_level)
    safety_stock = z_score * avg_std * (lead_time ** 0.5)
    
    # Calculate reorder point
    avg_daily_demand = sum(forecast_values) / len(forecast_values)
    reorder_point = avg_daily_demand * lead_time + safety_stock
    
    # Calculate economic order quantity
    annual_demand = avg_daily_demand * 365
    order_cost = product.order_cost or 100  # Default order cost
    unit_cost = product.unit_cost or 10  # Default unit cost
    holding_cost_rate = product.holding_cost_rate or 0.25  # Default holding cost rate
    
    eoq = (2 * annual_demand * order_cost / (holding_cost_rate * unit_cost)) ** 0.5
    
    # Calculate min/max levels
    min_level = safety_stock
    max_level = reorder_point + eoq
    
    # Create optimization result
    return {
        "product_id": product.id,
        "product_name": product.name,
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "economic_order_quantity": round(eoq, 2),
        "min_level": round(min_level, 2),
        "max_level": round(max_level, 2),
        "lead_time": lead_time,
        "service_level": service_level,
        "average_daily_demand": round(avg_daily_demand, 2),
    } 