"""
Inventory Optimization Models

This module contains advanced inventory optimization models for calculating:
- Min/Max inventory levels
- Reorder points 
- Safety stock levels
- Economic order quantities
- Storage constraints
- Product expiry considerations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class InventoryOptimizer:
    """
    Advanced inventory optimization model for calculating optimal inventory levels,
    reorder points, safety stock levels, and order quantities.
    """
    
    def __init__(self, 
                service_level: float = 0.95, 
                holding_cost_rate: float = 0.25,
                ordering_cost: float = 50.0):
        """
        Initialize the inventory optimizer.
        
        Args:
            service_level: Desired service level (default: 95%)
            holding_cost_rate: Annual holding cost as percentage of product value
            ordering_cost: Fixed cost per order
        """
        self.service_level = service_level
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost
        
        # Service level to z-score mapping
        self.z_score = self._service_level_to_z_score(service_level)
        
        logger.info(f"Initialized InventoryOptimizer with service level: {service_level}, z-score: {self.z_score}")
    
    def _service_level_to_z_score(self, service_level: float) -> float:
        """
        Convert service level to z-score.
        
        Common service levels:
        - 90% -> 1.28
        - 95% -> 1.65
        - 98% -> 2.05
        - 99% -> 2.33
        
        Args:
            service_level: Desired service level (0.0-1.0)
            
        Returns:
            Corresponding z-score
        """
        # Simple mapping for common service levels
        z_scores = {
            0.50: 0.00,
            0.75: 0.67,
            0.80: 0.84,
            0.85: 1.04,
            0.90: 1.28,
            0.95: 1.65,
            0.98: 2.05,
            0.99: 2.33,
            0.999: 3.09
        }
        
        # Find closest service level
        closest_level = min(z_scores.keys(), key=lambda x: abs(x - service_level))
        return z_scores[closest_level]
    
    def calculate_safety_stock(self, 
                              avg_demand: float, 
                              demand_std: float, 
                              lead_time: float, 
                              lead_time_std: Optional[float] = None,
                              service_level: Optional[float] = None) -> float:
        """
        Calculate safety stock considering demand and lead time variability.
        
        The formula accounts for both demand and lead time variability:
        Safety Stock = Z × √(L × σ_d² + D² × σ_l²)
        
        Where:
        - Z: Z-score for service level
        - L: Average lead time
        - σ_d: Standard deviation of daily demand
        - D: Average daily demand
        - σ_l: Standard deviation of lead time
        
        Args:
            avg_demand: Average demand per time unit
            demand_std: Standard deviation of demand
            lead_time: Average lead time
            lead_time_std: Standard deviation of lead time (if None, only demand variability is considered)
            service_level: Service level override (uses instance default if None)
            
        Returns:
            Safety stock level
        """
        z = self._service_level_to_z_score(service_level) if service_level else self.z_score
        
        if lead_time_std is None or lead_time_std == 0:
            # Only consider demand variability
            safety_stock = z * demand_std * np.sqrt(lead_time)
        else:
            # Consider both demand and lead time variability
            safety_stock = z * np.sqrt((lead_time * demand_std**2) + (avg_demand**2 * lead_time_std**2))
        
        return max(0, safety_stock)
    
    def calculate_reorder_point(self,
                               avg_demand: float,
                               lead_time: float,
                               safety_stock: Optional[float] = None,
                               demand_std: Optional[float] = None,
                               lead_time_std: Optional[float] = None,
                               service_level: Optional[float] = None) -> float:
        """
        Calculate reorder point.
        
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        
        Args:
            avg_demand: Average demand per time unit
            lead_time: Average lead time
            safety_stock: Pre-calculated safety stock (if None, it will be calculated)
            demand_std: Standard deviation of demand (needed if safety_stock is None)
            lead_time_std: Standard deviation of lead time (optional)
            service_level: Service level override (uses instance default if None)
            
        Returns:
            Reorder point
        """
        # Calculate safety stock if not provided
        if safety_stock is None:
            if demand_std is None:
                raise ValueError("Either safety_stock or demand_std must be provided")
            
            safety_stock = self.calculate_safety_stock(
                avg_demand=avg_demand,
                demand_std=demand_std,
                lead_time=lead_time,
                lead_time_std=lead_time_std,
                service_level=service_level
            )
        
        # Calculate reorder point
        rop = (avg_demand * lead_time) + safety_stock
        
        return max(0, rop)
    
    def calculate_economic_order_quantity(self,
                                         annual_demand: float,
                                         unit_cost: float,
                                         holding_cost_rate: Optional[float] = None,
                                         ordering_cost: Optional[float] = None) -> float:
        """
        Calculate Economic Order Quantity (EOQ).
        
        EOQ = √(2 × Annual Demand × Ordering Cost / (Unit Cost × Holding Cost Rate))
        
        Args:
            annual_demand: Annual demand quantity
            unit_cost: Cost per unit
            holding_cost_rate: Annual holding cost as percentage of product value
            ordering_cost: Fixed cost per order
            
        Returns:
            Economic Order Quantity
        """
        h_rate = holding_cost_rate if holding_cost_rate is not None else self.holding_cost_rate
        o_cost = ordering_cost if ordering_cost is not None else self.ordering_cost
        
        # Calculate annual holding cost per unit
        holding_cost = unit_cost * h_rate
        
        # Calculate EOQ
        if holding_cost <= 0 or annual_demand <= 0 or o_cost <= 0:
            return 0
            
        eoq = np.sqrt((2 * annual_demand * o_cost) / holding_cost)
        
        return max(1, eoq)
    
    def calculate_min_max_levels(self,
                                avg_demand: float,
                                lead_time: float,
                                review_period: float,
                                demand_std: float,
                                lead_time_std: Optional[float] = None,
                                service_level: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate Min/Max inventory levels.
        
        Min Level = Reorder Point
        Max Level = Reorder Point + Economic Order Quantity
        
        Args:
            avg_demand: Average demand per time unit
            lead_time: Average lead time
            review_period: Inventory review period
            demand_std: Standard deviation of demand
            lead_time_std: Standard deviation of lead time (optional)
            service_level: Service level override (uses instance default if None)
            
        Returns:
            Dictionary with min and max levels
        """
        z = self._service_level_to_z_score(service_level) if service_level else self.z_score
        
        # Calculate safety stock
        safety_stock = self.calculate_safety_stock(
            avg_demand=avg_demand,
            demand_std=demand_std,
            lead_time=lead_time,
            lead_time_std=lead_time_std,
            service_level=service_level
        )
        
        # Min level (reorder point)
        min_level = self.calculate_reorder_point(
            avg_demand=avg_demand,
            lead_time=lead_time,
            safety_stock=safety_stock
        )
        
        # Max level (min level + cycle stock)
        # For a periodic review system, we need to cover demand during lead time + review period
        max_level = min_level + (avg_demand * review_period)
        
        # Consider lead time & review period uncertainty for max level
        if lead_time_std is not None and lead_time_std > 0:
            max_level += z * np.sqrt(((lead_time + review_period) * demand_std**2) + 
                                     (avg_demand**2 * lead_time_std**2))
        
        return {
            "min_level": min_level,
            "max_level": max_level,
            "safety_stock": safety_stock,
            "reorder_point": min_level
        }
    
    def adjust_for_perishability(self,
                               max_level: float,
                               shelf_life: float,
                               avg_demand: float) -> float:
        """
        Adjust max inventory level for perishable items.
        
        Args:
            max_level: Calculated max inventory level
            shelf_life: Shelf life in days
            avg_demand: Average daily demand
            
        Returns:
            Adjusted max level
        """
        # Don't hold more than we can sell before expiry
        max_possible = avg_demand * shelf_life
        
        # Return the minimum of the calculated max level and what we can sell before expiry
        return min(max_level, max_possible)
    
    def adjust_for_storage_constraints(self,
                                     max_level: float,
                                     unit_volume: float,
                                     available_space: float) -> float:
        """
        Adjust max inventory level for storage constraints.
        
        Args:
            max_level: Calculated max inventory level
            unit_volume: Volume per unit
            available_space: Available storage space
            
        Returns:
            Adjusted max level
        """
        # Calculate maximum units that can be stored
        max_units = available_space / unit_volume if unit_volume > 0 else float('inf')
        
        # Return the minimum of the calculated max level and what can be stored
        return min(max_level, max_units)
    
    def optimize_inventory_policy(self,
                                 product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize inventory policy for a product.
        
        Args:
            product_data: Dictionary with product data including:
                - avg_demand: Average demand per time unit
                - demand_std: Standard deviation of demand
                - lead_time: Average lead time
                - lead_time_std: Standard deviation of lead time (optional)
                - unit_cost: Cost per unit
                - is_perishable: Whether the product is perishable
                - shelf_life: Shelf life in days (required if is_perishable is True)
                - unit_volume: Volume per unit (optional)
                - available_space: Available storage space (optional)
                - review_period: Inventory review period
                
        Returns:
            Dictionary with optimized inventory policy
        """
        # Extract required parameters
        avg_demand = product_data.get('avg_demand')
        demand_std = product_data.get('demand_std')
        lead_time = product_data.get('lead_time')
        
        if any(param is None for param in [avg_demand, demand_std, lead_time]):
            raise ValueError("Missing required parameters: avg_demand, demand_std, lead_time")
        
        # Extract optional parameters
        lead_time_std = product_data.get('lead_time_std')
        unit_cost = product_data.get('unit_cost', 1.0)
        is_perishable = product_data.get('is_perishable', False)
        shelf_life = product_data.get('shelf_life')
        unit_volume = product_data.get('unit_volume')
        available_space = product_data.get('available_space')
        review_period = product_data.get('review_period', 7)  # Default weekly review
        
        # Validate perishable products have shelf life
        if is_perishable and shelf_life is None:
            raise ValueError("Shelf life must be provided for perishable products")
        
        # Calculate EOQ if annual demand is provided
        annual_demand = product_data.get('annual_demand')
        eoq = None
        if annual_demand is not None and unit_cost is not None:
            eoq = self.calculate_economic_order_quantity(
                annual_demand=annual_demand,
                unit_cost=unit_cost
            )
        
        # Calculate min/max levels
        inventory_levels = self.calculate_min_max_levels(
            avg_demand=avg_demand,
            lead_time=lead_time,
            review_period=review_period,
            demand_std=demand_std,
            lead_time_std=lead_time_std
        )
        
        min_level = inventory_levels['min_level']
        max_level = inventory_levels['max_level']
        safety_stock = inventory_levels['safety_stock']
        
        # Adjust for perishability
        if is_perishable and shelf_life is not None:
            max_level = self.adjust_for_perishability(
                max_level=max_level,
                shelf_life=shelf_life,
                avg_demand=avg_demand
            )
        
        # Adjust for storage constraints
        if unit_volume is not None and available_space is not None:
            max_level = self.adjust_for_storage_constraints(
                max_level=max_level,
                unit_volume=unit_volume,
                available_space=available_space
            )
        
        # Prepare result
        result = {
            'min_level': min_level,
            'max_level': max_level,
            'safety_stock': safety_stock,
            'reorder_point': min_level
        }
        
        # Add EOQ if calculated
        if eoq is not None:
            result['eoq'] = eoq
        
        return result

class MultiEchelonOptimizer:
    """
    Multi-echelon inventory optimization for complex supply chain networks.
    
    Optimizes inventory levels across multiple tiers (e.g., manufacturer, 
    distributor, retailer) in the supply chain.
    """
    
    def __init__(self, service_level: float = 0.95):
        """
        Initialize the multi-echelon optimizer.
        
        Args:
            service_level: Desired overall service level
        """
        self.service_level = service_level
        self.single_optimizer = InventoryOptimizer(service_level=service_level)
        
        logger.info(f"Initialized MultiEchelonOptimizer with service level: {service_level}")
    
    def optimize_network(self, 
                        nodes: List[Dict[str, Any]], 
                        network_structure: List[Tuple[int, int]]) -> Dict[int, Dict[str, Any]]:
        """
        Optimize inventory across a multi-echelon supply chain network.
        
        Args:
            nodes: List of dictionaries with node data (each node represents a facility)
                  Each node should have an 'id' field and inventory parameters
            network_structure: List of (source_id, destination_id) tuples representing the network
            
        Returns:
            Dictionary mapping node IDs to their optimized inventory policies
        """
        # Validate inputs
        if not nodes:
            raise ValueError("Nodes list cannot be empty")
        
        # Create a mapping of node ID to its index in the nodes list
        node_id_to_index = {node['id']: i for i, node in enumerate(nodes)}
        
        # Create adjacency list to represent the network
        # For each node, which nodes receive from it?
        outgoing = {node['id']: [] for node in nodes}
        # For each node, which nodes supply to it?
        incoming = {node['id']: [] for node in nodes}
        
        for source_id, dest_id in network_structure:
            if source_id not in node_id_to_index or dest_id not in node_id_to_index:
                raise ValueError(f"Invalid node IDs in network structure: {source_id}, {dest_id}")
            
            outgoing[source_id].append(dest_id)
            incoming[dest_id].append(source_id)
        
        # Identify the tiers in the network (0 = retailers, max = suppliers)
        tiers = {}
        max_tier = 0
        
        # Start with retail nodes (no outgoing connections)
        for node in nodes:
            node_id = node['id']
            if not outgoing[node_id]:  # No outgoing = retail
                tiers[node_id] = 0
        
        # Assign tiers to the rest of the nodes
        changed = True
        while changed:
            changed = False
            for node in nodes:
                node_id = node['id']
                if node_id in tiers:
                    continue
                
                # If all destinations have been assigned tiers, this node's tier is one higher than the max
                if all(dest_id in tiers for dest_id in outgoing[node_id]):
                    dest_tiers = [tiers[dest_id] for dest_id in outgoing[node_id]]
                    tiers[node_id] = max(dest_tiers) + 1 if dest_tiers else 0
                    max_tier = max(max_tier, tiers[node_id])
                    changed = True
        
        # Initialize results
        results = {}
        
        # Process each tier, starting from retailers (tier 0)
        for tier in range(max_tier + 1):
            tier_nodes = [node for node in nodes if tiers.get(node['id'], -1) == tier]
            
            for node in tier_nodes:
                node_id = node['id']
                
                # For retailers (tier 0), use standard optimization
                if tier == 0:
                    results[node_id] = self.single_optimizer.optimize_inventory_policy(node)
                else:
                    # For higher tiers, we need to consider the demand patterns of downstream nodes
                    downstream_demand = []
                    downstream_std = []
                    
                    for dest_id in outgoing[node_id]:
                        # Get the reorder point of the destination node
                        dest_result = results.get(dest_id, {})
                        if 'reorder_point' in dest_result:
                            downstream_demand.append(dest_result['reorder_point'])
                        
                        # Get the safety stock of the destination node as an approximation of variability
                        if 'safety_stock' in dest_result:
                            downstream_std.append(dest_result['safety_stock'])
                    
                    # If we have downstream demand data, adjust this node's demand parameters
                    if downstream_demand:
                        # The node's demand is the sum of the reorder points of its destinations
                        node['avg_demand'] = sum(downstream_demand)
                        
                        # The node's demand variability is influenced by the safety stocks of destinations
                        # This is a simplification; a more complex approach would be to consider the 
                        # correlation between demands
                        node['demand_std'] = np.sqrt(sum(s**2 for s in downstream_std)) if downstream_std else node.get('demand_std', 0)
                    
                    # Now optimize this node with adjusted parameters
                    results[node_id] = self.single_optimizer.optimize_inventory_policy(node)
        
        return results 