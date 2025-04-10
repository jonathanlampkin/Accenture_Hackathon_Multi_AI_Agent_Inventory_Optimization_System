"""
Coordinator for the Multi-Agent Inventory Optimization System.
This module contains the MultiAgentCoordinator class that manages the collaboration
between specialized agents to optimize inventory, demand forecasting, and pricing.
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import agent classes
from src.agents.inventory_agent import InventoryAgent
from src.agents.demand_agent import DemandAgent
from src.agents.pricing_agent import PricingAgent 
from src.agents.supply_chain_agent import SupplyChainAgent
from src.agents.crew_agents import InventoryCrew

logger = logging.getLogger("InventorySystem.Coordinator")

class MultiAgentCoordinator:
    """
    Coordinator for the multi-agent inventory optimization system.
    Manages the collaboration between specialized agents.
    """
    
    def __init__(self, optimization_target='balanced', product_id=None, store_id=None, 
                 max_iterations=5, output_dir=None, use_gpu=False, use_crewai=True):
        """
        Initialize the coordinator with specialized agents.
        
        Args:
            optimization_target: Target optimization goal ('cost', 'availability', 'balanced')
            product_id: Optional specific product ID to focus on
            store_id: Optional specific store ID to focus on
            max_iterations: Maximum number of optimization iterations
            output_dir: Optional custom output directory
            use_gpu: Whether to use GPU acceleration if available
            use_crewai: Whether to use the new crewAI-based system
        """
        self.optimization_target = optimization_target
        self.product_id = product_id
        self.store_id = store_id
        self.max_iterations = max_iterations
        self.use_gpu = use_gpu
        self.use_crewai = use_crewai
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = os.path.join('output', datetime.now().strftime('%Y%m%d_%H%M%S'))
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize agents based on the chosen system
        if self.use_crewai:
            self.crew = InventoryCrew(
                data_path=os.path.join('data', 'processed'),
                output_dir=self.output_dir
            )
            logger.info("Initialized crewAI-based inventory optimization system")
        else:
            self.inventory_agent = InventoryAgent(use_gpu=use_gpu)
            self.demand_agent = DemandAgent(use_gpu=use_gpu)
            self.pricing_agent = PricingAgent(use_gpu=use_gpu)
            self.supply_chain_agent = SupplyChainAgent(use_gpu=use_gpu)
            logger.info("Initialized traditional multi-agent system")
        
        logger.info(f"MultiAgentCoordinator initialized with target: {optimization_target}")
        logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
        
        if product_id:
            logger.info(f"Product ID filter: {product_id}")
        else:
            logger.info("Product ID filter: All products")
            
        if store_id:
            logger.info(f"Store ID filter: {store_id}")
        else:
            logger.info("Store ID filter: All stores")
    
    def _get_inventory_weights(self):
        """
        Get optimization weights for inventory based on the target.
        
        Returns:
            dict: Optimization weights
        """
        if self.optimization_target == 'cost':
            return {
                'inventory_carrying_cost': 0.6,
                'stockout_cost': 0.2,
                'order_cost': 0.2
            }
        elif self.optimization_target == 'availability':
            return {
                'inventory_carrying_cost': 0.2,
                'stockout_cost': 0.6,
                'order_cost': 0.2
            }
        else:  # balanced
            return {
                'inventory_carrying_cost': 0.4,
                'stockout_cost': 0.4,
                'order_cost': 0.2
            }
    
    def _get_pricing_weights(self):
        """
        Get optimization weights for pricing based on the target.
        
        Returns:
            dict: Optimization weights
        """
        if self.optimization_target == 'cost':
            return {
                'profit_margin': 0.6,
                'sales_volume': 0.2,
                'market_share': 0.2
            }
        elif self.optimization_target == 'availability':
            return {
                'profit_margin': 0.2,
                'sales_volume': 0.6,
                'market_share': 0.2
            }
        else:  # balanced
            return {
                'profit_margin': 0.4,
                'sales_volume': 0.4,
                'market_share': 0.2
            }
    
    def run_optimization(self):
        """
        Run the optimization process using either crewAI or traditional agents.
        """
        try:
            if self.use_crewai:
                # Load and prepare data
                data = self._load_data()
                
                # Run crewAI optimization
                results = self.crew.run_optimization(data)
                
                # Save results
                self._save_results(results)
                
                return results
            else:
                # Use traditional multi-agent approach
                return self._run_traditional_optimization()
                
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            raise

    def _load_data(self):
        """Load and prepare data for optimization."""
        try:
            # Load your data here - implement based on your data structure
            data = pd.read_csv(os.path.join('data', 'processed', 'inventory_data.csv'))
            
            if self.product_id:
                data = data[data['product_id'] == self.product_id]
            if self.store_id:
                data = data[data['store_id'] == self.store_id]
                
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _save_results(self, results):
        """Save optimization results to the output directory."""
        try:
            output_file = os.path.join(self.output_dir, 'optimization_results.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, default=self._json_serializable)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def _run_traditional_optimization(self):
        """Run optimization using the traditional multi-agent approach."""
        logger.info(f"Starting optimization process with {self.max_iterations} iterations")
        
        results = {
            'optimization_target': self.optimization_target,
            'iterations_completed': 0,
            'error': None
        }
        
        try:
            # Initial data gathering and baseline analysis
            logger.info("Gathering initial data and performing baseline analysis")
            
            # Get inventory analysis
            inventory_data = self.inventory_agent.analyze()
            
            # Get demand analysis and forecast
            demand_data = self.demand_agent.analyze()
            
            # Get pricing analysis
            pricing_data = self.pricing_agent.analyze()
            
            # Iterative optimization
            for iteration in range(self.max_iterations):
                logger.info(f"Starting optimization iteration {iteration + 1}/{self.max_iterations}")
                
                # Exchange information between agents
                self._exchange_agent_information()
                
                # Get recommendations from each agent
                inventory_recommendations = self.inventory_agent.make_recommendation()
                demand_recommendations = self.demand_agent.make_recommendation()
                pricing_recommendations = self.pricing_agent.make_recommendation()
                
                # Integrate recommendations and resolve conflicts
                integrated_recommendations = self._integrate_recommendations(
                    inventory_recommendations,
                    demand_recommendations,
                    pricing_recommendations
                )
                
                # Update agents with integrated recommendations
                self._update_agents_with_feedback(integrated_recommendations)
                
                # Save iteration results
                self._save_iteration_results(iteration + 1, integrated_recommendations)
                
                results['iterations_completed'] = iteration + 1
            
            # Final analysis
            final_results = self._generate_final_results()
            results.update(final_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during optimization process: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def _exchange_agent_information(self):
        """
        Facilitate information exchange between agents.
        """
        # Share inventory agent's critical products with demand agent
        inventory_critical = self.inventory_agent.state.get('critical_products', [])
        if inventory_critical:
            self.demand_agent.receive_message(
                'InventoryAgent',
                {'critical_products': inventory_critical},
                'info'
            )
        
        # Share demand agent's forecasts with inventory agent
        demand_forecasts = self.demand_agent.state.get('forecasts', {})
        if demand_forecasts:
            self.inventory_agent.receive_message(
                'DemandAgent',
                {'demand_forecasts': demand_forecasts},
                'info'
            )
        
        # Share pricing elasticity with inventory agent
        price_elasticity = self.pricing_agent.state.get('price_elasticity', {})
        if price_elasticity:
            self.inventory_agent.receive_message(
                'PricingAgent',
                {'price_elasticity': price_elasticity},
                'info'
            )
    
    def _integrate_recommendations(self, inventory_recs, demand_recs, pricing_recs):
        """
        Integrate recommendations from all agents and resolve conflicts.
        
        Args:
            inventory_recs: Recommendations from inventory agent
            demand_recs: Recommendations from demand agent
            pricing_recs: Recommendations from pricing agent
            
        Returns:
            dict: Integrated recommendations
        """
        integrated_recs = {
            'inventory': inventory_recs,
            'demand': demand_recs,
            'pricing': pricing_recs,
            'conflicts_resolved': [],
            'prioritized_actions': []
        }
        
        # Identify and resolve conflicts
        conflicts = self._identify_conflicts(inventory_recs, demand_recs, pricing_recs)
        
        for conflict in conflicts:
            resolution = self._resolve_conflict(conflict)
            if resolution:
                integrated_recs['conflicts_resolved'].append({
                    'conflict': conflict,
                    'resolution': resolution
                })
        
        # Prioritize actions based on optimization target
        prioritized_actions = self._prioritize_actions(integrated_recs)
        integrated_recs['prioritized_actions'] = prioritized_actions
        
        return integrated_recs
    
    def _identify_conflicts(self, inventory_recs, demand_recs, pricing_recs):
        """
        Identify conflicts between agent recommendations.
        
        Returns:
            list: Conflicts identified
        """
        conflicts = []
        
        # Simplified conflict identification
        # In a real system, this would involve more complex logic
        
        # Example: Conflict between inventory reduction and demand growth
        inventory_reduce = inventory_recs.get('reduce_stock_recommendations', [])
        demand_growth = demand_recs.get('trend_based_recommendations', [])
        
        for inv_rec in inventory_reduce:
            for demand_rec in demand_growth:
                if (inv_rec.get('product_id') == demand_rec.get('product_id') and
                    demand_rec.get('trend') == 'Increasing'):
                    conflicts.append({
                        'type': 'inventory_demand_conflict',
                        'inventory_rec': inv_rec,
                        'demand_rec': demand_rec,
                        'description': 'Inventory suggests reducing stock while demand trend is increasing'
                    })
        
        # Example: Conflict between price increase and demand elasticity
        price_increases = pricing_recs.get('price_adjustment_recommendations', [])
        
        for price_rec in price_increases:
            product_id = price_rec.get('product_id')
            
            # Check if this conflicts with elastic demand
            for demand_rec in demand_growth:
                if demand_rec.get('product_id') == product_id:
                    conflicts.append({
                        'type': 'pricing_demand_conflict',
                        'pricing_rec': price_rec,
                        'demand_rec': demand_rec,
                        'description': 'Price increase suggested while demand may be elastic'
                    })
        
        return conflicts
    
    def _resolve_conflict(self, conflict):
        """
        Resolve a conflict based on optimization target.
        
        Args:
            conflict: Conflict information
            
        Returns:
            dict: Resolution
        """
        conflict_type = conflict.get('type')
        
        if conflict_type == 'inventory_demand_conflict':
            if self.optimization_target == 'availability':
                return {
                    'action': 'favor_demand',
                    'description': 'Maintain higher inventory levels to support increasing demand'
                }
            elif self.optimization_target == 'cost':
                return {
                    'action': 'favor_inventory',
                    'description': 'Reduce inventory despite demand trend to minimize carrying costs'
                }
            else:  # balanced
                return {
                    'action': 'moderate_reduction',
                    'description': 'Slightly reduce inventory while increasing order frequency'
                }
                
        elif conflict_type == 'pricing_demand_conflict':
            if self.optimization_target == 'availability':
                return {
                    'action': 'maintain_price',
                    'description': 'Maintain current price to support demand growth'
                }
            elif self.optimization_target == 'cost':
                return {
                    'action': 'incremental_increase',
                    'description': 'Implement smaller, gradual price increases'
                }
            else:  # balanced
                return {
                    'action': 'selective_increase',
                    'description': 'Increase prices only for less elastic products'
                }
        
        return None
    
    def _prioritize_actions(self, integrated_recs):
        """
        Prioritize actions based on optimization target.
        
        Args:
            integrated_recs: Integrated recommendations
            
        Returns:
            list: Prioritized actions
        """
        # Simplified prioritization
        # In a real system, this would involve more complex scoring and ranking
        
        prioritized = []
        
        if self.optimization_target == 'cost':
            # Prioritize cost-saving actions
            for rec in integrated_recs.get('inventory', {}).get('reduce_stock_recommendations', []):
                prioritized.append({
                    'type': 'reduce_inventory',
                    'product_id': rec.get('product_id'),
                    'priority': 'high',
                    'description': f"Reduce excess inventory for product {rec.get('product_id')}"
                })
                
            for rec in integrated_recs.get('pricing', {}).get('price_adjustment_recommendations', []):
                if rec.get('price_change_percentage', 0) > 0:
                    prioritized.append({
                        'type': 'increase_price',
                        'product_id': rec.get('product_id'),
                        'priority': 'high',
                        'description': f"Increase price for product {rec.get('product_id')}"
                    })
                    
        elif self.optimization_target == 'availability':
            # Prioritize availability-enhancing actions
            for rec in integrated_recs.get('inventory', {}).get('restock_recommendations', []):
                prioritized.append({
                    'type': 'restock_inventory',
                    'product_id': rec.get('product_id'),
                    'priority': 'high',
                    'description': f"Restock product {rec.get('product_id')} to prevent stockouts"
                })
                
            for rec in integrated_recs.get('demand', {}).get('trend_based_recommendations', []):
                if rec.get('trend') == 'Increasing':
                    prioritized.append({
                        'type': 'increase_inventory',
                        'product_id': rec.get('product_id'),
                        'priority': 'high',
                        'description': f"Increase inventory for growing demand of product {rec.get('product_id')}"
                    })
        else:  # balanced
            # Balance cost and availability
            for rec in integrated_recs.get('inventory', {}).get('reorder_point_adjustments', []):
                prioritized.append({
                    'type': 'adjust_reorder_point',
                    'product_id': rec.get('product_id'),
                    'priority': 'high',
                    'description': f"Optimize reorder point for product {rec.get('product_id')}"
                })
                
            for rec in integrated_recs.get('pricing', {}).get('competitive_pricing_recommendations', []):
                prioritized.append({
                    'type': 'adjust_competitive_pricing',
                    'product_id': rec.get('product_id'),
                    'priority': 'medium',
                    'description': f"Adjust pricing to be competitive for product {rec.get('product_id')}"
                })
        
        # Sort by priority
        return sorted(prioritized, key=lambda x: 0 if x['priority'] == 'high' else 1 if x['priority'] == 'medium' else 2)
    
    def _update_agents_with_feedback(self, integrated_recommendations):
        """
        Update agents with feedback from integrated recommendations.
        
        Args:
            integrated_recommendations: Integrated recommendations
        """
        # Provide feedback to each agent based on integration results
        
        # Inventory agent feedback
        inventory_feedback = {
            'integrated_recommendations': integrated_recommendations,
            'optimization_target': self.optimization_target
        }
        self.inventory_agent.update(inventory_feedback)
        
        # Demand agent feedback
        demand_feedback = {
            'integrated_recommendations': integrated_recommendations,
            'optimization_target': self.optimization_target
        }
        self.demand_agent.update(demand_feedback)
        
        # Pricing agent feedback
        pricing_feedback = {
            'integrated_recommendations': integrated_recommendations,
            'optimization_target': self.optimization_target
        }
        self.pricing_agent.update(pricing_feedback)
    
    def _save_iteration_results(self, iteration, integrated_recommendations):
        """
        Save results from the current optimization iteration.
        
        Args:
            iteration: Current iteration number
            integrated_recommendations: Integrated recommendations
        """
        # Create directory for this iteration
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save integrated recommendations
        recommendations_file = os.path.join(iteration_dir, 'integrated_recommendations.json')
        with open(recommendations_file, 'w') as f:
            json.dump(integrated_recommendations, f, indent=2, default=self._json_serializable)
        
        logger.info(f"Saved iteration {iteration} results to {iteration_dir}")
    
    def _generate_final_results(self):
        """
        Generate final results after all iterations.
        
        Returns:
            dict: Final results and metrics
        """
        # In a real system, this would involve aggregating results and calculating metrics
        
        # Simple metrics for demonstration
        final_results = {
            'completed': True,
            'output_directory': self.output_dir,
            'optimization_metrics': {
                'target': self.optimization_target,
                'iterations': self.max_iterations,
                'product_filter': self.product_id or 'All',
                'store_filter': self.store_id or 'All'
            }
        }
        
        # Save final results
        results_file = os.path.join(self.output_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=self._json_serializable)
        
        return final_results
    
    def _json_serializable(self, obj):
        """Helper method to make objects JSON serializable"""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable") 