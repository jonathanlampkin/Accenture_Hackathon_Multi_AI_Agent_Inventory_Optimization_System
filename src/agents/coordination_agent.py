"""
Coordination Agent for the Multi-Agent Inventory Optimization System.
This agent is responsible for coordinating and integrating insights from all
specialized agents to provide balanced recommendations that optimize across
inventory management, demand forecasting, and pricing goals.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.agents.base_agent import BaseAgent

class CoordinationAgent(BaseAgent):
    """
    Agent responsible for coordinating and integrating all specialized agents.
    """
    
    def __init__(self, inventory_agent=None, demand_agent=None, pricing_agent=None):
        """
        Initialize the Coordination Agent.
        
        Args:
            inventory_agent: InventoryAgent instance
            demand_agent: DemandAgent instance
            pricing_agent: PricingAgent instance
        """
        super().__init__("CoordinationAgent")
        
        # Store references to specialized agents
        self.agents = {
            'inventory': inventory_agent,
            'demand': demand_agent,
            'pricing': pricing_agent
        }
        
        # Initialize agent-specific state
        self.state.update({
            'integrated_insights': {},
            'conflict_resolution': {},
            'optimization_goals': {
                'inventory_turnover': True,  # Optimize inventory turnover
                'stockout_prevention': True,  # Prevent stockouts
                'revenue_maximization': True,  # Maximize revenue
                'cost_minimization': True     # Minimize costs
            },
            'last_coordination_time': None,
            'prioritized_recommendations': []
        })
        
        # Load previous state if available
        self.load_state()
    
    def register_agent(self, agent_type, agent):
        """
        Register a specialized agent with the coordination agent.
        
        Args:
            agent_type: Type of agent ('inventory', 'demand', 'pricing')
            agent: Agent instance
        """
        self.agents[agent_type] = agent
        self.log(f"Registered {agent_type} agent: {agent.name}")
    
    def analyze(self):
        """
        Coordinate analysis across all specialized agents and integrate their insights.
        
        Returns:
            dict: Integrated analysis results
        """
        self.log("Starting coordination analysis...")
        
        # Update last coordination time
        self.state['last_coordination_time'] = pd.Timestamp.now().isoformat()
        
        # Dictionary to store integrated insights
        integrated_insights = {}
        
        # Check which agents are registered
        for agent_type, agent in self.agents.items():
            if agent is not None:
                self.log(f"Requesting analysis from {agent_type} agent...")
                
                # Request analysis from the agent
                agent_insights = self._get_agent_insights(agent)
                
                # Store agent insights
                integrated_insights[agent_type] = agent_insights
                
                self.log(f"Received insights from {agent_type} agent")
        
        # Identify product overlaps and integrate insights
        self.log("Integrating insights across agents...")
        integrated_product_insights = self._integrate_product_insights(integrated_insights)
        
        # Resolve conflicting recommendations
        conflict_resolution = self._resolve_conflicts(integrated_product_insights)
        
        # Update state with integrated insights
        self.state['integrated_insights'] = integrated_insights
        self.state['conflict_resolution'] = conflict_resolution
        
        # Create analysis summary
        analysis_results = {
            'timestamp': self.state['last_coordination_time'],
            'integrated_insights': {
                'product_count': len(integrated_product_insights),
                'agent_coverage': {agent_type: (agent is not None) for agent_type, agent in self.agents.items()}
            },
            'conflict_resolution': {
                'conflicts_identified': len(conflict_resolution.get('conflicts', [])),
                'conflicts_resolved': len(conflict_resolution.get('resolutions', []))
            }
        }
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        # Save state after analysis
        self.save_state()
        
        return analysis_results
    
    def _get_agent_insights(self, agent):
        """
        Get insights from a specialized agent using the messaging system.
        
        Args:
            agent: Agent instance
            
        Returns:
            dict: Agent's insights
        """
        # Request analysis results from the agent
        response = self.send_message(agent, {'get_analysis_results': True}, 'request')
        
        # If agent doesn't respond with analysis results, request recommendations
        if not response or 'analysis_results' not in response:
            response = self.send_message(agent, {'get_recommendations': True}, 'request')
            
            if response and 'recommendations' in response:
                return {'recommendations': response['recommendations']}
        else:
            return {'analysis_results': response['analysis_results']}
        
        # If all else fails, return empty insights
        return {}
    
    def _integrate_product_insights(self, integrated_insights):
        """
        Integrate insights across agents for each product.
        
        Args:
            integrated_insights: Dictionary with insights from each agent
            
        Returns:
            dict: Integrated insights by product
        """
        # Dictionary to store integrated insights by product
        product_insights = {}
        
        # Extract product-specific insights from each agent
        
        # Inventory agent insights
        if 'inventory' in integrated_insights:
            inventory_data = integrated_insights['inventory']
            
            # Extract critical products
            if 'analysis_results' in inventory_data and 'critical_products_count' in inventory_data['analysis_results']:
                # Request critical products list from inventory agent
                response = self.send_message(self.agents['inventory'], {'get_critical_products': True}, 'request')
                
                if response and 'critical_products' in response:
                    critical_products = response['critical_products']
                    
                    # Add to product insights
                    for product in critical_products:
                        product_id = product['product_id']
                        
                        if product_id not in product_insights:
                            product_insights[product_id] = {}
                        
                        if 'inventory' not in product_insights[product_id]:
                            product_insights[product_id]['inventory'] = {}
                        
                        product_insights[product_id]['inventory']['critical'] = True
                        product_insights[product_id]['inventory']['risk_score'] = product['risk_score']
                        product_insights[product_id]['inventory']['current_stock'] = product['current_stock']
                        product_insights[product_id]['inventory']['reorder_point'] = product['reorder_point']
            
            # Extract surplus products
            if 'analysis_results' in inventory_data and 'surplus_products_count' in inventory_data['analysis_results']:
                # Request surplus products list from inventory agent
                response = self.send_message(self.agents['inventory'], {'get_surplus_products': True}, 'request')
                
                if response and 'surplus_products' in response:
                    surplus_products = response['surplus_products']
                    
                    # Add to product insights
                    for product in surplus_products:
                        product_id = product['product_id']
                        
                        if product_id not in product_insights:
                            product_insights[product_id] = {}
                        
                        if 'inventory' not in product_insights[product_id]:
                            product_insights[product_id]['inventory'] = {}
                        
                        product_insights[product_id]['inventory']['surplus'] = True
                        product_insights[product_id]['inventory']['excess_percentage'] = product['excess_percentage']
                        product_insights[product_id]['inventory']['days_of_supply'] = product['days_of_supply']
        
        # Demand agent insights
        if 'demand' in integrated_insights:
            demand_data = integrated_insights['demand']
            
            # Extract demand patterns
            if 'analysis_results' in demand_data and 'demand_patterns' in demand_data['analysis_results']:
                demand_patterns = demand_data['analysis_results']['demand_patterns']
                
                # Add to product insights
                for product_id, pattern in demand_patterns.items():
                    if product_id not in product_insights:
                        product_insights[product_id] = {}
                    
                    if 'demand' not in product_insights[product_id]:
                        product_insights[product_id]['demand'] = {}
                    
                    product_insights[product_id]['demand']['trend'] = pattern['trend']
                    product_insights[product_id]['demand']['volatility'] = pattern['volatility']
                    product_insights[product_id]['demand']['mean_demand'] = pattern['mean_demand']
            
            # Extract seasonality factors
            if 'analysis_results' in demand_data and 'seasonality_factors' in demand_data['analysis_results']:
                seasonality_factors = demand_data['analysis_results']['seasonality_factors']
                
                # Add to product insights
                for product_id, seasonality in seasonality_factors.items():
                    if product_id not in product_insights:
                        product_insights[product_id] = {}
                    
                    if 'demand' not in product_insights[product_id]:
                        product_insights[product_id]['demand'] = {}
                    
                    product_insights[product_id]['demand']['seasonality'] = seasonality
        
        # Pricing agent insights
        if 'pricing' in integrated_insights:
            pricing_data = integrated_insights['pricing']
            
            # Extract price elasticity
            if 'analysis_results' in pricing_data and 'price_elasticity' in pricing_data['analysis_results']:
                price_elasticity = pricing_data['analysis_results']['price_elasticity']
                
                # Add to product insights
                for product_id, elasticity in price_elasticity.items():
                    if product_id not in product_insights:
                        product_insights[product_id] = {}
                    
                    if 'pricing' not in product_insights[product_id]:
                        product_insights[product_id]['pricing'] = {}
                    
                    product_insights[product_id]['pricing']['elasticity'] = elasticity
            
            # Extract optimal prices
            if 'analysis_results' in pricing_data and 'optimal_prices' in pricing_data['analysis_results']:
                optimal_prices = pricing_data['analysis_results']['optimal_prices']
                
                # Add to product insights
                for key, price_data in optimal_prices.items():
                    product_id = price_data['product_id']
                    
                    if product_id not in product_insights:
                        product_insights[product_id] = {}
                    
                    if 'pricing' not in product_insights[product_id]:
                        product_insights[product_id]['pricing'] = {}
                    
                    # Store optimal price data
                    if 'optimal_prices' not in product_insights[product_id]['pricing']:
                        product_insights[product_id]['pricing']['optimal_prices'] = []
                    
                    product_insights[product_id]['pricing']['optimal_prices'].append(price_data)
        
        return product_insights
    
    def _resolve_conflicts(self, product_insights):
        """
        Identify and resolve conflicts between agent recommendations.
        
        Args:
            product_insights: Dictionary with integrated insights by product
            
        Returns:
            dict: Conflict resolution results
        """
        self.log("Resolving conflicts between agent recommendations...")
        
        # Lists to track conflicts and resolutions
        conflicts = []
        resolutions = []
        
        # Examine each product for potential conflicts
        for product_id, insights in product_insights.items():
            # Check for inventory vs. pricing conflicts
            if 'inventory' in insights and 'pricing' in insights:
                # Conflict: Critical inventory but price increase recommended
                if insights['inventory'].get('critical', False) and 'optimal_prices' in insights['pricing']:
                    for price_data in insights['pricing']['optimal_prices']:
                        if price_data['price_change_percentage'] > 0:
                            conflict = {
                                'product_id': product_id,
                                'type': 'inventory_pricing',
                                'description': 'Critical inventory but price increase recommended',
                                'inventory_risk': insights['inventory'].get('risk_score', 0),
                                'price_change': price_data['price_change_percentage'],
                                'revenue_impact': price_data.get('revenue_change_percentage', 0)
                            }
                            conflicts.append(conflict)
                            
                            # Resolution: Prioritize inventory if risk is high, otherwise consider pricing
                            if insights['inventory'].get('risk_score', 0) > 1.2:
                                resolution = {
                                    'conflict_id': len(conflicts) - 1,
                                    'resolution': 'prioritize_inventory',
                                    'action': 'Keep price stable or reduce slightly to address inventory risk',
                                    'rationale': 'High stockout risk outweighs potential revenue gain'
                                }
                            else:
                                resolution = {
                                    'conflict_id': len(conflicts) - 1,
                                    'resolution': 'balanced',
                                    'action': 'Moderate price increase with increased order quantity',
                                    'rationale': 'Balance revenue opportunity with inventory risk'
                                }
                            
                            resolutions.append(resolution)
                
                # Conflict: Surplus inventory but price decrease recommended
                if insights['inventory'].get('surplus', False) and 'optimal_prices' in insights['pricing']:
                    for price_data in insights['pricing']['optimal_prices']:
                        if price_data['price_change_percentage'] < 0:
                            conflict = {
                                'product_id': product_id,
                                'type': 'inventory_pricing',
                                'description': 'Surplus inventory with recommended price decrease',
                                'excess_percentage': insights['inventory'].get('excess_percentage', 0),
                                'price_change': price_data['price_change_percentage'],
                                'revenue_impact': price_data.get('revenue_change_percentage', 0)
                            }
                            conflicts.append(conflict)
                            
                            # Resolution: Combine price reduction with promotion to reduce inventory
                            resolution = {
                                'conflict_id': len(conflicts) - 1,
                                'resolution': 'synergistic',
                                'action': 'Apply suggested price reduction with promotional campaign',
                                'rationale': 'Use price reduction to accelerate inventory reduction'
                            }
                            
                            resolutions.append(resolution)
            
            # Check for demand vs. inventory conflicts
            if 'demand' in insights and 'inventory' in insights:
                # Conflict: Declining demand but critical inventory
                if insights['demand'].get('trend') == 'Decreasing' and insights['inventory'].get('critical', False):
                    conflict = {
                        'product_id': product_id,
                        'type': 'demand_inventory',
                        'description': 'Declining demand with critical inventory levels',
                        'demand_trend': 'Decreasing',
                        'inventory_risk': insights['inventory'].get('risk_score', 0)
                    }
                    conflicts.append(conflict)
                    
                    # Resolution: Adjust reorder points downward
                    resolution = {
                        'conflict_id': len(conflicts) - 1,
                        'resolution': 'adjust_reorder_points',
                        'action': 'Reduce reorder points to align with declining demand',
                        'rationale': 'Prevent future surplus inventory by adapting to demand trend'
                    }
                    
                    resolutions.append(resolution)
                
                # Conflict: Increasing demand but surplus inventory
                if insights['demand'].get('trend') == 'Increasing' and insights['inventory'].get('surplus', False):
                    conflict = {
                        'product_id': product_id,
                        'type': 'demand_inventory',
                        'description': 'Increasing demand with surplus inventory',
                        'demand_trend': 'Increasing',
                        'excess_percentage': insights['inventory'].get('excess_percentage', 0)
                    }
                    conflicts.append(conflict)
                    
                    # Resolution: Monitor closely without immediate action
                    resolution = {
                        'conflict_id': len(conflicts) - 1,
                        'resolution': 'monitor',
                        'action': 'Monitor closely but allow demand to consume excess inventory',
                        'rationale': 'Increasing demand will naturally reduce excess inventory'
                    }
                    
                    resolutions.append(resolution)
        
        return {
            'conflicts': conflicts,
            'resolutions': resolutions
        }
    
    def _save_analysis_results(self, results):
        """
        Save analysis results to a file.
        
        Args:
            results: Dictionary with analysis results
        """
        results_file = os.path.join(self.output_dir, 'coordination_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.log(f"Coordination results saved to {results_file}")
    
    def make_recommendation(self):
        """
        Generate integrated recommendations based on all agent insights.
        
        Returns:
            dict: Integrated recommendations
        """
        self.log("Generating integrated recommendations...")
        
        # Check if we have integrated insights
        if not self.state.get('integrated_insights'):
            self.analyze()
        
        # Get integrated insights and conflict resolutions
        integrated_insights = self.state.get('integrated_insights', {})
        conflict_resolution = self.state.get('conflict_resolution', {})
        
        # List to store prioritized recommendations
        prioritized_recommendations = []
        
        # Get recommendations from each agent
        agent_recommendations = {}
        for agent_type, agent in self.agents.items():
            if agent is not None:
                # Request recommendations from the agent
                response = self.send_message(agent, {'get_recommendations': True}, 'request')
                
                if response and 'recommendations' in response:
                    agent_recommendations[agent_type] = response['recommendations']
        
        # Process inventory agent recommendations
        if 'inventory' in agent_recommendations:
            inventory_recs = agent_recommendations['inventory']
            
            # Add critical restock recommendations
            if 'restock_recommendations' in inventory_recs:
                for rec in inventory_recs['restock_recommendations']:
                    if rec['urgency'] == 'High':
                        prioritized_recommendations.append({
                            'product_id': rec['product_id'],
                            'store_id': rec['store_id'],
                            'type': 'restock',
                            'action': f"Order {rec['recommended_order_quantity']} units of product {rec['product_id']} for store {rec['store_id']}",
                            'urgency': 'High',
                            'source': 'inventory',
                            'confidence': 0.9,
                            'impact': 'Prevent stockout',
                            'priority_score': 90
                        })
            
            # Add surplus reduction recommendations
            if 'reduce_stock_recommendations' in inventory_recs:
                for rec in inventory_recs['reduce_stock_recommendations']:
                    if rec['excess_percentage'] > 50:
                        prioritized_recommendations.append({
                            'product_id': rec['product_id'],
                            'store_id': rec['store_id'],
                            'type': 'reduce_stock',
                            'action': f"{rec['recommendation_type']} {rec['recommended_reduction']} units of product {rec['product_id']} from store {rec['store_id']}",
                            'urgency': 'Medium',
                            'source': 'inventory',
                            'confidence': 0.8,
                            'impact': 'Reduce carrying costs',
                            'priority_score': 75
                        })
        
        # Process pricing agent recommendations
        if 'pricing' in agent_recommendations:
            pricing_recs = agent_recommendations['pricing']
            
            # Add high-impact price adjustments
            if 'price_adjustment_recommendations' in pricing_recs:
                for rec in pricing_recs['price_adjustment_recommendations']:
                    if rec['priority'] == 'High' and abs(rec['expected_revenue_impact']) > 15:
                        action = "Increase" if rec['price_change_percentage'] > 0 else "Decrease"
                        prioritized_recommendations.append({
                            'product_id': rec['product_id'],
                            'store_id': rec['store_id'],
                            'type': 'price_adjustment',
                            'action': f"{action} price of product {rec['product_id']} in store {rec['store_id']} by {abs(rec['price_change_percentage']):.1f}% to ${rec['recommended_price']:.2f}",
                            'urgency': 'Medium',
                            'source': 'pricing',
                            'confidence': rec['confidence'],
                            'impact': f"Expected {rec['expected_revenue_impact']:.1f}% revenue impact",
                            'priority_score': 80 if abs(rec['expected_revenue_impact']) > 20 else 70
                        })
        
        # Process demand agent recommendations
        if 'demand' in agent_recommendations:
            demand_recs = agent_recommendations['demand']
            
            # Add seasonality recommendations
            if 'seasonality_recommendations' in demand_recs:
                for rec in demand_recs['seasonality_recommendations']:
                    if rec['seasonality_strength'] > 0.5:
                        prioritized_recommendations.append({
                            'product_id': rec['product_id'],
                            'type': 'seasonal_preparation',
                            'action': rec['recommendation'],
                            'urgency': 'Medium',
                            'source': 'demand',
                            'confidence': rec['confidence'],
                            'impact': 'Capitalize on seasonal demand',
                            'priority_score': 75 if rec['seasonality_strength'] > 0.7 else 65
                        })
        
        # Add conflict resolution recommendations
        for resolution in conflict_resolution.get('resolutions', []):
            conflict = conflict_resolution.get('conflicts', [])[resolution['conflict_id']]
            
            prioritized_recommendations.append({
                'product_id': conflict['product_id'],
                'type': 'conflict_resolution',
                'action': resolution['action'],
                'rationale': resolution['rationale'],
                'urgency': 'Medium',
                'source': 'coordination',
                'confidence': 0.85,
                'impact': 'Resolve conflicting recommendations',
                'priority_score': 85  # High priority for conflict resolutions
            })
        
        # Sort recommendations by priority score
        prioritized_recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Update state with prioritized recommendations
        self.state['prioritized_recommendations'] = prioritized_recommendations
        
        # Group recommendations for output
        grouped_recommendations = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'high_priority': [rec for rec in prioritized_recommendations if rec['priority_score'] >= 80],
            'medium_priority': [rec for rec in prioritized_recommendations if 60 <= rec['priority_score'] < 80],
            'low_priority': [rec for rec in prioritized_recommendations if rec['priority_score'] < 60],
            'conflict_resolutions': [rec for rec in prioritized_recommendations if rec['type'] == 'conflict_resolution']
        }
        
        # Add recommendations to agent's recommendation history
        self.add_recommendation(
            f"{len(prioritized_recommendations)} integrated recommendations generated",
            confidence=0.9,
            context="Coordination across all agents"
        )
        
        # Save recommendations
        rec_file = os.path.join(self.output_dir, 'integrated_recommendations.json')
        with open(rec_file, 'w') as f:
            json.dump(grouped_recommendations, f, indent=2)
        
        self.log(f"Integrated recommendations saved to {rec_file}")
        return grouped_recommendations
    
    def update(self, feedback=None):
        """
        Update the agent's state based on feedback.
        
        Args:
            feedback: Dictionary with feedback on recommendations
            
        Returns:
            bool: True if update was successful
        """
        self.log("Updating coordination agent state based on feedback...")
        
        if feedback:
            # Process feedback on recommendations
            self.log(f"Received feedback: {feedback}")
            
            # Update optimization goals if provided
            if 'optimization_goals' in feedback:
                self.state['optimization_goals'].update(feedback['optimization_goals'])
                self.log("Updated optimization goals based on feedback")
            
            # If feedback includes accepted recommendations, track them
            if 'accepted_recommendations' in feedback:
                for rec in feedback['accepted_recommendations']:
                    self.log(f"Recommendation accepted: {rec}")
            
            # If feedback requests a re-analysis, perform it
            if feedback.get('reanalyze', False):
                self.analyze()
        
        # Save state after update
        self.save_state()
        return True
    
    def receive_message(self, sender, message, message_type='info'):
        """
        Process a message from another agent or external source.
        
        Args:
            sender: The name of the agent sending the message
            message: The content of the message
            message_type: The type of message (info, request, response)
            
        Returns:
            Any: Optional response to the message
        """
        # Call parent method to log the message
        super().receive_message(sender, message, message_type)
        
        response = None
        
        # Process different types of messages
        if message_type == 'request':
            if 'get_integrated_recommendations' in message:
                # Generate recommendations if needed
                if not self.state.get('prioritized_recommendations'):
                    self.make_recommendation()
                
                # Return integrated recommendations
                response = {
                    'integrated_recommendations': self.state.get('prioritized_recommendations', [])
                }
                self.log(f"Sending integrated recommendations to {sender}")
                
            elif 'get_conflict_resolutions' in message:
                # Return conflict resolutions
                response = {
                    'conflict_resolutions': self.state.get('conflict_resolution', {})
                }
                self.log(f"Sending conflict resolutions to {sender}")
                
            elif 'update_optimization_goals' in message and 'goals' in message:
                # Update optimization goals
                self.state['optimization_goals'].update(message['goals'])
                self.save_state()
                
                response = {
                    'result': 'success',
                    'updated_goals': self.state['optimization_goals']
                }
                self.log(f"Updated optimization goals based on request from {sender}")
        
        return response 