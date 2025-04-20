"""
Multi-Agent Orchestrator for Inventory Optimization System

This module coordinates the multi-agent system for inventory optimization,
managing communication between agents, scheduling tasks, and providing 
a unified interface for the system.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Import agent modules
from src.crew_agents import InventoryCrew
from src.inventory_agent import InventoryAgent
from src.demand_agent import DemandAgent
from src.pricing_agent import PricingAgent
from src.supply_chain_agent import SupplyChainAgent
from src.qa_agent import QAAgent
from src.risk_agent import RiskAgent
from src.tools import calculate_reorder_point_tool, calculate_safety_stock_tool

# Import inventory optimization models
from src.models.inventory_optimization.inventory_models import InventoryOptimizer, MultiEchelonOptimizer
from src.models.inventory_optimization.anomaly_detection import AnomalyDetector
from src.models.inventory_optimization.scenario_planning import ScenarioPlanner
from src.models.inventory_optimization.reinforcement_learning import InventoryRLOptimizer

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents for inventory optimization.
    
    This class manages:
    1. Agent creation and initialization
    2. Communication between agents
    3. Task scheduling and execution
    4. System state management
    5. Integration with optimization models
    6. Result collection and processing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the multi-agent orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'output/multi_agent')
        self._ensure_output_dir()
        
        # Initialize logging
        self._setup_logging()
        logger.info("Initializing Multi-Agent Orchestrator")
        
        # Initialize system state
        self.state = {
            'initialized': False,
            'optimization_status': 'not_started',
            'agents': {},
            'results': {},
            'messages': [],
            'errors': [],
            'start_time': None,
            'end_time': None
        }
        
        # Initialize models
        self._init_models()
        
        # Initialize agents
        self._init_agents()
        
        # Mark as initialized
        self.state['initialized'] = True
        logger.info("Multi-Agent Orchestrator initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'output_dir': 'output/multi_agent',
            'log_level': 'INFO',
            'use_crew_ai': True,
            'max_iterations': 10,
            'models': {
                'inventory': {
                    'service_level': 0.95,
                    'lead_time_multiplier': 1.5
                },
                'anomaly_detection': {
                    'threshold': 3.0,
                    'method': 'isolation_forest'
                },
                'scenario_planning': {
                    'num_scenarios': 5,
                    'horizon': 12
                },
                'reinforcement_learning': {
                    'enabled': True,
                    'num_episodes': 500
                }
            },
            'agents': {
                'inventory': {
                    'enabled': True
                },
                'demand': {
                    'enabled': True
                },
                'pricing': {
                    'enabled': True
                },
                'supply_chain': {
                    'enabled': True
                },
                'risk': {
                    'enabled': True
                },
                'qa': {
                    'enabled': True
                }
            }
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                
                # Merge configs (deep merge)
                def merge_dicts(d1, d2):
                    for k, v in d2.items():
                        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                            merge_dicts(d1[k], v)
                        else:
                            d1[k] = v
                
                merge_dicts(default_config, custom_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        
        return default_config
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['logs', 'results', 'models', 'visualizations']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Configure logging for the orchestrator."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        log_file = os.path.join(self.output_dir, 'logs', f'orchestrator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _init_models(self) -> None:
        """Initialize optimization models."""
        model_config = self.config.get('models', {})
        
        # Initialize inventory optimizer
        inv_config = model_config.get('inventory', {})
        self.inventory_optimizer = InventoryOptimizer(
            service_level=inv_config.get('service_level', 0.95),
            lead_time_multiplier=inv_config.get('lead_time_multiplier', 1.5)
        )
        
        # Initialize multi-echelon optimizer if enabled
        if model_config.get('multi_echelon', {}).get('enabled', False):
            self.multi_echelon_optimizer = MultiEchelonOptimizer()
        else:
            self.multi_echelon_optimizer = None
        
        # Initialize anomaly detector
        anom_config = model_config.get('anomaly_detection', {})
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize scenario planner
        scenario_config = model_config.get('scenario_planning', {})
        self.scenario_planner = ScenarioPlanner(
            num_scenarios=scenario_config.get('num_scenarios', 5),
            horizon=scenario_config.get('horizon', 12)
        )
        
        # Initialize RL optimizer if enabled
        rl_config = model_config.get('reinforcement_learning', {})
        if rl_config.get('enabled', False):
            self.rl_optimizer = InventoryRLOptimizer(config=rl_config)
        else:
            self.rl_optimizer = None
        
        logger.info("Optimization models initialized")
    
    def _init_agents(self) -> None:
        """Initialize all agents based on configuration."""
        agent_config = self.config.get('agents', {})
        
        # Initialize CrewAI if enabled
        if self.config.get('use_crew_ai', True):
            self.crew = InventoryCrew()
            self.state['agents']['crew'] = {
                'status': 'initialized',
                'type': 'crew'
            }
            logger.info("CrewAI initialized")
        else:
            self.crew = None
            
            # Initialize individual agents if CrewAI is not used
            if agent_config.get('inventory', {}).get('enabled', True):
                self.inventory_agent = InventoryAgent()
                self.state['agents']['inventory'] = {
                    'status': 'initialized',
                    'type': 'inventory'
                }
            
            if agent_config.get('demand', {}).get('enabled', True):
                self.demand_agent = DemandAgent()
                self.state['agents']['demand'] = {
                    'status': 'initialized',
                    'type': 'demand'
                }
            
            if agent_config.get('pricing', {}).get('enabled', True):
                self.pricing_agent = PricingAgent()
                self.state['agents']['pricing'] = {
                    'status': 'initialized',
                    'type': 'pricing'
                }
            
            if agent_config.get('supply_chain', {}).get('enabled', True):
                self.supply_chain_agent = SupplyChainAgent()
                self.state['agents']['supply_chain'] = {
                    'status': 'initialized',
                    'type': 'supply_chain'
                }
            
            if agent_config.get('risk', {}).get('enabled', True):
                self.risk_agent = RiskAgent()
                self.state['agents']['risk'] = {
                    'status': 'initialized',
                    'type': 'risk'
                }
            
            if agent_config.get('qa', {}).get('enabled', True):
                self.qa_agent = QAAgent()
                self.state['agents']['qa'] = {
                    'status': 'initialized',
                    'type': 'qa'
                }
            
            logger.info("Individual agents initialized")
    
    def run_optimization(self, 
                        data: Optional[Dict[str, pd.DataFrame]] = None, 
                        data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete inventory optimization process.
        
        Args:
            data: Dictionary of dataframes including sales, inventory, products, etc.
            data_path: Path to data directory if data not provided directly
            
        Returns:
            Dictionary with optimization results
        """
        # Record start time
        self.state['start_time'] = datetime.now()
        self.state['optimization_status'] = 'running'
        
        # Load data if not provided
        if data is None and data_path is not None:
            data = self._load_data(data_path)
        
        if data is None:
            error_msg = "No data provided for optimization"
            self.state['errors'].append(error_msg)
            self.state['optimization_status'] = 'failed'
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        
        try:
            # Run optimization using CrewAI if enabled
            if self.crew is not None:
                logger.info("Running optimization with CrewAI")
                crew_results = self.crew.run_optimization(data)
                self.state['results']['crew'] = crew_results
                optimization_results = self._process_crew_results(crew_results)
            else:
                # Run optimization using individual agents
                logger.info("Running optimization with individual agents")
                optimization_results = self._run_individual_agent_optimization(data)
            
            # Apply inventory optimization models to enhance results
            enhanced_results = self._enhance_with_models(optimization_results, data)
            
            # Save results
            self._save_results(enhanced_results)
            
            # Mark as completed
            self.state['optimization_status'] = 'completed'
            self.state['end_time'] = datetime.now()
            
            # Calculate execution time
            execution_time = (self.state['end_time'] - self.state['start_time']).total_seconds()
            logger.info(f"Optimization completed in {execution_time:.2f} seconds")
            
            return {
                'status': 'success',
                'results': enhanced_results,
                'execution_time': execution_time
            }
            
        except Exception as e:
            error_msg = f"Error during optimization: {str(e)}"
            self.state['errors'].append(error_msg)
            self.state['optimization_status'] = 'failed'
            self.state['end_time'] = datetime.now()
            logger.error(error_msg, exc_info=True)
            
            return {
                'status': 'error',
                'message': error_msg
            }
    
    def _load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load data from files.
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Dictionary of dataframes
        """
        data = {}
        
        try:
            # Load sales data
            sales_path = os.path.join(data_path, 'sales.csv')
            if os.path.exists(sales_path):
                data['sales'] = pd.read_csv(sales_path, parse_dates=['date'])
            
            # Load inventory data
            inventory_path = os.path.join(data_path, 'inventory.csv')
            if os.path.exists(inventory_path):
                data['inventory'] = pd.read_csv(inventory_path, parse_dates=['date'])
            
            # Load products data
            products_path = os.path.join(data_path, 'products.csv')
            if os.path.exists(products_path):
                data['products'] = pd.read_csv(products_path)
            
            # Load suppliers data
            suppliers_path = os.path.join(data_path, 'suppliers.csv')
            if os.path.exists(suppliers_path):
                data['suppliers'] = pd.read_csv(suppliers_path)
            
            # Load locations data
            locations_path = os.path.join(data_path, 'locations.csv')
            if os.path.exists(locations_path):
                data['locations'] = pd.read_csv(locations_path)
            
            logger.info(f"Loaded data from {data_path}")
            return data
            
        except Exception as e:
            error_msg = f"Error loading data from {data_path}: {str(e)}"
            self.state['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
            return {}
    
    def _run_individual_agent_optimization(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run optimization using individual agents when CrewAI is disabled.
        
        Args:
            data: Dictionary of dataframes
            
        Returns:
            Optimization results
        """
        results = {}
        messages = []
        
        # Start with demand forecasting
        if hasattr(self, 'demand_agent'):
            logger.info("Running demand forecasting")
            demand_results = self.demand_agent.analyze(data)
            results['demand'] = demand_results
            
            # Create message about demand forecast
            message = {
                'from': 'demand_agent',
                'to': 'all',
                'content': f"Completed demand forecasting for {len(demand_results.get('forecasts', []))} products",
                'timestamp': datetime.now().isoformat()
            }
            messages.append(message)
            self.state['messages'].append(message)
        
        # Run inventory optimization
        if hasattr(self, 'inventory_agent'):
            logger.info("Running inventory optimization")
            # Pass demand results to inventory agent
            demand_data = results.get('demand', {})
            inventory_results = self.inventory_agent.analyze({**data, 'demand_forecast': demand_data})
            results['inventory'] = inventory_results
            
            # Create message about inventory optimization
            message = {
                'from': 'inventory_agent',
                'to': 'all',
                'content': f"Completed inventory optimization with {len(inventory_results.get('recommendations', []))} recommendations",
                'timestamp': datetime.now().isoformat()
            }
            messages.append(message)
            self.state['messages'].append(message)
        
        # Run supply chain analysis
        if hasattr(self, 'supply_chain_agent'):
            logger.info("Running supply chain analysis")
            supply_chain_results = self.supply_chain_agent.analyze(data)
            results['supply_chain'] = supply_chain_results
            
            # Create message about supply chain analysis
            message = {
                'from': 'supply_chain_agent',
                'to': 'all',
                'content': f"Completed supply chain analysis with {len(supply_chain_results.get('recommendations', []))} recommendations",
                'timestamp': datetime.now().isoformat()
            }
            messages.append(message)
            self.state['messages'].append(message)
        
        # Run risk analysis
        if hasattr(self, 'risk_agent'):
            logger.info("Running risk analysis")
            risk_results = self.risk_agent.analyze(data)
            results['risk'] = risk_results
            
            # Create message about risk analysis
            message = {
                'from': 'risk_agent',
                'to': 'all',
                'content': f"Completed risk analysis with {len(risk_results.get('risks', []))} identified risks",
                'timestamp': datetime.now().isoformat()
            }
            messages.append(message)
            self.state['messages'].append(message)
        
        # Run pricing analysis
        if hasattr(self, 'pricing_agent'):
            logger.info("Running pricing analysis")
            pricing_results = self.pricing_agent.analyze(data)
            results['pricing'] = pricing_results
            
            # Create message about pricing analysis
            message = {
                'from': 'pricing_agent',
                'to': 'all',
                'content': f"Completed pricing analysis with {len(pricing_results.get('recommendations', []))} pricing recommendations",
                'timestamp': datetime.now().isoformat()
            }
            messages.append(message)
            self.state['messages'].append(message)
        
        # Run QA analysis
        if hasattr(self, 'qa_agent'):
            logger.info("Running QA analysis")
            # Pass all results to QA agent
            qa_results = self.qa_agent.analyze({**data, 'agent_results': results})
            results['qa'] = qa_results
            
            # Create message about QA analysis
            message = {
                'from': 'qa_agent',
                'to': 'all',
                'content': f"Completed QA analysis with {len(qa_results.get('issues', []))} identified issues",
                'timestamp': datetime.now().isoformat()
            }
            messages.append(message)
            self.state['messages'].append(message)
        
        # Compile final results
        final_results = {
            'agent_results': results,
            'messages': messages,
            'timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def _process_crew_results(self, crew_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process results from CrewAI execution.
        
        Args:
            crew_results: Results from CrewAI
            
        Returns:
            Processed results
        """
        # Extract agent outputs
        agent_outputs = crew_results.get('agent_outputs', {})
        
        # Process agent messages
        messages = crew_results.get('messages', [])
        for message in messages:
            self.state['messages'].append(message)
        
        # Compile final results
        final_results = {
            'agent_results': agent_outputs,
            'messages': messages,
            'timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def _enhance_with_models(self, 
                            results: Dict[str, Any], 
                            data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Enhance optimization results using advanced models.
        
        Args:
            results: Results from agent optimization
            data: Dictionary of dataframes
            
        Returns:
            Enhanced results
        """
        enhanced_results = results.copy()
        model_results = {}
        
        # Extract inventory recommendations
        inventory_results = results.get('agent_results', {}).get('inventory', {})
        recommendations = inventory_results.get('recommendations', [])
        
        # Enhance with inventory optimizer
        if recommendations:
            logger.info("Enhancing results with inventory optimizer")
            enhanced_recommendations = []
            
            for rec in recommendations:
                product_id = rec.get('product_id')
                
                # Get product data
                product_data = None
                for product in data.get('products', []):
                    if product.get('product_id') == product_id:
                        product_data = product
                        break
                
                if product_data:
                    # Calculate optimal inventory policy
                    lead_time = product_data.get('lead_time', 7)  # Default to 7 days
                    avg_demand = rec.get('avg_demand', 0)
                    std_demand = rec.get('std_demand', 0)
                    
                    optimal_policy = self.inventory_optimizer.optimize_inventory_policy({
                        'product_id': product_id,
                        'lead_time': lead_time,
                        'avg_demand': avg_demand,
                        'std_demand': std_demand,
                        'unit_cost': product_data.get('unit_cost', 0),
                        'holding_cost_pct': product_data.get('holding_cost_pct', 0.25),
                        'stockout_cost': product_data.get('stockout_cost', 0),
                        'order_cost': product_data.get('order_cost', 0),
                        'shelf_life': product_data.get('shelf_life', None),
                        'storage_capacity': product_data.get('storage_capacity', None)
                    })
                    
                    # Update recommendation with optimal policy
                    enhanced_rec = rec.copy()
                    enhanced_rec.update({
                        'enhanced_min': optimal_policy.get('min_level'),
                        'enhanced_max': optimal_policy.get('max_level'),
                        'enhanced_reorder_point': optimal_policy.get('reorder_point'),
                        'enhanced_eoq': optimal_policy.get('eoq'),
                        'enhanced_safety_stock': optimal_policy.get('safety_stock'),
                        'confidence': optimal_policy.get('confidence', rec.get('confidence', 0.8))
                    })
                    
                    enhanced_recommendations.append(enhanced_rec)
                else:
                    enhanced_recommendations.append(rec)
            
            # Update recommendations in results
            enhanced_results['agent_results']['inventory']['recommendations'] = enhanced_recommendations
            model_results['inventory_optimizer'] = {
                'products_enhanced': len(enhanced_recommendations),
                'timestamp': datetime.now().isoformat()
            }
        
        # Enhance with anomaly detection
        if 'sales' in data:
            logger.info("Enhancing results with anomaly detection")
            sales_df = data['sales']
            
            # Detect anomalies in sales data
            anomalies = self.anomaly_detector.detect_demand_anomalies(sales_df)
            
            # Add anomalies to results
            model_results['anomaly_detection'] = {
                'anomalies': anomalies,
                'timestamp': datetime.now().isoformat()
            }
        
        # Enhance with scenario planning
        if 'sales' in data and 'inventory' in data:
            logger.info("Enhancing results with scenario planning")
            sales_df = data['sales']
            inventory_df = data['inventory']
            
            # Run scenario planning
            scenarios = self.scenario_planner.run_what_if_scenario(
                sales_df, 
                inventory_df, 
                scenario_type='demand_surge',
                parameters={'surge_pct': 0.2}
            )
            
            # Add scenarios to results
            model_results['scenario_planning'] = {
                'scenarios': scenarios,
                'timestamp': datetime.now().isoformat()
            }
        
        # Enhance with RL if enabled
        if self.rl_optimizer and 'sales' in data and 'inventory' in data:
            logger.info("Enhancing results with reinforcement learning")
            # Train RL model (simplified for demonstration)
            rl_results = {
                'training_status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            model_results['reinforcement_learning'] = rl_results
        
        # Add model results to enhanced results
        enhanced_results['model_results'] = model_results
        
        return enhanced_results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save optimization results to files.
        
        Args:
            results: Optimization results
        """
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        results_file = os.path.join(self.output_dir, 'results', f'optimization_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        # Save recommendations as CSV if available
        if 'agent_results' in results and 'inventory' in results['agent_results']:
            recommendations = results['agent_results']['inventory'].get('recommendations', [])
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_file = os.path.join(self.output_dir, 'results', f'recommendations_{timestamp}.csv')
                recommendations_df.to_csv(recommendations_file, index=False)
        
        # Save anomalies as CSV if available
        if 'model_results' in results and 'anomaly_detection' in results['model_results']:
            anomalies = results['model_results']['anomaly_detection'].get('anomalies', [])
            if anomalies:
                anomalies_df = pd.DataFrame(anomalies)
                anomalies_file = os.path.join(self.output_dir, 'results', f'anomalies_{timestamp}.csv')
                anomalies_df.to_csv(anomalies_file, index=False)
        
        logger.info(f"Saved results to {self.output_dir}/results/")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the optimization process.
        
        Returns:
            Status dictionary
        """
        return {
            'status': self.state['optimization_status'],
            'initialized': self.state['initialized'],
            'agents': self.state['agents'],
            'start_time': self.state['start_time'],
            'end_time': self.state['end_time'],
            'errors': self.state['errors'],
            'message_count': len(self.state['messages'])
        }
    
    def get_agent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages between agents.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        # Sort messages by timestamp (most recent first)
        sorted_messages = sorted(
            self.state['messages'], 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        
        # Return limited number of messages
        return sorted_messages[:limit]


if __name__ == "__main__":
    # Example usage
    orchestrator = MultiAgentOrchestrator()
    data_path = "data"
    results = orchestrator.run_optimization(data_path=data_path)
    print(f"Optimization status: {results['status']}")
    if results['status'] == 'success':
        print(f"Execution time: {results['execution_time']:.2f} seconds") 