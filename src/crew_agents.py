"""
Crew AI implementation for Inventory Optimization System

This module uses CrewAI framework to orchestrate multiple agents that collaborate
to optimize inventory management policies. Each agent has specialized roles and abilities,
and they work together to analyze data, forecast demand, optimize inventory levels,
and recommend actions.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

try:
    from crewai import Agent, Task, Process, Crew
    from crewai.tasks import Task as CrewTask
    CREW_AI_AVAILABLE = True
except ImportError:
    # Mock objects if CrewAI is not available
    class Agent:
        def __init__(self, *args, **kwargs): pass
    class Task:
        def __init__(self, *args, **kwargs): pass
    class Process:
        SEQUENTIAL = "sequential"
        HIERARCHICAL = "hierarchical"
    class Crew:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass
    class CrewTask:
        def __init__(self, *args, **kwargs): pass
    CREW_AI_AVAILABLE = False
    
from src.tools import (
    calculate_reorder_point_tool, 
    calculate_safety_stock_tool,
    analyze_product_performance_tool,
    identify_anomalies_tool,
    forecast_demand_tool
)

logger = logging.getLogger(__name__)

class InventoryCrew:
    """
    CrewAI implementation for inventory optimization system.
    
    This class creates and orchestrates a crew of specialized agents
    that collaborate to optimize inventory management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the inventory crew with agents and configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Check if CrewAI is available
        if not CREW_AI_AVAILABLE:
            logger.warning("CrewAI package not found. Using limited functionality.")
            self.crew_available = False
            return
        
        self.crew_available = True
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Initialize tasks
        self.tasks = []
        
        # Create crew
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process.SEQUENTIAL,
            verbose=self.config.get('verbose', 2)
        )
        
        logger.info("InventoryCrew initialized with CrewAI")
    
    def _create_agents(self) -> Dict[str, Agent]:
        """
        Create specialized agents for inventory optimization.
        
        Returns:
            Dictionary of agents
        """
        agents = {}
        
        # Demand Forecasting Agent
        agents['demand'] = Agent(
            role="Demand Forecasting Specialist",
            goal="Analyze historical sales data to create accurate demand forecasts",
            backstory=(
                "You are an expert in demand forecasting with years of experience in "
                "retail and supply chain analytics. Your forecasts are known for their "
                "accuracy and ability to account for seasonality, trends, and special events."
            ),
            verbose=True,
            tools=[forecast_demand_tool, identify_anomalies_tool],
            allow_delegation=True
        )
        
        # Inventory Optimization Agent
        agents['inventory'] = Agent(
            role="Inventory Optimization Specialist",
            goal="Determine optimal inventory levels to minimize costs while maintaining service levels",
            backstory=(
                "You are an inventory management expert with deep knowledge of inventory "
                "theory and optimization techniques. Your recommendations consistently "
                "help companies reduce holding costs while avoiding stockouts."
            ),
            verbose=True,
            tools=[calculate_reorder_point_tool, calculate_safety_stock_tool],
            allow_delegation=True
        )
        
        # Supply Chain Agent
        agents['supply_chain'] = Agent(
            role="Supply Chain Manager",
            goal="Optimize the entire supply chain to improve efficiency and reduce lead times",
            backstory=(
                "You manage complex supply chains and are skilled at identifying "
                "bottlenecks and inefficiencies. Your holistic approach considers "
                "suppliers, transportation, warehousing, and distribution."
            ),
            verbose=True,
            tools=[analyze_product_performance_tool],
            allow_delegation=True
        )
        
        # Pricing Analyst Agent
        agents['pricing'] = Agent(
            role="Pricing Strategy Analyst",
            goal="Develop pricing strategies that maximize profit while considering inventory constraints",
            backstory=(
                "You are a pricing strategy expert who understands the relationship "
                "between price elasticity, demand, and inventory management. You create "
                "pricing policies that balance revenue goals with inventory optimization."
            ),
            verbose=True,
            allow_delegation=True
        )
        
        # Risk Management Agent
        agents['risk'] = Agent(
            role="Risk Management Specialist",
            goal="Identify and mitigate inventory and supply chain risks",
            backstory=(
                "You specialize in identifying and quantifying risks in inventory management "
                "and supply chains. Your risk mitigation strategies help companies prepare "
                "for disruptions and maintain business continuity."
            ),
            verbose=True,
            tools=[identify_anomalies_tool],
            allow_delegation=True
        )
        
        # Quality Assurance Agent
        agents['qa'] = Agent(
            role="Quality Assurance Expert",
            goal="Validate all analyses and recommendations for accuracy and feasibility",
            backstory=(
                "You have a keen eye for detail and a deep understanding of inventory "
                "management principles. You review all recommendations to ensure they are "
                "practical, accurate, and aligned with business objectives."
            ),
            verbose=True,
            allow_delegation=False
        )
        
        return agents
    
    def _create_tasks(self, data: Dict[str, pd.DataFrame]) -> List[CrewTask]:
        """
        Create tasks for the crew based on available data.
        
        Args:
            data: Dictionary of dataframes for analysis
            
        Returns:
            List of tasks
        """
        # Reset tasks
        self.tasks = []
        
        # 1. Analyze historical demand data
        self.tasks.append(Task(
            description=(
                "Analyze historical sales data to identify patterns, trends, and seasonality. "
                "Generate demand forecasts for each product for the next 30, 60, and 90 days."
            ),
            expected_output=(
                "A comprehensive demand forecast report including: "
                "1. Forecasted daily demand for each product "
                "2. Confidence intervals for the forecasts "
                "3. Identified seasonal patterns "
                "4. Anomalies in historical demand "
                "5. Key drivers of demand "
            ),
            agent=self.agents['demand'],
            context={
                "data": {
                    "sales": data.get('sales', pd.DataFrame()).to_dict('records') if isinstance(data.get('sales'), pd.DataFrame) else [],
                    "products": data.get('products', pd.DataFrame()).to_dict('records') if isinstance(data.get('products'), pd.DataFrame) else []
                }
            }
        ))
        
        # 2. Calculate optimal inventory levels
        self.tasks.append(Task(
            description=(
                "Using the demand forecasts, determine optimal inventory levels for each product. "
                "Calculate reorder points, safety stock levels, and economic order quantities. "
                "Consider lead times, service level targets, holding costs, and ordering costs."
            ),
            expected_output=(
                "An inventory optimization report including: "
                "1. Recommended min and max inventory levels for each product "
                "2. Reorder points "
                "3. Safety stock levels "
                "4. Economic order quantities "
                "5. Expected service levels "
                "6. Estimated holding costs "
            ),
            agent=self.agents['inventory'],
            context={
                "data": {
                    "inventory": data.get('inventory', pd.DataFrame()).to_dict('records') if isinstance(data.get('inventory'), pd.DataFrame) else [],
                    "products": data.get('products', pd.DataFrame()).to_dict('records') if isinstance(data.get('products'), pd.DataFrame) else []
                },
                "previous_task_output": "{demand_task_output}"
            }
        ))
        
        # 3. Analyze supply chain efficiency
        self.tasks.append(Task(
            description=(
                "Analyze the supply chain for inefficiencies and bottlenecks. "
                "Identify opportunities to reduce lead times and improve reliability. "
                "Evaluate supplier performance and recommend improvements."
            ),
            expected_output=(
                "A supply chain analysis report including: "
                "1. Identified bottlenecks in the supply chain "
                "2. Supplier performance metrics "
                "3. Recommendations for lead time reduction "
                "4. Strategies for improving supply chain reliability "
                "5. Cost-saving opportunities "
            ),
            agent=self.agents['supply_chain'],
            context={
                "data": {
                    "suppliers": data.get('suppliers', pd.DataFrame()).to_dict('records') if isinstance(data.get('suppliers'), pd.DataFrame) else [],
                    "inventory": data.get('inventory', pd.DataFrame()).to_dict('records') if isinstance(data.get('inventory'), pd.DataFrame) else []
                },
                "previous_task_output": "{inventory_task_output}"
            }
        ))
        
        # 4. Create pricing recommendations
        self.tasks.append(Task(
            description=(
                "Develop pricing strategies that balance revenue optimization with inventory management. "
                "Identify products where price adjustments could help manage inventory levels. "
                "Consider price elasticity and competitive positioning."
            ),
            expected_output=(
                "A pricing strategy report including: "
                "1. Recommended price adjustments for specific products "
                "2. Expected impact on demand and inventory "
                "3. Promotion recommendations for slow-moving inventory "
                "4. Price optimization for high-demand items "
                "5. Competitive pricing analysis "
            ),
            agent=self.agents['pricing'],
            context={
                "data": {
                    "sales": data.get('sales', pd.DataFrame()).to_dict('records') if isinstance(data.get('sales'), pd.DataFrame) else [],
                    "products": data.get('products', pd.DataFrame()).to_dict('records') if isinstance(data.get('products'), pd.DataFrame) else []
                },
                "previous_task_output": "{supply_chain_task_output}"
            }
        ))
        
        # 5. Perform risk assessment
        self.tasks.append(Task(
            description=(
                "Identify and assess risks related to inventory management and the supply chain. "
                "Calculate the likelihood and impact of stockouts, oversupply, supplier disruptions, and demand shocks. "
                "Develop mitigation strategies for each identified risk."
            ),
            expected_output=(
                "A risk assessment report including: "
                "1. Identified inventory and supply chain risks "
                "2. Risk scores based on likelihood and impact "
                "3. Early warning indicators for each risk "
                "4. Recommended mitigation strategies "
                "5. Contingency plans for high-impact risks "
            ),
            agent=self.agents['risk'],
            context={
                "data": {
                    "inventory": data.get('inventory', pd.DataFrame()).to_dict('records') if isinstance(data.get('inventory'), pd.DataFrame) else [],
                    "suppliers": data.get('suppliers', pd.DataFrame()).to_dict('records') if isinstance(data.get('suppliers'), pd.DataFrame) else []
                },
                "previous_task_output": "{pricing_task_output}"
            }
        ))
        
        # 6. Quality assurance review
        self.tasks.append(Task(
            description=(
                "Review all analyses and recommendations for accuracy, feasibility, and alignment with business objectives. "
                "Identify any contradictions or impractical recommendations. "
                "Ensure all calculations are correct and assumptions are valid."
            ),
            expected_output=(
                "A QA report including: "
                "1. Validation of demand forecasts and inventory recommendations "
                "2. Identification of any errors or inconsistencies "
                "3. Assessment of recommendation feasibility "
                "4. Suggested improvements or adjustments "
                "5. Final recommendations with confidence levels "
            ),
            agent=self.agents['qa'],
            context={
                "previous_task_outputs": {
                    "demand": "{demand_task_output}",
                    "inventory": "{inventory_task_output}",
                    "supply_chain": "{supply_chain_task_output}",
                    "pricing": "{pricing_task_output}",
                    "risk": "{risk_task_output}"
                }
            }
        ))
        
        return self.tasks
    
    def run_optimization(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run inventory optimization using CrewAI.
        
        Args:
            data: Dictionary of dataframes for analysis
            
        Returns:
            Dictionary with optimization results
        """
        if not self.crew_available:
            logger.error("CrewAI is not available. Cannot run optimization.")
            return {
                'status': 'error',
                'message': 'CrewAI is not available'
            }
        
        try:
            # Create tasks based on data
            self._create_tasks(data)
            
            # Update crew with new tasks
            self.crew.tasks = self.tasks
            
            # Run the crew
            logger.info("Running CrewAI optimization...")
            result = self.crew.run()
            
            # Process results
            processed_results = self._process_results(result)
            
            logger.info("CrewAI optimization completed successfully")
            return {
                'status': 'success',
                'agent_outputs': processed_results,
                'messages': self._extract_agent_messages(result),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error running CrewAI optimization: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _process_results(self, crew_result: Any) -> Dict[str, Any]:
        """
        Process and structure results from CrewAI.
        
        Args:
            crew_result: Raw output from CrewAI
            
        Returns:
            Structured results dictionary
        """
        processed_results = {}
        
        # Extract task results
        if hasattr(crew_result, 'task_outputs'):
            for task_name, task_output in crew_result.task_outputs.items():
                # Determine which agent
                agent_type = None
                for idx, task in enumerate(self.tasks):
                    if task_name == f"task_{idx+1}":
                        agent_type = task.agent.role.lower().replace(" specialist", "").replace(" expert", "").replace(" manager", "").replace(" analyst", "")
                        break
                
                if agent_type:
                    # Convert to structured format
                    processed_results[agent_type] = self._parse_task_output(task_output)
        
        return processed_results
    
    def _parse_task_output(self, output: str) -> Dict[str, Any]:
        """
        Parse unstructured text output into structured data.
        
        Args:
            output: Raw text output from an agent
            
        Returns:
            Structured data from the output
        """
        # This is a simplified parser - in a real implementation,
        # you would use more sophisticated NLP techniques
        structured_data = {
            'recommendations': [],
            'analysis': {},
            'raw_output': output
        }
        
        # Extract recommendations section
        if "recommendations:" in output.lower():
            rec_section = output.lower().split("recommendations:")[1].split("\n\n")[0]
            rec_lines = rec_section.strip().split("\n")
            for line in rec_lines:
                if line.strip():
                    structured_data['recommendations'].append({'text': line.strip()})
        
        # Extract metrics if available
        if "metrics:" in output.lower():
            metrics_section = output.lower().split("metrics:")[1].split("\n\n")[0]
            metrics_lines = metrics_section.strip().split("\n")
            for line in metrics_lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    structured_data['analysis'][key.strip()] = value.strip()
        
        return structured_data
    
    def _extract_agent_messages(self, result: Any) -> List[Dict[str, Any]]:
        """
        Extract agent messages from CrewAI result.
        
        Args:
            result: Result from CrewAI
            
        Returns:
            List of agent messages
        """
        messages = []
        
        # Extract messages if available
        if hasattr(result, 'agent_messages'):
            for msg in result.agent_messages:
                messages.append({
                    'from': msg.get('from', 'unknown'),
                    'to': msg.get('to', 'unknown'),
                    'content': msg.get('content', ''),
                    'timestamp': datetime.now().isoformat()
                })
        
        return messages 