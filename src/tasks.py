"""
Task definitions for the Multi-Agent Inventory Optimization System.
"""

from crewai import Task
from typing import List, Dict, Any
import pandas as pd
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class InventoryTasks:
    """Manages the definition and creation of tasks for the inventory optimization crew."""
    
    def __init__(self, inventory_data: pd.DataFrame, demand_data: pd.DataFrame):
        """Initializes tasks with shared data and parameters."""
        if inventory_data is None or inventory_data.empty:
            logger.error("Inventory data is missing or empty.")
            raise ValueError("Valid inventory data is required.")
        if demand_data is None or demand_data.empty:
            logger.error("Demand data is missing or empty.")
            raise ValueError("Valid demand data is required.")
            
        self.max_retries = 2 # Reduce max retries slightly
        self.task_timeout = 300  # 5 minutes timeout per task
        
        # Pre-calculate necessary statistics or context for tasks
        try:
            # Define required columns for each dataframe
            inv_required_cols = ['Stock Levels', 'Supplier Lead Time (days)'] # Add others if used: 'Product ID', 'Store ID'?
            demand_required_cols = ['Sales Quantity', 'Date'] # Add others if used: 'Product ID', 'Store ID'?
            
            inv_missing_cols = [col for col in inv_required_cols if col not in inventory_data.columns]
            demand_missing_cols = [col for col in demand_required_cols if col not in demand_data.columns]
            
            errors = []
            if inv_missing_cols:
                errors.append(f"Inventory data missing columns: {inv_missing_cols}")
            if demand_missing_cols:
                errors.append(f"Demand data missing columns: {demand_missing_cols}")
                
            if errors:
                raise ValueError("Data validation failed: " + "; ".join(errors))
                
            # Calculate stats using correct dataframes and column names
            self.avg_daily_demand = demand_data['Sales Quantity'].mean()
            self.demand_std = demand_data['Sales Quantity'].std()
            self.avg_inventory_level = inventory_data['Stock Levels'].mean()
            self.avg_lead_time = int(inventory_data['Supplier Lead Time (days)'].mean())
            self.historical_sales_list = demand_data['Sales Quantity'].tolist()
            self.current_inventory_list = inventory_data['Stock Levels'].tolist()
            
            logger.debug(f"Task context calculated: Avg Demand={self.avg_daily_demand:.2f}, Demand Std={self.demand_std:.2f}, Avg Inv={self.avg_inventory_level:.2f}, Avg Lead Time={self.avg_lead_time}")
        except KeyError as e:
            logger.error(f"Error accessing expected column in data: {e}. Check required_cols vs CSV headers.")
            raise
        except Exception as e:
            logger.error(f"Error during task initialization data processing: {e}")
            raise

    def _create_task(self, description: str, expected_output: str, agent, context: List[Task] = None) -> Task:
        """Helper function to create a Task instance with common parameters."""
        task_params = {
            "description": description,
            "expected_output": expected_output,
            "agent": agent,
            "max_retries": self.max_retries,
        }
        if context:
            # Ensure context is always a list
            task_params["context"] = context 
        
        return Task(**task_params)

    # --- Task Definitions --- 

    def create_demand_forecasting_task(self, agent) -> Task:
        """Task for the Demand Analyst to forecast future demand."""
        return self._create_task(
            description=(
                f"Analyze historical sales data to forecast future demand. "
                f"Consider patterns, seasonality (if identifiable), and recent trends present *within the provided data*. "
                f"Use the 'forecast_demand' tool with the historical sales data provided below. The tool expects a dictionary containing 'historical_demand' (a list of numbers) and 'period' (an integer, typically 3). "
                f"Only use the search tool if you need to investigate a *specific* external factor known to be relevant to the *time period* of the historical data. Avoid searching for general, current market trends as they may not apply to this dataset.\\n\\n" 
                f"Historical Sales Data: {self.historical_sales_list}\\n"
                f"Average Daily Demand (calculated): {self.avg_daily_demand:.2f}\\n"
                f"Standard Deviation of Demand: {self.demand_std:.2f}"
            ),
            expected_output=(
                "A demand forecast report including: "
                "1. Forecasted demand figure for the next period (e.g., using the tool output like 'Forecasted Demand (next period, 3-period SMA): 193.33 units'). "
                "2. Key assumptions made (e.g., using a 3-period SMA). "
                "3. Confidence level (State N/A if the tool doesn't provide one). "
                "Provide your final response using the 'Final Answer:' format. Ensure the forecast value is clearly stated."
            ),
            agent=agent
        )

    def create_safety_stock_calculation_task(self, agent, demand_task: Task) -> Task:
         """Task for the Inventory Optimizer OR Supply Chain Analyst to calculate safety stock."""
         return self._create_task(
            description=(
                "Based on the demand forecast (especially demand variability/standard deviation) "
                "and supply chain lead times, calculate the optimal safety stock levels. "
                "Use the 'calculate_safety_stock' tool. Assume a target service level of 95% "
                "(Z-score=1.65) unless specified otherwise. "
                "Use the Demand Standard Deviation provided below directly in your calculation. "
                "Also, use the 'analyze_stockout_risk' tool to assess current risk "
                "using the full 'Current Inventory Levels' list provided below before "
                "calculating the new safety stock.\\n\\n"
                "Key Inputs from Context:\\n"
                f"- Demand Standard Deviation: {self.demand_std:.2f}\\n"
                f"- Average Lead Time: {self.avg_lead_time} days\\n"
                f"- Current Inventory Levels: {self.current_inventory_list}\\n"
            ),
            expected_output=(
                "A report detailing the calculated safety stock level (in units), "
                "the Z-score/service level used (e.g., 95% service level with Z=1.65), "
                "justification for the chosen service level, "
                "and the assessment of the current stockout risk ('Low', 'Medium', 'High')."
            ),
            agent=agent,
            context=[demand_task]
        )
        
    def create_reorder_point_calculation_task(self, agent, demand_task: Task, safety_stock_task: Task) -> Task:
        """Task for the Inventory Optimizer to calculate reorder points."""
        return self._create_task(
            description=(
                "Using the demand forecast (average daily demand) and the calculated safety stock "
                "level, determine the optimal reorder points (ROP) for key products. "
                "Use the 'calculate_reorder_point' tool. Ensure you extract the safety "
                "stock value from the previous task's output context.\\n\\n"
                "Key Inputs from Context:\\n"
                f"- Average Daily Demand: {self.avg_daily_demand:.2f}\\n"
                f"- Average Lead Time: {self.avg_lead_time} days\\n"
                "- Safety Stock: (Use output from previous task)"
            ),
            expected_output=(
                "A report stating the calculated Reorder Point (ROP) in units, and confirming the "
                "demand, lead time, and safety stock values used in the calculation."
            ),
            agent=agent,
            context=[demand_task, safety_stock_task] 
        )
        
    def create_overall_optimization_report_task(self, agent, context_tasks: List[Task]) -> Task:
        """Final task to synthesize findings and create a consolidated report."""
        return self._create_task(
            description=(
                "Review the 'Final Answer:' sections from ALL previous task outputs available in the context. "
                "Synthesize the findings into a single, cohesive inventory optimization strategy report "
                "using the exact format specified in the expected output. "
                "Extract the final key numbers (Forecasted Demand, Calculated Safety Stock, Calculated ROP) "
                "directly from the context of the previous tasks. "
                "Highlight key recommendations and implementation steps based on these numbers. "
                "**ABSOLUTELY DO NOT use any calculation tools or perform any new calculations.** "
                "Focus *only* on reporting the results extracted from previous tasks."
            ),
            expected_output=(
                "A comprehensive inventory optimization strategy report including:\\n"
                "1. **Demand Forecast Summary:**\\n"
                "   - Forecasted Demand for next period: [Extract value from Demand Forecast Task Context]\\n"
                "   - Key Assumptions/Method: [Extract from Demand Forecast Task Context]\\n"
                "2. **Inventory Level Recommendations:**\\n"
                "   - Calculated Optimal Safety Stock: [Extract value from Safety Stock Task Context]\\n"
                "   - Calculated Reorder Point (ROP): [Extract value from ROP Task Context]\\n"
                "   - Target Service Level Used: [Extract from Safety Stock Task Context, e.g., 95%]\\n"
                "3. **Risk Assessment:**\\n"
                "   - Current Stockout Risk Assessment: [Extract from Safety Stock Task Context]\\n"
                "   - Impact of Recommendations on Risk: [Synthesize based on extracted values]\\n"
                "4. **Integrated Recommendations & Implementation Steps:**\\n"
                "   - Summarize the key actions based on the calculated figures.\\n"
                "   - Suggest steps for implementing the new safety stock and ROP levels.\\n"
                "   - Recommend monitoring frequency and adjustment triggers."
                "   - [Ensure all sections above are populated with extracted values]"
            ),
            agent=agent, 
            context=context_tasks # Depends on all previous tasks
        )

    def get_all_tasks(self, agents: Dict[str, Any]) -> List[Task]:
        """Creates and returns the sequence of tasks for the crew."""
        logger.debug("Creating all inventory optimization tasks...")
        try:
            # Assign agents to roles clearly for task assignment
            demand_analyst = agents['demand_analyst']
            inventory_optimizer = agents['inventory_optimizer']
            risk_analyst = agents['risk_analyst'] # Could be assigned the safety stock task? Or optimizer does it.

            # Define the sequence of tasks
            demand_task = self.create_demand_forecasting_task(demand_analyst)
            
            # Assign safety stock calc to Inventory Optimizer (could also be risk_analyst)
            safety_stock_task = self.create_safety_stock_calculation_task(inventory_optimizer, demand_task)
            
            reorder_point_task = self.create_reorder_point_calculation_task(inventory_optimizer, demand_task, safety_stock_task)
            
            # Final report task - assign to the Inventory Optimizer as the lead
            report_task = self.create_overall_optimization_report_task(
                inventory_optimizer, 
                [demand_task, safety_stock_task, reorder_point_task] # Removed pricing_task from context
            )

            tasks = [demand_task, safety_stock_task, reorder_point_task, report_task] # Removed pricing task
            logger.info(f"Successfully created {len(tasks)} tasks.")
            return tasks
            
        except KeyError as e:
            logger.error(f"Error assigning agent to task. Agent key '{e}' not found in provided agents dictionary.")
            raise
        except Exception as e:
            logger.error(f"Error creating tasks: {e}", exc_info=True)
            raise 