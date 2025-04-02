from crewai import Agent, Task, Crew, Process
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

class InventoryCrew:
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = output_dir
        self.ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
        self.llm = self._setup_llm()
        self.agents = self._create_agents()
        
    def _setup_llm(self):
        """Set up the Ollama LLM for agents."""
        try:
            return Ollama(model=self.ollama_model)
        except Exception as e:
            print(f"Failed to initialize Ollama LLM: {e}")
            print("Falling back to default LLM...")
            return None
        
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for inventory optimization."""
        return {
            'data_analyst': Agent(
                role='Data Analyst',
                goal='Analyze historical inventory data and identify patterns',
                backstory="""You are an experienced data analyst specializing in retail analytics.
                Your expertise lies in identifying patterns and trends in inventory data.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            ),
            
            'forecasting_expert': Agent(
                role='Forecasting Expert',
                goal='Predict future demand and optimize inventory levels',
                backstory="""You are a forecasting expert with deep knowledge of time series analysis
                and demand prediction. You excel at creating accurate forecasts for retail inventory.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            ),
            
            'inventory_optimizer': Agent(
                role='Inventory Optimizer',
                goal='Optimize inventory levels and reorder points',
                backstory="""You are an inventory optimization specialist who knows how to balance
                costs while maintaining optimal stock levels. You understand various inventory
                management strategies and their trade-offs.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            ),
            
            'pricing_analyst': Agent(
                role='Pricing Analyst',
                goal='Optimize pricing strategies based on demand and inventory levels',
                backstory="""You are a pricing strategy expert who understands how to maximize
                revenue while maintaining healthy inventory turnover. You excel at dynamic pricing
                and promotional strategies.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            )
        }
    
    def _create_tasks(self, data: pd.DataFrame) -> List[Task]:
        """Create tasks for the crew to execute."""
        return [
            Task(
                description="""Analyze the historical inventory data to identify patterns and trends.
                Focus on seasonality, demand variability, and stockout patterns.
                Output should include key metrics and visualizations.""",
                agent=self.agents['data_analyst']
            ),
            
            Task(
                description="""Using the analyzed data, create demand forecasts for the next 30 days.
                Consider seasonality, trends, and any identified patterns.
                Provide confidence intervals and explain your methodology.""",
                agent=self.agents['forecasting_expert']
            ),
            
            Task(
                description="""Based on the forecasts, determine optimal inventory levels and reorder points.
                Consider holding costs, stockout costs, and lead times.
                Provide specific recommendations for inventory policy changes.""",
                agent=self.agents['inventory_optimizer']
            ),
            
            Task(
                description="""Develop pricing recommendations based on demand forecasts and inventory levels.
                Consider competitor pricing, market conditions, and inventory turnover goals.
                Provide specific pricing strategies and expected outcomes.""",
                agent=self.agents['pricing_analyst']
            )
        ]
    
    def run_optimization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the inventory optimization process using the crew."""
        tasks = self._create_tasks(data)
        
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            verbose=2,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        # Process and structure the results
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': result,
            'metrics': self._extract_metrics(result)
        }
        
        return optimization_results
    
    def _extract_metrics(self, results: str) -> Dict[str, float]:
        """Extract key metrics from the crew's results."""
        # This is a placeholder - implement actual metric extraction based on your needs
        return {
            'forecast_accuracy': 0.0,
            'inventory_turnover': 0.0,
            'stockout_rate': 0.0,
            'holding_cost': 0.0
        } 