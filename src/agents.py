from crewai import Agent
from langchain_community.llms import Ollama  # Import Ollama
from typing import List, Dict, Any
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import specific tool lists from the new tools module
from .tools import (
    demand_analyst_tools,
    inventory_optimizer_tools,
    supply_chain_analyst_tools,
    risk_analyst_tools
)

logger = logging.getLogger(__name__)

class InventoryAgents:
    def __init__(self, model_name: str = "llama3", ollama_base_url: str = "http://localhost:11434"):
        self.max_execution_time = 300  # 5 minutes timeout per task
        self.max_retries = 3
        
        # Instantiate the Ollama LLM
        try:
            self.llm = Ollama(
                model=model_name,
                base_url=ollama_base_url
            )
            logger.info(f"Successfully connected to Ollama model '{model_name}' at {ollama_base_url}")
        except Exception as e:
            logger.error(f"Failed to connect or initialize Ollama model '{model_name}' at {ollama_base_url}: {e}")
            logger.error("Ensure Ollama server is running and the model is available.")
            raise ConnectionError(f"Could not connect to Ollama: {e}")

    def _create_agent(self, role: str, goal: str, backstory: str, tools: List) -> Agent:
        """Helper method to create an agent with shared configurations."""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm,  # Pass the instantiated LLM object
            tools=tools,   # Pass the specific tool list for the agent
            verbose=True,
            max_iter=15, # Limit iterations to prevent excessive loops
            max_rpm=20, # Limit requests per minute (adjust based on Ollama/hardware)
            max_execution_time=self.max_execution_time,
            max_retries=self.max_retries,
            allow_delegation=False, # Keep delegation off for now
            memory=True
        )

    def create_demand_analyst(self) -> Agent:
        return self._create_agent(
            role='Lead Demand Forecasting Analyst',
            goal=(
                'Analyze historical sales data, market trends, and external factors (e.g., promotions, seasonality) ' 
                'to generate accurate demand forecasts for various products. ' 
                'Identify key demand drivers and patterns.'
            ),
            backstory=(
                """You are an expert Demand Forecasting Analyst with a strong background in statistics and machine learning. "
                "Your primary responsibility is to predict future customer demand accurately. You meticulously analyze past sales, "
                "consider market dynamics, and incorporate information about upcoming marketing campaigns or seasonal events. "
                "Your forecasts are crucial for inventory planning and supply chain operations. You leverage available tools "
                "to analyze trends and generate reliable demand predictions."""
            ),
            tools=demand_analyst_tools
        )

    def create_inventory_optimizer(self) -> Agent:
        return self._create_agent(
            role='Inventory Optimization Manager',
            goal=(
                'Optimize inventory levels across all products and locations to minimize holding costs, ordering costs, and stockout costs. ' 
                'Calculate optimal safety stock and reorder points based on forecasts, lead times, and service level targets.'
            ),
            backstory=(
                """You are a seasoned Inventory Optimization Manager specializing in quantitative analysis and supply chain efficiency. "
                "You are responsible for maintaining the delicate balance between having enough stock to meet demand and minimizing excess inventory. "
                "Using demand forecasts, supplier lead times, cost data, and service level goals, you determine the most cost-effective inventory policies. "
                "You utilize calculation tools to find optimal safety stock and reorder points."""
            ),
            tools=inventory_optimizer_tools
        )

    def create_supply_chain_analyst(self) -> Agent:
        return self._create_agent(
            role='Supply Chain Orchestration Director',
            goal=(
                'Analyze supply chain constraints, supplier reliability, lead times, and logistics costs. ' 
                'Coordinate with suppliers and logistics providers to ensure timely replenishment and mitigate supply risks. ' 
                'Evaluate the impact of supply chain performance on inventory levels.'
             ),
            backstory=(
                """You are a strategic Supply Chain Director overseeing the end-to-end flow of goods. You focus on supplier relationships, "
                "lead time variability, transportation efficiency, and potential disruptions. Your analysis helps in understanding "
                "how supply-side factors influence inventory needs and costs. You assess risks like stockouts due to supplier delays "
                "and calculate necessary buffers (safety stock) considering these factors."""
            ),
            tools=supply_chain_analyst_tools
        )

    def create_risk_analyst(self) -> Agent:
        return self._create_agent(
            role='Inventory Risk Management Specialist',
            goal=(
                'Identify, assess, and propose mitigation strategies for inventory-related risks, including stockouts, obsolescence, spoilage, demand volatility, and supply disruptions. ' 
                'Quantify the potential financial impact of these risks.'
            ),
            backstory=(
                """You are a meticulous Risk Management Specialist focused on the complexities of inventory. You proactively identify potential issues "
                "that could lead to financial losses, such as holding too much slow-moving stock (obsolescence risk) or running out of popular items (stockout risk). "
                "You analyze demand variability and supply chain uncertainties to assess these risks and recommend actions, like adjusting safety stock or improving forecasting, "
                "to minimize their impact."""
            ),
            tools=risk_analyst_tools
        )

    def get_all_agents(self) -> List[Agent]:
        """Returns a list containing all specialized inventory agents."""
        logger.debug("Creating all inventory agents...")
        try:
            agents = [
                self.create_demand_analyst(),
                self.create_inventory_optimizer(),
                self.create_supply_chain_analyst(),
                self.create_risk_analyst()
            ]
            logger.info(f"Successfully created {len(agents)} agents.")
            return agents
        except Exception as e:
            logger.error(f"Error creating agents: {e}", exc_info=True)
            raise 