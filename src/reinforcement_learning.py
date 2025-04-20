"""
Reinforcement Learning module for inventory optimization.

This module provides a reinforcement learning approach to inventory optimization,
allowing the system to learn optimal inventory policies through experience.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
import pickle
import json

logger = logging.getLogger(__name__)

class InventoryEnvironment:
    """
    A reinforcement learning environment for inventory optimization.
    
    This environment simulates inventory management decisions and their consequences,
    allowing agents to learn optimal inventory policies through trial and error.
    """
    
    def __init__(
        self,
        product_data: Dict[str, Any],
        demand_history: List[Dict[str, float]],
        lead_time: int,
        holding_cost_rate: float = 0.02,
        stockout_cost_rate: float = 0.1,
        order_cost: float = 10.0,
        max_inventory: int = 1000,
        max_order: int = 500
    ):
        """
        Initialize the inventory environment.
        
        Args:
            product_data: Product information
            demand_history: Historical demand data
            lead_time: Lead time for orders in periods
            holding_cost_rate: Holding cost as a fraction of product value per period
            stockout_cost_rate: Stockout cost as a fraction of product margin per unit
            order_cost: Fixed cost per order
            max_inventory: Maximum inventory level
            max_order: Maximum order quantity
        """
        self.product_id = product_data.get('product_id', 'unknown')
        self.product_price = product_data.get('price', 10.0)
        self.product_cost = product_data.get('cost', 5.0)
        self.is_perishable = product_data.get('is_perishable', False)
        
        # Convert demand history to numpy array
        self.demand_history = np.array([d.get('demand', 0) for d in demand_history])
        
        # Environment parameters
        self.lead_time = lead_time
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_rate = stockout_cost_rate
        self.order_cost = order_cost
        self.max_inventory = max_inventory
        self.max_order = max_order
        
        # State variables
        self.current_inventory = 0
        self.pending_orders = [0] * lead_time
        self.current_step = 0
        self.done = False
        
        # Set discretization for state and action spaces
        self.inventory_levels = 20  # Number of discrete inventory levels
        self.order_levels = 10      # Number of discrete order quantities
        
        # Episode history
        self.history = []
    
    def reset(self, initial_inventory: int = 100) -> np.ndarray:
        """
        Reset the environment to an initial state.
        
        Args:
            initial_inventory: Initial inventory level
            
        Returns:
            Initial state observation
        """
        self.current_inventory = min(initial_inventory, self.max_inventory)
        self.pending_orders = [0] * self.lead_time
        self.current_step = 0
        self.done = False
        self.history = []
        
        # Return initial state
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action in the environment and return the result.
        
        Args:
            action: Action to take (order quantity index)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert discrete action to order quantity
        order_quantity = self._action_to_quantity(action)
        
        # Apply the order
        order_placed = self._place_order(order_quantity)
        
        # Process demand for the current period
        period_demand = self._get_period_demand()
        sales, stockout = self._satisfy_demand(period_demand)
        
        # Calculate costs and revenue
        holding_cost = self.holding_cost_rate * self.product_cost * self.current_inventory
        stockout_cost = stockout * (self.product_price - self.product_cost) * self.stockout_cost_rate
        order_cost = self.order_cost if order_placed else 0
        revenue = sales * self.product_price
        cogs = sales * self.product_cost
        
        # Calculate reward (profit)
        reward = revenue - cogs - holding_cost - stockout_cost - order_cost
        
        # Update state and check if episode is done
        self.current_step += 1
        self.done = self.current_step >= len(self.demand_history)
        
        # Record this step in history
        step_info = {
            'step': self.current_step,
            'inventory': self.current_inventory,
            'order': order_quantity,
            'demand': period_demand,
            'sales': sales,
            'stockout': stockout,
            'reward': reward,
            'holding_cost': holding_cost,
            'stockout_cost': stockout_cost,
            'order_cost': order_cost,
            'revenue': revenue,
            'pending_orders': self.pending_orders.copy()
        }
        self.history.append(step_info)
        
        # Return next state, reward, done, info
        return self._get_state(), reward, self.done, step_info
    
    def _action_to_quantity(self, action: int) -> int:
        """Convert discrete action index to order quantity."""
        # Map action index to a fraction of max_order
        if action == 0:
            return 0  # Special case for not ordering
        
        fraction = action / self.order_levels
        return int(fraction * self.max_order)
    
    def _place_order(self, quantity: int) -> bool:
        """Place an order and update pending orders."""
        if quantity <= 0:
            return False
        
        # Add order to pending queue
        if self.lead_time > 0:
            self.pending_orders[-1] = quantity  # Add to end of queue
        else:
            # If lead time is 0, immediately receive the order
            self.current_inventory = min(self.current_inventory + quantity, self.max_inventory)
        
        return quantity > 0
    
    def _get_period_demand(self) -> int:
        """Get demand for the current period."""
        if self.current_step < len(self.demand_history):
            return int(self.demand_history[self.current_step])
        return 0  # Default if we're out of historical data
    
    def _satisfy_demand(self, demand: int) -> Tuple[int, int]:
        """
        Satisfy demand from current inventory.
        
        Args:
            demand: Current period demand
            
        Returns:
            Tuple of (sales, stockout)
        """
        # Process incoming orders
        if self.lead_time > 0:
            received_order = self.pending_orders[0]
            self.current_inventory = min(self.current_inventory + received_order, self.max_inventory)
            
            # Shift pending orders
            self.pending_orders = self.pending_orders[1:] + [0]
        
        # Calculate sales and stockout
        sales = min(demand, self.current_inventory)
        stockout = demand - sales
        
        # Update inventory
        self.current_inventory -= sales
        
        return sales, stockout
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            State vector including current inventory and pending orders
        """
        # Normalize inventory to [0, 1]
        norm_inventory = self.current_inventory / self.max_inventory
        
        # Normalize pending orders to [0, 1]
        norm_pending = [min(1.0, p / self.max_order) for p in self.pending_orders]
        
        # Combine inventory and pending orders
        state = [norm_inventory] + norm_pending
        
        return np.array(state, dtype=np.float32)
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Convert episode history to a DataFrame.
        
        Returns:
            DataFrame of episode history
        """
        return pd.DataFrame(self.history)


class QLearningAgent:
    """
    Q-Learning agent for inventory optimization.
    
    This agent learns a Q-table mapping states and actions to expected rewards,
    allowing it to determine optimal inventory policies.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01
    ):
        """
        Initialize the Q-Learning agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for updating Q-values
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate
            exploration_decay: Rate at which exploration decays
            min_exploration_rate: Minimum exploration rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        self.q_bins = {
            'inventory': 20,   # Number of bins for inventory level
            'pending': 5       # Number of bins for each pending order
        }
        
        # Create the Q-table
        self.q_table = {}
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Get the best action for the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Action index
        """
        # Discretize state
        state_key = self._discretize_state(state)
        
        # Exploration: random action
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_dim)
        
        # Exploitation: best known action
        if state_key not in self.q_table:
            # If state not seen before, initialize with zeros
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state_key])
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Update Q-values based on the observed transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Discretize states
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        
        # Get current Q value
        current_q = self.q_table[state_key][action]
        
        # Calculate max Q for next state
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        
        # Calculate target Q value
        target_q = reward + (self.discount_factor * max_next_q)
        
        # Update Q value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state into bins for the Q-table.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple representing discretized state
        """
        # Discretize inventory level
        inv_bin = min(int(state[0] * self.q_bins['inventory']), self.q_bins['inventory'] - 1)
        
        # Discretize pending orders
        pending_bins = []
        for i in range(1, len(state)):
            bin_idx = min(int(state[i] * self.q_bins['pending']), self.q_bins['pending'] - 1)
            pending_bins.append(bin_idx)
        
        # Return tuple for use as dictionary key
        return tuple([inv_bin] + pending_bins)
    
    def save(self, filepath: str) -> None:
        """
        Save the Q-table and agent parameters.
        
        Args:
            filepath: Path to save the agent
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Q-table and parameters
        agent_data = {
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'params': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay': self.exploration_decay,
                'min_exploration_rate': self.min_exploration_rate,
                'q_bins': self.q_bins
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'QLearningAgent':
        """
        Load a saved agent.
        
        Args:
            filepath: Path to the saved agent
            
        Returns:
            Loaded QLearningAgent instance
        """
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Get parameters
        params = agent_data['params']
        
        # Create new agent
        agent = cls(
            state_dim=params['state_dim'],
            action_dim=params['action_dim'],
            learning_rate=params['learning_rate'],
            discount_factor=params['discount_factor'],
            exploration_rate=params['exploration_rate'],
            exploration_decay=params['exploration_decay'],
            min_exploration_rate=params['min_exploration_rate']
        )
        
        # Set q_bins
        agent.q_bins = params['q_bins']
        
        # Convert Q-table back to correct format
        agent.q_table = {}
        for state_key_str, q_values in agent_data['q_table'].items():
            # Convert string representation of tuple back to actual tuple
            state_key = tuple(map(int, state_key_str.strip('()').split(', ')))
            agent.q_table[state_key] = np.array(q_values)
        
        return agent


class RLInventoryOptimizer:
    """
    Reinforcement Learning-based inventory optimizer.
    
    This class trains and applies RL models for inventory optimization.
    """
    
    def __init__(
        self,
        output_dir: str = './models/rl',
        num_episodes: int = 1000,
        lead_time: int = 3,
        holding_cost_rate: float = 0.02,
        stockout_cost_rate: float = 0.1
    ):
        """
        Initialize the RL inventory optimizer.
        
        Args:
            output_dir: Directory to save models and results
            num_episodes: Number of episodes for training
            lead_time: Default lead time for inventory replenishment
            holding_cost_rate: Cost rate for holding inventory
            stockout_cost_rate: Cost rate for stockouts
        """
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.lead_time = lead_time
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_rate = stockout_cost_rate
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store models
        self.models = {}
    
    def train_product_model(
        self,
        product_data: Dict[str, Any],
        demand_history: List[Dict[str, float]],
        save_model: bool = True
    ) -> Tuple[QLearningAgent, Dict[str, Any]]:
        """
        Train an RL model for a specific product.
        
        Args:
            product_data: Product information
            demand_history: Historical demand data
            save_model: Whether to save the trained model
            
        Returns:
            Tuple of (trained_agent, training_results)
        """
        product_id = product_data.get('product_id', 'unknown')
        logger.info(f"Training RL model for product {product_id}")
        
        # Get lead time for this product
        lead_time = product_data.get('lead_time', self.lead_time)
        
        # Create environment
        env = InventoryEnvironment(
            product_data=product_data,
            demand_history=demand_history,
            lead_time=lead_time,
            holding_cost_rate=self.holding_cost_rate,
            stockout_cost_rate=self.stockout_cost_rate
        )
        
        # Create agent
        state = env.reset()
        state_dim = len(state)
        action_dim = env.order_levels + 1  # +1 for the "no order" action
        
        agent = QLearningAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.995
        )
        
        # Training loop
        episode_rewards = []
        best_reward = float('-inf')
        
        for episode in range(self.num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Get action
                action = agent.get_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Update agent
                agent.update(state, action, reward, next_state, done)
                
                # Update state and total reward
                state = next_state
                total_reward += reward
            
            # Record this episode's reward
            episode_rewards.append(total_reward)
            
            # Log progress periodically
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode+1}/{self.num_episodes}, Average Reward: {avg_reward:.2f}, Exploration Rate: {agent.exploration_rate:.4f}")
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    if save_model:
                        model_path = os.path.join(self.output_dir, f"model_{product_id}.pkl")
                        agent.save(model_path)
        
        # Save final model if it's the best
        final_avg_reward = np.mean(episode_rewards[-100:])
        if final_avg_reward > best_reward and save_model:
            model_path = os.path.join(self.output_dir, f"model_{product_id}.pkl")
            agent.save(model_path)
        
        # Store model in memory
        self.models[product_id] = agent
        
        # Calculate training metrics
        window_size = 100
        smoothed_rewards = []
        for i in range(len(episode_rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(episode_rewards[i:i+window_size]))
        
        training_results = {
            'product_id': product_id,
            'total_episodes': self.num_episodes,
            'final_exploration_rate': agent.exploration_rate,
            'best_avg_reward': best_reward,
            'final_avg_reward': final_avg_reward,
            'rewards': episode_rewards,
            'smoothed_rewards': smoothed_rewards
        }
        
        return agent, training_results
    
    def get_optimal_policy(
        self,
        product_id: str,
        current_inventory: float,
        pending_orders: List[float]
    ) -> Dict[str, Any]:
        """
        Get the optimal inventory policy for the current state.
        
        Args:
            product_id: Product ID
            current_inventory: Current inventory level
            pending_orders: List of pending orders
            
        Returns:
            Dictionary with optimal action and policy details
        """
        # Check if model exists
        if product_id not in self.models:
            model_path = os.path.join(self.output_dir, f"model_{product_id}.pkl")
            if os.path.exists(model_path):
                try:
                    self.models[product_id] = QLearningAgent.load(model_path)
                except Exception as e:
                    logger.error(f"Error loading model for product {product_id}: {str(e)}")
                    return {
                        'error': f"Model for product {product_id} could not be loaded",
                        'order_quantity': 0
                    }
            else:
                return {
                    'error': f"No trained model found for product {product_id}",
                    'order_quantity': 0
                }
        
        # Get model
        agent = self.models[product_id]
        
        # Normalize state inputs
        max_inventory = 1000  # This should match the environment's max_inventory
        max_order = 500       # This should match the environment's max_order
        
        norm_inventory = min(1.0, current_inventory / max_inventory)
        norm_pending = [min(1.0, p / max_order) for p in pending_orders]
        
        # Fill or truncate pending orders list to match expected length
        expected_pending_len = agent.state_dim - 1
        if len(norm_pending) < expected_pending_len:
            norm_pending = norm_pending + [0] * (expected_pending_len - len(norm_pending))
        elif len(norm_pending) > expected_pending_len:
            norm_pending = norm_pending[:expected_pending_len]
        
        # Create state vector
        state = np.array([norm_inventory] + norm_pending, dtype=np.float32)
        
        # Get optimal action (in training=False mode to disable exploration)
        action = agent.get_action(state, training=False)
        
        # Convert action to order quantity
        order_quantity = self._action_to_quantity(action, max_order, agent.action_dim)
        
        # Get Q-values for all actions
        state_key = agent._discretize_state(state)
        q_values = agent.q_table.get(state_key, np.zeros(agent.action_dim))
        
        # Return policy information
        return {
            'product_id': product_id,
            'current_inventory': current_inventory,
            'pending_orders': pending_orders,
            'recommended_action': int(action),
            'order_quantity': int(order_quantity),
            'expected_value': float(q_values[action]) if action < len(q_values) else 0,
            'confidence': float(self._calculate_confidence(q_values))
        }
    
    def _action_to_quantity(self, action: int, max_order: int, action_dim: int) -> int:
        """Convert discrete action index to order quantity."""
        # Map action index to a fraction of max_order
        if action == 0:
            return 0  # Special case for not ordering
        
        fraction = action / (action_dim - 1)
        return int(fraction * max_order)
    
    def _calculate_confidence(self, q_values: np.ndarray) -> float:
        """
        Calculate confidence in the recommendation.
        
        Args:
            q_values: Q-values for all actions
            
        Returns:
            Confidence score (0-1)
        """
        if len(q_values) <= 1:
            return 0.0
            
        # Calculate how much better the best action is compared to alternatives
        best_q = np.max(q_values)
        second_best_q = np.partition(q_values, -2)[-2] if len(q_values) > 1 else 0
        
        # Calculate mean and standard deviation of Q-values
        mean_q = np.mean(q_values)
        std_q = np.std(q_values)
        
        # If all Q-values are the same, we have no confidence
        if std_q == 0:
            return 0.0
        
        # Calculate Z-score of best action
        z_score = (best_q - mean_q) / std_q if std_q > 0 else 0
        
        # Calculate advantage over second-best action
        advantage = (best_q - second_best_q) / (np.abs(mean_q) + 1e-6)
        
        # Combine metrics into overall confidence
        confidence = 0.5 * min(1.0, z_score / 3.0) + 0.5 * min(1.0, advantage)
        
        return confidence


def train_models(
    product_data_list: List[Dict[str, Any]],
    demand_history: Dict[str, List[Dict[str, float]]],
    output_dir: str = './models/rl',
    num_episodes: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Train RL models for multiple products.
    
    Args:
        product_data_list: List of product data dictionaries
        demand_history: Dictionary mapping product_id to demand history
        output_dir: Directory to save models and results
        num_episodes: Number of episodes for training
        
    Returns:
        Dictionary mapping product_id to training results
    """
    # Create optimizer
    optimizer = RLInventoryOptimizer(
        output_dir=output_dir,
        num_episodes=num_episodes
    )
    
    # Train models for each product
    training_results = {}
    
    for product_data in product_data_list:
        product_id = product_data.get('product_id', 'unknown')
        
        # Check if we have demand history for this product
        if product_id not in demand_history:
            logger.warning(f"No demand history found for product {product_id}, skipping training")
            continue
        
        # Train model
        _, results = optimizer.train_product_model(
            product_data=product_data,
            demand_history=demand_history[product_id],
            save_model=True
        )
        
        # Store results
        training_results[product_id] = results
    
    # Save training summary
    summary_path = os.path.join(output_dir, 'training_summary.json')
    
    with open(summary_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for product_id, results in training_results.items():
            serializable_results[product_id] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }
        
        json.dump(serializable_results, f, indent=2)
    
    return training_results


def get_optimal_policies(
    product_data_list: List[Dict[str, Any]],
    inventory_data: Dict[str, Dict[str, Any]],
    model_dir: str = './models/rl'
) -> Dict[str, Dict[str, Any]]:
    """
    Get optimal inventory policies for multiple products.
    
    Args:
        product_data_list: List of product data dictionaries
        inventory_data: Dictionary mapping product_id to current inventory data
        model_dir: Directory containing trained models
        
    Returns:
        Dictionary mapping product_id to optimal policy
    """
    # Create optimizer
    optimizer = RLInventoryOptimizer(output_dir=model_dir)
    
    # Get policies for each product
    policies = {}
    
    for product_data in product_data_list:
        product_id = product_data.get('product_id', 'unknown')
        
        # Check if we have inventory data for this product
        if product_id not in inventory_data:
            logger.warning(f"No inventory data found for product {product_id}, skipping")
            continue
        
        # Get current inventory and pending orders
        inventory_info = inventory_data[product_id]
        current_inventory = inventory_info.get('current_inventory', 0)
        pending_orders = inventory_info.get('pending_orders', [])
        
        # Get optimal policy
        policy = optimizer.get_optimal_policy(
            product_id=product_id,
            current_inventory=current_inventory,
            pending_orders=pending_orders
        )
        
        # Store policy
        policies[product_id] = policy
    
    return policies 