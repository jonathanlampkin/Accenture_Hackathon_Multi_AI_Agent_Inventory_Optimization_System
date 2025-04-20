"""
Reinforcement Learning for Inventory Optimization

This module implements reinforcement learning approaches to optimize inventory policies
dynamically based on experience and changing conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import copy
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

logger = logging.getLogger(__name__)

class InventoryEnvironment(gym.Env):
    """
    Gym environment for inventory optimization.
    
    The environment simulates inventory management where an agent
    needs to decide on order quantities.
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                demand_model: Callable = None):
        """
        Initialize the inventory environment.
        
        Args:
            config: Configuration parameters including:
                - holding_cost: Cost per unit held per time step
                - stockout_cost: Cost per unit short per time step
                - order_cost: Fixed cost per order
                - max_inventory: Maximum inventory capacity
                - max_order: Maximum order quantity
                - lead_time: Lead time in time steps
                - demand_mean: Mean demand if using random demand
                - demand_std: Standard deviation of demand if using random
                - initial_inventory: Initial inventory level
                - episode_length: Number of time steps per episode
            demand_model: Optional function to generate demand
        """
        super(InventoryEnvironment, self).__init__()
        
        # Store configuration
        self.config = config
        
        # Extract parameters
        self.holding_cost = config.get('holding_cost', 1.0)
        self.stockout_cost = config.get('stockout_cost', 10.0)
        self.order_cost = config.get('order_cost', 50.0)
        self.variable_cost = config.get('variable_cost', 5.0)
        self.max_inventory = config.get('max_inventory', 1000)
        self.max_order = config.get('max_order', 100)
        self.lead_time = config.get('lead_time', 1)
        self.demand_mean = config.get('demand_mean', 10)
        self.demand_std = config.get('demand_std', 2)
        self.initial_inventory = config.get('initial_inventory', 100)
        self.episode_length = config.get('episode_length', 100)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=self.max_order, shape=(1,), dtype=np.float32
        )
        
        # Observation: [inventory_level, pending_orders (for each lead time step), demand_history]
        demand_history_length = config.get('demand_history_length', 5)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(2 + self.lead_time + demand_history_length,),
            dtype=np.float32
        )
        
        # Custom demand model
        self.demand_model = demand_model
        
        # State variables
        self.inventory = self.initial_inventory
        self.pending_orders = [0] * self.lead_time
        self.time_step = 0
        self.demand_history = deque(maxlen=demand_history_length)
        self.rewards = []
        self.cumulative_cost = 0
        self.stockouts = 0
        
        # Fill demand history with mean demand initially
        for _ in range(demand_history_length):
            self.demand_history.append(self.demand_mean)
        
        logger.info(f"Initialized InventoryEnvironment with max_inventory={self.max_inventory}, lead_time={self.lead_time}")
    
    def _generate_demand(self) -> float:
        """
        Generate demand for the current time step.
        
        Returns:
            Generated demand
        """
        if self.demand_model:
            return max(0, self.demand_model(self.time_step))
        else:
            # Default: truncated normal distribution
            demand = np.random.normal(self.demand_mean, self.demand_std)
            return max(0, demand)
    
    def _calculate_reward(self, inventory: float, demand: float, order_quantity: float) -> float:
        """
        Calculate reward (negative cost) for the current state.
        
        Args:
            inventory: Current inventory level
            demand: Current demand
            order_quantity: Order quantity
            
        Returns:
            Reward (negative cost)
        """
        # Holding cost
        holding_cost = self.holding_cost * max(0, inventory)
        
        # Stockout cost (penalty for not meeting demand)
        stockout = max(0, demand - inventory)
        stockout_cost = self.stockout_cost * stockout
        
        # Order cost (fixed + variable)
        order_cost = self.order_cost * (1 if order_quantity > 0 else 0) + self.variable_cost * order_quantity
        
        # Total cost
        total_cost = holding_cost + stockout_cost + order_cost
        
        # Return negative cost as reward
        return -total_cost
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation state.
        
        Returns:
            Observation array
        """
        # Normalize inventory level
        norm_inventory = self.inventory / self.max_inventory
        
        # Normalize time step
        norm_time = self.time_step / self.episode_length
        
        # Normalize pending orders
        norm_pending = [order / self.max_order for order in self.pending_orders]
        
        # Normalize demand history
        max_demand = max(max(self.demand_history), self.demand_mean * 3)
        norm_demand_history = [d / max_demand for d in self.demand_history]
        
        # Combine all features
        observation = [norm_inventory, norm_time] + norm_pending + list(norm_demand_history)
        
        return np.array(observation, dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.inventory = self.initial_inventory
        self.pending_orders = [0] * self.lead_time
        self.time_step = 0
        self.demand_history = deque(maxlen=self.demand_history.maxlen)
        self.rewards = []
        self.cumulative_cost = 0
        self.stockouts = 0
        
        # Fill demand history with mean demand initially
        for _ in range(self.demand_history.maxlen):
            self.demand_history.append(self.demand_mean)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (order quantity)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Extract order quantity from action
        order_quantity = float(action[0])
        order_quantity = min(max(0, order_quantity), self.max_order)  # Clip to valid range
        
        # Receive orders that arrived (first pending order)
        received_quantity = self.pending_orders[0]
        self.inventory += received_quantity
        
        # Update pending orders (shift left and add new order at the end)
        self.pending_orders = self.pending_orders[1:] + [order_quantity]
        
        # Generate demand
        demand = self._generate_demand()
        self.demand_history.append(demand)
        
        # Fulfill demand
        fulfilled = min(demand, self.inventory)
        self.inventory -= fulfilled
        stockout = demand - fulfilled
        if stockout > 0:
            self.stockouts += 1
        
        # Calculate reward
        reward = self._calculate_reward(self.inventory, demand, order_quantity)
        self.rewards.append(reward)
        self.cumulative_cost -= reward  # Store cumulative cost (negative reward)
        
        # Increment time step
        self.time_step += 1
        
        # Check if episode is done
        done = self.time_step >= self.episode_length
        
        # Get new observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'inventory': self.inventory,
            'demand': demand,
            'order_quantity': order_quantity,
            'stockout': stockout > 0,
            'stockout_quantity': stockout,
            'cost': -reward,
            'cumulative_cost': self.cumulative_cost,
            'service_level': 1 - (self.stockouts / self.time_step) if self.time_step > 0 else 1.0
        }
        
        return observation, reward, done, info

class DQNAgent:
    """
    Deep Q-Network agent for inventory optimization.
    """
    
    def __init__(self, 
                state_size: int,
                action_size: int,
                config: Dict[str, Any] = None):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state/observation space
            action_size: Size of the action space (discretized)
            config: Agent configuration parameters
        """
        # Configuration
        self.config = config or {}
        
        # State and action parameters
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN hyperparameters
        self.gamma = self.config.get('gamma', 0.95)  # Discount factor
        self.epsilon = self.config.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.train_start = self.config.get('train_start', 1000)
        self.update_target_freq = self.config.get('update_target_freq', 100)
        
        # Experience replay memory
        self.memory = deque(maxlen=self.config.get('memory_size', 2000))
        
        # Models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Metrics
        self.train_loss_history = []
        self.step_count = 0
        
        logger.info(f"Initialized DQNAgent with state_size={state_size}, action_size={action_size}")
    
    def _build_model(self) -> Model:
        """
        Build a neural network model for DQN.
        
        Returns:
            Keras model
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self) -> None:
        """Update target model by copying weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Updated target model weights")
    
    def memorize(self, 
                state: np.ndarray,
                action: int,
                reward: float,
                next_state: np.ndarray,
                done: bool) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose an action based on current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action index
        """
        # Exploration
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation (use model)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> float:
        """
        Train the model using experience replay.
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.train_start:
            return 0.0
        
        # Sample minibatch from memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        states = np.zeros((len(minibatch), self.state_size))
        next_states = np.zeros((len(minibatch), self.state_size))
        
        # Extract states and next_states
        for i, (state, _, _, next_state, _) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
        
        # Predict Q-values
        targets = self.model.predict(states, verbose=0)
        next_targets = self.target_model.predict(next_states, verbose=0)
        
        # Update targets with Bellman equation
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(next_targets[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.train_loss_history.append(loss)
        
        # Update target model periodically
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def load(self, name: str) -> None:
        """
        Load model weights from file.
        
        Args:
            name: Filename
        """
        self.model.load_weights(name)
        self.update_target_model()
        logger.info(f"Loaded model weights from {name}")
    
    def save(self, name: str) -> None:
        """
        Save model weights to file.
        
        Args:
            name: Filename
        """
        self.model.save_weights(name)
        logger.info(f"Saved model weights to {name}")

class InventoryRLOptimizer:
    """
    Reinforcement Learning based optimizer for inventory management policies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RL optimizer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Environment parameters
        env_config = self.config.get('environment', {})
        self.env = InventoryEnvironment(env_config)
        
        # Agent parameters
        agent_config = self.config.get('agent', {})
        
        # Discretize action space
        self.discrete_actions = self.config.get('discrete_actions', 10)
        action_values = np.linspace(0, self.env.max_order, self.discrete_actions)
        self.action_map = {i: action_values[i] for i in range(self.discrete_actions)}
        
        # Create agent
        self.agent = DQNAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.discrete_actions,
            config=agent_config
        )
        
        # Training parameters
        self.num_episodes = self.config.get('num_episodes', 1000)
        self.eval_interval = self.config.get('eval_interval', 10)
        self.max_steps = self.config.get('max_steps', 100)
        
        # Metrics
        self.training_rewards = []
        self.eval_rewards = []
        self.training_costs = []
        self.training_service_levels = []
        
        logger.info(f"Initialized InventoryRLOptimizer with {self.discrete_actions} discrete actions")
    
    def train(self, 
             num_episodes: Optional[int] = None, 
             verbose: bool = False) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            num_episodes: Number of training episodes (override config)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training metrics
        """
        episodes = num_episodes or self.num_episodes
        
        # Reset metrics
        self.training_rewards = []
        self.eval_rewards = []
        self.training_costs = []
        self.training_service_levels = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            info_log = []
            
            for step in range(self.max_steps):
                # Select action
                action_idx = self.agent.act(state)
                action_value = self.action_map[action_idx]
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(np.array([action_value]))
                
                # Store experience in memory
                self.agent.memorize(state, action_idx, reward, next_state, done)
                
                # Train agent
                loss = self.agent.replay()
                
                # Update state and metrics
                state = next_state
                total_reward += reward
                info_log.append(info)
                
                if done:
                    break
            
            # Calculate episode metrics
            episode_reward = total_reward
            episode_cost = sum(info['cost'] for info in info_log)
            episode_service_level = 1 - sum(1 for info in info_log if info['stockout']) / len(info_log)
            
            # Store metrics
            self.training_rewards.append(episode_reward)
            self.training_costs.append(episode_cost)
            self.training_service_levels.append(episode_service_level)
            
            # Evaluate periodically
            if episode % self.eval_interval == 0:
                eval_reward = self.evaluate(1, False)
                self.eval_rewards.append(eval_reward)
                
                if verbose:
                    logger.info(f"Episode {episode}/{episodes} - Reward: {episode_reward:.1f}, "
                              f"Cost: {episode_cost:.1f}, Service Level: {episode_service_level:.2f}, "
                              f"Eval Reward: {eval_reward:.1f}, Epsilon: {self.agent.epsilon:.3f}")
        
        # Final evaluation
        final_eval_reward = self.evaluate(5, False)
        
        logger.info(f"Training completed - Final eval reward: {final_eval_reward:.1f}")
        
        # Return training metrics
        return {
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards,
            'training_costs': self.training_costs,
            'training_service_levels': self.training_service_levels,
            'final_eval_reward': final_eval_reward
        }
    
    def evaluate(self, 
                num_episodes: int = 1, 
                render: bool = False) -> float:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Average reward
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            info_log = []
            
            for step in range(self.max_steps):
                # Select best action (no exploration)
                q_values = self.agent.model.predict(state.reshape(1, -1), verbose=0)
                action_idx = np.argmax(q_values[0])
                action_value = self.action_map[action_idx]
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(np.array([action_value]))
                
                # Update state and metrics
                state = next_state
                total_reward += reward
                info_log.append(info)
                
                if done:
                    break
            
            total_rewards.append(total_reward)
        
        # Return average reward
        return sum(total_rewards) / len(total_rewards)
    
    def get_optimal_policy(self) -> Callable:
        """
        Get the learned optimal policy function.
        
        Returns:
            Policy function that takes a state and returns an order quantity
        """
        def policy_function(state):
            # Ensure state is in the right format
            state_array = np.array(state, dtype=np.float32)
            
            # Get Q-values from the model
            q_values = self.agent.model.predict(state_array.reshape(1, -1), verbose=0)
            
            # Select best action
            action_idx = np.argmax(q_values[0])
            action_value = self.action_map[action_idx]
            
            return action_value
        
        return policy_function
    
    def get_inventory_policy(self) -> Dict[str, Any]:
        """
        Extract inventory policy parameters from the learned policy.
        
        Returns:
            Dictionary with policy parameters
        """
        # Sample states with different inventory levels
        inventory_levels = np.linspace(0, self.env.max_inventory, 20)
        pending_orders = [0] * self.env.lead_time
        demand_history = [self.env.demand_mean] * self.env.demand_history.maxlen
        
        # Generate policy map
        policy_map = {}
        for inv_level in inventory_levels:
            # Create state with this inventory level
            state = [inv_level / self.env.max_inventory, 0.5]  # Normalize inventory, mid-time step
            state += [order / self.env.max_order for order in pending_orders]  # Normalize pending orders
            state += [d / (self.env.demand_mean * 3) for d in demand_history]  # Normalize demand history
            
            state_array = np.array(state, dtype=np.float32)
            
            # Get action from model
            q_values = self.agent.model.predict(state_array.reshape(1, -1), verbose=0)
            action_idx = np.argmax(q_values[0])
            order_quantity = self.action_map[action_idx]
            
            policy_map[float(inv_level)] = float(order_quantity)
        
        # Extract reorder point and target level
        ordered_levels = sorted(policy_map.items())
        reorder_point = None
        
        for inv, order in ordered_levels:
            if order > 0 and reorder_point is None:
                reorder_point = inv
        
        if reorder_point is None:
            reorder_point = 0
        
        # Calculate order-up-to level (S)
        # This is an approximation based on the observed policy
        target_level = reorder_point
        for inv, order in ordered_levels:
            if inv <= reorder_point:
                target_level = max(target_level, inv + order)
        
        # Calculate economic order quantity (EOQ)
        # For a (s,S) policy, EOQ ≈ S - s
        eoq = target_level - reorder_point
        
        # Return policy parameters
        return {
            'reorder_point': reorder_point,
            'target_level': target_level,
            'eoq': eoq,
            'policy_map': policy_map,
            'model_based': True,
            'policy_type': '(s,S) policy'
        } 