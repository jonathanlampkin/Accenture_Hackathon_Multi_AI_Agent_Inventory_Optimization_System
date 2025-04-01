"""
Base Agent Class for the Multi-Agent Inventory Optimization System
"""

import os
import json
import pandas as pd
import logging
from abc import ABC, abstractmethod
import sys
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'agents.log')),
        logging.StreamHandler()
    ]
)

class BaseAgent(ABC):
    """
    Base Agent class that all specialized agents will inherit from
    """
    
    def __init__(self, name, use_gpu=False):
        """
        Initialize the base agent
        
        Args:
            name: Name of the agent
            use_gpu: Whether to use GPU acceleration if available
        """
        self.name = name
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(f'Agent.{name}')
        self.messages = []
        self.state = {}
        self.recommendations = []
        
        # Create agent output directory if it doesn't exist
        self.output_dir = os.path.join(config.OUTPUT_DIR, self.name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"{self.name} agent initialized")
        if self.use_gpu:
            self.logger.info("GPU acceleration enabled")
            try:
                # Only attempt to import and set up GPU if use_gpu is True
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.device.type == "cuda":
                    self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.logger.warning("GPU requested but not available, falling back to CPU")
            except ImportError:
                self.logger.warning("PyTorch not installed, GPU acceleration disabled")
                self.device = None
        else:
            self.device = None
    
    def log(self, message, level='info'):
        """
        Log a message with the appropriate level
        
        Args:
            message: Message to log
            level: Logging level (info, warning, error, debug)
        """
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'debug':
            self.logger.debug(message)
        
        # Store message in agent's message history
        self.messages.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'level': level,
            'message': message
        })
    
    def save_state(self):
        """
        Save the agent's current state to a JSON file
        """
        state_file = os.path.join(self.output_dir, f"{self.name}_state.json")
        with open(state_file, 'w') as f:
            json.dump({
                'name': self.name,
                'state': self.state,
                'recommendations': self.recommendations,
                'messages': self.messages[-50:]  # Save only the last 50 messages
            }, f, indent=2, default=self._json_serializable)
        self.log(f"State saved to {state_file}")
    
    def _json_serializable(self, obj):
        """
        Convert objects to JSON serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable representation of the object
        """
        if isinstance(obj, pd.DataFrame):
            return {
                'type': 'DataFrame',
                'data': obj.to_dict(orient='records'),
                'index': obj.index.tolist(),
                'columns': obj.columns.tolist()
            }
        elif isinstance(obj, pd.Series):
            return {
                'type': 'Series',
                'data': obj.to_dict(),
                'index': obj.index.tolist(),
                'name': obj.name
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (pd.Timedelta, timedelta)):
            return str(obj)
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            # Default behavior for other types
            return str(obj)
    
    def load_state(self):
        """
        Load the agent's state from a JSON file if it exists
        
        Returns:
            bool: True if state was loaded, False otherwise
        """
        state_file = os.path.join(self.output_dir, f"{self.name}_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                data = json.load(f)
                
                # Get the state and restore any DataFrames
                loaded_state = data.get('state', {})
                self._restore_complex_objects(loaded_state)
                self.state = loaded_state
                
                self.recommendations = data.get('recommendations', [])
                loaded_messages = data.get('messages', [])
                
                # Only append messages that aren't already in the history
                existing_timestamps = {m['timestamp'] for m in self.messages}
                for msg in loaded_messages:
                    if msg['timestamp'] not in existing_timestamps:
                        self.messages.append(msg)
                
                self.log(f"State loaded from {state_file}")
                return True
        return False
    
    def _restore_complex_objects(self, obj):
        """
        Recursively restore complex objects like DataFrames from their serialized form
        
        Args:
            obj: Object to restore (in-place)
        """
        if isinstance(obj, dict):
            # Check if this is a serialized DataFrame
            if 'type' in obj and obj['type'] == 'DataFrame' and 'data' in obj and 'columns' in obj:
                # Convert back to DataFrame and return
                return pd.DataFrame(obj['data'], columns=obj['columns'])
            
            # Check if this is a serialized Series
            if 'type' in obj and obj['type'] == 'Series' and 'data' in obj and 'index' in obj:
                # Convert back to Series and return
                return pd.Series(obj['data'], index=obj['index'], name=obj.get('name'))
            
            # Process nested dictionaries
            for key, value in list(obj.items()):
                if isinstance(value, dict):
                    restored = self._restore_complex_objects(value)
                    if restored is not value:  # If a new object was created
                        obj[key] = restored
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            restored = self._restore_complex_objects(item)
                            if restored is not item:  # If a new object was created
                                value[i] = restored
        
        return obj
    
    def add_recommendation(self, recommendation, confidence=None, context=None):
        """
        Add a recommendation to the agent's list of recommendations
        
        Args:
            recommendation: The recommendation text or object
            confidence: Confidence level in the recommendation (0.0 to 1.0)
            context: Additional context or explanation for the recommendation
        """
        rec_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'recommendation': recommendation,
            'confidence': confidence,
            'context': context
        }
        self.recommendations.append(rec_entry)
        self.log(f"Added recommendation: {recommendation} (confidence: {confidence})")
        
        # Save recommendations to a separate file for easier access
        self._save_recommendations()
    
    def _save_recommendations(self):
        """
        Save all recommendations to a JSON file
        """
        rec_file = os.path.join(self.output_dir, f"{self.name}_recommendations.json")
        with open(rec_file, 'w') as f:
            json.dump(self.recommendations, f, indent=2, default=self._json_serializable)
    
    def get_latest_recommendations(self, n=5):
        """
        Get the n most recent recommendations
        
        Args:
            n: Number of recommendations to return
            
        Returns:
            list: The n most recent recommendations
        """
        return self.recommendations[-n:]
    
    def receive_message(self, sender, message, message_type='info'):
        """
        Process a message from another agent
        
        Args:
            sender: The name of the agent sending the message
            message: The content of the message
            message_type: The type of message (info, request, response)
            
        Returns:
            Any: Optional response to the message
        """
        self.log(f"Received message from {sender}: {message}", level='info')
        
        # Add to message history
        self.messages.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'sender': sender,
            'message': message,
            'type': message_type
        })
        
        # Default implementation simply logs the message
        # Subclasses should override this method to handle messages
        return None
    
    def send_message(self, recipient, message, message_type='info'):
        """
        Send a message to another agent
        
        Args:
            recipient: The agent object to send the message to
            message: The content of the message
            message_type: The type of message (info, request, response)
            
        Returns:
            Any: The response from the recipient agent
        """
        self.log(f"Sending message to {recipient.name}: {message}", level='info')
        
        # Add to message history
        self.messages.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'recipient': recipient.name,
            'message': message,
            'type': message_type
        })
        
        # Call the recipient's receive_message method
        return recipient.receive_message(self.name, message, message_type)
    
    @abstractmethod
    def analyze(self, data=None):
        """
        Analyze data and update the agent's state
        
        Args:
            data: Data to analyze (optional)
            
        Returns:
            dict: Analysis results
        """
        pass
    
    @abstractmethod
    def make_recommendation(self):
        """
        Generate a recommendation based on the agent's current state
        
        Returns:
            dict: The recommendation
        """
        pass
    
    @abstractmethod
    def update(self, feedback=None):
        """
        Update the agent's state based on feedback
        
        Args:
            feedback: Feedback to incorporate
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        pass 