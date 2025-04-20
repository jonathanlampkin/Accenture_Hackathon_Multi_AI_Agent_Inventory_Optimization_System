"""
Configuration module for the inventory optimization system.

This module provides configuration management for the multi-agent inventory
optimization system, including settings for forecasting, optimization,
and agent parameters.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "forecasting": {
        "horizon": 30,  # Default forecast horizon in days
        "confidence_interval": 0.95,  # Default confidence interval for forecasts
        "models": {
            "sarima": {"enabled": True},
            "prophet": {"enabled": True},
            "exponential_smoothing": {"enabled": True},
            "random_forest": {"enabled": True},
            "lstm": {"enabled": False},  # Disabled by default as it requires more data
            "ensemble": {"enabled": True},
        }
    },
    "optimization": {
        "service_level": 0.95,  # Default service level for safety stock calculation
        "holding_cost_rate": 0.25,  # Annual holding cost as a fraction of item value
        "ordering_cost": 25.0,  # Fixed cost per order
        "min_max_factor": 1.5,  # Factor for min/max inventory levels
        "review_period": 7  # Review period in days
    },
    "anomaly_detection": {
        "sensitivity": 0.05,  # Lower is more sensitive
        "methods": {
            "isolation_forest": {"enabled": True},
            "z_score": {"enabled": True},
            "moving_average": {"enabled": True}
        }
    },
    "scenario_planning": {
        "scenarios": {
            "base": {
                "demand_factor": 1.0,
                "lead_time_factor": 1.0
            },
            "high_demand": {
                "demand_factor": 1.2,
                "lead_time_factor": 1.0
            },
            "supply_disruption": {
                "demand_factor": 1.0,
                "lead_time_factor": 1.5
            },
            "combined_risk": {
                "demand_factor": 1.2,
                "lead_time_factor": 1.5
            }
        }
    },
    "agents": {
        "model": "gpt-4",
        "temperature": 0.2,
        "verbose": True
    },
    "data": {
        "input_path": "./data/demand_data.csv",
        "output_dir": "./output"
    }
}


class Config:
    """Configuration manager for the inventory optimization system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found: {config_path}")
                return
            
            # Determine file type and load accordingly
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {config_path}")
                return
            
            # Update configuration with custom values
            self._update_nested_dict(self.config, custom_config)
            logger.info(f"Loaded configuration from {config_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Update nested dictionary recursively.
        
        Args:
            d: Base dictionary
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.
        
        Args:
            path: Dot notation path to configuration value
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        try:
            keys = path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any) -> None:
        """
        Set configuration value using dot notation path.
        
        Args:
            path: Dot notation path to configuration value
            value: Value to set
        """
        keys = path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def save(self, config_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Determine file type and save accordingly
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif config_path.endswith('.json'):
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.warning(f"Unsupported configuration file format: {config_path}")
                return
            
            logger.info(f"Saved configuration to {config_path}")
        
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


def generate_default_config(output_path: str) -> None:
    """
    Generate default configuration file.
    
    Args:
        output_path: Path to save default configuration
    """
    config = Config()
    config.save(output_path)
    logger.info(f"Generated default configuration at {output_path}")


# Sample usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inventory Optimization Configuration")
    parser.add_argument("--generate", help="Generate default configuration file", action="store_true")
    parser.add_argument("--output", help="Output path for generated configuration", default="./config/inventory_config.yaml")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_default_config(args.output)
    else:
        parser.print_help() 