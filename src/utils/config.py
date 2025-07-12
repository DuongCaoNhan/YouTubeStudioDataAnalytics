"""
Configuration Management
Handles configuration loading and management for the analytics system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration management class."""
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_data: Dictionary with configuration data
        """
        self.data = config_data or self.get_default_config()
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'default_videos_file': 'data/sample/videos.csv',
                'default_subscribers_file': 'data/sample/subscribers.csv',
                'export_directory': 'data/exports',
                'charts_directory': 'data/exports/charts'
            },
            'ml': {
                'default_model_type': 'linear',
                'feature_columns': [
                    'Duration (minutes)',
                    'Likes',
                    'Comments',
                    'Like Rate (%)',
                    'Comment Rate (%)',
                    'Engagement Rate (%)'
                ],
                'hyperparameter_tuning': True,
                'random_state': 42,
                'test_size': 0.2
            },
            'visualization': {
                'default_theme': 'plotly_white',
                'color_palette': [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                    '#bcbd22', '#17becf'
                ],
                'figure_size': {
                    'width': 800,
                    'height': 600
                },
                'save_format': 'html'
            },
            'dashboard': {
                'streamlit': {
                    'host': 'localhost',
                    'port': 8501
                },
                'dash': {
                    'host': '127.0.0.1',
                    'port': 8050,
                    'debug': True
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/youtube_analytics.log'
            },
            'analysis': {
                'min_videos_for_ml': 10,
                'outlier_threshold': 3.0,
                'correlation_threshold': 0.3,
                'engagement_rate_threshold': 5.0
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).
        
        Args:
            key: Configuration key (e.g., 'ml.default_model_type')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dots).
        
        Args:
            key: Configuration key (e.g., 'ml.default_model_type')
            value: Value to set
        """
        keys = key.split('.')
        config = self.data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with updates
        """
        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.data, updates)
    
    def save_to_file(self, filepath: str, format: str = 'json') -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(path, 'w') as f:
                json.dump(self.data, f, indent=2)
        elif format.lower() == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(self.data, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() == '.json':
                loaded_config = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                loaded_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self.update(loaded_config)
        logger.info(f"Configuration loaded from {filepath}")
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check required sections
            required_sections = ['data', 'ml', 'visualization', 'dashboard']
            for section in required_sections:
                if section not in self.data:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate data paths
            data_config = self.data['data']
            for key in ['export_directory', 'charts_directory']:
                if key in data_config:
                    Path(data_config[key]).mkdir(parents=True, exist_ok=True)
            
            # Validate ML configuration
            ml_config = self.data['ml']
            valid_models = ['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']
            if ml_config.get('default_model_type') not in valid_models:
                logger.warning(f"Unknown model type: {ml_config.get('default_model_type')}")
            
            # Validate visualization configuration
            viz_config = self.data['visualization']
            if 'color_palette' in viz_config and not isinstance(viz_config['color_palette'], list):
                logger.error("Color palette must be a list")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.data.copy()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.data, indent=2)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    config = Config()
    
    if config_path:
        try:
            config.load_from_file(config_path)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        # Try to load from standard locations
        standard_paths = [
            'config/config.json',
            'config/config.yaml',
            'config.json',
            'config.yaml'
        ]
        
        for path in standard_paths:
            if Path(path).exists():
                try:
                    config.load_from_file(path)
                    logger.info(f"Loaded configuration from {path}")
                    break
                except Exception as e:
                    logger.debug(f"Could not load config from {path}: {e}")
        else:
            logger.info("No configuration file found, using defaults")
    
    # Validate configuration
    if not config.validate():
        logger.warning("Configuration validation failed, some features may not work correctly")
    
    return config
