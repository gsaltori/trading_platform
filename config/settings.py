# config/settings.py
"""
Configuration management for the trading platform.

Supports YAML configuration files with environment variable overrides.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    postgres_url: str = "sqlite:///data/sqlite/trading.db"
    influx_url: str = ""
    influx_token: str = ""
    influx_org: str = "trading"
    redis_url: str = ""
    
    def __post_init__(self):
        # Override with environment variables if set
        self.postgres_url = os.environ.get('DATABASE_URL', self.postgres_url)
        self.influx_url = os.environ.get('INFLUX_URL', self.influx_url)
        self.influx_token = os.environ.get('INFLUX_TOKEN', self.influx_token)
        self.redis_url = os.environ.get('REDIS_URL', self.redis_url)


@dataclass
class MT5Config:
    """MetaTrader 5 configuration."""
    path: str = "C:/Program Files/MetaTrader 5/terminal64.exe"
    server: str = ""
    login: int = 0
    password: str = ""
    timeout: int = 60000
    portable: bool = False
    
    def __post_init__(self):
        # Override with environment variables if set
        self.login = int(os.environ.get('MT5_LOGIN', self.login))
        self.password = os.environ.get('MT5_PASSWORD', self.password)
        self.server = os.environ.get('MT5_SERVER', self.server)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown: float = 0.15
    max_position_size: float = 0.1
    daily_loss_limit: float = 0.05
    correlation_threshold: float = 0.7
    max_correlated_positions: int = 3
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.5


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    n_workers: int = -1  # -1 for auto-detect


@dataclass
class TradingConfig:
    """Trading execution configuration."""
    max_slippage: float = 0.001
    default_commission: float = 0.001
    max_spread_multiplier: float = 2.0
    allow_hedging: bool = False
    default_timeframe: str = "H1"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/trading_platform.log"
    max_size_mb: int = 10
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ConfigManager:
    """
    Manages platform configuration.
    
    Loads from YAML file with support for environment variable overrides.
    """
    
    def __init__(self, config_path: str = "config/platform_config.yaml"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        
        # Initialize with defaults
        self.database = DatabaseConfig()
        self.mt5 = MT5Config()
        self.risk = RiskConfig()
        self.optimization = OptimizationConfig()
        self.trading = TradingConfig()
        self.logging = LoggingConfig()
        
        # Load from file if exists
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                    self._update_from_dict(config_data)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Error loading config, using defaults: {e}")
                self.save_config()
        else:
            logger.info(f"No config file found, creating default at {self.config_path}")
            self.save_config()
    
    def save_config(self):
        """Save current configuration to YAML file."""
        try:
            config_data = {
                'database': asdict(self.database),
                'mt5': asdict(self.mt5),
                'risk': asdict(self.risk),
                'optimization': asdict(self.optimization),
                'trading': asdict(self.trading),
                'logging': asdict(self.logging)
            }
            
            # Don't save sensitive data
            if config_data['mt5']['password']:
                config_data['mt5']['password'] = '***'
            if config_data['database']['influx_token']:
                config_data['database']['influx_token'] = '***'
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        section_map = {
            'database': self.database,
            'mt5': self.mt5,
            'risk': self.risk,
            'optimization': self.optimization,
            'trading': self.trading,
            'logging': self.logging
        }
        
        for section, values in config_dict.items():
            if section in section_map and isinstance(values, dict):
                section_obj = section_map[section]
                for key, value in values.items():
                    if hasattr(section_obj, key) and value is not None:
                        # Handle special cases
                        if key == 'password' and value == '***':
                            continue
                        if key == 'influx_token' and value == '***':
                            continue
                        setattr(section_obj, key, value)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        section_obj = getattr(self, section, None)
        if section_obj:
            return getattr(section_obj, key, default)
        return default
    
    def set(self, section: str, key: str, value: Any):
        """Set a specific configuration value."""
        section_obj = getattr(self, section, None)
        if section_obj and hasattr(section_obj, key):
            setattr(section_obj, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': asdict(self.database),
            'mt5': asdict(self.mt5),
            'risk': asdict(self.risk),
            'optimization': asdict(self.optimization),
            'trading': asdict(self.trading),
            'logging': asdict(self.logging)
        }
    
    def validate(self) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Validate risk parameters
        if not 0 < self.risk.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")
        if not 0 < self.risk.max_position_size <= 1:
            errors.append("max_position_size must be between 0 and 1")
        if not 0 < self.risk.daily_loss_limit <= 1:
            errors.append("daily_loss_limit must be between 0 and 1")
        
        # Validate optimization parameters
        if self.optimization.population_size < 10:
            errors.append("population_size should be at least 10")
        if self.optimization.generations < 1:
            errors.append("generations must be positive")
        
        if errors:
            for error in errors:
                logger.warning(f"Config validation error: {error}")
            return False
        
        return True


def load_config(config_path: str = "config/platform_config.yaml") -> ConfigManager:
    """Convenience function to load configuration."""
    return ConfigManager(config_path)
