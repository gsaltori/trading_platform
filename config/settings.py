# config/settings.py
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import yaml

@dataclass
class DatabaseConfig:
    postgres_url: str = "postgresql://user:pass@localhost:5432/trading"
    influx_url: str = "http://localhost:8086"
    influx_token: str = "your-token"
    influx_org: str = "trading"
    redis_url: str = "redis://localhost:6379/0"

@dataclass
class MT5Config:
    path: str = "C:/Program Files/MetaTrader 5/terminal64.exe"
    server: str = ""
    login: int = 0
    password: str = ""
    timeout: int = 60000
    portable: bool = False

@dataclass
class RiskConfig:
    max_drawdown: float = 0.15
    max_position_size: float = 0.1
    daily_loss_limit: float = 0.05
    correlation_threshold: float = 0.7

@dataclass
class OptimizationConfig:
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

class ConfigManager:
    def __init__(self, config_path: str = "config/platform_config.yaml"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        
        self.database = DatabaseConfig()
        self.mt5 = MT5Config()
        self.risk = RiskConfig()
        self.optimization = OptimizationConfig()
        
        self.load_config()
    
    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._update_from_dict(config_data)
        else:
            self.save_config()
    
    def save_config(self):
        config_data = {
            'database': asdict(self.database),
            'mt5': asdict(self.mt5),
            'risk': asdict(self.risk),
            'optimization': asdict(self.optimization)
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)