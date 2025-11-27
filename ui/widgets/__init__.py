# ui/widgets/__init__.py
"""
UI Widgets Module.

Provides all widget components for the trading platform GUI.
"""

from .mt5_connection_widget import MT5ConnectionWidget
from .data_panel import DataPanel
from .strategy_generator_widget import StrategyGeneratorWidget
from .backtest_results_widget import BacktestResultsWidget
from .charts_widget import ChartsWidget
from .log_widget import LogWidget
from .auto_generator_widget import AutoGeneratorWidget

__all__ = [
    'MT5ConnectionWidget',
    'DataPanel',
    'StrategyGeneratorWidget',
    'BacktestResultsWidget',
    'ChartsWidget',
    'LogWidget',
    'AutoGeneratorWidget'
]
