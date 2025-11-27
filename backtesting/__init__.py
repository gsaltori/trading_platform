# backtesting/__init__.py
"""Backtesting module for the trading platform."""

from .backtest_engine import BacktestEngine, BacktestResult, Trade, MultiStrategyBacktester

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'Trade',
    'MultiStrategyBacktester'
]
