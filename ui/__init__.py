# ui/__init__.py
"""
Trading Platform GUI module.

Provides a comprehensive graphical user interface for:
- MetaTrader 5 connection management
- Data downloading and visualization
- Strategy generation and optimization
- Backtesting and analysis
"""

from .main_window import run_gui, MainWindow

__all__ = ['run_gui', 'MainWindow']
