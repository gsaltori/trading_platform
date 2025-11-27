# data/__init__.py
"""Data module for the trading platform."""

from .mt5_connector import MT5ConnectionManager, SymbolInfo

__all__ = ['MT5ConnectionManager', 'SymbolInfo']
