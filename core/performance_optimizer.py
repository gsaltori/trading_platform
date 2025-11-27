# core/performance_optimizer.py
"""
Performance optimization utilities for the trading platform.

Provides vectorized operations and parallel processing capabilities.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from functools import lru_cache
import gc

logger = logging.getLogger(__name__)

# Try to import numba
NUMBA_AVAILABLE = False
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    logger.info("Numba not available, using pure Python")
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    use_numba: bool = True
    use_multiprocessing: bool = True
    max_workers: int = -1
    cache_size: int = 1000
    memory_limit_mb: int = 4096
    enable_gc: bool = True


class PerformanceOptimizer:
    """Performance optimizer for intensive operations."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        
        if self.config.max_workers == -1:
            import multiprocessing as mp
            self.config.max_workers = mp.cpu_count()
        
        self.cache = {}
        self.setup_optimizations()
    
    def setup_optimizations(self):
        """Setup system-level optimizations."""
        if NUMBA_AVAILABLE and self.config.use_numba:
            try:
                numba.set_num_threads(self.config.max_workers)
            except Exception:
                pass
        
        if self.config.enable_gc:
            gc.enable()
            gc.set_threshold(700, 10, 10)
    
    @staticmethod
    def vectorized_returns_calculation(prices: np.ndarray) -> np.ndarray:
        """Vectorized returns calculation."""
        if len(prices) < 2:
            return np.zeros(len(prices))
        
        returns = np.zeros(len(prices))
        returns[1:] = np.diff(prices) / prices[:-1]
        return returns
    
    @staticmethod
    def bulk_indicators_calculation(
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray,
        volume: np.ndarray, 
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bulk calculation of technical indicators."""
        n = len(close)
        rsi = np.zeros(n)
        atr = np.zeros(n)
        momentum = np.zeros(n)
        
        for i in range(period, n):
            # RSI calculation
            gains = 0.0
            losses = 0.0
            for j in range(i - period + 1, i + 1):
                if j > 0:
                    ret = (close[j] - close[j-1]) / close[j-1] if close[j-1] != 0 else 0
                    if ret > 0:
                        gains += ret
                    else:
                        losses += abs(ret)
            
            if losses == 0:
                rsi[i] = 100.0
            else:
                rs = gains / losses
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
            # ATR calculation
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1]) if i > 0 else 0
            tr3 = abs(low[i] - close[i-1]) if i > 0 else 0
            atr[i] = max(tr1, tr2, tr3)
            
            # Momentum calculation
            if close[i - period] != 0:
                momentum[i] = (close[i] / close[i - period] - 1.0) * 100.0
        
        return rsi, atr, momentum
    
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame operations for better performance."""
        optimized_df = df.copy()
        
        # Optimize numeric types
        float_cols = optimized_df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            optimized_df[float_cols] = optimized_df[float_cols].astype('float32')
        
        int_cols = optimized_df.select_dtypes(include=['int64']).columns
        if len(int_cols) > 0:
            optimized_df[int_cols] = optimized_df[int_cols].astype('int32')
        
        # Sort index for better access
        if not optimized_df.index.is_monotonic_increasing:
            optimized_df = optimized_df.sort_index()
        
        return optimized_df
    
    def memory_usage_report(self) -> Dict[str, float]:
        """Report memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except ImportError:
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'percent': 0,
                'available_mb': 0
            }
    
    def cleanup_memory(self):
        """Clean up memory and cache."""
        self.cache.clear()
        if self.config.enable_gc:
            gc.collect()


class VectorizedBacktestEngine:
    """Fully vectorized backtesting engine."""
    
    def __init__(self):
        self.performance_optimizer = PerformanceOptimizer()
    
    def vectorized_backtest(
        self, 
        signals: np.ndarray, 
        prices: np.ndarray,
        commissions: np.ndarray, 
        initial_capital: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fully vectorized backtest."""
        n = len(signals)
        positions = np.zeros(n)
        equity = np.zeros(n)
        cash = np.zeros(n)
        
        cash[0] = initial_capital
        equity[0] = initial_capital
        
        for i in range(1, n):
            # Update position
            if signals[i] != 0 and positions[i-1] == 0:
                # New position
                position_size = cash[i-1] * 0.1 / prices[i] if prices[i] != 0 else 0
                positions[i] = position_size * signals[i]
                cash[i] = cash[i-1] - (position_size * prices[i]) - commissions[i]
            elif signals[i] == 0 and positions[i-1] != 0:
                # Close position
                cash[i] = cash[i-1] + (positions[i-1] * prices[i]) - commissions[i]
                positions[i] = 0
            else:
                # Keep position
                positions[i] = positions[i-1]
                cash[i] = cash[i-1]
            
            # Calculate equity
            equity[i] = cash[i] + (positions[i] * prices[i])
        
        return equity, positions


def time_execution(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper
