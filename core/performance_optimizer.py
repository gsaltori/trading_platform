# core/performance_optimizer.py
import numpy as np
import pandas as pd
import numba
from numba import jit, prange, vectorize, float64, int64
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import gc

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    use_numba: bool = True
    use_multiprocessing: bool = True
    max_workers: int = -1
    cache_size: int = 1000
    memory_limit_mb: int = 4096
    enable_gc: bool = True

class PerformanceOptimizer:
    """Optimizador de performance para operaciones intensivas"""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        if self.config.max_workers == -1:
            self.config.max_workers = mp.cpu_count()
        
        self.cache = {}
        self.setup_optimizations()
    
    def setup_optimizations(self):
        """Configurar optimizaciones a nivel de sistema"""
        if self.config.use_numba:
            numba.set_num_threads(self.config.max_workers)
        
        if self.config.enable_gc:
            # Optimizar garbage collector
            gc.enable()
            gc.set_threshold(700, 10, 10)
    
    @lru_cache(maxsize=1000)
    def cached_technical_calculation(self, data_hash: str, calculation_type: str, **params):
        """Caché para cálculos técnicos frecuentes"""
        return self._perform_technical_calculation(calculation_type, **params)
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def vectorized_returns_calculation(prices: np.ndarray) -> np.ndarray:
        """Cálculo vectorizado de retornos optimizado con Numba"""
        n = len(prices)
        returns = np.zeros(n)
        for i in prange(1, n):
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
        return returns
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def bulk_indicators_calculation(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                  volume: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cálculo masivo de indicadores técnicos"""
        n = len(close)
        rsi = np.zeros(n)
        atr = np.zeros(n)
        momentum = np.zeros(n)
        
        for i in prange(period, n):
            # RSI calculation
            gains = 0.0
            losses = 0.0
            for j in range(i - period + 1, i + 1):
                ret = (close[j] - close[j-1]) / close[j-1]
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
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            atr[i] = max(tr1, tr2, tr3)
            
            # Momentum calculation
            momentum[i] = (close[i] / close[i - period] - 1.0) * 100.0
        
        return rsi, atr, momentum
    
    def parallel_backtest(self, strategies: List, data_dict: Dict[str, pd.DataFrame], 
                         **backtest_params) -> Dict[str, Any]:
        """Ejecutar múltiples backtests en paralelo"""
        from backtesting.backtest_engine import BacktestEngine
        
        def run_single_backtest(strategy, symbol, data):
            engine = BacktestEngine(initial_capital=10000)
            return engine.run_backtest(data, strategy, symbol, **backtest_params)
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            for strategy in strategies:
                for symbol in strategy.config.symbols:
                    if symbol in data_dict:
                        future = executor.submit(
                            run_single_backtest, strategy, symbol, data_dict[symbol]
                        )
                        futures[future] = (strategy.name, symbol)
            
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minutos timeout
                    strategy_name, symbol = futures[future]
                    if strategy_name not in results:
                        results[strategy_name] = {}
                    results[strategy_name][symbol] = result
                except Exception as e:
                    logger.error(f"Error en backtest paralelo: {e}")
        
        return results
    
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimizar operaciones con DataFrame para mejor performance"""
        # Convertir a tipos optimizados
        optimized_df = df.copy()
        
        # Optimizar tipos numéricos
        float_cols = optimized_df.select_dtypes(include=['float64']).columns
        optimized_df[float_cols] = optimized_df[float_cols].astype('float32')
        
        int_cols = optimized_df.select_dtypes(include=['int64']).columns
        optimized_df[int_cols] = optimized_df[int_cols].astype('int32')
        
        # Ordenar índice para mejor acceso
        if not optimized_df.index.is_monotonic_increasing:
            optimized_df = optimized_df.sort_index()
        
        return optimized_df
    
    def memory_usage_report(self) -> Dict[str, float]:
        """Reporte de uso de memoria"""
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
    
    def cleanup_memory(self):
        """Limpiar memoria y caché"""
        self.cache.clear()
        if self.config.enable_gc:
            gc.collect()
    
    def _perform_technical_calculation(self, calculation_type: str, **params):
        """Realizar cálculo técnico específico"""
        # Implementación de cálculos técnicos optimizados
        pass

class VectorizedBacktestEngine:
    """Motor de backtesting completamente vectorizado"""
    
    def __init__(self):
        self.performance_optimizer = PerformanceOptimizer()
    
    @jit(nopython=True, parallel=True)
    def vectorized_backtest(self, signals: np.ndarray, prices: np.ndarray, 
                          commissions: np.ndarray, initial_capital: float) -> Tuple[np.ndarray, np.ndarray]:
        """Backtest completamente vectorizado"""
        n = len(signals)
        positions = np.zeros(n)
        equity = np.zeros(n)
        cash = np.zeros(n)
        
        cash[0] = initial_capital
        equity[0] = initial_capital
        
        for i in prange(1, n):
            # Actualizar posición
            if signals[i] != 0 and positions[i-1] == 0:
                # Nueva posición
                position_size = cash[i-1] * 0.1 / prices[i]  # 10% del capital
                positions[i] = position_size * signals[i]
                cash[i] = cash[i-1] - (position_size * prices[i]) - commissions[i]
            elif signals[i] == 0 and positions[i-1] != 0:
                # Cerrar posición
                cash[i] = cash[i-1] + (positions[i-1] * prices[i]) - commissions[i]
                positions[i] = 0
            else:
                # Mantener posición
                positions[i] = positions[i-1]
                cash[i] = cash[i-1]
            
            # Calcular equity
            equity[i] = cash[i] + (positions[i] * prices[i])
        
        return equity, positions

# Decoradores de optimización para funciones críticas
def optimize_with_numba(func):
    """Decorador para optimizar funciones con Numba"""
    try:
        return jit(nopython=True, fastmath=True, cache=True)(func)
    except Exception as e:
        logger.warning(f"No se pudo optimizar {func.__name__} con Numba: {e}")
        return func

def time_execution(func):
    """Decorador para medir tiempo de ejecución"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} ejecutado en {execution_time:.4f} segundos")
        return result
    return wrapper