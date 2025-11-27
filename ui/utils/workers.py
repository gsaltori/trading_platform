# ui/utils/workers.py
"""
Background workers for long-running operations.

Uses PyQt6 signals for thread-safe GUI updates.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np

from PyQt6.QtCore import QObject, QThread, pyqtSignal, QRunnable, QThreadPool

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals for worker threads."""
    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str, str)  # error message, traceback
    progress = pyqtSignal(int, str)  # percentage, message
    result = pyqtSignal(object)  # result data
    log = pyqtSignal(str, str)  # level, message


class BaseWorker(QRunnable):
    """Base class for background workers."""
    
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self._is_cancelled = False
    
    def cancel(self):
        """Request cancellation of the worker."""
        self._is_cancelled = True
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._is_cancelled
    
    def emit_progress(self, percentage: int, message: str = ""):
        """Emit progress signal."""
        self.signals.progress.emit(percentage, message)
    
    def emit_log(self, level: str, message: str):
        """Emit log signal."""
        self.signals.log.emit(level, message)
    
    def run(self):
        """Execute the worker task. Override in subclasses."""
        self.signals.started.emit()
        try:
            result = self.execute()
            if not self._is_cancelled:
                self.signals.result.emit(result)
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.error(f"Worker error: {error_msg}\n{tb}")
            self.signals.error.emit(error_msg, tb)
        finally:
            self.signals.finished.emit()
    
    def execute(self) -> Any:
        """Execute the actual work. Override in subclasses."""
        raise NotImplementedError


class DataDownloadWorker(BaseWorker):
    """Worker for downloading historical data from MT5."""
    
    def __init__(self, mt5_connector, symbol: str, timeframe: str,
                 start_date: datetime, end_date: datetime = None,
                 count: int = None):
        super().__init__()
        self.mt5_connector = mt5_connector
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        self.count = count
    
    def execute(self) -> Optional[pd.DataFrame]:
        """Download historical data."""
        self.emit_progress(10, f"Connecting to MT5...")
        
        if not self.mt5_connector.connected:
            if not self.mt5_connector.initialize():
                raise ConnectionError("Could not connect to MT5")
        
        self.emit_progress(30, f"Downloading {self.symbol} {self.timeframe} data...")
        
        data = self.mt5_connector.get_historical_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date,
            count=self.count
        )
        
        if data is None or data.empty:
            raise ValueError(f"No data received for {self.symbol}")
        
        self.emit_progress(90, f"Downloaded {len(data)} candles")
        self.emit_log("INFO", f"Downloaded {len(data)} candles for {self.symbol} {self.timeframe}")
        
        return data


class MultiSymbolDataWorker(BaseWorker):
    """Worker for downloading data for multiple symbols."""
    
    def __init__(self, mt5_connector, symbols: List[str], timeframe: str,
                 start_date: datetime = None, end_date: datetime = None,
                 count: int = None):
        super().__init__()
        self.mt5_connector = mt5_connector
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        self.count = count
    
    def execute(self) -> Dict[str, pd.DataFrame]:
        """Download data for all symbols."""
        results = {}
        total = len(self.symbols)
        
        self.emit_progress(5, "Connecting to MT5...")
        
        if not self.mt5_connector.connected:
            if not self.mt5_connector.initialize():
                raise ConnectionError("Could not connect to MT5")
        
        for i, symbol in enumerate(self.symbols):
            if self.is_cancelled():
                break
            
            progress = int((i + 1) / total * 90) + 5
            self.emit_progress(progress, f"Downloading {symbol}...")
            
            try:
                data = self.mt5_connector.get_historical_data(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    count=self.count
                )
                
                if data is not None and not data.empty:
                    results[symbol] = data
                    self.emit_log("INFO", f"Downloaded {len(data)} candles for {symbol}")
                else:
                    self.emit_log("WARNING", f"No data for {symbol}")
                    
            except Exception as e:
                self.emit_log("ERROR", f"Error downloading {symbol}: {e}")
        
        return results


class BacktestWorker(BaseWorker):
    """Worker for running backtests."""
    
    def __init__(self, backtest_engine, strategy, data: pd.DataFrame,
                 symbol: str, commission: float = 0.001,
                 slippage_model: str = "fixed"):
        super().__init__()
        self.backtest_engine = backtest_engine
        self.strategy = strategy
        self.data = data
        self.symbol = symbol
        self.commission = commission
        self.slippage_model = slippage_model
    
    def execute(self):
        """Run the backtest."""
        self.emit_progress(10, "Initializing backtest...")
        
        self.emit_progress(30, f"Running strategy {self.strategy.name}...")
        
        result = self.backtest_engine.run_backtest(
            data=self.data,
            strategy=self.strategy,
            symbol=self.symbol,
            commission=self.commission,
            slippage_model=self.slippage_model
        )
        
        self.emit_progress(90, "Calculating metrics...")
        self.emit_log("INFO", f"Backtest completed: {result.total_return:.2f}% return")
        
        return result


class OptimizationWorker(BaseWorker):
    """Worker for strategy optimization."""
    
    def __init__(self, strategy_engine, strategy_type: str,
                 data: pd.DataFrame, parameter_ranges: Dict[str, List],
                 metric: str = 'sharpe_ratio',
                 callback: Callable[[int, Dict], None] = None):
        super().__init__()
        self.strategy_engine = strategy_engine
        self.strategy_type = strategy_type
        self.data = data
        self.parameter_ranges = parameter_ranges
        self.metric = metric
        self.callback = callback
    
    def execute(self) -> Dict[str, Any]:
        """Run the optimization."""
        from itertools import product
        
        self.emit_progress(5, "Preparing optimization...")
        
        param_combinations = list(product(*self.parameter_ranges.values()))
        param_names = list(self.parameter_ranges.keys())
        total = len(param_combinations)
        
        best_params = {}
        best_metric = -np.inf
        all_results = []
        
        self.emit_log("INFO", f"Testing {total} parameter combinations")
        
        for i, combination in enumerate(param_combinations):
            if self.is_cancelled():
                break
            
            progress = int((i + 1) / total * 90) + 5
            params = dict(zip(param_names, combination))
            
            self.emit_progress(progress, f"Testing combination {i+1}/{total}")
            
            try:
                from strategies.strategy_engine import StrategyConfig
                
                temp_config = StrategyConfig(
                    name="temp_optimization",
                    symbols=["OPTIMIZE"],
                    timeframe="H1",
                    parameters=params
                )
                
                temp_strategy = self.strategy_engine.available_strategies[self.strategy_type](temp_config)
                results = temp_strategy.run("OPTIMIZE", self.data.copy())
                
                metric_value = self._calculate_metric(results)
                
                all_results.append({
                    'params': params,
                    'metric': metric_value
                })
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()
                
                if self.callback:
                    self.callback(i, {'params': params, 'metric': metric_value})
                    
            except Exception as e:
                self.emit_log("WARNING", f"Error with params {params}: {e}")
                continue
        
        self.emit_log("INFO", f"Optimization complete. Best {self.metric}: {best_metric:.4f}")
        
        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'all_results': all_results
        }
    
    def _calculate_metric(self, data: pd.DataFrame) -> float:
        """Calculate performance metric."""
        if 'signal' not in data.columns:
            return -np.inf
        
        returns = data['close'].pct_change().shift(-1)
        strategy_returns = returns * data['signal'].shift(1)
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return -np.inf
        
        if self.metric == 'sharpe_ratio':
            if strategy_returns.std() > 0:
                return float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
            return -np.inf
        elif self.metric == 'total_return':
            return float(strategy_returns.sum())
        elif self.metric == 'win_rate':
            wins = (strategy_returns > 0).sum()
            total = (strategy_returns != 0).sum()
            return float(wins / total) if total > 0 else 0
        else:
            return float(strategy_returns.mean())


class StrategyGeneratorWorker(BaseWorker):
    """Worker for automatic strategy generation using ML."""
    
    def __init__(self, ml_engine, data: pd.DataFrame,
                 target_type: str = 'classification',
                 algorithms: List[str] = None,
                 optimize: bool = True):
        super().__init__()
        self.ml_engine = ml_engine
        self.data = data
        self.target_type = target_type
        self.algorithms = algorithms or ['random_forest', 'xgboost']
        self.optimize = optimize
    
    def execute(self) -> Dict[str, Any]:
        """Generate and evaluate strategies."""
        from ml.ml_engine import MLModelConfig
        
        results = {}
        total_steps = len(self.algorithms) * (2 if self.optimize else 1)
        current_step = 0
        
        self.emit_progress(5, "Preparing data...")
        
        for algorithm in self.algorithms:
            if self.is_cancelled():
                break
            
            current_step += 1
            progress = int(current_step / total_steps * 90) + 5
            
            self.emit_progress(progress, f"Training {algorithm} model...")
            self.emit_log("INFO", f"Training {algorithm} model...")
            
            try:
                config = MLModelConfig(
                    model_type=self.target_type,
                    algorithm=algorithm,
                    features=[],  # Will be auto-generated
                    target='price_direction',
                    lookback_window=50,
                    prediction_horizon=1,
                    train_test_split=0.8,
                    parameters={}
                )
                
                result = self.ml_engine.train_model(self.data.copy(), config)
                
                results[algorithm] = {
                    'metrics': result.metrics,
                    'feature_importance': result.feature_importance,
                    'classification_report': result.classification_report,
                    'model_name': result.model_name
                }
                
                self.emit_log("INFO", f"{algorithm} accuracy: {result.metrics.get('accuracy', 0):.3f}")
                
            except Exception as e:
                self.emit_log("ERROR", f"Error training {algorithm}: {e}")
                results[algorithm] = {'error': str(e)}
        
        # Find best model
        best_model = None
        best_accuracy = 0
        for alg, res in results.items():
            if 'metrics' in res:
                acc = res['metrics'].get('accuracy', 0)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = alg
        
        return {
            'results': results,
            'best_model': best_model,
            'best_accuracy': best_accuracy
        }


class LiveTradingWorker(BaseWorker):
    """Worker for live trading execution."""
    
    def __init__(self, execution_engine, strategy, mt5_connector,
                 symbol: str, check_interval: int = 1):
        super().__init__()
        self.execution_engine = execution_engine
        self.strategy = strategy
        self.mt5_connector = mt5_connector
        self.symbol = symbol
        self.check_interval = check_interval
    
    def execute(self):
        """Run live trading loop."""
        import time
        
        self.emit_progress(10, "Starting live trading...")
        self.emit_log("INFO", f"Live trading started for {self.symbol}")
        
        iteration = 0
        
        while not self.is_cancelled():
            iteration += 1
            
            try:
                # Get latest data
                data = self.mt5_connector.get_historical_data(
                    symbol=self.symbol,
                    timeframe=self.strategy.config.timeframe,
                    start_date=None,
                    count=200
                )
                
                if data is not None and not data.empty:
                    # Run strategy
                    signals = self.strategy.run(self.symbol, data)
                    
                    # Check for signals
                    if len(signals) > 0:
                        last_signal = signals.iloc[-1]
                        if last_signal.get('signal', 0) != 0:
                            self.emit_log("INFO", f"Signal detected: {'BUY' if last_signal['signal'] > 0 else 'SELL'}")
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.emit_log("ERROR", f"Error in trading loop: {e}")
                time.sleep(5)  # Wait longer on error
        
        self.emit_log("INFO", "Live trading stopped")
        return {'iterations': iteration}


class MarketScannerWorker(BaseWorker):
    """Worker for scanning multiple markets for signals."""
    
    def __init__(self, mt5_connector, strategy_engine, symbols: List[str],
                 timeframe: str, strategy_name: str):
        super().__init__()
        self.mt5_connector = mt5_connector
        self.strategy_engine = strategy_engine
        self.symbols = symbols
        self.timeframe = timeframe
        self.strategy_name = strategy_name
    
    def execute(self) -> Dict[str, Any]:
        """Scan markets for signals."""
        signals = {}
        total = len(self.symbols)
        
        self.emit_progress(5, "Starting market scan...")
        
        for i, symbol in enumerate(self.symbols):
            if self.is_cancelled():
                break
            
            progress = int((i + 1) / total * 90) + 5
            self.emit_progress(progress, f"Scanning {symbol}...")
            
            try:
                # Get data
                data = self.mt5_connector.get_historical_data(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    start_date=None,
                    count=200
                )
                
                if data is not None and not data.empty:
                    # Get signals
                    strategy_signals = self.strategy_engine.get_strategy_signals(
                        self.strategy_name, symbol, data
                    )
                    
                    if strategy_signals:
                        # Get the most recent signal
                        last_signal = strategy_signals[-1]
                        signals[symbol] = {
                            'direction': 'BUY' if last_signal.direction > 0 else 'SELL',
                            'strength': last_signal.strength,
                            'price': last_signal.price,
                            'timestamp': last_signal.timestamp,
                            'stop_loss': last_signal.stop_loss,
                            'take_profit': last_signal.take_profit
                        }
                        self.emit_log("INFO", f"Signal found for {symbol}: {signals[symbol]['direction']}")
                        
            except Exception as e:
                self.emit_log("WARNING", f"Error scanning {symbol}: {e}")
        
        return signals
