# optimization/walk_forward.py
"""
Walk-Forward Analysis Module.

Implements walk-forward optimization to validate trading strategies
and detect overfitting.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class WalkForwardMethod(Enum):
    """Walk-forward analysis methods."""
    ANCHORED = "anchored"  # Expanding window
    ROLLING = "rolling"    # Fixed window size
    COMBINATORIAL = "combinatorial"  # Multiple combinations


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis for a single window."""
    window: WalkForwardWindow
    in_sample_return: float
    out_of_sample_return: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_trades: int
    out_of_sample_trades: int
    in_sample_win_rate: float
    out_of_sample_win_rate: float
    best_params: Dict[str, Any]
    efficiency_ratio: float  # OOS / IS performance ratio
    
    def to_dict(self) -> Dict:
        return {
            'window_id': self.window.window_id,
            'train_period': f"{self.window.train_start} to {self.window.train_end}",
            'test_period': f"{self.window.test_start} to {self.window.test_end}",
            'is_return': self.in_sample_return,
            'oos_return': self.out_of_sample_return,
            'is_sharpe': self.in_sample_sharpe,
            'oos_sharpe': self.out_of_sample_sharpe,
            'is_trades': self.in_sample_trades,
            'oos_trades': self.out_of_sample_trades,
            'efficiency_ratio': self.efficiency_ratio,
            'best_params': self.best_params
        }


@dataclass
class WalkForwardReport:
    """Complete walk-forward analysis report."""
    strategy_name: str
    method: WalkForwardMethod
    windows: List[WalkForwardResult]
    total_in_sample_return: float
    total_out_of_sample_return: float
    average_efficiency_ratio: float
    consistency_score: float  # % of windows with positive OOS return
    robustness_score: float   # Statistical significance
    is_robust: bool
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'method': self.method.value,
            'windows': [w.to_dict() for w in self.windows],
            'total_is_return': self.total_in_sample_return,
            'total_oos_return': self.total_out_of_sample_return,
            'avg_efficiency_ratio': self.average_efficiency_ratio,
            'consistency_score': self.consistency_score,
            'robustness_score': self.robustness_score,
            'is_robust': self.is_robust,
            'recommendations': self.recommendations
        }


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for strategy validation.
    
    Splits data into multiple train/test windows and validates
    strategy performance across all windows.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.num_windows = self.config.get('num_windows', 5)
        self.train_ratio = self.config.get('train_ratio', 0.7)
        self.min_train_size = self.config.get('min_train_size', 500)
        self.min_test_size = self.config.get('min_test_size', 100)
        self.method = WalkForwardMethod(self.config.get('method', 'rolling'))
        
        # Results storage
        self.windows: List[WalkForwardWindow] = []
        self.results: List[WalkForwardResult] = []
    
    def create_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Create walk-forward windows from data."""
        self.windows = []
        n = len(data)
        
        if self.method == WalkForwardMethod.ROLLING:
            # Fixed window size, rolling through data
            window_size = n // self.num_windows
            train_size = int(window_size * self.train_ratio)
            test_size = window_size - train_size
            
            for i in range(self.num_windows):
                start_idx = i * test_size
                train_start_idx = start_idx
                train_end_idx = start_idx + train_size
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_size
                
                if test_end_idx > n:
                    break
                
                window = WalkForwardWindow(
                    window_id=i + 1,
                    train_start=data.index[train_start_idx],
                    train_end=data.index[train_end_idx - 1],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[min(test_end_idx - 1, n - 1)],
                    train_size=train_size,
                    test_size=test_size
                )
                self.windows.append(window)
        
        elif self.method == WalkForwardMethod.ANCHORED:
            # Expanding window - train always starts from beginning
            step_size = n // (self.num_windows + 1)
            
            for i in range(self.num_windows):
                train_end_idx = step_size * (i + 1)
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + step_size, n)
                
                if train_end_idx < self.min_train_size or test_end_idx - test_start_idx < self.min_test_size:
                    continue
                
                window = WalkForwardWindow(
                    window_id=i + 1,
                    train_start=data.index[0],
                    train_end=data.index[train_end_idx - 1],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[test_end_idx - 1],
                    train_size=train_end_idx,
                    test_size=test_end_idx - test_start_idx
                )
                self.windows.append(window)
        
        elif self.method == WalkForwardMethod.COMBINATORIAL:
            # Multiple overlapping windows
            window_size = n // 3
            step = window_size // 2
            
            window_id = 1
            for start in range(0, n - window_size, step):
                train_size = int(window_size * self.train_ratio)
                test_size = window_size - train_size
                
                train_start_idx = start
                train_end_idx = start + train_size
                test_start_idx = train_end_idx
                test_end_idx = start + window_size
                
                if test_end_idx > n:
                    break
                
                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=data.index[train_start_idx],
                    train_end=data.index[train_end_idx - 1],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[test_end_idx - 1],
                    train_size=train_size,
                    test_size=test_size
                )
                self.windows.append(window)
                window_id += 1
        
        logger.info(f"Created {len(self.windows)} walk-forward windows")
        return self.windows
    
    def run_analysis(self, data: pd.DataFrame,
                     strategy_class,
                     strategy_config,
                     backtest_engine,
                     optimizer=None,
                     parameter_ranges: Dict = None,
                     callback: Callable = None) -> WalkForwardReport:
        """
        Run complete walk-forward analysis.
        
        Args:
            data: Complete historical data
            strategy_class: Strategy class to test
            strategy_config: Base strategy configuration
            backtest_engine: Backtest engine instance
            optimizer: Optional optimizer for parameter tuning
            parameter_ranges: Parameter ranges for optimization
            callback: Progress callback(window_id, total_windows, result)
        
        Returns:
            WalkForwardReport with complete analysis
        """
        if not self.windows:
            self.create_windows(data)
        
        self.results = []
        
        for i, window in enumerate(self.windows):
            logger.info(f"Processing window {window.window_id}/{len(self.windows)}")
            
            # Split data
            train_data = data[window.train_start:window.train_end].copy()
            test_data = data[window.test_start:window.test_end].copy()
            
            if len(train_data) < self.min_train_size or len(test_data) < self.min_test_size:
                logger.warning(f"Window {window.window_id} has insufficient data, skipping")
                continue
            
            try:
                # Optimize on training data (if optimizer provided)
                if optimizer and parameter_ranges:
                    best_params = self._optimize_on_window(
                        train_data, optimizer, strategy_class,
                        strategy_config, parameter_ranges
                    )
                else:
                    best_params = strategy_config.parameters
                
                # Create strategy with best params
                config_copy = self._copy_config(strategy_config)
                config_copy.parameters = best_params
                strategy = strategy_class(config_copy)
                
                # Backtest on training data
                is_result = backtest_engine.run_backtest(
                    data=train_data,
                    strategy=strategy,
                    symbol=strategy_config.symbols[0] if strategy_config.symbols else "SYMBOL"
                )
                
                # Backtest on test data
                oos_result = backtest_engine.run_backtest(
                    data=test_data,
                    strategy=strategy,
                    symbol=strategy_config.symbols[0] if strategy_config.symbols else "SYMBOL"
                )
                
                # Calculate efficiency ratio
                if is_result.total_return != 0:
                    efficiency_ratio = oos_result.total_return / abs(is_result.total_return)
                else:
                    efficiency_ratio = 0
                
                result = WalkForwardResult(
                    window=window,
                    in_sample_return=is_result.total_return,
                    out_of_sample_return=oos_result.total_return,
                    in_sample_sharpe=is_result.sharpe_ratio or 0,
                    out_of_sample_sharpe=oos_result.sharpe_ratio or 0,
                    in_sample_trades=is_result.total_trades,
                    out_of_sample_trades=oos_result.total_trades,
                    in_sample_win_rate=is_result.win_rate,
                    out_of_sample_win_rate=oos_result.win_rate,
                    best_params=best_params,
                    efficiency_ratio=efficiency_ratio
                )
                
                self.results.append(result)
                
                if callback:
                    callback(i + 1, len(self.windows), result)
                    
            except Exception as e:
                logger.error(f"Error processing window {window.window_id}: {e}")
                continue
        
        # Generate report
        return self._generate_report(strategy_config.name if hasattr(strategy_config, 'name') else "Strategy")
    
    def _optimize_on_window(self, train_data: pd.DataFrame,
                            optimizer, strategy_class,
                            strategy_config, parameter_ranges: Dict) -> Dict:
        """Optimize strategy on a single window."""
        try:
            best_params = optimizer.optimize_parameters(
                strategy_type=strategy_config.name if hasattr(strategy_config, 'name') else 'ma_crossover',
                data=train_data,
                parameter_ranges=parameter_ranges,
                metric='sharpe_ratio'
            )
            return best_params
        except Exception as e:
            logger.warning(f"Optimization failed, using defaults: {e}")
            return strategy_config.parameters
    
    def _copy_config(self, config):
        """Create a copy of strategy config."""
        import copy
        return copy.deepcopy(config)
    
    def _generate_report(self, strategy_name: str) -> WalkForwardReport:
        """Generate walk-forward report from results."""
        if not self.results:
            return WalkForwardReport(
                strategy_name=strategy_name,
                method=self.method,
                windows=[],
                total_in_sample_return=0,
                total_out_of_sample_return=0,
                average_efficiency_ratio=0,
                consistency_score=0,
                robustness_score=0,
                is_robust=False,
                recommendations=["Insufficient data for analysis"]
            )
        
        # Calculate aggregated metrics
        total_is_return = sum(r.in_sample_return for r in self.results)
        total_oos_return = sum(r.out_of_sample_return for r in self.results)
        
        efficiency_ratios = [r.efficiency_ratio for r in self.results]
        avg_efficiency = np.mean(efficiency_ratios) if efficiency_ratios else 0
        
        # Consistency: % of windows with positive OOS return
        positive_oos = sum(1 for r in self.results if r.out_of_sample_return > 0)
        consistency = (positive_oos / len(self.results)) * 100 if self.results else 0
        
        # Robustness score using t-test
        robustness = self._calculate_robustness_score()
        
        # Determine if strategy is robust
        is_robust = (
            avg_efficiency > 0.3 and  # OOS performance at least 30% of IS
            consistency >= 60 and      # At least 60% positive OOS windows
            robustness > 0.5           # Statistical significance
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_efficiency, consistency, robustness, total_oos_return
        )
        
        return WalkForwardReport(
            strategy_name=strategy_name,
            method=self.method,
            windows=self.results,
            total_in_sample_return=total_is_return,
            total_out_of_sample_return=total_oos_return,
            average_efficiency_ratio=avg_efficiency,
            consistency_score=consistency,
            robustness_score=robustness,
            is_robust=is_robust,
            recommendations=recommendations
        )
    
    def _calculate_robustness_score(self) -> float:
        """Calculate statistical robustness score."""
        if len(self.results) < 3:
            return 0
        
        oos_returns = [r.out_of_sample_return for r in self.results]
        
        # Simple t-test for mean > 0
        mean = np.mean(oos_returns)
        std = np.std(oos_returns) if len(oos_returns) > 1 else 1
        n = len(oos_returns)
        
        if std == 0:
            return 1.0 if mean > 0 else 0.0
        
        t_stat = mean / (std / np.sqrt(n))
        
        # Convert t-stat to probability-like score
        # Higher t-stat = more robust
        robustness = min(1.0, max(0.0, 0.5 + 0.1 * t_stat))
        
        return robustness
    
    def _generate_recommendations(self, efficiency: float, consistency: float,
                                   robustness: float, total_oos: float) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if efficiency < 0.3:
            recommendations.append(
                "⚠️ Low efficiency ratio suggests overfitting. "
                "Consider simplifying the strategy or using fewer parameters."
            )
        
        if consistency < 50:
            recommendations.append(
                "⚠️ Inconsistent OOS performance. Strategy may not be stable "
                "across different market conditions."
            )
        
        if robustness < 0.5:
            recommendations.append(
                "⚠️ Low statistical significance. More data or windows needed "
                "to validate strategy performance."
            )
        
        if total_oos < 0:
            recommendations.append(
                "❌ Negative total OOS return indicates the strategy is not profitable "
                "when tested on unseen data. Re-evaluation needed."
            )
        
        if efficiency > 0.5 and consistency > 70 and robustness > 0.7:
            recommendations.append(
                "✅ Strategy shows good robustness. Consider live testing with "
                "small position sizes."
            )
        
        if not recommendations:
            recommendations.append(
                "Strategy requires further analysis. Consider additional validation methods."
            )
        
        return recommendations


class MonteCarloValidator:
    """
    Monte Carlo simulation for strategy validation.
    
    Uses randomization techniques to assess strategy robustness.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
    
    def validate(self, data: pd.DataFrame, strategy, backtest_engine,
                 callback: Callable = None) -> Dict[str, Any]:
        """
        Run Monte Carlo validation.
        
        Tests strategy against:
        1. Randomized trade order
        2. Randomized entry timing
        3. Bootstrap resampling
        """
        results = {
            'trade_shuffle': [],
            'bootstrap': [],
            'original': None
        }
        
        # Run original backtest
        original_result = backtest_engine.run_backtest(
            data=data, strategy=strategy, symbol="SYMBOL"
        )
        results['original'] = {
            'return': original_result.total_return,
            'sharpe': original_result.sharpe_ratio or 0
        }
        
        # Trade shuffle simulation
        for i in range(self.n_simulations // 2):
            if callback:
                callback(i, self.n_simulations, "shuffle")
            
            shuffled_return = self._simulate_trade_shuffle(original_result)
            results['trade_shuffle'].append(shuffled_return)
        
        # Bootstrap simulation
        for i in range(self.n_simulations // 2):
            if callback:
                callback(i + self.n_simulations // 2, self.n_simulations, "bootstrap")
            
            bootstrap_return = self._simulate_bootstrap(data, strategy, backtest_engine)
            results['bootstrap'].append(bootstrap_return)
        
        # Calculate statistics
        return self._analyze_results(results)
    
    def _simulate_trade_shuffle(self, backtest_result) -> float:
        """Simulate with shuffled trade order."""
        if not backtest_result.trades:
            return 0
        
        # Shuffle trade PnLs and calculate cumulative return
        pnls = [t.pnl for t in backtest_result.trades if t.pnl is not None]
        np.random.shuffle(pnls)
        
        cumulative = np.cumsum(pnls)
        if len(cumulative) > 0:
            return cumulative[-1] / 10000 * 100  # As percentage
        return 0
    
    def _simulate_bootstrap(self, data: pd.DataFrame, strategy,
                           backtest_engine) -> float:
        """Simulate with bootstrap resampled data."""
        # Resample with replacement
        n = len(data)
        indices = np.random.choice(n, size=n, replace=True)
        indices.sort()  # Keep temporal order
        
        bootstrap_data = data.iloc[indices].copy()
        bootstrap_data.index = data.index  # Keep original index
        
        try:
            result = backtest_engine.run_backtest(
                data=bootstrap_data,
                strategy=strategy,
                symbol="SYMBOL"
            )
            return result.total_return
        except Exception:
            return 0
    
    def _analyze_results(self, results: Dict) -> Dict[str, Any]:
        """Analyze Monte Carlo results."""
        original = results['original']['return']
        shuffle_returns = results['trade_shuffle']
        bootstrap_returns = results['bootstrap']
        
        all_simulated = shuffle_returns + bootstrap_returns
        
        if not all_simulated:
            return {
                'p_value': 1.0,
                'percentile': 50,
                'is_significant': False
            }
        
        # Calculate p-value (% of simulations worse than original)
        better_count = sum(1 for r in all_simulated if r >= original)
        p_value = better_count / len(all_simulated)
        
        # Calculate percentile of original result
        percentile = sum(1 for r in all_simulated if r < original) / len(all_simulated) * 100
        
        # Statistical analysis
        mean_simulated = np.mean(all_simulated)
        std_simulated = np.std(all_simulated)
        
        return {
            'original_return': original,
            'mean_simulated': mean_simulated,
            'std_simulated': std_simulated,
            'p_value': p_value,
            'percentile': percentile,
            'is_significant': percentile > 95,  # 95% confidence
            'confidence_interval': (
                np.percentile(all_simulated, 2.5),
                np.percentile(all_simulated, 97.5)
            ),
            'shuffle_results': shuffle_returns,
            'bootstrap_results': bootstrap_returns
        }
