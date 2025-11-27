# tests/test_suite.py
"""
Comprehensive test suite for the trading platform.

Tests run in lite mode by default (no PostgreSQL/Redis required).
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path

# Set lite mode for tests
os.environ['TRADING_LITE_MODE'] = '1'

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress logging during tests
logging.basicConfig(level=logging.WARNING)

# Import platform components
from core.platform import TradingPlatform, reset_platform
from strategies.strategy_engine import StrategyEngine, StrategyConfig, TradeSignal
from backtesting.backtest_engine import BacktestEngine, BacktestResult
from risk_management.risk_engine import RiskEngine, RiskConfig


class TestTradingPlatform(unittest.TestCase):
    """Test suite for the trading platform."""
    
    @classmethod
    def setUpClass(cls):
        """Setup for all tests."""
        # Reset any existing platform
        reset_platform()
        
        # Create test data
        cls.test_data = cls._create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests."""
        reset_platform()
    
    @staticmethod
    def _create_test_data(days: int = 100) -> pd.DataFrame:
        """Create realistic test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Generate prices with trend and volatility
        returns = np.random.normal(0.0005, 0.015, days)
        prices = 100 * (1 + returns).cumprod()
        
        # Generate OHLCV
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, days)),
            'high': prices * (1 + np.abs(np.random.normal(0.005, 0.005, days))),
            'low': prices * (1 - np.abs(np.random.normal(0.005, 0.005, days))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
        
        return data
    
    def test_01_platform_initialization(self):
        """Test platform initialization."""
        platform = TradingPlatform(lite_mode=True)
        result = platform.initialize()
        
        self.assertTrue(result)
        self.assertTrue(platform.initialized)
        
        platform.shutdown()
    
    def test_02_strategy_engine(self):
        """Test strategy engine."""
        strategy_engine = StrategyEngine()
        
        # Create test strategy
        config = StrategyConfig(
            name="TestStrategy",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        strategy = strategy_engine.create_strategy('ma_crossover', config)
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, "TestStrategy")
        
        # Test signal generation
        signals = strategy_engine.get_strategy_signals("TestStrategy", "TEST", self.test_data)
        self.assertIsInstance(signals, list)
    
    def test_03_backtesting_engine(self):
        """Test backtesting engine."""
        backtest_engine = BacktestEngine(initial_capital=10000)
        
        # Create strategy for backtest
        strategy_engine = StrategyEngine()
        config = StrategyConfig(
            name="BacktestStrategy",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 5, 'slow_period': 15}
        )
        strategy = strategy_engine.create_strategy('ma_crossover', config)
        
        # Run backtest
        result = backtest_engine.run_backtest(
            data=self.test_data,
            strategy=strategy,
            symbol="TEST",
            commission=0.001
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.total_return, float)
        self.assertIsInstance(result.total_trades, int)
        self.assertGreaterEqual(result.win_rate, 0)
        self.assertLessEqual(result.win_rate, 100)
    
    def test_04_risk_management(self):
        """Test risk management engine."""
        risk_engine = RiskEngine()
        
        # Test position sizing
        signal = TradeSignal(
            symbol="TEST",
            direction=1,
            strength=0.8,
            timestamp=datetime.now(),
            price=100.0
        )
        
        account_info = {'balance': 10000, 'equity': 10000}
        position_size = risk_engine.calculate_position_size(signal, 0.02, account_info)
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
    
    def test_05_strategy_signals(self):
        """Test strategy signal generation."""
        strategy_engine = StrategyEngine()
        
        # Test MA Crossover
        ma_config = StrategyConfig(
            name="MA_Test",
            symbols=["TEST"],
            timeframe="D1",
            parameters={
                'fast_period': 5,
                'slow_period': 20,
                'ma_type': 'sma',
                'rsi_period': 14
            }
        )
        
        strategy = strategy_engine.create_strategy('ma_crossover', ma_config)
        result = strategy.run("TEST", self.test_data)
        
        self.assertIn('signal', result.columns)
        self.assertIn('ma_fast', result.columns)
        self.assertIn('ma_slow', result.columns)
    
    def test_06_rsi_strategy(self):
        """Test RSI strategy."""
        strategy_engine = StrategyEngine()
        
        config = StrategyConfig(
            name="RSI_Test",
            symbols=["TEST"],
            timeframe="D1",
            parameters={
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        )
        
        strategy = strategy_engine.create_strategy('rsi', config)
        result = strategy.run("TEST", self.test_data)
        
        self.assertIn('signal', result.columns)
        self.assertIn('rsi', result.columns)
    
    def test_07_macd_strategy(self):
        """Test MACD strategy."""
        strategy_engine = StrategyEngine()
        
        config = StrategyConfig(
            name="MACD_Test",
            symbols=["TEST"],
            timeframe="D1",
            parameters={
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            }
        )
        
        strategy = strategy_engine.create_strategy('macd', config)
        result = strategy.run("TEST", self.test_data)
        
        self.assertIn('signal', result.columns)
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
    
    def test_08_backtest_metrics(self):
        """Test backtest metrics calculation."""
        backtest_engine = BacktestEngine(initial_capital=10000)
        
        strategy_engine = StrategyEngine()
        config = StrategyConfig(
            name="MetricsTest",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 30}
        )
        strategy = strategy_engine.create_strategy('ma_crossover', config)
        
        result = backtest_engine.run_backtest(
            data=self.test_data,
            strategy=strategy,
            symbol="TEST"
        )
        
        # Check all metrics are present
        self.assertIsInstance(result.total_return, float)
        self.assertIsInstance(result.max_drawdown, float)
        self.assertIsInstance(result.win_rate, float)
        
        # Check trades list
        self.assertIsInstance(result.trades, list)
    
    def test_09_risk_config(self):
        """Test risk configuration."""
        config = RiskConfig(
            max_drawdown=0.15,
            max_position_size=0.10,
            daily_loss_limit=0.05
        )
        
        risk_engine = RiskEngine(config)
        
        self.assertEqual(risk_engine.config.max_drawdown, 0.15)
        self.assertEqual(risk_engine.config.max_position_size, 0.10)
    
    def test_10_error_handling(self):
        """Test error handling."""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        strategy_engine = StrategyEngine()
        config = StrategyConfig(
            name="ErrorTest",
            symbols=["INVALID"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        try:
            strategy = strategy_engine.create_strategy('ma_crossover', config)
            # This should handle empty data gracefully
            result = strategy.run("INVALID", empty_data)
            self.assertTrue(True)
        except Exception as e:
            # Also acceptable to raise controlled exception
            self.assertIsInstance(e, (ValueError, KeyError, IndexError))
    
    def test_11_multiple_strategies(self):
        """Test running multiple strategies."""
        strategy_engine = StrategyEngine()
        
        # Create multiple strategies
        strategies = [
            ('ma_crossover', {'fast_period': 5, 'slow_period': 20}),
            ('rsi', {'rsi_period': 14}),
            ('macd', {'fast_period': 12, 'slow_period': 26})
        ]
        
        for strat_type, params in strategies:
            config = StrategyConfig(
                name=f"Test_{strat_type}",
                symbols=["TEST"],
                timeframe="D1",
                parameters=params
            )
            strategy_engine.create_strategy(strat_type, config)
        
        self.assertEqual(len(strategy_engine.strategies), 3)
        
        # Run all strategies
        data_dict = {"TEST": self.test_data}
        all_signals = strategy_engine.batch_run_strategies(data_dict)
        
        self.assertEqual(len(all_signals), 3)


class IntegrationTests(unittest.TestCase):
    """Integration tests for the platform."""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow."""
        # Reset platform
        reset_platform()
        
        # Initialize platform in lite mode
        platform = TradingPlatform(lite_mode=True)
        
        try:
            self.assertTrue(platform.initialize())
            
            # Create and test strategy
            strategy_engine = StrategyEngine()
            config = StrategyConfig(
                name="E2E_Test",
                symbols=["TEST"],
                timeframe="D1",
                parameters={'fast_period': 8, 'slow_period': 21}
            )
            strategy = strategy_engine.create_strategy('ma_crossover', config)
            
            # Create test data
            test_data = TestTradingPlatform._create_test_data(50)
            
            # Backtest
            backtest_engine = BacktestEngine()
            result = backtest_engine.run_backtest(
                data=test_data,
                strategy=strategy,
                symbol="TEST"
            )
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertIsInstance(result.total_trades, int)
            
        finally:
            platform.shutdown()
    
    def test_risk_integration(self):
        """Test risk management integration."""
        # Create components
        strategy_engine = StrategyEngine()
        risk_engine = RiskEngine()
        backtest_engine = BacktestEngine()
        
        # Create strategy
        config = StrategyConfig(
            name="RiskIntegration",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 25}
        )
        strategy = strategy_engine.create_strategy('ma_crossover', config)
        
        # Create test data
        test_data = TestTradingPlatform._create_test_data(100)
        
        # Get signals
        signals = strategy_engine.get_strategy_signals("RiskIntegration", "TEST", test_data)
        
        # Check risk for each signal
        account_info = {'balance': 10000, 'equity': 10000}
        for signal in signals[:5]:  # Check first 5 signals
            size = risk_engine.calculate_position_size(signal, 0.02, account_info)
            self.assertGreater(size, 0)


def run_all_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestTradingPlatform))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
