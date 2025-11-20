# tests/test_suite.py
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.platform import TradingPlatform
from strategies.strategy_engine import StrategyEngine, StrategyConfig
from backtesting.backtest_engine import BacktestEngine
from ml.ml_engine import MLEngine, MLModelConfig
from optimization.genetic_optimizer import GeneticOptimizer, OptimizationConfig
from execution.live_execution import LiveExecutionEngine, LiveTradingConfig
from risk_management.risk_engine import RiskEngine

class TestTradingPlatform(unittest.TestCase):
    """Suite de pruebas para la plataforma de trading"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todas las pruebas"""
        logging.basicConfig(level=logging.WARNING)  # Reducir log para pruebas
        cls.platform = TradingPlatform()
        cls.platform.initialize()
        
        # Datos de prueba
        cls.test_data = cls._create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza después de todas las pruebas"""
        if cls.platform.initialized:
            cls.platform.shutdown()
    
    @staticmethod
    def _create_test_data(days: int = 100) -> pd.DataFrame:
        """Crear datos de prueba realistas"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Generar precios con tendencia y volatilidad realistas
        returns = np.random.normal(0.0005, 0.015, days)
        prices = 100 * (1 + returns).cumprod()
        
        # Generar OHLCV
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, days)),
            'high': prices * (1 + np.abs(np.random.normal(0.005, 0.005, days))),
            'low': prices * (1 - np.abs(np.random.normal(0.005, 0.005, days))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
        
        return data
    
    def test_01_platform_initialization(self):
        """Prueba de inicialización de la plataforma"""
        self.assertTrue(self.platform.initialized)
        self.assertIsNotNone(self.platform.mt5_connector)
        self.assertIsNotNone(self.platform.data_manager)
    
    def test_02_data_management(self):
        """Prueba de gestión de datos"""
        # Probar almacenamiento y recuperación
        symbol, timeframe = "TEST", "D1"
        self.platform.data_manager.store_market_data(symbol, timeframe, self.test_data)
        
        # Probar cache
        cached_data = self.platform.data_manager.get_cached_data(symbol, timeframe)
        self.assertIsNotNone(cached_data)
        self.assertEqual(len(cached_data), len(self.test_data))
    
    def test_03_strategy_engine(self):
        """Prueba del motor de estrategias"""
        strategy_engine = StrategyEngine()
        
        # Crear estrategia de prueba
        config = StrategyConfig(
            name="TestStrategy",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        strategy = strategy_engine.create_strategy('ma_crossover', config)
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, "TestStrategy")
        
        # Probar generación de señales
        signals = strategy_engine.get_strategy_signals("TestStrategy", "TEST", self.test_data)
        self.assertIsInstance(signals, list)
    
    def test_04_backtesting_engine(self):
        """Prueba del motor de backtesting"""
        backtest_engine = BacktestEngine(initial_capital=10000)
        
        # Crear estrategia para backtest
        strategy_engine = StrategyEngine()
        config = StrategyConfig(
            name="BacktestStrategy",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 5, 'slow_period': 15}
        )
        strategy = strategy_engine.create_strategy('ma_crossover', config)
        
        # Ejecutar backtest
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
    
    def test_05_machine_learning_engine(self):
        """Prueba del motor de Machine Learning"""
        ml_engine = MLEngine()
        
        # Configurar modelo de ML
        ml_config = MLModelConfig(
            model_type='classification',
            algorithm='random_forest',
            features=[],
            target='price_direction',
            parameters={'n_estimators': 50, 'max_depth': 5}  # Reducido para pruebas
        )
        
        # Entrenar modelo
        result = ml_engine.train_model(self.test_data, ml_config)
        
        self.assertIsNotNone(result)
        self.assertIn('accuracy', result.metrics)
        self.assertGreaterEqual(result.metrics['accuracy'], 0)
        self.assertLessEqual(result.metrics['accuracy'], 1)
    
    def test_06_genetic_optimization(self):
        """Prueba de optimización genética"""
        # Crear estrategia para optimizar
        strategy_engine = StrategyEngine()
        config = StrategyConfig(
            name="OptimizationTest",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        strategy = strategy_engine.create_strategy('ma_crossover', config)
        
        # Configurar optimización
        backtest_engine = BacktestEngine()
        genetic_optimizer = GeneticOptimizer(backtest_engine)
        
        opt_config = OptimizationConfig(
            strategy_name="OptimizationTest",
            parameter_ranges={
                'fast_period_int': (5, 15),
                'slow_period_int': (15, 30)
            },
            objective='sharpe',
            population_size=10,  # Reducido para pruebas
            generations=5       # Reducido para pruebas
        )
        
        # Ejecutar optimización
        result = genetic_optimizer.optimize_strategy(strategy, self.test_data, opt_config)
        
        self.assertIsNotNone(result)
        self.assertIn('best_parameters', result)
        self.assertIn('best_fitness', result)
    
    def test_07_risk_management(self):
        """Prueba del motor de gestión de riesgo"""
        risk_engine = RiskEngine()
        
        # Probar cálculo de posición
        from strategies.strategy_engine import TradeSignal
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
    
    def test_08_live_execution(self):
        """Prueba del motor de ejecución en vivo"""
        live_engine = LiveExecutionEngine()
        
        # Configurar estrategia para trading en vivo
        live_config = LiveTradingConfig(
            strategy_name="LiveTest",
            symbols=["TEST"],
            timeframe="D1",
            enabled=False,  # Deshabilitado para pruebas
            max_positions=1,
            risk_per_trade=0.01
        )
        
        # En pruebas reales, aquí se agregaría una estrategia real
        # Por ahora probamos solo la configuración
        self.assertIsInstance(live_config, LiveTradingConfig)
    
    def test_09_performance_optimization(self):
        """Prueba de optimizaciones de performance"""
        from core.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        
        # Probar cálculo vectorizado
        prices = np.array([100, 101, 102, 101, 103, 104], dtype=np.float64)
        returns = optimizer.vectorized_returns_calculation(prices)
        
        self.assertIsInstance(returns, np.ndarray)
        self.assertEqual(len(returns), len(prices))
        
        # Probar reporte de memoria
        memory_report = optimizer.memory_usage_report()
        self.assertIn('rss_mb', memory_report)
        self.assertIsInstance(memory_report['rss_mb'], float)
    
    def test_10_error_handling(self):
        """Prueba de manejo de errores"""
        # Probar con datos inválidos
        invalid_data = pd.DataFrame()
        
        strategy_engine = StrategyEngine()
        config = StrategyConfig(
            name="ErrorTest",
            symbols=["INVALID"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        # Esto debería manejar el error gracefulmente
        try:
            strategy = strategy_engine.create_strategy('ma_crossover', config)
            signals = strategy_engine.get_strategy_signals("ErrorTest", "INVALID", invalid_data)
            # Si llegamos aquí, el manejo de errores funciona
            self.assertTrue(True)
        except Exception as e:
            # También es aceptable que lance una excepción controlada
            self.assertIsInstance(e, (ValueError, KeyError))

class IntegrationTests(unittest.TestCase):
    """Pruebas de integración de componentes"""
    
    def test_end_to_end_workflow(self):
        """Prueba de flujo de trabajo completo"""
        platform = TradingPlatform()
        
        try:
            # Inicializar plataforma
            self.assertTrue(platform.initialize())
            
            # Crear y probar estrategia
            strategy_engine = StrategyEngine()
            config = StrategyConfig(
                name="E2E_Test",
                symbols=["TEST"],
                timeframe="D1",
                parameters={'fast_period': 8, 'slow_period': 21}
            )
            strategy = strategy_engine.create_strategy('ma_crossover', config)
            
            # Backtest
            backtest_engine = BacktestEngine()
            result = backtest_engine.run_backtest(
                data=TestTradingPlatform._create_test_data(50),
                strategy=strategy,
                symbol="TEST"
            )
            
            # Verificar resultados básicos
            self.assertIsNotNone(result)
            self.assertIsInstance(result.total_trades, int)
            
        finally:
            platform.shutdown()

def run_all_tests():
    """Ejecutar todas las pruebas"""
    # Crear test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestTradingPlatform))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)