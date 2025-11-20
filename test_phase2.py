# test_phase2.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from core.platform import TradingPlatform
from strategies.strategy_engine import StrategyEngine, StrategyConfig, MovingAverageCrossover, RSIStrategy
from backtesting.backtest_engine import BacktestEngine

def test_phase2():
    """Prueba completa de los componentes de la Fase 2"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🧪 Probando Fase 2: Interfaz Gráfica y Motor de Estrategias...")
    
    # Crear plataforma
    with TradingPlatform() as platform:
        if not platform.initialized:
            print("❌ Error inicializando la plataforma")
            return
        
        print("✅ Plataforma inicializada correctamente")
        
        # Inicializar motor de estrategias
        strategy_engine = StrategyEngine()
        backtest_engine = BacktestEngine()
        
        # Crear configuraciones de estrategias
        ma_config = StrategyConfig(
            name="MA Crossover Avanzado",
            symbols=["EURUSD", "GBPUSD"],
            timeframe="H1",
            parameters={
                'fast_period': 10,
                'slow_period': 20,
                'ma_type': 'sma',
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'min_ma_diff': 0.001
            },
            risk_management={
                'atr_multiplier': 2.0,
                'risk_reward_ratio': 1.5
            }
        )
        
        rsi_config = StrategyConfig(
            name="RSI con Divergencias",
            symbols=["EURUSD", "USDJPY"],
            timeframe="H1",
            parameters={
                'rsi_period': 14,
                'ma_period': 21,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'use_divergences': True
            }
        )
        
        # Crear estrategias
        ma_strategy = strategy_engine.create_strategy('ma_crossover', ma_config)
        rsi_strategy = strategy_engine.create_strategy('rsi', rsi_config)
        
        print(f"✅ Estrategias creadas: {list(strategy_engine.strategies.keys())}")
        
        # Obtener datos de prueba
        print("\n📊 Obteniendo datos para backtesting...")
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        data_dict = {}
        
        for symbol in symbols:
            data = platform.get_market_data(symbol, "H1", days=90)
            if data is not None:
                data_dict[symbol] = data
                print(f"   {symbol}: {len(data)} velas")
        
        if data_dict:
            print("\n📈 Ejecutando backtests avanzados...")
            
            # Probar MA Crossover
            print("\n🔧 Probando MA Crossover...")
            for symbol in ma_config.symbols:
                if symbol in data_dict:
                    result = backtest_engine.run_backtest(
                        data=data_dict[symbol],
                        strategy=ma_strategy,
                        symbol=symbol,
                        commission=0.001,
                        slippage_model="dynamic",
                        position_sizing="risk_based",
                        risk_per_trade=0.02
                    )
                    
                    print(f"📊 {symbol} - Resultados MA Crossover:")
                    print(f"   Retorno Total: {result.total_return:.2f}%")
                    print(f"   Total Trades: {result.total_trades}")
                    print(f"   Win Rate: {result.win_rate:.1f}%")
                    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "   Sharpe Ratio: N/A")
                    print(f"   Drawdown Máximo: {result.max_drawdown:.2f}%")
                    print(f"   Profit Factor: {result.profit_factor:.2f}" if result.profit_factor else "   Profit Factor: N/A")
            
            # Probar RSI con Divergencias
            print("\n🔧 Probando RSI con Divergencias...")
            for symbol in rsi_config.symbols:
                if symbol in data_dict:
                    result = backtest_engine.run_backtest(
                        data=data_dict[symbol],
                        strategy=rsi_strategy,
                        symbol=symbol,
                        commission=0.001,
                        slippage_model="dynamic",
                        position_sizing="risk_based",
                        risk_per_trade=0.02
                    )
                    
                    print(f"📊 {symbol} - Resultados RSI:")
                    print(f"   Retorno Total: {result.total_return:.2f}%")
                    print(f"   Total Trades: {result.total_trades}")
                    print(f"   Win Rate: {result.win_rate:.1f}%")
                    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "   Sharpe Ratio: N/A")
            
            # Probar optimización de parámetros
            print("\n🔧 Probando optimización de parámetros...")
            eurusd_data = data_dict.get("EURUSD")
            if eurusd_data is not None:
                parameter_ranges = {
                    'fast_period': [5, 10, 15],
                    'slow_period': [20, 25, 30],
                    'rsi_period': [10, 14, 18]
                }
                
                best_params = strategy_engine.optimize_parameters(
                    'ma_crossover', eurusd_data, parameter_ranges, 'sharpe_ratio'
                )
                
                print(f"✅ Mejores parámetros encontrados: {best_params}")
        
        else:
            print("❌ Error obteniendo datos para backtesting")
        
        print("\n🎉 Prueba de Fase 2 completada exitosamente!")

if __name__ == "__main__":
    # Para ejecutar la interfaz gráfica:
    # from ui.main_window import run_gui
    # run_gui()
    
    # Para ejecutar las pruebas:
    test_phase2()