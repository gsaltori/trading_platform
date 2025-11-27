#!/usr/bin/env python3
# quick_check.py
"""Quick check script to verify the trading platform is working."""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("Trading Platform - Quick Check")
    print("=" * 60)
    
    # Check imports
    print("\n📦 Checking imports...")
    
    modules = [
        ("core.platform", "TradingPlatform"),
        ("strategies.strategy_engine", "StrategyEngine"),
        ("backtesting.backtest_engine", "BacktestEngine"),
        ("risk_management.risk_engine", "RiskEngine"),
        ("optimization.genetic_optimizer", "GeneticOptimizer"),
        ("execution.live_execution", "LiveExecutionEngine"),
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some imports failed. Check the error messages above.")
        return 1
    
    # Quick functional test
    print("\n🧪 Running quick functional test...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = 100 * (1 + np.random.normal(0, 0.02, 100)).cumprod()
        
        data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test strategy
        from strategies.strategy_engine import StrategyEngine, StrategyConfig
        
        engine = StrategyEngine()
        config = StrategyConfig(
            name="QuickTest",
            symbols=["TEST"],
            timeframe="D1",
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        strategy = engine.create_strategy('ma_crossover', config)
        signals = engine.get_strategy_signals("QuickTest", "TEST", data)
        
        print(f"  ✅ Strategy created and generated {len(signals)} signals")
        
        # Test backtest
        from backtesting.backtest_engine import BacktestEngine
        
        bt = BacktestEngine(initial_capital=10000)
        result = bt.run_backtest(data, strategy, "TEST")
        
        print(f"  ✅ Backtest completed: {result.total_return:.2f}% return, {result.total_trades} trades")
        
        # Test risk engine
        from risk_management.risk_engine import RiskEngine
        from strategies.strategy_engine import TradeSignal
        from datetime import datetime
        
        risk = RiskEngine()
        signal = TradeSignal(
            symbol="TEST",
            direction=1,
            strength=0.8,
            timestamp=datetime.now(),
            price=100.0
        )
        
        size = risk.calculate_position_size(signal, 0.02, {'balance': 10000})
        print(f"  ✅ Risk engine calculated position size: {size:.4f}")
        
    except Exception as e:
        print(f"  ❌ Functional test failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Platform is ready to use.")
    print("=" * 60)
    print("\nUsage:")
    print("  python main.py --lite --test    # Run tests")
    print("  python main.py --health-check   # Check health")
    print("  python main.py --headless       # Run headless")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
