#!/usr/bin/env python3
# demo_autogen.py
"""
Demo Script for Auto Strategy Generation.

Demonstrates the automatic strategy generation capabilities:
1. Random strategy generation
2. Genetic algorithm optimization
3. Walk-forward validation
4. Overfitting detection
5. MQL5 export

Usage:
    python demo_autogen.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(symbol: str = "EURUSD", days: int = 365):
    """Generate sample OHLCV data for testing."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate price data
    n = len(dates)
    
    # Random walk for close prices
    returns = np.random.normal(0, 0.0005, n)
    close = 1.1000 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.001, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, n)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    # Generate volume
    volume = np.random.randint(100, 10000, n)
    
    df = pd.DataFrame({
        'time': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.set_index('time', inplace=True)
    
    return df


def demo_random_generation():
    """Demo: Generate random strategies."""
    print("\n" + "="*60)
    print("DEMO 1: Random Strategy Generation")
    print("="*60)
    
    from strategies.auto_generator import AutoStrategyGenerator
    
    generator = AutoStrategyGenerator()
    
    # Generate 5 random strategies
    strategies = generator.generate_batch(
        count=5,
        symbols=['EURUSD'],
        timeframe='H1'
    )
    
    print(f"\nGenerated {len(strategies)} strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy.name}")
        print(f"   Indicators: {len(strategy.indicators)}")
        for ind in strategy.indicators:
            print(f"   - {ind.name}: {ind.parameters}")
        print(f"   Entry rules: {len(strategy.entry_rules)}")
        print(f"   Risk management: {strategy.risk_management}")
    
    return strategies


def demo_genetic_optimization(data):
    """Demo: Genetic algorithm optimization."""
    print("\n" + "="*60)
    print("DEMO 2: Genetic Algorithm Optimization")
    print("="*60)
    
    from strategies.genetic_strategy import GeneticStrategyOptimizer, GeneticConfig
    from backtesting.backtest_engine import BacktestEngine
    
    # Configure genetic algorithm
    config = GeneticConfig(
        population_size=20,
        generations=10,  # Small for demo
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_count=2,
        tournament_size=3,
        min_trades=5
    )
    
    optimizer = GeneticStrategyOptimizer(config)
    backtest_engine = BacktestEngine(initial_capital=10000)
    
    print("\nRunning genetic optimization...")
    print("(This may take a few minutes)")
    
    def progress_callback(generation, stats):
        print(f"  Gen {generation + 1}: Best={stats['best_fitness']:.4f}, "
              f"Avg={stats['avg_fitness']:.4f}, Return={stats['best_return']:.2f}%")
    
    best_strategy = optimizer.run(
        data=data,
        backtest_engine=backtest_engine,
        symbols=['EURUSD'],
        timeframe='H1',
        callback=progress_callback
    )
    
    if best_strategy:
        print(f"\n✅ Best strategy found: {best_strategy.name}")
        print(f"   Fitness score: {best_strategy.fitness_score:.4f}")
        print(f"   Indicators: {[i.name for i in best_strategy.indicators]}")
    
    # Get top strategies
    top_strategies = optimizer.get_top_strategies(5)
    print(f"\nTop 5 strategies:")
    for i, strat in enumerate(top_strategies, 1):
        print(f"  {i}. {strat.name} - Fitness: {strat.fitness_score:.4f}")
    
    return best_strategy


def demo_walk_forward(data, strategy):
    """Demo: Walk-forward analysis."""
    print("\n" + "="*60)
    print("DEMO 3: Walk-Forward Analysis")
    print("="*60)
    
    from optimization.walk_forward import WalkForwardAnalyzer, WalkForwardMethod
    from backtesting.backtest_engine import BacktestEngine
    
    analyzer = WalkForwardAnalyzer({
        'num_windows': 4,
        'train_ratio': 0.7,
        'method': 'rolling'
    })
    
    # Create windows
    windows = analyzer.create_windows(data)
    
    print(f"\nCreated {len(windows)} walk-forward windows:")
    for w in windows:
        print(f"  Window {w.window_id}:")
        print(f"    Train: {w.train_start.date()} to {w.train_end.date()} ({w.train_size} bars)")
        print(f"    Test: {w.test_start.date()} to {w.test_end.date()} ({w.test_size} bars)")
    
    # Note: Full walk-forward requires strategy class, simplified demo
    print("\n⚠️ Full walk-forward analysis requires strategy class implementation")
    print("   Use the GUI for complete walk-forward testing")
    
    return windows


def demo_overfitting_detection(strategy, data):
    """Demo: Overfitting detection."""
    print("\n" + "="*60)
    print("DEMO 4: Overfitting Detection")
    print("="*60)
    
    from optimization.overfitting_detector import OverfittingDetector
    
    detector = OverfittingDetector()
    
    # Create mock backtest results for demo
    class MockResult:
        def __init__(self):
            self.total_return = 25.5
            self.sharpe_ratio = 1.8
            self.max_drawdown = 12.3
            self.win_rate = 58.0
            self.profit_factor = 1.65
            self.total_trades = 150
    
    in_sample = MockResult()
    out_of_sample = MockResult()
    out_of_sample.total_return = 15.2  # Lower OOS return
    out_of_sample.sharpe_ratio = 1.2
    
    report = detector.analyze(
        strategy=strategy,
        in_sample_result=in_sample,
        out_of_sample_result=out_of_sample,
        data_points=len(data)
    )
    
    print(f"\n{report.summary}")
    
    print("\nIndicators checked:")
    for indicator in report.indicators:
        status = "⚠️" if indicator.is_warning else "✅"
        print(f"  {status} {indicator.name}: {indicator.description}")
    
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  {rec}")
    
    return report


def demo_mql5_export(strategy, output_dir: str = "exported_strategies"):
    """Demo: Export to MQL5."""
    print("\n" + "="*60)
    print("DEMO 5: MQL5 Export")
    print("="*60)
    
    from strategies.mql5_exporter import MQL5Exporter
    
    exporter = MQL5Exporter()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Export strategy
    output_path = f"{output_dir}/{strategy.name}.mq5"
    code = exporter.export(strategy, output_path)
    
    print(f"\n✅ Strategy exported to: {output_path}")
    print(f"   Code length: {len(code)} characters")
    
    # Show preview
    print("\nCode preview (first 50 lines):")
    print("-" * 40)
    lines = code.split('\n')[:50]
    for line in lines:
        print(line)
    print("...")
    print("-" * 40)
    
    return output_path


def demo_indicator_library():
    """Demo: Available indicators."""
    print("\n" + "="*60)
    print("BONUS: Available Indicators Library")
    print("="*60)
    
    from strategies.auto_generator import IndicatorLibrary, IndicatorType
    
    library = IndicatorLibrary()
    
    for ind_type in IndicatorType:
        indicators = library.get_by_type(ind_type)
        if indicators:
            print(f"\n{ind_type.value.upper()} Indicators ({len(indicators)}):")
            for ind in indicators:
                params = ", ".join([f"{k}" for k in ind.parameters.keys()])
                print(f"  • {ind.name}({params})")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("   AUTO STRATEGY GENERATION - DEMO")
    print("="*60)
    
    # Check dependencies
    missing = []
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
    
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install pandas numpy")
        return 1
    
    # Generate sample data
    print("\nGenerating sample data...")
    data = generate_sample_data(days=180)
    print(f"✅ Generated {len(data)} hourly candles")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   Price range: {data['close'].min():.5f} to {data['close'].max():.5f}")
    
    # Run demos
    try:
        # Demo 1: Random generation
        strategies = demo_random_generation()
        
        # Demo 2: Genetic optimization (if we have data)
        if len(data) > 0:
            best_strategy = demo_genetic_optimization(data)
        else:
            best_strategy = strategies[0] if strategies else None
        
        # Use the best strategy for remaining demos
        if best_strategy is None and strategies:
            best_strategy = strategies[0]
        
        if best_strategy:
            # Demo 3: Walk-forward
            demo_walk_forward(data, best_strategy)
            
            # Demo 4: Overfitting detection
            demo_overfitting_detection(best_strategy, data)
            
            # Demo 5: MQL5 export
            demo_mql5_export(best_strategy)
        
        # Bonus: Show indicator library
        demo_indicator_library()
        
        print("\n" + "="*60)
        print("   DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run the GUI: python run_gui.py")
        print("  2. Load real data from MT5")
        print("  3. Use the Auto Generator tab for full functionality")
        print("  4. Export best strategies to MQL5 and test in MT5")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
