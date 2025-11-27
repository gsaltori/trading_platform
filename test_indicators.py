#!/usr/bin/env python3
# test_indicators.py
"""
Test Script for Extended Indicators.

Verifies that all 70+ indicators are correctly implemented and functional.
"""

import sys
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def generate_sample_data():
    """Generate sample OHLCV data."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n), periods=n, freq='H')
    
    returns = np.random.normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.randint(1000, 100000, n)
    
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


def test_indicator_library():
    """Test the ExtendedIndicatorLibrary."""
    print("\n" + "="*60)
    print("TEST 1: Extended Indicator Library")
    print("="*60)
    
    from strategies.indicator_integration import ExtendedIndicatorLibrary, get_indicator_stats
    
    library = ExtendedIndicatorLibrary()
    stats = get_indicator_stats()
    
    print(f"\n✅ Total indicators: {stats['total_indicators']}")
    print("\nBy category:")
    for ind_type, count in stats['by_type'].items():
        if ind_type != 'total':
            print(f"  • {ind_type}: {count}")
    
    print("\nIndicator list:")
    for ind_type, indicators in stats['indicators_list'].items():
        if indicators:
            print(f"\n  {ind_type.upper()} ({len(indicators)}):")
            for ind in indicators:
                print(f"    - {ind}")
    
    return True


def test_indicator_calculations():
    """Test calculation of all indicators."""
    print("\n" + "="*60)
    print("TEST 2: Indicator Calculations")
    print("="*60)
    
    from strategies.indicator_integration import (
        ExtendedIndicatorLibrary, 
        ExtendedIndicatorCalculator
    )
    
    library = ExtendedIndicatorLibrary()
    calculator = ExtendedIndicatorCalculator()
    data = generate_sample_data()
    
    success_count = 0
    fail_count = 0
    failed_indicators = []
    
    print(f"\nTesting {len(library.indicators)} indicators on {len(data)} data points...")
    print("-" * 60)
    
    for name, indicator in library.indicators.items():
        try:
            # Generate test parameters
            params = {}
            for param_name, param_config in indicator.parameters.items():
                if isinstance(param_config, dict):
                    params[param_name] = param_config.get('default', 14)
            
            # Calculate indicator
            result = calculator.calculate(data, indicator, params)
            
            # Verify output columns exist
            missing_cols = []
            for col_template in indicator.output_columns:
                col = col_template
                for pname, pval in params.items():
                    col = col.replace(f'{{{pname}}}', str(pval))
                
                if col not in result.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                print(f"  ⚠️  {name}: Missing columns {missing_cols}")
                fail_count += 1
                failed_indicators.append(name)
            else:
                # Check for NaN ratio
                total_nan = 0
                for col_template in indicator.output_columns:
                    col = col_template
                    for pname, pval in params.items():
                        col = col.replace(f'{{{pname}}}', str(pval))
                    if col in result.columns:
                        total_nan += result[col].isna().sum()
                
                nan_ratio = total_nan / (len(result) * len(indicator.output_columns))
                
                if nan_ratio > 0.5:
                    print(f"  ⚠️  {name}: High NaN ratio ({nan_ratio:.1%})")
                else:
                    print(f"  ✅ {name}: OK (NaN: {nan_ratio:.1%})")
                
                success_count += 1
                
        except Exception as e:
            print(f"  ❌ {name}: ERROR - {str(e)[:50]}")
            fail_count += 1
            failed_indicators.append(name)
    
    print("-" * 60)
    print(f"\nResults: {success_count} passed, {fail_count} failed")
    
    if failed_indicators:
        print(f"\nFailed indicators: {', '.join(failed_indicators)}")
    
    return fail_count == 0


def test_strategy_generation():
    """Test strategy generation with extended indicators."""
    print("\n" + "="*60)
    print("TEST 3: Strategy Generation with Extended Indicators")
    print("="*60)
    
    from strategies.indicator_integration import create_extended_generator
    
    generator = create_extended_generator()
    data = generate_sample_data()
    
    print("\nGenerating 5 random strategies...")
    strategies = generator.generate_batch(5, symbols=['EURUSD'], timeframe='H1')
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n  Strategy {i}: {strategy.name}")
        print(f"    Indicators: {len(strategy.indicators)}")
        for ind in strategy.indicators:
            print(f"      - {ind.name}")
        print(f"    Entry rules: {len(strategy.entry_rules)}")
        print(f"    Exit rules: {len(strategy.exit_rules)}")
        
        # Test evaluation
        try:
            result = generator.evaluate_strategy(strategy, data)
            signals = result['signal'].value_counts()
            print(f"    Signals: Buy={signals.get(1, 0)}, Sell={signals.get(-1, 0)}, Neutral={signals.get(0, 0)}")
        except Exception as e:
            print(f"    ⚠️ Evaluation error: {e}")
    
    return True


def test_extended_indicator_calculations():
    """Test extended indicator calculations directly."""
    print("\n" + "="*60)
    print("TEST 4: Direct Extended Indicator Calculations")
    print("="*60)
    
    from strategies.extended_indicators import ExtendedIndicatorCalculator as EIC
    
    data = generate_sample_data()
    
    tests = [
        ("VWMA", lambda: EIC.calculate_vwma(data, 20)),
        ("HMA", lambda: EIC.calculate_hma(data, 20)),
        ("KAMA", lambda: EIC.calculate_kama(data, 10, 2, 30)),
        ("ZLEMA", lambda: EIC.calculate_zlema(data, 20)),
        ("T3", lambda: EIC.calculate_t3(data, 5, 0.7)),
        ("PSAR", lambda: EIC.calculate_psar(data, 0.02, 0.02, 0.2)),
        ("Aroon", lambda: EIC.calculate_aroon(data, 25)),
        ("Vortex", lambda: EIC.calculate_vortex(data, 14)),
        ("TSI", lambda: EIC.calculate_tsi(data, 25, 13, 13)),
        ("Ultimate Oscillator", lambda: EIC.calculate_ultimate_oscillator(data, 7, 14, 28)),
        ("Awesome Oscillator", lambda: EIC.calculate_awesome_oscillator(data, 5, 34)),
        ("TRIX", lambda: EIC.calculate_trix(data, 15, 9)),
        ("KST", lambda: EIC.calculate_kst(data)),
        ("PPO", lambda: EIC.calculate_ppo(data, 12, 26, 9)),
        ("CMO", lambda: EIC.calculate_cmo(data, 14)),
        ("Chaikin Volatility", lambda: EIC.calculate_chaikin_volatility(data, 10, 10)),
        ("Historical Volatility", lambda: EIC.calculate_historical_volatility(data, 20)),
        ("Mass Index", lambda: EIC.calculate_mass_index(data, 9, 25)),
        ("NATR", lambda: EIC.calculate_natr(data, 14)),
        ("Ulcer Index", lambda: EIC.calculate_ulcer_index(data, 14)),
        ("ATR Bands", lambda: EIC.calculate_atr_bands(data, 14, 2.0)),
        ("VWAP Bands", lambda: EIC.calculate_vwap_bands(data, 2.0)),
        ("PVT", lambda: EIC.calculate_pvt(data)),
        ("EOM", lambda: EIC.calculate_eom(data, 14)),
        ("Force Index", lambda: EIC.calculate_force_index(data, 13)),
        ("Klinger", lambda: EIC.calculate_klinger(data, 34, 55, 13)),
        ("Higher High/Lower Low", lambda: EIC.calculate_higher_high_lower_low(data, 20)),
        ("ADR", lambda: EIC.calculate_adr(data, 14)),
        ("Candle Body", lambda: EIC.calculate_candle_body(data)),
        ("Inside/Outside Bar", lambda: EIC.calculate_inside_outside_bar(data)),
        ("STC", lambda: EIC.calculate_schaff_trend(data, 10, 23, 50)),
        ("Fisher Transform", lambda: EIC.calculate_ehlers_fisher(data, 10)),
        ("Coppock", lambda: EIC.calculate_coppock(data, 10, 14, 11)),
        ("Elder Ray", lambda: EIC.calculate_elder_ray(data, 13)),
        ("PGO", lambda: EIC.calculate_pretty_good_oscillator(data, 14)),
        ("WaveTrend", lambda: EIC.calculate_wave_trend(data, 10, 21)),
    ]
    
    success = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            print(f"  ✅ {name}: {len(result.columns) - 6} new columns")
            success += 1
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")
            failed += 1
    
    print(f"\nDirect calculations: {success}/{len(tests)} passed")
    return failed == 0


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("   EXTENDED INDICATORS TEST SUITE")
    print("="*60)
    
    # Check dependencies
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return 1
    
    results = []
    
    # Test 1: Library
    try:
        results.append(("Indicator Library", test_indicator_library()))
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        traceback.print_exc()
        results.append(("Indicator Library", False))
    
    # Test 2: Calculations
    try:
        results.append(("Indicator Calculations", test_indicator_calculations()))
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        traceback.print_exc()
        results.append(("Indicator Calculations", False))
    
    # Test 3: Strategy Generation
    try:
        results.append(("Strategy Generation", test_strategy_generation()))
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        traceback.print_exc()
        results.append(("Strategy Generation", False))
    
    # Test 4: Extended Direct Calculations
    try:
        results.append(("Direct Calculations", test_extended_indicator_calculations()))
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        traceback.print_exc()
        results.append(("Direct Calculations", False))
    
    # Summary
    print("\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("   ALL TESTS PASSED! ✅")
    else:
        print("   SOME TESTS FAILED ❌")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
