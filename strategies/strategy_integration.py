# strategies/strategy_integration.py
"""
Módulo de integración que extiende el StrategyEngine con todas las nuevas estrategias
Importar esto después del StrategyEngine original
"""

def extend_strategy_engine(strategy_engine):
    """
    Extender el StrategyEngine con todas las estrategias nuevas
    
    Args:
        strategy_engine: Instancia del StrategyEngine a extender
    """
    
    # Importar estrategias nuevas
    try:
        from strategies.bollinger_strategy import BollingerBandsStrategy
        strategy_engine.available_strategies['bollinger'] = BollingerBandsStrategy
        print("✅ Bollinger Bands cargado")
    except ImportError as e:
        print(f"⚠️  No se pudo cargar Bollinger Bands: {e}")
    
    try:
        from strategies.stochastic_strategy import StochasticStrategy
        strategy_engine.available_strategies['stochastic'] = StochasticStrategy
        print("✅ Stochastic cargado")
    except ImportError as e:
        print(f"⚠️  No se pudo cargar Stochastic: {e}")
    
    try:
        from strategies.adx_strategy import ADXStrategy
        strategy_engine.available_strategies['adx'] = ADXStrategy
        print("✅ ADX cargado")
    except ImportError as e:
        print(f"⚠️  No se pudo cargar ADX: {e}")
    
    try:
        from strategies.cci_strategy import CCIStrategy
        strategy_engine.available_strategies['cci'] = CCIStrategy
        print("✅ CCI cargado")
    except ImportError as e:
        print(f"⚠️  No se pudo cargar CCI: {e}")
    
    try:
        from strategies.ichimoku_strategy import IchimokuStrategy
        strategy_engine.available_strategies['ichimoku'] = IchimokuStrategy
        print("✅ Ichimoku cargado")
    except ImportError as e:
        print(f"⚠️  No se pudo cargar Ichimoku: {e}")
    
    try:
        from strategies.psar_strategy import ParabolicSARStrategy
        strategy_engine.available_strategies['psar'] = ParabolicSARStrategy
        print("✅ Parabolic SAR cargado")
    except ImportError as e:
        print(f"⚠️  No se pudo cargar Parabolic SAR: {e}")
    
    # Híbridas
    try:
        from strategies.hybrid_strategies import (
            MARSIMACDStrategy, 
            BBRSIStrategy, 
            ADXMARSIStrategy
        )
        strategy_engine.available_strategies['ma_rsi_macd'] = MARSIMACDStrategy
        strategy_engine.available_strategies['bb_rsi'] = BBRSIStrategy
        strategy_engine.available_strategies['adx_ma_rsi'] = ADXMARSIStrategy
        print("✅ Estrategias híbridas cargadas")
    except ImportError as e:
        print(f"⚠️  No se pudieron cargar híbridas: {e}")
    
    return strategy_engine


# Biblioteca completa de estrategias expandida
COMPLETE_STRATEGY_LIBRARY = {
    # ============================================
    # MOVING AVERAGE STRATEGIES (10 variantes)
    # ============================================
    'ma_crossover': [
        {'fast_period': 20, 'slow_period': 50, 'rsi_period': 14, 'ma_type': 'ema', 
         'min_ma_diff': 0.002, 'rsi_oversold': 25, 'rsi_overbought': 75, 
         'name': 'MA_20_50_EMA'},
        {'fast_period': 50, 'slow_period': 200, 'rsi_period': 14, 'ma_type': 'sma', 
         'min_ma_diff': 0.003, 'rsi_oversold': 20, 'rsi_overbought': 80, 
         'name': 'MA_GoldenCross'},
        {'fast_period': 30, 'slow_period': 100, 'rsi_period': 14, 'ma_type': 'ema', 
         'min_ma_diff': 0.0025, 'rsi_oversold': 25, 'rsi_overbought': 75, 
         'name': 'MA_30_100_EMA'},
        {'fast_period': 15, 'slow_period': 40, 'rsi_period': 14, 'ma_type': 'ema', 
         'min_ma_diff': 0.0015, 'rsi_oversold': 30, 'rsi_overbought': 70, 
         'name': 'MA_15_40_EMA'},
        {'fast_period': 25, 'slow_period': 75, 'rsi_period': 14, 'ma_type': 'sma', 
         'min_ma_diff': 0.002, 'rsi_oversold': 25, 'rsi_overbought': 75, 
         'name': 'MA_25_75_SMA'},
        {'fast_period': 10, 'slow_period': 20, 'rsi_period': 14, 'ma_type': 'ema', 
         'min_ma_diff': 0.001, 'rsi_oversold': 30, 'rsi_overbought': 70, 
         'name': 'MA_Triple_10_20'},
        {'fast_period': 12, 'slow_period': 26, 'rsi_period': 14, 'ma_type': 'sma', 
         'min_ma_diff': 0.001, 'rsi_oversold': 30, 'rsi_overbought': 70, 
         'name': 'MA_MACD_Periods'},
        {'fast_period': 8, 'slow_period': 21, 'rsi_period': 14, 'ma_type': 'ema', 
         'min_ma_diff': 0.001, 'rsi_oversold': 30, 'rsi_overbought': 70, 
         'name': 'MA_Fib_8_21'},
        {'fast_period': 13, 'slow_period': 34, 'rsi_period': 14, 'ma_type': 'ema', 
         'min_ma_diff': 0.0015, 'rsi_oversold': 25, 'rsi_overbought': 75, 
         'name': 'MA_Fib_13_34'},
        {'fast_period': 21, 'slow_period': 55, 'rsi_period': 14, 'ma_type': 'sma', 
         'min_ma_diff': 0.002, 'rsi_oversold': 25, 'rsi_overbought': 75, 
         'name': 'MA_Fib_21_55'},
    ],
    
    # ============================================
    # RSI STRATEGIES (8 variantes)
    # ============================================
    'rsi': [
        {'rsi_period': 14, 'rsi_oversold': 20, 'rsi_overbought': 80, 'ma_period': 50, 
         'use_divergences': True, 'name': 'RSI_Ultra_20_80'},
        {'rsi_period': 21, 'rsi_oversold': 25, 'rsi_overbought': 75, 'ma_period': 100, 
         'use_divergences': True, 'name': 'RSI_Long_25_75'},
        {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'ma_period': 50, 
         'use_divergences': True, 'name': 'RSI_Classic_30_70'},
        {'rsi_period': 14, 'rsi_oversold': 25, 'rsi_overbought': 75, 'ma_period': 50, 
         'use_divergences': False, 'name': 'RSI_Conservative'},
        {'rsi_period': 9, 'rsi_oversold': 30, 'rsi_overbought': 70, 'ma_period': 21, 
         'use_divergences': True, 'name': 'RSI_Fast_9'},
        {'rsi_period': 7, 'rsi_oversold': 25, 'rsi_overbought': 75, 'ma_period': 21, 
         'use_divergences': True, 'name': 'RSI_VeryFast_7'},
        {'rsi_period': 28, 'rsi_oversold': 30, 'rsi_overbought': 70, 'ma_period': 100, 
         'use_divergences': True, 'name': 'RSI_Slow_28'},
        {'rsi_period': 21, 'rsi_oversold': 35, 'rsi_overbought': 65, 'ma_period': 50, 
         'use_divergences': False, 'name': 'RSI_Moderate'},
    ],
    
    # ============================================
    # MACD STRATEGIES (6 variantes)
    # ============================================
    'macd': [
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'name': 'MACD_Standard'},
        {'fast_period': 8, 'slow_period': 17, 'signal_period': 9, 'name': 'MACD_Fast'},
        {'fast_period': 5, 'slow_period': 13, 'signal_period': 5, 'name': 'MACD_VeryFast'},
        {'fast_period': 15, 'slow_period': 30, 'signal_period': 10, 'name': 'MACD_Slow'},
        {'fast_period': 20, 'slow_period': 40, 'signal_period': 12, 'name': 'MACD_VerySlow'},
        {'fast_period': 10, 'slow_period': 22, 'signal_period': 8, 'name': 'MACD_Custom'},
    ],
    
    # ============================================
    # BOLLINGER BANDS (5 variantes)
    # ============================================
    'bollinger': [
        {'period': 20, 'std_dev': 2.0, 'rsi_period': 14, 'rsi_oversold': 30, 
         'rsi_overbought': 70, 'name': 'BB_Classic'},
        {'period': 20, 'std_dev': 2.5, 'rsi_period': 14, 'rsi_oversold': 25, 
         'rsi_overbought': 75, 'name': 'BB_Wide'},
        {'period': 20, 'std_dev': 1.5, 'rsi_period': 14, 'rsi_oversold': 30, 
         'rsi_overbought': 70, 'name': 'BB_Tight'},
        {'period': 30, 'std_dev': 2.0, 'rsi_period': 14, 'rsi_oversold': 30, 
         'rsi_overbought': 70, 'name': 'BB_Long'},
        {'period': 10, 'std_dev': 2.0, 'rsi_period': 14, 'rsi_oversold': 30, 
         'rsi_overbought': 70, 'name': 'BB_Short'},
    ],
    
    # ============================================
    # STOCHASTIC (4 variantes)
    # ============================================
    'stochastic': [
        {'k_period': 14, 'd_period': 3, 'oversold': 20, 'overbought': 80, 
         'name': 'Stoch_Classic'},
        {'k_period': 14, 'd_period': 3, 'oversold': 30, 'overbought': 70, 
         'name': 'Stoch_Moderate'},
        {'k_period': 21, 'd_period': 5, 'oversold': 20, 'overbought': 80, 
         'name': 'Stoch_Slow'},
        {'k_period': 5, 'd_period': 3, 'oversold': 20, 'overbought': 80, 
         'name': 'Stoch_Fast'},
    ],
    
    # ============================================
    # ADX TREND (4 variantes)
    # ============================================
    'adx': [
        {'adx_period': 14, 'adx_threshold': 25, 'rsi_period': 14, 'rsi_oversold': 30, 
         'rsi_overbought': 70, 'name': 'ADX_Standard'},
        {'adx_period': 14, 'adx_threshold': 30, 'rsi_period': 14, 'rsi_oversold': 25, 
         'rsi_overbought': 75, 'name': 'ADX_Strong'},
        {'adx_period': 21, 'adx_threshold': 25, 'rsi_period': 14, 'rsi_oversold': 30, 
         'rsi_overbought': 70, 'name': 'ADX_Long'},
        {'adx_period': 14, 'adx_threshold': 20, 'rsi_period': 14, 'rsi_oversold': 30, 
         'rsi_overbought': 70, 'name': 'ADX_Sensitive'},
    ],
    
    # ============================================
    # CCI (3 variantes)
    # ============================================
    'cci': [
        {'cci_period': 20, 'oversold': -100, 'overbought': 100, 'name': 'CCI_Standard'},
        {'cci_period': 14, 'oversold': -150, 'overbought': 150, 'name': 'CCI_Wide'},
        {'cci_period': 30, 'oversold': -100, 'overbought': 100, 'name': 'CCI_Long'},
    ],
    
    # ============================================
    # ICHIMOKU (2 variantes)
    # ============================================
    'ichimoku': [
        {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'name': 'Ichimoku_Standard'},
        {'tenkan': 7, 'kijun': 22, 'senkou_b': 44, 'name': 'Ichimoku_Fast'},
    ],
    
    # ============================================
    # PARABOLIC SAR (3 variantes)
    # ============================================
    'psar': [
        {'acceleration': 0.02, 'maximum': 0.2, 'name': 'PSAR_Standard'},
        {'acceleration': 0.01, 'maximum': 0.15, 'name': 'PSAR_Slow'},
        {'acceleration': 0.03, 'maximum': 0.25, 'name': 'PSAR_Fast'},
    ],
    
    # ============================================
    # ESTRATEGIAS HÍBRIDAS
    # ============================================
    'ma_rsi_macd': [
        {
            'ma_fast': 12, 'ma_slow': 26, 
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'name': 'Hybrid_MA_RSI_MACD_Classic'
        },
        {
            'ma_fast': 20, 'ma_slow': 50, 
            'rsi_period': 14, 'rsi_oversold': 25, 'rsi_overbought': 75,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'name': 'Hybrid_MA_RSI_MACD_Conservative'
        },
        {
            'ma_fast': 8, 'ma_slow': 21, 
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'macd_fast': 8, 'macd_slow': 17, 'macd_signal': 9,
            'name': 'Hybrid_MA_RSI_MACD_Fast'
        },
    ],
    
    'bb_rsi': [
        {
            'bb_period': 20, 'bb_std': 2.0,
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'name': 'Hybrid_BB_RSI_Classic'
        },
        {
            'bb_period': 20, 'bb_std': 2.5,
            'rsi_period': 14, 'rsi_oversold': 20, 'rsi_overbought': 80,
            'name': 'Hybrid_BB_RSI_Wide'
        },
        {
            'bb_period': 30, 'bb_std': 2.0,
            'rsi_period': 21, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'name': 'Hybrid_BB_RSI_Long'
        },
    ],
    
    'adx_ma_rsi': [
        {
            'adx_period': 14, 'adx_threshold': 25,
            'ma_fast': 20, 'ma_slow': 50,
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'name': 'Hybrid_ADX_MA_RSI_Trend'
        },
        {
            'adx_period': 14, 'adx_threshold': 30,
            'ma_fast': 50, 'ma_slow': 200,
            'rsi_period': 14, 'rsi_oversold': 20, 'rsi_overbought': 80,
            'name': 'Hybrid_ADX_MA_RSI_Strong'
        },
    ],
}


def count_total_strategies():
    """Contar total de estrategias disponibles"""
    total = 0
    for strategy_type, configs in COMPLETE_STRATEGY_LIBRARY.items():
        total += len(configs)
    return total


if __name__ == "__main__":
    print("📚 BIBLIOTECA COMPLETA DE ESTRATEGIAS")
    print("="*70)
    
    for strategy_type, configs in COMPLETE_STRATEGY_LIBRARY.items():
        print(f"   {strategy_type}: {len(configs)} variantes")
    
    total = count_total_strategies()
    print("\n" + "="*70)
    print(f"📊 TOTAL: {total} estrategias pre-configuradas")
    print("="*70)