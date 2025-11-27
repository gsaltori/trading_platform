# 📊 RESUMEN DE INDICADORES DISPONIBLES
# Trading Platform - Sistema de Indicadores Extendido

"""
Este archivo documenta todos los indicadores disponibles en la plataforma.
Total: 100+ indicadores organizados por categoría.

USO:
    from strategies import get_indicator_stats
    stats = get_indicator_stats()
    print(f"Total indicadores: {stats['total_indicators']}")
"""

# ============================================================================
# INDICADORES BASE (31 indicadores)
# ============================================================================
BASE_INDICATORS = {
    'TREND': [
        'SMA',           # Simple Moving Average
        'EMA',           # Exponential Moving Average
        'BB',            # Bollinger Bands
        'Ichimoku',      # Ichimoku Cloud
        'ADX',           # Average Directional Index
        'Supertrend',    # Supertrend
    ],
    'MOMENTUM': [
        'RSI',           # Relative Strength Index
        'MACD',          # Moving Average Convergence Divergence
        'Stochastic',    # Stochastic Oscillator
        'StochRSI',      # Stochastic RSI
        'CCI',           # Commodity Channel Index
        'Williams_R',    # Williams %R
        'ROC',           # Rate of Change
        'MOM',           # Momentum
    ],
    'VOLATILITY': [
        'ATR',           # Average True Range
        'Keltner',       # Keltner Channels
        'DonchianChannel', # Donchian Channels
    ],
    'VOLUME': [
        'OBV',           # On Balance Volume
        'MFI',           # Money Flow Index
        'VWAP',          # Volume Weighted Average Price
        'AD',            # Accumulation/Distribution
        'CMF',           # Chaikin Money Flow
    ],
    'PRICE_ACTION': [
        'FibonacciLevels',  # Fibonacci Retracement
        'PivotPoints',      # Pivot Points (Standard)
        'Candle_Patterns',  # Candlestick Patterns
    ]
}

# ============================================================================
# INDICADORES EXTENDIDOS (46 indicadores)
# ============================================================================
EXTENDED_INDICATORS = {
    'TREND': [
        'VWMA',          # Volume Weighted Moving Average
        'HMA',           # Hull Moving Average
        'KAMA',          # Kaufman Adaptive Moving Average
        'ZLEMA',         # Zero Lag EMA
        'T3',            # T3 Moving Average
        'PSAR',          # Parabolic SAR
        'Aroon',         # Aroon Indicator
        'VIDYA',         # Variable Index Dynamic Average
        'Vortex',        # Vortex Indicator
        'DPO',           # Detrended Price Oscillator
    ],
    'MOMENTUM': [
        'TSI',           # True Strength Index
        'UltimateOscillator',  # Ultimate Oscillator
        'AwesomeOscillator',   # Awesome Oscillator
        'TRIX',          # Triple Exponential Average
        'KST',           # Know Sure Thing
        'PPO',           # Percentage Price Oscillator
        'PVO',           # Percentage Volume Oscillator
        'DMI',           # Directional Movement Index
        'RVI_Momentum',  # Relative Vigor Index
        'CMO',           # Chande Momentum Oscillator
        'STC',           # Schaff Trend Cycle
        'FisherTransform',  # Fisher Transform
        'Coppock',       # Coppock Curve
        'ElderRay',      # Elder Ray Index
        'PGO',           # Pretty Good Oscillator
        'WaveTrend',     # Wave Trend Oscillator
    ],
    'VOLATILITY': [
        'ChaikinVolatility',   # Chaikin Volatility
        'HistoricalVolatility', # Historical Volatility
        'MassIndex',     # Mass Index
        'NATR',          # Normalized ATR
        'UlcerIndex',    # Ulcer Index
        'ATRBands',      # ATR Bands
        'RVI_Volatility', # Relative Volatility Index
    ],
    'VOLUME': [
        'VWAPBands',     # VWAP with Bands
        'PVT',           # Price Volume Trend
        'NVI',           # Negative Volume Index
        'PVI',           # Positive Volume Index
        'EOM',           # Ease of Movement
        'ForceIndex',    # Force Index
        'MFI_Enhanced',  # Enhanced MFI
        'Klinger',       # Klinger Volume Oscillator
    ],
    'PRICE_ACTION': [
        'HHLL',          # Higher High / Lower Low
        'ZigZag',        # ZigZag Indicator
        'ADR',           # Average Daily Range
        'CandleBody',    # Candle Body Analysis
        'InsideOutsideBar',  # Inside/Outside Bar
    ]
}

# ============================================================================
# INDICADORES DE SESIONES (7 indicadores) - NUEVO
# ============================================================================
SESSION_INDICATORS = {
    'SESSION_RANGES': [
        'AsianSession',    # Rango de sesión asiática (00:00-09:00 UTC)
        'LondonSession',   # Rango de sesión de Londres (07:00-16:00 UTC)
        'NewYorkSession',  # Rango de sesión de NY (12:00-21:00 UTC)
        'AllSessions',     # Todas las sesiones combinadas
    ],
    'SESSION_ANALYSIS': [
        'SessionStats',    # Estadísticas de sesión (rango promedio, percentil)
        'ORB',             # Opening Range Breakout
        'Killzones',       # ICT Killzones (horarios óptimos de trading)
    ]
}

# Detalles de salida por indicador de sesión:
SESSION_OUTPUTS = {
    'AsianSession': [
        'asian_high',      # Máximo de la sesión
        'asian_low',       # Mínimo de la sesión
        'asian_range',     # Rango (high - low)
        'asian_mid',       # Punto medio
        'in_asian',        # ¿Estamos en sesión asiática?
        'asian_breakout',  # Señal de rompimiento (+1 bullish, -1 bearish)
    ],
    'ORB': [
        'orb_high',        # Máximo del opening range
        'orb_low',         # Mínimo del opening range
        'orb_range',       # Tamaño del rango
        'orb_breakout',    # Señal de rompimiento
        'orb_target_1',    # Target 1x rango
        'orb_target_1_5',  # Target 1.5x rango
        'orb_target_2',    # Target 2x rango
    ],
    'Killzones': [
        'asian_kz',        # Killzone asiática
        'london_kz',       # Killzone de Londres
        'nyam_kz',         # Killzone NY AM
        'nypm_kz',         # Killzone NY PM
        'in_killzone',     # ¿Estamos en alguna killzone?
    ]
}

# ============================================================================
# INDICADORES ICT (6 indicadores) - NUEVO
# ============================================================================
ICT_INDICATORS = {
    'PRICE_DELIVERY': [
        'FairValueGap',    # Fair Value Gaps (imbalances)
        'OrderBlocks',     # Order Blocks (última vela opuesta antes de movimiento fuerte)
        'BreakerBlocks',   # Breaker Blocks (order blocks fallidos)
    ],
    'LIQUIDITY': [
        'LiquidityLevels', # BSL (Buy Side) y SSL (Sell Side)
    ],
    'STRUCTURE': [
        'MarketStructure', # HH, HL, LH, LL, BOS, CHoCH, MSS
    ],
    'ZONES': [
        'PremiumDiscount', # Zonas de premium (50-100%) y discount (0-50%)
        'OTE',             # Optimal Trade Entry (61.8% - 79%)
    ]
}

# Detalles de salida por indicador ICT:
ICT_OUTPUTS = {
    'FairValueGap': [
        'fvg_bullish',          # FVG alcista detectado
        'fvg_bearish',          # FVG bajista detectado
        'fvg_bullish_top',      # Tope del FVG alcista
        'fvg_bullish_bottom',   # Base del FVG alcista
        'fvg_bearish_top',      # Tope del FVG bajista
        'fvg_bearish_bottom',   # Base del FVG bajista
        'fvg_bullish_unfilled', # ¿Hay FVG alcista sin rellenar?
        'fvg_bearish_unfilled', # ¿Hay FVG bajista sin rellenar?
    ],
    'OrderBlocks': [
        'ob_bullish',       # Order block alcista detectado
        'ob_bearish',       # Order block bajista detectado
        'ob_bullish_top',   # Tope del OB alcista
        'ob_bullish_bottom', # Base del OB alcista
        'ob_bearish_top',   # Tope del OB bajista
        'ob_bearish_bottom', # Base del OB bajista
    ],
    'MarketStructure': [
        'swing_high',       # Es un swing high
        'swing_low',        # Es un swing low
        'swing_high_price', # Precio del swing high
        'swing_low_price',  # Precio del swing low
        'hh',               # Higher High
        'hl',               # Higher Low
        'lh',               # Lower High
        'll',               # Lower Low
        'structure',        # 'bullish', 'bearish', 'ranging'
        'bos_bullish',      # Break of Structure alcista
        'bos_bearish',      # Break of Structure bajista
        'choch_bullish',    # Change of Character alcista
        'choch_bearish',    # Change of Character bajista
        'mss',              # Market Structure Shift (+1, -1, 0)
    ],
    'PremiumDiscount': [
        'range_high',       # Máximo del rango
        'range_low',        # Mínimo del rango
        'equilibrium',      # Nivel 50% (equilibrio)
        'premium_start',    # Inicio zona premium (50%)
        'premium_70',       # Nivel 70%
        'premium_79',       # Nivel 79% (OTE)
        'discount_end',     # Fin zona discount (50%)
        'discount_30',      # Nivel 30%
        'discount_21',      # Nivel 21% (OTE)
        'zone',             # 'premium', 'discount', 'equilibrium'
        'zone_pct',         # Porcentaje actual (0-100)
    ],
    'OTE': [
        'ote_bullish_62',   # Nivel OTE 61.8% para compras
        'ote_bullish_70',   # Nivel OTE 70.5% para compras
        'ote_bullish_79',   # Nivel OTE 79% para compras
        'ote_bearish_62',   # Nivel OTE 61.8% para ventas
        'ote_bearish_70',   # Nivel OTE 70.5% para ventas
        'ote_bearish_79',   # Nivel OTE 79% para ventas
        'in_bullish_ote',   # ¿Precio en zona OTE alcista?
        'in_bearish_ote',   # ¿Precio en zona OTE bajista?
    ]
}

# ============================================================================
# INDICADORES SMART MONEY (4 indicadores) - NUEVO
# ============================================================================
SMART_MONEY_INDICATORS = {
    'CONCEPTS': [
        'Displacement',    # Movimientos impulsivos fuertes
        'Inducement',      # Niveles que atrapan retail traders
        'LiquiditySweep',  # Stop hunts (sweeps de liquidez)
        'BreakerBlocks',   # Order blocks que fallaron
    ]
}

# Detalles de salida:
SMART_MONEY_OUTPUTS = {
    'Displacement': [
        'displacement_bullish',  # Desplazamiento alcista
        'displacement_bearish',  # Desplazamiento bajista
        'displacement_strength', # Fuerza del desplazamiento
    ],
    'LiquiditySweep': [
        'sweep_bullish',   # Sweep de liquidez alcista (stops debajo tomados, precio sube)
        'sweep_bearish',   # Sweep de liquidez bajista (stops arriba tomados, precio baja)
    ]
}

# ============================================================================
# INDICADORES ADICIONALES (5 indicadores) - NUEVO
# ============================================================================
ADDITIONAL_INDICATORS = {
    'LEVELS': [
        'PivotPoints',     # Standard, Fibonacci, Camarilla
        'VolumeProfile',   # POC, VAH, VAL
    ],
    'TREND_FOLLOWING': [
        'VWAPBandsExtended',  # VWAP con 3 niveles de bandas
        'RangeFilter',        # Filtro de rango (trend following)
        'ChandelierExit',     # Exit basado en ATR
    ]
}

# ============================================================================
# RESUMEN TOTAL
# ============================================================================
TOTAL_SUMMARY = {
    'base_indicators': 31,
    'extended_indicators': 46,
    'session_indicators': 7,
    'ict_indicators': 6,
    'smart_money_indicators': 4,
    'additional_indicators': 5,
    'TOTAL': 99  # Aproximadamente 100 indicadores
}

# ============================================================================
# EJEMPLOS DE USO
# ============================================================================
USAGE_EXAMPLES = """
# === Ejemplo 1: Estrategia de Rompimiento de Sesión Asiática ===
from strategies import SessionIndicators

# Calcular rangos de sesión
df = SessionIndicators.calculate_session_ranges(df, 'asian', utc_offset=0)
df = SessionIndicators.calculate_session_breakout(df, 'asian')

# Señal de compra: rompimiento alcista del rango asiático
buy_signal = df['asian_breakout'] == 1

# === Ejemplo 2: Estrategia ICT con FVG y Order Blocks ===
from strategies import ICTIndicators

# Calcular FVGs
df = ICTIndicators.calculate_fair_value_gaps(df, min_gap_percent=0.1)

# Calcular Order Blocks
df = ICTIndicators.calculate_order_blocks(df, lookback=10, strength=3)

# Señal: precio en zona de descuento + FVG alcista sin rellenar + OB alcista cercano
buy_conditions = (
    (df['zone'] == 'discount') &
    (df['fvg_bullish_unfilled'] == True) &
    (df['ob_bullish'] == True)
)

# === Ejemplo 3: Market Structure Shift ===
from strategies import ICTIndicators

df = ICTIndicators.calculate_market_structure(df, swing_length=5)

# Señal de cambio de tendencia
bullish_reversal = df['choch_bullish'] == True  # CHoCH alcista
bearish_reversal = df['choch_bearish'] == True  # CHoCH bajista

# === Ejemplo 4: Smart Money - Liquidity Sweep ===
from strategies import SmartMoneyIndicators

df = SmartMoneyIndicators.calculate_liquidity_sweep(df, lookback=20)

# Señal: después de sweep de liquidez, buscar reversión
reversal_long = df['sweep_bullish'] == True   # Stops tomados abajo, esperar subida
reversal_short = df['sweep_bearish'] == True  # Stops tomados arriba, esperar bajada

# === Ejemplo 5: Usando el Generador Automático con Todos los Indicadores ===
from strategies import create_extended_generator, get_indicator_stats

# Crear generador con indicadores extendidos
generator = create_extended_generator()

# Ver estadísticas
stats = get_indicator_stats()
print(f"Total indicadores disponibles: {stats['total_indicators']}")
print(f"Por categoría: {stats['by_type']}")
"""

if __name__ == "__main__":
    print("=" * 60)
    print("📊 RESUMEN DE INDICADORES - TRADING PLATFORM")
    print("=" * 60)
    
    for category, count in TOTAL_SUMMARY.items():
        if category != 'TOTAL':
            print(f"  {category.replace('_', ' ').title()}: {count}")
    
    print("-" * 60)
    print(f"  TOTAL: {TOTAL_SUMMARY['TOTAL']} indicadores")
    print("=" * 60)
