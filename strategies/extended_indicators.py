# strategies/extended_indicators.py
"""
Extended Indicator Library.

Adds additional technical indicators to the strategy generator.
Organized by category: Trend, Momentum, Volatility, Volume, Price Action, Cycle.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ExtendedIndicatorCalculator:
    """
    Extended indicator calculations.
    
    Provides implementations for 50+ additional technical indicators.
    """
    
    # ========================================================================
    # TREND INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_vwma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Volume Weighted Moving Average."""
        df = df.copy()
        df[f'vwma_{period}'] = (
            (df['close'] * df['volume']).rolling(window=period).sum() /
            df['volume'].rolling(window=period).sum()
        )
        return df
    
    @staticmethod
    def calculate_hma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Hull Moving Average - Faster and smoother."""
        df = df.copy()
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = df['close'].rolling(window=half_period).apply(
            lambda x: np.dot(x, np.arange(1, half_period + 1)) / np.arange(1, half_period + 1).sum(),
            raw=True
        )
        wma_full = df['close'].rolling(window=period).apply(
            lambda x: np.dot(x, np.arange(1, period + 1)) / np.arange(1, period + 1).sum(),
            raw=True
        )
        
        raw_hma = 2 * wma_half - wma_full
        df[f'hma_{period}'] = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_period + 1)) / np.arange(1, sqrt_period + 1).sum(),
            raw=True
        )
        return df
    
    @staticmethod
    def calculate_kama(df: pd.DataFrame, period: int = 10, 
                       fast_period: int = 2, slow_period: int = 30) -> pd.DataFrame:
        """Kaufman Adaptive Moving Average."""
        df = df.copy()
        
        # Efficiency Ratio
        change = abs(df['close'] - df['close'].shift(period))
        volatility = df['close'].diff().abs().rolling(window=period).sum()
        er = change / volatility
        er = er.fillna(0)
        
        # Smoothing constants
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama = pd.Series(index=df.index, dtype=float)
        kama.iloc[period] = df['close'].iloc[period]
        
        for i in range(period + 1, len(df)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df['close'].iloc[i] - kama.iloc[i-1])
        
        df[f'kama_{period}'] = kama
        return df
    
    @staticmethod
    def calculate_zlema(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Zero Lag Exponential Moving Average."""
        df = df.copy()
        lag = int((period - 1) / 2)
        ema_data = df['close'] + (df['close'] - df['close'].shift(lag))
        df[f'zlema_{period}'] = ema_data.ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_t3(df: pd.DataFrame, period: int = 5, v_factor: float = 0.7) -> pd.DataFrame:
        """T3 Moving Average - Triple smoothed EMA."""
        df = df.copy()
        
        c1 = -v_factor ** 3
        c2 = 3 * v_factor ** 2 + 3 * v_factor ** 3
        c3 = -6 * v_factor ** 2 - 3 * v_factor - 3 * v_factor ** 3
        c4 = 1 + 3 * v_factor + v_factor ** 3 + 3 * v_factor ** 2
        
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        ema4 = ema3.ewm(span=period, adjust=False).mean()
        ema5 = ema4.ewm(span=period, adjust=False).mean()
        ema6 = ema5.ewm(span=period, adjust=False).mean()
        
        df[f't3_{period}'] = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
        return df
    
    @staticmethod
    def calculate_psar(df: pd.DataFrame, af_start: float = 0.02, 
                       af_step: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """Parabolic SAR."""
        df = df.copy()
        
        length = len(df)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        psar = np.zeros(length)
        psarbull = np.zeros(length)
        psarbear = np.zeros(length)
        bull = True
        af = af_start
        ep = low[0]
        hp = high[0]
        lp = low[0]
        
        for i in range(2, length):
            if bull:
                psar[i] = psar[i-1] + af * (hp - psar[i-1])
            else:
                psar[i] = psar[i-1] + af * (lp - psar[i-1])
            
            reverse = False
            
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = af_start
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = af_start
            
            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + af_step, af_max)
                    if low[i-1] < psar[i]:
                        psar[i] = low[i-1]
                    if low[i-2] < psar[i]:
                        psar[i] = low[i-2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + af_step, af_max)
                    if high[i-1] > psar[i]:
                        psar[i] = high[i-1]
                    if high[i-2] > psar[i]:
                        psar[i] = high[i-2]
            
            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]
        
        df['psar'] = psar
        df['psar_bull'] = np.where(psarbull > 0, psarbull, np.nan)
        df['psar_bear'] = np.where(psarbear > 0, psarbear, np.nan)
        df['psar_direction'] = np.where(df['close'] > psar, 1, -1)
        return df
    
    @staticmethod
    def calculate_aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """Aroon Indicator."""
        df = df.copy()
        
        aroon_up = 100 * df['high'].rolling(window=period + 1).apply(
            lambda x: x.argmax(), raw=True
        ) / period
        aroon_down = 100 * df['low'].rolling(window=period + 1).apply(
            lambda x: x.argmin(), raw=True
        ) / period
        
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        df['aroon_oscillator'] = aroon_up - aroon_down
        return df
    
    @staticmethod
    def calculate_vidya(df: pd.DataFrame, period: int = 14, 
                        cmo_period: int = 9) -> pd.DataFrame:
        """Variable Index Dynamic Average."""
        df = df.copy()
        
        # Chande Momentum Oscillator
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        sum_gain = gain.rolling(window=cmo_period).sum()
        sum_loss = loss.rolling(window=cmo_period).sum()
        
        cmo = abs((sum_gain - sum_loss) / (sum_gain + sum_loss))
        
        # VIDYA
        alpha = 2 / (period + 1)
        vidya = pd.Series(index=df.index, dtype=float)
        vidya.iloc[period] = df['close'].iloc[period]
        
        for i in range(period + 1, len(df)):
            vidya.iloc[i] = alpha * cmo.iloc[i] * df['close'].iloc[i] + (1 - alpha * cmo.iloc[i]) * vidya.iloc[i-1]
        
        df[f'vidya_{period}'] = vidya
        return df
    
    @staticmethod
    def calculate_vortex(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Vortex Indicator."""
        df = df.copy()
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        vm_plus = abs(df['high'] - df['low'].shift(1))
        vm_minus = abs(df['low'] - df['high'].shift(1))
        
        tr_sum = tr.rolling(window=period).sum()
        vm_plus_sum = vm_plus.rolling(window=period).sum()
        vm_minus_sum = vm_minus.rolling(window=period).sum()
        
        df['vortex_plus'] = vm_plus_sum / tr_sum
        df['vortex_minus'] = vm_minus_sum / tr_sum
        df['vortex_diff'] = df['vortex_plus'] - df['vortex_minus']
        return df
    
    @staticmethod
    def calculate_dpo(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Detrended Price Oscillator."""
        df = df.copy()
        shift = int(period / 2) + 1
        sma = df['close'].rolling(window=period).mean()
        df[f'dpo_{period}'] = df['close'].shift(shift) - sma
        return df
    
    # ========================================================================
    # BASE INDICATORS (missing from base calculator)
    # ========================================================================
    
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame, tenkan: int = 9, 
                           kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
        """Ichimoku Cloud."""
        df = df.copy()
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan).max()
        tenkan_low = df['low'].rolling(window=tenkan).min()
        df['tenkan'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun).max()
        kijun_low = df['low'].rolling(window=kijun).min()
        df['kijun'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_high = df['high'].rolling(window=senkou).max()
        senkou_low = df['low'].rolling(window=senkou).min()
        df['senkou_b'] = ((senkou_high + senkou_low) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        df['chikou'] = df['close'].shift(-kijun)
        
        return df
    
    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, period: int = 10, 
                              multiplier: float = 3.0) -> pd.DataFrame:
        """Supertrend Indicator."""
        df = df.copy()
        
        # Calculate ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate basic bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Calculate Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                
                if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
        
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        return df
    
    @staticmethod
    def calculate_stochrsi(df: pd.DataFrame, rsi_period: int = 14, 
                           stoch_period: int = 14, k_period: int = 3, 
                           d_period: int = 3) -> pd.DataFrame:
        """Stochastic RSI."""
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic of RSI
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        
        stochrsi_k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
        stochrsi_d = stochrsi_k.rolling(window=d_period).mean()
        
        df['stochrsi_k'] = stochrsi_k
        df['stochrsi_d'] = stochrsi_d
        return df
    
    @staticmethod
    def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index."""
        df = df.copy()
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        return df
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Volume Weighted Average Price."""
        df = df.copy()
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    
    @staticmethod
    def calculate_ad(df: pd.DataFrame) -> pd.DataFrame:
        """Accumulation/Distribution Line."""
        df = df.copy()
        
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        df['ad'] = (clv * df['volume']).cumsum()
        return df
    
    @staticmethod
    def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Chaikin Money Flow."""
        df = df.copy()
        
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        
        df['cmf'] = (clv * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return df
    
    @staticmethod
    def calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Fibonacci Retracement Levels."""
        df = df.copy()
        
        high = df['high'].rolling(window=lookback).max()
        low = df['low'].rolling(window=lookback).min()
        diff = high - low
        
        df['fib_0'] = low
        df['fib_236'] = low + 0.236 * diff
        df['fib_382'] = low + 0.382 * diff
        df['fib_500'] = low + 0.500 * diff
        df['fib_618'] = low + 0.618 * diff
        df['fib_786'] = low + 0.786 * diff
        df['fib_100'] = high
        return df
    
    @staticmethod
    def calculate_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick Pattern Detection."""
        df = df.copy()
        
        body = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']
        
        # Doji: very small body
        df['doji'] = (body < full_range * 0.1).astype(int)
        
        # Hammer: small body at top, long lower wick
        df['hammer'] = (
            (body < full_range * 0.3) & 
            (lower_wick > body * 2) & 
            (upper_wick < body * 0.5)
        ).astype(int)
        
        # Engulfing: current candle engulfs previous
        bullish_engulfing = (
            (df['close'] > df['open']) &  # Current is bullish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous is bearish
            (df['open'] < df['close'].shift(1)) &  # Current open below previous close
            (df['close'] > df['open'].shift(1))  # Current close above previous open
        )
        bearish_engulfing = (
            (df['close'] < df['open']) &  # Current is bearish
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous is bullish
            (df['open'] > df['close'].shift(1)) &  # Current open above previous close
            (df['close'] < df['open'].shift(1))  # Current close below previous open
        )
        df['engulfing'] = np.where(bullish_engulfing, 1, np.where(bearish_engulfing, -1, 0))
        
        # Morning Star (simplified): bearish, small body, bullish
        df['morning_star'] = (
            (df['close'].shift(2) < df['open'].shift(2)) &  # First bearish
            (body.shift(1) < full_range.shift(1) * 0.3) &  # Second small body
            (df['close'] > df['open']) &  # Third bullish
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Close above midpoint
        ).astype(int)
        
        # Evening Star (simplified): bullish, small body, bearish
        df['evening_star'] = (
            (df['close'].shift(2) > df['open'].shift(2)) &  # First bullish
            (body.shift(1) < full_range.shift(1) * 0.3) &  # Second small body
            (df['close'] < df['open']) &  # Third bearish
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Close below midpoint
        ).astype(int)
        
        return df
    
    # ========================================================================
    # MOMENTUM INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_tsi(df: pd.DataFrame, long_period: int = 25, 
                      short_period: int = 13, signal_period: int = 13) -> pd.DataFrame:
        """True Strength Index."""
        df = df.copy()
        
        diff = df['close'].diff()
        
        double_smooth_pc = diff.ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
        double_smooth_abs_pc = diff.abs().ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
        
        df['tsi'] = 100 * (double_smooth_pc / double_smooth_abs_pc)
        df['tsi_signal'] = df['tsi'].ewm(span=signal_period, adjust=False).mean()
        df['tsi_histogram'] = df['tsi'] - df['tsi_signal']
        return df
    
    @staticmethod
    def calculate_ultimate_oscillator(df: pd.DataFrame, period1: int = 7, 
                                       period2: int = 14, period3: int = 28) -> pd.DataFrame:
        """Ultimate Oscillator."""
        df = df.copy()
        
        bp = df['close'] - pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
        
        df['ultimate_oscillator'] = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return df
    
    @staticmethod
    def calculate_awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.DataFrame:
        """Awesome Oscillator."""
        df = df.copy()
        median_price = (df['high'] + df['low']) / 2
        df['ao'] = median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()
        df['ao_color'] = np.where(df['ao'] > df['ao'].shift(1), 1, -1)
        return df
    
    @staticmethod
    def calculate_trix(df: pd.DataFrame, period: int = 15, signal: int = 9) -> pd.DataFrame:
        """TRIX - Triple Exponential Average."""
        df = df.copy()
        
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        
        df['trix'] = ema3.pct_change() * 10000
        df['trix_signal'] = df['trix'].ewm(span=signal, adjust=False).mean()
        df['trix_histogram'] = df['trix'] - df['trix_signal']
        return df
    
    @staticmethod
    def calculate_kst(df: pd.DataFrame) -> pd.DataFrame:
        """Know Sure Thing Oscillator."""
        df = df.copy()
        
        roc1 = df['close'].pct_change(10) * 100
        roc2 = df['close'].pct_change(15) * 100
        roc3 = df['close'].pct_change(20) * 100
        roc4 = df['close'].pct_change(30) * 100
        
        sma1 = roc1.rolling(window=10).mean()
        sma2 = roc2.rolling(window=10).mean()
        sma3 = roc3.rolling(window=10).mean()
        sma4 = roc4.rolling(window=15).mean()
        
        df['kst'] = sma1 + 2 * sma2 + 3 * sma3 + 4 * sma4
        df['kst_signal'] = df['kst'].rolling(window=9).mean()
        df['kst_histogram'] = df['kst'] - df['kst_signal']
        return df
    
    @staticmethod
    def calculate_ppo(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> pd.DataFrame:
        """Percentage Price Oscillator."""
        df = df.copy()
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['ppo'] = ((ema_fast - ema_slow) / ema_slow) * 100
        df['ppo_signal'] = df['ppo'].ewm(span=signal, adjust=False).mean()
        df['ppo_histogram'] = df['ppo'] - df['ppo_signal']
        return df
    
    @staticmethod
    def calculate_pvo(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                      signal: int = 9) -> pd.DataFrame:
        """Percentage Volume Oscillator."""
        df = df.copy()
        
        ema_fast = df['volume'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['volume'].ewm(span=slow, adjust=False).mean()
        
        df['pvo'] = ((ema_fast - ema_slow) / ema_slow) * 100
        df['pvo_signal'] = df['pvo'].ewm(span=signal, adjust=False).mean()
        df['pvo_histogram'] = df['pvo'] - df['pvo_signal']
        return df
    
    @staticmethod
    def calculate_dmi(df: pd.DataFrame, period: int = 14, adx_smooth: int = 14) -> pd.DataFrame:
        """Directional Movement Index (enhanced)."""
        df = df.copy()
        
        # True Range
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smooth
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=adx_smooth, adjust=False).mean()
        
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['dx'] = dx
        df['adx_enhanced'] = adx
        df['dmi_diff'] = plus_di - minus_di
        return df
    
    @staticmethod
    def calculate_rvi(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Relative Vigor Index."""
        df = df.copy()
        
        # Numerator: Close - Open
        close_open = df['close'] - df['open']
        numerator = (close_open + 2 * close_open.shift(1) + 2 * close_open.shift(2) + close_open.shift(3)) / 6
        
        # Denominator: High - Low
        high_low = df['high'] - df['low']
        denominator = (high_low + 2 * high_low.shift(1) + 2 * high_low.shift(2) + high_low.shift(3)) / 6
        
        # RVI
        numerator_sum = numerator.rolling(window=period).sum()
        denominator_sum = denominator.rolling(window=period).sum()
        
        df['rvi'] = numerator_sum / denominator_sum
        df['rvi_signal'] = (df['rvi'] + 2 * df['rvi'].shift(1) + 2 * df['rvi'].shift(2) + df['rvi'].shift(3)) / 6
        return df
    
    @staticmethod
    def calculate_cmo(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Chande Momentum Oscillator."""
        df = df.copy()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        sum_gain = gain.rolling(window=period).sum()
        sum_loss = loss.rolling(window=period).sum()
        
        df['cmo'] = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
        return df
    
    # ========================================================================
    # VOLATILITY INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_chaikin_volatility(df: pd.DataFrame, ema_period: int = 10,
                                      roc_period: int = 10) -> pd.DataFrame:
        """Chaikin Volatility."""
        df = df.copy()
        
        hl_spread = df['high'] - df['low']
        ema_spread = hl_spread.ewm(span=ema_period, adjust=False).mean()
        
        df['chaikin_volatility'] = ((ema_spread - ema_spread.shift(roc_period)) / 
                                     ema_spread.shift(roc_period)) * 100
        return df
    
    @staticmethod
    def calculate_historical_volatility(df: pd.DataFrame, period: int = 20,
                                         annual_factor: int = 252) -> pd.DataFrame:
        """Historical Volatility (annualized)."""
        df = df.copy()
        
        log_returns = np.log(df['close'] / df['close'].shift(1))
        df['historical_volatility'] = log_returns.rolling(window=period).std() * np.sqrt(annual_factor) * 100
        return df
    
    @staticmethod
    def calculate_mass_index(df: pd.DataFrame, ema_period: int = 9,
                             sum_period: int = 25) -> pd.DataFrame:
        """Mass Index."""
        df = df.copy()
        
        hl_spread = df['high'] - df['low']
        ema1 = hl_spread.ewm(span=ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
        
        ratio = ema1 / ema2
        df['mass_index'] = ratio.rolling(window=sum_period).sum()
        return df
    
    @staticmethod
    def calculate_natr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Normalized Average True Range."""
        df = df.copy()
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        df['natr'] = (atr / df['close']) * 100
        return df
    
    @staticmethod
    def calculate_ulcer_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Ulcer Index - Measures downside volatility."""
        df = df.copy()
        
        max_close = df['close'].rolling(window=period).max()
        percent_drawdown = ((df['close'] - max_close) / max_close) * 100
        squared_drawdown = percent_drawdown ** 2
        
        df['ulcer_index'] = np.sqrt(squared_drawdown.rolling(window=period).mean())
        return df
    
    @staticmethod
    def calculate_atr_bands(df: pd.DataFrame, period: int = 14,
                            multiplier: float = 2.0) -> pd.DataFrame:
        """ATR Bands."""
        df = df.copy()
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        df['atr_band_middle'] = df['close'].rolling(window=period).mean()
        df['atr_band_upper'] = df['atr_band_middle'] + (atr * multiplier)
        df['atr_band_lower'] = df['atr_band_middle'] - (atr * multiplier)
        return df
    
    @staticmethod
    def calculate_rvi_volatility(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Relative Volatility Index."""
        df = df.copy()
        
        std = df['close'].rolling(window=period).std()
        delta = std.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        
        df['rvi_volatility'] = 100 * avg_gain / (avg_gain + avg_loss)
        return df
    
    # ========================================================================
    # VOLUME INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_vwap_bands(df: pd.DataFrame, std_mult: float = 2.0) -> pd.DataFrame:
        """VWAP with standard deviation bands."""
        df = df.copy()
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cum_vol = df['volume'].cumsum()
        cum_vol_price = (typical_price * df['volume']).cumsum()
        
        vwap = cum_vol_price / cum_vol
        
        # Calculate squared differences
        sq_diff = ((typical_price - vwap) ** 2 * df['volume']).cumsum()
        std = np.sqrt(sq_diff / cum_vol)
        
        df['vwap'] = vwap
        df['vwap_upper'] = vwap + (std * std_mult)
        df['vwap_lower'] = vwap - (std * std_mult)
        return df
    
    @staticmethod
    def calculate_pvt(df: pd.DataFrame) -> pd.DataFrame:
        """Price Volume Trend."""
        df = df.copy()
        df['pvt'] = (((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']).cumsum()
        return df
    
    @staticmethod
    def calculate_nvi(df: pd.DataFrame) -> pd.DataFrame:
        """Negative Volume Index."""
        df = df.copy()
        
        roc = df['close'].pct_change()
        vol_down = df['volume'] < df['volume'].shift(1)
        
        nvi = pd.Series(index=df.index, dtype=float)
        nvi.iloc[0] = 1000
        
        for i in range(1, len(df)):
            if vol_down.iloc[i]:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + roc.iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        df['nvi'] = nvi
        df['nvi_signal'] = nvi.ewm(span=255, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_pvi(df: pd.DataFrame) -> pd.DataFrame:
        """Positive Volume Index."""
        df = df.copy()
        
        roc = df['close'].pct_change()
        vol_up = df['volume'] > df['volume'].shift(1)
        
        pvi = pd.Series(index=df.index, dtype=float)
        pvi.iloc[0] = 1000
        
        for i in range(1, len(df)):
            if vol_up.iloc[i]:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + roc.iloc[i])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        df['pvi'] = pvi
        df['pvi_signal'] = pvi.ewm(span=255, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_eom(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Ease of Movement."""
        df = df.copy()
        
        dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        br = (df['volume'] / 100000000) / (df['high'] - df['low'])
        
        eom = dm / br
        df['eom'] = eom.rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_force_index(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """Force Index."""
        df = df.copy()
        
        force = df['close'].diff() * df['volume']
        df['force_index'] = force.ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_mfi_enhanced(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index with signal line."""
        df = df.copy()
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        df['mfi_enhanced'] = 100 - (100 / (1 + money_ratio))
        df['mfi_signal'] = df['mfi_enhanced'].rolling(window=9).mean()
        return df
    
    @staticmethod
    def calculate_klinger(df: pd.DataFrame, fast: int = 34, slow: int = 55,
                          signal: int = 13) -> pd.DataFrame:
        """Klinger Volume Oscillator."""
        df = df.copy()
        
        hlc = df['high'] + df['low'] + df['close']
        dm = df['high'] - df['low']
        
        # Convert to Series for shift operations
        hlc_series = pd.Series(hlc.values, index=df.index)
        dm_series = pd.Series(dm.values, index=df.index)
        
        trend = pd.Series(np.where(hlc_series > hlc_series.shift(1), 1, -1), index=df.index)
        
        # Calculate CM using pandas operations
        cm = pd.Series(index=df.index, dtype=float)
        cm.iloc[0] = dm_series.iloc[0]
        for i in range(1, len(df)):
            if trend.iloc[i] == trend.iloc[i-1]:
                cm.iloc[i] = cm.iloc[i-1] + dm_series.iloc[i]
            else:
                cm.iloc[i] = dm_series.iloc[i]
        
        # Avoid division by zero
        cm = cm.replace(0, np.nan)
        
        vf = df['volume'] * abs(2 * (dm_series / cm) - 1) * trend * 100
        
        df['klinger'] = vf.ewm(span=fast, adjust=False).mean() - vf.ewm(span=slow, adjust=False).mean()
        df['klinger_signal'] = df['klinger'].ewm(span=signal, adjust=False).mean()
        return df
    
    # ========================================================================
    # PRICE ACTION INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_higher_high_lower_low(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Higher High and Lower Low detection."""
        df = df.copy()
        
        rolling_high = df['high'].rolling(window=period).max()
        rolling_low = df['low'].rolling(window=period).min()
        
        df['higher_high'] = (df['high'] >= rolling_high).astype(int)
        df['lower_low'] = (df['low'] <= rolling_low).astype(int)
        
        # Trend based on HH/HL vs LH/LL
        df['hh_count'] = df['higher_high'].rolling(window=period).sum()
        df['ll_count'] = df['lower_low'].rolling(window=period).sum()
        df['trend_strength'] = df['hh_count'] - df['ll_count']
        return df
    
    @staticmethod
    def calculate_zigzag(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
        """ZigZag indicator."""
        df = df.copy()
        
        threshold_pct = threshold / 100
        high = df['high'].values
        low = df['low'].values
        
        zigzag = np.zeros(len(df))
        zigzag[0] = (high[0] + low[0]) / 2
        
        trend = 0  # 1 for up, -1 for down
        last_pivot = zigzag[0]
        last_pivot_idx = 0
        
        for i in range(1, len(df)):
            if trend >= 0:
                if high[i] > last_pivot * (1 + threshold_pct):
                    zigzag[i] = high[i]
                    last_pivot = high[i]
                    last_pivot_idx = i
                    trend = 1
                elif low[i] < last_pivot * (1 - threshold_pct):
                    zigzag[last_pivot_idx] = last_pivot
                    zigzag[i] = low[i]
                    last_pivot = low[i]
                    last_pivot_idx = i
                    trend = -1
            else:
                if low[i] < last_pivot * (1 - threshold_pct):
                    zigzag[i] = low[i]
                    last_pivot = low[i]
                    last_pivot_idx = i
                    trend = -1
                elif high[i] > last_pivot * (1 + threshold_pct):
                    zigzag[last_pivot_idx] = last_pivot
                    zigzag[i] = high[i]
                    last_pivot = high[i]
                    last_pivot_idx = i
                    trend = 1
        
        df['zigzag'] = np.where(zigzag > 0, zigzag, np.nan)
        return df
    
    @staticmethod
    def calculate_adr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Daily Range."""
        df = df.copy()
        daily_range = df['high'] - df['low']
        df['adr'] = daily_range.rolling(window=period).mean()
        df['adr_percent'] = (df['adr'] / df['close']) * 100
        return df
    
    @staticmethod
    def calculate_candle_body(df: pd.DataFrame) -> pd.DataFrame:
        """Candle body analysis."""
        df = df.copy()
        
        body = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        upper_wick = df['high'] - pd.concat([df['close'], df['open']], axis=1).max(axis=1)
        lower_wick = pd.concat([df['close'], df['open']], axis=1).min(axis=1) - df['low']
        
        df['body_size'] = body
        df['body_percent'] = (body / full_range) * 100
        df['upper_wick_percent'] = (upper_wick / full_range) * 100
        df['lower_wick_percent'] = (lower_wick / full_range) * 100
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        return df
    
    @staticmethod
    def calculate_inside_outside_bar(df: pd.DataFrame) -> pd.DataFrame:
        """Inside and Outside bar detection."""
        df = df.copy()
        
        # Inside bar: current bar inside previous bar
        inside = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
        
        # Outside bar: current bar engulfs previous bar
        outside = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        
        df['inside_bar'] = inside.astype(int)
        df['outside_bar'] = outside.astype(int)
        return df
    
    # ========================================================================
    # CYCLE INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_schaff_trend(df: pd.DataFrame, period: int = 10,
                               fast: int = 23, slow: int = 50) -> pd.DataFrame:
        """Schaff Trend Cycle."""
        df = df.copy()
        
        # MACD
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        
        # Stochastic of MACD
        lowest_macd = macd.rolling(window=period).min()
        highest_macd = macd.rolling(window=period).max()
        
        k = 100 * (macd - lowest_macd) / (highest_macd - lowest_macd)
        
        # Double smooth
        d = k.ewm(span=period, adjust=False).mean()
        
        lowest_d = d.rolling(window=period).min()
        highest_d = d.rolling(window=period).max()
        
        df['stc'] = 100 * (d - lowest_d) / (highest_d - lowest_d)
        df['stc_signal'] = df['stc'].rolling(window=3).mean()
        return df
    
    @staticmethod
    def calculate_ehlers_fisher(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Ehlers Fisher Transform."""
        df = df.copy()
        
        mid = (df['high'] + df['low']) / 2
        lowest = mid.rolling(window=period).min()
        highest = mid.rolling(window=period).max()
        
        value = 2 * ((mid - lowest) / (highest - lowest) - 0.5)
        value = value.clip(-0.999, 0.999)
        
        fish = 0.5 * np.log((1 + value) / (1 - value))
        df['fisher'] = fish.ewm(span=period, adjust=False).mean()
        df['fisher_signal'] = df['fisher'].shift(1)
        return df
    
    @staticmethod
    def calculate_coppock(df: pd.DataFrame, wma_period: int = 10,
                          roc1: int = 14, roc2: int = 11) -> pd.DataFrame:
        """Coppock Curve."""
        df = df.copy()
        
        roc_long = df['close'].pct_change(roc1) * 100
        roc_short = df['close'].pct_change(roc2) * 100
        
        combined = roc_long + roc_short
        weights = np.arange(1, wma_period + 1)
        
        df['coppock'] = combined.rolling(window=wma_period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return df
    
    @staticmethod
    def calculate_elder_ray(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """Elder Ray Index (Bull/Bear Power)."""
        df = df.copy()
        
        ema = df['close'].ewm(span=period, adjust=False).mean()
        df['bull_power'] = df['high'] - ema
        df['bear_power'] = df['low'] - ema
        df['elder_force'] = df['bull_power'] + df['bear_power']
        return df
    
    @staticmethod
    def calculate_pretty_good_oscillator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Pretty Good Oscillator."""
        df = df.copy()
        
        sma = df['close'].rolling(window=period).mean()
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=period, adjust=False).mean()
        
        df['pgo'] = (df['close'] - sma) / atr
        return df
    
    @staticmethod
    def calculate_wave_trend(df: pd.DataFrame, channel: int = 10,
                             average: int = 21) -> pd.DataFrame:
        """WaveTrend Oscillator."""
        df = df.copy()
        
        ap = (df['high'] + df['low'] + df['close']) / 3
        esa = ap.ewm(span=channel, adjust=False).mean()
        d = abs(ap - esa).ewm(span=channel, adjust=False).mean()
        
        ci = (ap - esa) / (0.015 * d)
        
        df['wt1'] = ci.ewm(span=average, adjust=False).mean()
        df['wt2'] = df['wt1'].rolling(window=4).mean()
        df['wt_diff'] = df['wt1'] - df['wt2']
        return df


def get_extended_indicator_configs() -> List[Dict]:
    """
    Returns configuration for all extended indicators.
    Use this to register them in IndicatorLibrary.
    """
    return [
        # Trend Indicators
        {'name': 'VWMA', 'type': 'trend',
         'params': {'period': {'min': 5, 'max': 100, 'default': 20}},
         'outputs': ['vwma_{period}']},
        
        {'name': 'HMA', 'type': 'trend',
         'params': {'period': {'min': 5, 'max': 100, 'default': 20}},
         'outputs': ['hma_{period}']},
        
        {'name': 'KAMA', 'type': 'trend',
         'params': {'period': {'min': 5, 'max': 50, 'default': 10},
                    'fast_period': {'min': 2, 'max': 5, 'default': 2},
                    'slow_period': {'min': 20, 'max': 50, 'default': 30}},
         'outputs': ['kama_{period}']},
        
        {'name': 'ZLEMA', 'type': 'trend',
         'params': {'period': {'min': 5, 'max': 100, 'default': 20}},
         'outputs': ['zlema_{period}']},
        
        {'name': 'T3', 'type': 'trend',
         'params': {'period': {'min': 3, 'max': 20, 'default': 5},
                    'v_factor': {'min': 0.5, 'max': 0.9, 'default': 0.7}},
         'outputs': ['t3_{period}']},
        
        {'name': 'PSAR', 'type': 'trend',
         'params': {'af_start': {'min': 0.01, 'max': 0.05, 'default': 0.02},
                    'af_step': {'min': 0.01, 'max': 0.05, 'default': 0.02},
                    'af_max': {'min': 0.1, 'max': 0.3, 'default': 0.2}},
         'outputs': ['psar', 'psar_bull', 'psar_bear', 'psar_direction']},
        
        {'name': 'Aroon', 'type': 'trend',
         'params': {'period': {'min': 10, 'max': 50, 'default': 25}},
         'outputs': ['aroon_up', 'aroon_down', 'aroon_oscillator']},
        
        {'name': 'VIDYA', 'type': 'trend',
         'params': {'period': {'min': 7, 'max': 30, 'default': 14},
                    'cmo_period': {'min': 5, 'max': 15, 'default': 9}},
         'outputs': ['vidya_{period}']},
        
        {'name': 'Vortex', 'type': 'trend',
         'params': {'period': {'min': 7, 'max': 30, 'default': 14}},
         'outputs': ['vortex_plus', 'vortex_minus', 'vortex_diff']},
        
        {'name': 'DPO', 'type': 'trend',
         'params': {'period': {'min': 10, 'max': 30, 'default': 20}},
         'outputs': ['dpo_{period}']},
        
        # Momentum Indicators
        {'name': 'TSI', 'type': 'momentum',
         'params': {'long_period': {'min': 15, 'max': 35, 'default': 25},
                    'short_period': {'min': 7, 'max': 20, 'default': 13},
                    'signal_period': {'min': 7, 'max': 20, 'default': 13}},
         'outputs': ['tsi', 'tsi_signal', 'tsi_histogram']},
        
        {'name': 'UltimateOscillator', 'type': 'momentum',
         'params': {'period1': {'min': 5, 'max': 10, 'default': 7},
                    'period2': {'min': 10, 'max': 20, 'default': 14},
                    'period3': {'min': 20, 'max': 40, 'default': 28}},
         'outputs': ['ultimate_oscillator']},
        
        {'name': 'AwesomeOscillator', 'type': 'momentum',
         'params': {'fast': {'min': 3, 'max': 10, 'default': 5},
                    'slow': {'min': 20, 'max': 50, 'default': 34}},
         'outputs': ['ao', 'ao_color']},
        
        {'name': 'TRIX', 'type': 'momentum',
         'params': {'period': {'min': 10, 'max': 25, 'default': 15},
                    'signal': {'min': 5, 'max': 15, 'default': 9}},
         'outputs': ['trix', 'trix_signal', 'trix_histogram']},
        
        {'name': 'KST', 'type': 'momentum',
         'params': {},
         'outputs': ['kst', 'kst_signal', 'kst_histogram']},
        
        {'name': 'PPO', 'type': 'momentum',
         'params': {'fast': {'min': 8, 'max': 16, 'default': 12},
                    'slow': {'min': 20, 'max': 35, 'default': 26},
                    'signal': {'min': 5, 'max': 15, 'default': 9}},
         'outputs': ['ppo', 'ppo_signal', 'ppo_histogram']},
        
        {'name': 'PVO', 'type': 'momentum',
         'params': {'fast': {'min': 8, 'max': 16, 'default': 12},
                    'slow': {'min': 20, 'max': 35, 'default': 26},
                    'signal': {'min': 5, 'max': 15, 'default': 9}},
         'outputs': ['pvo', 'pvo_signal', 'pvo_histogram']},
        
        {'name': 'DMI', 'type': 'momentum',
         'params': {'period': {'min': 7, 'max': 30, 'default': 14},
                    'adx_smooth': {'min': 7, 'max': 21, 'default': 14}},
         'outputs': ['plus_di', 'minus_di', 'dx', 'adx_enhanced', 'dmi_diff']},
        
        {'name': 'RVI_Momentum', 'type': 'momentum',
         'params': {'period': {'min': 5, 'max': 20, 'default': 10}},
         'outputs': ['rvi', 'rvi_signal']},
        
        {'name': 'CMO', 'type': 'momentum',
         'params': {'period': {'min': 7, 'max': 25, 'default': 14}},
         'outputs': ['cmo']},
        
        # Volatility Indicators
        {'name': 'ChaikinVolatility', 'type': 'volatility',
         'params': {'ema_period': {'min': 5, 'max': 20, 'default': 10},
                    'roc_period': {'min': 5, 'max': 20, 'default': 10}},
         'outputs': ['chaikin_volatility']},
        
        {'name': 'HistoricalVolatility', 'type': 'volatility',
         'params': {'period': {'min': 10, 'max': 50, 'default': 20}},
         'outputs': ['historical_volatility']},
        
        {'name': 'MassIndex', 'type': 'volatility',
         'params': {'ema_period': {'min': 5, 'max': 15, 'default': 9},
                    'sum_period': {'min': 15, 'max': 35, 'default': 25}},
         'outputs': ['mass_index']},
        
        {'name': 'NATR', 'type': 'volatility',
         'params': {'period': {'min': 7, 'max': 25, 'default': 14}},
         'outputs': ['natr']},
        
        {'name': 'UlcerIndex', 'type': 'volatility',
         'params': {'period': {'min': 7, 'max': 25, 'default': 14}},
         'outputs': ['ulcer_index']},
        
        {'name': 'ATRBands', 'type': 'volatility',
         'params': {'period': {'min': 7, 'max': 25, 'default': 14},
                    'multiplier': {'min': 1.0, 'max': 4.0, 'default': 2.0}},
         'outputs': ['atr_band_upper', 'atr_band_middle', 'atr_band_lower']},
        
        {'name': 'RVI_Volatility', 'type': 'volatility',
         'params': {'period': {'min': 5, 'max': 20, 'default': 10}},
         'outputs': ['rvi_volatility']},
        
        # Volume Indicators
        {'name': 'VWAPBands', 'type': 'volume',
         'params': {'std_mult': {'min': 1.0, 'max': 3.0, 'default': 2.0}},
         'outputs': ['vwap', 'vwap_upper', 'vwap_lower']},
        
        {'name': 'PVT', 'type': 'volume',
         'params': {},
         'outputs': ['pvt']},
        
        {'name': 'NVI', 'type': 'volume',
         'params': {},
         'outputs': ['nvi', 'nvi_signal']},
        
        {'name': 'PVI', 'type': 'volume',
         'params': {},
         'outputs': ['pvi', 'pvi_signal']},
        
        {'name': 'EOM', 'type': 'volume',
         'params': {'period': {'min': 7, 'max': 25, 'default': 14}},
         'outputs': ['eom']},
        
        {'name': 'ForceIndex', 'type': 'volume',
         'params': {'period': {'min': 7, 'max': 25, 'default': 13}},
         'outputs': ['force_index']},
        
        {'name': 'MFI_Enhanced', 'type': 'volume',
         'params': {'period': {'min': 7, 'max': 25, 'default': 14}},
         'outputs': ['mfi_enhanced', 'mfi_signal']},
        
        {'name': 'Klinger', 'type': 'volume',
         'params': {'fast': {'min': 25, 'max': 45, 'default': 34},
                    'slow': {'min': 45, 'max': 70, 'default': 55},
                    'signal': {'min': 7, 'max': 20, 'default': 13}},
         'outputs': ['klinger', 'klinger_signal']},
        
        # Price Action Indicators
        {'name': 'HHLL', 'type': 'price_action',
         'params': {'period': {'min': 10, 'max': 50, 'default': 20}},
         'outputs': ['higher_high', 'lower_low', 'hh_count', 'll_count', 'trend_strength']},
        
        {'name': 'ZigZag', 'type': 'price_action',
         'params': {'threshold': {'min': 1.0, 'max': 10.0, 'default': 5.0}},
         'outputs': ['zigzag']},
        
        {'name': 'ADR', 'type': 'price_action',
         'params': {'period': {'min': 7, 'max': 30, 'default': 14}},
         'outputs': ['adr', 'adr_percent']},
        
        {'name': 'CandleBody', 'type': 'price_action',
         'params': {},
         'outputs': ['body_size', 'body_percent', 'upper_wick_percent', 'lower_wick_percent', 'is_bullish']},
        
        {'name': 'InsideOutsideBar', 'type': 'price_action',
         'params': {},
         'outputs': ['inside_bar', 'outside_bar']},
        
        # Cycle Indicators
        {'name': 'STC', 'type': 'cycle',
         'params': {'period': {'min': 5, 'max': 20, 'default': 10},
                    'fast': {'min': 15, 'max': 35, 'default': 23},
                    'slow': {'min': 35, 'max': 70, 'default': 50}},
         'outputs': ['stc', 'stc_signal']},
        
        {'name': 'FisherTransform', 'type': 'cycle',
         'params': {'period': {'min': 5, 'max': 20, 'default': 10}},
         'outputs': ['fisher', 'fisher_signal']},
        
        {'name': 'Coppock', 'type': 'cycle',
         'params': {'wma_period': {'min': 5, 'max': 15, 'default': 10},
                    'roc1': {'min': 10, 'max': 20, 'default': 14},
                    'roc2': {'min': 8, 'max': 15, 'default': 11}},
         'outputs': ['coppock']},
        
        {'name': 'ElderRay', 'type': 'cycle',
         'params': {'period': {'min': 7, 'max': 20, 'default': 13}},
         'outputs': ['bull_power', 'bear_power', 'elder_force']},
        
        {'name': 'PGO', 'type': 'cycle',
         'params': {'period': {'min': 7, 'max': 25, 'default': 14}},
         'outputs': ['pgo']},
        
        {'name': 'WaveTrend', 'type': 'cycle',
         'params': {'channel': {'min': 5, 'max': 20, 'default': 10},
                    'average': {'min': 15, 'max': 30, 'default': 21}},
         'outputs': ['wt1', 'wt2', 'wt_diff']},
    ]
