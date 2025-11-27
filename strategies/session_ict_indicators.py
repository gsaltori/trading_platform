# strategies/session_ict_indicators.py
"""
Advanced Session and ICT (Inner Circle Trader) Indicators.

Includes:
- Trading Sessions (Asian, London, New York)
- Opening Range Breakout (ORB)
- ICT Concepts (FVG, Order Blocks, Liquidity, etc.)
- Market Structure
- Smart Money Concepts
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, time, timedelta

logger = logging.getLogger(__name__)


class SessionIndicators:
    """
    Trading Session Analysis.
    
    Calculates session ranges, breakouts, and statistics for:
    - Asian Session (Tokyo)
    - London Session
    - New York Session
    - Session Overlaps
    """
    
    # Default session times (UTC)
    SESSIONS = {
        'asian': {'start': time(0, 0), 'end': time(9, 0)},      # 00:00 - 09:00 UTC
        'london': {'start': time(7, 0), 'end': time(16, 0)},    # 07:00 - 16:00 UTC
        'newyork': {'start': time(12, 0), 'end': time(21, 0)},  # 12:00 - 21:00 UTC
        'london_ny_overlap': {'start': time(12, 0), 'end': time(16, 0)},  # Overlap
    }
    
    @classmethod
    def calculate_session_ranges(cls, df: pd.DataFrame, 
                                  session: str = 'asian',
                                  utc_offset: int = 0) -> pd.DataFrame:
        """
        Calculate session high/low ranges.
        
        Args:
            df: DataFrame with OHLCV data (index must be datetime)
            session: 'asian', 'london', 'newyork', or 'london_ny_overlap'
            utc_offset: Hours offset from UTC for your broker's time
        
        Returns:
            DataFrame with session_high, session_low, session_range columns
        """
        df = df.copy()
        
        session_config = cls.SESSIONS.get(session, cls.SESSIONS['asian'])
        start_time = session_config['start']
        end_time = session_config['end']
        
        # Adjust for UTC offset
        def adjust_time(t, offset):
            dt = datetime.combine(datetime.today(), t)
            dt = dt + timedelta(hours=offset)
            return dt.time()
        
        adj_start = adjust_time(start_time, utc_offset)
        adj_end = adjust_time(end_time, utc_offset)
        
        # Initialize columns
        df[f'{session}_high'] = np.nan
        df[f'{session}_low'] = np.nan
        df[f'{session}_range'] = np.nan
        df[f'{session}_mid'] = np.nan
        df[f'in_{session}'] = False
        
        # Group by date and calculate session ranges
        if hasattr(df.index, 'date'):
            for date in df.index.date:
                # Filter for this date
                day_mask = df.index.date == date
                day_data = df[day_mask]
                
                # Find session candles
                if adj_start < adj_end:
                    session_mask = (day_data.index.time >= adj_start) & (day_data.index.time < adj_end)
                else:
                    # Session crosses midnight
                    session_mask = (day_data.index.time >= adj_start) | (day_data.index.time < adj_end)
                
                session_data = day_data[session_mask]
                
                if len(session_data) > 0:
                    session_high = session_data['high'].max()
                    session_low = session_data['low'].min()
                    session_range = session_high - session_low
                    session_mid = (session_high + session_low) / 2
                    
                    # Apply to all candles after session end on this day
                    after_session = day_data.index.time >= adj_end
                    df.loc[day_data[after_session].index, f'{session}_high'] = session_high
                    df.loc[day_data[after_session].index, f'{session}_low'] = session_low
                    df.loc[day_data[after_session].index, f'{session}_range'] = session_range
                    df.loc[day_data[after_session].index, f'{session}_mid'] = session_mid
                    df.loc[session_data.index, f'in_{session}'] = True
        
        # Forward fill session values
        df[f'{session}_high'] = df[f'{session}_high'].ffill()
        df[f'{session}_low'] = df[f'{session}_low'].ffill()
        df[f'{session}_range'] = df[f'{session}_range'].ffill()
        df[f'{session}_mid'] = df[f'{session}_mid'].ffill()
        
        return df
    
    @classmethod
    def calculate_session_breakout(cls, df: pd.DataFrame,
                                    session: str = 'asian') -> pd.DataFrame:
        """
        Detect session range breakouts.
        
        Returns signals:
        - 1: Bullish breakout (price breaks above session high)
        - -1: Bearish breakout (price breaks below session low)
        - 0: No breakout
        """
        df = df.copy()
        
        # Ensure session ranges are calculated
        if f'{session}_high' not in df.columns:
            df = cls.calculate_session_ranges(df, session)
        
        # Detect breakouts
        df[f'{session}_breakout'] = 0
        
        # Bullish breakout
        bullish = (df['close'] > df[f'{session}_high']) & \
                  (df['close'].shift(1) <= df[f'{session}_high'].shift(1))
        df.loc[bullish, f'{session}_breakout'] = 1
        
        # Bearish breakout
        bearish = (df['close'] < df[f'{session}_low']) & \
                  (df['close'].shift(1) >= df[f'{session}_low'].shift(1))
        df.loc[bearish, f'{session}_breakout'] = -1
        
        # Breakout distance (how far price moved from range)
        df[f'{session}_breakout_distance'] = 0.0
        df.loc[bullish, f'{session}_breakout_distance'] = df['close'] - df[f'{session}_high']
        df.loc[bearish, f'{session}_breakout_distance'] = df[f'{session}_low'] - df['close']
        
        return df
    
    @classmethod
    def calculate_all_sessions(cls, df: pd.DataFrame, 
                                utc_offset: int = 0) -> pd.DataFrame:
        """Calculate ranges and breakouts for all sessions."""
        df = df.copy()
        
        for session in ['asian', 'london', 'newyork']:
            df = cls.calculate_session_ranges(df, session, utc_offset)
            df = cls.calculate_session_breakout(df, session)
        
        return df
    
    @classmethod
    def calculate_session_stats(cls, df: pd.DataFrame,
                                 session: str = 'asian',
                                 lookback: int = 20) -> pd.DataFrame:
        """
        Calculate session statistics.
        
        Returns:
        - Average session range
        - Range percentile
        - Session volatility
        """
        df = df.copy()
        
        if f'{session}_range' not in df.columns:
            df = cls.calculate_session_ranges(df, session)
        
        # Rolling statistics
        df[f'{session}_avg_range'] = df[f'{session}_range'].rolling(lookback).mean()
        df[f'{session}_range_std'] = df[f'{session}_range'].rolling(lookback).std()
        
        # Range percentile (is today's range larger or smaller than usual?)
        def percentile_rank(x):
            if len(x) < 2:
                return 50
            return (x.rank().iloc[-1] / len(x)) * 100
        
        df[f'{session}_range_percentile'] = df[f'{session}_range'].rolling(lookback).apply(
            percentile_rank, raw=False
        )
        
        return df


class OpeningRangeBreakout:
    """
    Opening Range Breakout (ORB) Strategy Indicators.
    
    Calculates:
    - Opening range (first N minutes)
    - ORB levels
    - Breakout signals
    - Extension targets
    """
    
    @staticmethod
    def calculate_orb(df: pd.DataFrame, 
                      minutes: int = 30,
                      session_start: time = time(9, 30)) -> pd.DataFrame:
        """
        Calculate Opening Range Breakout levels.
        
        Args:
            df: DataFrame with OHLCV data
            minutes: Opening range duration in minutes
            session_start: Market open time
        
        Returns:
            DataFrame with ORB levels and signals
        """
        df = df.copy()
        
        df['orb_high'] = np.nan
        df['orb_low'] = np.nan
        df['orb_range'] = np.nan
        df['orb_breakout'] = 0
        
        if hasattr(df.index, 'date'):
            for date in df.index.date:
                day_mask = df.index.date == date
                day_data = df[day_mask]
                
                # Find opening range candles
                orb_end_time = (datetime.combine(datetime.today(), session_start) + 
                               timedelta(minutes=minutes)).time()
                
                orb_mask = (day_data.index.time >= session_start) & \
                          (day_data.index.time < orb_end_time)
                orb_data = day_data[orb_mask]
                
                if len(orb_data) > 0:
                    orb_high = orb_data['high'].max()
                    orb_low = orb_data['low'].min()
                    orb_range = orb_high - orb_low
                    
                    # Apply to candles after ORB period
                    after_orb = day_data.index.time >= orb_end_time
                    df.loc[day_data[after_orb].index, 'orb_high'] = orb_high
                    df.loc[day_data[after_orb].index, 'orb_low'] = orb_low
                    df.loc[day_data[after_orb].index, 'orb_range'] = orb_range
        
        # Forward fill
        df['orb_high'] = df['orb_high'].ffill()
        df['orb_low'] = df['orb_low'].ffill()
        df['orb_range'] = df['orb_range'].ffill()
        
        # Breakout detection
        bullish = (df['close'] > df['orb_high']) & (df['close'].shift(1) <= df['orb_high'].shift(1))
        bearish = (df['close'] < df['orb_low']) & (df['close'].shift(1) >= df['orb_low'].shift(1))
        
        df.loc[bullish, 'orb_breakout'] = 1
        df.loc[bearish, 'orb_breakout'] = -1
        
        # Extension targets (1x, 1.5x, 2x range)
        df['orb_target_1'] = np.where(df['orb_breakout'] == 1, 
                                       df['orb_high'] + df['orb_range'],
                                       np.where(df['orb_breakout'] == -1,
                                                df['orb_low'] - df['orb_range'], np.nan))
        df['orb_target_1_5'] = np.where(df['orb_breakout'] == 1,
                                         df['orb_high'] + 1.5 * df['orb_range'],
                                         np.where(df['orb_breakout'] == -1,
                                                  df['orb_low'] - 1.5 * df['orb_range'], np.nan))
        df['orb_target_2'] = np.where(df['orb_breakout'] == 1,
                                       df['orb_high'] + 2 * df['orb_range'],
                                       np.where(df['orb_breakout'] == -1,
                                                df['orb_low'] - 2 * df['orb_range'], np.nan))
        
        return df


class ICTIndicators:
    """
    ICT (Inner Circle Trader) Concepts.
    
    Implements:
    - Fair Value Gaps (FVG) / Imbalances
    - Order Blocks (OB)
    - Breaker Blocks
    - Mitigation Blocks
    - Liquidity Pools (BSL/SSL)
    - Market Structure Shifts (MSS)
    - Change of Character (CHoCH)
    - Break of Structure (BOS)
    - Premium/Discount Zones
    - Optimal Trade Entry (OTE)
    - Killzones
    """
    
    @staticmethod
    def calculate_fair_value_gaps(df: pd.DataFrame, 
                                   min_gap_percent: float = 0.1) -> pd.DataFrame:
        """
        Identify Fair Value Gaps (Imbalances).
        
        A FVG occurs when there's a gap between:
        - Bullish FVG: Previous candle high and next candle low
        - Bearish FVG: Previous candle low and next candle high
        """
        df = df.copy()
        
        df['fvg_bullish'] = False
        df['fvg_bearish'] = False
        df['fvg_bullish_top'] = np.nan
        df['fvg_bullish_bottom'] = np.nan
        df['fvg_bearish_top'] = np.nan
        df['fvg_bearish_bottom'] = np.nan
        
        for i in range(2, len(df)):
            # Bullish FVG: gap between candle[i-2] high and candle[i] low
            prev_high = df['high'].iloc[i-2]
            curr_low = df['low'].iloc[i]
            
            if curr_low > prev_high:
                gap_size = (curr_low - prev_high) / df['close'].iloc[i] * 100
                if gap_size >= min_gap_percent:
                    df.iloc[i, df.columns.get_loc('fvg_bullish')] = True
                    df.iloc[i, df.columns.get_loc('fvg_bullish_top')] = curr_low
                    df.iloc[i, df.columns.get_loc('fvg_bullish_bottom')] = prev_high
            
            # Bearish FVG: gap between candle[i-2] low and candle[i] high
            prev_low = df['low'].iloc[i-2]
            curr_high = df['high'].iloc[i]
            
            if prev_low > curr_high:
                gap_size = (prev_low - curr_high) / df['close'].iloc[i] * 100
                if gap_size >= min_gap_percent:
                    df.iloc[i, df.columns.get_loc('fvg_bearish')] = True
                    df.iloc[i, df.columns.get_loc('fvg_bearish_top')] = prev_low
                    df.iloc[i, df.columns.get_loc('fvg_bearish_bottom')] = curr_high
        
        # Track unfilled FVGs
        df['fvg_bullish_unfilled'] = False
        df['fvg_bearish_unfilled'] = False
        
        # Forward track FVGs until filled
        bullish_fvg_active = []
        bearish_fvg_active = []
        
        for i in range(len(df)):
            # Check if price filled any active FVGs
            bullish_fvg_active = [
                fvg for fvg in bullish_fvg_active 
                if df['low'].iloc[i] > fvg['bottom']
            ]
            bearish_fvg_active = [
                fvg for fvg in bearish_fvg_active
                if df['high'].iloc[i] < fvg['top']
            ]
            
            # Add new FVGs
            if df['fvg_bullish'].iloc[i]:
                bullish_fvg_active.append({
                    'top': df['fvg_bullish_top'].iloc[i],
                    'bottom': df['fvg_bullish_bottom'].iloc[i]
                })
            
            if df['fvg_bearish'].iloc[i]:
                bearish_fvg_active.append({
                    'top': df['fvg_bearish_top'].iloc[i],
                    'bottom': df['fvg_bearish_bottom'].iloc[i]
                })
            
            df.iloc[i, df.columns.get_loc('fvg_bullish_unfilled')] = len(bullish_fvg_active) > 0
            df.iloc[i, df.columns.get_loc('fvg_bearish_unfilled')] = len(bearish_fvg_active) > 0
        
        return df
    
    @staticmethod
    def calculate_order_blocks(df: pd.DataFrame,
                                lookback: int = 10,
                                strength: int = 3) -> pd.DataFrame:
        """
        Identify Order Blocks.
        
        Order Block: The last opposing candle before a strong move.
        - Bullish OB: Last bearish candle before strong bullish move
        - Bearish OB: Last bullish candle before strong bearish move
        """
        df = df.copy()
        
        df['ob_bullish'] = False
        df['ob_bearish'] = False
        df['ob_bullish_top'] = np.nan
        df['ob_bullish_bottom'] = np.nan
        df['ob_bearish_top'] = np.nan
        df['ob_bearish_bottom'] = np.nan
        
        # Identify strong moves
        df['body'] = df['close'] - df['open']
        df['body_abs'] = df['body'].abs()
        df['avg_body'] = df['body_abs'].rolling(lookback).mean()
        
        for i in range(strength, len(df)):
            # Check for strong bullish move (next 'strength' candles all bullish and large)
            future_candles = df.iloc[i:i+strength] if i+strength <= len(df) else df.iloc[i:]
            
            if len(future_candles) >= strength:
                all_bullish = all(future_candles['close'] > future_candles['open'])
                strong_move = future_candles['body_abs'].sum() > df['avg_body'].iloc[i] * strength * 1.5
                
                if all_bullish and strong_move:
                    # Find last bearish candle before this move
                    for j in range(i-1, max(0, i-lookback), -1):
                        if df['close'].iloc[j] < df['open'].iloc[j]:
                            df.iloc[j, df.columns.get_loc('ob_bullish')] = True
                            df.iloc[j, df.columns.get_loc('ob_bullish_top')] = df['high'].iloc[j]
                            df.iloc[j, df.columns.get_loc('ob_bullish_bottom')] = df['low'].iloc[j]
                            break
                
                # Check for strong bearish move
                all_bearish = all(future_candles['close'] < future_candles['open'])
                
                if all_bearish and strong_move:
                    # Find last bullish candle before this move
                    for j in range(i-1, max(0, i-lookback), -1):
                        if df['close'].iloc[j] > df['open'].iloc[j]:
                            df.iloc[j, df.columns.get_loc('ob_bearish')] = True
                            df.iloc[j, df.columns.get_loc('ob_bearish_top')] = df['high'].iloc[j]
                            df.iloc[j, df.columns.get_loc('ob_bearish_bottom')] = df['low'].iloc[j]
                            break
        
        # Clean up
        df.drop(['body', 'body_abs', 'avg_body'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_liquidity_levels(df: pd.DataFrame,
                                    lookback: int = 20,
                                    equal_threshold: float = 0.0005) -> pd.DataFrame:
        """
        Identify Liquidity Pools (Equal Highs/Lows).
        
        - BSL (Buy Side Liquidity): Equal highs where stops are placed
        - SSL (Sell Side Liquidity): Equal lows where stops are placed
        """
        df = df.copy()
        
        df['bsl'] = False  # Buy Side Liquidity (equal highs)
        df['ssl'] = False  # Sell Side Liquidity (equal lows)
        df['bsl_level'] = np.nan
        df['ssl_level'] = np.nan
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Find equal highs (BSL)
            equal_highs = window[
                abs(window['high'] - current_high) / current_high < equal_threshold
            ]
            if len(equal_highs) >= 2:
                df.iloc[i, df.columns.get_loc('bsl')] = True
                df.iloc[i, df.columns.get_loc('bsl_level')] = current_high
            
            # Find equal lows (SSL)
            equal_lows = window[
                abs(window['low'] - current_low) / current_low < equal_threshold
            ]
            if len(equal_lows) >= 2:
                df.iloc[i, df.columns.get_loc('ssl')] = True
                df.iloc[i, df.columns.get_loc('ssl_level')] = current_low
        
        return df
    
    @staticmethod
    def calculate_market_structure(df: pd.DataFrame,
                                    swing_length: int = 5) -> pd.DataFrame:
        """
        Analyze Market Structure.
        
        Identifies:
        - Swing Highs/Lows
        - Higher Highs (HH), Higher Lows (HL)
        - Lower Highs (LH), Lower Lows (LL)
        - Break of Structure (BOS)
        - Change of Character (CHoCH)
        - Market Structure Shift (MSS)
        """
        df = df.copy()
        
        # Identify swing points
        df['swing_high'] = False
        df['swing_low'] = False
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        for i in range(swing_length, len(df) - swing_length):
            # Swing High: highest point in window
            window_highs = df['high'].iloc[i-swing_length:i+swing_length+1]
            if df['high'].iloc[i] == window_highs.max():
                df.iloc[i, df.columns.get_loc('swing_high')] = True
                df.iloc[i, df.columns.get_loc('swing_high_price')] = df['high'].iloc[i]
            
            # Swing Low: lowest point in window
            window_lows = df['low'].iloc[i-swing_length:i+swing_length+1]
            if df['low'].iloc[i] == window_lows.min():
                df.iloc[i, df.columns.get_loc('swing_low')] = True
                df.iloc[i, df.columns.get_loc('swing_low_price')] = df['low'].iloc[i]
        
        # Forward fill swing prices for comparison
        df['last_swing_high'] = df['swing_high_price'].ffill()
        df['last_swing_low'] = df['swing_low_price'].ffill()
        df['prev_swing_high'] = df['last_swing_high'].shift(1)
        df['prev_swing_low'] = df['last_swing_low'].shift(1)
        
        # Identify structure
        df['hh'] = (df['swing_high']) & (df['swing_high_price'] > df['prev_swing_high'])
        df['hl'] = (df['swing_low']) & (df['swing_low_price'] > df['prev_swing_low'])
        df['lh'] = (df['swing_high']) & (df['swing_high_price'] < df['prev_swing_high'])
        df['ll'] = (df['swing_low']) & (df['swing_low_price'] < df['prev_swing_low'])
        
        # Market Structure
        df['structure'] = 'ranging'
        
        # Bullish structure: HH and HL
        bullish_mask = df['hh'] | df['hl']
        # Bearish structure: LH and LL
        bearish_mask = df['lh'] | df['ll']
        
        df.loc[bullish_mask, 'structure'] = 'bullish'
        df.loc[bearish_mask, 'structure'] = 'bearish'
        
        # Break of Structure (BOS) - continuation
        df['bos_bullish'] = (df['close'] > df['last_swing_high']) & \
                           (df['close'].shift(1) <= df['last_swing_high'].shift(1))
        df['bos_bearish'] = (df['close'] < df['last_swing_low']) & \
                           (df['close'].shift(1) >= df['last_swing_low'].shift(1))
        
        # Change of Character (CHoCH) - reversal
        # Bullish CHoCH: Break above swing high after bearish structure
        # Bearish CHoCH: Break below swing low after bullish structure
        df['choch_bullish'] = False
        df['choch_bearish'] = False
        
        structure_series = df['structure'].shift(1)
        df.loc[(df['bos_bullish']) & (structure_series == 'bearish'), 'choch_bullish'] = True
        df.loc[(df['bos_bearish']) & (structure_series == 'bullish'), 'choch_bearish'] = True
        
        # Market Structure Shift (combination)
        df['mss'] = 0
        df.loc[df['choch_bullish'], 'mss'] = 1
        df.loc[df['choch_bearish'], 'mss'] = -1
        
        # Clean up
        df.drop(['prev_swing_high', 'prev_swing_low'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_premium_discount(df: pd.DataFrame,
                                    lookback: int = 20) -> pd.DataFrame:
        """
        Calculate Premium and Discount Zones.
        
        Based on the range between swing high and swing low:
        - Premium Zone: Upper 50% (overvalued - look for sells)
        - Discount Zone: Lower 50% (undervalued - look for buys)
        - Equilibrium: 50% level
        """
        df = df.copy()
        
        # Rolling high and low
        df['range_high'] = df['high'].rolling(lookback).max()
        df['range_low'] = df['low'].rolling(lookback).min()
        df['range'] = df['range_high'] - df['range_low']
        
        # Equilibrium (50%)
        df['equilibrium'] = (df['range_high'] + df['range_low']) / 2
        
        # Premium zone levels (50% - 100%)
        df['premium_start'] = df['equilibrium']  # 50%
        df['premium_70'] = df['range_low'] + 0.7 * df['range']
        df['premium_79'] = df['range_low'] + 0.79 * df['range']  # OTE level
        
        # Discount zone levels (0% - 50%)
        df['discount_end'] = df['equilibrium']  # 50%
        df['discount_30'] = df['range_low'] + 0.3 * df['range']
        df['discount_21'] = df['range_low'] + 0.21 * df['range']  # OTE level
        
        # Current zone
        df['zone'] = 'equilibrium'
        df.loc[df['close'] > df['equilibrium'], 'zone'] = 'premium'
        df.loc[df['close'] < df['equilibrium'], 'zone'] = 'discount'
        
        # Zone percentage
        df['zone_pct'] = (df['close'] - df['range_low']) / df['range'] * 100
        
        return df
    
    @staticmethod
    def calculate_optimal_trade_entry(df: pd.DataFrame,
                                       fib_levels: List[float] = [0.618, 0.705, 0.79]) -> pd.DataFrame:
        """
        Calculate Optimal Trade Entry (OTE) Zones.
        
        OTE is typically between 61.8% and 79% Fibonacci retracement.
        """
        df = df.copy()
        
        # Calculate market structure first
        if 'swing_high_price' not in df.columns:
            df = ICTIndicators.calculate_market_structure(df)
        
        # Get last significant swing points
        df['last_sh'] = df['swing_high_price'].ffill()
        df['last_sl'] = df['swing_low_price'].ffill()
        
        # For bullish OTE (buying in retracement)
        swing_range = df['last_sh'] - df['last_sl']
        
        for level in fib_levels:
            # Bullish OTE levels (retracement from high)
            df[f'ote_bullish_{int(level*100)}'] = df['last_sh'] - (swing_range * level)
            # Bearish OTE levels (retracement from low)
            df[f'ote_bearish_{int(level*100)}'] = df['last_sl'] + (swing_range * level)
        
        # Check if price is in OTE zone
        df['in_bullish_ote'] = (df['close'] >= df['ote_bullish_79']) & \
                               (df['close'] <= df['ote_bullish_62'])
        df['in_bearish_ote'] = (df['close'] <= df['ote_bearish_79']) & \
                               (df['close'] >= df['ote_bearish_62'])
        
        return df
    
    @staticmethod
    def calculate_killzones(df: pd.DataFrame,
                            utc_offset: int = 0) -> pd.DataFrame:
        """
        Identify ICT Killzones.
        
        Killzones are optimal trading times:
        - Asian Killzone: 20:00 - 00:00 (NY time)
        - London Killzone: 02:00 - 05:00 (NY time)
        - New York AM Killzone: 07:00 - 10:00 (NY time)
        - New York PM Killzone: 13:30 - 16:00 (NY time)
        """
        df = df.copy()
        
        # Killzone times (adjusted from NY time to UTC)
        killzones = {
            'asian_kz': {'start': time(1, 0), 'end': time(5, 0)},      # ~20:00-00:00 NY
            'london_kz': {'start': time(7, 0), 'end': time(10, 0)},    # ~02:00-05:00 NY
            'nyam_kz': {'start': time(12, 0), 'end': time(15, 0)},     # ~07:00-10:00 NY
            'nypm_kz': {'start': time(18, 30), 'end': time(21, 0)},    # ~13:30-16:00 NY
        }
        
        for kz_name, kz_times in killzones.items():
            df[kz_name] = False
            
            if hasattr(df.index, 'time'):
                start = kz_times['start']
                end = kz_times['end']
                
                if start < end:
                    mask = (df.index.time >= start) & (df.index.time < end)
                else:
                    mask = (df.index.time >= start) | (df.index.time < end)
                
                df.loc[mask, kz_name] = True
        
        # Combined killzone indicator
        df['in_killzone'] = df['asian_kz'] | df['london_kz'] | df['nyam_kz'] | df['nypm_kz']
        
        return df


class SmartMoneyIndicators:
    """
    Additional Smart Money Concepts.
    """
    
    @staticmethod
    def calculate_displacement(df: pd.DataFrame,
                                threshold_mult: float = 2.0,
                                lookback: int = 14) -> pd.DataFrame:
        """
        Identify Displacement (Strong impulsive moves).
        
        Displacement occurs when price makes a strong, aggressive move
        typically with large-bodied candles.
        """
        df = df.copy()
        
        # Body size
        df['body_size'] = abs(df['close'] - df['open'])
        df['avg_body'] = df['body_size'].rolling(lookback).mean()
        
        # Displacement detection
        df['displacement_bullish'] = (df['close'] > df['open']) & \
                                     (df['body_size'] > df['avg_body'] * threshold_mult)
        df['displacement_bearish'] = (df['close'] < df['open']) & \
                                     (df['body_size'] > df['avg_body'] * threshold_mult)
        
        # Displacement strength
        df['displacement_strength'] = df['body_size'] / df['avg_body']
        
        df.drop(['body_size', 'avg_body'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_inducement(df: pd.DataFrame,
                              lookback: int = 10) -> pd.DataFrame:
        """
        Identify Inducement levels.
        
        Inducement: Minor swing points that trap retail traders
        before the real move happens.
        """
        df = df.copy()
        
        # Minor swings (smaller lookback than main structure)
        minor_lookback = max(2, lookback // 3)
        
        df['minor_high'] = df['high'].rolling(minor_lookback * 2 + 1, center=True).max()
        df['minor_low'] = df['low'].rolling(minor_lookback * 2 + 1, center=True).min()
        
        df['inducement_high'] = df['high'] == df['minor_high']
        df['inducement_low'] = df['low'] == df['minor_low']
        
        df.drop(['minor_high', 'minor_low'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_liquidity_sweep(df: pd.DataFrame,
                                   lookback: int = 20) -> pd.DataFrame:
        """
        Identify Liquidity Sweeps (Stop Hunts).
        
        A liquidity sweep occurs when price briefly breaks a level
        (taking out stops) then reverses.
        """
        df = df.copy()
        
        # Recent highs and lows
        df['recent_high'] = df['high'].rolling(lookback).max().shift(1)
        df['recent_low'] = df['low'].rolling(lookback).min().shift(1)
        
        # Sweep detection
        # Bullish sweep: Price breaks below recent low then closes above
        df['sweep_bullish'] = (df['low'] < df['recent_low']) & (df['close'] > df['recent_low'])
        
        # Bearish sweep: Price breaks above recent high then closes below
        df['sweep_bearish'] = (df['high'] > df['recent_high']) & (df['close'] < df['recent_high'])
        
        df.drop(['recent_high', 'recent_low'], axis=1, inplace=True)
        
        return df
    
    @staticmethod  
    def calculate_breaker_block(df: pd.DataFrame,
                                 lookback: int = 10) -> pd.DataFrame:
        """
        Identify Breaker Blocks.
        
        A breaker block is a failed order block that becomes 
        support/resistance after being broken.
        """
        df = df.copy()
        
        # First calculate order blocks
        if 'ob_bullish' not in df.columns:
            df = ICTIndicators.calculate_order_blocks(df, lookback)
        
        df['breaker_bullish'] = False
        df['breaker_bearish'] = False
        
        # Track order blocks and check if they get broken
        active_bullish_obs = []
        active_bearish_obs = []
        
        for i in range(len(df)):
            # Add new order blocks
            if df['ob_bullish'].iloc[i]:
                active_bullish_obs.append({
                    'idx': i,
                    'top': df['ob_bullish_top'].iloc[i],
                    'bottom': df['ob_bullish_bottom'].iloc[i],
                    'broken': False
                })
            
            if df['ob_bearish'].iloc[i]:
                active_bearish_obs.append({
                    'idx': i,
                    'top': df['ob_bearish_top'].iloc[i],
                    'bottom': df['ob_bearish_bottom'].iloc[i],
                    'broken': False
                })
            
            # Check if any OB gets broken (becomes a breaker)
            for ob in active_bullish_obs:
                if not ob['broken'] and df['close'].iloc[i] < ob['bottom']:
                    ob['broken'] = True
                    df.iloc[i, df.columns.get_loc('breaker_bearish')] = True
            
            for ob in active_bearish_obs:
                if not ob['broken'] and df['close'].iloc[i] > ob['top']:
                    ob['broken'] = True
                    df.iloc[i, df.columns.get_loc('breaker_bullish')] = True
        
        return df


class AdditionalIndicators:
    """
    Additional Technical Indicators.
    """
    
    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame,
                                pivot_type: str = 'standard') -> pd.DataFrame:
        """
        Calculate Pivot Points.
        
        Types: standard, fibonacci, woodie, camarilla, demark
        """
        df = df.copy()
        
        # Use previous day's data
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        
        # Standard Pivot Point
        df['pivot'] = (df['prev_high'] + df['prev_low'] + df['prev_close']) / 3
        
        if pivot_type == 'standard':
            df['r1'] = 2 * df['pivot'] - df['prev_low']
            df['r2'] = df['pivot'] + (df['prev_high'] - df['prev_low'])
            df['r3'] = df['prev_high'] + 2 * (df['pivot'] - df['prev_low'])
            df['s1'] = 2 * df['pivot'] - df['prev_high']
            df['s2'] = df['pivot'] - (df['prev_high'] - df['prev_low'])
            df['s3'] = df['prev_low'] - 2 * (df['prev_high'] - df['pivot'])
        
        elif pivot_type == 'fibonacci':
            range_hl = df['prev_high'] - df['prev_low']
            df['r1'] = df['pivot'] + 0.382 * range_hl
            df['r2'] = df['pivot'] + 0.618 * range_hl
            df['r3'] = df['pivot'] + 1.000 * range_hl
            df['s1'] = df['pivot'] - 0.382 * range_hl
            df['s2'] = df['pivot'] - 0.618 * range_hl
            df['s3'] = df['pivot'] - 1.000 * range_hl
        
        elif pivot_type == 'camarilla':
            range_hl = df['prev_high'] - df['prev_low']
            df['r1'] = df['prev_close'] + range_hl * 1.1 / 12
            df['r2'] = df['prev_close'] + range_hl * 1.1 / 6
            df['r3'] = df['prev_close'] + range_hl * 1.1 / 4
            df['r4'] = df['prev_close'] + range_hl * 1.1 / 2
            df['s1'] = df['prev_close'] - range_hl * 1.1 / 12
            df['s2'] = df['prev_close'] - range_hl * 1.1 / 6
            df['s3'] = df['prev_close'] - range_hl * 1.1 / 4
            df['s4'] = df['prev_close'] - range_hl * 1.1 / 2
        
        df.drop(['prev_high', 'prev_low', 'prev_close'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame,
                                  bins: int = 20,
                                  lookback: int = 100) -> pd.DataFrame:
        """
        Calculate Volume Profile (simplified).
        
        Identifies:
        - Point of Control (POC): Price with highest volume
        - Value Area High (VAH)
        - Value Area Low (VAL)
        """
        df = df.copy()
        
        df['poc'] = np.nan
        df['vah'] = np.nan
        df['val'] = np.nan
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]
            
            # Create price bins
            price_min = window['low'].min()
            price_max = window['high'].max()
            bin_edges = np.linspace(price_min, price_max, bins + 1)
            
            # Assign volume to bins
            volume_by_price = np.zeros(bins)
            
            for _, row in window.iterrows():
                # Distribute candle volume across its range
                low_bin = np.searchsorted(bin_edges, row['low']) - 1
                high_bin = np.searchsorted(bin_edges, row['high']) - 1
                
                low_bin = max(0, min(bins - 1, low_bin))
                high_bin = max(0, min(bins - 1, high_bin))
                
                if high_bin >= low_bin:
                    vol_per_bin = row['volume'] / (high_bin - low_bin + 1)
                    volume_by_price[low_bin:high_bin + 1] += vol_per_bin
            
            # POC: highest volume bin
            poc_bin = np.argmax(volume_by_price)
            df.iloc[i, df.columns.get_loc('poc')] = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2
            
            # Value Area (70% of volume)
            total_vol = volume_by_price.sum()
            target_vol = total_vol * 0.7
            
            sorted_bins = np.argsort(volume_by_price)[::-1]
            cumsum = 0
            value_area_bins = []
            
            for bin_idx in sorted_bins:
                cumsum += volume_by_price[bin_idx]
                value_area_bins.append(bin_idx)
                if cumsum >= target_vol:
                    break
            
            if value_area_bins:
                df.iloc[i, df.columns.get_loc('vah')] = bin_edges[max(value_area_bins) + 1]
                df.iloc[i, df.columns.get_loc('val')] = bin_edges[min(value_area_bins)]
        
        return df
    
    @staticmethod
    def calculate_vwap_bands(df: pd.DataFrame,
                              std_mult: List[float] = [1, 2, 3]) -> pd.DataFrame:
        """
        Calculate VWAP with standard deviation bands.
        """
        df = df.copy()
        
        # Cumulative VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Rolling VWAP for bands
        period = 20
        df['vwap_rolling'] = (typical_price * df['volume']).rolling(period).sum() / \
                             df['volume'].rolling(period).sum()
        
        # Standard deviation
        df['vwap_std'] = typical_price.rolling(period).std()
        
        for mult in std_mult:
            df[f'vwap_upper_{mult}'] = df['vwap_rolling'] + mult * df['vwap_std']
            df[f'vwap_lower_{mult}'] = df['vwap_rolling'] - mult * df['vwap_std']
        
        df.drop('vwap_std', axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_range_filter(df: pd.DataFrame,
                                period: int = 100,
                                multiplier: float = 2.6) -> pd.DataFrame:
        """
        Calculate Range Filter indicator.
        
        Trend-following indicator that filters noise.
        """
        df = df.copy()
        
        # Average Range
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].ewm(span=period, adjust=False).mean() * multiplier
        
        # Range Filter
        df['rf_upper'] = df['close'].rolling(1).max()
        df['rf_lower'] = df['close'].rolling(1).min()
        
        rf = pd.Series(index=df.index, dtype=float)
        rf.iloc[0] = df['close'].iloc[0]
        
        for i in range(1, len(df)):
            prev_rf = rf.iloc[i-1]
            curr_close = df['close'].iloc[i]
            avg_range = df['avg_range'].iloc[i]
            
            if curr_close > prev_rf:
                rf.iloc[i] = max(prev_rf, curr_close - avg_range)
            else:
                rf.iloc[i] = min(prev_rf, curr_close + avg_range)
        
        df['range_filter'] = rf
        df['rf_trend'] = np.where(df['close'] > df['range_filter'], 1, -1)
        
        df.drop(['range', 'avg_range', 'rf_upper', 'rf_lower'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_chandelier_exit(df: pd.DataFrame,
                                   period: int = 22,
                                   multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculate Chandelier Exit.
        
        Trailing stop based on ATR.
        """
        df = df.copy()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        # Chandelier Exit
        df['chandelier_long'] = df['high'].rolling(period).max() - multiplier * atr
        df['chandelier_short'] = df['low'].rolling(period).min() + multiplier * atr
        
        # Signal
        df['chandelier_signal'] = np.where(df['close'] > df['chandelier_long'], 1,
                                           np.where(df['close'] < df['chandelier_short'], -1, 0))
        
        return df
