# strategies/indicator_integration.py
"""
Indicator Integration Module.

Integrates extended indicators into the main IndicatorLibrary and IndicatorCalculator.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .auto_generator import IndicatorLibrary, IndicatorCalculator, IndicatorConfig, IndicatorType
from .extended_indicators import ExtendedIndicatorCalculator, get_extended_indicator_configs
from .session_ict_indicators import (
    SessionIndicators, 
    OpeningRangeBreakout, 
    ICTIndicators, 
    SmartMoneyIndicators,
    AdditionalIndicators
)

logger = logging.getLogger(__name__)


class ExtendedIndicatorLibrary(IndicatorLibrary):
    """
    Extended Indicator Library with 70+ indicators.
    
    Includes all base indicators plus extended indicators.
    """
    
    def __init__(self):
        super().__init__()
        self._register_extended_indicators()
        self._register_cycle_indicators()
        self._register_session_indicators()
        self._register_ict_indicators()
        self._register_smart_money_indicators()
        self._register_additional_indicators()
    
    def _register_extended_indicators(self):
        """Register all extended indicators."""
        
        # TREND INDICATORS
        self.register(IndicatorConfig(
            name="VWMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 100, 'default': 20}},
            output_columns=['vwma_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="HMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 100, 'default': 20}},
            output_columns=['hma_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="KAMA",
            indicator_type=IndicatorType.TREND,
            parameters={
                'period': {'min': 5, 'max': 50, 'default': 10},
                'fast_period': {'min': 2, 'max': 5, 'default': 2},
                'slow_period': {'min': 20, 'max': 50, 'default': 30}
            },
            output_columns=['kama_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="ZLEMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 100, 'default': 20}},
            output_columns=['zlema_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="T3",
            indicator_type=IndicatorType.TREND,
            parameters={
                'period': {'min': 3, 'max': 20, 'default': 5},
                'v_factor': {'min': 0.5, 'max': 0.9, 'default': 0.7}
            },
            output_columns=['t3_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="PSAR",
            indicator_type=IndicatorType.TREND,
            parameters={
                'af_start': {'min': 0.01, 'max': 0.05, 'default': 0.02},
                'af_step': {'min': 0.01, 'max': 0.05, 'default': 0.02},
                'af_max': {'min': 0.1, 'max': 0.3, 'default': 0.2}
            },
            output_columns=['psar', 'psar_bull', 'psar_bear', 'psar_direction']
        ))
        
        self.register(IndicatorConfig(
            name="Aroon",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 10, 'max': 50, 'default': 25}},
            output_columns=['aroon_up', 'aroon_down', 'aroon_oscillator']
        ))
        
        self.register(IndicatorConfig(
            name="VIDYA",
            indicator_type=IndicatorType.TREND,
            parameters={
                'period': {'min': 7, 'max': 30, 'default': 14},
                'cmo_period': {'min': 5, 'max': 15, 'default': 9}
            },
            output_columns=['vidya_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="Vortex",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 7, 'max': 30, 'default': 14}},
            output_columns=['vortex_plus', 'vortex_minus', 'vortex_diff']
        ))
        
        self.register(IndicatorConfig(
            name="DPO",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 10, 'max': 30, 'default': 20}},
            output_columns=['dpo_{period}']
        ))
        
        # MOMENTUM INDICATORS
        self.register(IndicatorConfig(
            name="TSI",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'long_period': {'min': 15, 'max': 35, 'default': 25},
                'short_period': {'min': 7, 'max': 20, 'default': 13},
                'signal_period': {'min': 7, 'max': 20, 'default': 13}
            },
            output_columns=['tsi', 'tsi_signal', 'tsi_histogram']
        ))
        
        self.register(IndicatorConfig(
            name="UltimateOscillator",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'period1': {'min': 5, 'max': 10, 'default': 7},
                'period2': {'min': 10, 'max': 20, 'default': 14},
                'period3': {'min': 20, 'max': 40, 'default': 28}
            },
            output_columns=['ultimate_oscillator']
        ))
        
        self.register(IndicatorConfig(
            name="AwesomeOscillator",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'fast': {'min': 3, 'max': 10, 'default': 5},
                'slow': {'min': 20, 'max': 50, 'default': 34}
            },
            output_columns=['ao', 'ao_color']
        ))
        
        self.register(IndicatorConfig(
            name="TRIX",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'period': {'min': 10, 'max': 25, 'default': 15},
                'signal': {'min': 5, 'max': 15, 'default': 9}
            },
            output_columns=['trix', 'trix_signal', 'trix_histogram']
        ))
        
        self.register(IndicatorConfig(
            name="KST",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={},
            output_columns=['kst', 'kst_signal', 'kst_histogram']
        ))
        
        self.register(IndicatorConfig(
            name="PPO",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'fast': {'min': 8, 'max': 16, 'default': 12},
                'slow': {'min': 20, 'max': 35, 'default': 26},
                'signal': {'min': 5, 'max': 15, 'default': 9}
            },
            output_columns=['ppo', 'ppo_signal', 'ppo_histogram']
        ))
        
        self.register(IndicatorConfig(
            name="PVO",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'fast': {'min': 8, 'max': 16, 'default': 12},
                'slow': {'min': 20, 'max': 35, 'default': 26},
                'signal': {'min': 5, 'max': 15, 'default': 9}
            },
            output_columns=['pvo', 'pvo_signal', 'pvo_histogram']
        ))
        
        self.register(IndicatorConfig(
            name="DMI",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'period': {'min': 7, 'max': 30, 'default': 14},
                'adx_smooth': {'min': 7, 'max': 21, 'default': 14}
            },
            output_columns=['plus_di', 'minus_di', 'dx', 'adx_enhanced', 'dmi_diff']
        ))
        
        self.register(IndicatorConfig(
            name="RVI_Momentum",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 5, 'max': 20, 'default': 10}},
            output_columns=['rvi', 'rvi_signal']
        ))
        
        self.register(IndicatorConfig(
            name="CMO",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 7, 'max': 25, 'default': 14}},
            output_columns=['cmo']
        ))
        
        # VOLATILITY INDICATORS
        self.register(IndicatorConfig(
            name="ChaikinVolatility",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                'ema_period': {'min': 5, 'max': 20, 'default': 10},
                'roc_period': {'min': 5, 'max': 20, 'default': 10}
            },
            output_columns=['chaikin_volatility']
        ))
        
        self.register(IndicatorConfig(
            name="HistoricalVolatility",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={'period': {'min': 10, 'max': 50, 'default': 20}},
            output_columns=['historical_volatility']
        ))
        
        self.register(IndicatorConfig(
            name="MassIndex",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                'ema_period': {'min': 5, 'max': 15, 'default': 9},
                'sum_period': {'min': 15, 'max': 35, 'default': 25}
            },
            output_columns=['mass_index']
        ))
        
        self.register(IndicatorConfig(
            name="NATR",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={'period': {'min': 7, 'max': 25, 'default': 14}},
            output_columns=['natr']
        ))
        
        self.register(IndicatorConfig(
            name="UlcerIndex",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={'period': {'min': 7, 'max': 25, 'default': 14}},
            output_columns=['ulcer_index']
        ))
        
        self.register(IndicatorConfig(
            name="ATRBands",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                'period': {'min': 7, 'max': 25, 'default': 14},
                'multiplier': {'min': 1.0, 'max': 4.0, 'default': 2.0}
            },
            output_columns=['atr_band_upper', 'atr_band_middle', 'atr_band_lower']
        ))
        
        self.register(IndicatorConfig(
            name="RVI_Volatility",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={'period': {'min': 5, 'max': 20, 'default': 10}},
            output_columns=['rvi_volatility']
        ))
        
        # VOLUME INDICATORS
        self.register(IndicatorConfig(
            name="VWAPBands",
            indicator_type=IndicatorType.VOLUME,
            parameters={'std_mult': {'min': 1.0, 'max': 3.0, 'default': 2.0}},
            output_columns=['vwap', 'vwap_upper', 'vwap_lower']
        ))
        
        self.register(IndicatorConfig(
            name="PVT",
            indicator_type=IndicatorType.VOLUME,
            parameters={},
            output_columns=['pvt']
        ))
        
        self.register(IndicatorConfig(
            name="NVI",
            indicator_type=IndicatorType.VOLUME,
            parameters={},
            output_columns=['nvi', 'nvi_signal']
        ))
        
        self.register(IndicatorConfig(
            name="PVI",
            indicator_type=IndicatorType.VOLUME,
            parameters={},
            output_columns=['pvi', 'pvi_signal']
        ))
        
        self.register(IndicatorConfig(
            name="EOM",
            indicator_type=IndicatorType.VOLUME,
            parameters={'period': {'min': 7, 'max': 25, 'default': 14}},
            output_columns=['eom']
        ))
        
        self.register(IndicatorConfig(
            name="ForceIndex",
            indicator_type=IndicatorType.VOLUME,
            parameters={'period': {'min': 7, 'max': 25, 'default': 13}},
            output_columns=['force_index']
        ))
        
        self.register(IndicatorConfig(
            name="MFI_Enhanced",
            indicator_type=IndicatorType.VOLUME,
            parameters={'period': {'min': 7, 'max': 25, 'default': 14}},
            output_columns=['mfi_enhanced', 'mfi_signal']
        ))
        
        self.register(IndicatorConfig(
            name="Klinger",
            indicator_type=IndicatorType.VOLUME,
            parameters={
                'fast': {'min': 25, 'max': 45, 'default': 34},
                'slow': {'min': 45, 'max': 70, 'default': 55},
                'signal': {'min': 7, 'max': 20, 'default': 13}
            },
            output_columns=['klinger', 'klinger_signal']
        ))
        
        # PRICE ACTION INDICATORS
        self.register(IndicatorConfig(
            name="HHLL",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'period': {'min': 10, 'max': 50, 'default': 20}},
            output_columns=['higher_high', 'lower_low', 'hh_count', 'll_count', 'trend_strength']
        ))
        
        self.register(IndicatorConfig(
            name="ZigZag",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'threshold': {'min': 1.0, 'max': 10.0, 'default': 5.0}},
            output_columns=['zigzag']
        ))
        
        self.register(IndicatorConfig(
            name="ADR",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'period': {'min': 7, 'max': 30, 'default': 14}},
            output_columns=['adr', 'adr_percent']
        ))
        
        self.register(IndicatorConfig(
            name="CandleBody",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={},
            output_columns=['body_size', 'body_percent', 'upper_wick_percent', 'lower_wick_percent', 'is_bullish']
        ))
        
        self.register(IndicatorConfig(
            name="InsideOutsideBar",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={},
            output_columns=['inside_bar', 'outside_bar']
        ))
    
    def _register_cycle_indicators(self):
        """Register cycle indicators (new category)."""
        
        # Add CYCLE type if not exists (we'll handle this gracefully)
        self.register(IndicatorConfig(
            name="STC",
            indicator_type=IndicatorType.MOMENTUM,  # Using MOMENTUM as fallback
            parameters={
                'period': {'min': 5, 'max': 20, 'default': 10},
                'fast': {'min': 15, 'max': 35, 'default': 23},
                'slow': {'min': 35, 'max': 70, 'default': 50}
            },
            output_columns=['stc', 'stc_signal']
        ))
        
        self.register(IndicatorConfig(
            name="FisherTransform",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 5, 'max': 20, 'default': 10}},
            output_columns=['fisher', 'fisher_signal']
        ))
        
        self.register(IndicatorConfig(
            name="Coppock",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'wma_period': {'min': 5, 'max': 15, 'default': 10},
                'roc1': {'min': 10, 'max': 20, 'default': 14},
                'roc2': {'min': 8, 'max': 15, 'default': 11}
            },
            output_columns=['coppock']
        ))
        
        self.register(IndicatorConfig(
            name="ElderRay",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 7, 'max': 20, 'default': 13}},
            output_columns=['bull_power', 'bear_power', 'elder_force']
        ))
        
        self.register(IndicatorConfig(
            name="PGO",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 7, 'max': 25, 'default': 14}},
            output_columns=['pgo']
        ))
        
        self.register(IndicatorConfig(
            name="WaveTrend",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'channel': {'min': 5, 'max': 20, 'default': 10},
                'average': {'min': 15, 'max': 30, 'default': 21}
            },
            output_columns=['wt1', 'wt2', 'wt_diff']
        ))
    
    def _register_session_indicators(self):
        """Register session-based indicators."""
        
        # SESSION INDICATORS
        self.register(IndicatorConfig(
            name="AsianSession",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'utc_offset': {'min': -12, 'max': 12, 'default': 0}},
            output_columns=['asian_high', 'asian_low', 'asian_range', 'asian_mid', 'in_asian', 'asian_breakout']
        ))
        
        self.register(IndicatorConfig(
            name="LondonSession",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'utc_offset': {'min': -12, 'max': 12, 'default': 0}},
            output_columns=['london_high', 'london_low', 'london_range', 'london_mid', 'in_london', 'london_breakout']
        ))
        
        self.register(IndicatorConfig(
            name="NewYorkSession",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'utc_offset': {'min': -12, 'max': 12, 'default': 0}},
            output_columns=['newyork_high', 'newyork_low', 'newyork_range', 'newyork_mid', 'in_newyork', 'newyork_breakout']
        ))
        
        self.register(IndicatorConfig(
            name="AllSessions",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'utc_offset': {'min': -12, 'max': 12, 'default': 0}},
            output_columns=['asian_high', 'asian_low', 'asian_breakout', 'london_high', 'london_low', 'london_breakout', 'newyork_high', 'newyork_low', 'newyork_breakout']
        ))
        
        self.register(IndicatorConfig(
            name="SessionStats",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                'session': {'options': ['asian', 'london', 'newyork'], 'default': 'asian'},
                'lookback': {'min': 5, 'max': 50, 'default': 20}
            },
            output_columns=['session_avg_range', 'session_range_std', 'session_range_percentile']
        ))
        
        # OPENING RANGE BREAKOUT
        self.register(IndicatorConfig(
            name="ORB",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={
                'minutes': {'min': 5, 'max': 120, 'default': 30},
            },
            output_columns=['orb_high', 'orb_low', 'orb_range', 'orb_breakout', 'orb_target_1', 'orb_target_1_5', 'orb_target_2']
        ))
        
        # KILLZONES
        self.register(IndicatorConfig(
            name="Killzones",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'utc_offset': {'min': -12, 'max': 12, 'default': 0}},
            output_columns=['asian_kz', 'london_kz', 'nyam_kz', 'nypm_kz', 'in_killzone']
        ))
    
    def _register_ict_indicators(self):
        """Register ICT (Inner Circle Trader) indicators."""
        
        # FAIR VALUE GAPS
        self.register(IndicatorConfig(
            name="FairValueGap",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'min_gap_percent': {'min': 0.05, 'max': 1.0, 'default': 0.1}},
            output_columns=['fvg_bullish', 'fvg_bearish', 'fvg_bullish_top', 'fvg_bullish_bottom', 'fvg_bearish_top', 'fvg_bearish_bottom', 'fvg_bullish_unfilled', 'fvg_bearish_unfilled']
        ))
        
        # ORDER BLOCKS
        self.register(IndicatorConfig(
            name="OrderBlocks",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={
                'lookback': {'min': 5, 'max': 30, 'default': 10},
                'strength': {'min': 2, 'max': 5, 'default': 3}
            },
            output_columns=['ob_bullish', 'ob_bearish', 'ob_bullish_top', 'ob_bullish_bottom', 'ob_bearish_top', 'ob_bearish_bottom']
        ))
        
        # LIQUIDITY LEVELS
        self.register(IndicatorConfig(
            name="LiquidityLevels",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={
                'lookback': {'min': 10, 'max': 50, 'default': 20},
                'equal_threshold': {'min': 0.0001, 'max': 0.001, 'default': 0.0005}
            },
            output_columns=['bsl', 'ssl', 'bsl_level', 'ssl_level']
        ))
        
        # MARKET STRUCTURE
        self.register(IndicatorConfig(
            name="MarketStructure",
            indicator_type=IndicatorType.TREND,
            parameters={'swing_length': {'min': 3, 'max': 15, 'default': 5}},
            output_columns=['swing_high', 'swing_low', 'swing_high_price', 'swing_low_price', 'hh', 'hl', 'lh', 'll', 'structure', 'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish', 'mss']
        ))
        
        # PREMIUM/DISCOUNT ZONES
        self.register(IndicatorConfig(
            name="PremiumDiscount",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'lookback': {'min': 10, 'max': 50, 'default': 20}},
            output_columns=['range_high', 'range_low', 'equilibrium', 'premium_start', 'premium_70', 'premium_79', 'discount_end', 'discount_30', 'discount_21', 'zone', 'zone_pct']
        ))
        
        # OPTIMAL TRADE ENTRY
        self.register(IndicatorConfig(
            name="OTE",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={},
            output_columns=['ote_bullish_62', 'ote_bullish_70', 'ote_bullish_79', 'ote_bearish_62', 'ote_bearish_70', 'ote_bearish_79', 'in_bullish_ote', 'in_bearish_ote']
        ))
    
    def _register_smart_money_indicators(self):
        """Register Smart Money Concept indicators."""
        
        # DISPLACEMENT
        self.register(IndicatorConfig(
            name="Displacement",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'threshold_mult': {'min': 1.5, 'max': 4.0, 'default': 2.0},
                'lookback': {'min': 7, 'max': 25, 'default': 14}
            },
            output_columns=['displacement_bullish', 'displacement_bearish', 'displacement_strength']
        ))
        
        # INDUCEMENT
        self.register(IndicatorConfig(
            name="Inducement",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'lookback': {'min': 5, 'max': 20, 'default': 10}},
            output_columns=['inducement_high', 'inducement_low']
        ))
        
        # LIQUIDITY SWEEP
        self.register(IndicatorConfig(
            name="LiquiditySweep",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'lookback': {'min': 10, 'max': 50, 'default': 20}},
            output_columns=['sweep_bullish', 'sweep_bearish']
        ))
        
        # BREAKER BLOCKS
        self.register(IndicatorConfig(
            name="BreakerBlocks",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'lookback': {'min': 5, 'max': 30, 'default': 10}},
            output_columns=['breaker_bullish', 'breaker_bearish']
        ))
    
    def _register_additional_indicators(self):
        """Register additional technical indicators."""
        
        # PIVOT POINTS
        self.register(IndicatorConfig(
            name="PivotPoints",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'pivot_type': {'options': ['standard', 'fibonacci', 'camarilla'], 'default': 'standard'}},
            output_columns=['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']
        ))
        
        # VOLUME PROFILE (simplified)
        self.register(IndicatorConfig(
            name="VolumeProfile",
            indicator_type=IndicatorType.VOLUME,
            parameters={
                'bins': {'min': 10, 'max': 50, 'default': 20},
                'lookback': {'min': 50, 'max': 200, 'default': 100}
            },
            output_columns=['poc', 'vah', 'val']
        ))
        
        # VWAP BANDS
        self.register(IndicatorConfig(
            name="VWAPBandsExtended",
            indicator_type=IndicatorType.VOLUME,
            parameters={},
            output_columns=['vwap', 'vwap_rolling', 'vwap_upper_1', 'vwap_lower_1', 'vwap_upper_2', 'vwap_lower_2', 'vwap_upper_3', 'vwap_lower_3']
        ))
        
        # RANGE FILTER
        self.register(IndicatorConfig(
            name="RangeFilter",
            indicator_type=IndicatorType.TREND,
            parameters={
                'period': {'min': 50, 'max': 200, 'default': 100},
                'multiplier': {'min': 1.5, 'max': 4.0, 'default': 2.6}
            },
            output_columns=['range_filter', 'rf_trend']
        ))
        
        # CHANDELIER EXIT
        self.register(IndicatorConfig(
            name="ChandelierExit",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                'period': {'min': 14, 'max': 30, 'default': 22},
                'multiplier': {'min': 2.0, 'max': 4.0, 'default': 3.0}
            },
            output_columns=['chandelier_long', 'chandelier_short', 'chandelier_signal']
        ))
    
    def get_indicator_count(self) -> Dict[str, int]:
        """Get count of indicators by type."""
        counts = {}
        for ind_type in IndicatorType:
            counts[ind_type.value] = len(self.get_by_type(ind_type))
        counts['total'] = len(self.indicators)
        return counts


class ExtendedIndicatorCalculator(IndicatorCalculator):
    """
    Extended Indicator Calculator.
    
    Supports calculation of all extended indicators.
    """
    
    def __init__(self):
        super().__init__()
        self.ext_calc = ExtendedIndicatorCalculator.__new__(ExtendedIndicatorCalculator)
    
    def calculate(self, data: pd.DataFrame, 
                  indicator: IndicatorConfig,
                  params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate indicator with extended support."""
        
        # Try base calculator first
        name = indicator.name
        
        # Extended Trend Indicators
        if name == "VWMA":
            return self._calc_vwma(data, params)
        elif name == "HMA":
            return self._calc_hma(data, params)
        elif name == "KAMA":
            return self._calc_kama(data, params)
        elif name == "ZLEMA":
            return self._calc_zlema(data, params)
        elif name == "T3":
            return self._calc_t3(data, params)
        elif name == "PSAR":
            return self._calc_psar(data, params)
        elif name == "Aroon":
            return self._calc_aroon(data, params)
        elif name == "VIDYA":
            return self._calc_vidya(data, params)
        elif name == "Vortex":
            return self._calc_vortex(data, params)
        elif name == "DPO":
            return self._calc_dpo(data, params)
        
        # Extended Momentum Indicators
        elif name == "TSI":
            return self._calc_tsi(data, params)
        elif name == "UltimateOscillator":
            return self._calc_ultimate_oscillator(data, params)
        elif name == "AwesomeOscillator":
            return self._calc_awesome_oscillator(data, params)
        elif name == "TRIX":
            return self._calc_trix(data, params)
        elif name == "KST":
            return self._calc_kst(data, params)
        elif name == "PPO":
            return self._calc_ppo(data, params)
        elif name == "PVO":
            return self._calc_pvo(data, params)
        elif name == "DMI":
            return self._calc_dmi(data, params)
        elif name == "RVI_Momentum":
            return self._calc_rvi(data, params)
        elif name == "CMO":
            return self._calc_cmo(data, params)
        
        # Extended Volatility Indicators
        elif name == "ChaikinVolatility":
            return self._calc_chaikin_volatility(data, params)
        elif name == "HistoricalVolatility":
            return self._calc_historical_volatility(data, params)
        elif name == "MassIndex":
            return self._calc_mass_index(data, params)
        elif name == "NATR":
            return self._calc_natr(data, params)
        elif name == "UlcerIndex":
            return self._calc_ulcer_index(data, params)
        elif name == "ATRBands":
            return self._calc_atr_bands(data, params)
        elif name == "RVI_Volatility":
            return self._calc_rvi_volatility(data, params)
        
        # Extended Volume Indicators
        elif name == "VWAPBands":
            return self._calc_vwap_bands(data, params)
        elif name == "PVT":
            return self._calc_pvt(data, params)
        elif name == "NVI":
            return self._calc_nvi(data, params)
        elif name == "PVI":
            return self._calc_pvi(data, params)
        elif name == "EOM":
            return self._calc_eom(data, params)
        elif name == "ForceIndex":
            return self._calc_force_index(data, params)
        elif name == "MFI_Enhanced":
            return self._calc_mfi_enhanced(data, params)
        elif name == "Klinger":
            return self._calc_klinger(data, params)
        
        # Extended Price Action Indicators
        elif name == "HHLL":
            return self._calc_hhll(data, params)
        elif name == "ZigZag":
            return self._calc_zigzag(data, params)
        elif name == "ADR":
            return self._calc_adr(data, params)
        elif name == "CandleBody":
            return self._calc_candle_body(data, params)
        elif name == "InsideOutsideBar":
            return self._calc_inside_outside_bar(data, params)
        
        # Cycle Indicators
        elif name == "STC":
            return self._calc_stc(data, params)
        elif name == "FisherTransform":
            return self._calc_fisher(data, params)
        elif name == "Coppock":
            return self._calc_coppock(data, params)
        elif name == "ElderRay":
            return self._calc_elder_ray(data, params)
        elif name == "PGO":
            return self._calc_pgo(data, params)
        elif name == "WaveTrend":
            return self._calc_wave_trend(data, params)
        
        # Base indicators that need extended implementation
        elif name == "Ichimoku":
            return self._calc_ichimoku(data, params)
        elif name == "Supertrend":
            return self._calc_supertrend(data, params)
        elif name == "StochRSI":
            return self._calc_stochrsi(data, params)
        elif name == "MFI":
            return self._calc_mfi(data, params)
        elif name == "VWAP":
            return self._calc_vwap(data, params)
        elif name == "AD":
            return self._calc_ad(data, params)
        elif name == "CMF":
            return self._calc_cmf(data, params)
        elif name == "FibonacciLevels":
            return self._calc_fibonacci(data, params)
        elif name == "Candle_Patterns":
            return self._calc_candle_patterns(data, params)
        
        # ===== SESSION INDICATORS =====
        elif name == "AsianSession":
            return self._calc_asian_session(data, params)
        elif name == "LondonSession":
            return self._calc_london_session(data, params)
        elif name == "NewYorkSession":
            return self._calc_newyork_session(data, params)
        elif name == "AllSessions":
            return self._calc_all_sessions(data, params)
        elif name == "SessionStats":
            return self._calc_session_stats(data, params)
        elif name == "ORB":
            return self._calc_orb(data, params)
        elif name == "Killzones":
            return self._calc_killzones(data, params)
        
        # ===== ICT INDICATORS =====
        elif name == "FairValueGap":
            return self._calc_fvg(data, params)
        elif name == "OrderBlocks":
            return self._calc_order_blocks(data, params)
        elif name == "LiquidityLevels":
            return self._calc_liquidity_levels(data, params)
        elif name == "MarketStructure":
            return self._calc_market_structure(data, params)
        elif name == "PremiumDiscount":
            return self._calc_premium_discount(data, params)
        elif name == "OTE":
            return self._calc_ote(data, params)
        
        # ===== SMART MONEY INDICATORS =====
        elif name == "Displacement":
            return self._calc_displacement(data, params)
        elif name == "Inducement":
            return self._calc_inducement(data, params)
        elif name == "LiquiditySweep":
            return self._calc_liquidity_sweep(data, params)
        elif name == "BreakerBlocks":
            return self._calc_breaker_blocks(data, params)
        
        # ===== ADDITIONAL INDICATORS =====
        elif name == "PivotPoints":
            return self._calc_pivot_points(data, params)
        elif name == "VolumeProfile":
            return self._calc_volume_profile(data, params)
        elif name == "VWAPBandsExtended":
            return self._calc_vwap_bands_extended(data, params)
        elif name == "RangeFilter":
            return self._calc_range_filter(data, params)
        elif name == "ChandelierExit":
            return self._calc_chandelier_exit(data, params)
        
        # Fall back to base calculator
        return super().calculate(data, indicator, params)
    
    # ===== TREND INDICATOR IMPLEMENTATIONS =====
    
    def _calc_vwma(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        period = params.get('period', 20)
        df = df.copy()
        df[f'vwma_{period}'] = (
            (df['close'] * df['volume']).rolling(window=period).sum() /
            df['volume'].rolling(window=period).sum()
        )
        return df
    
    def _calc_hma(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        period = params.get('period', 20)
        df = df.copy()
        half_period = max(int(period / 2), 1)
        sqrt_period = max(int(np.sqrt(period)), 1)
        
        weights_half = np.arange(1, half_period + 1)
        weights_full = np.arange(1, period + 1)
        weights_sqrt = np.arange(1, sqrt_period + 1)
        
        wma_half = df['close'].rolling(window=half_period).apply(
            lambda x: np.dot(x, weights_half[:len(x)]) / weights_half[:len(x)].sum() if len(x) == half_period else np.nan,
            raw=True
        )
        wma_full = df['close'].rolling(window=period).apply(
            lambda x: np.dot(x, weights_full[:len(x)]) / weights_full[:len(x)].sum() if len(x) == period else np.nan,
            raw=True
        )
        
        raw_hma = 2 * wma_half - wma_full
        df[f'hma_{period}'] = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: np.dot(x, weights_sqrt[:len(x)]) / weights_sqrt[:len(x)].sum() if len(x) == sqrt_period else np.nan,
            raw=True
        )
        return df
    
    def _calc_kama(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        period = params.get('period', 10)
        fast = params.get('fast_period', 2)
        slow = params.get('slow_period', 30)
        df = df.copy()
        
        change = abs(df['close'] - df['close'].shift(period))
        volatility = df['close'].diff().abs().rolling(window=period).sum()
        er = (change / volatility).fillna(0)
        
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama = pd.Series(index=df.index, dtype=float)
        kama.iloc[period] = df['close'].iloc[period]
        
        for i in range(period + 1, len(df)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df['close'].iloc[i] - kama.iloc[i-1])
        
        df[f'kama_{period}'] = kama
        return df
    
    def _calc_zlema(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        period = params.get('period', 20)
        df = df.copy()
        lag = int((period - 1) / 2)
        ema_data = df['close'] + (df['close'] - df['close'].shift(lag))
        df[f'zlema_{period}'] = ema_data.ewm(span=period, adjust=False).mean()
        return df
    
    def _calc_t3(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        period = params.get('period', 5)
        v_factor = params.get('v_factor', 0.7)
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
    
    def _calc_psar(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_psar(df, 
                                   params.get('af_start', 0.02),
                                   params.get('af_step', 0.02),
                                   params.get('af_max', 0.2))
    
    def _calc_aroon(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_aroon(df, params.get('period', 25))
    
    def _calc_vidya(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_vidya(df, 
                                    params.get('period', 14),
                                    params.get('cmo_period', 9))
    
    def _calc_vortex(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_vortex(df, params.get('period', 14))
    
    def _calc_dpo(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_dpo(df, params.get('period', 20))
    
    # ===== MOMENTUM INDICATOR IMPLEMENTATIONS =====
    
    def _calc_tsi(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_tsi(df,
                                  params.get('long_period', 25),
                                  params.get('short_period', 13),
                                  params.get('signal_period', 13))
    
    def _calc_ultimate_oscillator(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_ultimate_oscillator(df,
                                                   params.get('period1', 7),
                                                   params.get('period2', 14),
                                                   params.get('period3', 28))
    
    def _calc_awesome_oscillator(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_awesome_oscillator(df,
                                                  params.get('fast', 5),
                                                  params.get('slow', 34))
    
    def _calc_trix(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_trix(df,
                                   params.get('period', 15),
                                   params.get('signal', 9))
    
    def _calc_kst(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_kst(df)
    
    def _calc_ppo(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_ppo(df,
                                  params.get('fast', 12),
                                  params.get('slow', 26),
                                  params.get('signal', 9))
    
    def _calc_pvo(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_pvo(df,
                                  params.get('fast', 12),
                                  params.get('slow', 26),
                                  params.get('signal', 9))
    
    def _calc_dmi(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_dmi(df,
                                  params.get('period', 14),
                                  params.get('adx_smooth', 14))
    
    def _calc_rvi(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_rvi(df, params.get('period', 10))
    
    def _calc_cmo(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_cmo(df, params.get('period', 14))
    
    # ===== VOLATILITY INDICATOR IMPLEMENTATIONS =====
    
    def _calc_chaikin_volatility(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_chaikin_volatility(df,
                                                  params.get('ema_period', 10),
                                                  params.get('roc_period', 10))
    
    def _calc_historical_volatility(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_historical_volatility(df, params.get('period', 20))
    
    def _calc_mass_index(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_mass_index(df,
                                         params.get('ema_period', 9),
                                         params.get('sum_period', 25))
    
    def _calc_natr(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_natr(df, params.get('period', 14))
    
    def _calc_ulcer_index(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_ulcer_index(df, params.get('period', 14))
    
    def _calc_atr_bands(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_atr_bands(df,
                                        params.get('period', 14),
                                        params.get('multiplier', 2.0))
    
    def _calc_rvi_volatility(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_rvi_volatility(df, params.get('period', 10))
    
    # ===== VOLUME INDICATOR IMPLEMENTATIONS =====
    
    def _calc_vwap_bands(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_vwap_bands(df, params.get('std_mult', 2.0))
    
    def _calc_pvt(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_pvt(df)
    
    def _calc_nvi(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_nvi(df)
    
    def _calc_pvi(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_pvi(df)
    
    def _calc_eom(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_eom(df, params.get('period', 14))
    
    def _calc_force_index(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_force_index(df, params.get('period', 13))
    
    def _calc_mfi_enhanced(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_mfi_enhanced(df, params.get('period', 14))
    
    def _calc_klinger(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_klinger(df,
                                      params.get('fast', 34),
                                      params.get('slow', 55),
                                      params.get('signal', 13))
    
    # ===== PRICE ACTION INDICATOR IMPLEMENTATIONS =====
    
    def _calc_hhll(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_higher_high_lower_low(df, params.get('period', 20))
    
    def _calc_zigzag(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_zigzag(df, params.get('threshold', 5.0))
    
    def _calc_adr(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_adr(df, params.get('period', 14))
    
    def _calc_candle_body(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_candle_body(df)
    
    def _calc_inside_outside_bar(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_inside_outside_bar(df)
    
    # ===== CYCLE INDICATOR IMPLEMENTATIONS =====
    
    def _calc_stc(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_schaff_trend(df,
                                           params.get('period', 10),
                                           params.get('fast', 23),
                                           params.get('slow', 50))
    
    def _calc_fisher(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_ehlers_fisher(df, params.get('period', 10))
    
    def _calc_coppock(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_coppock(df,
                                      params.get('wma_period', 10),
                                      params.get('roc1', 14),
                                      params.get('roc2', 11))
    
    def _calc_elder_ray(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_elder_ray(df, params.get('period', 13))
    
    def _calc_pgo(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_pretty_good_oscillator(df, params.get('period', 14))
    
    def _calc_wave_trend(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_wave_trend(df,
                                         params.get('channel', 10),
                                         params.get('average', 21))
    
    # ===== BASE INDICATORS (missing implementations) =====
    
    def _calc_ichimoku(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_ichimoku(df,
                                       params.get('tenkan', 9),
                                       params.get('kijun', 26),
                                       params.get('senkou', 52))
    
    def _calc_supertrend(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_supertrend(df,
                                         params.get('period', 10),
                                         params.get('multiplier', 3.0))
    
    def _calc_stochrsi(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_stochrsi(df,
                                       params.get('rsi_period', 14),
                                       params.get('stoch_period', 14),
                                       params.get('k_period', 3),
                                       params.get('d_period', 3))
    
    def _calc_mfi(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_mfi(df, params.get('period', 14))
    
    def _calc_vwap(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_vwap(df)
    
    def _calc_ad(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_ad(df)
    
    def _calc_cmf(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_cmf(df, params.get('period', 20))
    
    def _calc_fibonacci(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_fibonacci(df, params.get('lookback', 50))
    
    def _calc_candle_patterns(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from .extended_indicators import ExtendedIndicatorCalculator as EIC
        return EIC.calculate_candle_patterns(df)
    
    # ===== SESSION INDICATOR IMPLEMENTATIONS =====
    
    def _calc_asian_session(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        df = SessionIndicators.calculate_session_ranges(df, 'asian', params.get('utc_offset', 0))
        df = SessionIndicators.calculate_session_breakout(df, 'asian')
        return df
    
    def _calc_london_session(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        df = SessionIndicators.calculate_session_ranges(df, 'london', params.get('utc_offset', 0))
        df = SessionIndicators.calculate_session_breakout(df, 'london')
        return df
    
    def _calc_newyork_session(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        df = SessionIndicators.calculate_session_ranges(df, 'newyork', params.get('utc_offset', 0))
        df = SessionIndicators.calculate_session_breakout(df, 'newyork')
        return df
    
    def _calc_all_sessions(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return SessionIndicators.calculate_all_sessions(df, params.get('utc_offset', 0))
    
    def _calc_session_stats(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        session = params.get('session', 'asian')
        lookback = params.get('lookback', 20)
        return SessionIndicators.calculate_session_stats(df, session, lookback)
    
    def _calc_orb(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        from datetime import time
        minutes = params.get('minutes', 30)
        return OpeningRangeBreakout.calculate_orb(df, minutes)
    
    def _calc_killzones(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return ICTIndicators.calculate_killzones(df, params.get('utc_offset', 0))
    
    # ===== ICT INDICATOR IMPLEMENTATIONS =====
    
    def _calc_fvg(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return ICTIndicators.calculate_fair_value_gaps(df, params.get('min_gap_percent', 0.1))
    
    def _calc_order_blocks(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return ICTIndicators.calculate_order_blocks(
            df,
            params.get('lookback', 10),
            params.get('strength', 3)
        )
    
    def _calc_liquidity_levels(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return ICTIndicators.calculate_liquidity_levels(
            df,
            params.get('lookback', 20),
            params.get('equal_threshold', 0.0005)
        )
    
    def _calc_market_structure(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return ICTIndicators.calculate_market_structure(df, params.get('swing_length', 5))
    
    def _calc_premium_discount(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return ICTIndicators.calculate_premium_discount(df, params.get('lookback', 20))
    
    def _calc_ote(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return ICTIndicators.calculate_optimal_trade_entry(df)
    
    # ===== SMART MONEY INDICATOR IMPLEMENTATIONS =====
    
    def _calc_displacement(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return SmartMoneyIndicators.calculate_displacement(
            df,
            params.get('threshold_mult', 2.0),
            params.get('lookback', 14)
        )
    
    def _calc_inducement(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return SmartMoneyIndicators.calculate_inducement(df, params.get('lookback', 10))
    
    def _calc_liquidity_sweep(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return SmartMoneyIndicators.calculate_liquidity_sweep(df, params.get('lookback', 20))
    
    def _calc_breaker_blocks(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return SmartMoneyIndicators.calculate_breaker_block(df, params.get('lookback', 10))
    
    # ===== ADDITIONAL INDICATOR IMPLEMENTATIONS =====
    
    def _calc_pivot_points(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return AdditionalIndicators.calculate_pivot_points(df, params.get('pivot_type', 'standard'))
    
    def _calc_volume_profile(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return AdditionalIndicators.calculate_volume_profile(
            df,
            params.get('bins', 20),
            params.get('lookback', 100)
        )
    
    def _calc_vwap_bands_extended(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return AdditionalIndicators.calculate_vwap_bands(df)
    
    def _calc_range_filter(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return AdditionalIndicators.calculate_range_filter(
            df,
            params.get('period', 100),
            params.get('multiplier', 2.6)
        )
    
    def _calc_chandelier_exit(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return AdditionalIndicators.calculate_chandelier_exit(
            df,
            params.get('period', 22),
            params.get('multiplier', 3.0)
        )


def create_extended_generator():
    """
    Factory function to create an AutoStrategyGenerator with extended indicators.
    """
    from .auto_generator import AutoStrategyGenerator
    
    # Create generator
    generator = AutoStrategyGenerator()
    
    # Replace with extended library and calculator
    generator.indicator_library = ExtendedIndicatorLibrary()
    generator.indicator_calculator = ExtendedIndicatorCalculator()
    
    return generator


# Convenience function to get indicator statistics
def get_indicator_stats() -> Dict[str, Any]:
    """Get statistics about available indicators."""
    library = ExtendedIndicatorLibrary()
    
    stats = {
        'total_indicators': len(library.indicators),
        'by_type': library.get_indicator_count(),
        'indicators_list': {
            ind_type.value: [ind.name for ind in library.get_by_type(ind_type)]
            for ind_type in IndicatorType
        }
    }
    
    return stats
