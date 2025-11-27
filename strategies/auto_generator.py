# strategies/auto_generator.py
"""
Automatic Strategy Generator.

Generates trading strategies automatically by combining indicators,
conditions, and rules using genetic algorithms and machine learning.
"""

import logging
import random
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
import json
import copy

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of technical indicators."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"
    CUSTOM = "custom"


class ConditionOperator(Enum):
    """Operators for conditions."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUALS = "=="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"


class SignalType(Enum):
    """Signal types."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0


@dataclass
class IndicatorConfig:
    """Configuration for an indicator."""
    name: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    output_columns: List[str]
    calculate_func: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.indicator_type.value,
            'parameters': self.parameters,
            'output_columns': self.output_columns
        }


@dataclass
class Condition:
    """A trading condition."""
    left_operand: str  # Column name or value
    operator: ConditionOperator
    right_operand: str  # Column name or value
    right_operand2: Optional[str] = None  # For BETWEEN operator
    
    def to_dict(self) -> Dict:
        return {
            'left': self.left_operand,
            'operator': self.operator.value,
            'right': self.right_operand,
            'right2': self.right_operand2
        }
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate the condition on data."""
        # Get left operand
        if self.left_operand in data.columns:
            left = data[self.left_operand]
            left_is_series = True
        else:
            try:
                left = float(self.left_operand)
                left_is_series = False
            except ValueError:
                left = data[self.left_operand] if self.left_operand in data.columns else 0
                left_is_series = isinstance(left, pd.Series)
        
        # Get right operand
        if self.right_operand in data.columns:
            right = data[self.right_operand]
            right_is_series = True
        else:
            try:
                right = float(self.right_operand)
                right_is_series = False
            except ValueError:
                right = data[self.right_operand] if self.right_operand in data.columns else 0
                right_is_series = isinstance(right, pd.Series)
        
        # Evaluate based on operator
        if self.operator == ConditionOperator.GREATER_THAN:
            return left > right
        elif self.operator == ConditionOperator.LESS_THAN:
            return left < right
        elif self.operator == ConditionOperator.GREATER_EQUAL:
            return left >= right
        elif self.operator == ConditionOperator.LESS_EQUAL:
            return left <= right
        elif self.operator == ConditionOperator.EQUALS:
            return left == right
        elif self.operator == ConditionOperator.CROSSES_ABOVE:
            # Handle scalar operands for cross conditions
            if left_is_series and right_is_series:
                return (left > right) & (left.shift(1) <= right.shift(1))
            elif left_is_series:
                return (left > right) & (left.shift(1) <= right)
            elif right_is_series:
                return (left > right) & (left <= right.shift(1))
            else:
                # Both scalars - can't cross
                return pd.Series(False, index=data.index)
        elif self.operator == ConditionOperator.CROSSES_BELOW:
            # Handle scalar operands for cross conditions
            if left_is_series and right_is_series:
                return (left < right) & (left.shift(1) >= right.shift(1))
            elif left_is_series:
                return (left < right) & (left.shift(1) >= right)
            elif right_is_series:
                return (left < right) & (left >= right.shift(1))
            else:
                # Both scalars - can't cross
                return pd.Series(False, index=data.index)
        elif self.operator == ConditionOperator.BETWEEN:
            if self.right_operand2 in data.columns:
                right2 = data[self.right_operand2]
            else:
                right2 = float(self.right_operand2) if self.right_operand2 else right
            return (left >= right) & (left <= right2)
        elif self.operator == ConditionOperator.NOT_BETWEEN:
            if self.right_operand2 in data.columns:
                right2 = data[self.right_operand2]
            else:
                right2 = float(self.right_operand2) if self.right_operand2 else right
            return (left < right) | (left > right2)
        
        return pd.Series(False, index=data.index)


@dataclass
class TradingRule:
    """A trading rule combining multiple conditions."""
    name: str
    signal_type: SignalType
    conditions: List[Condition]
    logic: str = "AND"  # "AND" or "OR"
    priority: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'signal': self.signal_type.value,
            'conditions': [c.to_dict() for c in self.conditions],
            'logic': self.logic,
            'priority': self.priority
        }
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate the rule on data."""
        if not self.conditions:
            return pd.Series(False, index=data.index)
        
        results = [cond.evaluate(data) for cond in self.conditions]
        
        if self.logic == "AND":
            combined = results[0]
            for r in results[1:]:
                combined = combined & r
        else:  # OR
            combined = results[0]
            for r in results[1:]:
                combined = combined | r
        
        return combined


@dataclass
class GeneratedStrategy:
    """A generated trading strategy."""
    id: str
    name: str
    description: str
    indicators: List[IndicatorConfig]
    entry_rules: List[TradingRule]
    exit_rules: List[TradingRule]
    risk_management: Dict[str, Any]
    timeframe: str
    symbols: List[str]
    generation_date: datetime
    fitness_score: float = 0.0
    backtest_results: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'indicators': [i.to_dict() for i in self.indicators],
            'entry_rules': [r.to_dict() for r in self.entry_rules],
            'exit_rules': [r.to_dict() for r in self.exit_rules],
            'risk_management': self.risk_management,
            'timeframe': self.timeframe,
            'symbols': self.symbols,
            'generation_date': self.generation_date.isoformat(),
            'fitness_score': self.fitness_score,
            'backtest_results': self.backtest_results,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GeneratedStrategy':
        """Create strategy from dictionary."""
        # This would need full implementation to deserialize
        pass


class IndicatorLibrary:
    """Library of available indicators for strategy generation."""
    
    def __init__(self):
        self.indicators: Dict[str, IndicatorConfig] = {}
        self._register_default_indicators()
    
    def _register_default_indicators(self):
        """Register default indicators."""
        # Trend Indicators
        self.register(IndicatorConfig(
            name="SMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 200, 'default': 20}},
            output_columns=['sma_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="EMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 200, 'default': 20}},
            output_columns=['ema_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="DEMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 100, 'default': 20}},
            output_columns=['dema_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="TEMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 100, 'default': 20}},
            output_columns=['tema_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="WMA",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 5, 'max': 200, 'default': 20}},
            output_columns=['wma_{period}']
        ))
        
        self.register(IndicatorConfig(
            name="MACD",
            indicator_type=IndicatorType.TREND,
            parameters={
                'fast_period': {'min': 8, 'max': 20, 'default': 12},
                'slow_period': {'min': 20, 'max': 35, 'default': 26},
                'signal_period': {'min': 5, 'max': 15, 'default': 9}
            },
            output_columns=['macd', 'macd_signal', 'macd_histogram']
        ))
        
        self.register(IndicatorConfig(
            name="ADX",
            indicator_type=IndicatorType.TREND,
            parameters={'period': {'min': 7, 'max': 30, 'default': 14}},
            output_columns=['adx', 'di_plus', 'di_minus']
        ))
        
        self.register(IndicatorConfig(
            name="Ichimoku",
            indicator_type=IndicatorType.TREND,
            parameters={
                'tenkan_period': {'min': 7, 'max': 12, 'default': 9},
                'kijun_period': {'min': 20, 'max': 30, 'default': 26},
                'senkou_period': {'min': 45, 'max': 60, 'default': 52}
            },
            output_columns=['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou']
        ))
        
        self.register(IndicatorConfig(
            name="Supertrend",
            indicator_type=IndicatorType.TREND,
            parameters={
                'period': {'min': 7, 'max': 20, 'default': 10},
                'multiplier': {'min': 1.5, 'max': 4.0, 'default': 3.0}
            },
            output_columns=['supertrend', 'supertrend_direction']
        ))
        
        # Momentum Indicators
        self.register(IndicatorConfig(
            name="RSI",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 5, 'max': 30, 'default': 14}},
            output_columns=['rsi']
        ))
        
        self.register(IndicatorConfig(
            name="Stochastic",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'k_period': {'min': 5, 'max': 21, 'default': 14},
                'd_period': {'min': 3, 'max': 7, 'default': 3}
            },
            output_columns=['stoch_k', 'stoch_d']
        ))
        
        self.register(IndicatorConfig(
            name="StochRSI",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                'rsi_period': {'min': 7, 'max': 21, 'default': 14},
                'stoch_period': {'min': 7, 'max': 21, 'default': 14}
            },
            output_columns=['stochrsi_k', 'stochrsi_d']
        ))
        
        self.register(IndicatorConfig(
            name="CCI",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 10, 'max': 30, 'default': 20}},
            output_columns=['cci']
        ))
        
        self.register(IndicatorConfig(
            name="Williams_R",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 7, 'max': 21, 'default': 14}},
            output_columns=['williams_r']
        ))
        
        self.register(IndicatorConfig(
            name="MFI",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 7, 'max': 21, 'default': 14}},
            output_columns=['mfi']
        ))
        
        self.register(IndicatorConfig(
            name="ROC",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 5, 'max': 30, 'default': 12}},
            output_columns=['roc']
        ))
        
        self.register(IndicatorConfig(
            name="Momentum",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={'period': {'min': 5, 'max': 30, 'default': 10}},
            output_columns=['momentum']
        ))
        
        # Volatility Indicators
        self.register(IndicatorConfig(
            name="BollingerBands",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                'period': {'min': 10, 'max': 30, 'default': 20},
                'std_dev': {'min': 1.5, 'max': 3.0, 'default': 2.0}
            },
            output_columns=['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent']
        ))
        
        self.register(IndicatorConfig(
            name="ATR",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={'period': {'min': 7, 'max': 21, 'default': 14}},
            output_columns=['atr']
        ))
        
        self.register(IndicatorConfig(
            name="KeltnerChannel",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                'ema_period': {'min': 10, 'max': 30, 'default': 20},
                'atr_period': {'min': 7, 'max': 21, 'default': 10},
                'multiplier': {'min': 1.0, 'max': 3.0, 'default': 2.0}
            },
            output_columns=['kc_upper', 'kc_middle', 'kc_lower']
        ))
        
        self.register(IndicatorConfig(
            name="DonchianChannel",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={'period': {'min': 10, 'max': 55, 'default': 20}},
            output_columns=['dc_upper', 'dc_middle', 'dc_lower']
        ))
        
        self.register(IndicatorConfig(
            name="Volatility",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={'period': {'min': 10, 'max': 30, 'default': 20}},
            output_columns=['volatility']
        ))
        
        # Volume Indicators
        self.register(IndicatorConfig(
            name="OBV",
            indicator_type=IndicatorType.VOLUME,
            parameters={},
            output_columns=['obv']
        ))
        
        self.register(IndicatorConfig(
            name="VWAP",
            indicator_type=IndicatorType.VOLUME,
            parameters={},
            output_columns=['vwap']
        ))
        
        self.register(IndicatorConfig(
            name="VolumeMA",
            indicator_type=IndicatorType.VOLUME,
            parameters={'period': {'min': 10, 'max': 50, 'default': 20}},
            output_columns=['volume_ma', 'volume_ratio']
        ))
        
        self.register(IndicatorConfig(
            name="AD",
            indicator_type=IndicatorType.VOLUME,
            parameters={},
            output_columns=['ad']
        ))
        
        self.register(IndicatorConfig(
            name="CMF",
            indicator_type=IndicatorType.VOLUME,
            parameters={'period': {'min': 10, 'max': 30, 'default': 20}},
            output_columns=['cmf']
        ))
        
        # Price Action
        self.register(IndicatorConfig(
            name="PivotPoints",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={},
            output_columns=['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']
        ))
        
        self.register(IndicatorConfig(
            name="FibonacciLevels",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'lookback': {'min': 20, 'max': 100, 'default': 50}},
            output_columns=['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100']
        ))
        
        self.register(IndicatorConfig(
            name="SwingHighLow",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={'lookback': {'min': 5, 'max': 20, 'default': 10}},
            output_columns=['swing_high', 'swing_low']
        ))
        
        self.register(IndicatorConfig(
            name="Candle_Patterns",
            indicator_type=IndicatorType.PRICE_ACTION,
            parameters={},
            output_columns=['doji', 'hammer', 'engulfing', 'morning_star', 'evening_star']
        ))
    
    def register(self, indicator: IndicatorConfig):
        """Register an indicator."""
        self.indicators[indicator.name] = indicator
    
    def get(self, name: str) -> Optional[IndicatorConfig]:
        """Get indicator by name."""
        return self.indicators.get(name)
    
    def get_by_type(self, indicator_type: IndicatorType) -> List[IndicatorConfig]:
        """Get all indicators of a type."""
        return [ind for ind in self.indicators.values() 
                if ind.indicator_type == indicator_type]
    
    def get_all(self) -> List[IndicatorConfig]:
        """Get all indicators."""
        return list(self.indicators.values())
    
    def get_random(self, count: int = 1, 
                   indicator_type: IndicatorType = None) -> List[IndicatorConfig]:
        """Get random indicators."""
        if indicator_type:
            pool = self.get_by_type(indicator_type)
        else:
            pool = self.get_all()
        
        return random.sample(pool, min(count, len(pool)))


class IndicatorCalculator:
    """Calculates indicators on data."""
    
    def __init__(self):
        self.ta_available = False
        try:
            import ta
            self.ta_available = True
        except ImportError:
            logger.warning("ta library not available, using basic calculations")
    
    def calculate(self, data: pd.DataFrame, 
                  indicator: IndicatorConfig,
                  params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate an indicator and add columns to data."""
        df = data.copy()
        name = indicator.name
        
        try:
            if name == "SMA":
                period = params.get('period', 20)
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
            elif name == "EMA":
                period = params.get('period', 20)
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
            elif name == "DEMA":
                period = params.get('period', 20)
                ema1 = df['close'].ewm(span=period, adjust=False).mean()
                ema2 = ema1.ewm(span=period, adjust=False).mean()
                df[f'dema_{period}'] = 2 * ema1 - ema2
                
            elif name == "TEMA":
                period = params.get('period', 20)
                ema1 = df['close'].ewm(span=period, adjust=False).mean()
                ema2 = ema1.ewm(span=period, adjust=False).mean()
                ema3 = ema2.ewm(span=period, adjust=False).mean()
                df[f'tema_{period}'] = 3 * ema1 - 3 * ema2 + ema3
                
            elif name == "WMA":
                period = params.get('period', 20)
                weights = np.arange(1, period + 1)
                df[f'wma_{period}'] = df['close'].rolling(window=period).apply(
                    lambda x: np.dot(x, weights) / weights.sum(), raw=True
                )
                
            elif name == "MACD":
                fast = params.get('fast_period', 12)
                slow = params.get('slow_period', 26)
                signal = params.get('signal_period', 9)
                ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
                ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
                df['macd'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                
            elif name == "RSI":
                period = params.get('period', 14)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
            elif name == "Stochastic":
                k_period = params.get('k_period', 14)
                d_period = params.get('d_period', 3)
                lowest_low = df['low'].rolling(window=k_period).min()
                highest_high = df['high'].rolling(window=k_period).max()
                df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
                df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
                
            elif name == "CCI":
                period = params.get('period', 20)
                tp = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = tp.rolling(window=period).mean()
                mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
                df['cci'] = (tp - sma_tp) / (0.015 * mad)
                
            elif name == "Williams_R":
                period = params.get('period', 14)
                highest_high = df['high'].rolling(window=period).max()
                lowest_low = df['low'].rolling(window=period).min()
                df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
                
            elif name == "BollingerBands":
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2.0)
                df['bb_middle'] = df['close'].rolling(window=period).mean()
                rolling_std = df['close'].rolling(window=period).std()
                df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
                df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
            elif name == "ATR":
                period = params.get('period', 14)
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift())
                tr3 = abs(df['low'] - df['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['atr'] = tr.rolling(window=period).mean()
                
            elif name == "ADX":
                period = params.get('period', 14)
                df['atr'] = self._calculate_atr(df, period)
                
                plus_dm = df['high'].diff()
                minus_dm = -df['low'].diff()
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
                
                atr = df['atr']
                df['di_plus'] = 100 * (plus_dm.ewm(span=period).mean() / atr)
                df['di_minus'] = 100 * (minus_dm.ewm(span=period).mean() / atr)
                
                dx = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
                df['adx'] = dx.ewm(span=period).mean()
                
            elif name == "OBV":
                df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                
            elif name == "VolumeMA":
                period = params.get('period', 20)
                df['volume_ma'] = df['volume'].rolling(window=period).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
                
            elif name == "Momentum":
                period = params.get('period', 10)
                df['momentum'] = df['close'] / df['close'].shift(period) - 1
                
            elif name == "ROC":
                period = params.get('period', 12)
                df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
                
            elif name == "Volatility":
                period = params.get('period', 20)
                df['volatility'] = df['close'].pct_change().rolling(window=period).std() * np.sqrt(252)
                
            elif name == "DonchianChannel":
                period = params.get('period', 20)
                df['dc_upper'] = df['high'].rolling(window=period).max()
                df['dc_lower'] = df['low'].rolling(window=period).min()
                df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
                
            elif name == "KeltnerChannel":
                ema_period = params.get('ema_period', 20)
                atr_period = params.get('atr_period', 10)
                multiplier = params.get('multiplier', 2.0)
                
                df['kc_middle'] = df['close'].ewm(span=ema_period, adjust=False).mean()
                atr = self._calculate_atr(df, atr_period)
                df['kc_upper'] = df['kc_middle'] + (atr * multiplier)
                df['kc_lower'] = df['kc_middle'] - (atr * multiplier)
                
            elif name == "PivotPoints":
                df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
                df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
                df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
                df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
                df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
                df['r3'] = df['high'].shift(1) + 2 * (df['pivot'] - df['low'].shift(1))
                df['s3'] = df['low'].shift(1) - 2 * (df['high'].shift(1) - df['pivot'])
                
            elif name == "SwingHighLow":
                lookback = params.get('lookback', 10)
                df['swing_high'] = df['high'].rolling(window=lookback*2+1, center=True).max()
                df['swing_low'] = df['low'].rolling(window=lookback*2+1, center=True).min()
                
            else:
                logger.warning(f"Unknown indicator: {name}")
                
        except Exception as e:
            logger.error(f"Error calculating {name}: {e}")
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


class AutoStrategyGenerator:
    """
    Automatic Strategy Generator.
    
    Generates trading strategies by combining indicators and rules.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.indicator_library = IndicatorLibrary()
        self.indicator_calculator = IndicatorCalculator()
        self.generated_strategies: List[GeneratedStrategy] = []
        
        # Generation parameters
        self.min_indicators = self.config.get('min_indicators', 2)
        self.max_indicators = self.config.get('max_indicators', 5)
        self.min_entry_rules = self.config.get('min_entry_rules', 1)
        self.max_entry_rules = self.config.get('max_entry_rules', 3)
        self.min_conditions_per_rule = self.config.get('min_conditions_per_rule', 1)
        self.max_conditions_per_rule = self.config.get('max_conditions_per_rule', 4)
    
    def generate_random_strategy(self, 
                                  name: str = None,
                                  symbols: List[str] = None,
                                  timeframe: str = "H1") -> GeneratedStrategy:
        """Generate a completely random strategy."""
        strategy_id = str(uuid.uuid4())[:8]
        name = name or f"AutoGen_{strategy_id}"
        symbols = symbols or ["EURUSD"]
        
        # Select random indicators
        num_indicators = random.randint(self.min_indicators, self.max_indicators)
        
        # Ensure diversity - at least one from different types
        selected_indicators = []
        
        # Get one trend indicator
        trend_indicators = self.indicator_library.get_by_type(IndicatorType.TREND)
        if trend_indicators:
            selected_indicators.append(random.choice(trend_indicators))
        
        # Get one momentum indicator
        momentum_indicators = self.indicator_library.get_by_type(IndicatorType.MOMENTUM)
        if momentum_indicators:
            selected_indicators.append(random.choice(momentum_indicators))
        
        # Fill rest randomly
        all_indicators = self.indicator_library.get_all()
        remaining = num_indicators - len(selected_indicators)
        if remaining > 0:
            available = [i for i in all_indicators if i not in selected_indicators]
            selected_indicators.extend(random.sample(available, min(remaining, len(available))))
        
        # Generate indicator configurations with random parameters
        indicator_configs = []
        available_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for indicator in selected_indicators:
            params = {}
            for param_name, param_config in indicator.parameters.items():
                if isinstance(param_config, dict):
                    min_val = param_config.get('min', 5)
                    max_val = param_config.get('max', 50)
                    default = param_config.get('default', (min_val + max_val) // 2)
                    
                    if isinstance(min_val, float):
                        params[param_name] = round(random.uniform(min_val, max_val), 2)
                    else:
                        params[param_name] = random.randint(min_val, max_val)
            
            config = IndicatorConfig(
                name=indicator.name,
                indicator_type=indicator.indicator_type,
                parameters=params,
                output_columns=self._get_output_columns(indicator, params)
            )
            indicator_configs.append(config)
            available_columns.extend(config.output_columns)
        
        # Generate entry rules
        entry_rules = self._generate_rules(
            available_columns, 
            SignalType.BUY,
            self.min_entry_rules,
            self.max_entry_rules
        )
        
        # Generate exit rules (opposite logic)
        exit_rules = self._generate_rules(
            available_columns,
            SignalType.SELL,
            1, 2
        )
        
        # Risk management
        risk_management = {
            'atr_multiplier': round(random.uniform(1.5, 3.0), 1),
            'risk_reward_ratio': round(random.uniform(1.0, 3.0), 1),
            'max_risk_per_trade': round(random.uniform(0.01, 0.03), 3),
            'trailing_stop': random.choice([True, False]),
            'break_even': random.choice([True, False])
        }
        
        strategy = GeneratedStrategy(
            id=strategy_id,
            name=name,
            description=f"Auto-generated strategy with {len(indicator_configs)} indicators",
            indicators=indicator_configs,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            risk_management=risk_management,
            timeframe=timeframe,
            symbols=symbols,
            generation_date=datetime.now()
        )
        
        self.generated_strategies.append(strategy)
        return strategy
    
    def _get_output_columns(self, indicator: IndicatorConfig, params: Dict) -> List[str]:
        """Get output column names with parameters substituted."""
        columns = []
        for col in indicator.output_columns:
            for param_name, param_value in params.items():
                col = col.replace(f'{{{param_name}}}', str(param_value))
            columns.append(col)
        return columns
    
    def _generate_rules(self, available_columns: List[str],
                        signal_type: SignalType,
                        min_rules: int, max_rules: int) -> List[TradingRule]:
        """Generate random trading rules."""
        rules = []
        num_rules = random.randint(min_rules, max_rules)
        
        for i in range(num_rules):
            conditions = self._generate_conditions(available_columns)
            
            rule = TradingRule(
                name=f"Rule_{signal_type.name}_{i+1}",
                signal_type=signal_type,
                conditions=conditions,
                logic=random.choice(["AND", "OR"]),
                priority=i + 1
            )
            rules.append(rule)
        
        return rules
    
    def _generate_conditions(self, available_columns: List[str]) -> List[Condition]:
        """Generate random conditions."""
        conditions = []
        num_conditions = random.randint(
            self.min_conditions_per_rule,
            self.max_conditions_per_rule
        )
        
        # Common thresholds for different indicator types
        thresholds = {
            'rsi': [20, 30, 50, 70, 80],
            'stoch': [20, 30, 50, 70, 80],
            'cci': [-200, -100, 0, 100, 200],
            'williams': [-80, -50, -20],
            'adx': [20, 25, 30, 40],
            'macd': [0],
            'bb_percent': [0, 0.2, 0.5, 0.8, 1.0],
            'momentum': [-0.02, 0, 0.02],
            'volume_ratio': [0.5, 1.0, 1.5, 2.0]
        }
        
        operators = [
            ConditionOperator.GREATER_THAN,
            ConditionOperator.LESS_THAN,
            ConditionOperator.CROSSES_ABOVE,
            ConditionOperator.CROSSES_BELOW
        ]
        
        for _ in range(num_conditions):
            left_col = random.choice(available_columns)
            operator = random.choice(operators)
            
            # Determine right operand
            if operator in [ConditionOperator.CROSSES_ABOVE, ConditionOperator.CROSSES_BELOW]:
                # Cross conditions usually compare two indicators
                numeric_cols = [c for c in available_columns 
                              if c not in ['open', 'high', 'low', 'close', 'volume']]
                if numeric_cols and random.random() > 0.3:
                    right_col = random.choice(numeric_cols)
                else:
                    # Use a threshold
                    right_col = str(self._get_threshold_for_column(left_col, thresholds))
            else:
                # Regular comparison
                if random.random() > 0.5:
                    right_col = str(self._get_threshold_for_column(left_col, thresholds))
                else:
                    similar_cols = [c for c in available_columns 
                                   if c != left_col and self._are_similar_columns(left_col, c)]
                    if similar_cols:
                        right_col = random.choice(similar_cols)
                    else:
                        right_col = str(self._get_threshold_for_column(left_col, thresholds))
            
            condition = Condition(
                left_operand=left_col,
                operator=operator,
                right_operand=right_col
            )
            conditions.append(condition)
        
        return conditions
    
    def _get_threshold_for_column(self, column: str, thresholds: Dict) -> float:
        """Get an appropriate threshold for a column."""
        for key, values in thresholds.items():
            if key in column.lower():
                return random.choice(values)
        
        # Default threshold
        return round(random.uniform(-1, 1), 3)
    
    def _are_similar_columns(self, col1: str, col2: str) -> bool:
        """Check if two columns are similar (can be compared)."""
        # Extract base indicator name
        def get_base(col):
            parts = col.split('_')
            return parts[0] if parts else col
        
        return get_base(col1) == get_base(col2)
    
    def generate_batch(self, count: int, **kwargs) -> List[GeneratedStrategy]:
        """Generate multiple strategies."""
        strategies = []
        for i in range(count):
            strategy = self.generate_random_strategy(
                name=f"AutoGen_{i+1}",
                **kwargs
            )
            strategies.append(strategy)
        return strategies
    
    def calculate_strategy_indicators(self, strategy: GeneratedStrategy, 
                                       data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for a strategy."""
        df = data.copy()
        
        for indicator in strategy.indicators:
            df = self.indicator_calculator.calculate(df, indicator, indicator.parameters)
        
        return df
    
    def evaluate_strategy(self, strategy: GeneratedStrategy,
                          data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate strategy signals on data."""
        # Calculate indicators
        df = self.calculate_strategy_indicators(strategy, data)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Evaluate entry rules
        for rule in strategy.entry_rules:
            try:
                mask = rule.evaluate(df)
                df.loc[mask, 'signal'] = rule.signal_type.value
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule.name}: {e}")
        
        # Evaluate exit rules (override if conditions met)
        for rule in strategy.exit_rules:
            try:
                mask = rule.evaluate(df)
                # Only apply exit if we have a position (previous signal != 0)
                df.loc[mask & (df['signal'].shift(1) != 0), 'signal'] = 0
            except Exception as e:
                logger.warning(f"Error evaluating exit rule {rule.name}: {e}")
        
        return df
    
    def export_strategy(self, strategy: GeneratedStrategy, 
                        filepath: str, format: str = 'json'):
        """Export strategy to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                f.write(strategy.to_json())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_strategy(self, filepath: str) -> GeneratedStrategy:
        """Import strategy from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return GeneratedStrategy.from_dict(data)
