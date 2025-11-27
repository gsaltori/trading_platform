# strategies/strategy_engine.py
"""
Strategy engine with multiple trading strategies.

Provides base strategy class and implementations for common strategies.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import ta library
TA_AVAILABLE = False
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    logger.info("ta library not available, using basic indicators")


@dataclass
class TradeSignal:
    """Trade signal container."""
    symbol: str
    direction: int  # 1 for long, -1 for short, 0 for neutral
    strength: float
    timestamp: pd.Timestamp
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    symbols: List[str]
    timeframe: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default risk management parameters
        if 'atr_multiplier' not in self.risk_management:
            self.risk_management['atr_multiplier'] = 2.0
        if 'risk_reward_ratio' not in self.risk_management:
            self.risk_management['risk_reward_ratio'] = 1.5


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.data = {}
        self.signals = []
        self.indicators = {}
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        pass
    
    def calculate_risk_management(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate stop loss and take profit levels."""
        # Calculate ATR if not present
        if 'atr' not in data.columns:
            data['atr'] = self._calculate_atr(data)
        
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        atr_multiplier = self.config.risk_management.get('atr_multiplier', 2.0)
        risk_reward = self.config.risk_management.get('risk_reward_ratio', 1.5)
        
        for idx in signals.index:
            if signals.loc[idx, 'signal'] != 0:
                current_atr = data.loc[idx, 'atr'] if idx in data.index else 0
                current_price = data.loc[idx, 'close'] if idx in data.index else 0
                
                if current_atr > 0 and current_price > 0:
                    if signals.loc[idx, 'signal'] > 0:  # Long
                        signals.loc[idx, 'stop_loss'] = current_price - (current_atr * atr_multiplier)
                        signals.loc[idx, 'take_profit'] = current_price + (current_atr * atr_multiplier * risk_reward)
                    else:  # Short
                        signals.loc[idx, 'stop_loss'] = current_price + (current_atr * atr_multiplier)
                        signals.loc[idx, 'take_profit'] = current_price - (current_atr * atr_multiplier * risk_reward)
        
        return signals
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR without external library."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def run(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Run the complete strategy."""
        data = data.copy()
        data = self.calculate_indicators(data)
        data = self.generate_signals(data)
        data = self.calculate_risk_management(data, data)
        return data


class MovingAverageCrossover(BaseStrategy):
    """Enhanced moving average crossover strategy."""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_period = self.config.parameters.get('fast_period', 10)
        slow_period = self.config.parameters.get('slow_period', 20)
        ma_type = self.config.parameters.get('ma_type', 'sma')
        
        if TA_AVAILABLE:
            if ma_type == 'sma':
                data['ma_fast'] = SMAIndicator(data['close'], window=fast_period).sma_indicator()
                data['ma_slow'] = SMAIndicator(data['close'], window=slow_period).sma_indicator()
            else:
                data['ma_fast'] = EMAIndicator(data['close'], window=fast_period).ema_indicator()
                data['ma_slow'] = EMAIndicator(data['close'], window=slow_period).ema_indicator()
            
            rsi_period = self.config.parameters.get('rsi_period', 14)
            data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        else:
            if ma_type == 'sma':
                data['ma_fast'] = self._calculate_sma(data['close'], fast_period)
                data['ma_slow'] = self._calculate_sma(data['close'], slow_period)
            else:
                data['ma_fast'] = self._calculate_ema(data['close'], fast_period)
                data['ma_slow'] = self._calculate_ema(data['close'], slow_period)
            
            rsi_period = self.config.parameters.get('rsi_period', 14)
            data['rsi'] = self._calculate_rsi(data['close'], rsi_period)
        
        data['ma_diff'] = data['ma_fast'] - data['ma_slow']
        data['ma_ratio'] = data['ma_fast'] / data['ma_slow'] - 1
        data['atr'] = self._calculate_atr(data)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        oversold = self.config.parameters.get('rsi_oversold', 30)
        overbought = self.config.parameters.get('rsi_overbought', 70)
        min_ma_diff = self.config.parameters.get('min_ma_diff', 0.0001)
        
        data['signal'] = 0
        
        # Buy signal: Fast MA crosses above slow MA and RSI not overbought
        buy_condition = (
            (data['ma_diff'] > min_ma_diff) & 
            (data['ma_diff'].shift(1) <= min_ma_diff) &
            (data['rsi'] < overbought)
        )
        
        # Sell signal: Fast MA crosses below slow MA and RSI not oversold
        sell_condition = (
            (data['ma_diff'] < -min_ma_diff) & 
            (data['ma_diff'].shift(1) >= -min_ma_diff) &
            (data['rsi'] > oversold)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data


class RSIStrategy(BaseStrategy):
    """RSI strategy with divergence detection."""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi_period = self.config.parameters.get('rsi_period', 14)
        ma_period = self.config.parameters.get('ma_period', 21)
        
        if TA_AVAILABLE:
            data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
            data['rsi_ma'] = SMAIndicator(data['rsi'], window=ma_period).sma_indicator()
            
            bb = BollingerBands(data['rsi'], window=20, window_dev=2)
            data['rsi_upper'] = bb.bollinger_hband()
            data['rsi_lower'] = bb.bollinger_lband()
        else:
            data['rsi'] = self._calculate_rsi(data['close'], rsi_period)
            data['rsi_ma'] = self._calculate_sma(data['rsi'], ma_period)
            
            # Simple Bollinger Bands on RSI
            data['rsi_middle'] = data['rsi'].rolling(window=20).mean()
            rsi_std = data['rsi'].rolling(window=20).std()
            data['rsi_upper'] = data['rsi_middle'] + 2 * rsi_std
            data['rsi_lower'] = data['rsi_middle'] - 2 * rsi_std
        
        # Detect divergences
        data = self._calculate_divergences(data)
        data['atr'] = self._calculate_atr(data)
        
        return data
    
    def _calculate_divergences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate divergences between price and RSI."""
        data['price_high'] = data['high'].rolling(window=5, center=True).max()
        data['price_low'] = data['low'].rolling(window=5, center=True).min()
        data['rsi_high'] = data['rsi'].rolling(window=5, center=True).max()
        data['rsi_low'] = data['rsi'].rolling(window=5, center=True).min()
        
        data['price_peak'] = (data['high'] == data['price_high']).astype(int)
        data['price_trough'] = (data['low'] == data['price_low']).astype(int)
        data['rsi_peak'] = (data['rsi'] == data['rsi_high']).astype(int)
        data['rsi_trough'] = (data['rsi'] == data['rsi_low']).astype(int)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        oversold = self.config.parameters.get('rsi_oversold', 30)
        overbought = self.config.parameters.get('rsi_overbought', 70)
        use_divergences = self.config.parameters.get('use_divergences', True)
        
        data['signal'] = 0
        
        # Basic overbought/oversold signals
        buy_condition_basic = (data['rsi'] < oversold) & (data['rsi'].shift(1) >= oversold)
        sell_condition_basic = (data['rsi'] > overbought) & (data['rsi'].shift(1) <= overbought)
        
        if use_divergences:
            # Bullish divergence
            bullish_div = (
                (data['price_trough'] == 1) & 
                (data['rsi_trough'] == 1) &
                (data['low'] < data['low'].shift(2)) &
                (data['rsi'] > data['rsi'].shift(2))
            )
            
            # Bearish divergence
            bearish_div = (
                (data['price_peak'] == 1) & 
                (data['rsi_peak'] == 1) &
                (data['high'] > data['high'].shift(2)) &
                (data['rsi'] < data['rsi'].shift(2))
            )
            
            buy_condition = buy_condition_basic | bullish_div
            sell_condition = sell_condition_basic | bearish_div
        else:
            buy_condition = buy_condition_basic
            sell_condition = sell_condition_basic
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data


class MACDStrategy(BaseStrategy):
    """MACD strategy with histogram analysis."""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_period = self.config.parameters.get('fast_period', 12)
        slow_period = self.config.parameters.get('slow_period', 26)
        signal_period = self.config.parameters.get('signal_period', 9)
        
        if TA_AVAILABLE:
            macd = MACD(
                data['close'], 
                window_slow=slow_period,
                window_fast=fast_period,
                window_sign=signal_period
            )
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_histogram'] = macd.macd_diff()
        else:
            ema_fast = self._calculate_ema(data['close'], fast_period)
            ema_slow = self._calculate_ema(data['close'], slow_period)
            data['macd'] = ema_fast - ema_slow
            data['macd_signal'] = self._calculate_ema(data['macd'], signal_period)
            data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        data['macd_hist_smooth'] = data['macd_histogram'].rolling(window=3).mean()
        data['atr'] = self._calculate_atr(data)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data['signal'] = 0
        
        # Buy signal: MACD crosses above signal line with positive histogram
        buy_condition = (
            (data['macd'] > data['macd_signal']) & 
            (data['macd'].shift(1) <= data['macd_signal'].shift(1)) &
            (data['macd_histogram'] > 0)
        )
        
        # Sell signal: MACD crosses below signal line with negative histogram
        sell_condition = (
            (data['macd'] < data['macd_signal']) & 
            (data['macd'].shift(1) >= data['macd_signal'].shift(1)) &
            (data['macd_histogram'] < 0)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data


class StrategyEngine:
    """Main strategy engine for managing multiple strategies."""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.available_strategies = {
            'ma_crossover': MovingAverageCrossover,
            'rsi': RSIStrategy,
            'macd': MACDStrategy
        }
    
    def create_strategy(self, strategy_type: str, config: StrategyConfig) -> BaseStrategy:
        """Create a new strategy."""
        if strategy_type not in self.available_strategies:
            raise ValueError(f"Strategy type not available: {strategy_type}")
        
        strategy_class = self.available_strategies[strategy_type]
        strategy = strategy_class(config)
        self.strategies[config.name] = strategy
        
        logger.info(f"Strategy created: {config.name} ({strategy_type})")
        return strategy
    
    def run_strategy(self, strategy_name: str, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Run a specific strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.run(symbol, data)
    
    def get_strategy_signals(self, strategy_name: str, symbol: str, 
                           data: pd.DataFrame) -> List[TradeSignal]:
        """Get trading signals from a strategy."""
        result_data = self.run_strategy(strategy_name, symbol, data)
        signals = []
        
        for idx, row in result_data.iterrows():
            if row.get('signal', 0) != 0:
                signal = TradeSignal(
                    symbol=symbol,
                    direction=int(row['signal']),
                    strength=abs(float(row['signal'])),
                    timestamp=idx,
                    price=float(row['close']),
                    stop_loss=float(row.get('stop_loss', 0)) if row.get('stop_loss', 0) != 0 else None,
                    take_profit=float(row.get('take_profit', 0)) if row.get('take_profit', 0) != 0 else None,
                    metadata={
                        'strategy': strategy_name,
                        'rsi': float(row.get('rsi', 0)) if 'rsi' in row else None,
                        'atr': float(row.get('atr', 0)) if 'atr' in row else None
                    }
                )
                signals.append(signal)
        
        return signals
    
    def batch_run_strategies(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[TradeSignal]]:
        """Run multiple strategies on multiple symbols."""
        all_signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            strategy_signals = []
            for symbol in strategy.config.symbols:
                if symbol in data_dict:
                    try:
                        signals = self.get_strategy_signals(strategy_name, symbol, data_dict[symbol])
                        strategy_signals.extend(signals)
                    except Exception as e:
                        logger.error(f"Error running {strategy_name} on {symbol}: {e}")
            
            all_signals[strategy_name] = strategy_signals
        
        return all_signals
    
    def optimize_parameters(self, strategy_type: str, data: pd.DataFrame, 
                          parameter_ranges: Dict[str, List], 
                          metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        from itertools import product
        
        best_params = {}
        best_metric = -np.inf
        
        param_combinations = list(product(*parameter_ranges.values()))
        param_names = list(parameter_ranges.keys())
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            
            try:
                temp_config = StrategyConfig(
                    name="temp_optimization",
                    symbols=["OPTIMIZE"],
                    timeframe="H1",
                    parameters=params
                )
                
                temp_strategy = self.available_strategies[strategy_type](temp_config)
                results = temp_strategy.run("OPTIMIZE", data)
                
                metric_value = self._calculate_performance_metric(results, metric)
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue
        
        logger.info(f"Best parameters: {best_params} with {metric}: {best_metric:.4f}")
        return best_params
    
    def _calculate_performance_metric(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate performance metric."""
        if 'signal' not in data.columns:
            return -np.inf
        
        returns = data['close'].pct_change().shift(-1)
        strategy_returns = returns * data['signal'].shift(1)
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return -np.inf
        
        if metric == 'sharpe_ratio':
            if strategy_returns.std() > 0:
                return strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            return -np.inf
        elif metric == 'total_return':
            return strategy_returns.sum()
        else:
            return strategy_returns.mean()
    
    def list_strategies(self) -> List[str]:
        """List all registered strategies."""
        return list(self.strategies.keys())
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(name)
    
    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy."""
        if name in self.strategies:
            del self.strategies[name]
            return True
        return False
