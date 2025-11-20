# strategies/strategy_engine.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, field
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import numba
from numba import jit

logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
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
    name: str
    symbols: List[str]
    timeframe: str
    parameters: Dict[str, Any]
    risk_management: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
    """Clase base abstracta para todas las estrategias"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.data = {}
        self.signals = []
        self.indicators = {}
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos para la estrategia"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        pass
    
    def calculate_risk_management(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calcular stop loss y take profit"""
        if 'atr' not in data.columns:
            atr = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
            data['atr'] = atr.average_true_range()
        
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        # Estrategia básica de risk management basada en ATR
        atr_multiplier = self.config.risk_management.get('atr_multiplier', 2.0)
        risk_reward = self.config.risk_management.get('risk_reward_ratio', 1.5)
        
        for idx, row in signals.iterrows():
            if row['signal'] != 0:
                current_atr = data.loc[idx, 'atr']
                current_price = data.loc[idx, 'close']
                
                if row['signal'] > 0:  # Long
                    signals.loc[idx, 'stop_loss'] = current_price - (current_atr * atr_multiplier)
                    signals.loc[idx, 'take_profit'] = current_price + (current_atr * atr_multiplier * risk_reward)
                else:  # Short
                    signals.loc[idx, 'stop_loss'] = current_price + (current_atr * atr_multiplier)
                    signals.loc[idx, 'take_profit'] = current_price - (current_atr * atr_multiplier * risk_reward)
        
        return signals
    
    def run(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Ejecutar la estrategia completa"""
        data = data.copy()
        data = self.calculate_indicators(data)
        data = self.generate_signals(data)
        data = self.calculate_risk_management(data, data)
        return data

class MovingAverageCrossover(BaseStrategy):
    """Estrategia de cruce de medias móviles mejorada"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_period = self.config.parameters.get('fast_period', 10)
        slow_period = self.config.parameters.get('slow_period', 20)
        ma_type = self.config.parameters.get('ma_type', 'sma')
        
        if ma_type == 'sma':
            data['ma_fast'] = SMAIndicator(data['close'], window=fast_period).sma_indicator()
            data['ma_slow'] = SMAIndicator(data['close'], window=slow_period).sma_indicator()
        elif ma_type == 'ema':
            data['ma_fast'] = EMAIndicator(data['close'], window=fast_period).ema_indicator()
            data['ma_slow'] = EMAIndicator(data['close'], window=slow_period).ema_indicator()
        
        data['ma_diff'] = data['ma_fast'] - data['ma_slow']
        data['ma_ratio'] = data['ma_fast'] / data['ma_slow'] - 1
        
        # Añadir RSI para confirmación
        rsi_period = self.config.parameters.get('rsi_period', 14)
        data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        oversold = self.config.parameters.get('rsi_oversold', 30)
        overbought = self.config.parameters.get('rsi_overbought', 70)
        min_ma_diff = self.config.parameters.get('min_ma_diff', 0.001)
        
        data['signal'] = 0
        
        # Señal de compra: MA rápida cruza por encima de la lenta y RSI no sobrecomprado
        buy_condition = (
            (data['ma_diff'] > min_ma_diff) & 
            (data['ma_diff'].shift(1) <= min_ma_diff) &
            (data['rsi'] < overbought)
        )
        
        # Señal de venta: MA rápida cruza por debajo de la lenta y RSI no sobrevendido
        sell_condition = (
            (data['ma_diff'] < -min_ma_diff) & 
            (data['ma_diff'].shift(1) >= -min_ma_diff) &
            (data['rsi'] > oversold)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data

class RSIStrategy(BaseStrategy):
    """Estrategia RSI mejorada con divergencias"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi_period = self.config.parameters.get('rsi_period', 14)
        ma_period = self.config.parameters.get('ma_period', 21)
        
        # RSI básico
        data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        
        # Media móvil del RSI
        data['rsi_ma'] = SMAIndicator(data['rsi'], window=ma_period).sma_indicator()
        
        # Detección de divergencias
        data = self._calculate_divergences(data)
        
        # Bandas de Bollinger en RSI
        bb = BollingerBands(data['rsi'], window=20, window_dev=2)
        data['rsi_upper'] = bb.bollinger_hband()
        data['rsi_lower'] = bb.bollinger_lband()
        
        return data
    
    def _calculate_divergences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular divergencias entre precio y RSI"""
        data['price_high'] = data['high'].rolling(window=5, center=True).max()
        data['price_low'] = data['low'].rolling(window=5, center=True).min()
        data['rsi_high'] = data['rsi'].rolling(window=5, center=True).max()
        data['rsi_low'] = data['rsi'].rolling(window=5, center=True).min()
        
        # Detectar máximos y mínimos
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
        
        # Señales básicas de sobrecompra/sobreventa
        buy_condition_basic = data['rsi'] < oversold
        sell_condition_basic = data['rsi'] > overbought
        
        if use_divergences:
            # Divergencia alcista: Precio hace mínimos más bajos pero RSI hace mínimos más altos
            bullish_div = (
                (data['price_trough'] == 1) & 
                (data['rsi_trough'] == 1) &
                (data['low'] < data['low'].shift(2)) &
                (data['rsi'] > data['rsi'].shift(2))
            )
            
            # Divergencia bajista: Precio hace máximos más altos pero RSI hace máximos más bajos
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
    """Estrategia MACD con múltiples timeframe"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_period = self.config.parameters.get('fast_period', 12)
        slow_period = self.config.parameters.get('slow_period', 26)
        signal_period = self.config.parameters.get('signal_period', 9)
        
        macd = MACD(
            data['close'], 
            window_slow=slow_period,
            window_fast=fast_period,
            window_sign=signal_period
        )
        
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        # Histograma suavizado
        data['macd_hist_smooth'] = data['macd_histogram'].rolling(window=3).mean()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data['signal'] = 0
        
        # Señal de compra: MACD cruza por encima de la línea de señal y histograma positivo
        buy_condition = (
            (data['macd'] > data['macd_signal']) & 
            (data['macd'].shift(1) <= data['macd_signal'].shift(1)) &
            (data['macd_histogram'] > 0)
        )
        
        # Señal de venta: MACD cruza por debajo de la línea de señal y histograma negativo
        sell_condition = (
            (data['macd'] < data['macd_signal']) & 
            (data['macd'].shift(1) >= data['macd_signal'].shift(1)) &
            (data['macd_histogram'] < 0)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data

class StrategyEngine:
    """Motor principal de estrategias avanzado"""
    
    def __init__(self):
        self.strategies = {}
        self.available_strategies = {
            'ma_crossover': MovingAverageCrossover,
            'rsi': RSIStrategy,
            'macd': MACDStrategy
        }
        
    def create_strategy(self, strategy_type: str, config: StrategyConfig) -> BaseStrategy:
        """Crear una nueva estrategia"""
        if strategy_type not in self.available_strategies:
            raise ValueError(f"Tipo de estrategia no disponible: {strategy_type}")
        
        strategy_class = self.available_strategies[strategy_type]
        strategy = strategy_class(config)
        self.strategies[config.name] = strategy
        
        logger.info(f"Estrategia creada: {config.name} ({strategy_type})")
        return strategy
    
    def run_strategy(self, strategy_name: str, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Ejecutar una estrategia específica"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Estrategia no encontrada: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.run(symbol, data)
    
    def get_strategy_signals(self, strategy_name: str, symbol: str, 
                           data: pd.DataFrame) -> List[TradeSignal]:
        """Obtener señales de trading de una estrategia"""
        result_data = self.run_strategy(strategy_name, symbol, data)
        signals = []
        
        for idx, row in result_data.iterrows():
            if row['signal'] != 0:
                signal = TradeSignal(
                    symbol=symbol,
                    direction=row['signal'],
                    strength=abs(row['signal']),
                    timestamp=idx,
                    price=row['close'],
                    stop_loss=row.get('stop_loss'),
                    take_profit=row.get('take_profit'),
                    metadata={
                        'strategy': strategy_name,
                        'rsi': row.get('rsi'),
                        'atr': row.get('atr')
                    }
                )
                signals.append(signal)
        
        return signals
    
    def batch_run_strategies(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[TradeSignal]]:
        """Ejecutar múltiples estrategias en múltiples símbolos"""
        all_signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            strategy_signals = []
            for symbol in strategy.config.symbols:
                if symbol in data_dict:
                    signals = self.get_strategy_signals(strategy_name, symbol, data_dict[symbol])
                    strategy_signals.extend(signals)
            
            all_signals[strategy_name] = strategy_signals
        
        return all_signals
    
    def optimize_parameters(self, strategy_type: str, data: pd.DataFrame, 
                          parameter_ranges: Dict[str, List], 
                          metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimizar parámetros de estrategia usando grid search básico"""
        best_params = {}
        best_metric = -np.inf
        
        # Generar todas las combinaciones de parámetros
        from itertools import product
        param_combinations = list(product(*parameter_ranges.values()))
        param_names = list(parameter_ranges.keys())
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            
            try:
                # Crear estrategia temporal con estos parámetros
                temp_config = StrategyConfig(
                    name="temp_optimization",
                    symbols=["OPTIMIZE"],
                    timeframe="H1",
                    parameters=params
                )
                
                temp_strategy = self.available_strategies[strategy_type](temp_config)
                results = temp_strategy.run("OPTIMIZE", data)
                
                # Calcular métrica (simplificado)
                metric_value = self._calculate_performance_metric(results, metric)
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Error en combinación {params}: {e}")
                continue
        
        logger.info(f"Mejores parámetros: {best_params} con {metric}: {best_metric:.4f}")
        return best_params
    
    def _calculate_performance_metric(self, data: pd.DataFrame, metric: str) -> float:
        """Calcular métrica de performance (simplificado)"""
        # Esta es una implementación básica - se expandirá en el módulo de backtesting
        if 'signal' not in data.columns:
            return -np.inf
        
        returns = data['close'].pct_change().shift(-1)
        strategy_returns = returns * data['signal'].shift(1)
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return -np.inf
        
        if metric == 'sharpe_ratio':
            return strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else -np.inf
        elif metric == 'total_return':
            return strategy_returns.sum()
        else:
            return strategy_returns.mean()