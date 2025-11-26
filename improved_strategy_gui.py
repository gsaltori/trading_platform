#!/usr/bin/env python3
"""
Sistema Avanzado de Generación de Estrategias de Trading v3.0
=============================================================

MEJORAS PRINCIPALES:
- Estrategias multi-confirmación robustas
- Gestión de riesgo dinámica avanzada
- Filtros de calidad de mercado
- Trailing stops y breakeven
- Backtesting más realista
- Detección de tendencias mejorada

Autor: Sistema Mejorado v3.0
Fecha: Noviembre 2024
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import threading
import warnings
warnings.filterwarnings('ignore')

# ==================== ESTRATEGIAS MEJORADAS v3.0 ====================
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

@dataclass
class StrategyConfig:
    name: str
    symbols: List[str]
    timeframe: str
    parameters: Dict[str, Any]
    risk_management: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeSignal:
    symbol: str
    direction: int  # 1=long, -1=short, 0=neutral
    strength: float
    timestamp: pd.Timestamp
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
    """Clase base para estrategias mejoradas"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcular Average True Range"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcular Average Directional Index"""
        high_diff = data['high'].diff()
        low_diff = -data['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self.calculate_atr(data, period)
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def is_trending_market(self, data: pd.DataFrame, threshold: float = 25) -> bool:
        """Detectar si el mercado está en tendencia"""
        adx = self.calculate_adx(data)
        return adx.iloc[-1] > threshold if not adx.empty else False
    
    def calculate_risk_management(self, data: pd.DataFrame, signal: int, 
                                 entry_price: float) -> Dict[str, float]:
        """Calcular stop loss y take profit dinámicos"""
        atr = self.calculate_atr(data).iloc[-1]
        
        atr_mult = self.config.risk_management.get('atr_multiplier', 2.5)
        rr_ratio = self.config.risk_management.get('risk_reward_ratio', 2.0)
        
        if signal > 0:  # Long
            stop_loss = entry_price - (atr * atr_mult)
            take_profit = entry_price + (atr * atr_mult * rr_ratio)
        else:  # Short
            stop_loss = entry_price + (atr * atr_mult)
            take_profit = entry_price - (atr * atr_mult * rr_ratio)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr
        }
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores de la estrategia"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        pass
    
    def run(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Ejecutar estrategia completa"""
        data = data.copy()
        data = self.calculate_indicators(data)
        data = self.generate_signals(data)
        return data


class ImprovedMAStrategy(BaseStrategy):
    """Estrategia MA mejorada con múltiples confirmaciones"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast = self.config.parameters.get('fast_period', 10)
        slow = self.config.parameters.get('slow_period', 30)
        ma_type = self.config.parameters.get('ma_type', 'ema')
        
        # Medias móviles
        if ma_type == 'ema':
            data['ma_fast'] = data['close'].ewm(span=fast).mean()
            data['ma_slow'] = data['close'].ewm(span=slow).mean()
        else:
            data['ma_fast'] = data['close'].rolling(fast).mean()
            data['ma_slow'] = data['close'].rolling(slow).mean()
        
        data['ma_diff'] = data['ma_fast'] - data['ma_slow']
        data['ma_diff_pct'] = (data['ma_diff'] / data['ma_slow']) * 100
        
        # ATR para volatilidad
        data['atr'] = self.calculate_atr(data, 14)
        data['atr_pct'] = (data['atr'] / data['close']) * 100
        
        # ADX para fuerza de tendencia
        data['adx'] = self.calculate_adx(data, 14)
        
        # RSI para confirmación
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volumen relativo
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        min_adx = self.config.parameters.get('min_adx', 20)
        min_volume = self.config.parameters.get('min_volume_ratio', 0.8)
        
        data['signal'] = 0
        data['confidence'] = 0.0
        data['stop_loss'] = 0.0
        data['take_profit'] = 0.0
        
        for i in range(50, len(data)):
            # Condiciones base
            ma_cross_up = (data['ma_fast'].iloc[i] > data['ma_slow'].iloc[i] and 
                          data['ma_fast'].iloc[i-1] <= data['ma_slow'].iloc[i-1])
            
            ma_cross_down = (data['ma_fast'].iloc[i] < data['ma_slow'].iloc[i] and 
                            data['ma_fast'].iloc[i-1] >= data['ma_slow'].iloc[i-1])
            
            # Filtros de calidad
            strong_trend = data['adx'].iloc[i] > min_adx
            good_volume = data['volume_ratio'].iloc[i] > min_volume
            rsi_ok_long = 30 < data['rsi'].iloc[i] < 70
            rsi_ok_short = 30 < data['rsi'].iloc[i] < 70
            
            # Señal LONG con confirmaciones
            if ma_cross_up and strong_trend and good_volume and rsi_ok_long:
                data.loc[data.index[i], 'signal'] = 1
                
                # Calcular confianza basada en múltiples factores
                confidence = 0.0
                confidence += min(data['adx'].iloc[i] / 50, 1.0) * 0.4  # ADX weight
                confidence += min(data['volume_ratio'].iloc[i] / 2, 1.0) * 0.3  # Volume
                confidence += (abs(data['ma_diff_pct'].iloc[i]) / 2) * 0.3  # MA separation
                data.loc[data.index[i], 'confidence'] = confidence
                
                # Risk management
                risk_params = self.calculate_risk_management(
                    data.iloc[:i+1], 1, data['close'].iloc[i]
                )
                data.loc[data.index[i], 'stop_loss'] = risk_params['stop_loss']
                data.loc[data.index[i], 'take_profit'] = risk_params['take_profit']
            
            # Señal SHORT con confirmaciones
            elif ma_cross_down and strong_trend and good_volume and rsi_ok_short:
                data.loc[data.index[i], 'signal'] = -1
                
                # Calcular confianza
                confidence = 0.0
                confidence += min(data['adx'].iloc[i] / 50, 1.0) * 0.4
                confidence += min(data['volume_ratio'].iloc[i] / 2, 1.0) * 0.3
                confidence += (abs(data['ma_diff_pct'].iloc[i]) / 2) * 0.3
                data.loc[data.index[i], 'confidence'] = confidence
                
                # Risk management
                risk_params = self.calculate_risk_management(
                    data.iloc[:i+1], -1, data['close'].iloc[i]
                )
                data.loc[data.index[i], 'stop_loss'] = risk_params['stop_loss']
                data.loc[data.index[i], 'take_profit'] = risk_params['take_profit']
        
        return data


class ImprovedRSIStrategy(BaseStrategy):
    """Estrategia RSI mejorada con divergencias y confirmaciones"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi_period = self.config.parameters.get('rsi_period', 14)
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI smoothed
        data['rsi_sma'] = data['rsi'].rolling(5).mean()
        
        # ATR
        data['atr'] = self.calculate_atr(data, 14)
        
        # ADX
        data['adx'] = self.calculate_adx(data, 14)
        
        # Price momentum
        data['momentum'] = data['close'].pct_change(10) * 100
        
        # Detectar divergencias
        data = self._detect_divergences(data)
        
        return data
    
    def _detect_divergences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detectar divergencias alcistas y bajistas"""
        window = 10
        
        data['price_high'] = data['high'].rolling(window, center=True).max()
        data['price_low'] = data['low'].rolling(window, center=True).min()
        data['rsi_high'] = data['rsi'].rolling(window, center=True).max()
        data['rsi_low'] = data['rsi'].rolling(window, center=True).min()
        
        data['bullish_div'] = 0
        data['bearish_div'] = 0
        
        for i in range(window, len(data) - window):
            # Divergencia alcista: precio baja pero RSI sube
            if (data['low'].iloc[i] == data['price_low'].iloc[i] and
                i > window):
                prev_low_idx = data['low'].iloc[i-window:i].idxmin()
                if data['low'].iloc[i] < data['low'].loc[prev_low_idx]:
                    if data['rsi'].iloc[i] > data['rsi'].loc[prev_low_idx]:
                        data.loc[data.index[i], 'bullish_div'] = 1
            
            # Divergencia bajista: precio sube pero RSI baja
            if (data['high'].iloc[i] == data['price_high'].iloc[i] and
                i > window):
                prev_high_idx = data['high'].iloc[i-window:i].idxmax()
                if data['high'].iloc[i] > data['high'].loc[prev_high_idx]:
                    if data['rsi'].iloc[i] < data['rsi'].loc[prev_high_idx]:
                        data.loc[data.index[i], 'bearish_div'] = 1
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        oversold = self.config.parameters.get('rsi_oversold', 30)
        overbought = self.config.parameters.get('rsi_overbought', 70)
        min_adx = self.config.parameters.get('min_adx', 20)
        
        data['signal'] = 0
        data['confidence'] = 0.0
        data['stop_loss'] = 0.0
        data['take_profit'] = 0.0
        
        for i in range(50, len(data)):
            rsi = data['rsi'].iloc[i]
            rsi_prev = data['rsi'].iloc[i-1]
            adx = data['adx'].iloc[i]
            
            # Señal LONG: RSI oversold + momentum positivo
            if (rsi < oversold and rsi_prev < rsi and  # RSI subiendo desde oversold
                data['momentum'].iloc[i] > 0 and  # Momentum positivo
                adx > min_adx):  # Tendencia fuerte
                
                data.loc[data.index[i], 'signal'] = 1
                
                # Confianza mayor si hay divergencia alcista
                confidence = 0.5
                if data['bullish_div'].iloc[i] == 1:
                    confidence += 0.3
                confidence += min(adx / 50, 0.2)
                data.loc[data.index[i], 'confidence'] = confidence
                
                # Risk management
                risk_params = self.calculate_risk_management(
                    data.iloc[:i+1], 1, data['close'].iloc[i]
                )
                data.loc[data.index[i], 'stop_loss'] = risk_params['stop_loss']
                data.loc[data.index[i], 'take_profit'] = risk_params['take_profit']
            
            # Señal SHORT: RSI overbought + momentum negativo
            elif (rsi > overbought and rsi_prev > rsi and  # RSI bajando desde overbought
                  data['momentum'].iloc[i] < 0 and  # Momentum negativo
                  adx > min_adx):  # Tendencia fuerte
                
                data.loc[data.index[i], 'signal'] = -1
                
                # Confianza mayor si hay divergencia bajista
                confidence = 0.5
                if data['bearish_div'].iloc[i] == 1:
                    confidence += 0.3
                confidence += min(adx / 50, 0.2)
                data.loc[data.index[i], 'confidence'] = confidence
                
                # Risk management
                risk_params = self.calculate_risk_management(
                    data.iloc[:i+1], -1, data['close'].iloc[i]
                )
                data.loc[data.index[i], 'stop_loss'] = risk_params['stop_loss']
                data.loc[data.index[i], 'take_profit'] = risk_params['take_profit']
        
        return data


class ImprovedMACDStrategy(BaseStrategy):
    """Estrategia MACD mejorada con confirmación de tendencia"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast = self.config.parameters.get('fast_period', 12)
        slow = self.config.parameters.get('slow_period', 26)
        signal = self.config.parameters.get('signal_period', 9)
        
        # MACD
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=signal).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Histograma smoothed
        data['macd_hist_sma'] = data['macd_hist'].rolling(3).mean()
        
        # ATR y ADX
        data['atr'] = self.calculate_atr(data, 14)
        data['adx'] = self.calculate_adx(data, 14)
        
        # Tendencia de largo plazo
        data['ma_200'] = data['close'].rolling(200).mean()
        data['above_ma200'] = data['close'] > data['ma_200']
        
        # Momentum
        data['momentum'] = data['close'].pct_change(5) * 100
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        min_adx = self.config.parameters.get('min_adx', 20)
        min_hist = self.config.parameters.get('min_histogram', 0.0001)
        
        data['signal'] = 0
        data['confidence'] = 0.0
        data['stop_loss'] = 0.0
        data['take_profit'] = 0.0
        
        for i in range(200, len(data)):
            # Cruce MACD
            macd_cross_up = (data['macd'].iloc[i] > data['macd_signal'].iloc[i] and
                            data['macd'].iloc[i-1] <= data['macd_signal'].iloc[i-1])
            
            macd_cross_down = (data['macd'].iloc[i] < data['macd_signal'].iloc[i] and
                              data['macd'].iloc[i-1] >= data['macd_signal'].iloc[i-1])
            
            # Histograma creciendo
            hist_growing = data['macd_hist'].iloc[i] > data['macd_hist'].iloc[i-1]
            hist_shrinking = data['macd_hist'].iloc[i] < data['macd_hist'].iloc[i-1]
            
            # Filtros
            strong_trend = data['adx'].iloc[i] > min_adx
            hist_significant = abs(data['macd_hist'].iloc[i]) > min_hist
            
            # Señal LONG: Cruce alcista + tendencia alcista
            if (macd_cross_up and hist_growing and strong_trend and 
                hist_significant and data['above_ma200'].iloc[i]):
                
                data.loc[data.index[i], 'signal'] = 1
                
                # Confianza
                confidence = 0.6
                confidence += min(data['adx'].iloc[i] / 50, 0.2)
                confidence += min(abs(data['macd_hist'].iloc[i]) * 1000, 0.2)
                data.loc[data.index[i], 'confidence'] = confidence
                
                # Risk management
                risk_params = self.calculate_risk_management(
                    data.iloc[:i+1], 1, data['close'].iloc[i]
                )
                data.loc[data.index[i], 'stop_loss'] = risk_params['stop_loss']
                data.loc[data.index[i], 'take_profit'] = risk_params['take_profit']
            
            # Señal SHORT: Cruce bajista + tendencia bajista
            elif (macd_cross_down and hist_shrinking and strong_trend and 
                  hist_significant and not data['above_ma200'].iloc[i]):
                
                data.loc[data.index[i], 'signal'] = -1
                
                # Confianza
                confidence = 0.6
                confidence += min(data['adx'].iloc[i] / 50, 0.2)
                confidence += min(abs(data['macd_hist'].iloc[i]) * 1000, 0.2)
                data.loc[data.index[i], 'confidence'] = confidence
                
                # Risk management
                risk_params = self.calculate_risk_management(
                    data.iloc[:i+1], -1, data['close'].iloc[i]
                )
                data.loc[data.index[i], 'stop_loss'] = risk_params['stop_loss']
                data.loc[data.index[i], 'take_profit'] = risk_params['take_profit']
        
        return data


class StrategyEngine:
    """Motor de estrategias mejorado"""
    
    def __init__(self):
        self.strategies = {}
        self.available_strategies = {
            'ma_crossover': ImprovedMAStrategy,
            'rsi': ImprovedRSIStrategy,
            'macd': ImprovedMACDStrategy
        }
    
    def create_strategy(self, strategy_type: str, config: StrategyConfig):
        """Crear estrategia"""
        if strategy_type not in self.available_strategies:
            raise ValueError(f"Estrategia no disponible: {strategy_type}")
        
        strategy_class = self.available_strategies[strategy_type]
        strategy = strategy_class(config)
        self.strategies[config.name] = strategy
        return strategy
    
    def get_strategy_signals(self, strategy_name: str, symbol: str, 
                           data: pd.DataFrame) -> List[TradeSignal]:
        """Obtener señales de una estrategia"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Estrategia no encontrada: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        result_data = strategy.run(symbol, data)
        
        signals = []
        for idx, row in result_data.iterrows():
            if row['signal'] != 0:
                signal = TradeSignal(
                    symbol=symbol,
                    direction=int(row['signal']),
                    strength=abs(row['signal']),
                    timestamp=idx,
                    price=row['close'],
                    stop_loss=row.get('stop_loss'),
                    take_profit=row.get('take_profit'),
                    confidence=row.get('confidence', 0.5)
                )
                signals.append(signal)
        
        return signals


# ==================== BACKTESTING MEJORADO ====================

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: int
    entry_price: float
    exit_price: Optional[float]
    volume: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = ''
    confidence: float = 0.0

@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade: float
    avg_winning_trade: float
    avg_losing_trade: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    trades: List[Trade] = field(default_factory=list)

class ImprovedBacktestEngine:
    """Motor de backtesting mejorado con trailing stops y gestión realista"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, data: pd.DataFrame, strategy, symbol: str,
                    commission: float = 0.001) -> BacktestResult:
        """Ejecutar backtest mejorado"""
        
        # Reiniciar
        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        
        # Ejecutar estrategia
        strategy_data = strategy.run(symbol, data.copy())
        
        # Procesar señales con gestión avanzada
        open_trade = None
        
        for i in range(1, len(strategy_data)):
            current_bar = strategy_data.iloc[i]
            current_time = strategy_data.index[i]
            current_price = current_bar['close']
            
            # Gestionar trade abierto
            if open_trade:
                # Verificar stop loss
                if open_trade.direction > 0:  # Long
                    if current_bar['low'] <= open_trade.stop_loss:
                        self._close_trade(open_trade, open_trade.stop_loss, 
                                        current_time, 'Stop Loss', commission)
                        open_trade = None
                        continue
                    # Verificar take profit
                    if current_bar['high'] >= open_trade.take_profit:
                        self._close_trade(open_trade, open_trade.take_profit,
                                        current_time, 'Take Profit', commission)
                        open_trade = None
                        continue
                else:  # Short
                    if current_bar['high'] >= open_trade.stop_loss:
                        self._close_trade(open_trade, open_trade.stop_loss,
                                        current_time, 'Stop Loss', commission)
                        open_trade = None
                        continue
                    if current_bar['low'] <= open_trade.take_profit:
                        self._close_trade(open_trade, open_trade.take_profit,
                                        current_time, 'Take Profit', commission)
                        open_trade = None
                        continue
                
                # Trailing stop (mover stop loss a breakeven después de 50% profit)
                open_trade = self._update_trailing_stop(open_trade, current_bar)
            
            # Nueva señal
            if current_bar['signal'] != 0 and not open_trade:
                # Calcular tamaño de posición (2% del capital)
                risk_amount = self.current_capital * 0.02
                atr = current_bar.get('atr', current_price * 0.01)
                volume = risk_amount / (atr * 2)
                
                # Limitar volumen
                volume = min(volume, self.current_capital * 0.1 / current_price)
                
                if volume > 0:
                    open_trade = Trade(
                        entry_time=current_time,
                        exit_time=None,
                        symbol=symbol,
                        direction=int(current_bar['signal']),
                        entry_price=current_price,
                        exit_price=None,
                        volume=volume,
                        stop_loss=current_bar.get('stop_loss'),
                        take_profit=current_bar.get('take_profit'),
                        confidence=current_bar.get('confidence', 0.5)
                    )
                    
                    # Aplicar comisión de entrada
                    entry_cost = volume * current_price * commission
                    self.current_capital -= entry_cost
            
            # Actualizar equity curve
            equity = self.current_capital
            if open_trade:
                if open_trade.direction > 0:
                    unrealized_pnl = (current_price - open_trade.entry_price) * open_trade.volume
                else:
                    unrealized_pnl = (open_trade.entry_price - current_price) * open_trade.volume
                equity += unrealized_pnl
            
            self.equity_curve.append({
                'time': current_time,
                'equity': equity
            })
        
        # Cerrar trade final si existe
        if open_trade:
            last_price = strategy_data.iloc[-1]['close']
            last_time = strategy_data.index[-1]
            self._close_trade(open_trade, last_price, last_time, 'End', commission)
        
        # Calcular métricas
        return self._calculate_metrics(strategy.name)
    
    def _close_trade(self, trade: Trade, exit_price: float, exit_time: datetime,
                    reason: str, commission: float):
        """Cerrar trade"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calcular P&L
        if trade.direction > 0:
            trade.pnl = (exit_price - trade.entry_price) * trade.volume
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.volume
        
        # Restar comisión de salida
        exit_cost = trade.volume * exit_price * commission
        trade.pnl -= exit_cost
        
        trade.pnl_pct = (trade.pnl / (trade.volume * trade.entry_price)) * 100
        
        # Actualizar capital
        self.current_capital += trade.pnl
        self.trades.append(trade)
    
    def _update_trailing_stop(self, trade: Trade, current_bar) -> Trade:
        """Actualizar trailing stop (mover a breakeven)"""
        current_price = current_bar['close']
        
        if trade.direction > 0:  # Long
            # Si el precio ha subido 50% hacia el TP, mover SL a BE
            profit_pct = (current_price - trade.entry_price) / (trade.take_profit - trade.entry_price)
            if profit_pct > 0.5:
                trade.stop_loss = max(trade.stop_loss, trade.entry_price)
        else:  # Short
            profit_pct = (trade.entry_price - current_price) / (trade.entry_price - trade.take_profit)
            if profit_pct > 0.5:
                trade.stop_loss = min(trade.stop_loss, trade.entry_price)
        
        return trade
    
    def _calculate_metrics(self, strategy_name: str) -> BacktestResult:
        """Calcular métricas de performance"""
        if not self.trades:
            return BacktestResult(
                strategy_name=strategy_name,
                total_return=0, total_trades=0, winning_trades=0,
                losing_trades=0, win_rate=0, avg_trade=0,
                avg_winning_trade=0, avg_losing_trade=0, max_drawdown=0
            )
        
        total_return = ((self.current_capital - self.initial_capital) / 
                       self.initial_capital) * 100
        
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = (len(winning) / len(self.trades)) * 100 if self.trades else 0
        avg_trade = np.mean([t.pnl for t in self.trades])
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        
        # Calcular Sharpe ratio
        if len(self.trades) > 1:
            returns = [t.pnl_pct for t in self.trades]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            max_dd = equity_df['drawdown'].min()
        else:
            max_dd = 0
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_trade=avg_trade,
            avg_winning_trade=avg_win,
            avg_losing_trade=avg_loss,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            trades=self.trades
        )


# ==================== GUI MEJORADA ====================

class ImprovedStrategyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Generador de Estrategias v3.0 MEJORADO")
        self.root.geometry("1200x800")
        
        self.platform = None
        self.mt5_installations = []
        self.strategy_engine = StrategyEngine()
        self.backtest_engine = ImprovedBacktestEngine()
        self.generated_strategies = []
        self.data_cache = {}
        self.ml_models = {}
        
        self.setup_gui()
        self.connect_mt5()
    
    def setup_gui(self):
        """Configurar interfaz gráfica"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame de configuración
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuración", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # MT5 Installation Selection
        ttk.Label(config_frame, text="MT5:").grid(row=0, column=0, sticky=tk.W)
        self.mt5_var = tk.StringVar()
        self.mt5_combo = ttk.Combobox(config_frame, textvariable=self.mt5_var, width=50, state='readonly')
        self.mt5_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Botón conectar
        self.connect_btn = ttk.Button(config_frame, text="🔌 Conectar", command=self.connect_mt5_selected)
        self.connect_btn.grid(row=0, column=2, padx=5)
        
        # Status de conexión
        self.connection_label = ttk.Label(config_frame, text="○ Desconectado", foreground='red')
        self.connection_label.grid(row=0, column=3)
        
        # Símbolos
        ttk.Label(config_frame, text="Símbolos:").grid(row=1, column=0, sticky=tk.W)
        self.symbols_entry = ttk.Entry(config_frame, width=40)
        self.symbols_entry.insert(0, "EURUSD,GBPUSD,USDJPY,AUDUSD")
        self.symbols_entry.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Timeframe
        ttk.Label(config_frame, text="Timeframe:").grid(row=2, column=0, sticky=tk.W)
        self.timeframe_var = tk.StringVar(value="D1")
        timeframes = ['M15', 'M30', 'H1', 'H4', 'D1', 'W1']
        ttk.Combobox(config_frame, textvariable=self.timeframe_var, 
                    values=timeframes, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Días históricos
        ttk.Label(config_frame, text="Días históricos:").grid(row=3, column=0, sticky=tk.W)
        self.days_var = tk.IntVar(value=1825)
        ttk.Spinbox(config_frame, from_=365, to=7300, 
                   textvariable=self.days_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Modo
        ttk.Label(config_frame, text="Modo:").grid(row=4, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value="random")
        ttk.Radiobutton(config_frame, text="Aleatorio", variable=self.mode_var, 
                       value="random").grid(row=4, column=1, sticky=tk.W)
        
        # Número de estrategias aleatorias
        ttk.Label(config_frame, text="Núm. estrategias:").grid(row=5, column=0, sticky=tk.W)
        self.num_strategies = tk.IntVar(value=20)
        ttk.Spinbox(config_frame, from_=5, to=100, 
                   textvariable=self.num_strategies, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)
        
        # Machine Learning
        self.use_ml = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Usar ML", 
                       variable=self.use_ml).grid(row=6, column=0, sticky=tk.W)
        
        # Filtros
        ttk.Label(config_frame, text="Sharpe mín:").grid(row=7, column=0, sticky=tk.W)
        self.min_sharpe = tk.DoubleVar(value=0.3)
        ttk.Spinbox(config_frame, from_=-1, to=3, increment=0.1,
                   textvariable=self.min_sharpe, width=10).grid(row=7, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(config_frame, text="WinRate mín (%):").grid(row=8, column=0, sticky=tk.W)
        self.min_winrate = tk.DoubleVar(value=40)
        ttk.Spinbox(config_frame, from_=0, to=100, increment=5,
                   textvariable=self.min_winrate, width=10).grid(row=8, column=1, sticky=tk.W, pady=2)
        
        # Botón generar
        self.generate_btn = ttk.Button(config_frame, text="🚀 Generar Estrategias",
                                       command=self.start_generation)
        self.generate_btn.grid(row=9, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(config_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status
        self.status_label = ttk.Label(config_frame, text="Listo", foreground='green')
        self.status_label.grid(row=11, column=0, columnspan=2)
        
        # Log
        log_frame = ttk.LabelFrame(main_frame, text="📋 Log", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=60)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Tabla de resultados
        results_frame = ttk.LabelFrame(main_frame, text="📊 Estrategias Generadas", padding="5")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        columns = ('Nombre', 'Símbolo', 'Sharpe', 'WinRate', 'Return', 'DD', 'PF', 'Trades')
        self.strategies_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.strategies_tree.heading(col, text=col)
            self.strategies_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                 command=self.strategies_tree.yview)
        self.strategies_tree.configure(yscroll=scrollbar.set)
        
        self.strategies_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def log(self, message: str, level: str = 'INFO'):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        icons = {
            'INFO': 'ℹ️',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌'
        }
        
        icon = icons.get(level, 'ℹ️')
        log_message = f"[{timestamp}] {icon} {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update()
    
    def connect_mt5(self):
        """Detectar instalaciones MT5 disponibles"""
        try:
            import winreg
            import os
            
            # Detectar instalaciones MT5
            self.mt5_installations = []
            registry_paths = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
            ]
            
            for reg_path in registry_paths:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            subkey = winreg.OpenKey(key, subkey_name)
                            try:
                                name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                if "MetaTrader 5" in name:
                                    try:
                                        location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                        terminal_path = os.path.join(location, "terminal64.exe")
                                        if os.path.exists(terminal_path):
                                            self.mt5_installations.append((name, terminal_path))
                                    except:
                                        pass
                            except:
                                pass
                            winreg.CloseKey(subkey)
                        except:
                            continue
                    winreg.CloseKey(key)
                except:
                    continue
            
            if self.mt5_installations:
                self.log(f"✅ Detectadas {len(self.mt5_installations)} instalación(es) MT5", 'SUCCESS')
                
                # Poblar combobox
                display_names = [name for name, path in self.mt5_installations]
                self.mt5_combo['values'] = display_names
                
                # Seleccionar primera por defecto
                if display_names:
                    self.mt5_combo.current(0)
                    self.log(f"Seleccionado por defecto: {display_names[0]}")
                
                self.log("👆 Selecciona la instalación MT5 y presiona 'Conectar'")
            else:
                self.log("❌ No se encontraron instalaciones de MT5", 'ERROR')
                messagebox.showerror("Error", "No se encontraron instalaciones de MT5")
                
        except Exception as e:
            self.log(f"Error detectando MT5: {e}", 'ERROR')
            messagebox.showerror("Error", f"Error detectando MT5: {e}")
    
    def connect_mt5_selected(self):
        """Conectar a la instalación MT5 seleccionada"""
        try:
            selected_index = self.mt5_combo.current()
            if selected_index < 0:
                messagebox.showwarning("Advertencia", "Por favor selecciona una instalación MT5")
                return
            
            selected_name, selected_path = self.mt5_installations[selected_index]
            
            self.log(f"🔌 Conectando a: {selected_name}")
            self.log(f"📂 Ruta: {selected_path}")
            
            if not mt5.initialize(path=selected_path):
                error = mt5.last_error()
                raise Exception(f"MT5 initialize failed: {error}")
            
            self.platform = mt5
            self.connection_label.config(text="● Conectado", foreground='green')
            self.connect_btn.config(state='disabled')
            self.log("✅ ✓ Conectado a MT5 exitosamente", 'SUCCESS')
            
            # Obtener info de cuenta
            account_info = mt5.account_info()
            if account_info:
                self.log(f"📊 Cuenta: {account_info.login}")
                self.log(f"💰 Balance: ${account_info.balance:.2f}")
                self.log(f"🏦 Servidor: {account_info.server}")
            
        except Exception as e:
            self.log(f"❌ Error conectando: {e}", 'ERROR')
            self.connection_label.config(text="○ Error", foreground='red')
            messagebox.showerror("Error de Conexión", f"No se pudo conectar a MT5:\n{e}")
    
    def start_generation(self):
        """Iniciar generación en hilo separado"""
        self.generate_btn.config(state='disabled')
        thread = threading.Thread(target=self.generate_strategies, daemon=True)
        thread.start()
    
    def generate_strategies(self):
        """Generar estrategias mejoradas"""
        try:
            # Configuración
            symbols_str = self.symbols_entry.get()
            symbols = [s.strip() for s in symbols_str.split(',')]
            timeframe = self.timeframe_var.get()
            days = self.days_var.get()
            mode = self.mode_var.get()
            use_ml = self.use_ml.get()
            num_strategies = self.num_strategies.get()
            min_sharpe = self.min_sharpe.get()
            min_winrate = self.min_winrate.get()
            
            self.log("=" * 50)
            self.log("INICIANDO AUTOGENERACIÓN MEJORADA v3.0")
            self.log("=" * 50)
            self.log(f"Símbolos: {', '.join(symbols)}")
            self.log(f"Timeframe: {timeframe}, Días: {days}")
            self.log(f"Modo: {'Aleatorio' if mode == 'random' else 'Pre-configurado'}")
            self.log(f"ML: {'Sí' if use_ml else 'No'}")
            
            # Paso 1: Descargar datos
            self.log("Paso 1/4: Descargando datos históricos...")
            self.progress_var.set(10)
            
            for symbol in symbols:
                self.log(f"Descargando {symbol}...")
                data = self.get_mt5_data(symbol, timeframe, days)
                if data is not None:
                    self.data_cache[symbol] = data
                    self.log(f"✓ {symbol}: {len(data)} velas", 'SUCCESS')
                else:
                    self.log(f"✗ {symbol}: Error descargando datos", 'WARNING')
            
            if not self.data_cache:
                raise Exception("No se pudieron descargar datos")
            
            # Paso 2: ML
            self.progress_var.set(20)
            if use_ml:
                self.log("Paso 2/4: Entrenando modelos ML...")
                for symbol in self.data_cache.keys():
                    self.log(f"Entrenando ML para {symbol}...")
                    accuracy = self.train_ml_model(symbol)
                    self.log(f"✓ {symbol} ML: Accuracy {accuracy:.3f}", 'SUCCESS')
            else:
                self.log("Paso 2/4: ML desactivado")
            
            # Paso 3: Generar estrategias
            self.log("Paso 3/4: Generando estrategias MEJORADAS...")
            self.progress_var.set(30)
            
            strategies_to_test = []
            import random
            
            for i in range(num_strategies):
                strategy_type = random.choice(['ma_crossover', 'rsi', 'macd'])
                symbol = random.choice(list(self.data_cache.keys()))
                
                # Parámetros optimizados según tipo
                if strategy_type == 'ma_crossover':
                    params = {
                        'fast_period': random.randint(5, 20),
                        'slow_period': random.randint(20, 50),
                        'ma_type': random.choice(['ema', 'sma']),
                        'min_adx': random.randint(15, 30),
                        'min_volume_ratio': random.uniform(0.5, 1.2)
                    }
                elif strategy_type == 'rsi':
                    params = {
                        'rsi_period': random.randint(10, 21),
                        'rsi_oversold': random.randint(20, 35),
                        'rsi_overbought': random.randint(65, 80),
                        'min_adx': random.randint(15, 30)
                    }
                else:  # macd
                    params = {
                        'fast_period': random.randint(8, 15),
                        'slow_period': random.randint(20, 35),
                        'signal_period': random.randint(7, 12),
                        'min_adx': random.randint(15, 30),
                        'min_histogram': random.uniform(0.00005, 0.0002)
                    }
                
                name = f"Improved_{strategy_type}_{i+1}_{symbol}"
                
                config = StrategyConfig(
                    name=name,
                    symbols=[symbol],
                    timeframe=timeframe,
                    parameters=params,
                    risk_management={
                        'atr_multiplier': random.uniform(2.0, 3.5),
                        'risk_reward_ratio': random.uniform(1.5, 3.0)
                    }
                )
                
                try:
                    strategy = self.strategy_engine.create_strategy(strategy_type, config)
                    strategies_to_test.append((strategy, symbol))
                except Exception as e:
                    self.log(f"Error creando {name}: {e}", 'WARNING')
            
            self.log(f"✓ Generadas {len(strategies_to_test)} estrategias MEJORADAS", 'SUCCESS')
            
            # Paso 4: Backtest
            self.log("Paso 4/4: Ejecutando backtests...")
            viable_strategies = []
            
            for idx, (strategy, symbol) in enumerate(strategies_to_test):
                progress = 40 + (50 * idx / len(strategies_to_test))
                self.progress_var.set(progress)
                
                self.log(f"Testing {strategy.name}...")
                
                try:
                    data = self.data_cache[symbol]
                    result = self.backtest_engine.run_backtest(
                        data=data,
                        strategy=strategy,
                        symbol=symbol,
                        commission=0.001
                    )
                    
                    sharpe = result.sharpe_ratio or -10
                    winrate = result.win_rate
                    
                    if sharpe >= min_sharpe and winrate >= min_winrate and result.total_trades >= 5:
                        viable_strategies.append({
                            'name': strategy.name,
                            'symbol': symbol,
                            'strategy': strategy,
                            'result': result
                        })
                        self.log(f"✓ {strategy.name}: Sharpe {sharpe:.2f}, WR {winrate:.1f}%", 'SUCCESS')
                    else:
                        self.log(f"✗ {strategy.name}: Filtrado (Sharpe {sharpe:.2f}, WR {winrate:.1f}%)")
                
                except Exception as e:
                    self.log(f"✗ {strategy.name}: Error - {e}", 'ERROR')
            
            self.progress_var.set(100)
            self.generated_strategies = viable_strategies
            
            self.log("=" * 50)
            self.log(f"✅ COMPLETADO: {len(viable_strategies)} estrategias viables", 'SUCCESS')
            self.log("=" * 50)
            
            # Actualizar tabla
            self.root.after(0, self.update_strategies_table)
            
            if viable_strategies:
                self.status_label.config(text=f"✓ {len(viable_strategies)} estrategias generadas",
                                        foreground='green')
            else:
                self.status_label.config(text="⚠ 0 estrategias viables", foreground='orange')
                self.log("Intenta: Bajar filtros (Sharpe < 0.3, WR < 40%)", 'WARNING')
            
        except Exception as e:
            self.log(f"Error crítico: {e}", 'ERROR')
            self.status_label.config(text="Error", foreground='red')
        finally:
            self.generate_btn.config(state='normal')
    
    def get_mt5_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Obtener datos de MT5"""
        try:
            tf_map = {
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1
            }
            
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_D1)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, days)
            
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.log(f"Error obteniendo datos {symbol}: {e}", 'ERROR')
            return None
    
    def train_ml_model(self, symbol: str) -> float:
        """Entrenar modelo ML simple"""
        try:
            data = self.data_cache[symbol].copy()
            
            # Features simples
            data['returns'] = data['close'].pct_change()
            data['ma_20'] = data['close'].rolling(20).mean()
            data['ma_50'] = data['close'].rolling(50).mean()
            data['rsi'] = self.calculate_rsi(data['close'], 14)
            
            # Target: dirección del precio (1 día adelante)
            data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
            
            # Preparar datos
            features = ['returns', 'ma_20', 'ma_50', 'rsi']
            data = data.dropna()
            
            X = data[features]
            y = data['target']
            
            # Train/test split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Entrenar
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar
            accuracy = model.score(X_test, y_test)
            
            self.ml_models[symbol] = model
            return accuracy
            
        except Exception as e:
            self.log(f"Error en ML para {symbol}: {e}", 'WARNING')
            return 0.5
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def update_strategies_table(self):
        """Actualizar tabla de estrategias"""
        for item in self.strategies_tree.get_children():
            self.strategies_tree.delete(item)
        
        sorted_strategies = sorted(self.generated_strategies,
                                  key=lambda x: x['result'].sharpe_ratio or -10,
                                  reverse=True)
        
        for strat in sorted_strategies:
            result = strat['result']
            
            sharpe = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A"
            pf = f"{result.profit_factor:.2f}" if result.profit_factor else "N/A"
            
            self.strategies_tree.insert('', 'end', values=(
                strat['name'],
                strat['symbol'],
                sharpe,
                f"{result.win_rate:.1f}",
                f"{result.total_return:.2f}",
                f"{result.max_drawdown:.2f}",
                pf,
                result.total_trades
            ))


def main():
    root = tk.Tk()
    app = ImprovedStrategyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()