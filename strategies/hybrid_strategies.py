# strategies/hybrid_strategies.py
import pandas as pd
import numpy as np
from strategies.strategy_engine import BaseStrategy
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class MARSIMACDStrategy(BaseStrategy):
    """
    Estrategia Híbrida: MA + RSI + MACD
    
    Requiere confirmación de los 3 indicadores para señal
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        ma_fast = self.config.parameters.get('ma_fast', 12)
        ma_slow = self.config.parameters.get('ma_slow', 26)
        rsi_period = self.config.parameters.get('rsi_period', 14)
        macd_fast = self.config.parameters.get('macd_fast', 12)
        macd_slow = self.config.parameters.get('macd_slow', 26)
        macd_signal = self.config.parameters.get('macd_signal', 9)
        
        # Moving Averages
        data['ma_fast'] = EMAIndicator(data['close'], window=ma_fast).ema_indicator()
        data['ma_slow'] = EMAIndicator(data['close'], window=ma_slow).ema_indicator()
        
        # RSI
        data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        
        # MACD
        macd = MACD(data['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi_oversold = self.config.parameters.get('rsi_oversold', 30)
        rsi_overbought = self.config.parameters.get('rsi_overbought', 70)
        
        data['signal'] = 0
        
        # COMPRA: Los 3 indicadores alcistas
        buy_condition = (
            (data['ma_fast'] > data['ma_slow']) &  # MA alcista
            (data['rsi'] < rsi_overbought) &  # RSI no sobrecomprado
            (data['macd'] > data['macd_signal']) &  # MACD alcista
            (data['macd_histogram'] > 0)  # Histograma positivo
        )
        
        # VENTA: Los 3 indicadores bajistas
        sell_condition = (
            (data['ma_fast'] < data['ma_slow']) &  # MA bajista
            (data['rsi'] > rsi_oversold) &  # RSI no sobrevendido
            (data['macd'] < data['macd_signal']) &  # MACD bajista
            (data['macd_histogram'] < 0)  # Histograma negativo
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data


class BBRSIStrategy(BaseStrategy):
    """
    Estrategia Híbrida: Bollinger Bands + RSI
    
    Combina reversión a la media con momentum
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        bb_period = self.config.parameters.get('bb_period', 20)
        bb_std = self.config.parameters.get('bb_std', 2.0)
        rsi_period = self.config.parameters.get('rsi_period', 14)
        
        # Bollinger Bands
        bb = BollingerBands(data['close'], window=bb_period, window_dev=bb_std)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # RSI
        data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi_oversold = self.config.parameters.get('rsi_oversold', 30)
        rsi_overbought = self.config.parameters.get('rsi_overbought', 70)
        
        data['signal'] = 0
        
        # COMPRA: Precio en banda inferior + RSI oversold
        buy_condition = (
            (data['bb_position'] < 0.2) &
            (data['rsi'] < rsi_oversold)
        )
        
        # VENTA: Precio en banda superior + RSI overbought
        sell_condition = (
            (data['bb_position'] > 0.8) &
            (data['rsi'] > rsi_overbought)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data


class ADXMARSIStrategy(BaseStrategy):
    """
    Estrategia Híbrida: ADX + MA + RSI
    
    Opera solo en tendencias fuertes
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        adx_period = self.config.parameters.get('adx_period', 14)
        ma_fast = self.config.parameters.get('ma_fast', 20)
        ma_slow = self.config.parameters.get('ma_slow', 50)
        rsi_period = self.config.parameters.get('rsi_period', 14)
        
        # ADX
        adx = ADXIndicator(data['high'], data['low'], data['close'], window=adx_period)
        data['adx'] = adx.adx()
        data['di_plus'] = adx.adx_pos()
        data['di_minus'] = adx.adx_neg()
        
        # Moving Averages
        data['ma_fast'] = EMAIndicator(data['close'], window=ma_fast).ema_indicator()
        data['ma_slow'] = SMAIndicator(data['close'], window=ma_slow).sma_indicator()
        
        # RSI
        data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        adx_threshold = self.config.parameters.get('adx_threshold', 25)
        rsi_oversold = self.config.parameters.get('rsi_oversold', 30)
        rsi_overbought = self.config.parameters.get('rsi_overbought', 70)
        
        data['signal'] = 0
        
        # COMPRA: Tendencia alcista fuerte confirmada
        buy_condition = (
            (data['adx'] > adx_threshold) &  # Tendencia fuerte
            (data['di_plus'] > data['di_minus']) &  # Dirección alcista
            (data['ma_fast'] > data['ma_slow']) &  # MA alcista
            (data['rsi'] < rsi_overbought)  # No sobrecomprado
        )
        
        # VENTA: Tendencia bajista fuerte confirmada
        sell_condition = (
            (data['adx'] > adx_threshold) &  # Tendencia fuerte
            (data['di_minus'] > data['di_plus']) &  # Dirección bajista
            (data['ma_fast'] < data['ma_slow']) &  # MA bajista
            (data['rsi'] > rsi_oversold)  # No sobrevendido
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data