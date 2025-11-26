# strategies/adx_strategy.py
import pandas as pd
import numpy as np
from strategies.strategy_engine import BaseStrategy
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator

class ADXStrategy(BaseStrategy):
    """
    Estrategia de ADX (Average Directional Index)
    
    Señales:
    - COMPRA: ADX > threshold + DI+ > DI- + RSI no overbought
    - VENTA: ADX > threshold + DI- > DI+ + RSI no oversold
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular ADX y direccionales"""
        adx_period = self.config.parameters.get('adx_period', 14)
        rsi_period = self.config.parameters.get('rsi_period', 14)
        
        # ADX
        adx = ADXIndicator(data['high'], data['low'], data['close'], window=adx_period)
        data['adx'] = adx.adx()
        data['di_plus'] = adx.adx_pos()
        data['di_minus'] = adx.adx_neg()
        
        # RSI para filtro
        data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        adx_threshold = self.config.parameters.get('adx_threshold', 25)
        rsi_oversold = self.config.parameters.get('rsi_oversold', 30)
        rsi_overbought = self.config.parameters.get('rsi_overbought', 70)
        
        data['signal'] = 0
        
        # Señal de COMPRA: Tendencia alcista fuerte
        buy_condition = (
            (data['adx'] > adx_threshold) &  # Tendencia fuerte
            (data['di_plus'] > data['di_minus']) &  # Dirección alcista
            (data['di_plus'].shift(1) <= data['di_minus'].shift(1)) &  # Cruce reciente
            (data['rsi'] < rsi_overbought)  # No sobrecomprado
        )
        
        # Señal de VENTA: Tendencia bajista fuerte
        sell_condition = (
            (data['adx'] > adx_threshold) &  # Tendencia fuerte
            (data['di_minus'] > data['di_plus']) &  # Dirección bajista
            (data['di_minus'].shift(1) <= data['di_plus'].shift(1)) &  # Cruce reciente
            (data['rsi'] > rsi_oversold)  # No sobrevendido
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data