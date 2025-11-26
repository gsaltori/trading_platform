# strategies/bollinger_strategy.py
import pandas as pd
import numpy as np
from strategies.strategy_engine import BaseStrategy
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

class BollingerBandsStrategy(BaseStrategy):
    """
    Estrategia de Bollinger Bands
    
    Señales:
    - COMPRA: Precio toca banda inferior + RSI oversold
    - VENTA: Precio toca banda superior + RSI overbought
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular Bollinger Bands y RSI"""
        period = self.config.parameters.get('period', 20)
        std_dev = self.config.parameters.get('std_dev', 2.0)
        rsi_period = self.config.parameters.get('rsi_period', 14)
        
        # Bollinger Bands
        bb = BollingerBands(data['close'], window=period, window_dev=std_dev)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Posición del precio en las bandas (0 = banda inferior, 1 = banda superior)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # RSI para confirmación
        data['rsi'] = RSIIndicator(data['close'], window=rsi_period).rsi()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        rsi_oversold = self.config.parameters.get('rsi_oversold', 30)
        rsi_overbought = self.config.parameters.get('rsi_overbought', 70)
        
        data['signal'] = 0
        
        # Señal de COMPRA: Precio cerca/toca banda inferior + RSI oversold
        buy_condition = (
            (data['bb_position'] < 0.15) &  # Cerca de banda inferior
            (data['rsi'] < rsi_oversold) &
            (data['bb_width'] > 0.01)  # Bandas no muy estrechas
        )
        
        # Señal de VENTA: Precio cerca/toca banda superior + RSI overbought
        sell_condition = (
            (data['bb_position'] > 0.85) &  # Cerca de banda superior
            (data['rsi'] > rsi_overbought) &
            (data['bb_width'] > 0.01)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data