# strategies/stochastic_strategy.py
import pandas as pd
import numpy as np
from strategies.strategy_engine import BaseStrategy
from ta.momentum import StochasticOscillator

class StochasticStrategy(BaseStrategy):
    """
    Estrategia de Stochastic Oscillator
    
    Señales:
    - COMPRA: %K cruza por encima de %D en zona oversold
    - VENTA: %K cruza por debajo de %D en zona overbought
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular Stochastic Oscillator"""
        k_period = self.config.parameters.get('k_period', 14)
        d_period = self.config.parameters.get('d_period', 3)
        
        # Stochastic
        stoch = StochasticOscillator(
            data['high'], 
            data['low'], 
            data['close'],
            window=k_period,
            smooth_window=d_period
        )
        
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        oversold = self.config.parameters.get('oversold', 20)
        overbought = self.config.parameters.get('overbought', 80)
        
        data['signal'] = 0
        
        # Señal de COMPRA: %K cruza por encima de %D en zona oversold
        buy_condition = (
            (data['stoch_k'] > data['stoch_d']) &
            (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) &
            (data['stoch_k'] < oversold)
        )
        
        # Señal de VENTA: %K cruza por debajo de %D en zona overbought
        sell_condition = (
            (data['stoch_k'] < data['stoch_d']) &
            (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)) &
            (data['stoch_k'] > overbought)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data