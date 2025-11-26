# strategies/cci_strategy.py
import pandas as pd
import numpy as np
from strategies.strategy_engine import BaseStrategy

class CCIStrategy(BaseStrategy):
    """
    Estrategia de CCI (Commodity Channel Index)
    
    Señales:
    - COMPRA: CCI cruza por encima de -100 desde abajo
    - VENTA: CCI cruza por debajo de +100 desde arriba
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular CCI"""
        period = self.config.parameters.get('cci_period', 20)
        
        # CCI manual
        tp = (data['high'] + data['low'] + data['close']) / 3  # Typical Price
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        data['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        oversold = self.config.parameters.get('oversold', -100)
        overbought = self.config.parameters.get('overbought', 100)
        
        data['signal'] = 0
        
        # Señal de COMPRA: CCI cruza por encima de oversold
        buy_condition = (
            (data['cci'] > oversold) &
            (data['cci'].shift(1) <= oversold)
        )
        
        # Señal de VENTA: CCI cruza por debajo de overbought
        sell_condition = (
            (data['cci'] < overbought) &
            (data['cci'].shift(1) >= overbought)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data