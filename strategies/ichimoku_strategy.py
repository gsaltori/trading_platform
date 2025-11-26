# strategies/ichimoku_strategy.py
import pandas as pd
import numpy as np
from strategies.strategy_engine import BaseStrategy

class IchimokuStrategy(BaseStrategy):
    """
    Estrategia de Ichimoku Cloud
    
    Señales:
    - COMPRA: Tenkan cruza por encima de Kijun + precio sobre la nube
    - VENTA: Tenkan cruza por debajo de Kijun + precio bajo la nube
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular componentes de Ichimoku"""
        tenkan_period = self.config.parameters.get('tenkan', 9)
        kijun_period = self.config.parameters.get('kijun', 26)
        senkou_b_period = self.config.parameters.get('senkou_b', 52)
        
        # Tenkan-sen (Conversion Line)
        high_tenkan = data['high'].rolling(window=tenkan_period).max()
        low_tenkan = data['low'].rolling(window=tenkan_period).min()
        data['tenkan'] = (high_tenkan + low_tenkan) / 2
        
        # Kijun-sen (Base Line)
        high_kijun = data['high'].rolling(window=kijun_period).max()
        low_kijun = data['low'].rolling(window=kijun_period).min()
        data['kijun'] = (high_kijun + low_kijun) / 2
        
        # Senkou Span A (Leading Span A)
        data['senkou_a'] = ((data['tenkan'] + data['kijun']) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        high_senkou = data['high'].rolling(window=senkou_b_period).max()
        low_senkou = data['low'].rolling(window=senkou_b_period).min()
        data['senkou_b'] = ((high_senkou + low_senkou) / 2).shift(kijun_period)
        
        # Cloud top and bottom
        data['cloud_top'] = data[['senkou_a', 'senkou_b']].max(axis=1)
        data['cloud_bottom'] = data[['senkou_a', 'senkou_b']].min(axis=1)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        data['signal'] = 0
        
        # Señal de COMPRA: Tenkan cruza por encima de Kijun + precio sobre nube
        buy_condition = (
            (data['tenkan'] > data['kijun']) &
            (data['tenkan'].shift(1) <= data['kijun'].shift(1)) &
            (data['close'] > data['cloud_top'])
        )
        
        # Señal de VENTA: Tenkan cruza por debajo de Kijun + precio bajo nube
        sell_condition = (
            (data['tenkan'] < data['kijun']) &
            (data['tenkan'].shift(1) >= data['kijun'].shift(1)) &
            (data['close'] < data['cloud_bottom'])
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data