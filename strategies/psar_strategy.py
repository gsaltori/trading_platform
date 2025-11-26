# strategies/psar_strategy.py
import pandas as pd
import numpy as np
from strategies.strategy_engine import BaseStrategy

class ParabolicSARStrategy(BaseStrategy):
    """
    Estrategia de Parabolic SAR
    
    Señales:
    - COMPRA: SAR flip de arriba a abajo del precio
    - VENTA: SAR flip de abajo a arriba del precio
    """
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular Parabolic SAR"""
        af = self.config.parameters.get('acceleration', 0.02)
        max_af = self.config.parameters.get('maximum', 0.2)
        
        data['psar'] = self._calculate_psar(data, af, max_af)
        data['psar_position'] = np.where(data['close'] > data['psar'], 1, -1)
        
        return data
    
    def _calculate_psar(self, data: pd.DataFrame, af: float, max_af: float) -> pd.Series:
        """Calcular Parabolic SAR manualmente"""
        length = len(data)
        psar = data['close'].copy()
        bull = True
        iaf = af
        ep = data['low'].iloc[0]
        hp = data['high'].iloc[0]
        lp = data['low'].iloc[0]
        
        for i in range(1, length):
            if bull:
                psar.iloc[i] = psar.iloc[i-1] + iaf * (hp - psar.iloc[i-1])
            else:
                psar.iloc[i] = psar.iloc[i-1] + iaf * (lp - psar.iloc[i-1])
            
            reverse = False
            
            if bull:
                if data['low'].iloc[i] < psar.iloc[i]:
                    bull = False
                    reverse = True
                    psar.iloc[i] = hp
                    lp = data['low'].iloc[i]
                    iaf = af
            else:
                if data['high'].iloc[i] > psar.iloc[i]:
                    bull = True
                    reverse = True
                    psar.iloc[i] = lp
                    hp = data['high'].iloc[i]
                    iaf = af
            
            if not reverse:
                if bull:
                    if data['high'].iloc[i] > hp:
                        hp = data['high'].iloc[i]
                        iaf = min(iaf + af, max_af)
                    if data['low'].iloc[i-1] < psar.iloc[i]:
                        psar.iloc[i] = data['low'].iloc[i-1]
                    if data['low'].iloc[i-2] < psar.iloc[i]:
                        psar.iloc[i] = data['low'].iloc[i-2]
                else:
                    if data['low'].iloc[i] < lp:
                        lp = data['low'].iloc[i]
                        iaf = min(iaf + af, max_af)
                    if data['high'].iloc[i-1] > psar.iloc[i]:
                        psar.iloc[i] = data['high'].iloc[i-1]
                    if data['high'].iloc[i-2] > psar.iloc[i]:
                        psar.iloc[i] = data['high'].iloc[i-2]
        
        return psar
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading"""
        data['signal'] = 0
        
        # Señal de COMPRA: SAR flip de negativo a positivo
        buy_condition = (
            (data['psar_position'] == 1) &
            (data['psar_position'].shift(1) == -1)
        )
        
        # Señal de VENTA: SAR flip de positivo a negativo
        sell_condition = (
            (data['psar_position'] == -1) &
            (data['psar_position'].shift(1) == 1)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data