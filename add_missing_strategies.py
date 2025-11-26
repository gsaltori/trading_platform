#!/usr/bin/env python3
"""
Parche para agregar las estrategias faltantes a improved_strategy_gui.py
"""

# Código de las estrategias faltantes para insertar en el archivo

BOLLINGER_STRATEGY = '''
class BollingerStrategy(BaseStrategy):
    """Estrategia basada en Bollinger Bands"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.config.parameters.get('period', 20)
        std_dev = self.config.parameters.get('std_dev', 2.0)
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=period).mean()
        data['bb_std'] = data['close'].rolling(window=period).std()
        data['bb_upper'] = data['bb_middle'] + (std_dev * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (std_dev * data['bb_std'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # RSI para confirmación
        rsi_period = self.config.parameters.get('rsi_period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        oversold = self.config.parameters.get('rsi_oversold', 30)
        overbought = self.config.parameters.get('rsi_overbought', 70)
        
        data['signal'] = 0
        
        # Compra: precio cerca de banda inferior + RSI oversold
        buy_condition = (
            (data['bb_position'] < 0.15) &
            (data['rsi'] < oversold) &
            (data['bb_width'] > 0.01)
        )
        
        # Venta: precio cerca de banda superior + RSI overbought
        sell_condition = (
            (data['bb_position'] > 0.85) &
            (data['rsi'] > overbought) &
            (data['bb_width'] > 0.01)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
'''

STOCHASTIC_STRATEGY = '''
class StochasticStrategy(BaseStrategy):
    """Estrategia basada en Stochastic Oscillator"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        k_period = self.config.parameters.get('k_period', 14)
        d_period = self.config.parameters.get('d_period', 3)
        
        # Stochastic %K
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        data['stoch_k'] = 100 * (data['close'] - low_min) / (high_max - low_min)
        
        # Stochastic %D (señal)
        data['stoch_d'] = data['stoch_k'].rolling(window=d_period).mean()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        oversold = self.config.parameters.get('oversold', 20)
        overbought = self.config.parameters.get('overbought', 80)
        
        data['signal'] = 0
        
        # Compra: %K cruza por encima de %D en zona oversold
        buy_condition = (
            (data['stoch_k'] > data['stoch_d']) &
            (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) &
            (data['stoch_k'] < oversold)
        )
        
        # Venta: %K cruza por debajo de %D en zona overbought
        sell_condition = (
            (data['stoch_k'] < data['stoch_d']) &
            (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)) &
            (data['stoch_k'] > overbought)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
'''

ADX_STRATEGY = '''
class ADXStrategy(BaseStrategy):
    """Estrategia basada en ADX"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.config.parameters.get('adx_period', 14)
        
        # True Range
        data['tr1'] = data['high'] - data['low']
        data['tr2'] = abs(data['high'] - data['close'].shift(1))
        data['tr3'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Directional Movement
        data['dm_plus'] = np.where(
            (data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
            np.maximum(data['high'] - data['high'].shift(1), 0),
            0
        )
        data['dm_minus'] = np.where(
            (data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
            np.maximum(data['low'].shift(1) - data['low'], 0),
            0
        )
        
        # Smoothed values
        data['atr'] = data['tr'].rolling(window=period).mean()
        data['di_plus'] = 100 * (data['dm_plus'].rolling(window=period).mean() / data['atr'])
        data['di_minus'] = 100 * (data['dm_minus'].rolling(window=period).mean() / data['atr'])
        
        # ADX
        data['dx'] = 100 * abs(data['di_plus'] - data['di_minus']) / (data['di_plus'] + data['di_minus'])
        data['adx'] = data['dx'].rolling(window=period).mean()
        
        # RSI para filtro
        rsi_period = self.config.parameters.get('rsi_period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.parameters.get('adx_threshold', 25)
        
        data['signal'] = 0
        
        # Compra: ADX fuerte + DI+ > DI- + cruce reciente
        buy_condition = (
            (data['adx'] > threshold) &
            (data['di_plus'] > data['di_minus']) &
            (data['di_plus'].shift(1) <= data['di_minus'].shift(1)) &
            (data['rsi'] < 70)
        )
        
        # Venta: ADX fuerte + DI- > DI+ + cruce reciente
        sell_condition = (
            (data['adx'] > threshold) &
            (data['di_minus'] > data['di_plus']) &
            (data['di_minus'].shift(1) <= data['di_plus'].shift(1)) &
            (data['rsi'] > 30)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
'''

CCI_STRATEGY = '''
class CCIStrategy(BaseStrategy):
    """Estrategia basada en CCI"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.config.parameters.get('cci_period', 20)
        
        # Typical Price
        data['tp'] = (data['high'] + data['low'] + data['close']) / 3
        
        # SMA of Typical Price
        data['sma_tp'] = data['tp'].rolling(window=period).mean()
        
        # Mean Deviation
        data['mad'] = data['tp'].rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        # CCI
        data['cci'] = (data['tp'] - data['sma_tp']) / (0.015 * data['mad'])
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        oversold = self.config.parameters.get('oversold', -100)
        overbought = self.config.parameters.get('overbought', 100)
        
        data['signal'] = 0
        
        # Compra: CCI cruza por encima de oversold
        buy_condition = (
            (data['cci'] > oversold) &
            (data['cci'].shift(1) <= oversold)
        )
        
        # Venta: CCI cruza por debajo de overbought
        sell_condition = (
            (data['cci'] < overbought) &
            (data['cci'].shift(1) >= overbought)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
'''

ICHIMOKU_STRATEGY = '''
class IchimokuStrategy(BaseStrategy):
    """Estrategia basada en Ichimoku Cloud"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        tenkan = self.config.parameters.get('tenkan', 9)
        kijun = self.config.parameters.get('kijun', 26)
        senkou_b = self.config.parameters.get('senkou_b', 52)
        
        # Tenkan-sen (Conversion Line)
        high_tenkan = data['high'].rolling(window=tenkan).max()
        low_tenkan = data['low'].rolling(window=tenkan).min()
        data['tenkan'] = (high_tenkan + low_tenkan) / 2
        
        # Kijun-sen (Base Line)
        high_kijun = data['high'].rolling(window=kijun).max()
        low_kijun = data['low'].rolling(window=kijun).min()
        data['kijun'] = (high_kijun + low_kijun) / 2
        
        # Senkou Span A (Leading Span A)
        data['senkou_a'] = ((data['tenkan'] + data['kijun']) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        high_senkou = data['high'].rolling(window=senkou_b).max()
        low_senkou = data['low'].rolling(window=senkou_b).min()
        data['senkou_b'] = ((high_senkou + low_senkou) / 2).shift(kijun)
        
        # Cloud
        data['cloud_top'] = data[['senkou_a', 'senkou_b']].max(axis=1)
        data['cloud_bottom'] = data[['senkou_a', 'senkou_b']].min(axis=1)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data['signal'] = 0
        
        # Compra: Tenkan cruza por encima de Kijun + precio sobre nube
        buy_condition = (
            (data['tenkan'] > data['kijun']) &
            (data['tenkan'].shift(1) <= data['kijun'].shift(1)) &
            (data['close'] > data['cloud_top'])
        )
        
        # Venta: Tenkan cruza por debajo de Kijun + precio bajo nube
        sell_condition = (
            (data['tenkan'] < data['kijun']) &
            (data['tenkan'].shift(1) >= data['kijun'].shift(1)) &
            (data['close'] < data['cloud_bottom'])
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
'''

PSAR_STRATEGY = '''
class PSARStrategy(BaseStrategy):
    """Estrategia basada en Parabolic SAR"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        acceleration = self.config.parameters.get('acceleration', 0.02)
        maximum = self.config.parameters.get('maximum', 0.2)
        
        # Implementación simplificada de PSAR
        data['psar'] = data['close'].copy()
        data['psar_position'] = 0
        
        # Calcular PSAR (versión simplificada)
        af = acceleration
        for i in range(2, len(data)):
            if data['close'].iloc[i] > data['psar'].iloc[i-1]:
                # Tendencia alcista
                data.loc[data.index[i], 'psar'] = data['psar'].iloc[i-1] + af * (data['high'].iloc[i-1] - data['psar'].iloc[i-1])
                data.loc[data.index[i], 'psar_position'] = 1
                af = min(af + acceleration, maximum)
            else:
                # Tendencia bajista
                data.loc[data.index[i], 'psar'] = data['psar'].iloc[i-1] + af * (data['low'].iloc[i-1] - data['psar'].iloc[i-1])
                data.loc[data.index[i], 'psar_position'] = -1
                af = acceleration
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data['signal'] = 0
        
        # Compra: PSAR flip de bajista a alcista
        buy_condition = (
            (data['psar_position'] == 1) &
            (data['psar_position'].shift(1) == -1)
        )
        
        # Venta: PSAR flip de alcista a bajista
        sell_condition = (
            (data['psar_position'] == -1) &
            (data['psar_position'].shift(1) == 1)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
'''

# Código para registrar las estrategias en el motor
STRATEGY_REGISTRATION = '''
        # Estrategias originales
        self.available_strategies = {
            'ma_crossover': MovingAverageCrossover,
            'rsi': RSIStrategy,
            'macd': MACDStrategy,
            # Nuevas estrategias
            'bollinger': BollingerStrategy,
            'stochastic': StochasticStrategy,
            'adx': ADXStrategy,
            'cci': CCIStrategy,
            'ichimoku': IchimokuStrategy,
            'psar': PSARStrategy
        }
'''

print("Contenido para agregar a improved_strategy_gui.py:")
print("\n" + "="*80)
print("Agregar después de MACDStrategy y antes de StrategyEngine:")
print("="*80)
print(BOLLINGER_STRATEGY)
print(STOCHASTIC_STRATEGY)
print(ADX_STRATEGY)
print(CCI_STRATEGY)
print(ICHIMOKU_STRATEGY)
print(PSAR_STRATEGY)

print("\n" + "="*80)
print("Reemplazar self.available_strategies en StrategyEngine.__init__:")
print("="*80)
print(STRATEGY_REGISTRATION)