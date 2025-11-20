# data/mt5_connector.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

@dataclass
class SymbolInfo:
    name: str
    point: float
    digits: int
    spread: float
    trade_contract_size: float
    currency_base: str
    currency_profit: str
    currency_margin: str

class MT5ConnectionManager:
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.lock = threading.RLock()
        self.tick_queue = Queue()
        self.symbols_cache = {}
        
    def initialize(self) -> bool:
        """Inicializa conexión con MT5"""
        try:
            if not mt5.initialize(
                path=self.config.mt5.path,
                login=self.config.mt5.login,
                password=self.config.mt5.password,
                server=self.config.mt5.server,
                timeout=self.config.mt5.timeout,
                portable=self.config.mt5.portable
            ):
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            
            self.connected = True
            logger.info("MT5 connected successfully")
            self._cache_symbols_info()
            self._start_tick_monitor()
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def shutdown(self):
        """Cierra conexión con MT5"""
        with self.lock:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                logger.info("MT5 disconnected")
    
    @contextmanager
    def connection(self):
        """Context manager para manejo seguro de conexión"""
        try:
            if not self.connected:
                self.initialize()
            yield self
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
        finally:
            # No desconectamos automáticamente para mantener la conexión
            pass
    
    def _cache_symbols_info(self):
        """Cachea información de símbolos para mejor performance"""
        with self.lock:
            symbols = mt5.symbols_get()
            for symbol in symbols:
                self.symbols_cache[symbol.name] = SymbolInfo(
                    name=symbol.name,
                    point=symbol.point,
                    digits=symbol.digits,
                    spread=symbol.spread,
                    trade_contract_size=symbol.trade_contract_size,
                    currency_base=symbol.currency_base,
                    currency_profit=symbol.currency_profit,
                    currency_margin=symbol.currency_margin
                )
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime = None,
                           count: int = None) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de MT5
        
        Args:
            symbol: Símbolo del activo
            timeframe: Timeframe (M1, M5, H1, etc.)
            start_date: Fecha de inicio
            end_date: Fecha de fin (opcional)
            count: Número de velas (opcional)
        """
        with self.connection():
            try:
                tf_mapping = {
                    'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
                    'MN1': mt5.TIMEFRAME_MN1
                }
                
                timeframe_val = tf_mapping.get(timeframe, mt5.TIMEFRAME_H1)
                
                if count:
                    rates = mt5.copy_rates_from_pos(symbol, timeframe_val, 0, count)
                else:
                    rates = mt5.copy_rates_range(symbol, timeframe_val, start_date, end_date or datetime.now())
                
                if rates is None:
                    logger.error(f"No data for {symbol}, error: {mt5.last_error()}")
                    return None
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Calcular volumen en dólares si está disponible
                if 'tick_volume' in df.columns and 'real_volume' in df.columns:
                    df['dollar_volume'] = df['real_volume'] * df['close']
                
                return df
                
            except Exception as e:
                logger.error(f"Error getting historical data for {symbol}: {e}")
                return None
    
    def get_tick_data(self, symbol: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Obtiene datos de ticks recientes"""
        with self.connection():
            try:
                ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)
                if ticks is None:
                    return None
                
                df = pd.DataFrame(ticks)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return df
            except Exception as e:
                logger.error(f"Error getting tick data for {symbol}: {e}")
                return None
    
    def get_account_info(self) -> Dict:
        """Obtiene información de la cuenta"""
        with self.connection():
            try:
                account_info = mt5.account_info()
                if account_info:
                    return {
                        'login': account_info.login,
                        'balance': account_info.balance,
                        'equity': account_info.equity,
                        'margin': account_info.margin,
                        'free_margin': account_info.margin_free,
                        'leverage': account_info.leverage,
                        'currency': account_info.currency,
                        'server': account_info.server
                    }
                return {}
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
                return {}
    
    def place_order(self, symbol: str, order_type: str, volume: float,
                   price: float = None, sl: float = None, tp: float = None,
                   deviation: int = 10, comment: str = "") -> Dict:
        """
        Coloca una orden en MT5
        
        Args:
            symbol: Símbolo del activo
            order_type: 'BUY' o 'SELL'
            volume: Volumen en lotes
            price: Precio de entrada
            sl: Stop Loss
            tp: Take Profit
            deviation: Desviación máxima
            comment: Comentario de la orden
        """
        with self.connection():
            try:
                order_type = order_type.upper()
                if order_type == 'BUY':
                    order_type_mt5 = mt5.ORDER_TYPE_BUY
                    price = price or mt5.symbol_info_tick(symbol).ask
                elif order_type == 'SELL':
                    order_type_mt5 = mt5.ORDER_TYPE_SELL
                    price = price or mt5.symbol_info_tick(symbol).bid
                else:
                    return {'success': False, 'error': 'Invalid order type'}
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type_mt5,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": deviation,
                    "magic": 2024001,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    return {
                        'success': False,
                        'error': f"Order failed: {result.retcode}",
                        'retcode': result.retcode
                    }
                
                return {
                    'success': True,
                    'order_id': result.order,
                    'price': result.price,
                    'volume': result.volume,
                    'sl': result.sl,
                    'tp': result.tp
                }
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                return {'success': False, 'error': str(e)}
    
    def _start_tick_monitor(self):
        """Inicia monitorización de ticks en tiempo real"""
        def monitor_ticks():
            while self.connected:
                try:
                    # Obtener ticks para símbolos suscritos
                    for symbol in self.subscribed_symbols:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick:
                            self.tick_queue.put({
                                'symbol': symbol,
                                'time': datetime.now(),
                                'bid': tick.bid,
                                'ask': tick.ask,
                                'last': tick.last,
                                'volume': tick.volume
                            })
                    time.sleep(0.1)  # 100ms
                except Exception as e:
                    logger.error(f"Tick monitor error: {e}")
                    time.sleep(1)
        
        self.subscribed_symbols = set()
        self.monitor_thread = threading.Thread(target=monitor_ticks, daemon=True)
        self.monitor_thread.start()
    
    def subscribe_symbol(self, symbol: str):
        """Suscribe un símbolo para monitorización en tiempo real"""
        self.subscribed_symbols.add(symbol)
    
    def get_next_tick(self, timeout: float = 1.0) -> Optional[Dict]:
        """Obtiene el siguiente tick de la cola"""
        try:
            return self.tick_queue.get(timeout=timeout)
        except Empty:
            return None