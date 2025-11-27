# data/mt5_connector.py
"""
MetaTrader 5 connection manager with robust error handling.

Provides safe access to MT5 data and trading functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

# Try to import MT5, but handle gracefully if not available
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("MetaTrader5 package not installed. Running in offline mode.")
    mt5 = None


@dataclass
class SymbolInfo:
    """Symbol information container."""
    name: str
    point: float
    digits: int
    spread: float
    trade_contract_size: float
    currency_base: str
    currency_profit: str
    currency_margin: str


class MT5ConnectionManager:
    """
    Manages MetaTrader 5 connection with automatic reconnection
    and robust error handling.
    """
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
        'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
    }
    
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.lock = threading.RLock()
        self.tick_queue = Queue()
        self.symbols_cache: Dict[str, SymbolInfo] = {}
        self.subscribed_symbols: set = set()
        self.monitor_thread = None
        self._stop_monitoring = False
        
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available - connection manager will work in offline mode")
    
    def initialize(self) -> bool:
        """Initialize connection with MT5."""
        if not MT5_AVAILABLE:
            logger.info("MT5 not available - running in offline mode")
            return False
        
        try:
            with self.lock:
                # Try to initialize MT5
                init_params = {}
                
                if hasattr(self.config, 'mt5'):
                    if self.config.mt5.path:
                        init_params['path'] = self.config.mt5.path
                    if self.config.mt5.login:
                        init_params['login'] = self.config.mt5.login
                    if self.config.mt5.password:
                        init_params['password'] = self.config.mt5.password
                    if self.config.mt5.server:
                        init_params['server'] = self.config.mt5.server
                    if self.config.mt5.timeout:
                        init_params['timeout'] = self.config.mt5.timeout
                    if hasattr(self.config.mt5, 'portable'):
                        init_params['portable'] = self.config.mt5.portable
                
                if not mt5.initialize(**init_params):
                    error = mt5.last_error()
                    logger.error(f"MT5 initialize failed: {error}")
                    return False
                
                self.connected = True
                logger.info("MT5 connected successfully")
                
                # Cache symbols info
                self._cache_symbols_info()
                
                # Start tick monitor
                self._start_tick_monitor()
                
                return True
                
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def shutdown(self):
        """Close MT5 connection safely."""
        with self.lock:
            self._stop_monitoring = True
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            if self.connected and MT5_AVAILABLE:
                try:
                    mt5.shutdown()
                except Exception as e:
                    logger.warning(f"Error during MT5 shutdown: {e}")
            
            self.connected = False
            logger.info("MT5 disconnected")
    
    @contextmanager
    def connection(self):
        """Context manager for safe connection handling."""
        try:
            if not self.connected and MT5_AVAILABLE:
                self.initialize()
            yield self
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
    
    def _cache_symbols_info(self):
        """Cache symbol information for better performance."""
        if not MT5_AVAILABLE or not self.connected:
            return
        
        try:
            with self.lock:
                symbols = mt5.symbols_get()
                if symbols:
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
                    logger.info(f"Cached info for {len(self.symbols_cache)} symbols")
        except Exception as e:
            logger.warning(f"Error caching symbols: {e}")
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime = None, end_date: datetime = None,
                           count: int = None) -> Optional[pd.DataFrame]:
        """
        Get historical data from MT5.
        
        Args:
            symbol: Asset symbol
            timeframe: Timeframe (M1, M5, H1, etc.)
            start_date: Start date (optional if count is provided)
            end_date: End date (optional)
            count: Number of candles (optional, used if start_date is None)
        
        Returns:
            DataFrame with OHLCV data or None
        """
        if not MT5_AVAILABLE or not self.connected:
            return None
        
        with self.connection():
            try:
                # Ensure symbol is selected
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Symbol {symbol} not available")
                    return None
                
                # Map timeframe to MT5 constant
                tf_mapping = {
                    'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
                    'MN1': mt5.TIMEFRAME_MN1
                }
                
                timeframe_val = tf_mapping.get(timeframe, mt5.TIMEFRAME_H1)
                
                # Decide which method to use
                if count is not None and count > 0:
                    # Use count-based method
                    rates = mt5.copy_rates_from_pos(symbol, timeframe_val, 0, count)
                elif start_date is not None:
                    # Use date range method
                    end = end_date or datetime.now()
                    rates = mt5.copy_rates_range(symbol, timeframe_val, start_date, end)
                else:
                    # Default: get last 1000 candles
                    rates = mt5.copy_rates_from_pos(symbol, timeframe_val, 0, 1000)
                
                if rates is None or len(rates) == 0:
                    error = mt5.last_error()
                    logger.warning(f"No data for {symbol} {timeframe}, error: {error}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Rename columns for consistency
                df.rename(columns={
                    'tick_volume': 'volume'
                }, inplace=True)
                
                return df
                
            except Exception as e:
                logger.error(f"Error getting historical data for {symbol}: {e}")
                return None
    
    def get_tick_data(self, symbol: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get recent tick data."""
        if not MT5_AVAILABLE or not self.connected:
            return None
        
        with self.connection():
            try:
                ticks = mt5.copy_ticks_from(
                    symbol, datetime.now(), count, mt5.COPY_TICKS_ALL
                )
                
                if ticks is None or len(ticks) == 0:
                    return None
                
                df = pd.DataFrame(ticks)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return df
                
            except Exception as e:
                logger.error(f"Error getting tick data for {symbol}: {e}")
                return None
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not MT5_AVAILABLE or not self.connected:
            return {}
        
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
                        'server': account_info.server,
                        'profit': account_info.profit,
                        'trade_allowed': account_info.trade_allowed
                    }
                return {}
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
                return {}
    
    def place_order(self, symbol: str, order_type: str, volume: float,
                   price: float = None, sl: float = None, tp: float = None,
                   deviation: int = 10, comment: str = "") -> Dict[str, Any]:
        """
        Place an order in MT5.
        
        Args:
            symbol: Asset symbol
            order_type: 'BUY' or 'SELL'
            volume: Volume in lots
            price: Entry price (None for market price)
            sl: Stop Loss
            tp: Take Profit
            deviation: Maximum slippage
            comment: Order comment
        
        Returns:
            Dict with order result
        """
        if not MT5_AVAILABLE or not self.connected:
            return {'success': False, 'error': 'MT5 not connected'}
        
        with self.connection():
            try:
                order_type = order_type.upper()
                
                if order_type == 'BUY':
                    order_type_mt5 = mt5.ORDER_TYPE_BUY
                    tick = mt5.symbol_info_tick(symbol)
                    price = price or (tick.ask if tick else 0)
                elif order_type == 'SELL':
                    order_type_mt5 = mt5.ORDER_TYPE_SELL
                    tick = mt5.symbol_info_tick(symbol)
                    price = price or (tick.bid if tick else 0)
                else:
                    return {'success': False, 'error': f'Invalid order type: {order_type}'}
                
                if price <= 0:
                    return {'success': False, 'error': 'Could not get price'}
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(volume),
                    "type": order_type_mt5,
                    "price": float(price),
                    "deviation": deviation,
                    "magic": 2024001,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                if sl:
                    request["sl"] = float(sl)
                if tp:
                    request["tp"] = float(tp)
                
                result = mt5.order_send(request)
                
                if result is None:
                    return {'success': False, 'error': 'Order send returned None'}
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    return {
                        'success': False,
                        'error': f"Order failed: {result.retcode} - {result.comment}",
                        'retcode': result.retcode
                    }
                
                return {
                    'success': True,
                    'order_id': str(result.order),
                    'price': result.price,
                    'volume': result.volume,
                    'order_type': order_type
                }
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                return {'success': False, 'error': str(e)}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        if not MT5_AVAILABLE or not self.connected:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'time': datetime.fromtimestamp(pos.time)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, ticket: int) -> Dict[str, Any]:
        """Close a specific position by ticket."""
        if not MT5_AVAILABLE or not self.connected:
            return {'success': False, 'error': 'MT5 not connected'}
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': f'Position {ticket} not found'}
            
            position = position[0]
            
            # Determine close order type
            if position.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                tick = mt5.symbol_info_tick(position.symbol)
                price = tick.bid if tick else 0
            else:
                close_type = mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(position.symbol)
                price = tick.ask if tick else 0
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 10,
                "magic": 2024001,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f"Close failed: {result.retcode}",
                    'retcode': result.retcode
                }
            
            return {
                'success': True,
                'closed_ticket': ticket,
                'price': result.price
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}
    
    def _start_tick_monitor(self):
        """Start real-time tick monitoring."""
        if not MT5_AVAILABLE or not self.connected:
            return
        
        def monitor_ticks():
            while not self._stop_monitoring and self.connected:
                try:
                    for symbol in list(self.subscribed_symbols):
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
                    if not self._stop_monitoring:
                        logger.error(f"Tick monitor error: {e}")
                    time.sleep(1)
        
        self._stop_monitoring = False
        self.monitor_thread = threading.Thread(target=monitor_ticks, daemon=True)
        self.monitor_thread.start()
    
    def subscribe_symbol(self, symbol: str):
        """Subscribe to real-time tick updates for a symbol."""
        self.subscribed_symbols.add(symbol)
        
        if MT5_AVAILABLE and self.connected:
            try:
                mt5.symbol_select(symbol, True)
            except Exception as e:
                logger.warning(f"Could not select symbol {symbol}: {e}")
    
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from tick updates."""
        self.subscribed_symbols.discard(symbol)
    
    def get_next_tick(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get the next tick from the queue."""
        try:
            return self.tick_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information."""
        if symbol in self.symbols_cache:
            return self.symbols_cache[symbol]
        
        if MT5_AVAILABLE and self.connected:
            try:
                info = mt5.symbol_info(symbol)
                if info:
                    symbol_info = SymbolInfo(
                        name=info.name,
                        point=info.point,
                        digits=info.digits,
                        spread=info.spread,
                        trade_contract_size=info.trade_contract_size,
                        currency_base=info.currency_base,
                        currency_profit=info.currency_profit,
                        currency_margin=info.currency_margin
                    )
                    self.symbols_cache[symbol] = symbol_info
                    return symbol_info
            except Exception as e:
                logger.warning(f"Could not get symbol info for {symbol}: {e}")
        
        return None
