# execution/live_execution.py
"""
Live execution engine for real-time trading.

Handles order execution, position management, and risk controls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from threading import Thread, Event, Lock
from queue import Queue, Empty

logger = logging.getLogger(__name__)


@dataclass
class LiveTradingConfig:
    """Configuration for live trading."""
    strategy_name: str
    symbols: List[str]
    timeframe: str
    enabled: bool = True
    max_positions: int = 5
    risk_per_trade: float = 0.02
    daily_loss_limit: float = 0.05
    max_drawdown: float = 0.15
    trading_hours: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    news_filter: bool = True
    volatility_filter: bool = True


@dataclass
class LiveTrade:
    """Live trade record."""
    id: str
    symbol: str
    direction: int
    entry_price: float
    volume: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = 'open'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiveExecutionEngine:
    """Live execution engine with intelligent order management."""
    
    def __init__(self):
        self.platform = None
        self.strategy_engine = None
        self.risk_engine = None
        self.configs: Dict[str, LiveTradingConfig] = {}
        self.active_trades: Dict[str, LiveTrade] = {}
        self.trade_history: List[LiveTrade] = []
        self.is_running = False
        self.monitor_thread = None
        self.data_thread = None
        self.stop_event = Event()
        self.data_queue = Queue()
        self.lock = Lock()
        self.circuit_breaker = False
        
        # Real-time statistics
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.current_drawdown = 0.0
        self.daily_trades = 0
        self.equity_high = 0.0
        
        # Data cache
        self.data_cache: Dict[str, Any] = {}
        self.signal_cache: Dict[str, Any] = {}
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize required components."""
        try:
            from core.platform import get_platform
            from strategies.strategy_engine import StrategyEngine
            from risk_management.risk_engine import RiskEngine
            
            self.platform = get_platform()
            self.strategy_engine = StrategyEngine()
            self.risk_engine = RiskEngine()
        except ImportError as e:
            logger.warning(f"Could not import some components: {e}")
    
    def add_strategy(self, config: LiveTradingConfig) -> bool:
        """Add strategy for live trading."""
        if not self.strategy_engine:
            logger.error("Strategy engine not initialized")
            return False
        
        if config.strategy_name not in self.strategy_engine.strategies:
            logger.error(f"Strategy {config.strategy_name} not found")
            return False
        
        self.configs[config.strategy_name] = config
        
        # Subscribe symbols for real-time data
        if self.platform and hasattr(self.platform, 'mt5_connector'):
            for symbol in config.symbols:
                try:
                    self.platform.mt5_connector.subscribe_symbol(symbol)
                except Exception as e:
                    logger.warning(f"Could not subscribe to {symbol}: {e}")
        
        logger.info(f"Strategy {config.strategy_name} added for live trading")
        return True
    
    def start_trading(self):
        """Start live trading engine."""
        if self.is_running:
            logger.warning("Trading engine already running")
            return
        
        if not self.platform or not self.platform.initialized:
            logger.error("Platform not initialized")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Reset daily stats
        self._reset_daily_stats()
        
        # Start monitoring threads
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.data_thread = Thread(target=self._data_processing_loop, daemon=True)
        self.data_thread.start()
        
        logger.info("Live trading engine started")
    
    def stop_trading(self):
        """Stop live trading engine."""
        self.is_running = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        if self.data_thread:
            self.data_thread.join(timeout=10)
        
        logger.info("Live trading engine stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set() and self.is_running:
            try:
                # Check circuit breaker
                if self.circuit_breaker:
                    time.sleep(30)
                    continue
                
                # Process each active strategy
                for strategy_name, config in self.configs.items():
                    if not config.enabled:
                        continue
                    
                    # Check market conditions
                    if not self._check_market_conditions(config):
                        continue
                    
                    # Process each symbol
                    for symbol in config.symbols:
                        self._process_symbol(strategy_name, symbol, config)
                
                # Check risk management
                self._check_risk_management()
                
                # Update statistics
                self._update_statistics()
                
                # Check circuit breakers
                self._check_circuit_breakers()
                
                time.sleep(5)  # 5 seconds between iterations
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _data_processing_loop(self):
        """Real-time data processing loop."""
        while not self.stop_event.is_set() and self.is_running:
            try:
                # Get latest tick
                if self.platform and hasattr(self.platform, 'mt5_connector'):
                    tick = self.platform.mt5_connector.get_next_tick(timeout=1.0)
                    if tick:
                        self._process_tick_data(tick)
                else:
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in data processing: {e}")
                time.sleep(5)
    
    def _process_tick_data(self, tick: Dict):
        """Process real-time tick data."""
        symbol = tick.get('symbol')
        if not symbol:
            return
        
        # Update data cache
        if symbol not in self.data_cache:
            self.data_cache[symbol] = []
        
        self.data_cache[symbol].append(tick)
        
        # Keep only last 1000 ticks
        if len(self.data_cache[symbol]) > 1000:
            self.data_cache[symbol] = self.data_cache[symbol][-1000:]
    
    def _process_symbol(self, strategy_name: str, symbol: str, config: LiveTradingConfig):
        """Process a symbol for signal generation."""
        try:
            # Check if we already have an open position
            if self._has_open_position(symbol):
                return
            
            # Get recent data
            data = self._get_recent_data(symbol, config.timeframe)
            if data is None or len(data) < 100:
                return
            
            # Generate signals
            signals = self.strategy_engine.get_strategy_signals(strategy_name, symbol, data)
            if not signals:
                return
            
            # Take the most recent signal
            latest_signal = signals[-1]
            
            # Apply advanced filters
            if not self._apply_advanced_filters(latest_signal, data, config):
                return
            
            # Check risk
            if self.risk_engine:
                portfolio_status = self.get_portfolio_status()
                if not self.risk_engine.check_trade_risk(latest_signal, config, portfolio_status):
                    return
            
            # Execute trade
            self._execute_trade(latest_signal, strategy_name, config)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _get_recent_data(self, symbol: str, timeframe: str, lookback_bars: int = 100) -> Optional[pd.DataFrame]:
        """Get recent data with intelligent caching."""
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache first
        if cache_key in self.data_cache and isinstance(self.data_cache[cache_key], pd.DataFrame):
            if len(self.data_cache[cache_key]) >= lookback_bars:
                return self.data_cache[cache_key].tail(lookback_bars)
        
        # Get from platform
        if self.platform:
            data = self.platform.get_market_data(symbol, timeframe, days=30)
            if data is not None:
                self.data_cache[cache_key] = data
                return data.tail(lookback_bars)
        
        return None
    
    def _apply_advanced_filters(self, signal, data: pd.DataFrame, config: LiveTradingConfig) -> bool:
        """Apply advanced filters to signals."""
        # Volatility filter
        if config.volatility_filter:
            recent_volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            if recent_volatility > 0.02:  # 2% volatility threshold
                logger.info(f"Volatility filter activated for {signal.symbol}")
                return False
        
        # Trading hours filter
        if not self._is_within_trading_hours(config, signal.symbol):
            return False
        
        # News filter
        if config.news_filter:
            if self._is_high_impact_news_time():
                logger.info("News filter activated")
                return False
        
        # Correlation filter
        if self._has_high_correlation_trades(signal):
            return False
        
        return True
    
    def _execute_trade(self, signal, strategy_name: str, config: LiveTradingConfig):
        """Execute trade with advanced order management."""
        try:
            with self.lock:
                # Check limits
                if len(self.active_trades) >= config.max_positions:
                    logger.warning("Position limit reached")
                    return
                
                if self.daily_trades >= 20:  # Daily trade limit
                    logger.warning("Daily trade limit reached")
                    return
                
                # Calculate position size
                account_info = self.platform.get_account_summary() if self.platform else {'balance': 10000}
                
                if self.risk_engine:
                    volume = self.risk_engine.calculate_position_size(
                        signal, config.risk_per_trade, account_info
                    )
                else:
                    volume = account_info.get('balance', 10000) * 0.01 / signal.price
                
                if volume <= 0:
                    return
                
                # Place order
                order_type = 'BUY' if signal.direction > 0 else 'SELL'
                
                order_result = self._place_market_order(signal, order_type, volume)
                
                if order_result.get('success'):
                    # Record trade
                    trade = LiveTrade(
                        id=str(order_result.get('order_id', datetime.now().timestamp())),
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=order_result.get('price', signal.price),
                        volume=volume,
                        entry_time=datetime.now(),
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        strategy=strategy_name,
                        metadata={
                            'signal_strength': signal.strength,
                            'order_type': 'MARKET'
                        }
                    )
                    
                    self.active_trades[trade.id] = trade
                    self.daily_trades += 1
                    
                    logger.info(f"Trade executed: {order_type} {signal.symbol} "
                               f"at {trade.entry_price:.5f}, volume: {volume:.2f}")
                    
                    # Send notification
                    self._send_trade_notification(trade, "OPEN")
                
                else:
                    logger.error(f"Order failed: {order_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _place_market_order(self, signal, order_type: str, volume: float) -> Dict:
        """Place market order."""
        if self.platform and hasattr(self.platform, 'mt5_connector'):
            return self.platform.mt5_connector.place_order(
                symbol=signal.symbol,
                order_type=order_type,
                volume=volume,
                price=None,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                deviation=10,
                comment=f"LIVE_{order_type}"
            )
        else:
            # Simulated order for testing
            return {
                'success': True,
                'order_id': f"SIM_{datetime.now().timestamp()}",
                'price': signal.price,
                'volume': volume
            }
    
    def _check_risk_management(self):
        """Check and apply real-time risk management."""
        try:
            for trade_id, trade in list(self.active_trades.items()):
                if trade.status != 'open':
                    continue
                
                # Get current price
                current_tick = self._get_current_tick(trade.symbol)
                if not current_tick:
                    continue
                
                current_price = current_tick.get('bid') if trade.direction > 0 else current_tick.get('ask')
                if not current_price:
                    continue
                
                # Check stop loss
                if trade.stop_loss and self._check_stop_condition(trade, current_price):
                    self._close_trade(trade_id, current_price, 'Stop Loss')
                    continue
                
                # Check take profit
                if trade.take_profit and self._check_take_profit_condition(trade, current_price):
                    self._close_trade(trade_id, current_price, 'Take Profit')
                    continue
                
                # Update trailing stop
                self._update_trailing_stop(trade, current_price)
            
            # Check global risk limits
            self._check_global_risk_limits()
                
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
    
    def _check_stop_condition(self, trade: LiveTrade, current_price: float) -> bool:
        """Check stop loss condition."""
        if trade.direction > 0:  # Long
            return current_price <= trade.stop_loss
        else:  # Short
            return current_price >= trade.stop_loss
    
    def _check_take_profit_condition(self, trade: LiveTrade, current_price: float) -> bool:
        """Check take profit condition."""
        if trade.direction > 0:  # Long
            return current_price >= trade.take_profit
        else:  # Short
            return current_price <= trade.take_profit
    
    def _update_trailing_stop(self, trade: LiveTrade, current_price: float):
        """Update trailing stop."""
        if trade.direction > 0:  # Long
            new_stop = current_price * 0.99  # 1% below current price
            if trade.stop_loss and new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
        else:  # Short
            new_stop = current_price * 1.01  # 1% above current price
            if trade.stop_loss and new_stop < trade.stop_loss:
                trade.stop_loss = new_stop
    
    def _check_global_risk_limits(self):
        """Check global risk limits."""
        account_info = self.platform.get_account_summary() if self.platform else {}
        
        balance = account_info.get('balance', 10000)
        equity = account_info.get('equity', balance)
        
        # Check daily loss limit
        if balance > 0 and self.daily_pnl < -abs(balance * 0.05):  # 5% daily
            logger.warning("Daily loss limit reached")
            self._close_all_positions()
            return
        
        # Check max drawdown
        if balance > 0:
            drawdown = (balance - equity) / balance * 100
            if drawdown > 15:  # 15% max
                logger.warning("Max drawdown reached")
                self._close_all_positions()
                self.circuit_breaker = True
    
    def _close_trade(self, trade_id: str, price: float, reason: str):
        """Close specific trade."""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return
            
            # Update trade
            trade.exit_time = datetime.now()
            trade.exit_price = price
            trade.status = 'closed'
            
            # Calculate P&L
            if trade.direction > 0:
                trade.pnl = (price - trade.entry_price) * trade.volume
            else:
                trade.pnl = (trade.entry_price - price) * trade.volume
            
            trade.pnl_pct = (trade.pnl / (trade.volume * trade.entry_price)) * 100 if trade.volume * trade.entry_price > 0 else 0
            
            # Move to history
            self.trade_history.append(trade)
            del self.active_trades[trade_id]
            
            # Update statistics
            self.daily_pnl += trade.pnl
            self.total_pnl += trade.pnl
            
            logger.info(f"Trade closed: {trade.symbol} P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%), Reason: {reason}")
            
            # Send notification
            self._send_trade_notification(trade, "CLOSE")
        
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
    
    def _close_all_positions(self):
        """Close all positions."""
        logger.warning("Closing all positions for risk management")
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            current_tick = self._get_current_tick(trade.symbol)
            if current_tick:
                price = current_tick.get('bid') if trade.direction > 0 else current_tick.get('ask')
                if price:
                    self._close_trade(trade_id, price, 'Risk Management')
    
    def _check_circuit_breakers(self):
        """Check and activate circuit breakers."""
        recent_trades = [t for t in self.trade_history if t.entry_time > datetime.now() - timedelta(hours=1)]
        if len(recent_trades) >= 5:
            losing_trades = [t for t in recent_trades if t.pnl and t.pnl < 0]
            if len(losing_trades) >= 3:
                self.circuit_breaker = True
                logger.warning("Circuit breaker activated: 3 consecutive losses in 1 hour")
    
    def _get_current_tick(self, symbol: str) -> Optional[Dict]:
        """Get current tick for a symbol."""
        if symbol in self.data_cache and isinstance(self.data_cache[symbol], list) and self.data_cache[symbol]:
            return self.data_cache[symbol][-1]
        return None
    
    def _check_market_conditions(self, config: LiveTradingConfig) -> bool:
        """Check if market conditions are suitable."""
        # Basic check - always return True for now
        return True
    
    def _is_within_trading_hours(self, config: LiveTradingConfig, symbol: str) -> bool:
        """Check trading hours."""
        if not config.trading_hours:
            return True
        
        symbol_hours = config.trading_hours.get(symbol)
        if not symbol_hours:
            return True
        
        try:
            now = datetime.now().time()
            start_time = datetime.strptime(symbol_hours[0], '%H:%M').time()
            end_time = datetime.strptime(symbol_hours[1], '%H:%M').time()
            return start_time <= now <= end_time
        except Exception:
            return True
    
    def _is_high_impact_news_time(self) -> bool:
        """Check if it's high impact news time."""
        now = datetime.now()
        news_times = [8, 13, 15]
        current_hour = now.hour
        
        for news_hour in news_times:
            if abs(current_hour - news_hour) <= 1:
                return True
        
        return False
    
    def _has_high_correlation_trades(self, signal) -> bool:
        """Check for highly correlated trades."""
        open_symbols = [trade.symbol for trade in self.active_trades.values()]
        
        correlated_groups = [
            ['EURUSD', 'GBPUSD', 'AUDUSD'],
            ['USDJPY', 'USDCHF', 'USDCAD'],
            ['XAUUSD', 'XAGUSD']
        ]
        
        for group in correlated_groups:
            if signal.symbol in group:
                open_in_group = sum(1 for sym in open_symbols if sym in group)
                if open_in_group >= 2:
                    return True
        
        return False
    
    def _has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return any(trade.symbol == symbol and trade.status == 'open' 
                  for trade in self.active_trades.values())
    
    def _reset_daily_stats(self):
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
    
    def _update_statistics(self):
        """Update real-time statistics."""
        try:
            account_info = self.platform.get_account_summary() if self.platform else {}
            equity = account_info.get('equity', 10000)
            
            if equity > self.equity_high:
                self.equity_high = equity
            
            if self.equity_high > 0:
                self.current_drawdown = (self.equity_high - equity) / self.equity_high * 100
            else:
                self.current_drawdown = 0.0
        
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def _send_trade_notification(self, trade: LiveTrade, action: str):
        """Send trade notification."""
        message = f"{action} {trade.symbol} {trade.direction} @ {trade.entry_price}"
        if action == "CLOSE" and trade.pnl is not None:
            message += f" P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)"
        
        logger.info(f"NOTIFICATION: {message}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        open_positions = len(self.active_trades)
        total_trades = len(self.trade_history)
        
        winning_trades = [t for t in self.trade_history if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in self.trade_history if t.pnl and t.pnl <= 0]
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        return {
            'is_running': self.is_running,
            'open_positions': open_positions,
            'total_trades': total_trades,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'current_drawdown': self.current_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'circuit_breaker': self.circuit_breaker,
            'active_strategies': [name for name, config in self.configs.items() if config.enabled]
        }
