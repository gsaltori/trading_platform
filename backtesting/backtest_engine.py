# backtesting/backtest_engine.py
"""
Advanced backtesting engine with realistic execution simulation.

Supports multiple execution modes, slippage models, and comprehensive metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

# Try to import numba for performance
NUMBA_AVAILABLE = False
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class Trade:
    """Trade record."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: Optional[float]
    volume: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Backtest result container."""
    strategy_name: str
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade: float
    avg_winning_trade: float
    avg_losing_trade: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    recovery_factor: Optional[float] = None
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'total_return': self.total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_trade': self.avg_trade,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor
        }


if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def calculate_slippage_numba(volume: float, spread: float, volatility: float) -> float:
        """Calculate slippage with numba optimization."""
        base_slippage = spread * 0.1
        volume_impact = min(volume * 0.001, 0.01)
        volatility_impact = volatility * 0.05
        return base_slippage + volume_impact + volatility_impact
else:
    def calculate_slippage_numba(volume: float, spread: float, volatility: float) -> float:
        """Calculate slippage without numba."""
        base_slippage = spread * 0.1
        volume_impact = min(volume * 0.001, 0.01)
        volatility_impact = volatility * 0.05
        return base_slippage + volume_impact + volatility_impact


class BacktestEngine:
    """Advanced backtesting engine with multiple strategies support."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.open_trade = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
    
    def run_backtest(
        self, 
        data: pd.DataFrame, 
        strategy,
        symbol: str = "SYMBOL",
        commission: float = 0.001,
        slippage_model: str = "fixed",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_sizing: str = "fixed",
        risk_per_trade: float = 0.02
    ) -> BacktestResult:
        """
        Run advanced backtest on a strategy.
        
        Args:
            data: OHLCV data
            strategy: Strategy instance
            symbol: Trading symbol
            commission: Commission rate (0.001 = 0.1%)
            slippage_model: 'fixed' or 'dynamic'
            stop_loss: Fixed stop loss percentage
            take_profit: Fixed take profit percentage
            position_sizing: 'fixed' or 'risk_based'
            risk_per_trade: Risk per trade for risk-based sizing
        
        Returns:
            BacktestResult with complete metrics
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.position = 0
        self.open_trade = None
        self.trades = []
        self.equity_curve = []
        
        # Run strategy
        try:
            strategy_data = strategy.run(symbol, data.copy())
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            return self._create_empty_result(strategy.name)
        
        if strategy_data is None or len(strategy_data) < 2:
            return self._create_empty_result(strategy.name)
        
        # Process signals
        for i in range(1, len(strategy_data)):
            current_bar = strategy_data.iloc[i]
            previous_bar = strategy_data.iloc[i-1]
            current_time = strategy_data.index[i]
            
            # Check exit conditions for open trades
            if self.open_trade:
                self._check_exit_conditions(current_bar, current_time, commission, slippage_model)
            
            # Process new signals
            current_signal = current_bar.get('signal', 0)
            
            # Only open new position if no position is open
            if not self.open_trade and current_signal != 0:
                self._process_signal(
                    current_bar, current_time, symbol,
                    commission, slippage_model, position_sizing, risk_per_trade
                )
            
            # Update equity curve
            self._update_equity_curve(current_bar, current_time)
        
        # Close final position
        if self.open_trade:
            last_bar = strategy_data.iloc[-1]
            last_time = strategy_data.index[-1]
            self._close_trade(self.open_trade, last_bar, last_time, commission, slippage_model)
        
        # Calculate metrics
        result = self._calculate_advanced_metrics(strategy.name, strategy_data)
        
        logger.info(f"Backtest completed: {result.total_return:.2f}% return, {result.total_trades} trades")
        return result
    
    def _create_empty_result(self, strategy_name: str) -> BacktestResult:
        """Create empty result for failed backtests."""
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_trade=0.0,
            avg_winning_trade=0.0,
            avg_losing_trade=0.0,
            max_drawdown=0.0
        )
    
    def _process_signal(
        self, 
        bar: pd.Series, 
        timestamp: datetime,
        symbol: str,
        commission: float, 
        slippage_model: str,
        position_sizing: str, 
        risk_per_trade: float
    ):
        """Process a new trading signal."""
        signal = bar.get('signal', 0)
        price = bar['close']
        atr = bar.get('atr', price * 0.01)
        
        # Calculate position size
        if position_sizing == "fixed":
            volume = self.current_capital * 0.1 / price  # 10% of capital
        elif position_sizing == "risk_based":
            risk_amount = self.current_capital * risk_per_trade
            stop_distance = atr * 2 if atr > 0 else price * 0.02
            volume = risk_amount / stop_distance if stop_distance > 0 else 0
        else:
            volume = self.current_capital * 0.1 / price
        
        if volume <= 0:
            return
        
        # Calculate slippage
        if slippage_model == "fixed":
            slippage = 0.0001
        elif slippage_model == "dynamic":
            spread = bar.get('spread', 0.0002)
            volatility = bar.get('volatility', 0.01)
            slippage = calculate_slippage_numba(volume, spread, volatility)
        else:
            slippage = 0.0001
        
        # Apply slippage to entry price
        if signal > 0:  # Long
            entry_price = price * (1 + slippage)
        else:  # Short
            entry_price = price * (1 - slippage)
        
        # Calculate commission
        trade_commission = volume * entry_price * commission
        
        # Create trade
        self.open_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            symbol=symbol,
            direction=signal,
            entry_price=entry_price,
            exit_price=None,
            volume=volume,
            commission=trade_commission,
            slippage=slippage,
            stop_loss=bar.get('stop_loss'),
            take_profit=bar.get('take_profit')
        )
        
        self.current_capital -= trade_commission
        self.position = signal
        
        logger.debug(f"New position: {symbol} {'LONG' if signal > 0 else 'SHORT'} "
                    f"at {entry_price:.5f}, volume: {volume:.2f}")
    
    def _check_exit_conditions(
        self, 
        bar: pd.Series, 
        timestamp: datetime,
        commission: float, 
        slippage_model: str
    ):
        """Check exit conditions for open trade."""
        if not self.open_trade:
            return
        
        current_price = bar['close']
        trade = self.open_trade
        
        exit_signal = False
        exit_reason = ""
        
        # Check stop loss
        if trade.stop_loss is not None and trade.stop_loss > 0:
            if trade.direction > 0 and current_price <= trade.stop_loss:
                exit_signal = True
                exit_reason = "Stop Loss"
            elif trade.direction < 0 and current_price >= trade.stop_loss:
                exit_signal = True
                exit_reason = "Stop Loss"
        
        # Check take profit
        if trade.take_profit is not None and trade.take_profit > 0:
            if trade.direction > 0 and current_price >= trade.take_profit:
                exit_signal = True
                exit_reason = "Take Profit"
            elif trade.direction < 0 and current_price <= trade.take_profit:
                exit_signal = True
                exit_reason = "Take Profit"
        
        # Check for signal reversal
        signal = bar.get('signal', 0)
        if signal != 0 and signal != trade.direction:
            exit_signal = True
            exit_reason = "Signal Reversal"
        
        if exit_signal:
            self._close_trade(trade, bar, timestamp, commission, slippage_model, exit_reason)
            self.open_trade = None
    
    def _close_trade(
        self, 
        trade: Trade, 
        bar: pd.Series, 
        timestamp: datetime,
        commission: float, 
        slippage_model: str, 
        exit_reason: str = "Signal"
    ):
        """Close a position."""
        price = bar['close']
        
        # Calculate exit slippage
        if slippage_model == "fixed":
            exit_slippage = 0.0001
        elif slippage_model == "dynamic":
            spread = bar.get('spread', 0.0002)
            volatility = bar.get('volatility', 0.01)
            exit_slippage = calculate_slippage_numba(trade.volume, spread, volatility)
        else:
            exit_slippage = 0.0001
        
        # Apply slippage to exit price
        if trade.direction > 0:  # Close long
            exit_price = price * (1 - exit_slippage)
        else:  # Close short
            exit_price = price * (1 + exit_slippage)
        
        # Calculate P&L
        if trade.direction > 0:  # Long
            pnl = (exit_price - trade.entry_price) * trade.volume
        else:  # Short
            pnl = (trade.entry_price - exit_price) * trade.volume
        
        # Calculate exit commission
        exit_commission = trade.volume * exit_price * commission
        
        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.pnl = pnl - trade.commission - exit_commission
        trade.pnl_pct = (trade.pnl / (trade.volume * trade.entry_price)) * 100 if trade.volume * trade.entry_price > 0 else 0
        trade.metadata['exit_reason'] = exit_reason
        
        # Update capital
        self.current_capital += pnl - exit_commission
        self.position = 0
        
        # Add to trades list
        self.trades.append(trade)
        
        logger.debug(f"Position closed: {trade.symbol} P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%), Reason: {exit_reason}")
    
    def _update_equity_curve(self, bar: pd.Series, timestamp: datetime):
        """Update equity curve."""
        if self.open_trade:
            # Calculate floating P&L
            if self.open_trade.direction > 0:  # Long
                floating_pnl = (bar['close'] - self.open_trade.entry_price) * self.open_trade.volume
            else:  # Short
                floating_pnl = (self.open_trade.entry_price - bar['close']) * self.open_trade.volume
            
            current_equity = self.current_capital + floating_pnl
        else:
            current_equity = self.current_capital
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'capital': self.current_capital,
            'position': self.position
        })
    
    def _calculate_advanced_metrics(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return self._create_empty_result(strategy_name)
        
        # Basic metrics
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl <= 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        avg_trade = np.mean([t.pnl for t in self.trades if t.pnl]) if self.trades else 0
        avg_winning_trade = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_losing_trade = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Equity curve analysis
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown_pct'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            max_drawdown = abs(equity_df['drawdown_pct'].min())
        else:
            max_drawdown = 0.0
        
        # Advanced ratios
        sharpe_ratio = self._calculate_sharpe_ratio(equity_df)
        sortino_ratio = self._calculate_sortino_ratio(equity_df)
        calmar_ratio = self._calculate_calmar_ratio(total_return, max_drawdown)
        profit_factor = self._calculate_profit_factor(winning_trades, losing_trades)
        recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
        
        # Daily returns
        daily_returns = self._calculate_daily_returns(equity_df)
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_trade=avg_trade,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            trades=self.trades,
            equity_curve=equity_df,
            daily_returns=daily_returns
        )
    
    def _calculate_sharpe_ratio(self, equity_df: pd.DataFrame, risk_free_rate: float = 0.02) -> Optional[float]:
        """Calculate Sharpe Ratio."""
        if equity_df.empty or len(equity_df) < 2:
            return None
        
        returns = equity_df['equity'].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return None
        
        excess_returns = returns - (risk_free_rate / 252)
        return float(np.sqrt(252) * excess_returns.mean() / returns.std())
    
    def _calculate_sortino_ratio(self, equity_df: pd.DataFrame, risk_free_rate: float = 0.02) -> Optional[float]:
        """Calculate Sortino Ratio."""
        if equity_df.empty or len(equity_df) < 2:
            return None
        
        returns = equity_df['equity'].pct_change().dropna()
        if returns.empty:
            return None
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return None
        
        return float(np.sqrt(252) * excess_returns.mean() / downside_returns.std())
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> Optional[float]:
        """Calculate Calmar Ratio."""
        if max_drawdown == 0:
            return None
        return total_return / abs(max_drawdown)
    
    def _calculate_profit_factor(self, winning_trades: List[Trade], losing_trades: List[Trade]) -> Optional[float]:
        """Calculate Profit Factor."""
        gross_profit = sum(t.pnl for t in winning_trades if t.pnl) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl)) if losing_trades else 0
        
        if gross_loss == 0:
            return None if gross_profit == 0 else float('inf')
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> Optional[float]:
        """Calculate Recovery Factor."""
        if max_drawdown == 0:
            return None
        return total_return / abs(max_drawdown)
    
    def _calculate_daily_returns(self, equity_df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns."""
        if equity_df.empty:
            return pd.Series()
        
        try:
            daily_equity = equity_df.set_index('timestamp').resample('D').last()['equity'].dropna()
            daily_returns = daily_equity.pct_change().dropna()
            return daily_returns
        except Exception:
            return pd.Series()


class MultiStrategyBacktester:
    """Backtester for multiple strategies and symbols."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.backtest_engine = BacktestEngine(initial_capital)
    
    def run_portfolio_backtest(
        self, 
        strategies: Dict[str, Any],
        data_dict: Dict[str, pd.DataFrame],
        allocation: Dict[str, float] = None
    ) -> Dict[str, BacktestResult]:
        """Run backtest for a portfolio of strategies."""
        results = {}
        
        # If no allocation specified, distribute equally
        if allocation is None:
            allocation = {name: 1.0 / len(strategies) for name in strategies.keys()}
        
        total_allocation = sum(allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            logger.warning(f"Allocation sums to {total_allocation}, normalizing to 1.0")
            allocation = {k: v / total_allocation for k, v in allocation.items()}
        
        for strategy_name, strategy in strategies.items():
            strategy_capital = self.initial_capital * allocation.get(strategy_name, 0)
            
            if strategy_capital <= 0:
                continue
            
            strategy_results = []
            for symbol in strategy.config.symbols:
                if symbol in data_dict:
                    self.backtest_engine.initial_capital = strategy_capital / len(strategy.config.symbols)
                    try:
                        result = self.backtest_engine.run_backtest(data_dict[symbol], strategy, symbol)
                        strategy_results.append(result)
                    except Exception as e:
                        logger.error(f"Backtest failed for {strategy_name} on {symbol}: {e}")
            
            if strategy_results:
                results[strategy_name] = self._combine_strategy_results(strategy_results)
        
        return results
    
    def _combine_strategy_results(self, results: List[BacktestResult]) -> BacktestResult:
        """Combine results from multiple symbols for a strategy."""
        if not results:
            return None
        
        combined = BacktestResult(
            strategy_name=results[0].strategy_name,
            total_return=float(np.mean([r.total_return for r in results])),
            total_trades=sum(r.total_trades for r in results),
            winning_trades=sum(r.winning_trades for r in results),
            losing_trades=sum(r.losing_trades for r in results),
            win_rate=float(np.mean([r.win_rate for r in results])),
            avg_trade=float(np.mean([r.avg_trade for r in results])),
            avg_winning_trade=float(np.mean([r.avg_winning_trade for r in results if r.avg_winning_trade])),
            avg_losing_trade=float(np.mean([r.avg_losing_trade for r in results if r.avg_losing_trade])),
            max_drawdown=float(np.max([r.max_drawdown for r in results])),
            sharpe_ratio=float(np.mean([r.sharpe_ratio for r in results if r.sharpe_ratio])) if any(r.sharpe_ratio for r in results) else None,
            profit_factor=float(np.mean([r.profit_factor for r in results if r.profit_factor and r.profit_factor != float('inf')])) if any(r.profit_factor for r in results) else None
        )
        
        return combined
