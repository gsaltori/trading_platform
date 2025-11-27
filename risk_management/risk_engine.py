# risk_management/risk_engine.py
"""
Advanced Risk Management Engine for the trading platform.

Implements position sizing, risk limits, and portfolio risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_drawdown: float = 0.15  # 15% maximum drawdown
    max_position_size: float = 0.10  # 10% of capital per position
    max_portfolio_risk: float = 0.20  # 20% total portfolio risk
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    max_correlated_positions: int = 3  # Max positions in correlated assets
    correlation_threshold: float = 0.7  # Correlation threshold for grouping
    risk_free_rate: float = 0.02  # Annual risk-free rate
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.5  # Half-Kelly for safety
    min_position_size: float = 0.01  # Minimum 1% position
    max_leverage: float = 10.0  # Maximum leverage


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    direction: int
    entry_price: float
    current_price: float
    volume: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    risk_amount: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0


class PositionSizer:
    """Calculates optimal position sizes based on various methods."""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size.
        
        Uses risk-based sizing with optional Kelly Criterion adjustment.
        """
        if entry_price <= 0 or account_balance <= 0:
            return 0.0
        
        # Basic risk-based position sizing
        risk_amount = account_balance * risk_per_trade
        
        if stop_loss and stop_loss > 0:
            # Calculate distance to stop loss
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance > 0:
                basic_size = risk_amount / stop_distance
            else:
                basic_size = risk_amount / (entry_price * 0.02)  # Default 2% stop
        elif atr and atr > 0:
            # Use ATR-based stop (2 ATR)
            stop_distance = atr * 2
            basic_size = risk_amount / stop_distance
        else:
            # Default: 2% of price as stop distance
            stop_distance = entry_price * 0.02
            basic_size = risk_amount / stop_distance
        
        # Apply Kelly Criterion if enabled and we have the data
        if self.config.use_kelly_criterion and win_rate and avg_win_loss_ratio:
            kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
            kelly_size = account_balance * kelly_fraction / entry_price
            # Use the smaller of basic and Kelly-adjusted size
            position_size = min(basic_size, kelly_size)
        else:
            position_size = basic_size
        
        # Apply limits
        max_size = (account_balance * self.config.max_position_size) / entry_price
        min_size = (account_balance * self.config.min_position_size) / entry_price
        
        position_size = max(min_size, min(position_size, max_size))
        
        return round(position_size, 2)
    
    def _calculate_kelly_fraction(self, win_rate: float, avg_win_loss_ratio: float) -> float:
        """
        Calculate Kelly Criterion fraction.
        
        f* = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = ratio of average win to average loss
        """
        if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
            return self.config.min_position_size
        
        p = win_rate
        q = 1 - win_rate
        b = avg_win_loss_ratio
        
        kelly = (p * b - q) / b
        
        # Apply fractional Kelly for safety
        kelly = kelly * self.config.kelly_fraction
        
        # Ensure positive and bounded
        kelly = max(0, min(kelly, self.config.max_position_size))
        
        return kelly
    
    def volatility_adjusted_size(
        self,
        account_balance: float,
        target_volatility: float,
        asset_volatility: float,
        entry_price: float
    ) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Useful for maintaining consistent portfolio risk across different volatility regimes.
        """
        if asset_volatility <= 0 or entry_price <= 0:
            return 0.0
        
        # Calculate required position to achieve target volatility
        volatility_ratio = target_volatility / asset_volatility
        position_value = account_balance * volatility_ratio
        position_size = position_value / entry_price
        
        # Apply limits
        max_size = (account_balance * self.config.max_position_size) / entry_price
        position_size = min(position_size, max_size)
        
        return round(position_size, 2)


class RiskEngine:
    """
    Main risk management engine.
    
    Handles position sizing, portfolio risk, drawdown management,
    and risk limit enforcement.
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.position_sizer = PositionSizer(config)
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.positions: Dict[str, PositionRisk] = {}
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Historical data for analysis
        self.pnl_history: List[float] = []
        self.drawdown_history: List[float] = []
    
    def calculate_position_size(
        self,
        signal: Any,  # TradeSignal
        risk_per_trade: float,
        account_info: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate position size for a trade signal.
        
        Args:
            signal: Trade signal with symbol, direction, price, stop_loss
            risk_per_trade: Risk per trade as fraction (e.g., 0.02 for 2%)
            account_info: Account information with balance, equity
            historical_data: Optional historical data for ATR calculation
        
        Returns:
            Position size in units
        """
        account_balance = account_info.get('balance', 0)
        if account_balance <= 0:
            return 0.0
        
        entry_price = signal.price if hasattr(signal, 'price') else 0
        stop_loss = signal.stop_loss if hasattr(signal, 'stop_loss') else None
        
        # Calculate ATR if historical data provided
        atr = None
        if historical_data is not None and len(historical_data) >= 14:
            atr = self._calculate_atr(historical_data)
        
        # Get win rate and avg win/loss if available
        win_rate = None
        avg_win_loss_ratio = None
        
        # Calculate base position size
        position_size = self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss=stop_loss,
            atr=atr,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio
        )
        
        # Apply portfolio-level risk checks
        position_size = self._apply_portfolio_limits(
            position_size, entry_price, signal, account_info
        )
        
        return position_size
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(data) < period:
            return 0.0
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        
        return float(atr)
    
    def _apply_portfolio_limits(
        self,
        position_size: float,
        entry_price: float,
        signal: Any,
        account_info: Dict[str, Any]
    ) -> float:
        """Apply portfolio-level risk limits."""
        balance = account_info.get('balance', 0)
        
        # Check daily loss limit
        if self._is_daily_limit_reached(balance):
            logger.warning("Daily loss limit reached - reducing position size")
            return 0.0
        
        # Check maximum drawdown
        if self._is_max_drawdown_reached(account_info):
            logger.warning("Maximum drawdown reached - reducing position size")
            return position_size * 0.5
        
        # Check portfolio concentration
        position_value = position_size * entry_price
        total_exposure = self._calculate_total_exposure()
        
        if (total_exposure + position_value) / balance > self.config.max_portfolio_risk:
            # Reduce position to fit within limits
            available_exposure = (balance * self.config.max_portfolio_risk) - total_exposure
            if available_exposure > 0:
                position_size = available_exposure / entry_price
            else:
                position_size = 0.0
        
        return max(0, position_size)
    
    def check_trade_risk(
        self,
        signal: Any,
        config: Any,  # LiveTradingConfig
        portfolio_status: Dict[str, Any]
    ) -> bool:
        """
        Check if a trade passes all risk criteria.
        
        Returns True if trade is allowed, False otherwise.
        """
        # Check if we're within daily loss limit
        daily_pnl = portfolio_status.get('daily_pnl', 0)
        balance = portfolio_status.get('balance', 10000)
        
        if balance > 0 and daily_pnl / balance < -self.config.daily_loss_limit:
            logger.warning("Trade rejected: Daily loss limit exceeded")
            return False
        
        # Check drawdown
        current_drawdown = portfolio_status.get('current_drawdown', 0)
        if current_drawdown > self.config.max_drawdown * 100:
            logger.warning("Trade rejected: Maximum drawdown exceeded")
            return False
        
        # Check open positions limit
        open_positions = portfolio_status.get('open_positions', 0)
        max_positions = config.max_positions if hasattr(config, 'max_positions') else 5
        
        if open_positions >= max_positions:
            logger.warning("Trade rejected: Maximum positions reached")
            return False
        
        # Check correlation limits
        if not self._check_correlation_limits(signal, portfolio_status):
            logger.warning("Trade rejected: Correlation limit exceeded")
            return False
        
        return True
    
    def _check_correlation_limits(
        self,
        signal: Any,
        portfolio_status: Dict[str, Any]
    ) -> bool:
        """Check if adding this position would exceed correlation limits."""
        # Simplified correlation groups
        correlation_groups = {
            'major_usd': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
            'jpy_pairs': ['USDJPY', 'EURJPY', 'GBPJPY'],
            'commodity': ['XAUUSD', 'XAGUSD', 'USDCAD'],
        }
        
        symbol = signal.symbol if hasattr(signal, 'symbol') else ''
        
        for group_name, symbols in correlation_groups.items():
            if symbol in symbols:
                # Count how many correlated positions we have
                correlated_count = sum(
                    1 for pos in self.positions.values()
                    if pos.symbol in symbols
                )
                
                if correlated_count >= self.config.max_correlated_positions:
                    return False
        
        return True
    
    def _is_daily_limit_reached(self, balance: float) -> bool:
        """Check if daily loss limit has been reached."""
        self._reset_daily_stats_if_needed()
        
        if balance <= 0:
            return True
        
        daily_loss_pct = abs(self.daily_pnl) / balance
        return self.daily_pnl < 0 and daily_loss_pct >= self.config.daily_loss_limit
    
    def _is_max_drawdown_reached(self, account_info: Dict[str, Any]) -> bool:
        """Check if maximum drawdown has been reached."""
        equity = account_info.get('equity', 0)
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        if self.peak_equity <= 0:
            return False
        
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        self.drawdown_history.append(self.current_drawdown)
        
        return self.current_drawdown >= self.config.max_drawdown
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure."""
        return sum(
            pos.volume * pos.current_price
            for pos in self.positions.values()
        )
    
    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if it's a new day."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
    
    def update_position(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        current_price: float,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """Update or add a position in the risk tracking."""
        if direction > 0:  # Long
            unrealized_pnl = (current_price - entry_price) * volume
        else:  # Short
            unrealized_pnl = (entry_price - current_price) * volume
        
        unrealized_pnl_pct = (unrealized_pnl / (entry_price * volume)) * 100 if entry_price * volume > 0 else 0
        
        # Calculate risk amount (distance to stop loss)
        if stop_loss:
            risk_amount = abs(entry_price - stop_loss) * volume
        else:
            risk_amount = entry_price * volume * 0.02  # Default 2% risk
        
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            volume=volume,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            risk_amount=risk_amount,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def close_position(self, symbol: str, pnl: float):
        """Close a position and update daily P&L."""
        if symbol in self.positions:
            del self.positions[symbol]
        
        self.daily_pnl += pnl
        self.pnl_history.append(pnl)
        self.daily_trades += 1
    
    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary."""
        total_exposure = self._calculate_total_exposure()
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_positions': len(self.positions),
            'total_exposure': total_exposure,
            'total_risk_amount': total_risk,
            'total_unrealized_pnl': total_unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'positions': {
                symbol: {
                    'direction': pos.direction,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'risk_amount': pos.risk_amount
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            holding_period: Holding period in days
        
        Returns:
            VaR as a positive number (potential loss)
        """
        if len(returns) < 30:
            return 0.0
        
        # Scale for holding period
        scaled_returns = returns * np.sqrt(holding_period)
        
        # Calculate VaR at the specified confidence level
        var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        
        return abs(var)
    
    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR).
        
        This is the expected loss given that the loss exceeds VaR.
        """
        if len(returns) < 30:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        
        # Expected shortfall is the mean of losses beyond VaR
        losses = returns[returns < -var]
        
        if len(losses) == 0:
            return var
        
        return abs(np.mean(losses))
