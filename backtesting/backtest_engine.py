# backtesting/backtest_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from numba import jit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class Trade:
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
        return {
            'strategy_name': self.strategy_name,
            'total_return': self.total_return,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'avg_trade': self.avg_trade,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor
        }

@jit(nopython=True)
def calculate_slippage(volume: float, spread: float, volatility: float) -> float:
    """Calcular slippage basado en volumen, spread y volatilidad"""
    base_slippage = spread * 0.1  # 10% del spread
    volume_impact = min(volume * 0.001, 0.01)  # Impacto por volumen
    volatility_impact = volatility * 0.05  # Impacto por volatilidad
    
    return base_slippage + volume_impact + volatility_impact

class BacktestEngine:
    """Motor de backtesting avanzado con soporte para múltiples estrategias"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.open_trade = None
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, data: pd.DataFrame, strategy, 
                    symbol: str = "SYMBOL",
                    commission: float = 0.001,  # 0.1%
                    slippage_model: str = "fixed",
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    position_sizing: str = "fixed",
                    risk_per_trade: float = 0.02) -> BacktestResult:
        """
        Ejecutar backtest avanzado de una estrategia
        """
        logger.info(f"Iniciando backtest para {strategy.name}")
        
        # Reiniciar estado
        self.current_capital = self.initial_capital
        self.position = 0
        self.open_trade = None
        self.trades = []
        self.equity_curve = []
        
        # Ejecutar estrategia
        strategy_data = strategy.run(symbol, data.copy())
        
        # Procesar señales
        for i in range(1, len(strategy_data)):
            current_bar = strategy_data.iloc[i]
            previous_bar = strategy_data.iloc[i-1]
            current_time = strategy_data.index[i]
            
            # Verificar stop loss y take profit para trades abiertos
            if self.open_trade:
                self._check_exit_conditions(current_bar, current_time, commission, slippage_model)
            
            # Procesar nuevas señales
            current_signal = current_bar.get('signal', 0)
            previous_signal = previous_bar.get('signal', 0)
            
            # Solo considerar nuevas señales si no hay posición abierta
            if not self.open_trade and current_signal != 0:
                self._process_signal(current_bar, current_time, symbol, 
                                   commission, slippage_model, position_sizing, risk_per_trade)
            
            # Actualizar curva de equity
            self._update_equity_curve(current_bar, current_time)
        
        # Cerrar posición final si existe
        if self.open_trade:
            last_bar = strategy_data.iloc[-1]
            last_time = strategy_data.index[-1]
            self._close_trade(self.open_trade, last_bar, last_time, commission, slippage_model)
        
        # Calcular métricas avanzadas
        result = self._calculate_advanced_metrics(strategy.name, strategy_data)
        
        logger.info(f"Backtest completado: {result.total_return:.2f}% de retorno")
        return result
    
    def _process_signal(self, bar: pd.Series, timestamp: datetime, symbol: str,
                       commission: float, slippage_model: str,
                       position_sizing: str, risk_per_trade: float):
        """Procesar una nueva señal de trading"""
        signal = bar.get('signal', 0)
        price = bar['close']
        atr = bar.get('atr', 0.001)
        
        # Calcular tamaño de posición
        if position_sizing == "fixed":
            volume = self.current_capital * 0.1 / price  # 10% del capital
        elif position_sizing == "risk_based":
            # Basado en ATR y riesgo por trade
            risk_amount = self.current_capital * risk_per_trade
            volume = risk_amount / (atr * 2)  # Stop loss a 2 ATR
        else:
            volume = self.current_capital * 0.1 / price
        
        # Calcular slippage
        if slippage_model == "fixed":
            slippage = 0.0001
        elif slippage_model == "dynamic":
            spread = bar.get('spread', 0.0002)
            volatility = bar.get('volatility', 0.01)
            slippage = calculate_slippage(volume, spread, volatility)
        else:
            slippage = 0.0001
        
        # Aplicar slippage al precio de entrada
        if signal > 0:  # Long
            entry_price = price * (1 + slippage)
        else:  # Short
            entry_price = price * (1 - slippage)
        
        # Calcular comisión
        trade_commission = volume * entry_price * commission
        
        # Crear trade
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
        
        logger.debug(f"Nueva posición: {symbol} {'LONG' if signal > 0 else 'SHORT'} "
                    f"a {entry_price:.5f}, volumen: {volume:.2f}")
    
    def _check_exit_conditions(self, bar: pd.Series, timestamp: datetime,
                             commission: float, slippage_model: str):
        """Verificar condiciones de salida (stop loss, take profit)"""
        if not self.open_trade:
            return
        
        current_price = bar['close']
        trade = self.open_trade
        
        exit_signal = False
        exit_reason = ""
        
        # Verificar stop loss
        if trade.stop_loss is not None:
            if trade.direction > 0 and current_price <= trade.stop_loss:
                exit_signal = True
                exit_reason = "Stop Loss"
            elif trade.direction < 0 and current_price >= trade.stop_loss:
                exit_signal = True
                exit_reason = "Stop Loss"
        
        # Verificar take profit
        if trade.take_profit is not None:
            if trade.direction > 0 and current_price >= trade.take_profit:
                exit_signal = True
                exit_reason = "Take Profit"
            elif trade.direction < 0 and current_price <= trade.take_profit:
                exit_signal = True
                exit_reason = "Take Profit"
        
        if exit_signal:
            self._close_trade(trade, bar, timestamp, commission, slippage_model, exit_reason)
            self.open_trade = None
    
    def _close_trade(self, trade: Trade, bar: pd.Series, timestamp: datetime,
                    commission: float, slippage_model: str, exit_reason: str = "Signal"):
        """Cerrar una posición existente"""
        price = bar['close']
        
        # Calcular slippage de salida
        if slippage_model == "fixed":
            exit_slippage = 0.0001
        elif slippage_model == "dynamic":
            spread = bar.get('spread', 0.0002)
            volatility = bar.get('volatility', 0.01)
            exit_slippage = calculate_slippage(trade.volume, spread, volatility)
        else:
            exit_slippage = 0.0001
        
        # Aplicar slippage al precio de salida
        if trade.direction > 0:  # Cerrar long
            exit_price = price * (1 - exit_slippage)
        else:  # Cerrar short
            exit_price = price * (1 + exit_slippage)
        
        # Calcular P&L
        if trade.direction > 0:  # Long
            pnl = (exit_price - trade.entry_price) * trade.volume
        else:  # Short
            pnl = (trade.entry_price - exit_price) * trade.volume
        
        # Calcular comisión de salida
        exit_commission = trade.volume * exit_price * commission
        
        # Actualizar trade
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.pnl = pnl - trade.commission - exit_commission
        trade.pnl_pct = (trade.pnl / (trade.volume * trade.entry_price)) * 100
        trade.metadata['exit_reason'] = exit_reason
        
        # Actualizar capital
        self.current_capital += pnl - exit_commission
        self.position = 0
        
        # Agregar a la lista de trades
        self.trades.append(trade)
        
        logger.debug(f"Posición cerrada: {trade.symbol} P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%) "
                    f"Razón: {exit_reason}")
    
    def _update_equity_curve(self, bar: pd.Series, timestamp: datetime):
        """Actualizar la curva de equity"""
        if self.open_trade:
            # Calcular P&L flotante
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
            'drawdown': (current_equity - self.initial_capital) / self.initial_capital * 100
        })
    
    def _calculate_advanced_metrics(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """Calcular métricas de performance avanzadas"""
        if not self.trades:
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
        
        # Métricas básicas
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        avg_trade = np.mean([t.pnl for t in self.trades]) if self.trades else 0
        avg_winning_trade = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_losing_trade = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Drawdown máximo
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown_pct'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            max_drawdown = equity_df['drawdown_pct'].min()
        else:
            max_drawdown = 0.0
        
        # Ratios avanzados
        sharpe_ratio = self._calculate_sharpe_ratio(equity_df)
        sortino_ratio = self._calculate_sortino_ratio(equity_df)
        calmar_ratio = self._calculate_calmar_ratio(total_return, max_drawdown)
        profit_factor = self._calculate_profit_factor(winning_trades, losing_trades)
        recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
        
        # Retornos diarios
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
        """Calcular Sharpe Ratio"""
        if equity_df.empty or len(equity_df) < 2:
            return None
        
        returns = equity_df['equity'].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return None
        
        excess_returns = returns - (risk_free_rate / 252)
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, equity_df: pd.DataFrame, risk_free_rate: float = 0.02) -> Optional[float]:
        """Calcular Sortino Ratio"""
        if equity_df.empty or len(equity_df) < 2:
            return None
        
        returns = equity_df['equity'].pct_change().dropna()
        if returns.empty:
            return None
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return None
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> Optional[float]:
        """Calcular Calmar Ratio"""
        if max_drawdown == 0:
            return None
        return total_return / abs(max_drawdown)
    
    def _calculate_profit_factor(self, winning_trades: List[Trade], losing_trades: List[Trade]) -> Optional[float]:
        """Calcular Profit Factor"""
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        if gross_loss == 0:
            return None if gross_profit == 0 else float('inf')
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> Optional[float]:
        """Calcular Recovery Factor"""
        if max_drawdown == 0:
            return None
        return total_return / abs(max_drawdown)
    
    def _calculate_daily_returns(self, equity_df: pd.DataFrame) -> pd.Series:
        """Calcular retornos diarios"""
        if equity_df.empty:
            return pd.Series()
        
        # Agrupar por día y tomar el último equity del día
        daily_equity = equity_df.set_index('timestamp').resample('D').last()['equity'].dropna()
        daily_returns = daily_equity.pct_change().dropna()
        
        return daily_returns

class MultiStrategyBacktester:
    """Backtester para múltiples estrategias y símbolos"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.backtest_engine = BacktestEngine(initial_capital)
    
    def run_portfolio_backtest(self, strategies: Dict[str, Any], 
                             data_dict: Dict[str, pd.DataFrame],
                             allocation: Dict[str, float] = None) -> Dict[str, BacktestResult]:
        """Ejecutar backtest para un portafolio de estrategias"""
        results = {}
        
        # Si no se especifica asignación, distribuir equitativamente
        if allocation is None:
            allocation = {name: 1.0/len(strategies) for name in strategies.keys()}
        
        total_allocation = sum(allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError("La asignación debe sumar 1.0")
        
        for strategy_name, strategy in strategies.items():
            strategy_capital = self.initial_capital * allocation[strategy_name]
            
            # Ejecutar backtest para cada símbolo de la estrategia
            strategy_results = []
            for symbol in strategy.config.symbols:
                if symbol in data_dict:
                    self.backtest_engine.initial_capital = strategy_capital / len(strategy.config.symbols)
                    result = self.backtest_engine.run_backtest(data_dict[symbol], strategy, symbol)
                    strategy_results.append(result)
            
            # Combinar resultados (simplificado)
            if strategy_results:
                results[strategy_name] = self._combine_strategy_results(strategy_results)
        
        return results
    
    def _combine_strategy_results(self, results: List[BacktestResult]) -> BacktestResult:
        """Combinar resultados de múltiples símbolos para una estrategia"""
        if not results:
            return None
        
        # Combinar métricas (simplificado)
        combined = BacktestResult(
            strategy_name=results[0].strategy_name,
            total_return=np.mean([r.total_return for r in results]),
            total_trades=sum(r.total_trades for r in results),
            winning_trades=sum(r.winning_trades for r in results),
            losing_trades=sum(r.losing_trades for r in results),
            win_rate=np.mean([r.win_rate for r in results]),
            avg_trade=np.mean([r.avg_trade for r in results]),
            avg_winning_trade=np.mean([r.avg_winning_trade for r in results]),
            avg_losing_trade=np.mean([r.avg_losing_trade for r in results]),
            max_drawdown=np.max([r.max_drawdown for r in results])
        )
        
        return combined