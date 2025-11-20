# execution/live_execution.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from threading import Thread, Event, Lock
from queue import Queue, Empty
import warnings

from core.platform import get_platform
from strategies.strategy_engine import StrategyEngine, TradeSignal
from risk_management.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

@dataclass
class LiveTradingConfig:
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
    """Motor de ejecución en vivo avanzado con gestión inteligente"""
    
    def __init__(self):
        self.platform = get_platform()
        self.strategy_engine = StrategyEngine()
        self.risk_engine = RiskEngine()
        self.configs = {}
        self.active_trades = {}
        self.trade_history = []
        self.is_running = False
        self.monitor_thread = None
        self.stop_event = Event()
        self.data_queue = Queue()
        self.lock = Lock()
        self.circuit_breaker = False
        
        # Estadísticas en tiempo real
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.current_drawdown = 0.0
        self.daily_trades = 0
        self.equity_high = 0.0
        
        # Cache de datos
        self.data_cache = {}
        self.signal_cache = {}
        
    def add_strategy(self, config: LiveTradingConfig) -> bool:
        """Agregar estrategia para trading en vivo"""
        if config.strategy_name not in self.strategy_engine.strategies:
            logger.error(f"Estrategia {config.strategy_name} no encontrada")
            return False
        
        self.configs[config.strategy_name] = config
        
        # Suscribir símbolos para datos en tiempo real
        for symbol in config.symbols:
            self.platform.mt5_connector.subscribe_symbol(symbol)
        
        logger.info(f"Estrategia {config.strategy_name} agregada para trading en vivo")
        return True
    
    def start_trading(self):
        """Iniciar motor de trading en vivo"""
        if self.is_running:
            logger.warning("Motor de trading ya está en ejecución")
            return
        
        if not self.platform.initialized:
            logger.error("Plataforma no inicializada")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Reiniciar estadísticas diarias
        self._reset_daily_stats()
        
        # Iniciar hilos de monitorización
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Iniciar hilo de procesamiento de datos
        self.data_thread = Thread(target=self._data_processing_loop, daemon=True)
        self.data_thread.start()
        
        logger.info("Motor de trading en vivo iniciado")
    
    def stop_trading(self):
        """Detener motor de trading en vivo"""
        self.is_running = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        if self.data_thread:
            self.data_thread.join(timeout=10)
        
        logger.info("Motor de trading en vivo detenido")
    
    def _monitoring_loop(self):
        """Bucle principal de monitorización"""
        while not self.stop_event.is_set() and self.is_running:
            try:
                # Verificar circuit breaker
                if self.circuit_breaker:
                    time.sleep(30)
                    continue
                
                # Procesar cada estrategia activa
                for strategy_name, config in self.configs.items():
                    if not config.enabled:
                        continue
                    
                    # Verificar condiciones de mercado
                    if not self._check_market_conditions(config):
                        continue
                    
                    # Procesar cada símbolo de la estrategia
                    for symbol in config.symbols:
                        self._process_symbol(strategy_name, symbol, config)
                
                # Verificar gestión de riesgo y stops
                self._check_risk_management()
                
                # Actualizar estadísticas
                self._update_statistics()
                
                # Verificar circuit breakers
                self._check_circuit_breakers()
                
                time.sleep(5)  # 5 segundos entre iteraciones
                
            except Exception as e:
                logger.error(f"Error en bucle de monitorización: {e}")
                time.sleep(30)
    
    def _data_processing_loop(self):
        """Bucle de procesamiento de datos en tiempo real"""
        while not self.stop_event.is_set() and self.is_running:
            try:
                # Obtener tick más reciente
                tick = self.platform.mt5_connector.get_next_tick(timeout=1.0)
                if tick:
                    self._process_tick_data(tick)
                
            except Exception as e:
                logger.error(f"Error en procesamiento de datos: {e}")
                time.sleep(5)
    
    def _process_tick_data(self, tick: Dict):
        """Procesar datos de tick en tiempo real"""
        symbol = tick['symbol']
        
        # Actualizar cache de datos
        if symbol not in self.data_cache:
            self.data_cache[symbol] = []
        
        self.data_cache[symbol].append(tick)
        
        # Mantener solo los últimos 1000 ticks
        if len(self.data_cache[symbol]) > 1000:
            self.data_cache[symbol] = self.data_cache[symbol][-1000:]
    
    def _process_symbol(self, strategy_name: str, symbol: str, config: LiveTradingConfig):
        """Procesar un símbolo para generar señales"""
        try:
            # Verificar si ya tenemos posición abierta
            if self._has_open_position(symbol):
                return
            
            # Obtener datos recientes
            data = self._get_recent_data(symbol, config.timeframe)
            if data is None or len(data) < 100:
                return
            
            # Generar señales
            signals = self.strategy_engine.get_strategy_signals(strategy_name, symbol, data)
            if not signals:
                return
            
            # Tomar la señal más reciente
            latest_signal = signals[-1]
            
            # Verificar filtros avanzados
            if not self._apply_advanced_filters(latest_signal, data, config):
                return
            
            # Verificar riesgo
            if not self.risk_engine.check_trade_risk(latest_signal, config, self.get_portfolio_status()):
                return
            
            # Ejecutar trade
            self._execute_trade(latest_signal, strategy_name, config)
            
        except Exception as e:
            logger.error(f"Error procesando símbolo {symbol}: {e}")
    
    def _get_recent_data(self, symbol: str, timeframe: str, 
                        lookback_bars: int = 100) -> Optional[pd.DataFrame]:
        """Obtener datos recientes con cache inteligente"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Verificar cache primero
        if cache_key in self.data_cache and len(self.data_cache[cache_key]) >= lookback_bars:
            return self.data_cache[cache_key].tail(lookback_bars)
        
        # Obtener de MT5 si no hay cache
        data = self.platform.get_market_data(symbol, timeframe, days=30)
        if data is not None:
            self.data_cache[cache_key] = data
            return data.tail(lookback_bars)
        
        return None
    
    def _apply_advanced_filters(self, signal: TradeSignal, data: pd.DataFrame, 
                              config: LiveTradingConfig) -> bool:
        """Aplicar filtros avanzados a las señales"""
        
        # Filtro de volatilidad
        if config.volatility_filter:
            recent_volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            if recent_volatility > 0.02:  # 2% de volatilidad
                logger.info(f"Filtro de volatilidad activado para {signal.symbol}")
                return False
        
        # Filtro de horario de trading
        if not self._is_within_trading_hours(config, signal.symbol):
            return False
        
        # Filtro de noticias (implementación básica)
        if config.news_filter:
            if self._is_high_impact_news_time():
                logger.info("Filtro de noticias activado")
                return False
        
        # Filtro de correlación
        if self._has_high_correlation_trades(signal):
            return False
        
        return True
    
    def _execute_trade(self, signal: TradeSignal, strategy_name: str, config: LiveTradingConfig):
        """Ejecutar trade con gestión avanzada de órdenes"""
        try:
            with self.lock:
                # Verificar límites
                if len(self.active_trades) >= config.max_positions:
                    logger.warning("Límite de posiciones alcanzado")
                    return
                
                if self.daily_trades >= 20:  # Límite diario de operaciones
                    logger.warning("Límite diario de operaciones alcanzado")
                    return
                
                # Calcular tamaño de posición
                account_info = self.platform.get_account_summary()
                volume = self.risk_engine.calculate_position_size(
                    signal, config.risk_per_trade, account_info
                )
                
                if volume <= 0:
                    return
                
                # Colocar orden con tipo de ejecución inteligente
                order_type = 'BUY' if signal.direction > 0 else 'SELL'
                current_price = signal.price
                
                # Determinar tipo de orden basado en volatilidad
                volatility = self._get_current_volatility(signal.symbol)
                if volatility > 0.01:  # Alta volatilidad
                    # Usar orden limit para mejor precio
                    price = current_price * (0.999 if order_type == 'BUY' else 1.001)
                    order_result = self._place_limit_order(signal, order_type, volume, price)
                else:
                    # Usar orden market para ejecución rápida
                    order_result = self._place_market_order(signal, order_type, volume)
                
                if order_result['success']:
                    # Registrar trade
                    trade = LiveTrade(
                        id=order_result['order_id'],
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=order_result['price'],
                        volume=volume,
                        entry_time=datetime.now(),
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        strategy=strategy_name,
                        metadata={
                            'signal_strength': signal.strength,
                            'volatility': volatility,
                            'order_type': order_result.get('order_type', 'MARKET')
                        }
                    )
                    
                    self.active_trades[order_result['order_id']] = trade
                    self.daily_trades += 1
                    
                    logger.info(f"Trade ejecutado: {order_type} {signal.symbol} "
                               f"a {order_result['price']:.5f}, volumen: {volume:.2f}")
                    
                    # Enviar notificación
                    self._send_trade_notification(trade, "OPEN")
                
                else:
                    logger.error(f"Error ejecutando orden: {order_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error ejecutando trade: {e}")
    
    def _place_market_order(self, signal: TradeSignal, order_type: str, volume: float) -> Dict:
        """Colocar orden de mercado"""
        return self.platform.mt5_connector.place_order(
            symbol=signal.symbol,
            order_type=order_type,
            volume=volume,
            price=None,  # Precio de mercado
            sl=signal.stop_loss,
            tp=signal.take_profit,
            deviation=10,
            comment=f"LIVE_{order_type}"
        )
    
    def _place_limit_order(self, signal: TradeSignal, order_type: str, volume: float, price: float) -> Dict:
        """Colocar orden limitada (implementación básica)"""
        # Por simplicidad, usamos orden de mercado con precio específico
        return self.platform.mt5_connector.place_order(
            symbol=signal.symbol,
            order_type=order_type,
            volume=volume,
            price=price,
            sl=signal.stop_loss,
            tp=signal.take_profit,
            deviation=5,
            comment=f"LIVE_LIMIT_{order_type}"
        )
    
    def _check_risk_management(self):
        """Verificar y aplicar gestión de riesgo en tiempo real"""
        try:
            # Verificar stops para trades abiertos
            for trade_id, trade in list(self.active_trades.items()):
                if trade.status != 'open':
                    continue
                
                # Obtener precio actual
                current_tick = self._get_current_tick(trade.symbol)
                if not current_tick:
                    continue
                
                current_price = current_tick['bid'] if trade.direction > 0 else current_tick['ask']
                
                # Verificar stop loss
                if trade.stop_loss and self._check_stop_condition(trade, current_price):
                    self._close_trade(trade_id, current_price, 'Stop Loss')
                    continue
                
                # Verificar take profit
                if trade.take_profit and self._check_take_profit_condition(trade, current_price):
                    self._close_trade(trade_id, current_price, 'Take Profit')
                    continue
                
                # Verificar stop loss trailing
                self._update_trailing_stop(trade, current_price)
            
            # Verificar límites globales de riesgo
            self._check_global_risk_limits()
                
        except Exception as e:
            logger.error(f"Error en verificación de riesgo: {e}")
    
    def _check_stop_condition(self, trade: LiveTrade, current_price: float) -> bool:
        """Verificar condición de stop loss"""
        if trade.direction > 0:  # Long
            return current_price <= trade.stop_loss
        else:  # Short
            return current_price >= trade.stop_loss
    
    def _check_take_profit_condition(self, trade: LiveTrade, current_price: float) -> bool:
        """Verificar condición de take profit"""
        if trade.direction > 0:  # Long
            return current_price >= trade.take_profit
        else:  # Short
            return current_price <= trade.take_profit
    
    def _update_trailing_stop(self, trade: LiveTrade, current_price: float):
        """Actualizar stop loss trailing"""
        if trade.direction > 0:  # Long
            new_stop = current_price * 0.99  # 1% below current price
            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
        else:  # Short
            new_stop = current_price * 1.01  # 1% above current price
            if new_stop < trade.stop_loss:
                trade.stop_loss = new_stop
    
    def _check_global_risk_limits(self):
        """Verificar límites globales de riesgo"""
        account_info = self.platform.get_account_summary()
        if not account_info:
            return
        
        balance = account_info.get('balance', 0)
        equity = account_info.get('equity', 0)
        
        # Verificar drawdown diario
        if self.daily_pnl < -abs(balance * 0.05):  # 5% diario
            logger.warning("Límite de pérdida diaria alcanzado")
            self._close_all_positions()
            return
        
        # Verificar drawdown máximo
        drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        if drawdown > 15:  # 15% máximo
            logger.warning("Drawdown máximo alcanzado")
            self._close_all_positions()
            self.circuit_breaker = True
            return
        
        # Verificar margen
        margin_used = account_info.get('margin', 0)
        free_margin = account_info.get('free_margin', 0)
        if margin_used > 0 and free_margin / margin_used < 0.1:  # 10% margen libre
            logger.warning("Nivel de margen bajo")
            self._close_some_positions()
    
    def _close_trade(self, trade_id: str, price: float, reason: str):
        """Cerrar trade específico"""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return
            
            # Cerrar orden
            order_type = 'SELL' if trade.direction > 0 else 'BUY'
            result = self.platform.mt5_connector.place_order(
                symbol=trade.symbol,
                order_type=order_type,
                volume=trade.volume,
                price=price,
                comment=f"CLOSE_{reason}"
            )
            
            if result['success']:
                # Actualizar trade
                trade.exit_time = datetime.now()
                trade.exit_price = price
                trade.status = 'closed'
                
                # Calcular P&L
                if trade.direction > 0:
                    trade.pnl = (price - trade.entry_price) * trade.volume
                else:
                    trade.pnl = (trade.entry_price - price) * trade.volume
                
                trade.pnl_pct = (trade.pnl / (trade.volume * trade.entry_price)) * 100
                
                # Mover a historial
                self.trade_history.append(trade)
                del self.active_trades[trade_id]
                
                # Actualizar estadísticas
                self.daily_pnl += trade.pnl
                self.total_pnl += trade.pnl
                
                logger.info(f"Trade cerrado: {trade.symbol} P&L: {trade.pnl:.2f} "
                           f"({trade.pnl_pct:.2f}%), Razón: {reason}")
                
                # Enviar notificación
                self._send_trade_notification(trade, "CLOSE")
            
            else:
                logger.error(f"Error cerrando trade {trade_id}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error cerrando trade {trade_id}: {e}")
    
    def _close_all_positions(self):
        """Cerrar todas las posiciones"""
        logger.warning("Cerrando todas las posiciones por gestión de riesgo")
        for trade_id in list(self.active_trades.keys()):
            current_tick = self._get_current_tick(self.active_trades[trade_id].symbol)
            if current_tick:
                price = current_tick['bid'] if self.active_trades[trade_id].direction > 0 else current_tick['ask']
                self._close_trade(trade_id, price, 'Risk Management')
    
    def _close_some_positions(self):
        """Cerrar algunas posiciones para liberar margen"""
        # Cerrar las posiciones con peor P&L flotante
        floating_pnls = []
        for trade_id, trade in self.active_trades.items():
            current_tick = self._get_current_tick(trade.symbol)
            if current_tick:
                current_price = current_tick['bid'] if trade.direction > 0 else current_tick['ask']
                if trade.direction > 0:
                    floating_pnl = (current_price - trade.entry_price) * trade.volume
                else:
                    floating_pnl = (trade.entry_price - current_price) * trade.volume
                floating_pnls.append((trade_id, floating_pnl))
        
        # Ordenar por P&L flotante (de peor a mejor)
        floating_pnls.sort(key=lambda x: x[1])
        
        # Cerrar el 50% de las peores posiciones
        close_count = max(1, len(floating_pnls) // 2)
        for i in range(close_count):
            trade_id, _ = floating_pnls[i]
            current_tick = self._get_current_tick(self.active_trades[trade_id].symbol)
            if current_tick:
                price = current_tick['bid'] if self.active_trades[trade_id].direction > 0 else current_tick['ask']
                self._close_trade(trade_id, price, 'Margin Call')
    
    def _check_circuit_breakers(self):
        """Verificar y activar circuit breakers"""
        # Implementar lógica de circuit breakers basada en:
        # - Volatilidad extrema
        # - Pérdidas consecutivas
        # - Eventos de noticias
        # - Condiciones de mercado anormales
        
        # Ejemplo básico:
        recent_trades = [t for t in self.trade_history if t.entry_time > datetime.now() - timedelta(hours=1)]
        if len(recent_trades) >= 5:
            losing_trades = [t for t in recent_trades if t.pnl and t.pnl < 0]
            if len(losing_trades) >= 3:
                self.circuit_breaker = True
                logger.warning("Circuit breaker activado: 3 pérdidas consecutivas en 1 hora")
    
    def _get_current_tick(self, symbol: str) -> Optional[Dict]:
        """Obtener tick actual para un símbolo"""
        if symbol in self.data_cache and self.data_cache[symbol]:
            return self.data_cache[symbol][-1]
        return None
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Calcular volatilidad actual"""
        if symbol in self.data_cache and len(self.data_cache[symbol]) > 20:
            prices = [tick['last'] for tick in self.data_cache[symbol][-20:] if tick.get('last')]
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                return np.std(returns)
        return 0.0
    
    def _is_within_trading_hours(self, config: LiveTradingConfig, symbol: str) -> bool:
        """Verificar horario de trading"""
        if not config.trading_hours:
            return True
        
        symbol_hours = config.trading_hours.get(symbol)
        if not symbol_hours:
            return True
        
        now = datetime.now().time()
        start_time = datetime.strptime(symbol_hours[0], '%H:%M').time()
        end_time = datetime.strptime(symbol_hours[1], '%H:%M').time()
        
        return start_time <= now <= end_time
    
    def _is_high_impact_news_time(self) -> bool:
        """Verificar si es hora de noticias de alto impacto (implementación básica)"""
        # En una implementación real, integrar con API de calendario económico
        now = datetime.now()
        
        # Suponer noticias a las 8:00, 13:00, 15:00
        news_times = [8, 13, 15]
        current_hour = now.hour
        
        # Verificar si estamos en una hora de noticias ± 1 hora
        for news_hour in news_times:
            if abs(current_hour - news_hour) <= 1:
                return True
        
        return False
    
    def _has_high_correlation_trades(self, signal: TradeSignal) -> bool:
        """Verificar si ya hay trades altamente correlacionados"""
        # Implementación básica - en producción usar correlaciones reales
        open_symbols = [trade.symbol for trade in self.active_trades.values()]
        
        # Grupos correlacionados (ejemplo simplificado)
        correlated_groups = [
            ['EURUSD', 'GBPUSD', 'AUDUSD'],
            ['USDJPY', 'USDCHF', 'USDCAD'],
            ['XAUUSD', 'XAGUSD']
        ]
        
        for group in correlated_groups:
            if signal.symbol in group:
                # Contar cuántos símbolos del grupo están abiertos
                open_in_group = sum(1 for sym in open_symbols if sym in group)
                if open_in_group >= 2:  # Máximo 2 por grupo
                    return True
        
        return False
    
    def _has_open_position(self, symbol: str) -> bool:
        """Verificar si hay posición abierta para un símbolo"""
        return any(trade.symbol == symbol and trade.status == 'open' 
                  for trade in self.active_trades.values())
    
    def _reset_daily_stats(self):
        """Reiniciar estadísticas diarias"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
    
    def _update_statistics(self):
        """Actualizar estadísticas en tiempo real"""
        try:
            account_info = self.platform.get_account_summary()
            if account_info:
                equity = account_info.get('equity', 0)
                
                # Actualizar equity high
                if equity > self.equity_high:
                    self.equity_high = equity
                
                # Calcular drawdown actual
                if self.equity_high > 0:
                    self.current_drawdown = (self.equity_high - equity) / self.equity_high * 100
                else:
                    self.current_drawdown = 0.0
        
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def _send_trade_notification(self, trade: LiveTrade, action: str):
        """Enviar notificación de trade (implementación básica)"""
        message = f"{action} {trade.symbol} {trade.direction} @ {trade.entry_price}"
        if action == "CLOSE" and trade.pnl is not None:
            message += f" P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)"
        
        logger.info(f"NOTIFICATION: {message}")
        # En implementación real: enviar email, SMS, webhook, etc.
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Obtener estado actual del portafolio"""
        open_positions = len(self.active_trades)
        total_trades = len(self.trade_history)
        
        # Calcular métricas de performance
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