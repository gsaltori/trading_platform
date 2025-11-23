# risk_management/risk_engine.py
"""
Motor de Gestión de Riesgo Avanzado
Maneja cálculo de posiciones, límites de riesgo y gestión de capital
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    """Parámetros de gestión de riesgo"""
    max_position_size: float = 0.1  # 10% del capital por posición
    max_drawdown: float = 0.15  # 15% drawdown máximo
    daily_loss_limit: float = 0.05  # 5% pérdida diaria máxima
    max_correlation: float = 0.7  # Correlación máxima entre posiciones
    risk_per_trade: float = 0.02  # 2% de riesgo por trade
    use_kelly_criterion: bool = False
    use_volatility_sizing: bool = True
    max_leverage: float = 10.0
    max_open_positions: int = 5
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0

class RiskEngine:
    """
    Motor de gestión de riesgo avanzado
    
    Funcionalidades:
    - Cálculo de tamaño de posición
    - Verificación de límites de riesgo
    - Gestión de correlaciones
    - Position sizing adaptativo
    - Kelly Criterion
    - Volatility-based sizing
    """
    
    def __init__(self, parameters: RiskParameters = None):
        self.parameters = parameters or RiskParameters()
        
        # Métricas de riesgo en tiempo real
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.total_exposure = 0.0
        
        # Historial
        self.risk_history = []
        self.correlation_matrix = None
        
        logger.info("RiskEngine initialized with parameters: %s", self.parameters)
    
    def calculate_position_size(self, signal, risk_per_trade: float, 
                               account_info: Dict) -> float:
        """
        Calcular tamaño de posición óptimo basado en riesgo
        
        Args:
            signal: Señal de trading con información (TradeSignal object)
            risk_per_trade: Riesgo por trade como fracción (0.02 = 2%)
            account_info: Diccionario con información de cuenta
                         {'balance': float, 'equity': float, ...}
        
        Returns:
            float: Tamaño de posición en lotes (ej: 0.1, 0.5, 1.0)
        """
        try:
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', balance)
            
            if balance <= 0 or equity <= 0:
                logger.warning("Invalid balance or equity: %s", account_info)
                return 0.0
            
            # Usar equity para cálculo si está disponible (más conservador)
            base_capital = min(balance, equity)
            
            # Calcular monto de riesgo en dinero
            risk_amount = base_capital * risk_per_trade
            
            # Obtener información del signal
            entry_price = getattr(signal, 'price', 0)
            stop_loss = getattr(signal, 'stop_loss', None)
            
            if entry_price <= 0:
                logger.error("Invalid entry price: %s", entry_price)
                return 0.0
            
            # Calcular distancia de stop loss
            if stop_loss and stop_loss > 0:
                stop_distance = abs(entry_price - stop_loss)
            else:
                # Usar ATR si no hay stop loss definido
                metadata = getattr(signal, 'metadata', {})
                atr = metadata.get('atr', entry_price * 0.01)  # Default 1% del precio
                stop_distance = atr * self.parameters.stop_loss_atr_multiplier
            
            if stop_distance <= 0:
                logger.error("Invalid stop distance: %s", stop_distance)
                return 0.0
            
            # Calcular posición base: Riesgo / Stop Distance
            position_size = risk_amount / stop_distance
            
            # Ajustar por volatilidad si está habilitado
            if self.parameters.use_volatility_sizing:
                volatility = getattr(signal, 'metadata', {}).get('volatility', 0.01)
                # Reducir tamaño en alta volatilidad
                vol_adjustment = 1.0 / (1.0 + volatility * 10)
                position_size *= vol_adjustment
                logger.debug("Volatility adjustment: %.2f", vol_adjustment)
            
            # Aplicar límite máximo de posición
            max_position = (base_capital * self.parameters.max_position_size) / entry_price
            position_size = min(position_size, max_position)
            
            # Redondear a 2 decimales (estándar de forex)
            position_size = round(position_size, 2)
            
            # Mínimo 0.01 lotes (micro lote)
            if position_size < 0.01:
                position_size = 0.01
            
            # Verificar leverage
            position_value = position_size * entry_price
            leverage_used = position_value / base_capital
            
            if leverage_used > self.parameters.max_leverage:
                logger.warning("Leverage limit exceeded: %.2f > %.2f", 
                             leverage_used, self.parameters.max_leverage)
                position_size = (base_capital * self.parameters.max_leverage) / entry_price
                position_size = round(position_size, 2)
            
            logger.debug("Position size calculated: %.2f lots for %s at %.5f", 
                        position_size, signal.symbol, entry_price)
            
            return position_size
            
        except Exception as e:
            logger.error("Error calculating position size: %s", e, exc_info=True)
            return 0.0
    
    def check_trade_risk(self, signal, config, portfolio_status: Dict) -> bool:
        """
        Verificar si un trade cumple con todos los criterios de riesgo
        
        Args:
            signal: Señal de trading
            config: Configuración de trading (LiveTradingConfig)
            portfolio_status: Estado actual del portafolio
        
        Returns:
            bool: True si el trade pasa todas las verificaciones de riesgo
        """
        try:
            checks = []
            
            # 1. Verificar número de posiciones abiertas
            open_positions = portfolio_status.get('open_positions', 0)
            max_positions = getattr(config, 'max_positions', self.parameters.max_open_positions)
            
            if open_positions >= max_positions:
                logger.warning("Max positions limit reached: %d/%d", open_positions, max_positions)
                return False
            checks.append("positions_ok")
            
            # 2. Verificar límite de pérdida diaria
            daily_pnl = portfolio_status.get('daily_pnl', 0)
            daily_limit = getattr(config, 'daily_loss_limit', self.parameters.daily_loss_limit)
            
            # Asumir capital base de 10000 si no está disponible
            base_capital = portfolio_status.get('balance', 10000)
            daily_loss_amount = -abs(base_capital * daily_limit)
            
            if daily_pnl < daily_loss_amount:
                logger.warning("Daily loss limit reached: %.2f < %.2f", daily_pnl, daily_loss_amount)
                return False
            checks.append("daily_loss_ok")
            
            # 3. Verificar drawdown máximo
            current_drawdown = portfolio_status.get('current_drawdown', 0)
            max_dd = getattr(config, 'max_drawdown', self.parameters.max_drawdown) * 100
            
            if current_drawdown > max_dd:
                logger.warning("Max drawdown exceeded: %.2f%% > %.2f%%", current_drawdown, max_dd)
                return False
            checks.append("drawdown_ok")
            
            # 4. Verificar circuit breaker
            if portfolio_status.get('circuit_breaker', False):
                logger.warning("Circuit breaker is active")
                return False
            checks.append("circuit_breaker_ok")
            
            # 5. Verificar exposición total
            total_exposure = portfolio_status.get('total_exposure', 0)
            if total_exposure > base_capital * 0.5:  # Máximo 50% de exposición
                logger.warning("Total exposure too high: %.2f > %.2f", 
                             total_exposure, base_capital * 0.5)
                return False
            checks.append("exposure_ok")
            
            # 6. Verificar correlación con posiciones existentes (simplificado)
            # En producción, implementar verificación real de correlaciones
            checks.append("correlation_ok")
            
            logger.debug("Risk checks passed: %s", ", ".join(checks))
            return True
            
        except Exception as e:
            logger.error("Error checking trade risk: %s", e, exc_info=True)
            # En caso de error, ser conservador y rechazar
            return False
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, 
                                 avg_loss: float) -> float:
        """
        Calcular fracción óptima de Kelly
        
        Formula de Kelly: f* = (p*b - q) / b
        donde:
            p = probabilidad de ganar (win_rate)
            q = probabilidad de perder (1 - win_rate)
            b = ratio de ganancia/pérdida (avg_win / avg_loss)
        
        Args:
            win_rate: Tasa de ganancia (0-1)
            avg_win: Ganancia promedio en términos absolutos
            avg_loss: Pérdida promedio en términos absolutos (positivo)
        
        Returns:
            float: Fracción de Kelly (0-1), aplicando factor de seguridad
        """
        try:
            # Validaciones
            if not (0 < win_rate < 1):
                logger.warning("Invalid win_rate: %.2f", win_rate)
                return 0.0
            
            if avg_loss <= 0 or avg_win <= 0:
                logger.warning("Invalid avg_win/avg_loss: %.2f/%.2f", avg_win, avg_loss)
                return 0.0
            
            # Calcular componentes
            p = win_rate
            q = 1 - win_rate
            b = avg_win / avg_loss
            
            # Kelly Criterion
            kelly = (p * b - q) / b
            
            # Aplicar factor de seguridad (Half-Kelly o Quarter-Kelly)
            # Usar solo 25-50% del Kelly para reducir riesgo
            safety_factor = 0.5  # Half-Kelly
            kelly_safe = kelly * safety_factor
            
            # Limitar entre 0 y 0.25 (máximo 25% del capital)
            kelly_safe = max(0.0, min(kelly_safe, 0.25))
            
            logger.debug("Kelly Criterion: raw=%.4f, safe=%.4f (win_rate=%.2f, b=%.2f)", 
                        kelly, kelly_safe, win_rate, b)
            
            return kelly_safe
            
        except Exception as e:
            logger.error("Error calculating Kelly criterion: %s", e)
            return 0.0
    
    def update_risk_metrics(self, current_equity: float, daily_pnl: float, 
                           open_positions: List = None):
        """
        Actualizar métricas de riesgo en tiempo real
        
        Args:
            current_equity: Equity actual de la cuenta
            daily_pnl: P&L del día actual
            open_positions: Lista de posiciones abiertas (opcional)
        """
        try:
            self.daily_pnl = daily_pnl
            
            # Actualizar peak equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                logger.debug("New equity peak: %.2f", self.peak_equity)
            
            # Calcular drawdown actual
            if self.peak_equity > 0:
                self.current_drawdown = ((self.peak_equity - current_equity) / 
                                        self.peak_equity * 100)
            else:
                self.current_drawdown = 0.0
            
            # Calcular exposición total
            if open_positions:
                self.total_exposure = sum(
                    getattr(pos, 'volume', 0) * getattr(pos, 'entry_price', 0)
                    for pos in open_positions
                )
            
            # Guardar en historial
            self.risk_history.append({
                'timestamp': datetime.now(),
                'equity': current_equity,
                'daily_pnl': daily_pnl,
                'drawdown': self.current_drawdown,
                'exposure': self.total_exposure
            })
            
            # Mantener solo últimas 1000 entradas
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            # Alertas
            if self.current_drawdown > self.parameters.max_drawdown * 80:  # 80% del límite
                logger.warning("Approaching max drawdown: %.2f%% (limit: %.2f%%)", 
                             self.current_drawdown, 
                             self.parameters.max_drawdown * 100)
            
            if abs(daily_pnl) > current_equity * self.parameters.daily_loss_limit * 0.8:
                logger.warning("Approaching daily loss limit: %.2f", daily_pnl)
                
        except Exception as e:
            logger.error("Error updating risk metrics: %s", e)
    
    def calculate_stop_loss(self, entry_price: float, direction: int, 
                          atr: float) -> float:
        """
        Calcular stop loss basado en ATR
        
        Args:
            entry_price: Precio de entrada
            direction: Dirección del trade (1=long, -1=short)
            atr: Average True Range
        
        Returns:
            float: Precio de stop loss
        """
        try:
            multiplier = self.parameters.stop_loss_atr_multiplier
            
            if direction > 0:  # Long
                stop_loss = entry_price - (atr * multiplier)
            else:  # Short
                stop_loss = entry_price + (atr * multiplier)
            
            return round(stop_loss, 5)  # Redondear a 5 decimales para forex
            
        except Exception as e:
            logger.error("Error calculating stop loss: %s", e)
            return entry_price * 0.98 if direction > 0 else entry_price * 1.02
    
    def calculate_take_profit(self, entry_price: float, direction: int, 
                            atr: float) -> float:
        """
        Calcular take profit basado en ATR
        
        Args:
            entry_price: Precio de entrada
            direction: Dirección del trade (1=long, -1=short)
            atr: Average True Range
        
        Returns:
            float: Precio de take profit
        """
        try:
            multiplier = self.parameters.take_profit_atr_multiplier
            
            if direction > 0:  # Long
                take_profit = entry_price + (atr * multiplier)
            else:  # Short
                take_profit = entry_price - (atr * multiplier)
            
            return round(take_profit, 5)  # Redondear a 5 decimales para forex
            
        except Exception as e:
            logger.error("Error calculating take profit: %s", e)
            return entry_price * 1.03 if direction > 0 else entry_price * 0.97
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Obtener reporte completo de riesgo
        
        Returns:
            Dict con métricas de riesgo actuales
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'daily_pnl': round(self.daily_pnl, 2),
            'current_drawdown': round(self.current_drawdown, 2),
            'peak_equity': round(self.peak_equity, 2),
            'total_exposure': round(self.total_exposure, 2),
            'parameters': {
                'max_position_size': self.parameters.max_position_size,
                'max_drawdown': self.parameters.max_drawdown,
                'daily_loss_limit': self.parameters.daily_loss_limit,
                'risk_per_trade': self.parameters.risk_per_trade,
                'max_leverage': self.parameters.max_leverage,
                'use_kelly': self.parameters.use_kelly_criterion,
                'use_volatility_sizing': self.parameters.use_volatility_sizing
            },
            'history_entries': len(self.risk_history)
        }
    
    def reset_daily_metrics(self):
        """Resetear métricas diarias (llamar al inicio de cada día)"""
        self.daily_pnl = 0.0
        logger.info("Daily risk metrics reset")
    
    def get_risk_score(self) -> float:
        """
        Calcular score de riesgo general (0-100)
        100 = riesgo máximo, 0 = sin riesgo
        
        Returns:
            float: Score de riesgo
        """
        try:
            score = 0.0
            
            # Factor de drawdown (0-40 puntos)
            dd_factor = min(self.current_drawdown / (self.parameters.max_drawdown * 100), 1.0)
            score += dd_factor * 40
            
            # Factor de pérdida diaria (0-30 puntos)
            if self.peak_equity > 0:
                daily_loss_factor = abs(min(self.daily_pnl, 0)) / (self.peak_equity * self.parameters.daily_loss_limit)
                daily_loss_factor = min(daily_loss_factor, 1.0)
                score += daily_loss_factor * 30
            
            # Factor de exposición (0-30 puntos)
            if self.peak_equity > 0:
                exposure_factor = self.total_exposure / (self.peak_equity * 0.5)  # Relativo a 50% max
                exposure_factor = min(exposure_factor, 1.0)
                score += exposure_factor * 30
            
            return round(score, 2)
            
        except Exception as e:
            logger.error("Error calculating risk score: %s", e)
            return 50.0  # Score medio en caso de error

# Funciones de utilidad adicionales

def calculate_position_correlation(positions: List) -> np.ndarray:
    """
    Calcular matriz de correlación entre posiciones
    
    Args:
        positions: Lista de posiciones abiertas
    
    Returns:
        numpy.ndarray: Matriz de correlación
    """
    # Implementación simplificada
    # En producción, usar datos históricos reales
    n = len(positions)
    if n == 0:
        return np.array([])
    
    # Placeholder - retornar matriz identidad
    return np.eye(n)

def calculate_portfolio_var(positions: List, confidence: float = 0.95) -> float:
    """
    Calcular Value at Risk (VaR) del portafolio
    
    Args:
        positions: Lista de posiciones
        confidence: Nivel de confianza (0.95 = 95%)
    
    Returns:
        float: VaR en términos monetarios
    """
    # Implementación simplificada
    # En producción, usar distribución histórica o paramétrica
    if not positions:
        return 0.0
    
    total_value = sum(
        getattr(pos, 'volume', 0) * getattr(pos, 'entry_price', 0)
        for pos in positions
    )
    
    # VaR simplificado: 2% del valor total
    return total_value * 0.02

# Exportar clases y funciones principales
__all__ = [
    'RiskEngine',
    'RiskParameters',
    'calculate_position_correlation',
    'calculate_portfolio_var'
]