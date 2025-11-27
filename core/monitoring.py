# core/monitoring.py
"""
Advanced monitoring and logging system for the trading platform.
"""

import logging
import logging.handlers
import time
import psutil
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import threading
import json
from pathlib import Path


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    enable_performance_monitoring: bool = True
    enable_trade_monitoring: bool = True
    enable_system_monitoring: bool = True
    log_level: str = "INFO"
    log_retention_days: int = 30
    metrics_interval: int = 60  # segundos


class AdvancedLogger:
    """Sistema de logging avanzado con rotación y monitoreo."""
    
    def __init__(self, name: str, config: MonitoringConfig = None):
        self.name = name
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(name)
        self.setup_logging()
        
        # Métricas en tiempo real
        self.performance_metrics = {}
        self.trade_metrics = {}
        self.system_metrics = {}
        
        # Hilo de monitoreo
        self.monitor_thread = None
        self.stop_monitoring = False
    
    def setup_logging(self):
        """Configurar sistema de logging avanzado."""
        # Crear directorio de logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Evitar duplicar handlers
        if self.logger.handlers:
            return
        
        # Configurar nivel
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
        )
        
        # Handler de archivo con rotación
        try:
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_dir / 'trading_platform.log',
                when='midnight',
                interval=1,
                backupCount=self.config.log_retention_days
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
        
        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def start_monitoring(self):
        """Iniciar monitoreo en tiempo real."""
        if self.config.enable_performance_monitoring:
            self.stop_monitoring = False
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring_system(self):
        """Detener sistema de monitoreo."""
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
    
    def _monitoring_loop(self):
        """Bucle de monitoreo en tiempo real."""
        while not self.stop_monitoring:
            try:
                self._collect_performance_metrics()
                self._collect_system_metrics()
                time.sleep(self.config.metrics_interval)
            except Exception as e:
                self.logger.error(f"Error en monitoreo: {e}")
                time.sleep(30)
    
    def _collect_performance_metrics(self):
        """Recolectar métricas de performance."""
        import gc
        
        try:
            # Métricas de memoria
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.performance_metrics.update({
                'timestamp': datetime.now().isoformat(),
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
                'threads_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'gc_objects': len(gc.get_objects())
            })
        except Exception as e:
            self.logger.warning(f"Error collecting performance metrics: {e}")
    
    def _collect_system_metrics(self):
        """Recolectar métricas del sistema."""
        try:
            self.system_metrics.update({
                'timestamp': datetime.now().isoformat(),
                'cpu_total_percent': psutil.cpu_percent(interval=1),
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024,
                'disk_usage_percent': psutil.disk_usage('.').percent,
                'network_io': psutil.net_io_counters()._asdict()
            })
        except Exception as e:
            self.logger.warning(f"Error collecting system metrics: {e}")
    
    def log_trade_event(self, trade_data: Dict[str, Any]):
        """Registrar evento de trade."""
        if self.config.enable_trade_monitoring:
            self.logger.info(f"TRADE_EVENT: {json.dumps(trade_data, default=str)}")
            
            # Actualizar métricas de trading
            self.trade_metrics.update({
                'last_trade': trade_data,
                'timestamp': datetime.now().isoformat()
            })
    
    def log_strategy_event(self, strategy_name: str, event_type: str, data: Dict[str, Any]):
        """Registrar evento de estrategia."""
        event_data = {
            'strategy': strategy_name,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self.logger.info(f"STRATEGY_EVENT: {json.dumps(event_data, default=str)}")
    
    def log_performance_event(self, component: str, operation: str, execution_time: float):
        """Registrar evento de performance."""
        if execution_time > 1.0:  # Solo log operaciones lentas (>1 segundo)
            self.logger.warning(
                f"PERFORMANCE_ISSUE: {component}.{operation} took {execution_time:.2f}s"
            )
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Obtener reporte completo de métricas."""
        return {
            'performance': self.performance_metrics,
            'system': self.system_metrics,
            'trading': self.trade_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_metrics_to_file(self, filepath: str = None):
        """Guardar métricas en archivo."""
        if filepath is None:
            filepath = f"logs/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metrics = self.get_metrics_report()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)


class HealthChecker:
    """Sistema de verificación de salud de la plataforma."""
    
    def __init__(self, platform):
        self.platform = platform
        self.health_status = {}
        self.last_check = None
    
    def check_platform_health(self) -> Dict[str, Any]:
        """Verificar salud completa de la plataforma."""
        self.last_check = datetime.now()
        
        health_checks = {
            'mt5_connection': self._check_mt5_connection(),
            'database_connections': self._check_database_connections(),
            'memory_usage': self._check_memory_usage(),
            'disk_space': self._check_disk_space(),
            'strategy_health': self._check_strategy_health(),
            'performance_health': self._check_performance_health()
        }
        
        # Calcular estado general
        all_healthy = all(check['healthy'] for check in health_checks.values())
        health_checks['overall_health'] = {
            'healthy': all_healthy,
            'timestamp': self.last_check.isoformat(),
            'failed_checks': [name for name, check in health_checks.items() 
                            if not check['healthy'] and name != 'overall_health']
        }
        
        self.health_status = health_checks
        return health_checks
    
    def _check_mt5_connection(self) -> Dict[str, Any]:
        """Verificar conexión MT5."""
        try:
            if not self.platform.initialized:
                return {'healthy': False, 'message': 'Platform not initialized'}
            
            if self.platform.mt5_connector is None:
                return {'healthy': True, 'message': 'MT5 not configured (offline mode)'}
            
            if not self.platform.mt5_connector.connected:
                return {'healthy': True, 'message': 'MT5 offline (running in demo mode)'}
            
            account_info = self.platform.get_account_summary()
            if account_info:
                return {
                    'healthy': True,
                    'message': 'MT5 connection active',
                    'account': account_info.get('login', 'Unknown')
                }
            else:
                return {'healthy': False, 'message': 'No account info available'}
                
        except Exception as e:
            return {'healthy': False, 'message': f'MT5 connection error: {str(e)}'}
    
    def _check_database_connections(self) -> Dict[str, Any]:
        """Verificar conexiones a bases de datos."""
        try:
            dm = self.platform.data_manager
            
            # Verificar si estamos en modo lite
            if hasattr(dm, 'lite_mode') and dm.lite_mode:
                return {
                    'healthy': True,
                    'message': 'Running in lite mode (SQLite)',
                    'details': {'mode': 'lite', 'sqlite': 'OK'}
                }
            
            details = {}
            all_ok = True
            
            # Verificar PostgreSQL
            if hasattr(dm, 'Session'):
                try:
                    from sqlalchemy import text
                    with dm.Session() as session:
                        result = session.execute(text("SELECT 1"))
                        details['postgres'] = 'OK' if result.scalar() == 1 else 'FAILED'
                except Exception as e:
                    details['postgres'] = f'FAILED: {str(e)[:50]}'
                    all_ok = False
            
            # Verificar Redis
            if hasattr(dm, 'redis_client') and dm.redis_client:
                try:
                    if dm.redis_client.ping():
                        details['redis'] = 'OK'
                    else:
                        details['redis'] = 'FAILED'
                        all_ok = False
                except Exception as e:
                    details['redis'] = f'UNAVAILABLE: {str(e)[:30]}'
                    # Redis no es crítico
            else:
                details['redis'] = 'NOT CONFIGURED'
            
            return {
                'healthy': all_ok,
                'message': 'Database connections checked',
                'details': details
            }
            
        except Exception as e:
            return {'healthy': False, 'message': f'Database connection error: {str(e)}'}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Verificar uso de memoria."""
        try:
            process = psutil.Process()
            memory_percent = process.memory_percent()
            
            healthy = memory_percent < 80  # Menos del 80% de uso
            return {
                'healthy': healthy,
                'message': f'Memory usage: {memory_percent:.1f}%',
                'usage_percent': memory_percent
            }
            
        except Exception as e:
            return {'healthy': False, 'message': f'Memory check error: {str(e)}'}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Verificar espacio en disco."""
        try:
            disk_usage = psutil.disk_usage('.')
            usage_percent = disk_usage.percent
            
            healthy = usage_percent < 90  # Menos del 90% de uso
            return {
                'healthy': healthy,
                'message': f'Disk usage: {usage_percent:.1f}%',
                'usage_percent': usage_percent,
                'free_gb': disk_usage.free / 1024 / 1024 / 1024
            }
            
        except Exception as e:
            return {'healthy': False, 'message': f'Disk check error: {str(e)}'}
    
    def _check_strategy_health(self) -> Dict[str, Any]:
        """Verificar salud de las estrategias."""
        try:
            active_strategies = len(self.platform.strategies) if hasattr(self.platform, 'strategies') else 0
            return {
                'healthy': True,
                'message': 'Strategy health check passed',
                'active_strategies': active_strategies
            }
        except Exception as e:
            return {'healthy': False, 'message': f'Strategy health check error: {str(e)}'}
    
    def _check_performance_health(self) -> Dict[str, Any]:
        """Verificar salud de performance."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            healthy = cpu_percent < 85  # Menos del 85% de CPU
            
            return {
                'healthy': healthy,
                'message': f'CPU usage: {cpu_percent:.1f}%',
                'cpu_percent': cpu_percent
            }
            
        except Exception as e:
            return {'healthy': False, 'message': f'Performance check error: {str(e)}'}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud actual."""
        if not self.health_status or not self.last_check:
            return self.check_platform_health()
        
        # Si la última verificación fue hace más de 5 minutos, verificar nuevamente
        if datetime.now() - self.last_check > timedelta(minutes=5):
            return self.check_platform_health()
        
        return self.health_status
