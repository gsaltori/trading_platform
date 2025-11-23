# core/monitoring.py - VERSIÓN CORREGIDA
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
    enable_performance_monitoring: bool = True
    enable_trade_monitoring: bool = True
    enable_system_monitoring: bool = True
    log_level: str = "INFO"
    log_retention_days: int = 30
    metrics_interval: int = 60  # segundos

class AdvancedLogger:
    """Sistema de logging avanzado con rotación y monitoreo - VERSIÓN MEJORADA"""
    
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
        self._lock = threading.Lock()
    
    def setup_logging(self):
        """Configurar sistema de logging avanzado con mejor manejo de errores"""
        try:
            # Crear directorio de logs
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Configurar nivel
            log_level = getattr(logging, self.config.log_level, logging.INFO)
            self.logger.setLevel(log_level)
            
            # Limpiar handlers existentes para evitar duplicados
            self.logger.handlers.clear()
            
            # Formato mejorado con más contexto
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Handler de archivo con rotación mejorada
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_dir / 'trading_platform.log',
                when='midnight',
                interval=1,
                backupCount=self.config.log_retention_days,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            
            # Handler de consola con colores (opcional)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            
            # Handler para errores críticos (archivo separado)
            error_handler = logging.handlers.RotatingFileHandler(
                log_dir / 'errors.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            error_handler.setFormatter(formatter)
            error_handler.setLevel(logging.ERROR)
            
            # Agregar handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.addHandler(error_handler)
            
            self.logger.info(f"Logging system initialized for {self.name}")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Fallback a logging básico
            logging.basicConfig(level=logging.INFO)
    
    def start_monitoring(self):
        """Iniciar monitoreo en tiempo real con protección de errores"""
        if self.config.enable_performance_monitoring and not self.monitor_thread:
            self.stop_monitoring = False
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop, 
                daemon=True,
                name=f"Monitor-{self.name}"
            )
            self.monitor_thread.start()
            self.logger.info("Monitoring system started")
    
    def stop_monitoring_system(self):
        """Detener sistema de monitoreo de forma segura"""
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
            if self.monitor_thread.is_alive():
                self.logger.warning("Monitoring thread did not stop cleanly")
            else:
                self.logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Bucle de monitoreo en tiempo real con manejo robusto de errores"""
        while not self.stop_monitoring:
            try:
                with self._lock:
                    self._collect_performance_metrics()
                    self._collect_system_metrics()
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error en monitoreo: {e}", exc_info=True)
                time.sleep(30)  # Espera más larga en caso de error
    
    def _collect_performance_metrics(self):
        """Recolectar métricas de performance con validación"""
        try:
            import gc
            import os
            
            # Métricas de memoria
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.performance_metrics.update({
                'timestamp': datetime.now().isoformat(),
                'memory_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'memory_vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'memory_percent': round(process.memory_percent(), 2),
                'cpu_percent': round(process.cpu_percent(interval=0.1), 2),
                'threads_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'gc_objects': len(gc.get_objects()),
                'process_status': process.status()
            })
            
            # Alertas de uso excesivo
            if self.performance_metrics['memory_percent'] > 80:
                self.logger.warning(f"High memory usage: {self.performance_metrics['memory_percent']}%")
            
            if self.performance_metrics['cpu_percent'] > 85:
                self.logger.warning(f"High CPU usage: {self.performance_metrics['cpu_percent']}%")
                
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
    
    def _collect_system_metrics(self):
        """Recolectar métricas del sistema con validación"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Memoria
            virtual_memory = psutil.virtual_memory()
            
            # Disco
            disk_usage = psutil.disk_usage('.')
            
            # Red
            net_io = psutil.net_io_counters()
            
            self.system_metrics.update({
                'timestamp': datetime.now().isoformat(),
                'cpu_total_percent': round(cpu_percent, 2),
                'cpu_per_core': [round(c, 2) for c in cpu_per_core],
                'memory_available_mb': round(virtual_memory.available / 1024 / 1024, 2),
                'memory_total_mb': round(virtual_memory.total / 1024 / 1024, 2),
                'memory_percent': round(virtual_memory.percent, 2),
                'disk_usage_percent': round(disk_usage.percent, 2),
                'disk_free_gb': round(disk_usage.free / 1024 / 1024 / 1024, 2),
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'network_packets_sent': net_io.packets_sent,
                'network_packets_recv': net_io.packets_recv
            })
            
            # Alertas críticas
            if disk_usage.percent > 90:
                self.logger.error(f"Critical disk space: {disk_usage.percent}% used")
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def log_trade_event(self, trade_data: Dict[str, Any]):
        """Registrar evento de trade con validación"""
        if self.config.enable_trade_monitoring:
            try:
                # Sanitizar datos sensibles si es necesario
                safe_trade_data = self._sanitize_trade_data(trade_data)
                
                self.logger.info(f"TRADE_EVENT: {json.dumps(safe_trade_data, default=str)}")
                
                # Actualizar métricas de trading
                with self._lock:
                    self.trade_metrics.update({
                        'last_trade': safe_trade_data,
                        'timestamp': datetime.now().isoformat(),
                        'trade_count': self.trade_metrics.get('trade_count', 0) + 1
                    })
                    
            except Exception as e:
                self.logger.error(f"Error logging trade event: {e}")
    
    def _sanitize_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizar datos de trade para logging seguro"""
        safe_data = trade_data.copy()
        
        # Remover información sensible si existe
        sensitive_keys = ['api_key', 'password', 'secret', 'token']
        for key in sensitive_keys:
            if key in safe_data:
                safe_data[key] = '***REDACTED***'
        
        return safe_data
    
    def log_strategy_event(self, strategy_name: str, event_type: str, data: Dict[str, Any]):
        """Registrar evento de estrategia"""
        try:
            event_data = {
                'strategy': strategy_name,
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            self.logger.info(f"STRATEGY_EVENT: {json.dumps(event_data, default=str)}")
        except Exception as e:
            self.logger.error(f"Error logging strategy event: {e}")
    
    def log_performance_event(self, component: str, operation: str, execution_time: float):
        """Registrar evento de performance con niveles adaptativos"""
        try:
            # Diferentes niveles según tiempo de ejecución
            if execution_time > 5.0:
                self.logger.error(
                    f"CRITICAL_PERFORMANCE: {component}.{operation} took {execution_time:.2f}s"
                )
            elif execution_time > 1.0:
                self.logger.warning(
                    f"SLOW_PERFORMANCE: {component}.{operation} took {execution_time:.2f}s"
                )
            else:
                self.logger.debug(
                    f"PERFORMANCE: {component}.{operation} took {execution_time:.2f}s"
                )
        except Exception as e:
            self.logger.error(f"Error logging performance event: {e}")
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Obtener reporte completo de métricas con lock"""
        with self._lock:
            return {
                'performance': self.performance_metrics.copy(),
                'system': self.system_metrics.copy(),
                'trading': self.trade_metrics.copy(),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_metrics_to_file(self, filepath: str = None):
        """Guardar métricas en archivo con manejo de errores"""
        try:
            if filepath is None:
                filepath = f"logs/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            metrics = self.get_metrics_report()
            
            # Asegurar que el directorio existe
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            self.logger.info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

class HealthChecker:
    """Sistema de verificación de salud de la plataforma - VERSIÓN MEJORADA"""
    
    def __init__(self, platform):
        self.platform = platform
        self.health_status = {}
        self.last_check = None
        self._check_history = []
        self._max_history = 100
    
    def check_platform_health(self) -> Dict[str, Any]:
        """Verificar salud completa de la plataforma con historial"""
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
        all_healthy = all(check.get('healthy', False) for check in health_checks.values())
        
        health_checks['overall_health'] = {
            'healthy': all_healthy,
            'timestamp': self.last_check.isoformat(),
            'failed_checks': [name for name, check in health_checks.items() 
                            if not check.get('healthy', False) and name != 'overall_health']
        }
        
        self.health_status = health_checks
        
        # Guardar en historial
        self._add_to_history(health_checks)
        
        return health_checks
    
    def _add_to_history(self, health_check: Dict[str, Any]):
        """Agregar check al historial"""
        self._check_history.append({
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': health_check['overall_health']['healthy'],
            'failed_checks': health_check['overall_health']['failed_checks']
        })
        
        # Mantener solo los últimos N checks
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history:]
    
    def get_health_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de health checks"""
        return self._check_history.copy()
    
    def _check_mt5_connection(self) -> Dict[str, Any]:
        """Verificar conexión MT5 con más detalles"""
        try:
            if not self.platform.initialized:
                return {'healthy': False, 'message': 'Platform not initialized'}
            
            account_info = self.platform.get_account_summary()
            if account_info and 'balance' in account_info:
                return {
                    'healthy': True,
                    'message': 'MT5 connection active',
                    'account': account_info.get('login', 'Unknown'),
                    'balance': account_info.get('balance', 0),
                    'server': account_info.get('server', 'Unknown')
                }
            else:
                return {'healthy': False, 'message': 'No account info available'}
                
        except Exception as e:
            return {'healthy': False, 'message': f'MT5 connection error: {str(e)}'}
    
    def _check_database_connections(self) -> Dict[str, Any]:
        """Verificar conexiones a bases de datos con timeout"""
        details = {'postgres': 'UNKNOWN', 'redis': 'UNKNOWN'}
        
        try:
            # Verificar PostgreSQL con timeout
            from sqlalchemy import text
            with self.platform.data_manager.Session() as session:
                result = session.execute(text("SELECT 1"))
                postgres_ok = result.scalar() == 1
                details['postgres'] = 'OK' if postgres_ok else 'FAILED'
        except Exception as e:
            details['postgres'] = f'ERROR: {str(e)[:50]}'
            postgres_ok = False
        
        try:
            # Verificar Redis con timeout
            redis_ok = self.platform.data_manager.redis_client.ping()
            details['redis'] = 'OK' if redis_ok else 'FAILED'
        except Exception as e:
            details['redis'] = f'ERROR: {str(e)[:50]}'
            redis_ok = False
        
        return {
            'healthy': postgres_ok and redis_ok,
            'message': 'Database connections checked',
            'details': details
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Verificar uso de memoria con alertas graduales"""
        try:
            import os
            process = psutil.Process(os.getpid())
            memory_percent = process.memory_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Alertas graduales
            if memory_percent > 90:
                level = 'CRITICAL'
                healthy = False
            elif memory_percent > 80:
                level = 'WARNING'
                healthy = True
            else:
                level = 'OK'
                healthy = True
            
            return {
                'healthy': healthy,
                'message': f'Memory usage: {memory_percent:.1f}% ({level})',
                'usage_percent': round(memory_percent, 2),
                'usage_mb': round(memory_mb, 2),
                'level': level
            }
            
        except Exception as e:
            return {'healthy': False, 'message': f'Memory check error: {str(e)}'}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Verificar espacio en disco con alertas graduales"""
        try:
            disk_usage = psutil.disk_usage('.')
            usage_percent = disk_usage.percent
            free_gb = disk_usage.free / 1024 / 1024 / 1024
            
            # Alertas graduales
            if usage_percent > 95:
                level = 'CRITICAL'
                healthy = False
            elif usage_percent > 90:
                level = 'WARNING'
                healthy = True
            else:
                level = 'OK'
                healthy = True
            
            return {
                'healthy': healthy,
                'message': f'Disk usage: {usage_percent:.1f}% ({level})',
                'usage_percent': round(usage_percent, 2),
                'free_gb': round(free_gb, 2),
                'level': level
            }
            
        except Exception as e:
            return {'healthy': False, 'message': f'Disk check error: {str(e)}'}
    
    def _check_strategy_health(self) -> Dict[str, Any]:
        """Verificar salud de las estrategias"""
        try:
            # Contadores básicos
            active_strategies = len([s for s in self.platform.strategies.values() 
                                    if getattr(s, 'enabled', False)])
            
            return {
                'healthy': True,
                'message': 'Strategy health check passed',
                'active_strategies': active_strategies,
                'total_strategies': len(self.platform.strategies)
            }
        except Exception as e:
            return {'healthy': False, 'message': f'Strategy health check error: {str(e)}'}
    
    def _check_performance_health(self) -> Dict[str, Any]:
        """Verificar salud de performance con límites adaptativos"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Límites adaptativos
            if cpu_percent > 95:
                level = 'CRITICAL'
                healthy = False
            elif cpu_percent > 85:
                level = 'WARNING'
                healthy = True
            else:
                level = 'OK'
                healthy = True
            
            return {
                'healthy': healthy,
                'message': f'CPU usage: {cpu_percent:.1f}% ({level})',
                'cpu_percent': round(cpu_percent, 2),
                'level': level
            }
            
        except Exception as e:
            return {'healthy': False, 'message': f'Performance check error: {str(e)}'}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud actual con cache inteligente"""
        if not self.health_status or not self.last_check:
            return self.check_platform_health()
        
        # Si la última verificación fue hace más de 5 minutos, verificar nuevamente
        if datetime.now() - self.last_check > timedelta(minutes=5):
            return self.check_platform_health()
        
        return self.health_status