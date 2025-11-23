# core/platform.py - VERSIÓN CORREGIDA Y MEJORADA
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from config.settings import ConfigManager
from data.mt5_connector import MT5ConnectionManager
from database.data_manager import DataManager

# Configurar logging con mejor formato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingPlatform:
    """
    Plataforma principal de trading algorítmico con mejoras de robustez
    """
    
    def __init__(self, config_path: str = "config/platform_config.yaml"):
        # Crear directorios necesarios
        self._create_directories()
        
        # Inicializar componentes core con manejo de errores
        try:
            self.config = ConfigManager(config_path)
            self.data_manager = DataManager(self.config)
            self.mt5_connector = MT5ConnectionManager(self.config)
        except Exception as e:
            logger.error(f"Error inicializando componentes: {e}")
            raise
        
        # Estado de la plataforma
        self.initialized = False
        self.strategies = {}
        self.active_trades = {}
        self._connection_health = True
        
        logger.info("Trading Platform instance created successfully")
    
    def _create_directories(self):
        """Crea la estructura de directorios necesaria"""
        directories = [
            'logs',
            'data/cache',
            'data/backtests',
            'data/backups',
            'strategies/generated',
            'strategies/optimized',
            'config',
            'reports',
            'reports/html',
            'reports/pdf',
            'ml/models',
            'ml/datasets'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
    
    def initialize(self) -> bool:
        """Inicializa la plataforma completa con verificación de salud"""
        try:
            logger.info("Initializing Trading Platform...")
            
            # Inicializar conexión MT5 con reintentos
            max_attempts = 3
            for attempt in range(max_attempts):
                if self.mt5_connector.initialize():
                    logger.info(f"MT5 connection established on attempt {attempt + 1}")
                    break
                elif attempt < max_attempts - 1:
                    logger.warning(f"MT5 connection failed, retrying... (attempt {attempt + 1}/{max_attempts})")
                    import time
                    time.sleep(2)
                else:
                    logger.error("Failed to initialize MT5 connection after all attempts")
                    return False
            
            # Verificar conexiones a bases de datos con timeout
            try:
                # Verificación PostgreSQL
                with self.data_manager.Session() as session:
                    result = session.execute("SELECT 1")
                    if result.scalar() != 1:
                        raise Exception("PostgreSQL health check failed")
                
                # Verificación Redis
                if not self.data_manager.redis_client.ping():
                    raise Exception("Redis health check failed")
                
                logger.info("Database connections verified")
            except Exception as e:
                logger.warning(f"Database verification warning: {e}")
                # No falla la inicialización, pero registra el warning
            
            # Verificar símbolos disponibles
            self._verify_symbols()
            
            self.initialized = True
            logger.info("Trading Platform initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Platform initialization failed: {e}", exc_info=True)
            return False
    
    def _verify_symbols(self):
        """Verifica y cachea símbolos disponibles"""
        try:
            import MetaTrader5 as mt5
            symbols = mt5.symbols_get()
            if symbols:
                logger.info(f"Found {len(symbols)} available symbols")
            else:
                logger.warning("No symbols found or symbol retrieval failed")
        except Exception as e:
            logger.error(f"Symbol verification error: {e}")
    
    def shutdown(self):
        """Apaga la plataforma de manera segura con limpieza completa"""
        try:
            logger.info("Shutting down Trading Platform...")
            
            # Cerrar posiciones abiertas si es necesario (opcional)
            if self.active_trades:
                logger.warning(f"Shutting down with {len(self.active_trades)} active trades")
            
            # Cerrar conexiones
            self.mt5_connector.shutdown()
            
            # Cerrar conexiones de base de datos
            try:
                self.data_manager.influx_client.close()
                logger.info("InfluxDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing InfluxDB: {e}")
            
            # Guardar estado si es necesario
            self._save_state()
            
            self.initialized = False
            logger.info("Trading Platform shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def _save_state(self):
        """Guarda el estado actual de la plataforma"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'active_trades': len(self.active_trades),
                'strategies': list(self.strategies.keys())
            }
            
            state_file = Path('data/platform_state.json')
            import json
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info("Platform state saved")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       days: int = 30) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de mercado con cache inteligente y validación
        """
        if not self.initialized:
            logger.error("Platform not initialized")
            return None
        
        try:
            # Validar parámetros
            if not symbol or not timeframe:
                logger.error("Invalid parameters: symbol and timeframe required")
                return None
            
            # Primero verificar cache con TTL
            cached_data = self.data_manager.get_cached_data(symbol, timeframe)
            if cached_data is not None and not cached_data.empty:
                # Verificar frescura de datos
                if len(cached_data) > 0:
                    last_time = cached_data.index[-1]
                    age = datetime.now() - last_time
                    if age < timedelta(minutes=5):  # Cache válido por 5 minutos
                        logger.debug(f"Using fresh cached data for {symbol} {timeframe}")
                        return cached_data
            
            # Obtener datos de MT5
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = self.mt5_connector.get_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            if data is not None and not data.empty:
                # Validar calidad de datos
                if self._validate_data_quality(data):
                    # Almacenar en cache
                    self.data_manager.store_market_data(symbol, timeframe, data)
                    logger.info(f"Retrieved and cached {len(data)} bars for {symbol} {timeframe}")
                    return data
                else:
                    logger.warning(f"Data quality check failed for {symbol} {timeframe}")
            else:
                logger.warning(f"No data retrieved for {symbol} {timeframe}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}", exc_info=True)
            return None
    
    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Valida la calidad de los datos obtenidos"""
        try:
            # Verificar columnas requeridas
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                logger.error("Missing required columns in data")
                return False
            
            # Verificar valores nulos
            if data[required_cols].isnull().any().any():
                logger.warning("Data contains null values")
                # Podemos intentar llenar valores nulos
                data.fillna(method='ffill', inplace=True)
            
            # Verificar lógica OHLC
            invalid_rows = (data['high'] < data['low']) | \
                          (data['close'] > data['high']) | \
                          (data['close'] < data['low']) | \
                          (data['open'] > data['high']) | \
                          (data['open'] < data['low'])
            
            if invalid_rows.any():
                logger.warning(f"Found {invalid_rows.sum()} invalid OHLC rows")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la cuenta con información extendida"""
        if not self.initialized:
            return {}
        
        try:
            account_info = self.mt5_connector.get_account_info()
            
            # Agregar métricas calculadas
            if account_info:
                balance = account_info.get('balance', 0)
                equity = account_info.get('equity', 0)
                margin = account_info.get('margin', 0)
                
                account_info['margin_level'] = (equity / margin * 100) if margin > 0 else 0
                account_info['drawdown'] = ((balance - equity) / balance * 100) if balance > 0 else 0
                account_info['risk_exposure'] = (margin / equity * 100) if equity > 0 else 0
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}
    
    def health_check(self) -> Dict[str, bool]:
        """Verifica la salud de todos los componentes"""
        health = {
            'platform_initialized': self.initialized,
            'mt5_connected': self.mt5_connector.connected,
            'database_connected': False,
            'cache_available': False
        }
        
        try:
            # Verificar base de datos
            with self.data_manager.Session() as session:
                result = session.execute("SELECT 1")
                health['database_connected'] = result.scalar() == 1
        except:
            pass
        
        try:
            # Verificar Redis
            health['cache_available'] = self.data_manager.redis_client.ping()
        except:
            pass
        
        return health
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        
        # Propagar excepciones si las hay
        if exc_type is not None:
            logger.error(f"Exception in context: {exc_type.__name__}: {exc_val}")
        
        return False  # No suprimir excepciones

# Singleton global para acceso fácil con thread-safety
_platform_instance = None
_platform_lock = __import__('threading').Lock()

def get_platform(config_path: str = "config/platform_config.yaml") -> TradingPlatform:
    """Obtiene la instancia singleton de la plataforma de forma thread-safe"""
    global _platform_instance
    
    if _platform_instance is None:
        with _platform_lock:
            # Double-check locking
            if _platform_instance is None:
                _platform_instance = TradingPlatform(config_path)
    
    return _platform_instance

def reset_platform():
    """Resetea la instancia singleton (útil para testing)"""
    global _platform_instance
    
    with _platform_lock:
        if _platform_instance is not None:
            _platform_instance.shutdown()
            _platform_instance = None