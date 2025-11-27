# core/platform.py
"""
Main Trading Platform module with improved error handling and lite mode support.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

# Configurar logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'trading_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class TradingPlatform:
    """
    Plataforma principal de trading algorítmico.
    
    Soporta modo "lite" para desarrollo sin dependencias pesadas
    (PostgreSQL, Redis, InfluxDB).
    """
    
    def __init__(self, config_path: str = "config/platform_config.yaml", lite_mode: bool = None):
        """
        Inicializa la plataforma de trading.
        
        Args:
            config_path: Ruta al archivo de configuración
            lite_mode: Si True, usa SQLite y omite Redis/InfluxDB.
                      Si None, detecta automáticamente.
        """
        # Crear directorios necesarios
        self._create_directories()
        
        # Detectar modo lite automáticamente si no se especifica
        if lite_mode is None:
            lite_mode = self._should_use_lite_mode()
        
        self.lite_mode = lite_mode
        
        if lite_mode:
            logger.info("Running in LITE MODE (SQLite, no Redis/InfluxDB)")
        
        # Inicializar componentes core
        try:
            from config.settings import ConfigManager
            self.config = ConfigManager(config_path)
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            self.config = self._create_default_config()
        
        # Inicializar data manager con soporte lite mode
        try:
            from database.data_manager import DataManager
            self.data_manager = DataManager(self.config, lite_mode=lite_mode)
        except Exception as e:
            logger.warning(f"DataManager initialization failed, using fallback: {e}")
            self.data_manager = self._create_fallback_data_manager()
        
        # Inicializar MT5 connector
        try:
            from data.mt5_connector import MT5ConnectionManager
            self.mt5_connector = MT5ConnectionManager(self.config)
        except Exception as e:
            logger.warning(f"MT5 connector initialization failed: {e}")
            self.mt5_connector = None
        
        # Estado de la plataforma
        self.initialized = False
        self.strategies = {}
        self.active_trades = {}
        
        logger.info("Trading Platform instance created")
    
    def _should_use_lite_mode(self) -> bool:
        """Detecta si debe usar modo lite basado en disponibilidad de servicios."""
        # Verificar si PostgreSQL está disponible
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database="trading",
                user="trading_user",
                password="trading_password",
                connect_timeout=2
            )
            conn.close()
            return False  # PostgreSQL disponible, no necesita lite mode
        except Exception:
            pass
        
        # Verificar variable de entorno
        if os.environ.get('TRADING_LITE_MODE', '').lower() in ('true', '1', 'yes'):
            return True
        
        # Por defecto, usar lite mode en desarrollo
        return True
    
    def _create_directories(self):
        """Crea la estructura de directorios necesaria."""
        directories = [
            'logs',
            'data/cache',
            'data/backtests',
            'data/sqlite',
            'strategies/generated',
            'config',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _create_default_config(self):
        """Crea configuración por defecto cuando no hay archivo de config."""
        from dataclasses import dataclass
        
        @dataclass
        class DefaultDatabaseConfig:
            postgres_url: str = "sqlite:///data/sqlite/trading.db"
            influx_url: str = ""
            influx_token: str = ""
            influx_org: str = ""
            redis_url: str = ""
        
        @dataclass
        class DefaultMT5Config:
            path: str = "C:/Program Files/MetaTrader 5/terminal64.exe"
            server: str = ""
            login: int = 0
            password: str = ""
            timeout: int = 60000
            portable: bool = False
        
        @dataclass
        class DefaultConfig:
            database: DefaultDatabaseConfig = None
            mt5: DefaultMT5Config = None
            
            def __post_init__(self):
                if self.database is None:
                    self.database = DefaultDatabaseConfig()
                if self.mt5 is None:
                    self.mt5 = DefaultMT5Config()
        
        return DefaultConfig()
    
    def _create_fallback_data_manager(self):
        """Crea un data manager de fallback básico."""
        class FallbackDataManager:
            def __init__(self):
                self._cache = {}
            
            def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
                key = f"{symbol}_{timeframe}"
                self._cache[key] = data
            
            def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
                key = f"{symbol}_{timeframe}"
                return self._cache.get(key)
        
        return FallbackDataManager()
    
    def initialize(self) -> bool:
        """Inicializa la plataforma completa."""
        try:
            logger.info("Initializing Trading Platform...")
            
            # Inicializar conexión MT5 si está disponible
            if self.mt5_connector:
                if not self.mt5_connector.initialize():
                    logger.warning("MT5 connection not available - running in offline mode")
                else:
                    logger.info("MT5 connected successfully")
            else:
                logger.warning("MT5 connector not available - running in offline mode")
            
            self.initialized = True
            logger.info("Trading Platform initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Platform initialization failed: {e}")
            # Aún así marcamos como inicializado para permitir operaciones offline
            self.initialized = True
            return True
    
    def shutdown(self):
        """Apaga la plataforma de manera segura."""
        try:
            logger.info("Shutting down Trading Platform...")
            
            # Cerrar conexiones
            if self.mt5_connector:
                self.mt5_connector.shutdown()
            
            # Guardar estado si es necesario
            self.initialized = False
            
            logger.info("Trading Platform shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       days: int = 30) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de mercado con cache inteligente.
        
        Primero intenta desde cache, luego MT5, luego yfinance como fallback.
        """
        try:
            # Primero verificar cache
            cached_data = self.data_manager.get_cached_data(symbol, timeframe)
            if cached_data is not None and len(cached_data) > 0:
                logger.debug(f"Using cached data for {symbol} {timeframe}")
                return cached_data
            
            # Intentar obtener de MT5
            if self.mt5_connector and self.mt5_connector.connected:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                data = self.mt5_connector.get_historical_data(
                    symbol, timeframe, start_date, end_date
                )
                
                if data is not None and len(data) > 0:
                    # Almacenar en cache
                    self.data_manager.store_market_data(symbol, timeframe, data)
                    logger.info(f"Retrieved {len(data)} bars for {symbol} {timeframe} from MT5")
                    return data
            
            # Fallback a yfinance para datos de prueba
            data = self._get_yfinance_data(symbol, timeframe, days)
            if data is not None and len(data) > 0:
                self.data_manager.store_market_data(symbol, timeframe, data)
                logger.info(f"Retrieved {len(data)} bars for {symbol} {timeframe} from yfinance")
                return data
            
            logger.warning(f"No data retrieved for {symbol} {timeframe}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def _get_yfinance_data(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Obtiene datos de yfinance como fallback."""
        try:
            import yfinance as yf
            
            # Mapear símbolos de Forex a formato yfinance
            yf_symbol_map = {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'USDCAD': 'USDCAD=X',
                'AUDUSD': 'AUDUSD=X',
                'XAUUSD': 'GC=F',  # Gold futures
            }
            
            yf_symbol = yf_symbol_map.get(symbol, symbol)
            
            # Mapear timeframe
            tf_map = {
                'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
                'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1wk'
            }
            interval = tf_map.get(timeframe, '1h')
            
            # Ajustar período según timeframe (yfinance tiene límites)
            if interval in ['1m', '5m', '15m', '30m']:
                period = '7d'  # Máximo para intervalos pequeños
            elif interval in ['1h', '4h']:
                period = f'{min(days, 730)}d'
            else:
                period = f'{days}d'
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data is not None and len(data) > 0:
                # Normalizar nombres de columnas
                data.columns = [c.lower() for c in data.columns]
                if 'adj close' in data.columns:
                    data = data.drop(columns=['adj close'])
                return data
            
            return None
            
        except ImportError:
            logger.warning("yfinance not available for fallback data")
            return None
        except Exception as e:
            logger.warning(f"yfinance fallback failed: {e}")
            return None
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la cuenta."""
        if self.mt5_connector and self.mt5_connector.connected:
            try:
                account_info = self.mt5_connector.get_account_info()
                return account_info
            except Exception as e:
                logger.error(f"Error getting account summary: {e}")
        
        # Retornar datos de demo si MT5 no está disponible
        return {
            'login': 'DEMO',
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0,
            'leverage': 100,
            'currency': 'USD',
            'server': 'Offline'
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Singleton global para acceso fácil
_platform_instance = None


def get_platform(config_path: str = "config/platform_config.yaml", 
                lite_mode: bool = None) -> TradingPlatform:
    """Obtiene la instancia singleton de la plataforma."""
    global _platform_instance
    if _platform_instance is None:
        _platform_instance = TradingPlatform(config_path, lite_mode=lite_mode)
    return _platform_instance


def reset_platform():
    """Resetea la instancia singleton (útil para testing)."""
    global _platform_instance
    if _platform_instance is not None:
        _platform_instance.shutdown()
    _platform_instance = None
