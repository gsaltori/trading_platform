# core/platform.py
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import ConfigManager
from data.mt5_connector import MT5ConnectionManager
from database.data_manager import DataManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingPlatform:
    """
    Plataforma principal de trading algorítmico
    """
    
    def __init__(self, config_path: str = "config/platform_config.yaml"):
        # Crear directorios necesarios
        self._create_directories()
        
        # Inicializar componentes core
        self.config = ConfigManager(config_path)
        self.data_manager = DataManager(self.config)
        self.mt5_connector = MT5ConnectionManager(self.config)
        
        # Estado de la plataforma
        self.initialized = False
        self.strategies = {}
        self.active_trades = {}
        
        logger.info("Trading Platform instance created")
    
    def _create_directories(self):
        """Crea la estructura de directorios necesaria"""
        directories = [
            'logs',
            'data/cache',
            'data/backtests',
            'strategies/generated',
            'config',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> bool:
        """Inicializa la plataforma completa"""
        try:
            logger.info("Initializing Trading Platform...")
            
            # Inicializar conexión MT5
            if not self.mt5_connector.initialize():
                logger.error("Failed to initialize MT5 connection")
                return False
            
            # Verificar conexiones a bases de datos
            # (Podemos agregar verificaciones específicas aquí)
            
            self.initialized = True
            logger.info("Trading Platform initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Platform initialization failed: {e}")
            return False
    
    def shutdown(self):
        """Apaga la plataforma de manera segura"""
        try:
            logger.info("Shutting down Trading Platform...")
            
            # Cerrar conexiones
            self.mt5_connector.shutdown()
            
            # Guardar estado si es necesario
            self.initialized = False
            
            logger.info("Trading Platform shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       days: int = 30) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de mercado con cache inteligente
        """
        if not self.initialized:
            logger.error("Platform not initialized")
            return None
        
        try:
            # Primero verificar cache
            cached_data = self.data_manager.get_cached_data(symbol, timeframe)
            if cached_data is not None:
                logger.debug(f"Using cached data for {symbol} {timeframe}")
                return cached_data
            
            # Obtener datos de MT5
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = self.mt5_connector.get_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            if data is not None:
                # Almacenar en cache
                self.data_manager.store_market_data(symbol, timeframe, data)
                logger.info(f"Retrieved {len(data)} bars for {symbol} {timeframe}")
            else:
                logger.warning(f"No data retrieved for {symbol} {timeframe}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la cuenta"""
        if not self.initialized:
            return {}
        
        try:
            account_info = self.mt5_connector.get_account_info()
            return account_info
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

# Singleton global para acceso fácil
_platform_instance = None

def get_platform(config_path: str = "config/platform_config.yaml") -> TradingPlatform:
    """Obtiene la instancia singleton de la plataforma"""
    global _platform_instance
    if _platform_instance is None:
        _platform_instance = TradingPlatform(config_path)
    return _platform_instance