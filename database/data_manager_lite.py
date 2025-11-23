# database/data_manager_lite.py
"""
Data Manager LITE - Funciona sin PostgreSQL/Redis/InfluxDB
Usa SQLite y cache en memoria para desarrollo
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import pickle
import zlib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)
Base = declarative_base()

class TradingData(Base):
    __tablename__ = 'trading_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50))
    timeframe = Column(String(10))
    timestamp = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    spread = Column(Float)
    
    @classmethod
    def create_table(cls, engine):
        Base.metadata.create_all(engine)

class InMemoryCache:
    """Cache simple en memoria (reemplazo de Redis)"""
    
    def __init__(self):
        self.cache = {}
        self.expiry = {}
    
    def get(self, key):
        """Obtener valor del cache"""
        if key in self.cache:
            # Verificar expiración
            if key in self.expiry:
                if datetime.now() > self.expiry[key]:
                    del self.cache[key]
                    del self.expiry[key]
                    return None
            return self.cache[key]
        return None
    
    def setex(self, key, ttl, value):
        """Guardar con time-to-live"""
        self.cache[key] = value
        self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
    
    def set(self, key, value):
        """Guardar sin expiración"""
        self.cache[key] = value
    
    def ping(self):
        """Verificar que está funcionando"""
        return True
    
    def flushdb(self):
        """Limpiar cache"""
        self.cache.clear()
        self.expiry.clear()

class DataManager:
    """
    Gestor de datos LITE - sin dependencias externas
    Usa SQLite en lugar de PostgreSQL
    Usa cache en memoria en lugar de Redis
    """
    
    def __init__(self, config):
        self.config = config
        self.memory_cache = InMemoryCache()
        self.setup_databases()
    
    def setup_databases(self):
        """Configura bases de datos (solo SQLite para desarrollo)"""
        try:
            # Verificar si estamos en modo lite
            db_url = self.config.database.postgres_url
            
            # Si la URL contiene sqlite, usar SQLite
            if 'sqlite' in db_url.lower():
                logger.info("Using SQLite database (lite mode)")
                
                # Crear directorio de datos si no existe
                Path('data').mkdir(exist_ok=True)
                
                # SQLite
                self.pg_engine = create_engine(
                    db_url,
                    echo=False,
                    connect_args={'check_same_thread': False}  # Para SQLite
                )
                self.Session = sessionmaker(bind=self.pg_engine)
                
                # Crear tablas
                TradingData.create_table(self.pg_engine)
                
                # Cache en memoria (reemplazo de Redis)
                self.redis_client = self.memory_cache
                logger.info("Using in-memory cache (Redis disabled)")
                
                # InfluxDB deshabilitado
                self.influx_client = None
                self.write_api = None
                self.query_api = None
                logger.info("InfluxDB disabled (lite mode)")
                
            else:
                # Modo completo con PostgreSQL
                logger.info("Using PostgreSQL database (full mode)")
                self._setup_full_databases()
            
            logger.info("Databases setup completed")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            logger.warning("Falling back to lite mode...")
            self._setup_lite_fallback()
    
    def _setup_full_databases(self):
        """Setup completo con PostgreSQL/Redis/InfluxDB"""
        try:
            from influxdb_client import InfluxDBClient
            import redis
            
            # PostgreSQL
            self.pg_engine = create_engine(
                self.config.database.postgres_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={'connect_timeout': 10}
            )
            self.Session = sessionmaker(bind=self.pg_engine)
            
            # Redis
            self.redis_client = redis.Redis.from_url(
                self.config.database.redis_url,
                decode_responses=False
            )
            
            # InfluxDB
            self.influx_client = InfluxDBClient(
                url=self.config.database.influx_url,
                token=self.config.database.influx_token,
                org=self.config.database.influx_org
            )
            self.write_api = self.influx_client.write_api(write_option='SYNCHRONOUS')
            self.query_api = self.influx_client.query_api()
            
            # Crear tablas
            TradingData.create_table(self.pg_engine)
            
        except Exception as e:
            logger.error(f"Full database setup failed: {e}")
            raise
    
    def _setup_lite_fallback(self):
        """Fallback a modo lite si falla el setup completo"""
        try:
            # SQLite como fallback
            db_path = 'data/trading_fallback.db'
            Path('data').mkdir(exist_ok=True)
            
            self.pg_engine = create_engine(
                f'sqlite:///{db_path}',
                echo=False,
                connect_args={'check_same_thread': False}
            )
            self.Session = sessionmaker(bind=self.pg_engine)
            TradingData.create_table(self.pg_engine)
            
            # Cache en memoria
            self.redis_client = self.memory_cache
            
            # Sin InfluxDB
            self.influx_client = None
            self.write_api = None
            self.query_api = None
            
            logger.info("Fallback to lite mode successful")
            
        except Exception as e:
            logger.error(f"Lite fallback failed: {e}")
            raise
    
    def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena datos de mercado"""
        try:
            # PostgreSQL/SQLite
            self._store_in_postgres(symbol, timeframe, data)
            
            # InfluxDB (solo si está disponible)
            if self.influx_client and self.write_api:
                self._store_in_influx(symbol, timeframe, data)
            
            # Cache (Redis o memoria)
            self._store_in_redis(symbol, timeframe, data)
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    def _store_in_postgres(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en PostgreSQL o SQLite"""
        try:
            with self.Session() as session:
                for idx, row in data.iterrows():
                    record = TradingData(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=idx,
                        open=row.get('open'),
                        high=row.get('high'),
                        low=row.get('low'),
                        close=row.get('close'),
                        volume=row.get('volume', 0),
                        spread=row.get('spread', 0)
                    )
                    session.merge(record)
                session.commit()
            logger.debug(f"Stored {len(data)} rows for {symbol} {timeframe} in database")
        except Exception as e:
            logger.error(f"Error storing in database: {e}")
    
    def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en InfluxDB (solo si está disponible)"""
        try:
            from influxdb_client import Point
            
            points = []
            for idx, row in data.iterrows():
                point = Point("market_data") \
                    .tag("symbol", symbol) \
                    .tag("timeframe", timeframe) \
                    .field("open", float(row.get('open', 0))) \
                    .field("high", float(row.get('high', 0))) \
                    .field("low", float(row.get('low', 0))) \
                    .field("close", float(row.get('close', 0))) \
                    .field("volume", float(row.get('volume', 0))) \
                    .time(idx)
                points.append(point)
            
            self.write_api.write(bucket="trading", record=points)
            logger.debug(f"Stored {len(points)} points in InfluxDB")
        except Exception as e:
            logger.warning(f"Could not store in InfluxDB: {e}")
    
    def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en cache (Redis o memoria)"""
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            compressed_data = zlib.compress(pickle.dumps(data))
            self.redis_client.setex(cache_key, 3600, compressed_data)
            logger.debug(f"Cached data for {symbol} {timeframe}")
        except Exception as e:
            logger.warning(f"Could not cache data: {e}")
    
    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos cacheados"""
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(zlib.decompress(cached))
        except Exception as e:
            logger.debug(f"Cache miss or error: {e}")
        return None
    
    def store_strategy_result(self, strategy_name: str, result_data: Dict):
        """Almacena resultados de estrategias"""
        try:
            # Guardar en archivo JSON como fallback
            results_dir = Path('data/strategy_results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            
            logger.info(f"Strategy result saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing strategy result: {e}")
    
    def get_performance_history(self, strategy_name: str, days: int = 30) -> pd.DataFrame:
        """Obtiene historial de performance"""
        try:
            # Buscar archivos de resultados
            results_dir = Path('data/strategy_results')
            if not results_dir.exists():
                return pd.DataFrame()
            
            files = list(results_dir.glob(f"{strategy_name}_*.json"))
            
            if not files:
                return pd.DataFrame()
            
            # Leer archivos y crear DataFrame
            records = []
            for file in files[-30:]:  # Últimos 30 archivos
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        records.append(data)
                except:
                    continue
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()