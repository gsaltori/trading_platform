# database/data_manager.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import redis
import pickle
import zlib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import json

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

class DataManager:
    def __init__(self, config):
        self.config = config
        self.setup_databases()
    
    def setup_databases(self):
        """Configura todas las conexiones a bases de datos"""
        try:
            # PostgreSQL para datos estructurados
            self.pg_engine = create_engine(self.config.database.postgres_url)
            self.Session = sessionmaker(bind=self.pg_engine)
            
            # InfluxDB para series temporales
            self.influx_client = InfluxDBClient(
                url=self.config.database.influx_url,
                token=self.config.database.influx_token,
                org=self.config.database.influx_org
            )
            self.write_api = self.influx_client.write_api(write_option=SYNCHRONOUS)
            self.query_api = self.influx_client.query_api()
            
            # Redis para cache
            self.redis_client = redis.Redis.from_url(
                self.config.database.redis_url,
                decode_responses=False
            )
            
            # Crear tablas si no existen
            TradingData.create_table(self.pg_engine)
            
            logger.info("All databases connected successfully")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            raise
    
    def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena datos de mercado en múltiples bases de datos"""
        try:
            # PostgreSQL para consultas estructuradas
            self._store_in_postgres(symbol, timeframe, data)
            
            # InfluxDB para análisis temporal
            self._store_in_influx(symbol, timeframe, data)
            
            # Redis para cache
            self._store_in_redis(symbol, timeframe, data)
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    def _store_in_postgres(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en PostgreSQL"""
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
                session.merge(record)  # Upsert
            session.commit()
    
    def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en InfluxDB"""
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
    
    def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en Redis con compresión"""
        cache_key = f"market_data:{symbol}:{timeframe}"
        compressed_data = zlib.compress(pickle.dumps(data))
        self.redis_client.setex(cache_key, 3600, compressed_data)  # 1 hora
    
    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos cacheados de Redis"""
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(zlib.decompress(cached))
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
        return None
    
    def store_strategy_result(self, strategy_name: str, result_data: Dict):
        """Almacena resultados de estrategias"""
        try:
            # PostgreSQL
            with self.Session() as session:
                # Implementar lógica de almacenamiento de resultados
                pass
            
            # InfluxDB para métricas temporales
            point = Point("strategy_performance") \
                .tag("strategy", strategy_name) \
                .field("sharpe", result_data.get('sharpe', 0)) \
                .field("max_drawdown", result_data.get('max_drawdown', 0)) \
                .field("total_return", result_data.get('total_return', 0)) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket="trading", record=point)
            
        except Exception as e:
            logger.error(f"Error storing strategy result: {e}")
    
    def get_performance_history(self, strategy_name: str, days: int = 30) -> pd.DataFrame:
        """Obtiene historial de performance de una estrategia"""
        try:
            query = f'''
            from(bucket: "trading")
            |> range(start: -{days}d)
            |> filter(fn: (r) => r._measurement == "strategy_performance")
            |> filter(fn: (r) => r.strategy == "{strategy_name}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query_data_frame(query)
            if not result.empty:
                return result.set_index('_time')
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
        
        return pd.DataFrame()