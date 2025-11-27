# database/data_manager.py
"""
Data management module with support for multiple databases and lite mode.

Supports:
- PostgreSQL for production
- SQLite for development (lite mode)
- InfluxDB for time series (optional)
- Redis for caching (optional)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import json
import pickle
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)
Base = declarative_base()


class TradingData(Base):
    """Model for storing trading data."""
    __tablename__ = 'trading_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), index=True)
    timeframe = Column(String(10), index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    spread = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )
    
    @classmethod
    def create_table(cls, engine):
        Base.metadata.create_all(engine)


class DataManager:
    """
    Manages data storage and retrieval across multiple backends.
    
    Supports lite mode for development without heavy dependencies.
    """
    
    def __init__(self, config, lite_mode: bool = False):
        self.config = config
        self.lite_mode = lite_mode
        
        # Initialize storage backends
        self._cache = {}  # In-memory cache
        self.pg_engine = None
        self.Session = None
        self.influx_client = None
        self.redis_client = None
        
        # Service availability flags
        self.postgres_available = False
        self.influx_available = False
        self.redis_available = False
        
        self.setup_databases()
    
    def setup_databases(self):
        """Configure all database connections with graceful fallbacks."""
        if self.lite_mode:
            self._setup_lite_mode()
        else:
            self._setup_full_mode()
    
    def _setup_lite_mode(self):
        """Setup SQLite-only mode for development."""
        try:
            # Create data directory
            sqlite_dir = Path('data/sqlite')
            sqlite_dir.mkdir(parents=True, exist_ok=True)
            
            # SQLite database
            sqlite_path = sqlite_dir / 'trading.db'
            self.pg_engine = create_engine(f'sqlite:///{sqlite_path}')
            self.Session = sessionmaker(bind=self.pg_engine)
            
            # Create tables
            TradingData.create_table(self.pg_engine)
            
            self.postgres_available = True
            logger.info(f"Lite mode: SQLite database initialized at {sqlite_path}")
            
        except Exception as e:
            logger.error(f"SQLite setup error: {e}")
            self.postgres_available = False
    
    def _setup_full_mode(self):
        """Setup full mode with all databases."""
        # PostgreSQL
        self._setup_postgres()
        
        # InfluxDB (optional)
        self._setup_influx()
        
        # Redis (optional)
        self._setup_redis()
    
    def _setup_postgres(self):
        """Setup PostgreSQL connection."""
        try:
            postgres_url = getattr(self.config.database, 'postgres_url', '')
            
            if not postgres_url or 'sqlite' in postgres_url.lower():
                # Fall back to SQLite
                self._setup_lite_mode()
                return
            
            self.pg_engine = create_engine(
                postgres_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600
            )
            self.Session = sessionmaker(bind=self.pg_engine)
            
            # Test connection
            with self.pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Create tables
            TradingData.create_table(self.pg_engine)
            
            self.postgres_available = True
            logger.info("PostgreSQL connected successfully")
            
        except Exception as e:
            logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
            self.lite_mode = True
            self._setup_lite_mode()
    
    def _setup_influx(self):
        """Setup InfluxDB connection (optional)."""
        try:
            influx_url = getattr(self.config.database, 'influx_url', '')
            influx_token = getattr(self.config.database, 'influx_token', '')
            influx_org = getattr(self.config.database, 'influx_org', '')
            
            if not influx_url or not influx_token:
                logger.info("InfluxDB not configured - skipping")
                return
            
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            
            self.influx_client = InfluxDBClient(
                url=influx_url,
                token=influx_token,
                org=influx_org
            )
            
            # Test connection
            self.influx_client.ping()
            
            self.write_api = self.influx_client.write_api(write_option=SYNCHRONOUS)
            self.query_api = self.influx_client.query_api()
            
            self.influx_available = True
            logger.info("InfluxDB connected successfully")
            
        except ImportError:
            logger.info("InfluxDB client not installed - skipping")
        except Exception as e:
            logger.warning(f"InfluxDB not available: {e}")
    
    def _setup_redis(self):
        """Setup Redis connection (optional)."""
        try:
            redis_url = getattr(self.config.database, 'redis_url', '')
            
            if not redis_url:
                logger.info("Redis not configured - using in-memory cache")
                return
            
            import redis
            
            self.redis_client = redis.Redis.from_url(
                redis_url,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            
            self.redis_available = True
            logger.info("Redis connected successfully")
            
        except ImportError:
            logger.info("Redis client not installed - using in-memory cache")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
    
    def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store market data in available backends."""
        try:
            # Always store in cache
            self._store_in_cache(symbol, timeframe, data)
            
            # Store in PostgreSQL/SQLite
            if self.postgres_available:
                self._store_in_postgres(symbol, timeframe, data)
            
            # Store in InfluxDB if available
            if self.influx_available:
                self._store_in_influx(symbol, timeframe, data)
            
            # Store in Redis if available
            if self.redis_available:
                self._store_in_redis(symbol, timeframe, data)
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    def _store_in_cache(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store in in-memory cache."""
        cache_key = f"{symbol}_{timeframe}"
        self._cache[cache_key] = data.copy()
    
    def _store_in_postgres(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store in PostgreSQL/SQLite."""
        try:
            with self.Session() as session:
                for idx, row in data.iterrows():
                    # Check if record exists
                    existing = session.query(TradingData).filter(
                        TradingData.symbol == symbol,
                        TradingData.timeframe == timeframe,
                        TradingData.timestamp == idx
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.open = float(row.get('open', 0))
                        existing.high = float(row.get('high', 0))
                        existing.low = float(row.get('low', 0))
                        existing.close = float(row.get('close', 0))
                        existing.volume = float(row.get('volume', 0))
                        existing.spread = float(row.get('spread', 0)) if 'spread' in row else None
                    else:
                        # Insert new record
                        record = TradingData(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                            open=float(row.get('open', 0)),
                            high=float(row.get('high', 0)),
                            low=float(row.get('low', 0)),
                            close=float(row.get('close', 0)),
                            volume=float(row.get('volume', 0)),
                            spread=float(row.get('spread', 0)) if 'spread' in row else None
                        )
                        session.add(record)
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing in PostgreSQL/SQLite: {e}")
    
    def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store in InfluxDB."""
        if not self.influx_available:
            return
        
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
            
        except Exception as e:
            logger.warning(f"Error storing in InfluxDB: {e}")
    
    def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store in Redis with compression."""
        if not self.redis_available:
            return
        
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            compressed_data = zlib.compress(pickle.dumps(data))
            self.redis_client.setex(cache_key, 3600, compressed_data)  # 1 hour TTL
        except Exception as e:
            logger.warning(f"Error storing in Redis: {e}")
    
    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from the fastest available source."""
        cache_key = f"{symbol}_{timeframe}"
        
        # Try in-memory cache first
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Try Redis
        if self.redis_available:
            try:
                redis_key = f"market_data:{symbol}:{timeframe}"
                cached = self.redis_client.get(redis_key)
                if cached:
                    data = pickle.loads(zlib.decompress(cached))
                    self._cache[cache_key] = data  # Store in memory cache
                    return data.copy()
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Try PostgreSQL/SQLite
        if self.postgres_available:
            return self._get_from_postgres(symbol, timeframe)
        
        return None
    
    def _get_from_postgres(self, symbol: str, timeframe: str, 
                          days: int = 30) -> Optional[pd.DataFrame]:
        """Get data from PostgreSQL/SQLite."""
        try:
            with self.Session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                records = session.query(TradingData).filter(
                    TradingData.symbol == symbol,
                    TradingData.timeframe == timeframe,
                    TradingData.timestamp >= cutoff_date
                ).order_by(TradingData.timestamp).all()
                
                if not records:
                    return None
                
                data = pd.DataFrame([
                    {
                        'open': r.open,
                        'high': r.high,
                        'low': r.low,
                        'close': r.close,
                        'volume': r.volume,
                        'spread': r.spread
                    }
                    for r in records
                ], index=[r.timestamp for r in records])
                
                # Store in cache
                cache_key = f"{symbol}_{timeframe}"
                self._cache[cache_key] = data
                
                return data
                
        except Exception as e:
            logger.error(f"Error getting from PostgreSQL/SQLite: {e}")
            return None
    
    def store_strategy_result(self, strategy_name: str, result_data: Dict):
        """Store strategy results."""
        try:
            # Store in InfluxDB if available
            if self.influx_available:
                from influxdb_client import Point
                
                point = Point("strategy_performance") \
                    .tag("strategy", strategy_name) \
                    .field("sharpe", float(result_data.get('sharpe', 0))) \
                    .field("max_drawdown", float(result_data.get('max_drawdown', 0))) \
                    .field("total_return", float(result_data.get('total_return', 0))) \
                    .time(datetime.utcnow())
                
                self.write_api.write(bucket="trading", record=point)
            
            # Also store in cache
            cache_key = f"strategy_result:{strategy_name}"
            self._cache[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'data': result_data
            }
            
        except Exception as e:
            logger.error(f"Error storing strategy result: {e}")
    
    def get_performance_history(self, strategy_name: str, days: int = 30) -> pd.DataFrame:
        """Get strategy performance history."""
        try:
            if self.influx_available:
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
            logger.warning(f"Error getting performance history: {e}")
        
        return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all caches."""
        self._cache.clear()
        
        if self.redis_available:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get data manager status."""
        return {
            'lite_mode': self.lite_mode,
            'postgres_available': self.postgres_available,
            'influx_available': self.influx_available,
            'redis_available': self.redis_available,
            'cache_size': len(self._cache)
        }
