# data/free_data_sources.py
"""
Free data sources for historical market data.

Supports multiple providers: Yahoo Finance, Alpha Vantage, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
import os
import json
import hashlib

logger = logging.getLogger(__name__)

# Try importing data providers
YFINANCE_AVAILABLE = False
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.info("yfinance not installed. Install with: pip install yfinance")

PANDAS_DATAREADER_AVAILABLE = False
try:
    import pandas_datareader as pdr
    PANDAS_DATAREADER_AVAILABLE = True
except ImportError:
    logger.info("pandas_datareader not installed. Install with: pip install pandas-datareader")

# Try importing parquet support
PARQUET_AVAILABLE = False
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    logger.info("pyarrow not installed. Using CSV for cache. Install with: pip install pyarrow")


class DataCache:
    """Local cache for historical data."""
    
    def __init__(self, cache_dir: str = None):
        """Initialize cache."""
        if cache_dir is None:
            # Default cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".trading_platform_cache")
        
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self._ensure_cache_dir()
        self.metadata = self._load_metadata()
    
    def _ensure_cache_dir(self):
        """Create cache directory if not exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
        return {"entries": {}}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Error saving cache metadata: {e}")
    
    def _get_cache_key(self, symbol: str, timeframe: str, source: str) -> str:
        """Generate cache key."""
        key_string = f"{symbol}_{timeframe}_{source}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        ext = ".parquet" if PARQUET_AVAILABLE else ".csv"
        return os.path.join(self.cache_dir, f"{cache_key}{ext}")
    
    def get(self, symbol: str, timeframe: str, source: str) -> Optional[pd.DataFrame]:
        """Get data from cache."""
        cache_key = self._get_cache_key(symbol, timeframe, source)
        
        # Try both formats
        parquet_path = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        csv_path = os.path.join(self.cache_dir, f"{cache_key}.csv")
        
        cache_path = None
        if os.path.exists(parquet_path):
            cache_path = parquet_path
        elif os.path.exists(csv_path):
            cache_path = csv_path
        
        if cache_path:
            try:
                if cache_path.endswith('.parquet') and PARQUET_AVAILABLE:
                    df = pd.read_parquet(cache_path)
                else:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                
                entry = self.metadata["entries"].get(cache_key, {})
                
                logger.info(f"Cache hit for {symbol} {timeframe} from {source}")
                logger.info(f"  Cached: {entry.get('cached_at', 'unknown')}, Records: {len(df)}")
                
                return df
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}: {e}")
        
        return None
    
    def put(self, symbol: str, timeframe: str, source: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_key = self._get_cache_key(symbol, timeframe, source)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save data
            if PARQUET_AVAILABLE and cache_path.endswith('.parquet'):
                data.to_parquet(cache_path)
            else:
                # Fallback to CSV
                cache_path = cache_path.replace('.parquet', '.csv')
                data.to_csv(cache_path)
            
            # Update metadata
            self.metadata["entries"][cache_key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": source,
                "records": len(data),
                "start_date": str(data.index[0]) if len(data) > 0 else None,
                "end_date": str(data.index[-1]) if len(data) > 0 else None,
                "cached_at": datetime.now().isoformat(),
                "file_path": cache_path
            }
            self._save_metadata()
            
            logger.info(f"Cached {len(data)} records for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")
    
    def invalidate(self, symbol: str = None, timeframe: str = None, source: str = None):
        """Invalidate cache entries."""
        keys_to_remove = []
        
        for key, entry in self.metadata["entries"].items():
            match = True
            if symbol and entry.get("symbol") != symbol:
                match = False
            if timeframe and entry.get("timeframe") != timeframe:
                match = False
            if source and entry.get("source") != source:
                match = False
            
            if match:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            # Try both formats
            for ext in ['.parquet', '.csv']:
                cache_path = os.path.join(self.cache_dir, f"{key}{ext}")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            if key in self.metadata["entries"]:
                del self.metadata["entries"][key]
        
        self._save_metadata()
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def clear_all(self):
        """Clear all cache."""
        for key in list(self.metadata["entries"].keys()):
            for ext in ['.parquet', '.csv']:
                cache_path = os.path.join(self.cache_dir, f"{key}{ext}")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        self.metadata["entries"] = {}
        self._save_metadata()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_size = 0
        for key, entry in self.metadata["entries"].items():
            for ext in ['.parquet', '.csv']:
                cache_path = os.path.join(self.cache_dir, f"{key}{ext}")
                if os.path.exists(cache_path):
                    total_size += os.path.getsize(cache_path)
                    break
        
        return {
            "entries": len(self.metadata["entries"]),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": self.cache_dir
        }
    
    def list_cached_data(self) -> List[Dict]:
        """List all cached data entries."""
        return list(self.metadata["entries"].values())


class FreeDataProvider:
    """Base class for free data providers."""
    
    name = "Base"
    
    # Symbol mapping for different providers
    SYMBOL_MAP = {}
    
    # Timeframe mapping
    TIMEFRAME_MAP = {}
    
    @classmethod
    def get_data(cls, symbol: str, timeframe: str, 
                 start_date: datetime, end_date: datetime = None,
                 **kwargs) -> Optional[pd.DataFrame]:
        """Get historical data. Override in subclasses."""
        raise NotImplementedError
    
    @classmethod
    def map_symbol(cls, symbol: str) -> str:
        """Map trading symbol to provider symbol."""
        return cls.SYMBOL_MAP.get(symbol, symbol)
    
    @classmethod
    def get_available_symbols(cls) -> List[str]:
        """Get list of available symbols."""
        return []


class YahooFinanceProvider(FreeDataProvider):
    """Yahoo Finance data provider."""
    
    name = "Yahoo Finance"
    
    # Map MT5 symbols to Yahoo Finance symbols
    SYMBOL_MAP = {
        # Forex (Yahoo uses different format)
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
        "NZDUSD": "NZDUSD=X",
        "EURGBP": "EURGBP=X",
        "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X",
        "AUDJPY": "AUDJPY=X",
        "EURAUD": "EURAUD=X",
        "GBPAUD": "GBPAUD=X",
        "NZDJPY": "NZDJPY=X",
        # Metals
        "XAUUSD": "GC=F",  # Gold Futures
        "XAGUSD": "SI=F",  # Silver Futures
        # Indices
        "US30": "^DJI",    # Dow Jones
        "US500": "^GSPC",  # S&P 500
        "USTEC": "^IXIC",  # NASDAQ
        "DE40": "^GDAXI",  # DAX
        "UK100": "^FTSE",  # FTSE 100
        "JP225": "^N225",  # Nikkei 225
        # Crypto
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "LTCUSD": "LTC-USD",
        "XRPUSD": "XRP-USD",
        # Energy
        "XTIUSD": "CL=F",  # Crude Oil WTI
        "XBRUSD": "BZ=F",  # Brent Crude
        "XNGUSD": "NG=F",  # Natural Gas
    }
    
    # Yahoo Finance interval mapping
    TIMEFRAME_MAP = {
        "M1": "1m",
        "M5": "5m",
        "M15": "15m",
        "M30": "30m",
        "H1": "1h",
        "H4": "1h",  # Yahoo doesn't have 4h, will resample
        "D1": "1d",
        "W1": "1wk",
        "MN1": "1mo"
    }
    
    @classmethod
    def get_data(cls, symbol: str, timeframe: str,
                 start_date: datetime, end_date: datetime = None,
                 **kwargs) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not installed")
            return None
        
        try:
            # Map symbol
            yf_symbol = cls.map_symbol(symbol)
            interval = cls.TIMEFRAME_MAP.get(timeframe, "1d")
            
            # Yahoo Finance limitations for intraday data
            if interval in ["1m", "5m", "15m", "30m"]:
                # Max 7 days for minute data
                if (end_date or datetime.now()) - start_date > timedelta(days=7):
                    logger.warning(f"Yahoo Finance limits {interval} data to 7 days. Adjusting...")
                    start_date = (end_date or datetime.now()) - timedelta(days=7)
            elif interval == "1h":
                # Max 730 days for hourly data
                if (end_date or datetime.now()) - start_date > timedelta(days=730):
                    logger.warning(f"Yahoo Finance limits hourly data to 730 days. Adjusting...")
                    start_date = (end_date or datetime.now()) - timedelta(days=730)
            
            logger.info(f"Downloading {yf_symbol} {interval} from Yahoo Finance...")
            
            # Download data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start_date,
                end=end_date or datetime.now(),
                interval=interval
            )
            
            if df is None or df.empty:
                logger.warning(f"No data received for {symbol} from Yahoo Finance")
                return None
            
            # Standardize columns
            df = cls._standardize_dataframe(df)
            
            # Resample if needed (e.g., H4 from H1)
            if timeframe == "H4":
                df = cls._resample_to_4h(df)
            
            logger.info(f"Downloaded {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from Yahoo Finance: {e}")
            return None
    
    @classmethod
    def _standardize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        # Yahoo returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        df = df.copy()
        
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Keep only OHLCV
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols]
        
        # Ensure volume exists
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        return df
    
    @classmethod
    def _resample_to_4h(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-hour data to 4-hour."""
        return df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    @classmethod
    def get_available_symbols(cls) -> List[str]:
        """Get list of available symbols."""
        return list(cls.SYMBOL_MAP.keys())


class AlphaVantageProvider(FreeDataProvider):
    """Alpha Vantage data provider (requires free API key)."""
    
    name = "Alpha Vantage"
    
    SYMBOL_MAP = {
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD",
        "USDJPY": "USD/JPY",
        "BTCUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
    }
    
    TIMEFRAME_MAP = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "60min",
        "D1": "daily",
        "W1": "weekly",
        "MN1": "monthly"
    }
    
    @classmethod
    def get_data(cls, symbol: str, timeframe: str,
                 start_date: datetime, end_date: datetime = None,
                 api_key: str = None, **kwargs) -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage."""
        if not api_key:
            logger.warning("Alpha Vantage requires an API key. Get one free at alphavantage.co")
            return None
        
        # Implementation would go here
        # Requires requests library and API key
        logger.info("Alpha Vantage provider not fully implemented yet")
        return None


class TwelveDataProvider(FreeDataProvider):
    """Twelve Data provider (has free tier)."""
    
    name = "Twelve Data"
    
    @classmethod
    def get_data(cls, symbol: str, timeframe: str,
                 start_date: datetime, end_date: datetime = None,
                 api_key: str = None, **kwargs) -> Optional[pd.DataFrame]:
        """Get historical data from Twelve Data."""
        logger.info("Twelve Data provider not implemented yet")
        return None


class FreeDataManager:
    """
    Manager for free data sources with caching.
    
    Usage:
        manager = FreeDataManager()
        data = manager.get_data("EURUSD", "H1", start_date, source="yahoo")
    """
    
    # Available providers
    PROVIDERS = {
        "yahoo": YahooFinanceProvider,
        "alphavantage": AlphaVantageProvider,
        "twelvedata": TwelveDataProvider,
    }
    
    # Provider display names
    PROVIDER_NAMES = {
        "yahoo": "Yahoo Finance (Free)",
        "alphavantage": "Alpha Vantage (API Key)",
        "twelvedata": "Twelve Data (API Key)",
        "mt5": "MetaTrader 5"
    }
    
    def __init__(self, cache_dir: str = None, use_cache: bool = True):
        """Initialize data manager."""
        self.cache = DataCache(cache_dir) if use_cache else None
        self.use_cache = use_cache
        self.api_keys = {}
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider."""
        self.api_keys[provider] = api_key
    
    def get_available_providers(self) -> List[Dict[str, str]]:
        """Get list of available providers."""
        providers = []
        
        # Always include MT5 first
        providers.append({
            "id": "mt5",
            "name": "MetaTrader 5",
            "available": True,
            "requires_key": False
        })
        
        # Yahoo Finance
        providers.append({
            "id": "yahoo",
            "name": "Yahoo Finance (Free)",
            "available": YFINANCE_AVAILABLE,
            "requires_key": False
        })
        
        # Alpha Vantage
        providers.append({
            "id": "alphavantage",
            "name": "Alpha Vantage (Free API Key)",
            "available": True,
            "requires_key": True
        })
        
        return providers
    
    def get_data(self, symbol: str, timeframe: str,
                 start_date: datetime, end_date: datetime = None,
                 source: str = "yahoo",
                 force_download: bool = False,
                 progress_callback=None) -> Optional[pd.DataFrame]:
        """
        Get historical data from specified source.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, H1, D1, etc.)
            start_date: Start date
            end_date: End date (default: now)
            source: Data source ("yahoo", "alphavantage", etc.)
            force_download: Bypass cache
            progress_callback: Callback function(progress, message)
        
        Returns:
            DataFrame with OHLCV data
        """
        if progress_callback:
            progress_callback(10, f"Checking cache for {symbol}...")
        
        # Check cache first
        if self.use_cache and self.cache and not force_download:
            cached_data = self.cache.get(symbol, timeframe, source)
            if cached_data is not None:
                # Check if cached data covers requested range
                cache_start = cached_data.index[0]
                cache_end = cached_data.index[-1]
                
                if cache_start <= start_date and cache_end >= (end_date or datetime.now() - timedelta(days=1)):
                    if progress_callback:
                        progress_callback(100, f"Loaded {len(cached_data)} candles from cache")
                    
                    # Filter to requested range
                    mask = (cached_data.index >= start_date)
                    if end_date:
                        mask &= (cached_data.index <= end_date)
                    return cached_data[mask]
        
        if progress_callback:
            progress_callback(30, f"Downloading {symbol} from {source}...")
        
        # Get from provider
        provider = self.PROVIDERS.get(source)
        if not provider:
            logger.error(f"Unknown data source: {source}")
            return None
        
        # Get API key if needed
        api_key = self.api_keys.get(source)
        
        data = provider.get_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key
        )
        
        if progress_callback:
            progress_callback(80, f"Processing data...")
        
        # Cache the data
        if data is not None and self.use_cache and self.cache:
            self.cache.put(symbol, timeframe, source, data)
        
        if progress_callback:
            if data is not None:
                progress_callback(100, f"Downloaded {len(data)} candles")
            else:
                progress_callback(100, f"No data received")
        
        return data
    
    def get_cached_symbols(self) -> List[Dict]:
        """Get list of cached symbols."""
        if self.cache:
            return self.cache.list_cached_data()
        return []
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"entries": 0, "total_size_mb": 0}
    
    def clear_cache(self, symbol: str = None):
        """Clear cache."""
        if self.cache:
            if symbol:
                self.cache.invalidate(symbol=symbol)
            else:
                self.cache.clear_all()
    
    def get_symbol_suggestions(self, source: str = "yahoo") -> List[str]:
        """Get suggested symbols for a source."""
        provider = self.PROVIDERS.get(source)
        if provider:
            return provider.get_available_symbols()
        return []
